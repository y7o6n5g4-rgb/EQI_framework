# -*- coding: utf-8 -*-
"""
EQI_Holistic_Chunk_Emergence_v1_1.py  (S115, Architect: Claude)   Maestro: MAPSI(맏이)
[v1 -> v1.1] 월별 eigenfrequency 를 '미분적 출현'으로 교정 (MAPSI S115)
   v1: F = N_cycle/ΣT (평균 eigenperiod 의 역수 = 적분형) → 미적용 (비교용 F_avg 로만 남김)
   v1.1 (Solar 월별 trinity 계승):
     D2_month = ln(Σ_cycle T_k)                 적분 쪽 — 사이클 종료 조건(란다우 비용 n_c)이 닫음
     D1_month = exp(⟨|ΔT_k|⟩ / σ(ΔT_k))         미분 쪽 — 사이클 내 일별 eigenperiod 열의 변화율
     F_month  = √(D1_month / D2_month)           월별 eigenfrequency 미분적 출현 (Solar eigenfrequency 정의)
     T_month  = 1/F_month                         (다음 층의 일별 eigenperiod 후보)

Solar 로직(일별 eigenperiod 란다우 잔류 누적 → 임계 → 란다우 비용 지불 → 월별 eigenfrequency 출현)을
LIGO 청크에 이식.

  · 청크([x]=24, 4/3초) = 일별 블록 (미시/글로벌, EQI extended smallest unit)
  · 파일 전체 청크 = 월별 블록 (거시/로컬, EQI smallest unit)
  · 청크 trinity (Solar 계승): D1 = exp(⟨|Δφ'|⟩/σ(Δφ')), D2 = ln(Σ|φ − φ₀|)
      eigenperiod T_k = √(D2/D1) (센터 공식), eigenfrequency F_k = 1/T_k
  · 란다우 잔류 r_k = T_k·GAP_D2 (S59 단방향 소거 비용), GAP 단위 누적 R = Σ r_k / GAP
  · R ≥ n_c = TIP/GAP (S48 트리거) → 란다우 비용 지불, 슈냐 θ = π·R/n_c, Z = 1 + e^{iθ}
      → 사이클 월별 eigenfrequency F_cycle = N_cycle / Σ T_k (적분 누적 → 주파수 출현), 잔여 이월
      지불 횟수는 N_demon = |ln GAP|/GAP_D2 = 213.5 장부와 대조
  · 청크마다 holistic cycloid wave (cycloid v11.4: EHCW = D1·cos(MAXWELL_5th·ωt), PHCW = D2·sin(ωt))
  · 백색화: notch → bandpass(35~350) → 전체 해상도 주기도 1 Hz 이동평균으로 나눔
      (v3.9 의 거친 PSD 보간 방식은 신호 없는 데이터에서 파일 중앙 가짜 검출·4초 주기 인공물 확인 → 미적용)
  · 사전 판정(이벤트, GPS 기반 on-source = 병합 청크·직전 청크):
      BBH 는 T_k 또는 HCW 급수의 최대 |z| 청크가 on-source 이고 GW170814 H-L·V-L 같은 방향이면 인정.
      출현(지불) 시점이 on-source 에 오는지는 기록만 (우연 확률 ≈ 2·(T/2)/n_c 로 약한 검사)

사용:
  python EQI_Holistic_Chunk_Emergence_v1.py --h5h H.hdf5 --h5l L.hdf5 --gps <병합 GPS> [--shell 껍데기.json]
  python EQI_Holistic_Chunk_Emergence_v1.py --h H.csv --l L.csv --t0 <시작 GPS> --gps <병합 GPS>
  (껍데기: 긴 배경 파일을 넣으면 블록(32초)마다 같은 24청크 처리)
"""
import io, os, json, argparse, contextlib
import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d

import eqi_maxwell_layers as ML
from eqi_cycloid_wave_series_generator_v11_4 import EQI_cycloid_wave_series_generator_v11_4 as HCW

GAP = float(ML.GAP); GD = float(ML.GD)
N_C = float((ML.PHI ** 1.5 - 1) / ML.GAP)
N_DEMON = float(abs(np.log(float(ML.GAP))) / float(ML.GD))
N_CHUNK = int(ML.X_INT)
BLOCK = 131072
EDGE = (0, N_CHUNK - 1)
METRICS = ("T_eigenperiod", "hcw_series")


# ── 입력 ────────────────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    s = df["strain"].values if "strain" in df.columns else df.iloc[:, -1].values
    return np.nan_to_num(np.asarray(s, float))


def load_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        d = f["strain/Strain"]
        return (np.nan_to_num(np.asarray(d[:], float)), 1.0 / float(d.attrs.get("Xspacing", 1 / 4096)),
                float(d.attrs.get("Xstart", d.attrs.get("GPSstart", np.nan))))


# ── 백색화 (1 Hz 평활, 인공물 검사 통과) ─────────────────────────────
def whiten(x, fs):
    x = np.asarray(x, float); x = x - x.mean()
    for f0 in (60.0, 120.0, 180.0, 240.0, 300.0):
        b, a = iirnotch(f0, 30.0, fs=fs); x = filtfilt(b, a, x)
    b, a = butter(4, [35 / (fs / 2), 350 / (fs / 2)], btype="band"); x = filtfilt(b, a, x)
    X = np.fft.rfft(x)
    P = np.maximum(uniform_filter1d(np.abs(X) ** 2, max(3, int(round(len(x) / fs))), mode="nearest"), 1e-300)
    y = np.fft.irfft(X / np.sqrt(P), n=len(x))
    return y / (np.std(y) + 1e-300)


# ── 청크 trinity (Solar 계승) ───────────────────────────────────────
def trinity(phase, phasediff):
    rate = np.diff(phasediff)
    d1 = float(np.exp(np.mean(np.abs(rate)) / (np.std(rate) + 1e-300)))
    d2 = float(np.log(np.sum(np.abs(phase - phase[0])) + 1.0))
    return d1, d2, float(np.sqrt(d2 / d1))


import tempfile
_TMP = tempfile.mkdtemp(prefix="eqi_hcw_")


def chunk_hcw(feat, F, T):
    """청크의 holistic cycloid wave (v11.4). 반환: Infinite Series, 기저 항등식 오차."""
    step = max(1, len(feat) // 256)
    uf = feat[::step]
    with contextlib.redirect_stdout(io.StringIO()):
        r = HCW(uf, np.zeros(len(uf), int), np.array([F]), np.array([T]), time_steps=256,
                output_folder=_TMP)
    s_ = np.asarray(r["eqi_infinite_series"], float).ravel()
    return float(s_[-1]), float(r["base_error"])   # 누적 급수의 최종값


def process_block(phase, phasediff, feat):
    rows = []
    for k, idx in enumerate(np.array_split(np.arange(len(phase)), N_CHUNK)):
        d1, d2, T = trinity(phase[idx], phasediff[idx])
        series, berr = chunk_hcw(feat[idx], 1.0 / T, T)
        rows.append({"chunk": k, "D1": d1, "D2": d2, "T_eigenperiod": T, "F_eigenfrequency": 1.0 / T,
                     "residue_GAP": T * GD / GAP, "hcw_series": series, "hcw_base_error": berr})
    return rows


def emergence(rows):
    """일별 잔류 적분 누적 → n_c 도달 시 란다우 비용 지불 → 월별 eigenfrequency 출현."""
    R, cyc, events = 0.0, [], []
    for r in rows:
        R += r["residue_GAP"]; cyc.append(r)
        theta = np.pi * R / N_C
        r["R_cum"] = R; r["theta_over_pi"] = theta / np.pi; r["abs_Z"] = float(abs(1 + np.exp(1j * theta)))
        r["payment"] = False
        while R >= N_C:
            Tc = np.array([c["T_eigenperiod"] for c in cyc])
            D2m = float(np.log(np.sum(Tc)))                                   # 적분: 란다우 비용이 닫음
            dT = np.diff(Tc)
            D1m = float(np.exp(np.mean(np.abs(dT)) / (np.std(dT) + 1e-300))) if len(dT) >= 2 else float("nan")
            Fm = float(np.sqrt(D1m / D2m)) if np.isfinite(D1m) else float("nan")  # 미분적 출현
            events.append({"chunk": r["block_chunk"], "N_cycle": len(cyc), "D1_month": D1m, "D2_month": D2m,
                           "F_month": Fm, "T_month": 1.0 / Fm if Fm == Fm and Fm > 0 else float("nan"),
                           "F_avg_v1": len(cyc) / float(np.sum(Tc)), "carry_GAP": R - N_C})
            r["payment"] = True
            R -= N_C; cyc = []
    return events


def robust_z(v, ref):
    m = np.median(ref); s = np.median(np.abs(ref - m)) * 1.4826
    return (v - m) / (s + 1e-300)


def run(h, l, fs, t0=None, gps=None, white=True):
    n = min(len(h), len(l)); h, l = h[:n], l[:n]
    if white:
        h, l = whiten(h, fs), whiten(l, fs)
    phase = np.unwrap(np.angle(hilbert(h))) - np.unwrap(np.angle(hilbert(l)))
    phasediff = np.gradient(phase); curv = np.gradient(phasediff)
    ent = -np.cumsum(np.abs(curv) * np.log(np.abs(curv) + 1e-300))
    feat = np.column_stack([phase, curv, ent, phasediff])
    nb = max(1, n // BLOCK)
    rows = []
    for b in range(nb):
        sl = slice(b * BLOCK, (b + 1) * BLOCK) if n >= BLOCK else slice(0, n)
        for r in process_block(phase[sl], phasediff[sl], feat[sl]):
            r["block"] = b; r["block_chunk"] = b * N_CHUNK + r["chunk"]; rows.append(r)
    events = emergence(rows)
    out = {"white": white, "n_blocks": nb, "n_chunks": len(rows), "N_C": N_C, "N_demon": N_DEMON,
           "payments": len(events), "payments_over_N_demon": len(events) / N_DEMON,
           "events": events, "chunks": rows,
           "T_median": float(np.median([r["T_eigenperiod"] for r in rows])),
           "chunks_per_payment": len(rows) / max(len(events), 1)}
    if nb == 1 and t0 is not None and gps is not None:
        tm = gps - t0
        bounds = np.cumsum([len(a) for a in np.array_split(np.arange(n), N_CHUNK)]) / fs
        km = int(np.searchsorted(bounds, tm, side="right"))
        on = (km - 1, km)
        ks = np.array([r["chunk"] for r in rows])
        valid_off = ~np.isin(ks, on) & ~np.isin(ks, EDGE)
        test = {"merger_sec": tm, "merger_chunk": km, "on_set": list(on),
                "payment_in_on_source": any(e["chunk"] in on for e in events)}
        for m in METRICS:
            v = np.array([r[m] for r in rows]); z = robust_z(v, v[valid_off])
            pk = int(np.argmax(np.abs(np.where(np.isin(ks, EDGE), 0, z))))
            test[m] = {"on_z": [float(z[k]) for k in on], "peak_chunk": pk, "peak_z": float(z[pk]),
                       "peak_in_on_source": bool(pk in on)}
        out["onsource_test"] = test
    return out


def report(o):
    print("=" * 78)
    print(f" EQI Holistic Chunk Emergence v1.1 — 일별 적분 누적 → 란다우 비용 → 월별 미분 출현   [{'백색화' if o['white'] else '원시'}]")
    print("=" * 78)
    print(f"  블록 {o['n_blocks']}개, 청크 {o['n_chunks']}개, T(eigenperiod) 중앙값 {o['T_median']:.6f}")
    print(f"  란다우 비용 지불(R ≥ n_c={N_C:.4f}) {o['payments']}회, 지불당 청크 {o['chunks_per_payment']:.2f}"
          f",  지불/N_demon(213.5) = {o['payments_over_N_demon']:.4f}")
    for e in o["events"][:12]:
        print(f"    지불 @청크 {e['chunk']:4d}  사이클 {e['N_cycle']:2d}청크  D1_m {e['D1_month']:.5f}  D2_m {e['D2_month']:.5f}"
              f"  F_month(미분) {e['F_month']:.6f}  [평균형 {e['F_avg_v1']:.5f}]  이월 {e['carry_GAP']:.3f}")
    if len(o["events"]) > 12:
        F = [e["F_month"] for e in o["events"] if e["F_month"] == e["F_month"]]
        print(f"    … 월별 eigenfrequency(미분) 중앙값 {np.median(F):.6f}, 10~90% [{np.quantile(F, .1):.6f}, {np.quantile(F, .9):.6f}]")
    if o["n_blocks"] == 1:
        print("  청크별:  k    T_k        HCW 급수      R_cum   θ/π   지불")
        for r in o["chunks"]:
            print(f"         {r['chunk']:2d}  {r['T_eigenperiod']:.5f}  {r['hcw_series']:+.5e}  {r['R_cum']:7.3f}"
                  f"  {r['theta_over_pi']:.3f}  {'★' if r['payment'] else ''}")
    if "onsource_test" in o:
        t = o["onsource_test"]
        print(f"  ── GPS on-source: 병합 {t['merger_sec']:.2f}초 → 청크 {t['merger_chunk']}, on = {tuple(t['on_set'])} ──")
        for m in METRICS:
            x = t[m]
            print(f"    {m:<15} on z = {x['on_z'][0]:+7.2f}, {x['on_z'][1]:+7.2f}  최대|z| 청크 {x['peak_chunk']:2d}"
                  f" (z {x['peak_z']:+7.2f}) {'★ on-source' if x['peak_in_on_source'] else ''}")
        print(f"    란다우 비용 지불이 on-source 에서 일어남: {t['payment_in_on_source']} (기록만)")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h"); ap.add_argument("--l"); ap.add_argument("--h5h"); ap.add_argument("--h5l")
    ap.add_argument("--t0", type=float); ap.add_argument("--gps", type=float)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--raw", action="store_true"); ap.add_argument("--out", default="holistic_chunk_v1_1.json")
    a = ap.parse_args()
    if a.h5h and a.h5l:
        H, fs, t0 = load_h5(a.h5h); L, _, _ = load_h5(a.h5l); a.fs = fs
        a.t0 = a.t0 if a.t0 is not None else t0
    else:
        H, L = load_csv(a.h), load_csv(a.l)
    o = run(H, L, a.fs, a.t0, a.gps, white=not a.raw)
    report(o)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1, default=float)
    print(f"저장: {a.out}")

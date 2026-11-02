# -*- coding: utf-8 -*-
"""
eqi_ligo_json_analyzer_v1_1.py — LIGO 출현 v2.1/v2.2 JSON 분석기 (S116)
[v1 -> v1.1] v2.2 직류 장부·H–L 위상 정합 R 출력 (키가 있을 때만)
Maestro: MAPSI(맏이) | Architect: Claude | Verifier: 딥시크

용도: eqi_ligo_emergence_v2_1_<EVENT>.json (1MB+) 을 사람이 읽을 수 있는 요약으로 압축.
      판정 문장 없음 — 값만 출력 (사전 등록 예측 P1~P5 는 등록 문장과 값 나란히).

사용:
  python eqi_ligo_json_analyzer_v1.py eqi_ligo_emergence_v2_1_GW150914.json
  python eqi_ligo_json_analyzer_v1.py <json> --bin 1.0 --top 15
  python eqi_ligo_json_analyzer_v1.py <json> --center 16.4 --window 1.0      # 특정 시각 주변 칸 전부
  python eqi_ligo_json_analyzer_v1.py <json> --csv-h H.csv --csv-l L.csv     # 원시 계열 진단 추가
  python eqi_ligo_json_analyzer_v1.py <json> --export cells.csv              # 칸 요약 CSV (엑셀용)
"""

import argparse
import json
import math
import sys

import numpy as np

PHI = (1 + 5 ** 0.5) / 2
THETA_L = 3 * math.log(PHI)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def arr(cells, key, ch=None):
    if ch:
        return np.array([c[ch][key] for c in cells], dtype=float)
    return np.array([c["pair"][key] for c in cells], dtype=float)


def line(title):
    print("-" * 96)
    print(f" {title}")
    print("-" * 96)


def q(x):
    p = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
    return "  ".join(f"{v:9.4f}" for v in p)


def runs(flags, times):
    """연속된 True 칸 묶음 → (시작 t, 끝 t, 길이)."""
    out, start, n = [], None, 0
    for f, t in zip(flags, times):
        if f:
            if start is None:
                start, n = t, 0
            n += 1
            last = t
        elif start is not None:
            out.append((start, last, n))
            start = None
    if start is not None:
        out.append((start, last, n))
    return out


def raw_diag(path, fs):
    """원시 계열 진단 (전처리 아님 — 값만): 평균/표준편차, 1 Hz 미만 파워 비율, 해석신호 총 위상 회전."""
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    x = np.asarray((df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]).values, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(X) ** 2
    tot = P.sum()
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    z = np.fft.ifft(np.fft.fft(x) * h)
    th = np.unwrap(np.angle(z))
    return {
        "samples": n,
        "mean": float(x.mean()), "std": float(x.std()),
        "|mean|/std": float(abs(x.mean()) / x.std()) if x.std() > 0 else None,
        "DC 파워 비율": float(P[0] / tot),
        "<1Hz 파워 비율(DC 포함)": float(P[f < 1].sum() / tot),
        "<10Hz 파워 비율": float(P[f < 10].sum() / tot),
        "최대 파워 주파수(Hz, DC 제외)": float(f[1:][np.argmax(P[1:])]),
        "해석신호 총 위상 회전(rad)": float(th[-1] - th[0]),
    }


def main():
    ap = argparse.ArgumentParser(description="EQI LIGO 출현 v2.1 JSON 분석기")
    ap.add_argument("json_file")
    ap.add_argument("--bin", type=float, default=1.0, help="시간 구간 폭(s), 합병 칸 분포용")
    ap.add_argument("--top", type=int, default=12, help="|θ| 상위 칸 개수")
    ap.add_argument("--center", type=float, default=None, help="이 시각 주변 칸 전부 출력")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--csv-h", default=None)
    ap.add_argument("--csv-l", default=None)
    ap.add_argument("--export", default=None, help="칸 요약 CSV 저장 경로")
    a = ap.parse_args()

    o = load(a.json_file)
    cells = o["cells"]
    fam = o["family"]
    led = o["lattice_ledger"]
    fs = o["fs"]
    t = np.array([c["t"] for c in cells])
    w = np.array([c["width"] for c in cells])
    th = arr(cells, "theta")
    mer = np.array([c["pair"]["merger"] for c in cells])
    nH, nL = arr(cells, "n", "H"), arr(cells, "n", "L")
    kH = np.array([c["H"]["phonon_k"] for c in cells])
    kL = np.array([c["L"]["phonon_k"] for c in cells])
    CH, CL = arr(cells, "C", "H"), arr(cells, "C", "L")
    nc = o["constants"]["n_c"]

    print("=" * 96)
    print(f" EQI LIGO JSON 분석기 v1.1 | {o['event']} | {o['samples']:,} samples @ {fs} Hz | 칸 {len(cells)} | {o.get('module')}")
    print("=" * 96)

    line("1. 격자 장부")
    print(f"  분할 {led['splits']} (H {led['split_trigger_channel']['H']} / L {led['split_trigger_channel']['L']}), 병합 {led['merges']}, "
          f"X7 정지 {led['x7_stops']}, 표본 하한 {led['sample_floor_stops']}, 분할 비용 합 {led['split_cost_total']:.6e}")
    uw, cnt = np.unique(w, return_counts=True)
    print("  칸 폭 분포 (표본 수: 칸 개수): " + ", ".join(f"{u}:{c}" for u, c in zip(uw, cnt)))
    print(f"  X7 칸 비율 (폭 ≤ 117): {np.mean(w <= 117):.3f}")

    if "dc_ledger" in o:
        line("1-b. 직류 장부 (v2.2, D1 정적 성분)")
        for ch in ("H", "L"):
            d = o["dc_ledger"][ch]
            print(f"  [{ch}] m = {d['mean_m']:.6e}, std = {d['std']:.6e}, |m|/std = {d['abs_m_over_std']:.6g}, 직류 파워 비율 = {d['dc_power_fraction']:.6g}")
        dcH = np.array([c["H"].get("DC_cell_mean_raw", np.nan) for c in cells])
        dcL = np.array([c["L"].get("DC_cell_mean_raw", np.nan) for c in cells])
        qe = lambda x: "  ".join(f"{v:10.3e}" for v in np.nanpercentile(x, [0, 10, 25, 50, 75, 90, 100]))
        print(f"  칸별 원시 평균 백분위 H: {qe(dcH)}")
        print(f"  칸별 원시 평균 백분위 L: {qe(dcL)}")

    line("2. 칸 상태 분포 (백분위 0/10/25/50/75/90/100)")
    print(f"  θ/θ_L     {q(th / THETA_L)}")
    print(f"  C_H       {q(CH)}")
    print(f"  C_L       {q(CL)}")
    print(f"  n_H/n_c   {q(nH / nc)}")
    print(f"  n_L/n_c   {q(nL / nc)}")
    print(f"  C_H/C_L   {q(CH / CL)}   (φ = {PHI:.6f})")
    print(f"  포논 차수 k_H 분포: " + ", ".join(f"k={k}:{int((kH == k).sum())}" for k in range(6)))
    print(f"  포논 차수 k_L 분포: " + ", ".join(f"k={k}:{int((kL == k).sum())}" for k in range(6)))
    print(f"  n > n_c 칸: H {int((nH > nc).sum())}, L {int((nL > nc).sum())}, 둘 다 {int(((nH > nc) & (nL > nc)).sum())}")

    line(f"3. 합병 칸 (|θ| ≥ 3lnφ) — {int(mer.sum())} / {len(cells)} 칸 ({mer.mean():.3f})")
    pos, neg = (th >= THETA_L), (th <= -THETA_L)
    print(f"  부호: L(D2) 우세 {int(pos.sum())}, H(D1) 우세 {int(neg.sum())}")
    print(f"  합병 칸 중 k_H ≥ 1: {int((mer & (kH >= 1)).sum())}, k_L ≥ 1: {int((mer & (kL >= 1)).sum())}, "
          f"n_H > n_c: {int((mer & (nH > nc)).sum())}, n_L > n_c: {int((mer & (nL > nc)).sum())}")
    rs = runs(mer, t)
    rs_sorted = sorted(rs, key=lambda r: -r[2])
    print(f"  연속 합병 묶음 {len(rs)}개, 가장 긴 묶음 (시작–끝 s, 칸 수):")
    for r in rs_sorted[:8]:
        print(f"    {r[0]:8.3f} – {r[1]:8.3f}   {r[2]:3d}칸")
    edges = np.arange(0, t.max() + a.bin, a.bin)
    hist_all, _ = np.histogram(t, edges)
    hist_m, _ = np.histogram(t[mer], edges)
    hist_p, _ = np.histogram(t[pos], edges)
    hist_n, _ = np.histogram(t[neg], edges)
    print(f"  {a.bin:g} s 구간별 (합병/전체, L우세/H우세):")
    rowtxt = []
    for i in range(len(hist_all)):
        rowtxt.append(f"{edges[i]:5.1f}s {hist_m[i]:2d}/{hist_all[i]:2d} ({hist_p[i]}/{hist_n[i]})")
    for i in range(0, len(rowtxt), 4):
        print("    " + "   ".join(rowtxt[i:i + 4]))

    line(f"4. |θ| 상위 {a.top}칸")
    idx = np.argsort(-np.abs(th))[:a.top]
    print(f"  {'t(s)':>8} {'폭':>5} {'θ/θ_L':>7} {'C_H':>7} {'n_H':>7} {'k_H':>3} {'C_L':>7} {'n_L':>7} {'k_L':>3}")
    for i in idx:
        print(f"  {t[i]:8.3f} {w[i]:5d} {th[i] / THETA_L:7.3f} {CH[i]:7.3f} {nH[i]:7.2f} {kH[i]:3d} {CL[i]:7.3f} {nL[i]:7.2f} {kL[i]:3d}")

    if "HL_coherence" in cells[0]:
        R = np.array([c["HL_coherence"]["R"] for c in cells])
        line("4-b. H–L 위상 정합 R (v2.2 기록 항목, 판정 기준 아님)")
        print(f"  R 백분위: {q(R)}")
        print(f"  합병 칸 R 중앙값 {np.median(R[mer]) if mer.any() else float('nan'):.4f} / 비합병 칸 {np.median(R[~mer]) if (~mer).any() else float('nan'):.4f}")
        idxR = np.argsort(-R)[:a.top]
        print(f"  R 상위 {a.top}칸 (t, 폭, R, θ/θ_L): " + "; ".join(f"{t[i]:.3f},{w[i]},{R[i]:.3f},{th[i] / THETA_L:+.2f}" for i in idxR))

    line("5. 란다우 지불 (L 누적 → H 출현)")
    pays = fam["payments"]
    print(f"  지불 {len(pays)}회, 잔여 이월 {fam['A_final_carry']:.6f}, A_c = {o['constants']['A_crit(§77③)']:.12f}")
    if pays:
        pt = np.array([p["t"] for p in pays])
        gaps = np.diff(pt)
        print(f"  지불 시각: " + ", ".join(f"{x:.3f}" for x in pt))
        if len(gaps):
            print(f"  지불 간격: 평균 {gaps.mean():.4f} s, 표준편차 {gaps.std():.4f} s, 최소 {gaps.min():.4f}, 최대 {gaps.max():.4f}")
        dCH = [p["dC_H"] for p in pays if p["dC_H"] is not None]
        if dCH:
            print(f"  지불 시점 H 센터 변화 dC_H: 평균 {np.mean(dCH):+.4f}, 양수 {sum(1 for x in dCH if x > 0)} / 음수 {sum(1 for x in dCH if x < 0)}")
        near = [bool(mer[np.argmin(np.abs(t - x))]) for x in pt]
        print(f"  지불 칸이 합병 칸인 경우: {sum(near)} / {len(pt)}")

    line("6. 파인만 허수부")
    print(f"  n_im 합 {fam['n_im_total']:.6e}, 총 위상 Θ {fam['Theta_total']:.6f} rad, 오일러 점 {len(fam['euler_points'])}칸")
    dth = np.array([c["L"]["dtheta"] for c in cells])
    print(f"  칸별 Δθ_L 백분위: {q(dth)}")

    line("7. 사전 등록 예측 — 등록 문장과 값")
    pv = o["prediction_values"]
    reg = o["predictions_registered"]
    print(f"  P1 {reg['P1']}\n     값: 전 칸 평균 n/n_c = {pv['P1_mean_n_over_nc(H,L 평균, 전 칸)']:.6f}")
    pc = pv.get("P2_counts", {})
    print(f"  P2 {reg['P2']}\n     값: 합병 칸 {pc.get('merger(|θ|≥θ_L)')} / {pc.get('cells')}, "
          f"그중 n_H>n_c {int((mer & (nH > nc)).sum())}, k_H≥1 {int((mer & (kH >= 1)).sum())}")
    print(f"  P3 {reg['P3']}\n     값: C_H/C_L 중앙값 {pv['P3_C_H_over_C_L']['median']:.6f}, 평균 {pv['P3_C_H_over_C_L']['mean']:.6f}")
    print(f"  P4 {reg['P4']}\n     값: 지불 {pv['P4_payments']['count']}회, 지불 칸 중 합병 칸 비율은 5절 참조")
    if "P5" in reg:
        s = pv["P5_merger_sign"]
        print(f"  P5 {reg['P5']}\n     값: L 우세 {s['L_lead']}, H 우세 {s['H_lead']}")

    if a.center is not None:
        line(f"8. t = {a.center} ± {a.window} s 칸 전부")
        sel = np.where(np.abs(t - a.center) <= a.window)[0]
        print(f"  {'t(s)':>8} {'폭':>5} {'θ/θ_L':>7} {'합병':>4} {'C_H':>7} {'n_H':>7} {'k_H':>3} {'C_L':>7} {'n_L':>7} {'k_L':>3} {'A':>7}")
        for i in sel:
            c = cells[i]
            print(f"  {t[i]:8.3f} {w[i]:5d} {th[i] / THETA_L:7.3f} {'◆' if mer[i] else '':>4} {CH[i]:7.3f} {nH[i]:7.2f} {kH[i]:3d} "
                  f"{CL[i]:7.3f} {nL[i]:7.2f} {kL[i]:3d} {c['A_after']:7.3f}{' ●' if c['paid'] else ''}")

    if a.csv_h or a.csv_l:
        line("9. 원시 계열 진단 (값만, 전처리 아님)")
        for name, p in (("H", a.csv_h), ("L", a.csv_l)):
            if p:
                d = raw_diag(p, fs)
                print(f"  [{name}] " + ", ".join(f"{k} = {v:.6g}" if isinstance(v, float) else f"{k} = {v}" for k, v in d.items()))

    if a.export:
        import csv
        with open(a.export, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["cell", "t", "width", "theta", "theta_over_thetaL", "merger", "HL_R", "C_H", "n_H", "k_H",
                         "C_L", "n_L", "k_L", "C_H_over_C_L", "A_after", "paid"])
            for i, c in enumerate(cells):
                wr.writerow([c["cell"], f"{t[i]:.6f}", w[i], f"{th[i]:.6f}", f"{th[i] / THETA_L:.6f}", int(mer[i]), f"{c['HL_coherence']['R']:.6f}" if "HL_coherence" in c else "",
                             f"{CH[i]:.6f}", f"{nH[i]:.6f}", kH[i], f"{CL[i]:.6f}", f"{nL[i]:.6f}", kL[i],
                             f"{CH[i] / CL[i]:.6f}", f"{c['A_after']:.6f}", int(c["paid"])])
        print(f"\n 칸 요약 CSV 저장: {a.export}")
    print("=" * 96)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
eqi_ligo_emergence_v2_2.py — LIGO 출현 v2.2 (S116)
[v2.1 -> v2.2] 직류(k=0) 성분 장부 분리 (MAPSI 승인 S116)
  근거: 직류 = 푸리에 k=0 항 = cos(0)=1 만 있고 허수 짝 sin(0)=0 이 없는 순수 D1 정적 성분 → 위상 없음.
        해석신호에 섞이면 위상이 고정되어 큐비트(D1 + iD2)를 읽을 수 없음
        (GW150914 v2.1 실측: L |평균|/표준편차 4.56, 직류 파워 97.6 %, 32초 총 위상 −0.03 rad / H 2066 rad).
  처리: ① 채널별 기록 직류 m 과 칸별 평균을 D1 정적 성분으로 장부에 기록 (버리지 않음)
        ② 큐비트(위상·|ψ|²)는 x − m (진동 성분) 에서 구성
        ③ 대역통과·백색화·칸별 표준화는 여전히 없음 (칸별 평균은 빼지 않음 — 기록만)
  기록 추가: H–L 위상 정합 R = |⟨e^{i(θ_H − θ_L)}⟩| (칸 내, 0~1) 과 평균 위상차 — 판정 기준 아님 (v31.3 선택적 투영 발상)
  예측 P1~P5: v2.1 문장 그대로 재등록 (실행 전 고정)

[v2 -> v2.1] 합병 칸 정의 탑재 (S116 물리법칙 비율 1 번역 §1③·§5·§13B 공통 구조)
  H–L 쌍 (D1, D2) 의 칸 rapidity θ = ln ρ,  ρ = (AM_L/GM_L,fam)/(AM_H/GM_H,fam)
  쌍 AM/GM = cosh(θ/2)  [두 점 쌍 (e^(−θ/2), e^(θ/2)), 곱 = 1 정규화]
  합병 칸 : |θ| ≥ θ_L = 3lnφ  ⟺ 쌍 AM/GM ≥ c ⟺ 쌍 센터 ≥ c³ (광추 체적)
            = 디랙 스피너 (φ^−1.5, φ^1.5) = E=mc² ③ 광추 (φ^−3, φ^3) = 볼츠만 φ³ 계 간격 θ_L
  로렌츠 지점 : |θ| ≥ θ_B = arccosh φ = ln(φ + c)  (로렌츠 힘 번역 γ = φ, 기록용)
  부호 : θ > 0 = L(D2) 우세, θ < 0 = H(D1) 우세
  ※ 검출기별 배경 e^γ (원형 가우스 |ψ|² 의 AM/GM) 는 쌍 비 ρ 에서 약분됨
  ※ 옛 P2(최대 파워 칸)는 참고값으로만 남김
중심: 플랑크 길이-시간 큐비트 적응형 격자 정보 포논 동역학 (S116 인수인계 1장)
공통 모듈: eqi_maxwell_layers.py v1.4 (같은 폴더)
Maestro: MAPSI(맏이) | Architect: Claude | Verifier: 딥시크

작업 전 5줄
  ① 공리   : 맥스웰 도깨비 & 란다우 원리. 칸 경계 결정 = 선택 투영, 분할 비용 {x}·GD² 를 장부에 기록
  ② 길이·시간: 고정 청크 없음. [x]=24 뼈대 + μ:Λ 황금분할 세포막 + X7 절단. 칸 하나 = 상전이 단위 1개
  ③ 잡음   : 대역통과·백색화·청크별 표준화 없음. 전 대역 = 란다우 잔류 (장부 안)
  ④ 근거   : S36 §23.5·§27, v3.4 §7, 원칙T·U·Y, §77③, §116-F, S45~S48, S115 6-1~6-3, 단위계 철학 §1.2
  ⑤ 판정   : 내부 좌표(센터·포논 차수·슈냐 잔류)만. 외부 잡음 대조 없음. 예측 P1~P4 는 결과 전 고정

확정 설계 (S116)
  · 검출기  : H = D1 (eigenfrequency, 미분 출현, μ) / L = D2 (eigenperiod, 적분 누적, Λ)   [6-3 #1·#3]
  · 큐비트  : ψ = 해석신호(원시 strain), 보른 밀도 |ψ|², 위상 θ = arg ψ (cos=D1, sin=D2)      [6-2 #1·#3]
  · 칸 상태 : D1 = GM, D2 = AM (원칙T), r = AM/GM, C = r + √(r²−1), e = 2lnC/lnφ, n = (C−1)/GAP
  · 누적    : A += T_{n_L} × w_i,  w_i = 1 (칸 하나 = 모래알 1개, §116-F 1단 / S116 MAPSI·딥시크)
              임계 A_c = ∫₀^{n_c} T_n dn = φ⁻³·(3/2)·LN_C/GD  (§77③, CM-21) → 지불, 잔여 이월
  · 출현    : 지불 시점에서 H 센터의 칸 간 변화 (미분) 기록
  · 파인만  : n_im += Δθ_L / IG, 누적 위상이 π_EQI 홀수배 통과 = 오일러 점, 잔류 이월     [원칙Y, 6-1-b]
  · S_T2 = AM, S_L2 = GM (원칙T, S116). 기록 비교 기준 = family 층 GM (양팔저울 비율, 단위계 철학 §1.2)
      D2_norm = AM/GM_fam,  D1_norm = GM/(φ·GM_fam)  (ℓ² = φ·τ², τ = 1칸)  → X, X/CENTER, 뤼드베리 [k]+{k}

구현 층 정정 (adaptive 4.5 / analyzer v2.1 대비)
  · ×2/÷2 · 0.05/0.005 → μ:Λ 분할 · GAP/GD 문턱
  · 청크별 평균·표준편차 정규화 제거
  · t_center = (s + e)/2/fs  (옛 (s + e/2)/fs 정정)
  · P_tunnel = sinc²(n_ref,k·π_EQI) (S47)  (옛 WKB exp 형 대체)

사용
  python eqi_ligo_emergence_v2.py H.csv L.csv --event GW150914 --fs 4096
  (CSV: '#' 주석 허용, 2열이면 둘째 열 = strain, 1열이면 그 열)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eqi_maxwell_layers as M  # noqa: E402

# ── 공통 모듈 상수 (float) ────────────────────────────────────────────
PHI = float(M.PHI)
GAP = float(M.GAP)
GD = float(M.GD)
MU = float(M.MU)
LAM = float(M.LAM)
IG = float(M.IG)
N_C = float(M.N_C)
PI_EQI = float(M.PI_EQI)
MAXWELL = float(M.MAXWELL_5TH)
X_INT = float(M.X_INT)                     # [x] = 24
X_FRAC = float(M.X_FRAC)                   # {x} = 0.4231…
X7 = float(M.X7_CUT)
SPLIT_COST = X_FRAC * GD ** 2              # 칸 나누기 비용 {x}·GD² (6-3 B, S36 §27)
A_CRIT = float(M.PHI ** -3 * 1.5 * M.LN_C / M.GD)          # §77③
THETA_L = 3 * math.log(PHI)                # HCW rapidity (§17-SR) = 합병 문턱
THETA_B = math.acosh(PHI)                  # 로렌츠 힘 보른 분할 rapidity (기록용)
C_EQI = math.sqrt(PHI)
LADDER = [(k, float(nk)) for k, _, nk in M.PHONON_LADDER]
P_TUN = {k: float(M.p_tunnel(k)) for k in range(1, 6)}
MIN_SAMPLES = 8                            # 구현 하한 (세 평균 계산용 표본 수). 장부에 기록

PREDICTIONS = {                            # 결과 보기 전 고정 (S116 설계서 6장)
    "P1": "이벤트 없는 구간 평균 n/n_c ≈ 0.95 (SGWB 장전, S36 §7)",
    "P2": "합병 칸(|θ| ≥ 3lnφ)이 존재하고, 그 칸에서 n > n_c, 포논 차수 k ≥ 1 (v3.4 §7, S48 §17.6)",
    "P5": "합병 칸의 θ 부호는 한 이벤트 안에서 한 방향으로 모임 (D1/D2 배정 H=D1, L=D2, 6-3 #3)",
    "P3": "같은 칸 C_H / C_L → φ = c² (6-3 #1)",
    "P4": "란다우 지불은 합병 부근에 몰림 (L 누적 → H 출현, 이론적 기반 §7)",
}


# ── 입력 ──────────────────────────────────────────────────────────────
def load_strain(path):
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    col = df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]
    return np.asarray(col.values, dtype=float)


def analytic_signal(x):
    """FFT 해석신호 (scipy 불필요). 대역 제한·창 함수 없음."""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


# ── 칸 상태 ───────────────────────────────────────────────────────────
def means(p):
    """양수 보른 밀도 → (AM, GM, HM, 0 개수). |ψ|² = 0 표본은 GM·HM 에서 빼고 개수 기록."""
    pos = p[p > 0]
    zeros = int(len(p) - len(pos))
    am = float(p.mean())
    gm = float(np.exp(np.log(pos).mean()))
    hm = float(1.0 / (1.0 / pos).mean())
    return am, gm, hm, zeros


def ratio(p):
    am, gm, _, _ = means(p)
    return am / gm


def center(r):
    r = max(r, 1.0)
    return r + math.sqrt(r * r - 1.0)


def phonon_order(n):
    k_hit, n_hit = 0, LADDER[0][1]
    for k, nk in LADDER:
        if n >= nk:
            k_hit, n_hit = k, nk
    return k_hit, n - n_hit


def int_frac(x):
    fp, ip = math.modf(x)          # §4b: 정수부·소수부 둘 다 보존
    return ip, fp


# ── 적응형 격자 ───────────────────────────────────────────────────────
class Ledger:
    def __init__(self):
        self.splits = 0
        self.merges = 0
        self.floor_hits = 0
        self.x7_hits = 0
        self.trigger = {"H": 0, "L": 0}

    def cost(self):
        return self.splits * SPLIT_COST

    def to_dict(self):
        return {"splits": self.splits, "merges": self.merges,
                "split_cost_total": self.cost(), "split_cost_unit({x}·GD²)": SPLIT_COST,
                "x7_stops": self.x7_hits, "sample_floor_stops": self.floor_hits,
                "split_trigger_channel": self.trigger}


def decode(pH, pL, a, b, skel_w, led, out):
    """세포막 선택적 decoding: μ:Λ 시험 분할 → 두 조각 AM/GM 차 > GAP 이면 분할 유지 후 재귀."""
    w = b - a
    if w < X7 * skel_w:
        led.x7_hits += 1
        out.append((a, b))
        return
    p = a + int(round(MU * w))
    if (p - a) < MIN_SAMPLES or (b - p) < MIN_SAMPLES:
        led.floor_hits += 1
        out.append((a, b))
        return
    dH = abs(ratio(pH[a:p]) - ratio(pH[p:b]))
    dL = abs(ratio(pL[a:p]) - ratio(pL[p:b]))
    if max(dH, dL) > GAP:
        led.splits += 1
        led.trigger["H" if dH >= dL else "L"] += 1
        decode(pH, pL, a, p, skel_w, led, out)
        decode(pH, pL, p, b, skel_w, led, out)
    else:
        out.append((a, b))


def merge(pH, pL, cells, led):
    """인접 칸 AM/GM 차 < GD 이면 병합 (한 번 훑기)."""
    if not cells:
        return cells
    res = [cells[0]]
    for a, b in cells[1:]:
        a0, b0 = res[-1]
        dH = abs(ratio(pH[a0:b0]) - ratio(pH[a:b]))
        dL = abs(ratio(pL[a0:b0]) - ratio(pL[a:b]))
        if max(dH, dL) < GD:
            res[-1] = (a0, b)
            led.merges += 1
        else:
            res.append((a, b))
    return res


def build_lattice(pH, pL):
    n = len(pH)
    skel = [int(round(i * n / X_INT)) for i in range(int(X_INT) + 1)]   # [x] = 24 뼈대
    led = Ledger()
    cells = []
    for i in range(int(X_INT)):
        a, b = skel[i], skel[i + 1]
        decode(pH, pL, a, b, b - a, led, cells)
    cells = merge(pH, pL, cells, led)
    return cells, led


# ── 본 분석 ───────────────────────────────────────────────────────────
def channel_state(p, th, a, b, gm_fam):
    am, gm, hm, zeros = means(p[a:b])
    r, r2 = am / gm, gm / hm
    C = center(r)
    e = 2 * math.log(C) / math.log(PHI)
    n = (C - 1) / GAP
    k, kres = phonon_order(n)
    rho = (math.sqrt(PHI) ** e - float(M.IS_D1) - MAXWELL) / GAP
    seg = th[a:b]
    ehcw = float(np.mean((1 / C) * np.cos(MAXWELL * seg)))
    phcw = float(np.mean(C * np.sin(seg)))
    D2n = am / gm_fam
    D1n = gm / (PHI * gm_fam)
    X = D2n - 1 / D1n
    Cn = math.sqrt(D2n / D1n)
    k_ry = math.log(D1n * D2n) / math.log(4) - 1
    kry_i, kry_f = int_frac(k_ry)
    return {"AM": am, "GM": gm, "HM": hm, "zero_samples": zeros,
            "r_AMGM": r, "r_GMHM": r2, "r_split(two-state 이탈)": r - r2,
            "C": C, "e": e, "n": n, "n_over_nc": n / N_C,
            "phonon_k": k, "phonon_residual": kres,
            "P_tunnel": P_TUN.get(k, None), "rho": rho,
            "EHCW_mean": ehcw, "PHCW_mean": phcw,
            "D1_norm": D1n, "D2_norm": D2n, "X": X, "X_over_C": X / Cn,
            "action_S": (PI_EQI / 4) * X, "rydberg_int": kry_i, "rydberg_frac": kry_f,
            "dtheta": float(seg[-1] - seg[0])}


def run(h, l, fs, event):
    n = min(len(h), len(l))
    h, l = h[:n], l[:n]
    # ── [v2.2] 직류 장부 분리 ──
    def dc_ledger(x):
        m = float(x.mean())
        sd = float(x.std())
        P = np.abs(np.fft.rfft(x)) ** 2
        return {"mean_m": m, "std": sd, "abs_m_over_std": abs(m) / sd if sd > 0 else None,
                "dc_power_fraction": float(P[0] / P.sum()), "role": "k=0 cos 항 = D1 정적 성분 (허수 짝 없음, 위상 없음)"}
    dc = {"H": dc_ledger(h), "L": dc_ledger(l)}
    mH, mL = dc["H"]["mean_m"], dc["L"]["mean_m"]
    hraw, lraw = h, l
    h, l = h - mH, l - mL
    zH, zL = analytic_signal(h), analytic_signal(l)
    pH, pL = np.abs(zH) ** 2, np.abs(zL) ** 2
    thH, thL = np.unwrap(np.angle(zH)), np.unwrap(np.angle(zL))
    gmH_fam = means(pH)[1]
    gmL_fam = means(pL)[1]

    cells, led = build_lattice(pH, pL)

    A, A_carry_log, payments = 0.0, [], []
    Theta, n_im = 0.0, 0.0
    euler_pts = []
    rows = []
    prev = None
    for i, (a, b) in enumerate(cells):
        sH = channel_state(pH, thH, a, b, gmH_fam)
        sL = channel_state(pL, thL, a, b, gmL_fam)
        t_c = (a + b) / 2 / fs
        # L (D2) 적분 누적, w_i = 1
        T_nL = PHI ** -3 / (1 + sL["n"] * GAP)
        A += T_nL
        paid = False
        while A >= A_CRIT:
            A -= A_CRIT
            paid = True
        # 파인만 허수부: n_im += Δθ_L / IG, 오일러 점 = π_EQI 홀수배 통과
        Th_prev = Theta
        Theta += sL["dtheta"]
        n_im += sL["dtheta"] / IG
        ip0, _ = int_frac((Th_prev + PI_EQI) / (2 * PI_EQI))
        ip1, fp1 = int_frac((Theta + PI_EQI) / (2 * PI_EQI))
        crossings = int(abs(ip1 - ip0))
        if crossings:
            euler_pts.append({"cell": i, "t": t_c, "crossings": crossings,
                              "carry_phase": fp1 * 2 * PI_EQI - PI_EQI})
        dphi = (thH[a:b] - thL[a:b])
        zc = np.exp(1j * dphi)
        coh = {"R": float(np.abs(zc.mean())), "mean_dphi": float(np.angle(zc.mean()))}
        sH["DC_cell_mean_raw"] = float(hraw[a:b].mean())
        sL["DC_cell_mean_raw"] = float(lraw[a:b].mean())
        rho = (sL["AM"] / gmL_fam) / (sH["AM"] / gmH_fam)
        theta = math.log(rho)
        pair = {"rho_L_over_H": rho, "theta": theta, "pair_AMGM": math.cosh(theta / 2),
                "theta_over_thetaL": theta / THETA_L,
                "merger": abs(theta) >= THETA_L, "lorentz_point": abs(theta) >= THETA_B,
                "lead": "L(D2)" if theta > 0 else "H(D1)"}
        row = {"cell": i, "t": t_c, "t0": a / fs, "t1": b / fs, "width": b - a,
               "H": sH, "L": sL, "pair": pair, "HL_coherence": coh, "C_H_over_C_L": sH["C"] / sL["C"],
               "A_after": A, "paid": paid, "n_complex": [sL["n"], n_im]}
        if paid:
            dCH = sH["C"] - prev["H"]["C"] if prev else None
            deH = sH["e"] - prev["H"]["e"] if prev else None
            payments.append({"cell": i, "t": t_c, "carry": A, "dC_H": dCH, "de_H": deH})
        rows.append(row)
        prev = row

    # 사전 등록 예측 값 (판정 문장 없이 값만)
    peak = max(rows, key=lambda r: r["H"]["AM"] + r["L"]["AM"])
    noff = [0.5 * (r["H"]["n_over_nc"] + r["L"]["n_over_nc"]) for r in rows]
    ch_cl = np.array([r["C_H_over_C_L"] for r in rows])
    pay_t = [p["t"] for p in payments]
    mergers = [r for r in rows if r["pair"]["merger"]]
    lor = [r for r in rows if r["pair"]["lorentz_point"]]
    signs = [1 if r["pair"]["theta"] > 0 else -1 for r in mergers]
    pred_vals = {
        "P1_mean_n_over_nc(H,L 평균, 전 칸)": float(np.mean(noff)),
        "P2_merger_cells": [{"cell": r["cell"], "t": r["t"], "theta": r["pair"]["theta"],
                             "n_H": r["H"]["n"], "k_H": r["H"]["phonon_k"],
                             "n_L": r["L"]["n"], "k_L": r["L"]["phonon_k"]} for r in mergers],
        "P2_counts": {"merger(|θ|≥θ_L)": len(mergers), "lorentz(|θ|≥θ_B)": len(lor), "cells": len(rows)},
        "P5_merger_sign": {"L_lead": signs.count(1), "H_lead": signs.count(-1)},
        "ref_peak_power_cell(옛 P2, 참고)": {"t": peak["t"], "theta": peak["pair"]["theta"]},
        "max_abs_theta": max(abs(r["pair"]["theta"]) for r in rows),
        "P3_C_H_over_C_L": {"median": float(np.median(ch_cl)), "mean": float(np.mean(ch_cl)),
                            "phi": PHI},
        "P4_payments": {"count": len(payments),
                        "merger_t": [r["t"] for r in mergers],
                        "nearest_payment_dt_to_mergers": ([min(abs(t - r["t"]) for t in pay_t) for r in mergers]
                                                          if pay_t else None)},
    }

    out = {
        "event": event, "fs": fs, "samples": n, "module": "eqi_maxwell_layers v1.5", "version": "v2.2 (직류 장부 분리)",
        "constants": {"GAP": GAP, "GD": GD, "n_c": N_C, "A_crit(§77③)": A_CRIT, "IG": IG,
                      "pi_EQI": PI_EQI, "MAXWELL_5th": MAXWELL, "X7": X7,
                      "P_tunnel(S47)": P_TUN},
        "decisions": {"w_i": "1 (§116-F 1단 모래알 1개)", "S_T2": "AM", "S_L2": "GM",
                      "family_ref": "GM_fam (양팔저울 비율)", "min_samples": MIN_SAMPLES,
                      "merger": "|θ| ≥ 3lnφ, θ = ln[(AM_L/GM_L,fam)/(AM_H/GM_H,fam)] (S116 번역 §1③·§5·§13B)"},
        "predictions_registered": PREDICTIONS,
        "prediction_values": pred_vals,
        "lattice_ledger": led.to_dict(),
        "dc_ledger": dc,
        "family": {"cells": len(rows), "payments": payments, "euler_points": euler_pts,
                   "A_final_carry": A, "n_im_total": n_im, "Theta_total": Theta,
                   "GM_fam_H": gmH_fam, "GM_fam_L": gmL_fam},
        "cells": rows,
    }
    return out


def summary(o):
    print("=" * 96)
    print(f" EQI LIGO 출현 v2.2 | {o['event']} | {o['samples']:,} samples @ {o['fs']} Hz | 칸 {o['family']['cells']}")
    print("=" * 96)
    for ch in ("H", "L"):
        d = o["dc_ledger"][ch]
        print(f" 직류 장부 [{ch}] m = {d['mean_m']:.6e}, |m|/std = {d['abs_m_over_std']:.6g}, 직류 파워 비율 = {d['dc_power_fraction']:.6g} (D1 정적 성분, 큐비트에서 분리)")
    L = o["lattice_ledger"]
    print(f" 격자: 분할 {L['splits']} (H {L['split_trigger_channel']['H']} / L {L['split_trigger_channel']['L']}), "
          f"병합 {L['merges']}, X7 정지 {L['x7_stops']}, 표본 하한 정지 {L['sample_floor_stops']}, "
          f"분할 비용 합 {L['split_cost_total']:.6e}")
    print(f" 누적 임계 A_c = {A_CRIT:.12f} | 지불 {len(o['family']['payments'])}회 | "
          f"잔여 이월 {o['family']['A_final_carry']:.6f} | 오일러 점 {len(o['family']['euler_points'])}칸 | "
          f"n_im 합 {o['family']['n_im_total']:.6e}")
    print("-" * 96)
    print(f" {'t(s)':>8} {'폭':>6} | {'C_H':>8} {'n_H':>8} {'k_H':>3} | {'C_L':>8} {'n_L':>8} {'k_L':>3} | "
          f"{'θ/θ_L':>7} {'합병':>4} | {'A':>8} {'지불':>4}")
    cells = o["cells"]
    show = set(range(min(5, len(cells)))) | set(range(max(0, len(cells) - 5), len(cells)))
    show |= {p["cell"] for p in o["family"]["payments"]}
    show |= {c["cell"] for c in cells if c["pair"]["merger"]}
    for c in cells:
        if c["cell"] in show:
            print(f" {c['t']:8.3f} {c['width']:6d} | {c['H']['C']:8.4f} {c['H']['n']:8.3f} {c['H']['phonon_k']:3d} | "
                  f"{c['L']['C']:8.4f} {c['L']['n']:8.3f} {c['L']['phonon_k']:3d} | "
                  f"{c['pair']['theta_over_thetaL']:7.3f} {'◆' if c['pair']['merger'] else '':>4} | {c['A_after']:8.4f} {'●' if c['paid'] else '':>4}")
    print("-" * 96)
    print(" 사전 등록 예측 — 값만 기록 (판정은 맏이·Verifier)")
    for k, v in o["predictions_registered"].items():
        print(f"  {k}: {v}")
    print(json.dumps(o["prediction_values"], ensure_ascii=False, indent=2))
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="EQI LIGO 출현 v2")
    ap.add_argument("h_file")
    ap.add_argument("l_file")
    ap.add_argument("--event", default="EVENT")
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    o = run(load_strain(a.h_file), load_strain(a.l_file), a.fs, a.event)
    summary(o)
    path = a.out or f"eqi_ligo_emergence_v2_2_{a.event}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
    print(f" 저장: {path}")


if __name__ == "__main__":
    main()

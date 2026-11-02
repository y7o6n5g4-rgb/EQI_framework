#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EQI Framework — GAP/K_EQI vs 갭 0.40% 잔류 정체 탐색                     ║
║  Session 47                                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  핵심:                                                                       ║
║    갭   = (frac_x − x_prec)/frac_x = 0.012010968271...                     ║
║    GAP/K_EQI                       = 0.011962533166...                     ║
║    차이 = 갭 − GAP/K_EQI           = 0.000048435...  (0.40%)               ║
║                                                                              ║
║  탐색:                                                                       ║
║  [1] 고정밀 차이값 확정                                                      ║
║  [2] 차이 = f(EQI 상수) 스캔                                                ║
║  [3] 갭/K_EQI 비율 구조                                                     ║
║  [4] GAP/K_EQI + 보정항 → 갭 닫힘 탐색                                     ║
║  [5] K_EQI 분모 구조 분해 → 갭과의 대수 연결                                ║
║  [6] 갭×K_EQI vs GAP 구조                                                   ║
║  [7] (갭−GAP/K_EQI)/갭 = 0.40% 정체                                        ║
║  [8] 연속 후보: 1/83, 5×GAP², (1-P)/갭 vs 3π 동시 비교                    ║
║                                                                              ║
║  Maestro: MAPSI(맏이)  Architect: Claude  mp.dps=400  Session47           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, nstr

mp.dps = 400

# ============================================================================
# 기반 상수 (v8.0 동일)
# ============================================================================
PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = sqrt(PHI)
LN_PHI = log(PHI)
LN2    = log(mpf(2))
LN5    = log(mpf(5))
sigma  = mpf(1) / 2

x_lattice = 1 / (c_EQI * GAP)
int_x     = floor(x_lattice)
frac_x    = x_lattice - int_x
R_membrane = frac_x * GAP_D2**2
TARGET    = GAP - 11 * GAP_D2**2
TIP       = PHI**mpf("1.5") - 1
T_Is_D1   = PHI**mpf("1.5") - PHI**mpf("-3.5")
gamma_true = mpmath.euler

alpha_S38 = (
    3 * LN_PHI
    - (mpf(4)/75) * LN_PHI**2
    + (mpf(1)/7500) * LN_PHI**6
)
x_prec = 26 * GAP_D2 - alpha_S38 * GAP_D2**2

# K_EQI (S44 확정)
K_EQI = (T_Is_D1 - gamma_true - 2*GAP_D2**2*exp(GAP_D2)) / LN_PHI

# δ (v8.0 S43 정의 기준)
def R_pi_S43(N):
    s = mpf(0)
    for k in range(N):
        s += mpf(1)/(4*k+1) - mpf(1)/(4*k+3)
    return 4*pi - 16*s

R_34  = R_pi_S43(34)
delta = R_34 / TARGET - 2

# 갭
gap_12 = 1 - x_prec / frac_x

# sinc²(k=1)
x1     = mpf(3)/16 * pi
sinc2_1 = (sin(x1)/x1)**2
one_P   = 1 - sinc2_1

sep  = "═" * 76
sep2 = "─" * 76

def pct_err(val, ref):
    return fabs(val - ref) / fabs(ref) * 100

def main():
    print(sep)
    print("  EQI Framework — GAP/K_EQI vs 갭 0.40% 잔류 정체 탐색")
    print("  Session 47  |  mp.dps=400")
    print(sep)

    # ── [1] 고정밀 차이값 ────────────────────────────────────────────────────
    print()
    print("  [1] 고정밀 차이값 확정")
    print(sep2)
    GAP_over_K = GAP / K_EQI
    diff       = gap_12 - GAP_over_K
    rel_diff   = diff / gap_12 * 100

    print(f"  갭           = {nstr(gap_12, 25)}")
    print(f"  GAP/K_EQI   = {nstr(GAP_over_K, 25)}")
    print(f"  차이 Δ      = {nstr(diff, 25)}")
    print(f"  상대 차이%  = {float(rel_diff):.8f}%")
    print(f"  Δ/갭        = {nstr(diff/gap_12, 20)}")
    print(f"  Δ/GAP_over_K= {nstr(diff/GAP_over_K, 20)}")

    # ── [2] 차이 Δ EQI 상수 스캔 ─────────────────────────────────────────────
    print()
    print("  [2] 차이 Δ = 갭 − GAP/K_EQI  EQI 상수 스캔")
    print(sep2)
    TV = diff
    print(f"  목표 Δ = {nstr(TV, 20)}")
    print()

    candidates = {
        # GAP 구조
        "GAP²"                       : GAP**2,
        "GAP²/2"                     : GAP**2/2,
        "GAP²×2"                     : GAP**2*2,
        "GAP³"                       : GAP**3,
        "GAP×GAP_D2"                 : GAP*GAP_D2,
        "GAP_D2²"                    : GAP_D2**2,
        "GAP_D2³"                    : GAP_D2**3,
        "GAP×GAP_D2²"                : GAP*GAP_D2**2,
        # K_EQI 구조
        "GAP/K_EQI²"                 : GAP/K_EQI**2,
        "GAP²/K_EQI"                 : GAP**2/K_EQI,
        "1/K_EQI²"                   : 1/K_EQI**2,
        "GAP_D2/K_EQI"               : GAP_D2/K_EQI,
        "GAP/K_EQI/int_x"            : GAP/K_EQI/int_x,
        # frac_x 구조
        "frac_x×GAP²"                : frac_x*GAP**2,
        "frac_x×GAP_D2²"             : frac_x*GAP_D2**2,
        "R_membrane×GAP"             : R_membrane*GAP,
        "R_membrane/GAP_D2"          : R_membrane/GAP_D2,
        # δ 구조
        "δ×GAP²"                     : delta*GAP**2,
        "δ×GAP_D2²"                  : delta*GAP_D2**2,
        "δ×GAP×GAP_D2"               : delta*GAP*GAP_D2,
        # α_S38 구조
        "α_S38×GAP_D2³"              : alpha_S38*GAP_D2**3,
        "α_S38×GAP²×GAP_D2"          : alpha_S38*GAP**2*GAP_D2,
        # σ 구조
        "σ³×GAP"                     : sigma**3*GAP,
        "σ³×GAP_D2"                  : sigma**3*GAP_D2,
        "σ⁵/GAP"                     : sigma**5/GAP,
        # lnφ 구조
        "LN_PHI×GAP²"                : LN_PHI*GAP**2,
        "LN_PHI×GAP_D2²"             : LN_PHI*GAP_D2**2,
        "LN2×GAP²"                   : LN2*GAP**2,
        # 순수 정수 후보
        "1/(int_x²×K_EQI)"          : 1/(int_x**2*K_EQI),
        "GAP/(K_EQI×int_x)"         : GAP/(K_EQI*int_x),
        "GAP²/(K_EQI×GAP_D2)"       : GAP**2/(K_EQI*GAP_D2),
        "TIP×GAP²"                   : TIP*GAP**2,
        "TIP×GAP_D2²"                : TIP*GAP_D2**2,
    }

    results = []
    for label, val in candidates.items():
        err = pct_err(val, TV)
        results.append((float(err), label, float(val)))
    results.sort()

    print(f"  {'후보':35s}  {'값':>20}  {'오차%':>12}")
    print(sep2)
    for err, label, val in results[:20]:
        marker = "★★" if err < 0.1 else ("★" if err < 1.0 else "")
        print(f"  {label:35s}  {val:>20.12e}  {err:>12.6f}%  {marker}")

    # ── [3] 갭/K_EQI 비율 구조 ───────────────────────────────────────────────
    print()
    print("  [3] 갭×K_EQI 비율 구조")
    print(sep2)
    gap_times_K = gap_12 * K_EQI
    print(f"  갭×K_EQI              = {nstr(gap_times_K, 20)}")
    print(f"  GAP                   = {nstr(GAP, 20)}")
    print(f"  갭×K_EQI/GAP          = {nstr(gap_times_K/GAP, 20)}")
    print(f"  갭×K_EQI − GAP        = {nstr(gap_times_K - GAP, 20)}")
    print(f"  (갭×K_EQI − GAP)/GAP  = {nstr((gap_times_K-GAP)/GAP, 20)}")
    print()
    # 갭×K_EQI = GAP + ? 구조
    residual_K = gap_times_K - GAP
    print(f"  잔류 = 갭×K_EQI − GAP = {nstr(residual_K, 20)}")
    print(f"  잔류/GAP_D2           = {nstr(residual_K/GAP_D2, 20)}")
    print(f"  잔류/GAP²             = {nstr(residual_K/GAP**2, 20)}")
    print(f"  잔류/R_membrane       = {nstr(residual_K/R_membrane, 20)}")
    print(f"  잔류/δ                = {nstr(residual_K/delta, 20)}")
    print(f"  잔류/TARGET           = {nstr(residual_K/TARGET, 20)}")

    # ── [4] GAP/K_EQI + 보정항 → 갭 닫힘 ────────────────────────────────────
    print()
    print("  [4] GAP/K_EQI + 보정항 → 갭 완전 닫힘 탐색")
    print(sep2)
    print(f"  갭 = GAP/K_EQI + Δ")
    print(f"  Δ = {nstr(diff, 20)}")
    print()
    # Δ를 K_EQI로 표현
    print(f"  Δ×K_EQI              = {nstr(diff*K_EQI, 20)}")
    print(f"  Δ×K_EQI/GAP          = {nstr(diff*K_EQI/GAP, 20)}")
    print(f"  Δ×K_EQI/GAP_D2       = {nstr(diff*K_EQI/GAP_D2, 20)}")
    print()
    # 갭 = GAP/K_EQI × (1 + Δ/GAP_over_K)
    correction = diff / GAP_over_K
    print(f"  갭 = GAP/K_EQI × (1 + {nstr(correction, 15)})")
    print(f"  보정인수              = {nstr(1+correction, 20)}")
    print(f"  보정인수 − 1         = {nstr(correction, 20)}")
    print(f"  보정인수/GAP_D2       = {nstr(correction/GAP_D2, 15)}")
    print(f"  보정인수/GAP          = {nstr(correction/GAP, 15)}")
    print(f"  보정인수×K_EQI        = {nstr(correction*K_EQI, 15)}")
    print(f"  보정인수×int_x        = {nstr(correction*int_x, 15)}")

    # ── [5] K_EQI 분해 → 갭 연결 ─────────────────────────────────────────────
    print()
    print("  [5] K_EQI 분해 구조 vs 갭")
    print(sep2)
    # K_EQI = (T_Is_D1 - γ - 2GD²exp(GD)) / lnφ
    num_K = T_Is_D1 - gamma_true - 2*GAP_D2**2*exp(GAP_D2)
    print(f"  K_EQI 분자            = {nstr(num_K, 20)}")
    print(f"  K_EQI 분모(lnφ)       = {nstr(LN_PHI, 20)}")
    print(f"  K_EQI                 = {nstr(K_EQI, 20)}")
    print()
    # GAP/K_EQI = GAP×lnφ / (T_Is_D1 - γ - 2GD²exp(GD))
    print(f"  GAP/K_EQI = GAP×lnφ / K_EQI분자")
    print(f"  GAP×lnφ              = {nstr(GAP*LN_PHI, 20)}")
    print(f"  GAP×lnφ/K_EQI분자   = {nstr(GAP*LN_PHI/num_K, 20)}")
    print(f"  vs 갭 오차%           = {float(pct_err(GAP*LN_PHI/num_K, gap_12)):.6f}%")
    print()
    # 갭 = GAP×lnφ / K_EQI분자 + 보정?
    diff2 = gap_12 - GAP*LN_PHI/num_K
    print(f"  보정2 = 갭 − GAP×lnφ/K분자 = {nstr(diff2, 20)}")
    print(f"  보정2/갭              = {nstr(diff2/gap_12, 20)}")

    # ── [6] 갭×K_EQI = GAP + ? ───────────────────────────────────────────────
    print()
    print("  [6] 갭×K_EQI = GAP + 잔류 구조 정밀 분해")
    print(sep2)
    # 갭×K_EQI = GAP + residual_K
    # residual_K 스캔
    TV6 = residual_K
    print(f"  잔류 = 갭×K_EQI − GAP = {nstr(TV6, 20)}")
    print()
    cands6 = {
        "GAP×GAP_D2"         : GAP*GAP_D2,
        "GAP²/2"             : GAP**2/2,
        "GAP_D2²×K_EQI"      : GAP_D2**2*K_EQI,
        "GAP_D2×LN_PHI"      : GAP_D2*LN_PHI,
        "GAP×LN_PHI/int_x"   : GAP*LN_PHI/int_x,
        "LN2×GAP_D2"         : LN2*GAP_D2,
        "LN_PHI×GAP_D2"      : LN_PHI*GAP_D2,
        "2×GAP_D2²×exp(GAP_D2)/LN_PHI" : 2*GAP_D2**2*exp(GAP_D2)/LN_PHI,
        "gamma_true×GAP_D2"  : gamma_true*GAP_D2,
        "gamma_true×GAP²"    : gamma_true*GAP**2,
        "T_Is_D1×GAP_D2²"    : T_Is_D1*GAP_D2**2,
        "R_membrane×K_EQI"   : R_membrane*K_EQI,
        "δ×GAP_D2"           : delta*GAP_D2,
        "TARGET×GAP_D2"      : TARGET*GAP_D2,
        "frac_x×GAP²"        : frac_x*GAP**2,
        "GAP/K_EQI×GAP_D2"   : GAP/K_EQI*GAP_D2,
        "GAP/K_EQI×GAP"      : GAP/K_EQI*GAP,
        "sigma³×GAP"         : sigma**3*GAP,
        "sigma×GAP²"         : sigma*GAP**2,
        "alpha_S38×GAP_D2²"  : alpha_S38*GAP_D2**2,
    }
    r6 = [(float(pct_err(v, TV6)), k, float(v)) for k,v in cands6.items()]
    r6.sort()
    print(f"  {'후보':35s}  {'값':>20}  {'오차%':>12}")
    print(sep2)
    for err, label, val in r6[:15]:
        marker = "★★" if err < 0.1 else ("★" if err < 2.0 else "")
        print(f"  {label:35s}  {val:>20.12e}  {err:>12.6f}%  {marker}")

    # ── [7] (갭−GAP/K_EQI)/갭 = 0.40% 정체 ─────────────────────────────────
    print()
    print("  [7] (갭−GAP/K_EQI)/갭 = 0.40% 정체")
    print(sep2)
    TV7 = diff / gap_12  # = 0.40355...%/100
    print(f"  목표 = {nstr(TV7, 20)}")
    print(f"  = {float(TV7*100):.8f}%")
    print()
    cands7 = {
        "GAP_D2/K_EQI"            : GAP_D2/K_EQI,
        "GAP/K_EQI²/gap_12"       : GAP/K_EQI**2/gap_12,
        "GAP²/(K_EQI×gap_12)"     : GAP**2/(K_EQI*gap_12),
        "1/(K_EQI×int_x)"         : 1/(K_EQI*int_x),
        "GAP_D2/int_x"            : GAP_D2/int_x,
        "GAP/int_x/K_EQI"         : GAP/int_x/K_EQI,
        "LN_PHI/K_EQI/int_x"      : LN_PHI/K_EQI/int_x,
        "sigma/K_EQI/int_x"       : sigma/K_EQI/int_x,
        "R_membrane/gap_12"        : R_membrane/gap_12,
        "delta×GAP_D2/gap_12"     : delta*GAP_D2/gap_12,
        "alpha_S38×GAP_D2³/gap_12": alpha_S38*GAP_D2**3/gap_12,
        "GAP_D2²/GAP"             : GAP_D2**2/GAP,
        "sigma³/K_EQI"            : sigma**3/K_EQI,
        "LN2/K_EQI/int_x"         : LN2/K_EQI/int_x,
        "1/(K_EQI×frac_x/GAP_D2)" : 1/(K_EQI*frac_x/GAP_D2),
    }
    r7 = [(float(pct_err(v, TV7)), k, float(v)) for k,v in cands7.items()]
    r7.sort()
    print(f"  {'후보':35s}  {'값':>20}  {'오차%':>12}")
    print(sep2)
    for err, label, val in r7[:12]:
        marker = "★★" if err < 0.1 else ("★" if err < 2.0 else "")
        print(f"  {label:35s}  {val:>20.12e}  {err:>12.6f}%  {marker}")

    # ── [8] 병렬 후보 비교 ───────────────────────────────────────────────────
    print()
    print("  [8] 탐색 가치 후보 동시 비교")
    print(sep2)
    print(f"  갭 = {nstr(gap_12, 20)}")
    print()
    print(f"  {'후보':30s}  {'값':>20}  {'오차%':>10}  {'비고'}")
    print(sep2)
    parallel = [
        ("GAP/K_EQI",          GAP/K_EQI,           "★ EQI 내부"),
        ("1/83",               mpf(1)/83,            "정수"),
        ("1/84",               mpf(1)/84,            "정수"),
        ("GAP/K_EQI+GAP²",     GAP/K_EQI+GAP**2,    "2항"),
        ("GAP/K_EQI+GAP×GD",   GAP/K_EQI+GAP*GAP_D2,"2항"),
        ("GAP/K_EQI+GD²",      GAP/K_EQI+GAP_D2**2, "2항"),
        ("GAP/K_EQI+LN_PHI×GD²", GAP/K_EQI+LN_PHI*GAP_D2**2, "2항"),
        ("5×GAP²",             5*GAP**2,             "5×GAP²"),
        ("4×GAP²",             4*GAP**2,             "4×GAP²"),
        ("6×GAP²",             6*GAP**2,             "6×GAP²"),
        ("GAP×(1-GAP)/K_EQI",  GAP*(1-GAP)/K_EQI,   "2항"),
    ]
    for label, val, note in parallel:
        err = float(pct_err(val, gap_12))
        marker = "★★" if err < 0.1 else ("★" if err < 0.5 else "")
        print(f"  {label:30s}  {float(val):>20.12f}  {err:>10.6f}%  {note} {marker}")

    # ── [9] sinc² 연결 ───────────────────────────────────────────────────────
    print()
    print("  [9] (1-sinc²(k=1))/갭 vs 3π 정밀")
    print(sep2)
    ratio_sinc = one_P / gap_12
    print(f"  (1-P)/갭    = {nstr(ratio_sinc, 20)}")
    print(f"  3π          = {nstr(3*pi, 20)}")
    print(f"  오차%       = {float(pct_err(ratio_sinc, 3*pi)):.6f}%")
    print()
    # 3π − (1-P)/갭 잔류
    diff_pi = 3*pi - ratio_sinc
    print(f"  3π − (1-P)/갭 = {nstr(diff_pi, 20)}")
    print(f"  잔류/GAP      = {nstr(diff_pi/GAP, 15)}")
    print(f"  잔류/GAP_D2   = {nstr(diff_pi/GAP_D2, 15)}")
    print(f"  잔류/K_EQI    = {nstr(diff_pi/K_EQI, 15)}")
    print(f"  잔류×int_x    = {nstr(diff_pi*int_x, 15)}")

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  ▶▶ GAP/K_EQI vs 갭 탐색 요약")
    print(sep)
    print(f"  갭           = {nstr(gap_12, 20)}")
    print(f"  GAP/K_EQI   = {nstr(GAP_over_K, 20)}")
    print(f"  차이 Δ      = {nstr(diff, 20)}")
    print(f"  갭×K_EQI    = {nstr(gap_times_K, 20)}")
    print(f"  갭×K_EQI−GAP= {nstr(residual_K, 20)}")
    print()
    print("  최유력 닫힘 후보:")
    best = [
        ("GAP/K_EQI",          GAP/K_EQI),
        ("GAP/K_EQI+GAP²",     GAP/K_EQI+GAP**2),
        ("GAP/K_EQI+GAP×GD",   GAP/K_EQI+GAP*GAP_D2),
        ("5×GAP²",             5*GAP**2),
    ]
    for label, val in best:
        print(f"  {label:30s} = {float(val):.15f}  오차 {float(pct_err(val,gap_12)):.4f}%")
    print(sep)


if __name__ == "__main__":
    main()

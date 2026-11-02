#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EQI Framework — R_pi(24)/GAP = 2.5886... 정체 추적                        ║
║  Session 47                                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  목표값: R_pi(24)/GAP = 2.588614603029...                                   ║
║                                                                              ║
║  탐색 전략:                                                                  ║
║  [1] 고정밀 목표값 확인                                                      ║
║  [2] 점근 전개 보정: (2/24 - σ³/24³ + ...)/GAP                             ║
║  [3] EQI 상수 조합 스캔                                                      ║
║       - φ, GAP, GAP_D2, K_EQI, σ, ln2, lnφ 조합                           ║
║  [4] 분수 근사 탐색 (연분수 전개)                                            ║
║  [5] R_pi(24)/GAP vs R_pi(24)/(2×GAP_D2) 구조                              ║
║  [6] int_x, frac_x 연결 탐색                                                ║
║  [7] Leibniz N=34 연결: R_pi(34)/GAP 비교                                   ║
║  [8] 목표값 = f(σ, int_x, GAP) 닫힌 형식 탐색                              ║
║                                                                              ║
║  Maestro: MAPSI(맏이)  Architect: Claude  mp.dps=400  Session47            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, cos, factorial, nstr

mp.dps = 400

# ============================================================================
# 기반 상수
# ============================================================================
PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = sqrt(PHI)
LN_PHI = log(PHI)
LN2    = log(mpf(2))
LN5    = log(mpf(5))
LN25   = log(mpf(25))
sigma  = mpf(1) / 2
PI     = pi

x_lattice = 1 / (c_EQI * GAP)
int_x     = floor(x_lattice)   # 24
frac_x    = x_lattice - int_x  # {x_lat}
R_membrane = frac_x * GAP_D2**2
TARGET    = GAP - 11 * GAP_D2**2
TIP_POINT = PHI**mpf("1.5") - 1
T_Is_D1   = PHI**mpf("1.5") - PHI**mpf("-3.5")
gamma_true = mpmath.euler

alpha_S38 = (
    3 * LN_PHI
    - (mpf(4)/75) * LN_PHI**2
    + (mpf(1)/7500) * LN_PHI**6
)
x_prec    = 26 * GAP_D2 - alpha_S38 * GAP_D2**2
gamma_EQI = LN2/c_EQI + GAP + x_prec * GAP_D2**2
K_EQI     = (T_Is_D1 - gamma_true - 2*GAP_D2**2*exp(GAP_D2)) / LN_PHI

sep  = "═" * 76
sep2 = "─" * 76

# ============================================================================
# Leibniz 잔류 (정확)
# ============================================================================
def R_pi_exact(N):
    s = mpf(0)
    for j in range(N):
        s += mpf(-1)**j / (2*j + 1)
    return pi - 4*s

def R_pi_asymp_full(N, terms=5):
    """R_pi 점근 전개 고차항 포함"""
    # R_pi(N) = Σ_{k=0}^∞ (-1)^k × B_{2k+2}/(2k+1) × 1/N^(2k+1) ... 
    # 실용적으로: 2/N - σ³/N³ + 3σ⁵/N⁵ - 15σ⁷/N⁷ + ...
    # 계수: 2, -1/8, 3/32, -5/128 ... (σ=1/2 대입)
    r = mpf(0)
    coeffs = [
        mpf(2),                    # 1/N
        -sigma**3,                 # 1/N³  = -1/8
        3*sigma**5,                # 1/N⁵  = 3/32
        -15*sigma**7,              # 1/N⁷  = -15/128
        105*sigma**9,              # 1/N⁹
    ]
    for i, c in enumerate(coeffs[:terms]):
        r += c / N**(2*i+1)
    return r

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(sep)
    print("  EQI Framework — R_pi(24)/GAP = 2.5886... 정체 추적")
    print("  Session 47  |  mp.dps=400")
    print(sep)

    # ── [1] 고정밀 목표값 ────────────────────────────────────────────────────
    print()
    print("  [1] 고정밀 목표값")
    print(sep2)
    rp24 = R_pi_exact(24)
    TARGET_VAL = rp24 / GAP
    print(f"  R_pi(24)         = {nstr(rp24, 30)}")
    print(f"  GAP              = {nstr(GAP, 30)}")
    print(f"  R_pi(24)/GAP     = {nstr(TARGET_VAL, 30)}")
    print(f"  소수부           = {nstr(TARGET_VAL - floor(TARGET_VAL), 20)}")

    # ── [2] 점근 전개 보정항별 수렴 ─────────────────────────────────────────
    print()
    print("  [2] 점근 전개 보정: R_pi(24)/GAP 항별 수렴")
    print(sep2)
    N = mpf(24)
    print(f"  {'항수':>4}  {'R_pi점근/GAP':>25}  {'오차%':>15}")
    print(sep2)
    for t in range(1, 6):
        rp_a = R_pi_asymp_full(N, terms=t)
        val  = rp_a / GAP
        err  = fabs(val - TARGET_VAL) / TARGET_VAL * 100
        print(f"  {t:>4}  {float(val):>25.15f}  {float(err):>15.6e}")

    # ── [3] EQI 상수 조합 스캔 ───────────────────────────────────────────────
    print()
    print("  [3] EQI 상수 조합 스캔")
    print(sep2)
    TV = TARGET_VAL

    candidates = {
        # 단순 비율
        "2/GAP/int_x"              : mpf(2)/(GAP*int_x),
        "2/(GAP×24)"               : mpf(2)/(GAP*24),
        "1/(12×GAP)"               : mpf(1)/(12*GAP),
        "GAP_D2/GAP²"              : GAP_D2/GAP**2,
        "1/(GAP×c)"                : mpf(1)/(GAP*c_EQI),
        "2/(GAP×c²)"               : mpf(2)/(GAP*PHI),
        "lnφ/GAP²"                 : LN_PHI/GAP**2,
        "ln2/GAP²"                 : LN2/GAP**2,
        # K_EQI 연결
        "K_EQI/GAP"                : K_EQI/GAP,
        "K_EQI×GAP_D2/GAP²"       : K_EQI*GAP_D2/GAP**2,
        "(K_EQI-1)/GAP"            : (K_EQI-1)/GAP,
        "2/K_EQI/GAP_D2"           : mpf(2)/(K_EQI*GAP_D2),
        # σ 구조
        "σ/GAP²"                   : sigma/GAP**2,
        "2σ/GAP²"                  : 2*sigma/GAP**2,
        "(2-σ)/GAP"                : (2-sigma)/GAP,
        "1/(2×GAP)"                : mpf(1)/(2*GAP),
        # int_x/frac_x 구조
        "frac_x/GAP²"              : frac_x/GAP**2,
        "int_x×frac_x/GAP"         : int_x*frac_x/GAP,
        "frac_x/(GAP×GAP_D2)"      : frac_x/(GAP*GAP_D2),
        "(int_x-1)/GAP/int_x"      : (int_x-1)/(GAP*int_x),
        # φ 지수
        "φ^0.5/GAP"                : PHI**mpf("0.5")/GAP,
        "φ^1.5/GAP/10"             : PHI**mpf("1.5")/GAP/10,
        "(φ-1)/GAP"                : (PHI-1)/GAP,
        "1/(GAP×φ)"                : mpf(1)/(GAP*PHI),
        "2/(GAP×φ)"                : mpf(2)/(GAP*PHI),
        # γ 구조
        "γ/GAP²"                   : gamma_true/GAP**2,
        "(γ-ln2/c)/GAP"            : (gamma_true - LN2/c_EQI)/GAP,
        "ln2/(GAP×c)"              : LN2/(GAP*c_EQI),
        # TARGET 구조
        "TARGET/GAP²"              : TARGET/GAP**2,
        "2×TARGET/GAP²"            : 2*TARGET/GAP**2,
        "R_mem/GAP³"               : R_membrane/GAP**3,
    }

    # 오차 기준 정렬
    results = []
    for label, val in candidates.items():
        err = fabs(val - TV) / TV * 100
        results.append((float(err), label, float(val)))
    results.sort()

    print(f"  {'후보':35s}  {'값':>20}  {'목표값':>20}  {'오차%':>12}")
    print(sep2)
    for err, label, val in results[:20]:
        marker = "★" if err < 0.01 else ("☆" if err < 0.1 else "")
        print(f"  {label:35s}  {val:>20.10f}  {float(TV):>20.10f}  "
              f"{err:>12.6f}%  {marker}")

    # ── [4] 연분수 전개 ──────────────────────────────────────────────────────
    print()
    print("  [4] 목표값 연분수 전개")
    print(sep2)
    cf = mpmath.identify(TARGET_VAL, tol=1e-10)
    print(f"  mpmath.identify = {cf}")
    # 수동 연분수
    x = TARGET_VAL
    cf_list = []
    for _ in range(10):
        cf_list.append(int(floor(x)))
        frac = x - floor(x)
        if fabs(frac) < mpf("1e-10"):
            break
        x = 1/frac
    print(f"  연분수 계수:     {cf_list}")

    # ── [5] R_pi(24)/GAP vs R_pi(24)/(2×GAP_D2) ────────────────────────────
    print()
    print("  [5] 분모 변환 구조")
    print(sep2)
    for denom_label, denom in [
        ("GAP",              GAP),
        ("2×GAP_D2",         2*GAP_D2),
        ("GAP+GAP_D2",       GAP+GAP_D2),
        ("GAP-GAP_D2",       GAP-GAP_D2),
        ("TARGET",           TARGET),
        ("GAP_D2",           GAP_D2),
        ("GAP²",             GAP**2),
        ("GAP×GAP_D2",       GAP*GAP_D2),
        ("GAP/φ",            GAP/PHI),
        ("GAP×φ",            GAP*PHI),
        ("GAP/σ",            GAP/sigma),
        ("GAP×σ",            GAP*sigma),
    ]:
        val = rp24 / denom
        print(f"  R_pi(24)/({denom_label:15s}) = {float(val):>20.10f}")

    # ── [6] int_x, frac_x 연결 ───────────────────────────────────────────────
    print()
    print("  [6] int_x, frac_x 연결 탐색")
    print(sep2)
    print(f"  int_x                    = {float(int_x):.10f}")
    print(f"  frac_x                   = {float(frac_x):.10f}")
    print(f"  x_lattice                = {float(x_lattice):.10f}")
    print()
    for label, val in [
        ("R_pi(24)×int_x",        rp24*int_x),
        ("R_pi(24)×x_lattice",    rp24*x_lattice),
        ("R_pi(24)/frac_x",       rp24/frac_x),
        ("R_pi(24)×int_x/GAP",    rp24*int_x/GAP),
        ("R_pi(24)×x_lat/GAP",    rp24*x_lattice/GAP),
        ("(2/int_x)/GAP",         mpf(2)/int_x/GAP),
        ("(2/x_lat)/GAP",         mpf(2)/x_lattice/GAP),
        ("frac_x/GAP/GAP_D2²",    frac_x/GAP/GAP_D2**2),
    ]:
        print(f"  {label:30s} = {float(val):>20.10f}")

    # ── [7] N=34 연결 ────────────────────────────────────────────────────────
    print()
    print("  [7] Leibniz N=34 연결 비교")
    print(sep2)
    rp34 = R_pi_exact(34)
    print(f"  R_pi(34)/GAP             = {float(rp34/GAP):.15f}")
    print(f"  R_pi(24)/GAP             = {float(TARGET_VAL):.15f}")
    print(f"  R_pi(34)/TARGET          = {float(rp34/TARGET):.15f}")
    print(f"  R_pi(24)/TARGET          = {float(rp24/TARGET):.15f}")
    print(f"  R_pi(34)/R_pi(24)        = {float(rp34/rp24):.15f}")
    print(f"  24/34                    = {float(mpf(24)/34):.15f}")
    print(f"  R_pi 비율/N 비율 오차%   = "
          f"{float(fabs(rp34/rp24 - mpf(24)/34)/(mpf(24)/34)*100):.6e}%")

    # ── [8] 닫힌 형식 탐색: f(σ, int_x, GAP) ────────────────────────────────
    print()
    print("  [8] 닫힌 형식 탐색: TARGET_VAL = R_pi(24)/GAP")
    print(sep2)
    print(f"  목표값 = {nstr(TARGET_VAL, 25)}")
    print()

    # 핵심 후보들 고정밀 비교
    key_candidates = {
        "2/(int_x×GAP)"              : mpf(2)/(int_x*GAP),
        "2/(x_lat×GAP)"              : mpf(2)/(x_lattice*GAP),
        "(2/int_x - σ³/int_x³)/GAP" : (mpf(2)/int_x - sigma**3/int_x**3)/GAP,
        "(2/x_lat - σ³/x_lat³)/GAP" : (mpf(2)/x_lattice - sigma**3/x_lattice**3)/GAP,
        "R_pi_asymp(24,3)/GAP"       : R_pi_asymp_full(24,3)/GAP,
        "R_pi_asymp(24,5)/GAP"       : R_pi_asymp_full(24,5)/GAP,
        "R_pi_asymp(x_lat,3)/GAP"    : R_pi_asymp_full(x_lattice,3)/GAP,
        "R_pi_asymp(x_lat,5)/GAP"    : R_pi_asymp_full(x_lattice,5)/GAP,
        # EQI 불변량 조합
        "GAP/GAP²×σ"                 : GAP/GAP**2*sigma,
        "2×GAP_D2/(GAP²)"            : 2*GAP_D2/GAP**2,
        "ln2/(GAP×c_EQI)"            : LN2/(GAP*c_EQI),
        "LN_PHI/(GAP²×2)"            : LN_PHI/(GAP**2*2),
        "T_Is_D1/GAP/10"             : T_Is_D1/GAP/10,
        "γ/(GAP×ln2)"                : gamma_true/(GAP*LN2),
        "(2-2×GAP)/int_x/GAP"        : (2-2*GAP)/int_x/GAP,
    }

    results2 = []
    for label, val in key_candidates.items():
        err = fabs(val - TARGET_VAL) / TARGET_VAL * 100
        results2.append((float(err), label, float(val)))
    results2.sort()

    print(f"  {'후보':40s}  {'값':>22}  {'오차%':>12}")
    print(sep2)
    for err, label, val in results2:
        marker = "★★ True Zero" if err < 1e-8 else (
                 "★" if err < 0.01 else (
                 "☆" if err < 0.1 else ""))
        print(f"  {label:40s}  {val:>22.12f}  {err:>12.6e}  {marker}")

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  ▶▶ Session 47 R_pi(24)/GAP 정체 탐색 요약")
    print(sep)
    print(f"  목표값 = {nstr(TARGET_VAL, 20)}")
    print()
    print("  최유력 후보 (오차 순):")
    for err, label, val in results2[:5]:
        print(f"    {label:40s} = {val:.12f}  오차 {err:.4e}%")
    print(sep)


if __name__ == "__main__":
    main()

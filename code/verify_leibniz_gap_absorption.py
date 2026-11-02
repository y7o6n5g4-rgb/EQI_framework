#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EQI Framework — Leibniz 잔류 GAP 흡수 대수 경로 검증                      ║
║  Session 47                                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  핵심 가설:                                                                  ║
║    R_pi(N) = 2/N - σ³/N³ + O(N⁻⁵)  [S43 확정, σ=1/2]                     ║
║                                                                              ║
║  경로 A:  2/N = GAP × K_EQI  →  N ≈ 24 = int_x = 4!                       ║
║    K_EQI = 2.6907978271...  [S44 확정]                                      ║
║    int_x = floor(1/(c×GAP)) = 24                                            ║
║                                                                              ║
║  경로 B:  R_pi(N=34) = TARGET×(2+δ)  [S43 기존 확정]                       ║
║    → sinc² 잔류도 δ 경로로 흡수 가능성 탐색                                 ║
║                                                                              ║
║  경로 C:  n_ref,k × R_pi → GAP_D2^(2k+1) 직접 비교                        ║
║                                                                              ║
║  검증 항목:                                                                  ║
║  [1] K_EQI 재계산 (S44 확정값)                                              ║
║  [2] 경로 A: 2/(GAP×K_EQI) vs int_x=24 오차%                               ║
║  [3] R_pi(N) 점근 구조: 2/N - σ³/N³ 정밀도                                 ║
║  [4] R_pi(N=24) vs GAP×K_EQI 직접 비교                                     ║
║  [5] R_pi(N=24) 구조 분해: GAP, GAP_D2, R_membrane 관계                    ║
║  [6] 경로 B: sinc²잔류 vs δ 구조 비교                                       ║
║  [7] 경로 C: n_ref,k × R_pi vs GAP_D2^(2k+1)                              ║
║  [8] 완전 흡수 조건: sinc²(완전급수) True Zero 달성 J값                     ║
║  [9] 통합: Leibniz 잔류 → R_membrane → δ → sinc² 연결 고리                 ║
║                                                                              ║
║  Maestro: MAPSI(맏이)  Architect: Claude  mp.dps:400  Session47            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, cos, factorial

mp.dps = 400

# ============================================================================
# 기반 상수 (v7.9 동일)
# ============================================================================
PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = sqrt(PHI)
LN_PHI = log(PHI)
LN2    = log(mpf(2))
sigma  = mpf(1) / 2   # 리만 임계선

# v7.9 상수
x_lattice = 1 / (c_EQI * GAP)
int_x     = floor(x_lattice)          # 24 = 4!
frac_x    = x_lattice - int_x         # {x_lat}
R_membrane = frac_x * GAP_D2**2
TARGET    = GAP - 11 * GAP_D2**2
TIP_POINT = PHI**mpf("1.5") - 1

sep  = "═" * 76
sep2 = "─" * 76

# ============================================================================
# K_EQI (S44 확정)
# K_EQI = (T_Is_D1 - γ - 2×GAP_D2²×exp(GAP_D2)) / ln(φ)
# ============================================================================
T_Is_D1 = PHI**mpf("1.5") - PHI**mpf("-3.5")
gamma_true = mpmath.euler

# γ_EQI (§1 v7.9)
alpha_S38 = (
    3 * LN_PHI
    - (mpf(4)/75) * LN_PHI**2
    + (mpf(1)/7500) * LN_PHI**6
)
x_prec    = 26 * GAP_D2 - alpha_S38 * GAP_D2**2
gamma_EQI = LN2/c_EQI + GAP + x_prec * GAP_D2**2

K_EQI = (T_Is_D1 - gamma_true - 2*GAP_D2**2*exp(GAP_D2)) / LN_PHI

# ============================================================================
# Leibniz 급수 및 잔류
# ============================================================================

def leibniz_pi(J):
    s = mpf(0)
    for j in range(J):
        s += mpf(-1)**j / (2*j + 1)
    return 4 * s

def R_pi_exact(N):
    """R_pi(N) = π - Leibniz(N) [정확한 잔류]"""
    return pi - leibniz_pi(N)

def R_pi_asymp(N, terms=3):
    """R_pi 점근 전개: 2/N - σ³/N³ + σ⁵×3/N⁵ [S43 확정]"""
    r = mpf(2) / N
    if terms >= 2:
        r -= sigma**3 / N**3
    if terms >= 3:
        r += 3 * sigma**5 / N**5
    return r

# ============================================================================
# sinc² Taylor
# ============================================================================

def taylor_sinc(x, M=50):
    if fabs(x) < mpf("1e-300"):
        return mpf(1)
    s = mpf(0)
    for m in range(M):
        s += mpf(-1)**m * x**(2*m) / factorial(2*m+1)
    return s

def sinc2_sym(k):
    x = mpf(2*k+1) / mpf(4)**(k+1) * pi
    if fabs(x) < mpf("1e-300"):
        return mpf(1)
    return (sin(x)/x)**2

def n_ref(k):
    return mpf(2*k+1) / mpf(4)**(k+1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(sep)
    print("  EQI Framework — Leibniz 잔류 GAP 흡수 대수 경로 검증")
    print("  Session 47  |  mp.dps=400")
    print(sep)

    # ── [1] K_EQI 재확인 ────────────────────────────────────────────────────
    print()
    print("  [1] K_EQI 재확인 (S44 확정값)")
    print(sep2)
    print(f"  K_EQI         = {mpmath.nstr(K_EQI, 20)}")
    print(f"  S44 확정값    = 2.6907978271...")
    err_K = fabs(K_EQI - mpf("2.6907978271")) / mpf("2.6907978271") * 100
    print(f"  재현 오차%    = {float(err_K):.6e}")

    # ── [2] 경로 A: 2/(GAP×K_EQI) vs int_x=24 ──────────────────────────────
    print()
    print("  [2] 경로 A: 2/(GAP×K_EQI) vs int_x=24")
    print(sep2)
    N_A = 2 / (GAP * K_EQI)
    err_A = fabs(N_A - int_x) / int_x * 100
    print(f"  2/(GAP×K_EQI) = {float(N_A):.15f}")
    print(f"  int_x         = {int(int_x)}")
    print(f"  차이%         = {float(err_A):.6e}")
    print(f"  x_lattice     = {float(x_lattice):.15f}")
    print(f"  1/(c×GAP)     = {float(x_lattice):.15f}")
    # 추가: 2/(GAP×K_EQI) vs x_lattice
    err_xlat = fabs(N_A - x_lattice) / x_lattice * 100
    print(f"  vs x_lattice 차이% = {float(err_xlat):.6e}")

    # ── [3] R_pi(N) 점근 구조 정밀도 ────────────────────────────────────────
    print()
    print("  [3] R_pi(N) 점근 구조: 2/N - σ³/N³ 정밀도")
    print(sep2)
    print(f"  {'N':>6}  {'R_pi(정확)':>22}  {'점근(2항)':>22}  {'오차%':>15}")
    print(sep2)
    for N in [10, 24, 34, 100, 500, 1000]:
        rp_exact = R_pi_exact(N)
        rp_asymp = R_pi_asymp(N, terms=2)
        err = fabs(rp_exact - rp_asymp) / fabs(rp_exact) * 100
        print(f"  {N:>6}  {float(rp_exact):>22.15f}  "
              f"{float(rp_asymp):>22.15f}  {float(err):>15.6e}")

    # ── [4] R_pi(N=24) vs GAP×K_EQI ─────────────────────────────────────────
    print()
    print("  [4] R_pi(N=24) vs GAP×K_EQI 직접 비교")
    print(sep2)
    N24 = int(int_x)  # 24
    rp24 = R_pi_exact(N24)
    gap_keqi = GAP * K_EQI
    err_4 = fabs(rp24 - gap_keqi) / fabs(gap_keqi) * 100
    print(f"  R_pi(N=24)    = {float(rp24):.15f}")
    print(f"  GAP×K_EQI     = {float(gap_keqi):.15f}")
    print(f"  오차%         = {float(err_4):.6e}")
    # 비율
    ratio_4 = rp24 / gap_keqi
    print(f"  비율 R_pi/GAP×K_EQI = {float(ratio_4):.15f}")
    # 추가 후보: R_pi(24) vs 2/int_x
    print(f"  2/int_x(=2/24)= {float(mpf(2)/24):.15f}")
    err_2over24 = fabs(rp24 - mpf(2)/24) / fabs(mpf(2)/24) * 100
    print(f"  R_pi vs 2/24 오차% = {float(err_2over24):.6e}")

    # ── [5] R_pi(N=24) 구조 분해 ─────────────────────────────────────────────
    print()
    print("  [5] R_pi(N=24) 구조 분해: GAP, GAP_D2, R_membrane 관계")
    print(sep2)
    rp24_val = float(rp24)
    print(f"  R_pi(24)             = {float(rp24):.15f}")
    print(f"  GAP                  = {float(GAP):.15f}")
    print(f"  GAP_D2               = {float(GAP_D2):.15f}")
    print(f"  2×GAP_D2             = {float(2*GAP_D2):.15f}")
    print(f"  GAP+GAP_D2           = {float(GAP+GAP_D2):.15f}")
    print(f"  R_membrane           = {float(R_membrane):.15f}")
    print(f"  TARGET               = {float(TARGET):.15f}")
    print()
    # 비율들
    for label, val in [
        ("R_pi(24)/GAP",         rp24/GAP),
        ("R_pi(24)/GAP_D2",      rp24/GAP_D2),
        ("R_pi(24)/2GAP_D2",     rp24/(2*GAP_D2)),
        ("R_pi(24)/R_membrane",  rp24/R_membrane),
        ("R_pi(24)/TARGET",      rp24/TARGET),
        ("R_pi(24)/(GAP×K_EQI)", rp24/gap_keqi),
    ]:
        print(f"  {label:30s} = {float(val):.10f}")

    # ── [6] 경로 B: sinc² 잔류 vs δ 구조 ───────────────────────────────────
    print()
    print("  [6] 경로 B: sinc² 잔류(J=2000) vs δ 구조")
    print(sep2)
    J_use = 2000
    rp_J = R_pi_exact(J_use)

    # Leibniz 잔류가 sinc²에 미치는 1차 보정
    # ΔP_k ≈ 2 × sinc(x_k) × sinc'(x_k) × n_ref,k × R_pi(J)
    print(f"  R_pi(J=2000) = {float(rp_J):.15f}")
    print()
    print(f"  {'k':>3}  {'포논':>6}  {'sinc²잔류(측정)':>18}  "
          f"{'sinc²잔류(1차근사)':>20}  {'오차%':>12}  {'δ비교':>12}")
    print(sep2)

    # δ (v7.9 확정값)
    def _leibniz_residual(N):
        s = mpf(0)
        for k_l in range(N):
            s += mpf(1)/(4*k_l+1) - mpf(1)/(4*k_l+3)
        return 4*s
    R34 = _leibniz_residual(34)
    delta_num = R34 / TARGET - 2

    for k in range(1, 7):
        nr = n_ref(k)
        x_sym = nr * pi
        x_ser = nr * leibniz_pi(J_use)

        # sinc² 잔류 (측정)
        s2_sym = sinc2_sym(k)
        s2_ser = taylor_sinc(x_ser, 50)**2
        dP_meas = s2_sym - s2_ser

        # 1차 근사: dP ≈ 2×sinc×sinc'×n_ref×R_pi(J)
        # sinc'(x) = (cos(x) - sinc(x))/x
        sx = sin(x_sym)/x_sym if fabs(x_sym) > mpf("1e-300") else mpf(1)
        cx = cos(x_sym)
        sinc_prime = (cx - sx) / x_sym if fabs(x_sym) > mpf("1e-300") else mpf(0)
        dP_approx = 2 * sx * sinc_prime * nr * rp_J

        err_6 = fabs(dP_meas - dP_approx) / fabs(dP_meas) * 100 if fabs(dP_meas) > 0 else mpf(0)

        # δ와 비율
        delta_ratio = fabs(dP_meas) / fabs(delta_num) if delta_num != 0 else mpf(0)

        print(f"  {k:>3}  {2*k+1:>3}차    {float(dP_meas):>18.10e}  "
              f"{float(dP_approx):>20.10e}  {float(err_6):>12.4f}%  "
              f"{float(delta_ratio):>12.6e}")

    # ── [7] 경로 C: n_ref,k × R_pi vs GAP_D2^(2k+1) ────────────────────────
    print()
    print("  [7] 경로 C: n_ref,k × R_pi(J=2000) vs GAP_D2^(2k+1)")
    print(sep2)
    print(f"  {'k':>3}  {'포논':>6}  {'n_ref×R_pi':>20}  "
          f"{'GAP_D2^(2k+1)':>20}  {'비율':>15}")
    print(sep2)
    for k in range(1, 7):
        nr = n_ref(k)
        lhs = nr * rp_J
        rhs = GAP_D2**(2*k+1)
        ratio_C = lhs / rhs if rhs != 0 else mpf(0)
        print(f"  {k:>3}  {2*k+1:>3}차    {float(lhs):>20.10e}  "
              f"{float(rhs):>20.10e}  {float(ratio_C):>15.6f}")

    # ── [8] 완전 True Zero 달성 J값 탐색 ────────────────────────────────────
    print()
    print("  [8] sinc²(완전급수) True Zero 달성 J값 탐색 (k=1 기준)")
    print(sep2)
    s2_sym_1 = sinc2_sym(1)
    print(f"  sinc²(k=1, 기호기준) = {float(s2_sym_1):.15f}")
    print()
    print(f"  {'J':>8}  {'sinc²(완전급수)':>22}  {'오차%':>18}  {'판정':>12}")
    print(sep2)
    for J_test in [2000, 5000, 10000, 50000, 100000]:
        x_l = n_ref(1) * leibniz_pi(J_test)
        s2_full = taylor_sinc(x_l, 50)**2
        err_8 = fabs(s2_sym_1 - s2_full) / s2_sym_1 * 100
        judge = "★ True Zero" if float(err_8) < 1e-10 else (
                "Near TZ" if float(err_8) < 1e-5 else "")
        print(f"  {J_test:>8}  {float(s2_full):>22.15f}  "
              f"{float(err_8):>18.6e}  {judge}")

    # ── [9] 통합: Leibniz 잔류 → R_membrane → δ → sinc² 연결 고리 ─────────
    print()
    print("  [9] 통합 연결 고리 검증")
    print(sep2)
    print()
    print("  구조: R_pi(N=int_x) → R_membrane → δ → sinc² 잔류")
    print()

    rp_intx = R_pi_exact(int(int_x))  # R_pi(24)
    rp_asymp_intx = mpf(2) / int_x - sigma**3 / int_x**3

    print(f"  R_pi(24) 정확값   = {float(rp_intx):.15f}")
    print(f"  R_pi(24) 점근값   = {float(rp_asymp_intx):.15f}")
    print(f"  R_pi(24) 점근오차%= {float(fabs(rp_intx-rp_asymp_intx)/fabs(rp_intx)*100):.6e}")
    print()

    # R_pi(24) × GAP_D2 vs R_membrane
    rp24_x_gd = rp_intx * GAP_D2
    print(f"  R_pi(24) × GAP_D2      = {float(rp24_x_gd):.15f}")
    print(f"  R_membrane             = {float(R_membrane):.15f}")
    print(f"  비율                   = {float(rp24_x_gd/R_membrane):.10f}")
    print()

    # R_pi(24) × GAP_D2² vs R_membrane
    rp24_x_gd2 = rp_intx * GAP_D2**2
    print(f"  R_pi(24) × GAP_D2²     = {float(rp24_x_gd2):.15f}")
    print(f"  R_membrane             = {float(R_membrane):.15f}")
    print(f"  비율                   = {float(rp24_x_gd2/R_membrane):.10f}")
    print()

    # frac_x × GAP_D2² = R_membrane (정의)
    print(f"  frac_x                 = {float(frac_x):.15f}")
    print(f"  R_pi(24) / frac_x      = {float(rp_intx/frac_x):.10f}")
    print()

    # R_pi(24) 와 2/int_x 관계
    two_over_intx = mpf(2) / int_x
    print(f"  2/int_x(=2/24)         = {float(two_over_intx):.15f}")
    print(f"  R_pi(24)/( 2/int_x)    = {float(rp_intx/two_over_intx):.10f}")
    print()

    # sinc² k=1 잔류와 R_membrane 비율
    s2_1_sym = sinc2_sym(1)
    s2_1_ser = taylor_sinc(n_ref(1)*leibniz_pi(2000), 50)**2
    dP1 = s2_1_sym - s2_1_ser
    print(f"  sinc²잔류(k=1, J=2000) = {float(dP1):.15e}")
    print(f"  R_membrane             = {float(R_membrane):.15e}")
    print(f"  비율 sinc²잔류/R_mem   = {float(dP1/R_membrane):.10f}")
    print(f"  δ                      = {float(delta_num):.15e}")
    print(f"  비율 sinc²잔류/δ       = {float(dP1/delta_num):.10f}")

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  ▶▶ Session 47 Leibniz 잔류 GAP 흡수 대수 경로 요약")
    print(sep)
    print()
    print("  경로 A: 2/(GAP×K_EQI) ≈ int_x=24")
    print(f"          2/(GAP×K_EQI) = {float(N_A):.6f}  vs  int_x={int(int_x)}")
    print(f"          오차% = {float(err_A):.4f}%")
    print()
    print("  경로 B: sinc² 잔류 1차 근사 vs 측정값")
    print("          → 1차 미분 근사로 설명 가능 여부 [위 [6] 결과 참조]")
    print()
    print("  경로 C: n_ref,k × R_pi vs GAP_D2^(2k+1)")
    print("          → 직접 연결 여부 [위 [7] 결과 참조]")
    print()
    print("  [9] R_pi(N=24) 구조 분해 → R_membrane 연결 여부")
    print("      핵심 질문: R_pi(24) × [?] = R_membrane 달성 여부")
    print(sep)


if __name__ == "__main__":
    main()

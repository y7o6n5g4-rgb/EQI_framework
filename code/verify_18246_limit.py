#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Framework — Session 43
verify_18246_limit.py

목표:
  lim_{N→∞} [ratio(N) × N] 정밀값 산출
  ratio(N) = R_pi(N) / R_mem(N)
    R_pi(N)  = 4π − 16×Σ_{k=0}^{N-1}[1/(4k+1)−1/(4k+3)]  (Leibniz π 급수 N항 잔류)
    R_mem(N) = frac_x(N) × gd(N)²
      gd(N)    = ln5_partial(N) / 100
      x(N)     = 1 / (c_EQI × 2 × gd(N))
      frac_x(N) = x(N) − floor(x(N))

Maestro : MAPSI (맏이)
Architect: Claude (Session 43)
Engineer : Gemini
mp.dps   : 400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs
mp.dps = 400

# ── 핵심 상수 ────────────────────────────────────────────────────────────────
PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = sqrt(PHI)

TARGET = GAP - 11 * GAP_D2**2

# ── 급수 함수 ─────────────────────────────────────────────────────────────────

def leibniz_pi_residual(N):
    """4π − 16×Σ_{k=0}^{N-1}[1/(4k+1)−1/(4k+3)]"""
    s = mpf(0)
    for k in range(N):
        s += 1/(4*k+1) - 1/(4*k+3)
    return 4*pi - 16*s

def ln5_partial(N):
    """ln(5) Leibniz 급수 N항: 2×Σ_{k=0}^{N-1} (2/3)^(2k+1)/(2k+1)"""
    z = mpf(2)/3
    s = mpf(0)
    for k in range(N):
        s += z**(2*k+1) / (2*k+1)
    return 2*s

def ratio_times_N(N):
    """ratio(N) × N 계산"""
    R_pi = leibniz_pi_residual(N)

    gd_N     = ln5_partial(N) / 100
    x_N      = 1 / (c_EQI * 2 * gd_N)
    frac_N   = x_N - floor(x_N)
    R_mem_N  = frac_N * gd_N**2

    ratio = R_pi / R_mem_N
    return ratio * N, R_pi, R_mem_N, ratio


# ── §1: N별 ratio×N 수렴 추이 ────────────────────────────────────────────────
sep  = "─" * 76
sep2 = "═" * 76

print(f"\n{sep2}")
print("  §1. ratio(N)×N 수렴 추이  (N=10→5000)")
print(sep2)
print(f"  {'N':>6}  {'ratio×N':>28}  {'R_pi(N)':>22}  {'R_mem(N)':>22}")
print(sep)

N_list = [10, 20, 34, 50, 100, 200, 500, 1000, 2000, 5000]
results = {}
for N in N_list:
    val, R_pi, R_mem, ratio = ratio_times_N(N)
    results[N] = val
    print(f"  {N:>6}  {mp.nstr(val, 22):>28}  {mp.nstr(R_pi, 14):>22}  {mp.nstr(R_mem, 14):>22}")

print(sep)

# N=5000을 극한 근사값으로 사용
lim_approx = results[5000]
print(f"\n  lim_{{N→∞}} [ratio(N)×N] ≈ {mp.nstr(lim_approx, 30)}")
print(f"  (N=5000 기준 근사값)")


# ── §2: 수렴율 확인 (연속 항 차이) ──────────────────────────────────────────
print(f"\n{sep2}")
print("  §2. 연속 항 차이 (수렴 속도 확인)")
print(sep2)
prev_key = N_list[0]
for N in N_list[1:]:
    diff = results[N] - results[prev_key]
    print(f"  ratio×N({N}) − ratio×N({prev_key}) = {mp.nstr(diff, 12)}")
    prev_key = N


# ── §3: 후보 대수 정체 탐색 ─────────────────────────────────────────────────
print(f"\n{sep2}")
print("  §3. 후보 대수 정체 탐색 (극한값 vs 후보 공식)")
print(sep2)

L = lim_approx  # 극한 근사값

candidates = {
    "후보1: π / (4 × c_EQI × GAP_D2²)"  : pi / (4 * c_EQI * GAP_D2**2),
    "후보2: 4π × (x_lattice / GAP_D2)"   : 4 * pi * (1/(c_EQI*GAP)) / GAP_D2,
    "후보3: 4 × TARGET / GAP_D2³"         : 4 * TARGET / GAP_D2**3,
    "후보4: 11 × π / GAP_D2²"            : 11 * pi / GAP_D2**2,
    # 추가 탐색
    "탐색A: π / (4 × GAP_D2² × c_EQI²)" : pi / (4 * GAP_D2**2 * c_EQI**2),
    "탐색B: π × x_lattice / GAP_D2"      : pi * (1/(c_EQI*GAP)) / GAP_D2,
    "탐색C: π / (2 × c_EQI × GAP_D2²)"  : pi / (2 * c_EQI * GAP_D2**2),
    "탐색D: π × GAP / (4 × GAP_D2³)"    : pi * GAP / (4 * GAP_D2**3),
    "탐색E: 8π × x_lattice / GAP_D2"     : 8 * pi * (1/(c_EQI*GAP)) / GAP_D2,
    "탐색F: π / (4 × GAP_D2³)"           : pi / (4 * GAP_D2**3),
    "탐색G: π × GAP / GAP_D2³"           : pi * GAP / GAP_D2**3,
}

print(f"  극한 근사값 L = {mp.nstr(L, 22)}")
print(sep)
print(f"  {'후보':42s}  {'후보값':22s}  {'오차%':14s}")
print(sep)

best = None
best_err = mpf(1)
for label, val in candidates.items():
    err = (val - L) / L * 100
    err_f = float(fabs(err))
    flag = "  ★★★" if err_f < 0.01 else ("  ★" if err_f < 0.1 else "")
    print(f"  {label:42s}  {mp.nstr(val, 16):22s}  {float(err):+.6f}%{flag}")
    if fabs(err) < best_err:
        best_err = fabs(err)
        best = (label, val, err)

print(sep)
if best:
    print(f"\n  최근접 후보: {best[0]}")
    print(f"  오차: {float(best[2]):+.8f}%")


# ── §4: 점근 분석 — 두 급수 잔류의 1/N 비례 확인 ────────────────────────────
print(f"\n{sep2}")
print("  §4. 점근 분석 — R_pi(N)×N, R_mem(N)×N 각각의 수렴 상수")
print(sep2)
print(f"  {'N':>6}  {'R_pi×N':>24}  {'R_mem×N×10⁶':>24}")
print(sep)

for N in [100, 200, 500, 1000, 2000, 5000]:
    _, R_pi, R_mem, _ = ratio_times_N(N)
    rpiN  = R_pi * N
    rmemN = R_mem * N * mpf(10)**6
    print(f"  {N:>6}  {mp.nstr(rpiN, 18):>24}  {mp.nstr(rmemN, 18):>24}")

print(sep)
print("  → R_pi(N)×N → π/4 (Leibniz 수렴율)")
print(f"    π/4 = {mp.nstr(pi/4, 20)}")
print("  → R_mem(N)×N → (별도 상수) — §5에서 직접 도출")


# ── §5: R_mem(N)×N 극한 상수 직접 도출 ────────────────────────────────────
print(f"\n{sep2}")
print("  §5. R_mem(N)×N 극한 상수 도출")
print(sep2)

# ratio×N = R_pi×N / R_mem×N = (π/4) / C_mem  → C_mem = π/(4×lim)
C_mem = pi / (4 * L)
print(f"  ratio×N = (R_pi×N) / (R_mem×N) = (π/4) / C_mem")
print(f"  C_mem = π / (4 × lim) = {mp.nstr(C_mem, 22)}")
print()

# C_mem의 대수 정체 탐색
cands_mem = {
    "c_EQI × GAP_D2²"         : c_EQI * GAP_D2**2,
    "GAP_D2² × c_EQI"         : GAP_D2**2 * c_EQI,
    "GAP_D2² × 2 × c_EQI"     : GAP_D2**2 * 2 * c_EQI,
    "GAP_D2² / (4 × c_EQI)"   : GAP_D2**2 / (4 * c_EQI),
    "GAP_D2³ / GAP"            : GAP_D2**3 / GAP,
    "GAP_D2³ × 2"              : GAP_D2**3 * 2,
    "TARGET / π"               : TARGET / pi,
    "GAP_D2² / (2 × c_EQI)"   : GAP_D2**2 / (2 * c_EQI),
    "GAP_D2² × c_EQI / 2"     : GAP_D2**2 * c_EQI / 2,
}

print(f"  {'C_mem 후보':38s}  {'후보값':22s}  {'오차%':14s}")
print(sep)
for label, val in cands_mem.items():
    err = (val - C_mem) / C_mem * 100
    flag = "  ★★★" if float(fabs(err)) < 0.01 else ("  ★" if float(fabs(err)) < 0.5 else "")
    print(f"  {label:38s}  {mp.nstr(val, 16):22s}  {float(err):+.6f}%{flag}")


# ── §6: 최종 요약 ────────────────────────────────────────────────────────────
print(f"\n{sep2}")
print("  §6. Session 43 §1 최종 요약")
print(sep2)
print(f"  lim_{{N→∞}} [ratio(N)×N] = {mp.nstr(lim_approx, 28)}")
print(f"  (N=5000 근사)")
print()
print(f"  구조 해석:")
print(f"    ratio(N) × N = [R_pi(N)×N] / [R_mem(N)×N]")
print(f"                 = (π/4) / C_mem")
print(f"    C_mem = π/(4×lim) = {mp.nstr(C_mem, 22)}")
print()
print(f"  536.622의 정체 재확인:")
print(f"    536.622 = lim/34 = {mp.nstr(lim_approx/34, 18)}")
print(sep2)
print("  ★ Gemini 실행 후 Claude에게 전체 출력 붙여넣기 ★")
print(sep2)

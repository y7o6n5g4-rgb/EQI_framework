#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Framework — Session 43
verify_18246_fracx_analysis.py

문제 재진단:
  R_mem(N)×N이 N에 정비례 → R_mem(N) 자체가 N 비례 증가
  = frac_x(N)이 N 비례 증가
  → frac_x(N)은 상수가 아님

목표:
  1. frac_x(N) 거동 직접 확인
  2. R_mem(N)의 실제 점근 구조 파악
  3. ratio(N)×N 극한 18246...의 해석적 닫힘 재탐색

Maestro : MAPSI (맏이)
Architect: Claude (Session 43)
Engineer : Gemini
mp.dps   : 400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs
mp.dps = 400

PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = sqrt(PHI)

# v7.9 R_membrane (floor 기반, N→∞ 극한)
x_lattice  = 1 / (c_EQI * GAP)
frac_x_inf = x_lattice - floor(x_lattice)
R_mem_inf  = frac_x_inf * GAP_D2**2

TARGET = GAP - 11 * GAP_D2**2

sep  = "─" * 76
sep2 = "═" * 76

def ln5_partial(N):
    z = mpf(2)/3
    s = mpf(0)
    for k in range(N):
        s += z**(2*k+1) / (2*k+1)
    return 2*s

def get_components(N):
    gd_N    = ln5_partial(N) / 100
    x_N     = 1 / (c_EQI * 2 * gd_N)
    fx_N    = x_N - floor(x_N)
    Rmem_N  = fx_N * gd_N**2
    return gd_N, x_N, fx_N, Rmem_N

def leibniz_pi_residual(N):
    s = mpf(0)
    for k in range(N):
        s += 1/(4*k+1) - 1/(4*k+3)
    return 4*pi - 16*s


# ── §1: frac_x(N) 거동 직접 확인 ────────────────────────────────────────────
print(f"\n{sep2}")
print("  §1. frac_x(N), gd(N), x(N) 거동 직접 확인")
print(sep2)
print(f"  {'N':>6}  {'gd(N)':>22}  {'x(N)':>22}  {'frac_x(N)':>22}  {'frac_x×N':>22}")
print(sep)

for N in [10, 20, 34, 50, 100, 200, 500, 1000, 2000, 5000]:
    gd, x, fx, Rmem = get_components(N)
    print(f"  {N:>6}  {mp.nstr(gd,14):>22}  {mp.nstr(x,14):>22}  {mp.nstr(fx,14):>22}  {mp.nstr(fx*N,14):>22}")

print(sep)
print(f"  GAP_D2 (∞) = {mp.nstr(GAP_D2, 22)}")
print(f"  x_lattice  = {mp.nstr(x_lattice, 22)}")
print(f"  frac_x_inf = {mp.nstr(frac_x_inf, 22)}")


# ── §2: R_mem(N) 점근 구조 ───────────────────────────────────────────────────
print(f"\n{sep2}")
print("  §2. R_mem(N) 점근 구조 분석")
print(sep2)
print(f"  {'N':>6}  {'R_mem(N)':>26}  {'R_mem×N':>26}  {'R_mem×N²':>26}")
print(sep)

for N in [100, 200, 500, 1000, 2000, 5000]:
    gd, x, fx, Rmem = get_components(N)
    print(f"  {N:>6}  {mp.nstr(Rmem,18):>26}  {mp.nstr(Rmem*N,18):>26}  {mp.nstr(Rmem*N**2,18):>26}")

print(sep)
print("  → R_mem×N이 N 정비례면: R_mem ~ C/1 (상수)")
print("  → R_mem×N²이 N 정비례면: R_mem ~ C/N")
print("  → 실제 거동 확인 필요")


# ── §3: frac_x(N)×N 거동 — 핵심 ────────────────────────────────────────────
print(f"\n{sep2}")
print("  §3. frac_x(N)×N 거동 — 핵심 점근 분석")
print(sep2)
print(f"  {'N':>6}  {'frac_x(N)':>24}  {'frac_x×N':>24}  {'(frac_x-frac_x_inf)×N':>28}")
print(sep)

for N in [100, 200, 500, 1000, 2000, 5000, 10000]:
    gd, x, fx, Rmem = get_components(N)
    diff_fx = fx - frac_x_inf
    print(f"  {N:>6}  {mp.nstr(fx,16):>24}  {mp.nstr(fx*N,16):>24}  {mp.nstr(diff_fx*N,16):>28}")

print(sep)
print(f"  frac_x_inf = {mp.nstr(frac_x_inf, 22)}")
print(f"  → (frac_x(N) - frac_x_inf)×N 수렴 여부 확인")


# ── §4: ratio(N)×N 점근 전개 ─────────────────────────────────────────────────
print(f"\n{sep2}")
print("  §4. ratio(N)×N 점근 전개 구조")
print(sep2)
print(f"""
  ratio(N) = R_pi(N) / R_mem(N)

  R_pi(N)   = 4π − 16×S_pi(N)  ~ π/N  (Leibniz, 정확히는 π/(4N)+고차)
  R_mem(N)  = frac_x(N) × gd(N)²

  gd(N) → GAP_D2  (N≥50 실질 수렴)
  frac_x(N) → frac_x_inf  (N→∞)

  ratio(N)×N ~ [R_pi(N)×N] / [frac_x(N) × GAP_D2²]
             = [π/4 + O(1/N)] / [frac_x(N) × GAP_D2²]

  만약 frac_x(N) = frac_x_inf + A/N + O(1/N²) 이면:
    ratio(N)×N ~ (π/4) / [(frac_x_inf + A/N) × GAP_D2²]
    lim = (π/4) / (frac_x_inf × GAP_D2²)
        = (π/4) / R_mem_inf
        = {mp.nstr(pi/4/R_mem_inf, 20)}

  이것이 실측 18246.14...와 맞는지 확인:
""")

lim_formula = pi / (4 * R_mem_inf)
lim_N5000 = mpf("18246.14145400593471653612267")
err = (lim_formula - lim_N5000) / lim_N5000 * 100
print(f"  π/(4×R_mem_inf) = {mp.nstr(lim_formula, 24)}")
print(f"  lim (N=5000)    = {mp.nstr(lim_N5000, 24)}")
print(f"  오차            = {mp.nstr(err, 10)}%")
print()
print(f"  ※ R_mem_inf = frac_x_inf × GAP_D2² (floor 기반 v7.9 R_membrane)")
print(f"             = {mp.nstr(R_mem_inf, 22)}")


# ── §5: frac_x(N) 수렴 보정항 A 추출 ────────────────────────────────────────
print(f"\n{sep2}")
print("  §5. frac_x(N) 수렴 보정항 A 추출")
print(sep2)
print(f"  frac_x(N) ≈ frac_x_inf + A/N")
print(f"  A = (frac_x(N) - frac_x_inf) × N")
print(sep)
print(f"  {'N':>6}  {'A = (fx(N)-fx_inf)×N':>30}")
print(sep)

A_vals = {}
for N in [500, 1000, 2000, 5000, 10000]:
    gd, x, fx, Rmem = get_components(N)
    A = (fx - frac_x_inf) * N
    A_vals[N] = A
    print(f"  {N:>6}  {mp.nstr(A, 22):>30}")

print(sep)
# A 수렴값
A_inf = A_vals[10000]
print(f"  A (N=10000 근사) = {mp.nstr(A_inf, 22)}")
print()
# A의 대수 정체 탐색
print(f"  A 후보 탐색:")
A_cands = {
    "1/(4×c_EQI×GAP_D2×2)"  : 1/(4*c_EQI*GAP_D2*2),
    "1/(8×c_EQI×GAP_D2)"    : 1/(8*c_EQI*GAP_D2),
    "GAP/(4×c_EQI×GAP_D2²)" : GAP/(4*c_EQI*GAP_D2**2),
    "1/(4×c_EQI²×GAP_D2)"   : 1/(4*c_EQI**2*GAP_D2),
    "frac_x_inf / (4×GAP_D2)": frac_x_inf/(4*GAP_D2),
    "1/(4×GAP)"              : 1/(4*GAP),
    "x_lattice/4"            : x_lattice/4,
}
for label, val in A_cands.items():
    err_a = (val - A_inf)/A_inf*100
    flag = "  ★★★" if float(fabs(err_a)) < 0.1 else ""
    print(f"    {label:35s} = {mp.nstr(val,16):24s}  오차 {float(err_a):+.4f}%{flag}")


# ── §6: 최종 요약 ────────────────────────────────────────────────────────────
print(f"\n{sep2}")
print("  §6. 요약 및 다음 방향")
print(sep2)
print(f"  lim_{{N→∞}} [ratio(N)×N] = {mp.nstr(lim_N5000, 22)}")
print(f"  π/(4×R_mem_inf)         = {mp.nstr(lim_formula, 22)}")
print(f"  오차                    = {mp.nstr(err, 8)}%")
print()
print(f"  → π/(4×R_mem_inf)과 실측 극한의 관계 파악 후")
print(f"    보정항 구조 확인 예정")
print(sep2)
print("  ★ Gemini 실행 후 Claude에게 전체 출력 붙여넣기 ★")
print(sep2)

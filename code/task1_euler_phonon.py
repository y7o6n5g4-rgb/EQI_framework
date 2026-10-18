#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 — 오일러 수 포논 계수 검증 (TASKS.md v9.7, 엔진 수정 없음)
mp.dps = 400  |  상수 정의 = CLAUDE.md / v9.6 엔진 동일
  1a  엔진 §12~§13 격자 (Leibniz N=34, 68항)
  1b  §90a 격자 (137항)
  1c  S88 §92 ε 정확식
  1d  체 계층 갭 급수 (S48 §17.7)
Date: 2026-10-18
"""

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, fabs, nstr, exp, power

mp.dps = 400

# ════════════════════════════════════════════════════════════════
# 핵심 상수 (CLAUDE.md)
# ════════════════════════════════════════════════════════════════
PHI    = (sqrt(5) + 1) / 2
GAP    = log(25) / 100
GD     = log(5) / 100                       # GAP_D2
LN_C   = log(PHI) / 2
LC2GD  = LN_C**2 * GD                       # (B) 계단 기본량 B = LN_C²·GD
sigma  = mpf(1) / 2
alpha_inv = 360*PHI**-2 - 2*PHI**-3 + (3*PHI)**-5

# 오일러 수 (부호 없는 값) E_0..E_8
E = [mpf(1), mpf(1), mpf(5), mpf(61), mpf(1385)]   # E_{2k}, k=0..4
assert all(fabs(mpmath.eulernum(2*k)) == E[k] for k in range(5))


def err_pct(calc, target):
    """err% = 100*|calc - target|/|target|  (CLAUDE.md 원칙 5)"""
    return 100 * fabs(calc - target) / fabs(target)


def grade(e, algebraic=False):
    """CLAUDE.md 원칙 6 (e = err%)"""
    if algebraic:
        return "대수필연" if e < mpf(10)**-390 else "대수필연 미달"
    if e < mpf(10)**-6:
        return "TZ"
    if e < mpf("0.01"):
        return "Near TZ"
    return "-"


def row(tag, label, calc, target, e, g, src):
    print(f"[{tag}] {label} | {nstr(calc, 20)} | {nstr(target, 20)} | "
          f"{nstr(e, 6)} | {g} | {src}")


def L(M):
    """L(M) = 4·Σ_{k=0}^{M-1} (-1)^k/(2k+1)"""
    s = mpf(0)
    for k in range(M):
        s += mpf((-1)**k) / (2*k + 1)
    return 4 * s


def R_pi_exact(N):
    """엔진 §12 정의: R_pi(N) = 4π − 16Σ_{k<N}[1/(4k+1) − 1/(4k+3)]"""
    s = mpf(0)
    for k in range(N):
        s += mpf(1)/(4*k + 1) - mpf(1)/(4*k + 3)
    return 4*pi - 16*s


# ════════════════════════════════════════════════════════════════
# 1a. 엔진 §12~§13 격자 (N=34, 68항)
# ════════════════════════════════════════════════════════════════
print("=" * 78)
print("1a. R_pi(N=34) 오일러 전개  R_pi = Σ (-1)^k·2E_{2k}·σ^(4k)/N^(2k+1)")
print("=" * 78)

N = 34
R34 = R_pi_exact(N)
R34_L = 4 * (pi - L(2*N))
d_RL = fabs(R34 - R34_L)
row("1a-0", "R_pi(34) = 4(π − L(68))", R34_L, R34, d_RL,
    grade(d_RL, algebraic=True), "S43 §12")

# 계수 대수 확인: k=1 2σ⁴ = σ³,  k=2 10σ⁸ = 5σ⁵/4
c1 = 2*E[1]*sigma**4
c2 = 2*E[2]*sigma**8
row("1a-c1", "2E_2·σ⁴ = σ³", c1, sigma**3, fabs(c1 - sigma**3),
    grade(fabs(c1 - sigma**3), True), "S44 §13")
row("1a-c2", "2E_4·σ⁸ = 5σ⁵/4", c2, 5*sigma**5/4, fabs(c2 - 5*sigma**5/4),
    grade(fabs(c2 - 5*sigma**5/4), True), "S45 a₅_eff")

print()
print("  항 수 | k번째 항 (부호 포함)            | 누적 합                    | 절대 잔여 R−Σ             | 상대 잔여")
terms_a = []
acc = mpf(0)
prev_abs = None
for k in range(5):
    t = mpf((-1)**k) * 2 * E[k] * sigma**(4*k) / mpf(N)**(2*k + 1)
    if prev_abs is not None and fabs(t) > prev_abs:
        print(f"  k={k}: 항 크기 증가 → 점근급수 발산 시작, 여기서 중단")
        break
    prev_abs = fabs(t)
    acc += t
    terms_a.append(t)
    res = R34 - acc
    print(f"  {k+1:>4}  | {nstr(t, 22):>30} | {nstr(acc, 22):>26} | "
          f"{nstr(res, 12):>24} | {nstr(fabs(res)/R34, 6)}")

# 점근 발산 시작점 (기록용; 실제 사용 항수는 5)
kk = 0
while True:
    a = 2*fabs(mpmath.eulernum(2*kk))*sigma**(4*kk)/mpf(N)**(2*kk+1)
    b = 2*fabs(mpmath.eulernum(2*kk+2))*sigma**(4*kk+4)/mpf(N)**(2*kk+3)
    if b > a:
        break
    kk += 1
print(f"  (기록) N=34에서 항 크기 최소 지점 k={kk} — 5항 범위 내 발산 없음")

# 엔진 §13 _R_pi_asymp 3항 (5차 계수 3σ⁵) vs 오일러 5σ⁵/4
print()
eng3 = mpf(2)/N - sigma**3/N**3 + 3*sigma**5/N**5        # 엔진 §13 (기록, 수정 없음)
eul3 = mpf(2)/N - sigma**3/N**3 + 5*sigma**5/4/N**5      # 오일러 3항
row("1a-E3", "엔진 §13 3항 (5차 3σ⁵)", eng3, R34, err_pct(eng3, R34), grade(err_pct(eng3, R34)), "S43/S44 엔진")
row("1a-U3", "오일러 3항 (5차 5σ⁵/4)", eul3, R34, err_pct(eul3, R34), grade(err_pct(eul3, R34)), "S45 a₅_eff")
print(f"  5차 계수: 엔진 3σ⁵ = {nstr(3*sigma**5, 10)}  vs  오일러 5σ⁵/4 = {nstr(5*sigma**5/4, 10)}"
      f"  비 = {nstr(3*sigma**5/(5*sigma**5/4), 10)} (= 12/5)")
print(f"  잔여 실측 5차 계수: (R − 2/N + σ³/N³)·N⁵ = "
      f"{nstr((R34 - 2/mpf(N) + sigma**3/mpf(N)**3)*mpf(N)**5, 16)}  "
      f"(5σ⁵/4 = {nstr(5*sigma**5/4, 6)})")
print("  ※ 엔진 §13 수정 없음 — 기록만.")

# ════════════════════════════════════════════════════════════════
# 1b. §90a 격자 (137항)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("1b. §90a  L(M) − π = Σ (-1)^k E_{2k}/(4^k·M^(2k+1))   (M=137 홀수)")
print("=" * 78)

M = 137
s1 = L(M) - pi
s2 = s1 - 1/alpha_inv
s1_exp = mpf("7.2991728543872000722e-3")
s2_exp = mpf("1.8202894042787662893e-6")
row("1b-s1", "s1 = L(137) − π", s1, s1_exp, err_pct(s1, s1_exp), "기대값(20자리) 대조", "S88 §90a")
row("1b-s2", "s2 = s1 − 1/α⁻¹", s2, s2_exp, err_pct(s2, s2_exp), "기대값(20자리) 대조", "S88 §90a")

frac_a = alpha_inv - 137                  # {α⁻¹} — 137 명시 (int 금지)
lead = frac_a / (137 * alpha_inv)
d_lead = fabs(lead - (mpf(1)/137 - 1/alpha_inv))
row("1b-0", "{α⁻¹}/(137α⁻¹) = 1/137 − 1/α⁻¹", lead, mpf(1)/137 - 1/alpha_inv,
    d_lead, grade(d_lead, True), "항등")

print()
print("  최고 차수 | 추가 항                      | 누적 s2 근사               | 상대 잔여 |s2−Σ|/|s2|")
acc = lead
print(f"  {'1':>8}  | {nstr(lead, 20):>28} | {nstr(acc, 22):>26} | {nstr(fabs(s2-acc)/fabs(s2), 6)}")
s2_rel = {}
for k in range(1, 5):
    t = mpf((-1)**k) * E[k] / (mpf(4)**k * mpf(M)**(2*k + 1))
    acc += t
    rel = fabs(s2 - acc) / fabs(s2)
    s2_rel[2*k + 1] = rel
    print(f"  {2*k+1:>8}  | {nstr(t, 20):>28} | {nstr(acc, 22):>26} | {nstr(rel, 6)}")
print(f"  기대: 5차까지 ≈ 5.8e-10 → 실측 {nstr(s2_rel[5], 6)} ; 7차 포함 {nstr(s2_rel[7], 6)}")

# ════════════════════════════════════════════════════════════════
# 1c. S88 §92 ε 정확식
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("1c. ε = |s1|·α⁻¹ − 1  :  ε_true / ε_S88 / ε_Euler")
print("=" * 78)

eps_true = fabs(s1) * alpha_inv - 1
St = power(PHI, mpf(-11)/4)
eps_S88 = LC2GD * St                        # (B) 계단 계열 LC2GD = LN_C²·GD, × St
print(f"  ε_true  = {nstr(eps_true, 30)}")
print(f"  ε_S88   = LC2GD·φ^(-11/4) = {nstr(eps_S88, 30)}")
eul_sum = mpf(0)
eps_E = None
for k in range(5):
    eul_sum += mpf((-1)**k) * E[k] / (mpf(4)**k * mpf(M)**(2*k + 1))
    eps_E = alpha_inv * eul_sum - 1
    print(f"  ε_Euler (최고 {2*k+1}차) = {nstr(eps_E, 30)}   "
          f"err% vs ε_true = {nstr(err_pct(eps_E, eps_true), 6)}")
print()
row("1c-1", "ε_S88 vs ε_true", eps_S88, eps_true, err_pct(eps_S88, eps_true),
    grade(err_pct(eps_S88, eps_true)), "S88 §92")
row("1c-2", "ε_Euler(9차) vs ε_true", eps_E, eps_true, err_pct(eps_E, eps_true),
    grade(err_pct(eps_E, eps_true)), "Task 1b 전개")
row("1c-3", "ε_S88 vs ε_Euler(9차)", eps_S88, eps_E, err_pct(eps_S88, eps_E),
    grade(err_pct(eps_S88, eps_E)), "상호 대조")

# ════════════════════════════════════════════════════════════════
# 1d. 체 계층 갭 급수 (S48 §17.7)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("1d. S48 §17.7  갭×K_EQI = GAP + σ³·GAP² + σ⁵·GAP³ + ...  (k=0~5)")
print("=" * 78)
print("  정의 출처: S48 §17.7 / S47 §4.2 (급수), 엔진 §16 (갭), 엔진 §13 (K_EQI)")

T_Is_D1 = power(PHI, mpf(3)/2) - power(PHI, mpf(-7)/2)
K_EQI = (T_Is_D1 - mpmath.euler - 2*GD**2*exp(GD)) / log(PHI)

# 갭 = (frac_x − x_prec)/frac_x  — v9.6 엔진 §16 (S47) 정의를 엔진 객체에서 그대로 사용
# (엔진 §8의 floor는 엔진 내부 기존 코드. 본 스크립트는 값만 참조, 수정 없음)
import os, sys, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import eqi_unified_engine_v9_6 as eng
    C = eng.EQIConstants()
assert fabs(C.K_EQI - K_EQI) < mpf(10)**-390
gap12 = C.gap_12pct
print("  갭 = (frac_x − x_prec)/frac_x  (엔진 §16, S47)")
print(f"  갭          = {nstr(gap12, 30)}")
print(f"  K_EQI       = {nstr(K_EQI, 30)}")
target_d = gap12 * K_EQI
print(f"  target 갭·K_EQI = {nstr(target_d, 30)}")
row("1d-0", "(참고) GAP·K_EQI ≠ 대상", GAP*K_EQI, target_d, err_pct(GAP*K_EQI, target_d), "-", "정의 혼동 방지")

# (i) 순수 σ^(2k+1) 가중: k=0 항 계수 1 (S48 원문 "GAP + σ³GAP² + ..."), k≥1 σ^(2k+1)
pure = GAP
for k in range(1, 6):
    pure += sigma**(2*k + 1) * GAP**(k + 1)
# (ii) 오일러 가중: k≥1 계수 2E_{2k}σ^(4k), k=0 항은 S48과 동일하게 GAP (계수 1)
#      (2E_0σ⁰ = 2 를 그대로 쓰면 기저항이 2·GAP 가 되어 비교 무의미 → 기저 고정)
eul = GAP
for k in range(1, 6):
    ck = 2*fabs(mpmath.eulernum(2*k))*sigma**(4*k)
    eul += ck * GAP**(k + 1)
row("1d-1", "순수 σ^(2k+1) 가중 (k=0~5)", pure, target_d, err_pct(pure, target_d),
    grade(err_pct(pure, target_d)), "S48 §17.7 (원문 0.000731%)")
row("1d-2", "오일러 2E_{2k}σ^(4k) 가중 (k=0~5)", eul, target_d, err_pct(eul, target_d),
    grade(err_pct(eul, target_d)), "Task 1d 비교")
print(f"  두 급수 차이 (오일러 − 순수) = {nstr(eul - pure, 12)}  (k=2 항 차 = {nstr((5*sigma**8*2 - sigma**5)*GAP**3, 12)})")
print("  ※ k=1 계수는 두 경우 동일(σ³). 차이는 k≥2부터: 순수 σ⁵=1/32 vs 오일러 5/128 ...")

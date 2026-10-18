#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task M — MAXWELL 계보 전개 (TASKS.md v9.7, 독립 스크립트, 엔진 수정 없음)
mp.dps = 400  |  상수 정의 = CLAUDE.md / v9.6 엔진 동일
위계: MAXWELL 은 EQI 식이 정의 (S39 v7.6: MAXWELL_true SI 제거, T_Is_D1 경로). SI 비교 없음.
      순수 위상 MAXWELL_EQI = 2.0, 1.873… = 투영값 (S34 §14.1, S35).
  M1  세 언어 항별 출력 (값만)
  M2  기저항 항등식 (대수필연)
  M3  순수 위상 MAXWELL_EQI = 2 와 투영
  M4  세 평균 절대 부등식 (대수필연)
  M5  5차 계수와 137 격자 (274 = 2×137)
  M6  (MAXWELL − γ)/ln φ  (값만, 닫힘 시도 없음)
기호 주의: 전자기 언어의 S_D1·S_D2 (R0 정의: lnφ·M/(1+M), lnφ/(1+M)) 와
           격자 기저의 S_D1 = L_p² = φ^(-3/2), H_D2 = T_p² = φ^(-5/2) (§P1) 는 서로 다른 양.
           본 스크립트에서는 S_D1^em / S_D2^em, S_D1^lat / H_D2^lat 로 구분 표기.
Date: 2026-10-18
"""

import io
import os
import sys
import contextlib

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, fabs, nstr, power, exp

mp.dps = 400
TOL = mpf(10)**-390

# ════════════════════════════════════════════════════════════════
# 핵심 상수 (CLAUDE.md)
# ════════════════════════════════════════════════════════════════
PHI   = (sqrt(5) + 1) / 2
GAP   = log(25) / 100
GD    = log(5) / 100
LN_C  = log(PHI) / 2
sigma = mpf(1) / 2
mu    = 1 / PHI
Lam   = PHI**-2
c     = sqrt(PHI)
LN2   = log(2)
q     = lambda p, d: power(PHI, mpf(p)/d)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import eqi_unified_engine_v9_6 as eng
    C = eng.EQIConstants()


def alg(tag, label, calc, target, src):
    """[§tag] label | calc | target | 400자리 절대 차 | 등급 | 출처"""
    e = fabs(calc - target)
    g = "대수필연" if e < TOL else "대수필연 미달"
    print(f"[{tag}] {label} | {nstr(calc, 20)} | {nstr(target, 20)} | {nstr(e, 6)} | {g} | {src}")


def L(M_):
    s = mpf(0)
    for k in range(M_):
        s += mpf((-1)**k) / (2*k + 1)
    return 4 * s


# ════════════════════════════════════════════════════════════════
# M1. 세 언어 항별 출력 (값만)
# ════════════════════════════════════════════════════════════════
print("=" * 78)
print("M1. MAXWELL 세 언어 항별 출력 (값만)")
print("=" * 78)
t1 = q(3, 2) - q(-7, 2)
t2 = 2 * GD**2 * exp(GD)
t3 = (LN2 / 2) * GD**3
t4 = (LN2/19 + GD/26) * GD**4
t5 = LN_C**2 * GD**5 / 274
M5 = t1 + t2 + t3 + t4 + t5
print("  [기하 기원 — 엔진 §2·§5]")
for lab, v in (("t1 = φ^1.5 − φ^(-3.5)", t1), ("t2 = 2·GD²·e^GD", t2), ("t3 = (ln2/2)·GD³", t3),
               ("t4 = (ln2/19 + GD/26)·GD⁴", t4), ("t5 = LN_C²·GD⁵/274", t5)):
    print(f"     {lab:<28} = {nstr(v, 30)}")
print(f"     합 (엔진 5차)                = {nstr(M5, 30)}")
alg("M1-0", "Σt (스크립트) = 엔진 MAXWELL_5th", M5, C.MAXWELL_5th, "엔진 §5")

e1 = log(pi) * (1 + GAP)
e2 = LN2
e3 = -GAP**2 * log(pi) / LN2
e4 = 24 * GD**3                              # 4!·GD³
ME = e1 + e2 + e3 + e4
print("  [엔트로피장 — S34 §11.8]")
for lab, v in (("e1 = ln(π)·(1+GAP)", e1), ("e2 = ln2", e2), ("e3 = −GAP²·ln(π)/ln2", e3),
               ("e4 = 4!·GD³", e4)):
    print(f"     {lab:<28} = {nstr(v, 30)}")
print(f"     합 (엔트로피장 4차)          = {nstr(ME, 30)}")

S1em = log(PHI) * M5 / (1 + M5)
S2em = log(PHI) / (1 + M5)
print("  [전자기 언어 — EQI 정의 (R0)]")
print(f"     S_D1^em = lnφ·M/(1+M) = {nstr(S1em, 30)}")
print(f"     S_D2^em = lnφ/(1+M)   = {nstr(S2em, 30)}")
print(f"     S_D1^em + S_D2^em = {nstr(S1em + S2em, 30)} (= lnφ),  S_D1^em/S_D2^em = {nstr(S1em/S2em, 30)} (= M)")

d = ME - M5
print("  [두 식 대조]")
print(f"     엔트로피장 4차 − 엔진 5차 = {nstr(d, 30)}")
print(f"     상대 차 (엔진 5차 기준)     = {nstr(100*fabs(d)/M5, 10)} %")
print(f"     잔여 GD 차수: ln|차|/ln GD = {nstr(log(fabs(d))/log(GD), 12)}")
for k in range(1, 6):
    print(f"       차/GD^{k} = {nstr(d/GD**k, 15)}")

# ════════════════════════════════════════════════════════════════
# M2. 기저항 항등식 (대수필연)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("M2. 기저항 항등식")
print("=" * 78)
m_i, m_g = q(3, 2), q(-3, 2)
proton = q(-7, 2)
S1lat, H2lat = q(-3, 4)**2, q(-5, 4)**2       # 격자 기저 S_D1 = L_p², H_D2 = T_p²
alg("M2-1a", "t1 = m_i − proton", t1, m_i - proton, "엔진 §2 / S57")
alg("M2-1b", "t1 = μ·(φ^(5/2) − φ^(-5/2))", t1, mu*(q(5, 2) - q(-5, 2)), "대수")
alg("M2-1c", "t1 = m_i − (S_D1^lat − H_D2^lat)", t1, m_i - (S1lat - H2lat), "S88 §83")
alg("M2-1d", "t1 = 엔진 T_Is_D1", t1, C.T_Is_D1, "엔진 §2")
alg("M2-2", "exp(GD) = 5^(1/100)", exp(GD), power(5, mpf(1)/100), "S34 §15.2")

# ════════════════════════════════════════════════════════════════
# M3. 순수 위상 MAXWELL_EQI = 2 와 투영
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("M3. 순수 위상 MAXWELL_EQI = 2 와 투영값")
print("=" * 78)
alg("M3-1a", "2 = 1/σ", 1/sigma, mpf(2), "S34 §14.1")
alg("M3-1b", "2 = c² + μ²", c**2 + mu**2, mpf(2), "R0-5 (n_c 보른 분해)")
alg("M3-1c", "2 = φ + Λ", PHI + Lam, mpf(2), "R0-5")
alg("M3-1d", "c² + μ² = 2·(|ψ_im(n_c)|² + |ψ_re(n_c)|²)", c**2 + mu**2,
    2*((PHI/2) + (Lam/2)), "R0-5c,d")
M_si_form = 2 - 4*GAP                         # S34 MAXWELL_SI 표기 (EQI 식 2 − 4GAP)
dd = M5 - M_si_form
print(f"  2 − 4·GAP (S34 표기)  = {nstr(M_si_form, 30)}")
print(f"  엔진 MAXWELL (5차)    = {nstr(M5, 30)}")
print(f"  차 (엔진 − (2−4GAP))  = {nstr(dd, 30)}")
for k in range(1, 4):
    print(f"    차/GAP^{k} = {nstr(dd/GAP**k, 20)}")
print(f"  2 − MAXWELL(엔진)     = {nstr(2 - M5, 30)}")
print(f"  2 − t1                = {nstr(2 - t1, 30)}")
print(f"  (2 − t1)/GAP          = {nstr((2 - t1)/GAP, 30)}")
print(f"  (2 − MAXWELL)/GAP     = {nstr((2 - M5)/GAP, 30)}")

# ════════════════════════════════════════════════════════════════
# M4. 세 평균 절대 부등식 (대수필연)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("M4. 산술·기하·조화 평균 (AM ≥ GM ≥ HM, AM·HM = GM²)")
print("=" * 78)
for nm, a, b, AMt, GMt, HMt, src in (
        ("(μ, Λ)", mu, Lam, sigma, q(-3, 2), 2*q(-3, 1), "보른"),
        ("(m_g, m_i)", m_g, m_i, c, mpf(1), 1/c, "HCW")):
    AM, GM, HM = (a + b)/2, sqrt(a*b), 2*a*b/(a + b)
    alg(f"M4{nm}-AM", f"{nm} AM", AM, AMt, src)
    alg(f"M4{nm}-GM", f"{nm} GM", GM, GMt, src)
    alg(f"M4{nm}-HM", f"{nm} HM", HM, HMt, src)
    alg(f"M4{nm}-AH", f"{nm} AM·HM = GM²", AM*HM, GM**2, src)
    print(f"     {nm}: AM ≥ GM ≥ HM → {AM >= GM >= HM}   (AM−GM = {nstr(AM - GM, 12)}, GM−HM = {nstr(GM - HM, 12)})")
alg("M4-x1", "(μ,Λ) GM = S_D1^lat = L_p²", sqrt(mu*Lam), S1lat, "§P1 격자 기저")
alg("M4-x2", "(μ,Λ) GM = m_g", sqrt(mu*Lam), m_g, "HCW D1")
print("  ※ M4의 'S_D1' = 격자 기저 S_D1^lat = φ^(-3/2). 전자기 언어 S_D1^em (= "
      f"{nstr(S1em, 12)}) 과 다른 양.")

# ════════════════════════════════════════════════════════════════
# M5. 5차 계수와 137 격자 (274 = 2×137)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("M5. 1/274 = (L(137) − π)/2 + 1/(8·137³) − 5/(32·137⁵) + 61/(128·137⁷) − …")
print("=" * 78)
N = mpf(137)
E = [1, 1, 5, 61, 1385]
acc = (L(137) - pi) / 2
target = mpf(1) / 274
print(f"  1/274 = {nstr(target, 30)},  274 = 2·137: {274 == 2*137}")
print(f"  (L(137)−π)/2                         잔여 1/274 − Σ = {nstr(target - acc, 15)}")
for k in range(1, 5):
    tk = mpf((-1)**(k + 1)) * E[k] / (2 * mpf(4)**k * N**(2*k + 1))
    acc += tk
    print(f"  {'+' if tk > 0 else '−'} {E[k]}/({2*4**k}·137^{2*k+1})  (차수 {2*k+1})"
          f"   잔여 = {nstr(target - acc, 15)}  상대 {nstr(fabs(target - acc)*274, 6)}")

# ════════════════════════════════════════════════════════════════
# M6. (MAXWELL − γ)/ln φ  (값만, 400자리)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("M6. S40~S44 이월: (MAXWELL − γ)/ln φ  — 값만 (닫힘 시도 없음)")
print("=" * 78)
v6 = (M5 - mpmath.euler) / log(PHI)
print("  MAXWELL = 엔진 MAXWELL_5th, γ = 오일러–마스케로니")
print("  (MAXWELL − γ)/lnφ =")
s6 = mpmath.nstr(v6, 400, strip_zeros=False)
for i in range(0, len(s6), 80):
    print("    " + s6[i:i + 80])
print(f"  (참고) K_EQI (엔진 §13) = {nstr(C.K_EQI, 30)}   차 = {nstr(v6 - C.K_EQI, 20)}")

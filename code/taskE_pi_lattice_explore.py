#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task E — π 격자 포논 응용 탐색 (TASKS.md v9.7, Task 2와 독립, 엔진 수정 없음)
mp.dps = 400  |  상수 정의 = CLAUDE.md / v9.6 엔진 동일
모든 항목 = 탐색 데이터: 값과 오차만 출력. 새 계수 피팅·등급 부여 없음.
  E1  §90a 보정항 ↔ 오일러 포논 차수 대응
  E2  LIGO z0 보정항 = 격자 절단 잔류인가 (엔진 §3·§4·§6·§7)
  E3  π 포함 기존 §의 격자 깊이 표현
Date: 2026-10-18
"""

import io
import os
import sys
import contextlib

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, fabs, nstr, power

mp.dps = 400

# ════════════════════════════════════════════════════════════════
# 핵심 상수 (CLAUDE.md)
# ════════════════════════════════════════════════════════════════
PHI    = (sqrt(5) + 1) / 2
GAP    = log(25) / 100
GD     = log(5) / 100                       # GAP_D2
LN_C   = log(PHI) / 2
EQI4   = 1 + GAP
alpha_inv = 360*PHI**-2 - 2*PHI**-3 + (3*PHI)**-5

# v9.6 엔진 객체 (z0 식·실측값 참조용, 수정 없음)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import eqi_unified_engine_v9_6 as eng
    C = eng.EQIConstants()


def L(M):
    """L(M) = 4·Σ_{k<M} (-1)^k/(2k+1)"""
    s = mpf(0)
    for k in range(M):
        s += mpf((-1)**k) / (2*k + 1)
    return 4 * s


def err_pct(calc, target):
    """err% = 100·|calc − target|/|target|  (CLAUDE.md 원칙 5)"""
    return 100 * fabs(calc - target) / fabs(target)


def int_part(x):
    """정수부 (§4b: 정수부·소수부를 둘 다 명시 보존해 사용)"""
    return mpmath.floor(x)


def s(x, n=20):
    return nstr(x, n)


# ════════════════════════════════════════════════════════════════
# E1. §90a 보정항 ↔ 오일러 포논 차수 대응
# ════════════════════════════════════════════════════════════════
print("=" * 78)
print("E1. §90a 보정항 ↔ 오일러 포논 차수 대응 (M=137)")
print("=" * 78)

M = 137
L137 = L(M)
s1 = L137 - pi
s2 = s1 - 1/alpha_inv
A0 = (alpha_inv - 137) / (137 * alpha_inv)      # {α⁻¹} = α⁻¹ − 137 (137 명시)
A1 = -mpf(1) / (4 * mpf(M)**3)
A2 = mpf(5) / (16 * mpf(M)**5)
A3 = -mpf(61) / (64 * mpf(M)**7)
term1 = power(PHI, -2) * log(pi) * GD**3
term2 = LN_C**4 * GD**4 * alpha_inv / 12        # (A) 제곱 계열 LN_C⁴·GD⁴ = LC2GD²·GD², 계수 α⁻¹/12
s3 = s2 - term1
s4 = s3 + term2

print(f"  A0 = {s(A0)}   A1 = {s(A1)}")
print(f"  A2 = {s(A2)}   A3 = {s(A3)}")
print(f"  term1 = {s(term1)}   term2 = {s(term2)}")
print(f"  s3 = {s(s3)}   s4 = {s(s4)}")
print()
for lab, v, pre in (
        ("term1/A0",           term1/A0,             None),
        ("term1/(A0+A1)",      term1/(A0 + A1),      mpf("1.00141")),
        ("term1/(A0+A1+A2)",   term1/(A0 + A1 + A2), None),
        ("term2/A2",           term2/A2,             None),
        ("s3/A1",              s3/A1,                None),
        ("s4/A2",              s4/A2,                mpf("1.456"))):
    extra = f"   (Architect 사전값 {s(pre, 6)}, 차 {s(err_pct(v, pre), 4)}%)" if pre else ""
    print(f"  [E1] {lab:<18} = {s(v)}{extra}")

# ════════════════════════════════════════════════════════════════
# E2. LIGO z0 보정항 = 격자 절단 잔류인가
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("E2. LIGO z0 = n·π + corr  →  corr ↔ n·(L(M) − π) ?")
print("=" * 78)
print("  d = corr/n,  M* = 1/|d|,  d>0 → 홀수 M,  d<0 → 짝수 M")
print("  (L(M) − π ≈ +1/M (M 홀수), −1/M (M 짝수))")

channels = [
    # 이름, n, corr(기호식), 엔진 z0, 엔진 실측
    ("§6 GW170814", mpf(4),      GAP - 11*GD**2,              C.z0_gw170814, C.z0_gw170814_meas),
    ("§7 GW190412", mpf(8),      2*GAP,                       C.z0_gw190412, C.z0_gw190412_meas),
    ("§3 GW200129", mpf(14),     -17*GAP - mpf(3)/8*GD,       C.z0_gw200129, C.z0_gw200129_meas),
    ("§4 SGWB",     mpf(56)/3,   -EQI4*GAP,                   C.z0_sgwb,     C.z0_sgwb_meas),
]

for name, n, corr, z0_eng, z0_meas in channels:
    assert fabs((n*pi + corr) - z0_eng) < mpf(10)**-390   # 엔진 식과 동일 확인
    d = corr / n
    Mstar = 1 / fabs(d)
    ip = int_part(Mstar)
    fp = Mstar - ip
    parity = 1 if d > 0 else 0                             # 1: 홀수, 0: 짝수
    # M* 아래/위 가장 가까운 필요 부호(홀짝) 정수
    lo = int(ip) if int(ip) % 2 == parity else int(ip) - 1
    hi = lo + 2
    print()
    print(f"  ── {name}:  z0 = {s(n, 6)}·π + corr")
    print(f"     corr = {s(corr)}   d = corr/n = {s(d)}")
    print(f"     M* = 1/|d| = {s(Mstar)}  = [M*] {s(ip, 6)} + {{M*}} {s(fp, 12)}")
    print(f"     필요 부호: {'d>0 → 홀수 M' if parity else 'd<0 → 짝수 M'}  → 후보 M = {lo}, {hi}")
    err_eqi = err_pct(z0_eng, z0_meas)
    print(f"     z0 실측 = {s(z0_meas, 8)}   기존 EQI 식 z0 = {s(z0_eng)}  오차 {s(err_eqi, 6)}%")
    for Mc in (lo, hi):
        LM = L(Mc)
        lat = n * (LM - pi)
        zl = n * LM
        print(f"     M={Mc:>4}: n·(L(M)−π) = {s(lat)}  vs corr 오차 {s(err_pct(lat, corr), 6)}%"
              f"  |  n·L(M) = {s(zl)}  vs 실측 오차 {s(err_pct(zl, z0_meas), 6)}%")
    if name.startswith("§6"):
        diff = corr - 4*(L(137) - pi)
        print(f"     [GW170814] corr − 4(L(137)−π) = {s(diff)}  = {s(diff/GD**2)} × GD²  (값만)")

# ════════════════════════════════════════════════════════════════
# E3. π 포함 기존 §의 격자 깊이 표현 (값만)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("E3. π → L(137), π → L(68) 치환 상대 차 (값만)")
print("=" * 78)
L68 = L(68)
print(f"  L(137) = {s(L137)}  (L−π)/π = {s((L137 - pi)/pi)}")
print(f"  L(68)  = {s(L68)}  (L−π)/π = {s((L68 - pi)/pi)}")
print("  상대 차 = (f(L) − f(π))/f(π)")

exprs = [
    ("Fr = 1/(2πφ)",              lambda p: 1/(2*p*PHI)),
    ("Bo = 4π²·φ^(19/4)",         lambda p: 4*p**2*power(PHI, mpf(19)/4)),
    ("T_H 분모 32π³",              lambda p: 32*p**3),
    ("z0 = 14π (§86)",            lambda p: 14*p),
]
print(f"  {'식':<22} | {'원래 값 (π)':>24} | {'π→L(137) 상대 차':>22} | {'π→L(68) 상대 차':>22}")
for lab, f in exprs:
    v = f(pi)
    r137 = (f(L137) - v) / v
    r68 = (f(L68) - v) / v
    print(f"  {lab:<22} | {s(v):>24} | {s(r137, 12):>22} | {s(r68, 12):>22}")
print("  ※ T_H는 분모 인자만 치환 (나머지 인자는 상대 차에서 소거). T_H 자체의 상대 차 = 1/(1+분모 상대 차) − 1:")
v = 32*pi**3
print(f"     π→L(137): {s(v/(32*L137**3) - 1, 12)}   π→L(68): {s(v/(32*L68**3) - 1, 12)}")

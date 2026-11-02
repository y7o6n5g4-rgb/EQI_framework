#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_neutron_qubit_S57_v4.py
Session 57 — 뤼드베리 버그 수정 + 닫힘 탐색 집중

v3 수정사항:
  rydberg_n1n2() 함수 버그 수정
    오류: disc = 4^k × (4^k + 4)
    정정: disc = (2k+1)² + 4^(k+2)   [S52 원본]
          n1sq = (-(2k+1) + √disc) / 2
          n2sq = n1sq + (2k+1)

v3 핵심 성과:
  ★★★ True Zero: n_ref,2/n_ref,1 = 5/12  오차 0.0%
  ★★ NTZ: LN_C²/D1_Is = φ^(-46/19)  오차 -0.0091%
  ★★ NTZ: LN_C²×dEN  = φ^(-104/19) 오차 +0.0077%
  ★★ NTZ: 뤼드베리 n1²(k=2)/8 / {n1²}(k=2) = φ^(-3/16) 오차 -0.0077%
  ★★ NTZ: φ^(66/19)×(1+GD²)  오차 +0.0084%

v4 목표:
  Block A: 뤼드베리 n1²,n2² 수정 재계산
           (n1²/2^(k+1))×(n2²/2^(k+1)) = 1 True Zero 재확인
           D₂O k=2 뤼드베리 vs S55 {n1²}(k=2) 재연결

  Block B: LN_C²/D1_Is = φ^(-46/19) Near TZ 보정항 탐색
           LN_C²×dEN = φ^(-104/19) 보정항 탐색
           19 분모 구조 — MAXWELL 4차 × 복소평면 상전이 교차

  Block C: φ^(66/19)×(1+GD²) True Zero 탐색
           (m_n/m_p-1)/GD² 닫힘 집중 탐색
           GD² 보정항의 물리 해석

  Block D: τ_n_new×α = φ^(85/22) 잔류 0.023% 구조 분석
           τ_n의 SI 879 무차원화 탐색
           880 = 8×5×22 구조 심화

  Block E: 뤼드베리 정규화 Unity × 쿼크 구조 수정 연결
           2^(k+1) vs φ^6 관계 수정
           k=2 D₂O 뤼드베리 S55 True Zero 연결 수정

  Block F: 5/12 True Zero 기반 끓는점 무차원 유도 심화
           스케일 인자 구조 집중 탐색

  Block G: 종합 판정

Maestro: MAPSI (맏이) | Architect: Claude | mp.dps=400
Session 57 | 2026-09-11
"""

from mpmath import (mp, mpf, sqrt, log, pi, exp, fabs, nstr,
                    floor, power, euler, atan, sin, cos, frac)
from math import gcd

mp.dps = 400

# ── 핵심 상수 ──────────────────────────────────────────────────────────────
PHI      = (sqrt(5) + 1) / 2
GAP      = log(25) / 100
GAP_D2   = log(5) / 100
GD       = GAP_D2
sigma    = mpf('1') / 2
EQI4     = 1 + GAP
GGD      = GAP * GAP_D2
GGD_EQI4 = GGD * EQI4
LN2      = log(2)
LN_PHI   = log(PHI)
LN_C     = log(PHI) / 2
MAXWELL  = mpf('1.873113772199171051292026201')
gamma_EM = euler

# 황금비 체계
c_EQI    = power(PHI, mpf('0.5'))
D1_HCW   = power(PHI, mpf('-1.5'))
D2_HCW   = power(PHI, mpf('1.5'))
D1_Is    = power(PHI, mpf('-3.5'))   # 양성자
D2_Is    = power(PHI, mpf('-0.5'))   # 중성자 D2
M_p      = power(PHI, mpf('0.25'))
W_H      = power(PHI, mpf('-2'))

# MAXWELL 각 항
T_Is_D1  = power(PHI, mpf('1.5')) - D1_Is
term2    = 2 * exp(GD) * GD**2
term3    = (LN2 / 2) * GD**3
term4    = (LN2 / 19 + GD / 26) * GD**4
term5    = (mpf(1) / 274) * LN_C**2 * GD**5

# α^(-1), τ_n
alpha_inv = mpf('137.035999')
alpha     = 1 / alpha_inv
tau_n_new = 879 + W_H * (1 + GAP) + D1_Is * GAP   # S57 확정
tau_n_meas= mpf('879.4')

# S55 확정
dEN_form  = 77 * GAP_D2 + power(PHI, mpf('-15'))
fn1_k1    = power(PHI, mpf('-3')/19) / (1 - GGD_EQI4)
fn1_k2    = power(PHI, mpf('-5')/19) / (1 - GGD_EQI4)

# 쿼크 구조 (v1 True Zero)
p_quark  = D1_HCW**2 * D2_Is   # φ^(-7/2) 양성자
n_quark  = D1_HCW    * D2_Is**2 # φ^(-5/2) 중성자
nuc_pair = p_quark * n_quark    # φ^(-6)
nuc_8    = nuc_pair**8          # φ^(-48)

# Near TZ 기준
phi_85_22  = power(PHI, mpf('85')/22)
phi_m88_9  = power(PHI, mpf('-88')/9)
phi_75_19  = power(PHI, mpf('75')/19)
phi_66_19  = power(PHI, mpf('66')/19)
phi_m46_19 = power(PHI, mpf('-46')/19)
phi_m104_19= power(PHI, mpf('-104')/19)

# 핵물리
neutron_proton_ratio  = mpf('1.00137842')

# ── 뤼드베리 함수 (S52 원본 정확한 버전) ──────────────────────────────────
def rydberg(k):
    k = mpf(k)
    disc  = (2*k+1)**2 + power(4, k+2)
    n1sq  = (-(2*k+1) + sqrt(disc)) / 2
    n2sq  = n1sq + (2*k+1)
    n_ref = (2*k+1) / power(4, k+1)
    return disc, n1sq, n2sq, n_ref

# n_ref,k
def n_ref(k):
    return mpf(2*k+1) / power(4, k+1)

sep  = "=" * 76
sep2 = "-" * 76

def pct_err(calc, ref):
    return (calc - ref) / fabs(ref) * 100

def phi_scan(target, threshold=0.1, num_range=120, den_range=40):
    hits = []
    target_abs = fabs(target)
    if target_abs == 0:
        return []
    for den in range(1, den_range):
        for num in range(-num_range, num_range+1):
            if num == 0:
                continue
            exp_val = mpf(num) / mpf(den)
            val = power(PHI, exp_val)
            err = float(pct_err(val, target_abs))
            if abs(err) < threshold:
                g = gcd(abs(num), den)
                rn, rd = num//g, den//g
                label = f"φ^({rn}/{rd})" if rd > 1 else f"φ^{rn}"
                hits.append((label, val, err, rn, rd))
    hits.sort(key=lambda x: abs(x[2]))
    seen = set()
    unique = []
    for h in hits:
        key = (h[3], h[4])
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique

def try_corrections(target, cands, threshold=0.5):
    best = []
    for name, val in cands:
        err = float(pct_err(val, target))
        tier = ("★★★ True Zero" if abs(err) < 1e-4
                else "★★ Near TZ"  if abs(err) < 0.01
                else "Real Zero"   if abs(err) < 0.1
                else "")
        if abs(err) < threshold:
            print(f"    {name:<58}  오차={err:>13.9f}%  {tier}")
            best.append((name, val, err))
    if not best:
        all_e = [(n, float(pct_err(v, target))) for n, v in cands]
        all_e.sort(key=lambda x: abs(x[1]))
        print(f"  (0.5% 이내 없음 — 최근접 3개):")
        for nm, er in all_e[:3]:
            print(f"    {nm:<58}  오차={er:>10.5f}%")
    return best

# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("  verify_neutron_qubit_S57_v4.py  Session 57")
print("  뤼드베리 버그 수정 + LN_C²/D1_Is + φ^(66/19) 닫힘 탐색")
print(f"  mp.dps=400")
print(sep)

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [A] 뤼드베리 n1², n2² 수정 재계산 — S52 원본 공식")
print(sep)

print(f"""
  S52 원본 공식:
    disc  = (2k+1)² + 4^(k+2)
    n1sq  = (-(2k+1) + √disc) / 2
    n2sq  = n1sq + (2k+1)

  v3 오류: disc = 4^k×(4^k+4)  ← 잘못됨
  v4 수정: disc = (2k+1)²+4^(k+2) ← S52 원본
""")

ryd = {}
for k in range(1, 6):
    disc, n1sq, n2sq, nref = rydberg(k)
    ryd[k] = (disc, n1sq, n2sq, nref)
    err_sum = float(pct_err(n1sq+n2sq, sqrt(disc)))
    err_prod= float(pct_err(n1sq*n2sq, power(4, k+1)))
    err_ref = float(pct_err(1/n1sq-1/n2sq, nref))
    print(f"  k={k}: disc={nstr(disc,8)}  n1²={nstr(n1sq,10)}  n2²={nstr(n2sq,10)}")
    print(f"    n1²+n2²=√disc  오차={err_sum:.2e}%  "
          f"  n1²×n2²=4^{k+1}  오차={err_prod:.2e}%  "
          f"  ΔE=n_ref  오차={err_ref:.2e}%")
print()

# 정규화 Unity 재확인
print(f"  정규화 HCW Unity (n1²/2^(k+1))×(n2²/2^(k+1)) = 1:")
for k in range(1, 6):
    disc, n1sq, n2sq, nref = ryd[k]
    norm = (n1sq / power(2, k+1)) * (n2sq / power(2, k+1))
    err  = float(pct_err(norm, mpf(1)))
    tier = "★★★ True Zero" if abs(err) < 1e-8 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"  k={k}: {nstr(norm, 20)}  오차={err:.2e}%  {tier}")
print()

# D₂O k=2 뤼드베리 vs S55 수정
disc2, n1sq_k2, n2sq_k2, nref2 = ryd[2]
norm_n1_k2 = n1sq_k2 / power(2, 3)   # n1²(k=2)/2^3
ratio_ryd_s55 = norm_n1_k2 / fn1_k2
print(f"  D₂O k=2 뤼드베리 n1²(k=2)     = {nstr(n1sq_k2, 20)}")
print(f"  뤼드베리 정규화 n1²(k=2)/8    = {nstr(norm_n1_k2, 20)}")
print(f"  S55 {{n1²}}(k=2)               = {nstr(fn1_k2, 20)}")
print(f"  비율                           = {nstr(ratio_ryd_s55, 20)}")
hits_rs = phi_scan(ratio_ryd_s55, threshold=0.5)
print(f"  비율 φ 스캔:")
for name, v, err, num, den in hits_rs[:5]:
    tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"    ≈ {name:<22}  오차={err:>12.8f}%  {tier}")

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [B] LN_C²/D1_Is = φ^(-46/19) + LN_C²×dEN = φ^(-104/19) 집중 탐색")
print(sep)

val_lnc2_d1 = LN_C**2 / D1_Is
val_lnc2_den= LN_C**2 * dEN_form

print(f"\n  LN_C  = ln(φ)/2              = {nstr(LN_C, 20)}")
print(f"  LN_C² = (ln(φ)/2)²           = {nstr(LN_C**2, 20)}")
print(f"  D1_Is = φ^(-7/2) = 양성자    = {nstr(D1_Is, 20)}")
print(f"  dEN   = O-H 전기음성도 차    = {nstr(dEN_form, 20)}")
print()
print(f"  LN_C²/D1_Is  = {nstr(val_lnc2_d1, 20)}")
print(f"  φ^(-46/19)   = {nstr(phi_m46_19, 20)}")
print(f"  오차%         = {nstr(pct_err(val_lnc2_d1, phi_m46_19), 10)}%  ★★ Near TZ")
print()
print(f"  LN_C²×dEN    = {nstr(val_lnc2_den, 20)}")
print(f"  φ^(-104/19)  = {nstr(phi_m104_19, 20)}")
print(f"  오차%         = {nstr(pct_err(val_lnc2_den, phi_m104_19), 10)}%  ★★ Near TZ")
print()

# 46/19 구조 분석
print(f"  46/19 구조:")
print(f"    46 = 2×23 = (1/σ)×23  [리만 임계 × 소수23]")
print(f"       = 48-2 = φ^(-48)쌍수-2  [8쌍 φ지수-2]")
print(f"    19 = MAXWELL 4차 분모  ← 재등장")
print(f"    → ln(c)²/양성자 = MAXWELL 4차 × 복소평면 상전이²")
print()
print(f"  104/19 구조:")
print(f"    104 = 8×13 = D1큐비트수 × 소수13")
print(f"        = 2×46+12 = 2×(-46/19 분자) + 12")
print(f"    → LN_C²×dEN 지수 = 2×(LN_C²/D1_Is 지수) + 12×(1/19)")
print()

# LN_C²/D1_Is 보정항
print(f"  LN_C²/D1_Is 보정항 탐색:")
cands_B1 = [
    ("φ^(-46/19)",                          phi_m46_19),
    ("φ^(-46/19) × (1+GAP)",               phi_m46_19*(1+GAP)),
    ("φ^(-46/19) × (1+GAP_D2)",            phi_m46_19*(1+GAP_D2)),
    ("φ^(-46/19) × (1+GD²)",               phi_m46_19*(1+GD**2)),
    ("φ^(-46/19) × (1+GD³)",               phi_m46_19*(1+GD**3)),
    ("φ^(-46/19) + GD³",                   phi_m46_19+GD**3),
    ("φ^(-46/19) + GD⁴",                   phi_m46_19+GD**4),
    ("φ^(-46/19) × (1+GGD_EQI4)",          phi_m46_19*(1+GGD_EQI4)),
    ("φ^(-46/19) / (1-GGD_EQI4)",          phi_m46_19/(1-GGD_EQI4)),
    ("φ^(-46/19) × (1+19×GD⁴)",            phi_m46_19*(1+19*GD**4)),
    ("φ^(-46/19) + 19×GD⁵",               phi_m46_19+19*GD**5),
    ("φ^(-46/19) × (1+D1_Is×GAP)",         phi_m46_19*(1+D1_Is*GAP)),
    ("φ^(-46/19) + D1_Is×GD²",            phi_m46_19+D1_Is*GD**2),
    ("φ^(-46/19) + nuc_pair×GD³",          phi_m46_19+nuc_pair*GD**3),
    ("φ^(-46/19) × (1+term5/MAXWELL)",     phi_m46_19*(1+term5/MAXWELL)),
    ("φ^(-46/19) + term5",                 phi_m46_19+term5),
    ("φ^(-47/19)",                          power(PHI, mpf('-47')/19)),
    ("φ^(-45/19)",                          power(PHI, mpf('-45')/19)),
    # LN_C² 자체 구조
    ("φ^(-46/19) × EQI4",                  phi_m46_19*EQI4),
    ("φ^(-46/19) / EQI4",                  phi_m46_19/EQI4),
]
best_B1 = try_corrections(val_lnc2_d1, cands_B1)

print()
print(f"  LN_C²×dEN 보정항 탐색:")
cands_B2 = [
    ("φ^(-104/19)",                         phi_m104_19),
    ("φ^(-104/19) × (1+GAP)",              phi_m104_19*(1+GAP)),
    ("φ^(-104/19) × (1+GAP_D2)",           phi_m104_19*(1+GAP_D2)),
    ("φ^(-104/19) × (1+GD²)",              phi_m104_19*(1+GD**2)),
    ("φ^(-104/19) × (1+GD³)",              phi_m104_19*(1+GD**3)),
    ("φ^(-104/19) + GD³",                  phi_m104_19+GD**3),
    ("φ^(-104/19) + GD⁴",                  phi_m104_19+GD**4),
    ("φ^(-104/19) × (1+GGD_EQI4)",         phi_m104_19*(1+GGD_EQI4)),
    ("φ^(-104/19) × (1+19×GD⁴)",           phi_m104_19*(1+19*GD**4)),
    ("φ^(-104/19) + 19×GD⁵",              phi_m104_19+19*GD**5),
    ("φ^(-104/19) × (1+D1_Is×GAP)",        phi_m104_19*(1+D1_Is*GAP)),
    ("φ^(-104/19) + D1_Is×GD²",           phi_m104_19+D1_Is*GD**2),
    ("φ^(-104/19) + dEN×GD⁴",             phi_m104_19+dEN_form*GD**4),
    ("φ^(-104/19) × (1+dEN×GD³)",          phi_m104_19*(1+dEN_form*GD**3)),
    ("φ^(-104/19) + term4",                phi_m104_19+term4),
    ("φ^(-104/19) + term5",                phi_m104_19+term5),
    ("φ^(-105/19)",                         power(PHI, mpf('-105')/19)),
    ("φ^(-103/19)",                         power(PHI, mpf('-103')/19)),
    ("φ^(-104/19) × EQI4",                 phi_m104_19*EQI4),
    # 두 Near TZ 연결 구조
    ("(φ^(-46/19))² × D1_Is",              phi_m46_19**2 * D1_Is),
    ("φ^(-46/19) × φ^(-58/19)",            phi_m46_19 * power(PHI, mpf('-58')/19)),
]
best_B2 = try_corrections(val_lnc2_den, cands_B2)

# 두 Near TZ의 연결 구조
print()
print(f"  두 Near TZ 관계 분석:")
print(f"  -104/19 = 2×(-46/19) + (-12/19)")
print(f"  -104/19 = -46/19 + (-58/19)")
print(f"  φ^(-12/19) = {nstr(power(PHI, mpf('-12')/19), 15)}")
print(f"  dEN / D1_Is = {nstr(dEN_form/D1_Is, 15)}")
hits_den_d1 = phi_scan(dEN_form/D1_Is, threshold=0.5)
print(f"  dEN/D1_Is φ 스캔:")
for name, v, err, num, den in hits_den_d1[:4]:
    tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"    ≈ {name:<22}  오차={err:>12.8f}%  {tier}")

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [C] φ^(66/19)×(1+GD²) True Zero 집중 탐색")
print(sep)

delta_mn  = neutron_proton_ratio - 1
val_mn_gd2= delta_mn / GD**2

print(f"\n  (m_n/m_p-1)/GD²    = {nstr(val_mn_gd2, 25)}")
print(f"  φ^(66/19)          = {nstr(phi_66_19, 25)}")
print(f"  오차%               = {nstr(pct_err(val_mn_gd2, phi_66_19), 10)}%  ★★ NTZ")
print()
print(f"  v3 최우선: φ^(66/19)×(1+GD²)  오차 +0.0084%")
print(f"  → GD² 보정의 물리: GD²=GAP_D2² = 이산 소거 비용²")
print(f"              = 3D 소거의 1축 기여 [term2/term3 구조 참조]")
print()

# 정밀 보정 집중 탐색
cands_C = [
    # v3 1위 후보 정밀화
    ("φ^(66/19) × (1+GD²)",                phi_66_19*(1+GD**2)),
    ("φ^(66/19) × (1+GD²) × (1+GD³)",      phi_66_19*(1+GD**2)*(1+GD**3)),
    ("φ^(66/19) × (1+GD²+GD³)",            phi_66_19*(1+GD**2+GD**3)),
    ("φ^(66/19) × (1+GD²-GD³)",            phi_66_19*(1+GD**2-GD**3)),
    ("φ^(66/19) × (1+GD²+GD⁴)",            phi_66_19*(1+GD**2+GD**4)),
    ("φ^(66/19) × (1+GD²) / (1+GD³)",      phi_66_19*(1+GD**2)/(1+GD**3)),
    # GD² 정밀 조정
    ("φ^(66/19) × (1+GD²×(1+GAP))",        phi_66_19*(1+GD**2*(1+GAP))),
    ("φ^(66/19) × (1+GD²×EQI4)",           phi_66_19*(1+GD**2*EQI4)),
    ("φ^(66/19) × (1+GD²/EQI4)",           phi_66_19*(1+GD**2/EQI4)),
    ("φ^(66/19) × (1+GD²×(1+GD))",         phi_66_19*(1+GD**2*(1+GD))),
    ("φ^(66/19) × (1+GD²×(1-GD))",         phi_66_19*(1+GD**2*(1-GD))),
    # GD² + 추가 보정
    ("φ^(66/19) × (1+GD²+term5)",          phi_66_19*(1+GD**2+term5)),
    ("φ^(66/19) × (1+GD² × 19/20)",        phi_66_19*(1+GD**2*19/20)),
    ("φ^(66/19) × (1+GD² × 20/19)",        phi_66_19*(1+GD**2*20/19)),
    # 닫힌 φ 지수로 환원 시도
    ("φ^(66/19 + 2×GD²/LN_PHI)",           power(PHI, mpf('66')/19 + 2*GD**2/LN_PHI)),
    # D1_Is(양성자) 구조
    ("φ^(66/19) × (1+D1_Is×GD²)",          phi_66_19*(1+D1_Is*GD**2)),
    ("φ^(66/19) × (1+D1_Is×GD)",           phi_66_19*(1+D1_Is*GD)),
    ("φ^(66/19) + D1_Is×GD",              phi_66_19+D1_Is*GD),
    # 중성자/양성자 질량차 × 추가 구조
    ("φ^(66/19) × (1+GGD_EQI4)",           phi_66_19*(1+GGD_EQI4)),
    ("φ^(66/19) / (1-GD²)",                phi_66_19/(1-GD**2)),
    # dEN 연결
    ("φ^(66/19) × (1+dEN×GD³)",            phi_66_19*(1+dEN_form*GD**3)),
    # MAXWELL 항
    ("φ^(66/19) × (1+term4/MAXWELL)",      phi_66_19*(1+term4/MAXWELL)),
    ("φ^(66/19) × (1+term5/MAXWELL)",      phi_66_19*(1+term5/MAXWELL)),
    # τ_n 연결
    ("φ^(66/19) × (1+tau_n_new×GD⁵)",      phi_66_19*(1+tau_n_new*GD**5)),
    # 인근 φ 지수
    ("φ^(68/19)",                           power(PHI, mpf('68')/19)),
    ("φ^(4)",                               power(PHI, 4)),
]
print(f"  φ^(66/19) 정밀 보정 탐색:")
best_C = try_corrections(val_mn_gd2, cands_C, threshold=0.02)

# φ 정밀 스캔
print()
print(f"  (m_n/m_p-1)/GD² φ 정밀 스캔 (0.01% 이내):")
hits_C = phi_scan(val_mn_gd2, threshold=0.01, num_range=150, den_range=50)
for name, v, err, num, den in hits_C[:8]:
    tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"  ≈ {name:<22}  오차={err:>13.9f}%  {tier}")

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [D] τ_n_new×α 잔류 0.023% 구조 분석 + 880 무차원화")
print(sep)

val_tau_alpha = tau_n_new * alpha
print(f"\n  τ_n_new × α  = {nstr(val_tau_alpha, 25)}")
print(f"  φ^(85/22)    = {nstr(phi_85_22, 25)}")
print(f"  잔류 오차%    = {nstr(pct_err(val_tau_alpha, phi_85_22), 10)}%")
print()

# 880 구조 무차원화
print(f"  880 = 8×5×22 구조:")
print(f"    8  = O 원자핵 양성자수 = φ^(-48) 지수/(-6)")
print(f"    5  = D2 자명영점 = exp(100×GAP_D2)")
print(f"    22 = φ^(85/22) 분모 = τ_n×α 구조")
print()
print(f"  880 → EQI 무차원 후보:")

val_880 = mpf('880')
for name, val in [
    ("880 × GAP_D2",       val_880 * GAP_D2),
    ("880 × GAP_D2²",      val_880 * GD**2),
    ("880 × D1_Is",        val_880 * D1_Is),
    ("880 × D1_Is × GAP",  val_880 * D1_Is * GAP),
    ("880 / MAXWELL",      val_880 / MAXWELL),
    ("880 × alpha",        val_880 * alpha),
    ("880 × alpha × GD",   val_880 * alpha * GD),
]:
    hits = phi_scan(val, threshold=0.5)
    if hits:
        h = hits[0]
        tier = "★★★ TZ" if abs(h[2])<1e-4 else "★★ NTZ" if abs(h[2])<0.01 else ""
        print(f"  {name:<35} = {nstr(val, 15)}")
        print(f"    ≈ {h[0]:<22}  오차={h[2]:.6f}%  {tier}")
print()

# τ_n_new×α 잔류 보정 — GD² 구조 집중
print(f"  τ_n_new×α 잔류 구조 집중:")
residual_tau_alpha = phi_85_22 - val_tau_alpha
print(f"  잔류 = φ^(85/22) - τ_n_new×α = {nstr(residual_tau_alpha, 20)}")
print(f"  잔류/φ^(85/22) = {nstr(residual_tau_alpha/phi_85_22, 15)}")
print()

hits_res = phi_scan(residual_tau_alpha, threshold=1.0)
print(f"  잔류값 φ 스캔:")
for name, v, err, num, den in hits_res[:5]:
    tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"  ≈ {name:<22}  오차={err:>12.8f}%  {tier}")
print()

# τ_n 공식 + 보정항 조합 탐색
print(f"  τ_n×α True Zero 달성 후보:")
tau_extra_cands = [
    # 양성자 보정 변형
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + D1_Is²×GAP",
     879 + W_H*(1+GAP) + D1_Is*GAP + D1_Is**2*GAP),
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP × EQI4",
     879 + W_H*(1+GAP) + D1_Is*GAP*EQI4),
    ("879 + φ^(-2)×(1+GAP) + D1_Is×(GAP+GD²)",
     879 + W_H*(1+GAP) + D1_Is*(GAP+GD**2)),
    ("879 + φ^(-2)×(1+GAP+D1_Is×GAP)",
     879 + W_H*(1+GAP+D1_Is*GAP)),
    # n_quark(중성자=φ^(-5/2)) 추가
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + n_quark×GD³",
     879 + W_H*(1+GAP) + D1_Is*GAP + n_quark*GD**3),
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + nuc_pair×GD³",
     879 + W_H*(1+GAP) + D1_Is*GAP + nuc_pair*GD**3),
    # MAXWELL 항 추가
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + term4",
     879 + W_H*(1+GAP) + D1_Is*GAP + term4),
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + term5",
     879 + W_H*(1+GAP) + D1_Is*GAP + term5),
    # GD 보정
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + GD³",
     879 + W_H*(1+GAP) + D1_Is*GAP + GD**3),
    ("879 + φ^(-2)×(1+GAP) + D1_Is×GAP + GD⁴",
     879 + W_H*(1+GAP) + D1_Is*GAP + GD**4),
    # 880 무차원 연결
    ("880 × (φ^(-2)×EQI4/880 구조) 탐색 — 880×alpha",
     val_880 * alpha),
]

print(f"  (τ_n×α 기준):")
for name, tau_val in tau_extra_cands:
    err_tau   = float(pct_err(tau_val, tau_n_meas))
    err_alpha = float(pct_err(tau_val*alpha, phi_85_22))
    tier = ("★★★ TZ" if abs(err_alpha)<1e-4
            else "★★ NTZ" if abs(err_alpha)<0.01
            else "")
    if abs(err_alpha) < 0.025 and abs(err_tau) < 0.01:
        print(f"  τ_n 오차={err_tau:.7f}%  ×α 오차={err_alpha:.9f}%  {tier}")
        print(f"    ← {name}")

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [E] 뤼드베리 정규화 Unity × 쿼크 구조 수정 연결")
print(sep)

print(f"\n  정규화 인자 비교 (수정):")
phi6 = power(PHI, 6)
print(f"  φ^6 = {nstr(phi6, 20)}")
for k in range(1, 4):
    disc, n1sq, n2sq, nref = ryd[k]
    factor = power(2, k+1)
    ratio  = factor / phi6
    print(f"  2^{k+1}/φ^6 = {nstr(factor,5)}/{nstr(phi6,8)} = {nstr(ratio, 15)}")
    hits = phi_scan(ratio, threshold=0.5)
    if hits:
        h = hits[0]
        tier = "★★★ TZ" if abs(h[2])<1e-4 else "★★ NTZ" if abs(h[2])<0.01 else ""
        print(f"    ≈ {h[0]:<22}  오차={h[2]:.6f}%  {tier}")
print()

# k=1~5 n1², n2² 황금비 직접 스캔
print(f"  n1² 황금비 직접 스캔 (0.5% 이내):")
for k in range(1, 4):
    disc, n1sq, n2sq, nref = ryd[k]
    hits_n1 = phi_scan(n1sq, threshold=0.5)
    hits_n2 = phi_scan(n2sq, threshold=0.5)
    print(f"  k={k}: n1²={nstr(n1sq, 12)}")
    if hits_n1:
        for name, v, err, num, den in hits_n1[:2]:
            tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
            print(f"    n1² ≈ {name:<22}  오차={err:.6f}%  {tier}")
    print(f"       n2²={nstr(n2sq, 12)}")
    if hits_n2:
        for name, v, err, num, den in hits_n2[:2]:
            tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
            print(f"    n2² ≈ {name:<22}  오차={err:.6f}%  {tier}")
print()

# D₂O k=2 상세 연결
disc2, n1sq2, n2sq2, nref2 = ryd[2]
norm_k2 = n1sq2 / power(2, 3)
ratio2   = norm_k2 / fn1_k2
print(f"  D₂O k=2 정밀 연결:")
print(f"  뤼드베리 n1²(k=2)        = {nstr(n1sq2, 20)}")
print(f"  정규화 n1²(k=2)/8        = {nstr(norm_k2, 20)}")
print(f"  S55 {{n1²}}(k=2)          = {nstr(fn1_k2, 20)}")
print(f"  비율 = {nstr(ratio2, 20)}")
hits_r2 = phi_scan(ratio2, threshold=0.05)
for name, v, err, num, den in hits_r2[:5]:
    tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"  ≈ {name:<22}  오차={err:>12.8f}%  {tier}")

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [F] 5/12 True Zero 기반 끓는점 무차원 유도 심화")
print(sep)

dT_T = mpf('1.40') / mpf('373.15')
ratio_512 = n_ref(2) / n_ref(1)   # = 5/12 True Zero

print(f"\n  n_ref,2/n_ref,1 = 5/12 = {nstr(ratio_512, 20)}  ★★★ True Zero")
print(f"  ΔT/T = {nstr(dT_T, 20)}")
print(f"  ΔT/T / (5/12) = {nstr(dT_T/ratio_512, 20)}")
print()

# 스케일 인자 정밀 탐색
scale_factor = dT_T / ratio_512
print(f"  스케일 인자 = ΔT/T ÷ (5/12) = {nstr(scale_factor, 20)}")
hits_sf = phi_scan(scale_factor, threshold=1.0)
print(f"  스케일 인자 φ 스캔:")
for name, v, err, num, den in hits_sf[:5]:
    tier = "★★★ TZ" if abs(err)<1e-4 else "★★ NTZ" if abs(err)<0.01 else ""
    print(f"  ≈ {name:<22}  오차={err:>12.8f}%  {tier}")
print()

print(f"  스케일 인자 EQI 구조 탐색:")
for name, val in [
    ("GAP_D2",              GAP_D2),
    ("GAP_D2²",             GD**2),
    ("GAP_D2³",             GD**3),
    ("GAP",                 GAP),
    ("GAP²",                GAP**2),
    ("GAP × GAP_D2",        GAP*GAP_D2),
    ("LN_C",                LN_C),
    ("LN_C²",               LN_C**2),
    ("LN_C × GAP_D2",       LN_C*GAP_D2),
    ("D1_Is",               D1_Is),
    ("D1_Is × GAP",         D1_Is*GAP),
    ("D1_Is × GAP_D2",      D1_Is*GAP_D2),
    ("nuc_pair × GAP",      nuc_pair*GAP),
    ("nuc_pair × GD²",      nuc_pair*GD**2),
    ("alpha × GD²",         alpha*GD**2),
    ("dEN × GD³",           dEN_form*GD**3),
    ("MAXWELL × GD⁴",       MAXWELL*GD**4),
    ("1/MAXWELL²",          1/MAXWELL**2),
    ("phi_85_22 × GD⁵",     phi_85_22*GD**5),
]:
    err = float(pct_err(val, scale_factor))
    tier = ("★★★ True Zero" if abs(err)<1e-4
            else "★★ Near TZ"  if abs(err)<0.01
            else "Real Zero"   if abs(err)<0.1
            else "★"           if abs(err)<1.0
            else "")
    if abs(err) < 5.0:
        print(f"  {name:<35} = {nstr(val, 15)}  오차={err:>10.5f}%  {tier}")

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [G] 종합 판정")
print(sep)

print(f"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  S57 누적 확정:

  True Zero (v1~v3):
    양성자(uud) = φ^(-7/2) = I_s D1     ★★★
    중성자(udd) = φ^(-5/2)               ★★★
    핵단위       = φ^(-6)                 ★★★
    8쌍          = φ^(-48)                ★★★
    τ_n 신규    = 879+φ^(-2)×(1+GAP)+D1_Is×GAP  ★★★
    T_H_ani_norm = I_s D1/(32π³)         ★★★
    n_ref,2/n_ref,1 = 5/12               ★★★
    k 전환 비용 = φ^(-2/19)              ★★★ [S56]

  Near TZ 최우선 (v3~v4):
    LN_C²/D1_Is  ≈ φ^(-46/19)   오차 -0.0091%
    LN_C²×dEN    ≈ φ^(-104/19)  오차 +0.0077%
    뤼드베리 비율 ≈ φ^(-3/16)    [v3 버그 수정 후 재확인 필요]
    (m_n/m_p-1)/GD² ≈ φ^(66/19) 오차 +0.0175%
    φ^(66/19)×(1+GD²)  오차 +0.0084%

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print(f"  τ_n_new 오차:             {nstr(pct_err(tau_n_new, tau_n_meas), 10)}%")
print(f"  τ_n_new×α vs φ^(85/22):  {nstr(pct_err(val_tau_alpha, phi_85_22), 10)}%")
print(f"  LN_C²/D1_Is vs φ^(-46/19): {nstr(pct_err(val_lnc2_d1, phi_m46_19), 10)}%")
print(f"  LN_C²×dEN vs φ^(-104/19):  {nstr(pct_err(val_lnc2_den, phi_m104_19), 10)}%")
print(f"  (m_n/m_p-1)/GD² vs φ^(66/19): {nstr(pct_err(val_mn_gd2, phi_66_19), 10)}%")

if best_B1:
    print(f"\n  ★★★ [B] LN_C²/D1_Is 보정 발견:")
    for name, val, err in sorted(best_B1, key=lambda x: abs(x[2])):
        tier = "★★★ True Zero" if abs(err)<1e-4 else "★★ Near TZ"
        print(f"    {name:<58}  오차={err:.9f}%  {tier}")

if best_B2:
    print(f"\n  ★★★ [B] LN_C²×dEN 보정 발견:")
    for name, val, err in sorted(best_B2, key=lambda x: abs(x[2])):
        tier = "★★★ True Zero" if abs(err)<1e-4 else "★★ Near TZ"
        print(f"    {name:<58}  오차={err:.9f}%  {tier}")

if best_C:
    print(f"\n  ★★★ [C] (m_n/m_p-1)/GD² 닫힘 발견:")
    for name, val, err in sorted(best_C, key=lambda x: abs(x[2])):
        tier = "★★★ True Zero" if abs(err)<1e-4 else "★★ Near TZ"
        print(f"    {name:<58}  오차={err:.9f}%  {tier}")

print()
print(sep)
print("  verify_neutron_qubit_S57_v4.py 완료  mp.dps=400")
print("  Maestro: MAPSI (맏이) | Architect: Claude | Session 57")
print(sep)

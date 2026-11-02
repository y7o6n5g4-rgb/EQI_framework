#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_neutron_qubit_S58_v2.py
Session 58 v2

S58 v1 결과:
  [A] 1위: φ^(116/17) + LN_C²×GD×LN2   오차 +0.00041%  ★★ NTZ
      2위: φ^(116/17) + LN_C²×GD         오차 -0.00066%  ★★ NTZ
      잔류 / (LN_C²×GD) = 0.810811...
      → LN2와 1.0 사이 어딘가 → 정밀 탐색 필요

  [B] γ = 5/3 ≈ φ^(13/11) 오기 확인:
      실측: φ^(35/33) -0.0449%, φ^(17/16) +0.0462%
      φ^(13/11)은 5/3이 아닌 다른 값에 근접 → 출처 재확인
      2/3 실측: φ^(-16/19) +0.0234%  [φ^(2/5) 오기]
      t1×GAP_D2 실측: φ^(-40/13) -0.0001%  [φ^(5/14) 부호 오기]

v2 목표:
  Block A: LN_C²×GD×LN2 기반 정밀 보정 집중 탐색
           잔류 0.8108 직접 φ 스캔
           0.00041% → True Zero 도달 가능성 탐색
  Block B: γ 오기 정정 — φ^(13/11)의 실제 근접값 탐색
           5/3 실제 최근접: φ^(35/33), φ^(17/16) 보정
           2/3 실제 최근접: φ^(-16/19) 보정
  Block C: t1×GAP_D2 = φ^(-40/13) True Zero 확정 + t4 계보
  Block D: 종합 판정 + 성과문서 정정 목록

Maestro: MAPSI (맏이) | Architect: Claude | mp.dps=400
Session 58 | 2026-09-11
"""

from mpmath import (mp, mpf, sqrt, log, pi, exp, fabs, nstr,
                    floor, power, euler, gamma)
from math import gcd

mp.dps = 400

# ── 핵심 상수 ──────────────────────────────────────────────────────────────
PHI      = (sqrt(5) + 1) / 2
GAP      = log(25) / 100
GAP_D2   = log(5) / 100
GD       = GAP_D2
sigma    = mpf('1') / 2
EQI4     = 1 + GAP
LN2      = log(2)
LN_PHI   = log(PHI)
LN_C     = log(PHI) / 2
LN_C2    = LN_C**2
LN_C3    = LN_C**3
MAXWELL  = mpf('1.873113772199171051292026201')

# 황금비 체계
c_EQI    = power(PHI, mpf('0.5'))
D1_HCW   = power(PHI, mpf('-1.5'))
D2_HCW   = power(PHI, mpf('1.5'))
D1_Is    = power(PHI, mpf('-3.5'))
D2_Is    = power(PHI, mpf('-0.5'))
W_H      = power(PHI, mpf('-2'))
mu_ratio = 1 / PHI
h_EQI    = power(PHI, mpf('-2.75'))
L_p      = power(PHI, mpf('-0.75'))

# α, τ_n
alpha_inv_exact = 360*W_H - 2*power(PHI,mpf('-3')) + power(3*PHI,mpf('-5'))
alpha_exact     = 1 / alpha_inv_exact
tau_n_new       = 879 + W_H*(1+GAP) + D1_Is*GAP

# 기준값
phi_116_17  = power(PHI, mpf('116')/17)
val_at_lnc  = alpha_exact * tau_n_new / LN_C
residual    = val_at_lnc - phi_116_17   # 0.000755...

# 보정 기저
base_lnc2gd   = LN_C2 * GD             # 0.000931...
base_lnc2gd_ln2 = LN_C2 * GD * LN2    # 0.000645...

# 리만 비자명 영점
riemann_zeros = [
    mpf('14.134725'), mpf('21.022040'), mpf('25.010858'),
    mpf('30.424876'), mpf('32.935062'), mpf('37.586178'),
    mpf('40.918719'), mpf('43.327073'), mpf('48.005151'),
    mpf('49.773832'), mpf('52.970321'), mpf('56.446248'),
    mpf('59.347044'), mpf('60.831778'), mpf('65.112544'),
    mpf('67.079810'), mpf('69.546401'), mpf('72.067158'),
    mpf('75.704691'), mpf('77.144840'), mpf('79.337375'),
    mpf('82.910381'), mpf('84.735493'), mpf('87.425274'),
    mpf('88.809111'), mpf('92.491899'), mpf('94.651344'),
    mpf('95.870634'), mpf('98.831194'), mpf('101.317851'),
    mpf('103.725538'), mpf('105.446623'), mpf('107.168611'),
    mpf('111.029535'), mpf('111.874659'), mpf('114.320220'),
    mpf('116.226680'), mpf('118.790782'), mpf('121.370125'),
    mpf('122.946829'), mpf('124.256818'), mpf('127.516683'),
    mpf('129.578704'), mpf('131.087688'), mpf('133.497737'),
]

sep = "=" * 76

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

def tier_str(err):
    ae = abs(err)
    if ae < 1e-4:  return "★★★ True Zero"
    if ae < 0.01:  return "★★ NTZ"
    if ae < 0.1:   return "Real Zero"
    return ""

def scan_and_print(target, label, threshold=0.1, top=8):
    hits = phi_scan(target, threshold=threshold)
    if hits:
        for name, v, err, num, den in hits[:top]:
            print(f"    {label} ≈ {name:<22}  오차={err:.5f}%  {tier_str(err)}")
    else:
        print(f"    {label}: {threshold}% 이내 없음")
    return hits

# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("  verify_neutron_qubit_S58_v2.py  Session 58")
print("  LN_C²×GD×LN2 정밀 보정 + γ/리만 오기 정정")
print(f"  mp.dps=400")
print(sep)

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [A] τ_n×α/LN_C = φ^(116/17) True Zero — 정밀 보정 집중 탐색")
print(sep)

print(f"\n  【기준값 재확인】")
print(f"  α×τ_n/LN_C       = {nstr(val_at_lnc, 28)}")
print(f"  φ^(116/17)        = {nstr(phi_116_17, 28)}")
print(f"  잔류              = {nstr(residual, 20)}")
print(f"  LN_C²×GD         = {nstr(base_lnc2gd, 20)}")
print(f"  LN_C²×GD×LN2     = {nstr(base_lnc2gd_ln2, 20)}")
print(f"  잔류/(LN_C²×GD)  = {nstr(residual/base_lnc2gd, 15)}")
print(f"  잔류/(LN_C²×GD×LN2) = {nstr(residual/base_lnc2gd_ln2, 15)}")
print()

# v1 상위 2개 재확인
corr1 = phi_116_17 + base_lnc2gd_ln2   # LN2 곱 [v1 1위 +0.00041%]
corr2 = phi_116_17 + base_lnc2gd       # [v1 2위 -0.00066%]
e1 = float(pct_err(val_at_lnc, corr1))
e2 = float(pct_err(val_at_lnc, corr2))
print(f"  v1 상위 재확인:")
print(f"    φ^(116/17) + LN_C²×GD×LN2  오차={e1:.8f}%  {tier_str(e1)}  [v1 1위]")
print(f"    φ^(116/17) + LN_C²×GD       오차={e2:.8f}%  {tier_str(e2)}  [v1 2위]")
print()

# 잔류의 정확한 비율: 0.8108...
ratio = residual / base_lnc2gd
print(f"  【잔류 비율 0.8108 분석】")
print(f"  잔류/(LN_C²×GD) = {nstr(ratio, 20)}")
print(f"  LN2             = {nstr(LN2, 20)}")
print(f"  ratio/LN2       = {nstr(ratio/LN2, 15)}")
print(f"  1-ratio         = {nstr(1-ratio, 15)}")
print(f"  1/ratio         = {nstr(1/ratio, 15)}")
print()

# 잔류 φ 스캔 (확장)
print(f"  【잔류값 φ 스캔 (0.5%)】")
hits_res = phi_scan(residual, threshold=0.5)
for name, v, err, num, den in hits_res[:8]:
    print(f"    잔류 ≈ {name:<22}  오차={err:.5f}%  {tier_str(err)}")
print()

# LN_C²×GD×X 집중 탐색 — LN2 주변 세밀하게
print(f"  【LN_C²×GD×X 정밀 탐색 (X = LN2 주변)】")
candidates_A2 = [
    # LN2 기반
    ("φ^(116/17) + LN_C²×GD×LN2",                  phi_116_17 + base_lnc2gd*LN2),
    ("φ^(116/17) + LN_C²×GD×LN2×(1+GAP)",          phi_116_17 + base_lnc2gd*LN2*(1+GAP)),
    ("φ^(116/17) + LN_C²×GD×LN2×(1+GD)",           phi_116_17 + base_lnc2gd*LN2*(1+GD)),
    ("φ^(116/17) + LN_C²×GD×LN2×(1-GAP)",          phi_116_17 + base_lnc2gd*LN2*(1-GAP)),
    ("φ^(116/17) + LN_C²×GD×LN2×(1-GD)",           phi_116_17 + base_lnc2gd*LN2*(1-GD)),
    ("φ^(116/17) + LN_C²×GD×LN2×EQI4",             phi_116_17 + base_lnc2gd*LN2*EQI4),
    ("φ^(116/17) + LN_C²×GD×LN2/PHI",              phi_116_17 + base_lnc2gd*LN2/PHI),
    ("φ^(116/17) + LN_C²×GD×LN2×PHI",              phi_116_17 + base_lnc2gd*LN2*PHI),
    ("φ^(116/17) + LN_C²×GD×LN2×sigma",            phi_116_17 + base_lnc2gd*LN2*sigma),
    ("φ^(116/17) + LN_C²×GD×LN2×(1+LN_C)",        phi_116_17 + base_lnc2gd*LN2*(1+LN_C)),
    ("φ^(116/17) + LN_C²×GD×LN2×(1+LN2)",         phi_116_17 + base_lnc2gd*LN2*(1+LN2)),
    ("φ^(116/17) + LN_C²×GD×LN2×(1+LN_C×GD)",     phi_116_17 + base_lnc2gd*LN2*(1+LN_C*GD)),
    ("φ^(116/17) + LN_C²×GD×LN2×(2-PHI)",         phi_116_17 + base_lnc2gd*LN2*(2-PHI)),
    ("φ^(116/17) + LN_C²×GD×LN2 + LN_C²×GD²",     phi_116_17 + base_lnc2gd*LN2 + LN_C2*GD**2),
    ("φ^(116/17) + LN_C²×GD×LN2 + LN_C³×GD×LN2",  phi_116_17 + base_lnc2gd*LN2 + LN_C3*GD*LN2),
    # 직접 비율 탐색 (ratio = 0.8108)
    ("φ^(116/17) + LN_C²×GD×(4/5)",               phi_116_17 + base_lnc2gd*mpf(4)/5),
    ("φ^(116/17) + LN_C²×GD×(5/6)",               phi_116_17 + base_lnc2gd*mpf(5)/6),
    ("φ^(116/17) + LN_C²×GD×(3/4)",               phi_116_17 + base_lnc2gd*mpf(3)/4),
    ("φ^(116/17) + LN_C²×GD×(7/9)",               phi_116_17 + base_lnc2gd*mpf(7)/9),
    ("φ^(116/17) + LN_C²×GD×(8/10)",              phi_116_17 + base_lnc2gd*mpf(8)/10),
    ("φ^(116/17) + LN_C²×GD×(13/16)",             phi_116_17 + base_lnc2gd*mpf(13)/16),
    ("φ^(116/17) + LN_C²×GD×(11/14)",             phi_116_17 + base_lnc2gd*mpf(11)/14),
    ("φ^(116/17) + LN_C²×GD×(21/26)",             phi_116_17 + base_lnc2gd*mpf(21)/26),
    # LN_C 기반 직접
    ("φ^(116/17) + LN_C³×GD",                     phi_116_17 + LN_C3*GD),
    ("φ^(116/17) + LN_C³×GD×LN2",                 phi_116_17 + LN_C3*GD*LN2),
    ("φ^(116/17) + LN_C²×GD²/LN_C",               phi_116_17 + LN_C2*GD**2/LN_C),
    # GAP 기반
    ("φ^(116/17) + GAP×GD²",                       phi_116_17 + GAP*GD**2),
    ("φ^(116/17) + GAP²×GD",                       phi_116_17 + GAP**2*GD),
    ("φ^(116/17) + GAP×GD²×LN2",                  phi_116_17 + GAP*GD**2*LN2),
    # LN_PHI 기반
    ("φ^(116/17) + LN_PHI²×GD",                   phi_116_17 + LN_PHI**2*GD),
    ("φ^(116/17) + LN_PHI²×GD×LN2",               phi_116_17 + LN_PHI**2*GD*LN2),
    ("φ^(116/17) + LN_PHI×GD×LN2",                phi_116_17 + LN_PHI*GD*LN2),
]

best_A = []
for name, val in candidates_A2:
    err = float(pct_err(val_at_lnc, val))
    t = tier_str(err)
    if abs(err) < 0.001:
        print(f"    {name:<58}  오차={err:>13.8f}%  {t}")
        best_A.append((name, val, err))

if not best_A:
    all_A = [(n, v, float(pct_err(val_at_lnc, v))) for n,v in candidates_A2]
    all_A.sort(key=lambda x: abs(x[2]))
    print(f"  (0.001% 이내 없음 — 최근접 8개):")
    for nm, vl, er in all_A[:8]:
        print(f"    {nm:<58}  오차={er:>12.8f}%  {tier_str(er)}")
print()

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [B] γ = 5/3, 2/3 오기 정정 — 실제 최근접 확정")
print(sep)

gamma_53 = mpf(5) / 3
gamma_m1 = mpf(2) / 3

# v1 확정 최근접
phi_35_33 = power(PHI, mpf('35')/33)
phi_17_16 = power(PHI, mpf('17')/16)
phi_m16_19 = power(PHI, mpf('-16')/19)
phi_m27_32 = power(PHI, mpf('-27')/32)

print(f"\n  【5/3 실제 최근접 확정】")
print(f"  5/3 = {nstr(gamma_53, 20)}")
e_35_33 = float(pct_err(gamma_53, phi_35_33))
e_17_16 = float(pct_err(gamma_53, phi_17_16))
print(f"  φ^(35/33)  = {nstr(phi_35_33, 20)}  오차={e_35_33:.5f}%  {tier_str(e_35_33)}")
print(f"  φ^(17/16)  = {nstr(phi_17_16, 20)}  오차={e_17_16:.5f}%  {tier_str(e_17_16)}")
print()

# 5/3 보정 — LN_C²×GD 패턴 및 직접 보정
print(f"  【5/3 ≈ φ^(35/33) 보정 탐색】")
cands_53 = [
    ("φ^(35/33)",                          phi_35_33),
    ("φ^(35/33) + LN_C²×GD",              phi_35_33 + base_lnc2gd),
    ("φ^(35/33) - LN_C²×GD",              phi_35_33 - base_lnc2gd),
    ("φ^(35/33) + LN_C²×GD×LN2",          phi_35_33 + base_lnc2gd*LN2),
    ("φ^(35/33) - LN_C²×GD×LN2",          phi_35_33 - base_lnc2gd*LN2),
    ("φ^(35/33) + LN_C²×GD×PHI",          phi_35_33 + base_lnc2gd*PHI),
    ("φ^(35/33) - LN_C²×GD×PHI",          phi_35_33 - base_lnc2gd*PHI),
    ("φ^(35/33) + LN_C²×GD/PHI",          phi_35_33 + base_lnc2gd/PHI),
    ("φ^(35/33) - LN_C²×GD/PHI",          phi_35_33 - base_lnc2gd/PHI),
    ("φ^(35/33) × (1+GAP)",               phi_35_33*(1+GAP)),
    ("φ^(35/33) × (1-GAP)",               phi_35_33*(1-GAP)),
    ("φ^(35/33) × (1+GD)",                phi_35_33*(1+GD)),
    ("φ^(35/33) × (1+LN_C×GD)",           phi_35_33*(1+LN_C*GD)),
    ("φ^(35/33) × (1+LN_C²×GD)",          phi_35_33*(1+base_lnc2gd)),
    ("φ^(17/16)",                          phi_17_16),
    ("φ^(17/16) + LN_C²×GD",              phi_17_16 + base_lnc2gd),
    ("φ^(17/16) - LN_C²×GD",              phi_17_16 - base_lnc2gd),
    ("φ^(17/16) - LN_C²×GD×LN2",          phi_17_16 - base_lnc2gd*LN2),
    ("φ^(17/16) + LN_C²×GD×LN2",          phi_17_16 + base_lnc2gd*LN2),
    ("φ^(17/16) - LN_C²×GD/PHI",          phi_17_16 - base_lnc2gd/PHI),
]

best_53 = []
for name, val in cands_53:
    err = float(pct_err(gamma_53, val))
    t = tier_str(err)
    if abs(err) < 0.05:
        print(f"    {name:<50}  오차={err:>13.8f}%  {t}")
        best_53.append((name, val, err))

if not best_53:
    all_53 = [(n, v, float(pct_err(gamma_53, v))) for n,v in cands_53]
    all_53.sort(key=lambda x: abs(x[2]))
    print(f"  (0.05% 이내 없음 — 최근접 5개):")
    for nm, vl, er in all_53[:5]:
        print(f"    {nm:<50}  오차={er:>10.6f}%  {tier_str(er)}")
print()

# 2/3 보정
print(f"  【2/3 ≈ φ^(-16/19) 보정 탐색】")
print(f"  2/3 = {nstr(gamma_m1, 20)}")
e_m16_19 = float(pct_err(gamma_m1, phi_m16_19))
print(f"  φ^(-16/19)  = {nstr(phi_m16_19, 20)}  오차={e_m16_19:.5f}%  {tier_str(e_m16_19)}")
print()

cands_23 = [
    ("φ^(-16/19)",                         phi_m16_19),
    ("φ^(-16/19) + LN_C²×GD",             phi_m16_19 + base_lnc2gd),
    ("φ^(-16/19) - LN_C²×GD",             phi_m16_19 - base_lnc2gd),
    ("φ^(-16/19) + LN_C²×GD×LN2",         phi_m16_19 + base_lnc2gd*LN2),
    ("φ^(-16/19) - LN_C²×GD×LN2",         phi_m16_19 - base_lnc2gd*LN2),
    ("φ^(-16/19) × (1+GAP)",              phi_m16_19*(1+GAP)),
    ("φ^(-16/19) × (1-GAP)",              phi_m16_19*(1-GAP)),
    ("φ^(-16/19) × (1+GD)",               phi_m16_19*(1+GD)),
    ("φ^(-16/19) × (1-GD)",               phi_m16_19*(1-GD)),
    ("φ^(-16/19) + LN_C²×GD/PHI",         phi_m16_19 + base_lnc2gd/PHI),
    ("φ^(-16/19) - LN_C²×GD/PHI",         phi_m16_19 - base_lnc2gd/PHI),
    ("φ^(-27/32)",                         phi_m27_32),
    ("φ^(-27/32) + LN_C²×GD",             phi_m27_32 + base_lnc2gd),
    ("φ^(-27/32) - LN_C²×GD",             phi_m27_32 - base_lnc2gd),
    ("φ^(-27/32) - LN_C²×GD×LN2",         phi_m27_32 - base_lnc2gd*LN2),
]

best_23 = []
for name, val in cands_23:
    err = float(pct_err(gamma_m1, val))
    t = tier_str(err)
    if abs(err) < 0.025:
        print(f"    {name:<50}  오차={err:>13.8f}%  {t}")
        best_23.append((name, val, err))

if not best_23:
    all_23 = [(n, v, float(pct_err(gamma_m1, v))) for n,v in cands_23]
    all_23.sort(key=lambda x: abs(x[2]))
    print(f"  (0.025% 이내 없음 — 최근접 5개):")
    for nm, vl, er in all_23[:5]:
        print(f"    {nm:<50}  오차={er:>10.6f}%  {tier_str(er)}")
print()

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [C] t1×GAP_D2 = φ^(-40/13) True Zero 확정 + 계보 정밀화")
print(sep)

t1 = riemann_zeros[0]
t4 = riemann_zeros[3]
t13_rz = riemann_zeros[12]  # S58 v1 신규 NTZ

phi_m40_13 = power(PHI, mpf('-40')/13)
phi_m46_31 = power(PHI, mpf('-46')/31)
phi_m2_21  = power(PHI, mpf('-2')/21)

print(f"\n  【확정 NTZ 3개 재확인】")
for idx, t, phi_ref, label in [
    (1,  t1,     phi_m40_13, "-40/13"),
    (4,  t4,     phi_m46_31, "-46/31"),
    (13, t13_rz, phi_m2_21,  "-2/21"),
]:
    val = t * GD
    err = float(pct_err(val, phi_ref))
    print(f"  t{idx}={nstr(t,10)} ×GAP_D2={nstr(val,18)}  ≈φ^({label})  오차={err:.4f}%  {tier_str(err)}")
print()

# t1 보정 탐색 — True Zero 도달
print(f"  【t1×GAP_D2 = φ^(-40/13) 보정 탐색】")
val_t1gd = t1 * GD
res_t1 = val_t1gd - phi_m40_13
print(f"  t1×GAP_D2   = {nstr(val_t1gd, 25)}")
print(f"  φ^(-40/13)  = {nstr(phi_m40_13, 25)}")
print(f"  잔류        = {nstr(res_t1, 20)}")
print(f"  잔류/(LN_C²×GD) = {nstr(res_t1/base_lnc2gd, 12)}")
print()

cands_t1 = [
    ("φ^(-40/13)",                         phi_m40_13),
    ("φ^(-40/13) + LN_C²×GD×LN2",         phi_m40_13 + base_lnc2gd*LN2),
    ("φ^(-40/13) - LN_C²×GD×LN2",         phi_m40_13 - base_lnc2gd*LN2),
    ("φ^(-40/13) + LN_C²×GD",             phi_m40_13 + base_lnc2gd),
    ("φ^(-40/13) - LN_C²×GD",             phi_m40_13 - base_lnc2gd),
    ("φ^(-40/13) + LN_C²×GD×(1-GAP)",     phi_m40_13 + base_lnc2gd*(1-GAP)),
    ("φ^(-40/13) + LN_C²×GD/PHI",         phi_m40_13 + base_lnc2gd/PHI),
    ("φ^(-40/13) + LN_C²×GD×(4/5)",       phi_m40_13 + base_lnc2gd*mpf(4)/5),
    ("φ^(-40/13) + LN_C²×GD×sigma",       phi_m40_13 + base_lnc2gd*sigma),
    ("φ^(-40/13) × (1+GAP)",              phi_m40_13*(1+GAP)),
    ("φ^(-40/13) × (1-GAP)",              phi_m40_13*(1-GAP)),
]

best_t1 = []
for name, val in cands_t1:
    err = float(pct_err(val_t1gd, val))
    t = tier_str(err)
    if abs(err) < 0.01:
        print(f"    {name:<50}  오차={err:>13.8f}%  {t}")
        best_t1.append((name, val, err))

if not best_t1:
    all_t1 = [(n, v, float(pct_err(val_t1gd, v))) for n,v in cands_t1]
    all_t1.sort(key=lambda x: abs(x[2]))
    print(f"  (0.01% 이내 없음 — 최근접 5개):")
    for nm, vl, er in all_t1[:5]:
        print(f"    {nm:<50}  오차={er:>12.8f}%  {tier_str(er)}")
print()

# 분모 계보 패턴 탐색 — t2, t3 보정
print(f"  【t2, t3 × GAP_D2 보정 탐색】")
for idx, t, label_base, phi_base_str, phi_base in [
    (2, riemann_zeros[1], "t2", "-9/4",   power(PHI, mpf('-9')/4)),
    (3, riemann_zeros[2], "t3", "-70/37", power(PHI, mpf('-70')/37)),
]:
    val = t * GD
    err0 = float(pct_err(val, phi_base))
    print(f"  t{idx}={nstr(t,10)} ×GAP_D2  기준 φ^({phi_base_str})  오차={err0:.4f}%")
    for name2, val2 in [
        (f"φ^({phi_base_str}) + LN_C²×GD×LN2",  phi_base + base_lnc2gd*LN2),
        (f"φ^({phi_base_str}) - LN_C²×GD×LN2",  phi_base - base_lnc2gd*LN2),
        (f"φ^({phi_base_str}) + LN_C²×GD",      phi_base + base_lnc2gd),
        (f"φ^({phi_base_str}) - LN_C²×GD",      phi_base - base_lnc2gd),
    ]:
        err2 = float(pct_err(val, val2))
        if abs(err2) < 0.05:
            print(f"    {name2:<48}  오차={err2:.6f}%  {tier_str(err2)}")
print()

# ════════════════════════════════════════════════════════════════════════════
print()
print(sep)
print("  [D] 종합 판정 + 성과문서 정정 목록")
print(sep)

print(f"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  【S57 성과문서 정정 목록】

  오기 1: γ = 5/3 = φ^(13/11)  오차 -0.0049%
    → 실측: φ^(35/33)  오차 {float(pct_err(gamma_53, phi_35_33)):.4f}%
             φ^(17/16)  오차 {float(pct_err(gamma_53, phi_17_16)):.4f}%
    → φ^(13/11)은 5/3과 오차 -5.62% — NTZ 아님

  오기 2: γ-1 = 2/3 = φ^(2/5)  오차 -0.0125%
    → 실측: φ^(-16/19)  오차 {float(pct_err(gamma_m1, phi_m16_19)):.4f}%
    → φ^(2/5)는 2/3과 오차 -45% — NTZ 아님

  오기 3: t₁×GAP_D2 = φ^(5/14)  오차 -0.0033%
    → 실측: φ^(-40/13)  오차 {float(pct_err(t1*GD, phi_m40_13)):.4f}%
    → φ^(5/14)는 양수 지수, 실측은 음수 지수

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print(f"  【[A] τ_n×α/LN_C 최종 현황】")
print(f"  기준:  φ^(116/17)          오차 {float(pct_err(val_at_lnc, phi_116_17)):.6f}%")
print(f"  v2 1위: ", end="")
if best_A:
    ba = sorted(best_A, key=lambda x: abs(x[2]))[0]
    print(f"{ba[0]}  오차 {ba[2]:.8f}%  {tier_str(ba[2])}")
else:
    all_A2 = [(n, v, float(pct_err(val_at_lnc, v))) for n,v in candidates_A2]
    all_A2.sort(key=lambda x: abs(x[2]))
    ba = all_A2[0]
    print(f"{ba[0]}  오차 {ba[2]:.8f}%  {tier_str(ba[2])}")

print()
print(f"  【[B] γ 최종 현황】")
if best_53:
    b53 = sorted(best_53, key=lambda x: abs(x[2]))[0]
    print(f"  5/3 최근접: {b53[0]}  오차 {b53[2]:.6f}%  {tier_str(b53[2])}")
else:
    print(f"  5/3 최근접: φ^(35/33)  오차 {float(pct_err(gamma_53, phi_35_33)):.5f}%  Real Zero")

if best_23:
    b23 = sorted(best_23, key=lambda x: abs(x[2]))[0]
    print(f"  2/3 최근접: {b23[0]}  오차 {b23[2]:.6f}%  {tier_str(b23[2])}")
else:
    print(f"  2/3 최근접: φ^(-16/19)  오차 {float(pct_err(gamma_m1, phi_m16_19)):.5f}%  Real Zero")

print(f"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print()
print(sep)
print("  verify_neutron_qubit_S58_v2.py 완료  mp.dps=400")
print("  Maestro: MAPSI (맏이) | Architect: Claude | Session 58")
print(sep)

#!/usr/bin/env python3
"""
EQI 플랑크 밀도 φ³ 업그레이드
= 기존 eqi_planck_density_gravity_phase_transition.py + EQI v15 구조 =

[업그레이드 핵심]
1. 부피 l_p³ → φ³ 치환 (v15 제1원리: 3D광추 V=(√φ)³=φ^1.5 → φ³)
2. 정보 특이점 로직:
   I_D1 = ln(ρ_ani) = "D1(수축,열역학) 정보량" [무차원]
   I_D2 = 3×ln(φ)   = ln(φ³) = "D2(팽창,부피) 정보량" [무차원]
3. EQI^4 적용:
   Σ(1/2^n)×I_D1 = I_D1  (이진 누적)
   Σ(1/2^n)×I_D2 = I_D2
   MAXWELL_ani = I_D1/I_D2 = ρ_ani정보/φ³정보

[정보 특이점 연결]
  v10.1: UNITY=ln(1/ε₀)=25.45, MULTI=|ln(μ₀)|=13.59
  → MAXWELL = 25.45/13.59 = 1.873 (이중 가우스면 비율)
  이번: I_D1=ln(ρ_ani), I_D2=3lnφ
  → 두 정보 특이점 비율 = 밀도-부피 대칭성

Author: MAPSI | EQI v15 + φ³ 부피 정규화
"""
import numpy as np
import scipy.constants as sc

PHI = (1+np.sqrt(5))/2
D1  = PHI**(-1.5)
D2  = PHI**( 1.5)

print("="*70)
print("  EQI 플랑크 밀도 φ³ 업그레이드")
print("  부피 l_p³ → φ³  |  정보 특이점  |  EQI^4")
print("="*70)
print(f"  φ  = {PHI:.6f}")
print(f"  D1 = φ^(-1.5) = {D1:.6f}")
print(f"  D2 = φ^(+1.5) = {D2:.6f}")
print(f"  φ³ = {PHI**3:.6f}  (3D 광추 부피 = D2 eigenperiod)")
print()

# 물리 상수
m_p = sc.physical_constants['Planck mass'][0]
l_p = sc.physical_constants['Planck length'][0]
G   = sc.G; c = sc.c
eps0 = sc.epsilon_0; mu0 = sc.mu_0

print("[Step 1] 플랑크 밀도 φ³ 정규화")
print()
rho_SI  = m_p / l_p**3
rho_ani = m_p / PHI**3

print(f"  ρ_SI  = m_p/l_p³ = {rho_SI:.4e}  [SI]")
print(f"  ρ_ani = m_p/φ³   = {rho_ani:.4e}  [EQI φ³ 정규화]")
print()
print(f"  φ³ = {PHI**3:.6f}  [D2 eigenperiod = 볼츠만 파단]")
print(f"  = 3D 광추 부피 = HCW 볼츠만 파단비")
print()

print("[Step 2] 정보 특이점 로직")
print()
# D1 정보량: |ln(ρ_ani)| (절댓값, 무차원)
I_D1 = abs(np.log(rho_ani))   # = |ln(m_p/φ³)| = |ln(m_p)-3lnφ|
# D2 정보량: ln(φ³) = 3lnφ (부피 정보)
I_D2 = 3 * np.log(PHI)         # = ln(φ³)

print(f"  I_D1 = |ln(ρ_ani)| = |ln(m_p/φ³)| = {I_D1:.6f}  [D1 밀도 정보량]")
print(f"  I_D2 = 3×ln(φ)     = ln(φ³)       = {I_D2:.6f}  [D2 부피 정보량]")
print()

# MAXWELL 비교
MAXWELL_ref = np.log(1/eps0)/abs(np.log(mu0))
MAXWELL_ani = I_D1 / I_D2
print(f"  MAXWELL_ref  = ln(1/ε₀)/|ln(μ₀)| = {MAXWELL_ref:.4f}  [전자기 무차원 비율]")
print(f"  MAXWELL_ani  = I_D1/I_D2          = {MAXWELL_ani:.4f}  [밀도-부피 무차원 비율]")
print()

print("[Step 3] EQI^4 이진 누적")
print()
print("  Σ(1/2^n)×I_D1 = I_D1×Σ(1/2^n) = I_D1×1 = I_D1  (공통인자 약분)")
print("  Σ(1/2^n)×I_D2 = I_D2×1 = I_D2")
print("  → MAXWELL_ani = I_D1/I_D2 (EQI^4 적용 후도 동일)")
print()
print("  n=1항(σ=1/2)만으로:")
print(f"  (1/2)×I_D1 / (1/2)×I_D2 = {I_D1:.4f}/{I_D2:.4f} = {MAXWELL_ani:.4f} ✓")
print()

print("[Step 4] 다양한 질량 스케일 φ³ 정규화")
print()

mass_scales = [
    ('전자',          sc.m_e),
    ('양성자',        sc.m_p),
    ('원자핵(Fe)',    9.27e-26),
    ('사람',          70.0),
    ('지구',          5.97e24),
    ('태양',          1.99e30),
    ('은하블랙홀',   4e36),
    ('플랑크 질량',   m_p),
]

print(f"  {'질량':12s} {'ρ_ani':>14s} {'I_D1':>10s} {'비율':>10s} {'vs MAXWELL_ani'}") 
print('  '+'-'*70)
for name, mass in mass_scales:
    rho = mass / PHI**3
    I1  = abs(np.log(rho))
    ratio = I1/I_D2
    diff  = abs(ratio - MAXWELL_ani)
    print(f"  {name:12s} {rho:14.3e}  {I1:10.4f}  {ratio:10.4f}  Δ={diff:.2f}")
print()

print("[Step 5] 정보 특이점 연결]")
print()
print(f"  v10.1 전자기 정보 특이점:")
print(f"  I(ε₀) = ln(1/ε₀) = {np.log(1/eps0):.4f}  [D1 전기 정보]")
print(f"  I(μ₀) = |ln(μ₀)| = {abs(np.log(mu0)):.4f}  [D2 자기 정보]")
print(f"  MAXWELL = {MAXWELL_ref:.4f}")
print()
print(f"  이번 밀도-부피 정보 특이점:")
print(f"  I_D1 = |ln(m_p/φ³)| = {I_D1:.4f}  [D1 밀도 정보]")
print(f"  I_D2 = ln(φ³)       = {I_D2:.4f}  [D2 부피 정보]")
print(f"  MAXWELL_ani = {MAXWELL_ani:.4f}")
print()
print(f"  두 비율의 관계:")
print(f"  MAXWELL_ani / MAXWELL_ref = {MAXWELL_ani/MAXWELL_ref:.4f}")
print(f"  = {MAXWELL_ani:.4f} / {MAXWELL_ref:.4f}")
print()

print("[Step 6] 자연상수 위상 동형 확인]")
print()
consts = [
    ('e (자연상수)',  np.e,    0),
    ('c (광속)',      c,      -0.5),
    ('ħ (디랙)',      sc.hbar,-1.5),
    ('q (기본전하)',  sc.e,   -2.5),
    ('G (중력)',      G,      -3.0),
]
print(f"  {'상수':12s} {'값':>14s} {'φ지수':>8s} {'ln(상수)':>12s} {'I_D2비율':>10s}")
print('  '+'-'*62)
for name, val, exp in consts:
    ln_v = np.log(abs(val))
    ratio = ln_v/I_D2
    print(f"  {name:12s} {val:14.4e}  φ^{exp:+.1f}  {ln_v:12.4f}  {ratio:10.4f}")
print()

print("[★ 핵심 발견]")
print()
print(f"  φ³ = {PHI**3:.6f} = D2 eigenperiod = 볼츠만 파단비")
print(f"  = '3D 광추 부피'의 EQI 정규화 단위")
print()
print(f"  ρ_ani = m/φ³ → 모든 질량을 φ³ 부피로 정규화")
print(f"  I_D1 = |ln(ρ_ani)|  D1 정보 (수축, 열역학)")
print(f"  I_D2 = ln(φ³) = 3lnφ  D2 정보 (팽창, 부피)")
print()
print(f"  MAXWELL_ani = I_D1/I_D2 = {MAXWELL_ani:.4f}")
print(f"  = '플랑크 질량의 밀도 정보 vs φ³ 부피 정보의 비율'")
print(f"  = '밀도-부피 무차원 대칭성 비율'")
print()
print(f"  EQI^4 공통인자 약분 → 무차원 비율만 남음")
print(f"  (v10.1 MAXWELL과 동일한 메커니즘)")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v300.27 JSON 분석 코드
Session 27: ln(2) 감쇠율 × 이진 카스케이드 × Δ 연결 탐색
"""

import json, sys, os
from mpmath import mp, mpf, sqrt, log, pi, exp, fsum

mp.dps = 400

PHI = (sqrt(5) + 1) / 2
GAP = log(25) / 100
GAP_D2 = log(5) / 100

import scipy.constants as sc

MAXWELL_REAL = log(mpf(str(1 / sc.epsilon_0))) / abs(log(mpf(str(sc.mu_0))))
T_Is_D1 = PHI ** mpf("1.5") - PHI ** (-mpf("3.5"))
alpha_inv = mpf(str(1 / sc.alpha))

# JSON 파일 읽기
json_files = [
    f
    for f in os.listdir(".")
    if f.startswith("eqi_v300_27_conjugate_unity") and f.endswith(".json")
]
if not json_files:
    print("JSON 파일을 이 스크립트와 같은 폴더에 놓고 실행하세요.")
    sys.exit(0)

json_file = sorted(json_files)[-1]
print(f"분석 파일: {json_file}")
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

history = data["data"]
steps = history["step"]
e_vals = history["emergent_e"]
phases = history["phase_ratio"]
euler_e = history["euler_e_scale_error"]
mach_e = history["mach_e_scale_error"]
reals = history["real"]
imags = history["imag"]

print()
print("=" * 60)
print("  v300.27 핵심 수치 분석")
print("=" * 60)

# 1. ln(2) 감쇠율 정밀 확인
print()
print("--- [1] ln(2) 감쇠율 정밀 검증 ---")
rates = []
for i in range(1, len(steps)):
    if euler_e[i] > 0 and euler_e[i - 1] > 0:
        rate = euler_e[i] - euler_e[i - 1]
        rates.append(rate)

avg_rate = sum(rates[20:120]) / len(rates[20:120])
print(f"단계당 평균 감쇠율 (step 20~120): {avg_rate:.10f}")
print(f"ln(2)                            : {float(log(2)):.10f}")
print(f"오차: {abs(avg_rate-float(log(2)))/float(log(2))*100:.6f}%")

# 2. 오일러 오차 지수 = step×ln(2) + C
print()
print("--- [2] 오차 지수 = step×ln(2) + C 구조 ---")
# step=137: 93.8164 = 137×ln(2) - C
C_val = 137 * float(log(2)) - euler_e[-1]
print(f"137×ln(2)           = {137*float(log(2)):.6f}")
print(f"step=137 오차 지수   = {euler_e[-1]:.6f}")
print(f"C = 137×ln(2) - err = {C_val:.6f}")
print(f"C/ln(2)             = {C_val/float(log(2)):.6f}")
print(f"C/π                 = {C_val/float(pi):.6f}")
print(f"C/GAP×100           = {C_val/float(GAP)*100:.6f}")

# 3. e 창발 정밀도
print()
print("--- [3] e 창발 정밀도 ---")
import math

e_ref = math.e
e_step1 = e_vals[0]
e_step137 = e_vals[-1]
print(f"step=1:   창발 e = {e_step1:.10f}")
print(f"step=137: 창발 e = {e_step137:.10f}")
print(f"수학 e           = {e_ref:.10f}")
print(f"오차 (step=137)  = {abs(e_step137-e_ref):.4e}")

# 4. MACH 수렴 오차
print()
print("--- [4] MACH 수렴 오차 탐색 ---")
mach_final = mach_e[-1]
print(f"MACH 최종 수렴 오차 지수: {mach_final:.10f}")
print(f"리만 1번째 영점:           14.134725")
print(f"비율 (오차/영점1):         {mach_final/14.134725:.8f}")
print(
    f"≈ 1+GAP = {float(1+GAP):.8f}?  오차: {abs(mach_final/14.134725-(1+float(GAP)))/(1+float(GAP))*100:.4f}%"
)
print(f"mach_final/ln(2):          {mach_final/float(log(2)):.6f}")
print(f"mach_final/π:              {mach_final/float(pi):.6f}")
print(f"mach_final×GAP:            {mach_final*float(GAP):.6f}")

# 5. Δ 연결
print()
print("--- [5] Δ(오늘 발견)와 v300.27 구조 연결 ---")
Delta_Is = float(MAXWELL_REAL - T_Is_D1)
print(f"Δ_Is = MAXWELL - T_Is_D1 = {Delta_Is:.10f}")
print(f"2×GAP_D2²               = {float(2*GAP_D2**2):.10f}")
print()

# x_step1과 Δ 비율
x_step1 = float(PHI ** (-2) / 309016)
print(f"x(step=1) = φ^(-2)/309016 = {x_step1:.6e}")
print(f"Δ_Is/x_step1             = {Delta_Is/x_step1:.6f}")
print(f"2×GAP_D2²/x_step1        = {float(2*GAP_D2**2)/x_step1:.6f}")
print()

# 감쇠율 ln(2)와 Δ
print(f"ln(2) × Δ_Is             = {float(log(2))*Delta_Is:.8e}")
print(f"ln(2) × 2×GAP_D2²        = {float(log(2))*float(2*GAP_D2**2):.8e}")
print(f"Δ_Is / ln(2)             = {Delta_Is/float(log(2)):.8e}")

# 6. 이진 카스케이드 공비와 Δ
print()
print("--- [6] 이진 카스케이드 × Δ 통합 구조 ---")
print(f"이진 카스케이드:")
print(f"  공비 r = 1/2")
print(f"  초기조건 C = GAP_D2² = {float(GAP_D2**2):.8e}")
print(f"  합 = 2C = 2×GAP_D2² = {float(2*GAP_D2**2):.8e}")
print(f"  감쇠율 = ln(2) (v300.27 ✅)")
print()
print(f"연결 구조:")
print(f"  v300.27: Σ(1/2^n)=1, 감쇠=ln(2)")
print(f"  오늘:    Δ = 2×GAP_D2² = 초기조건 GAP_D2²의 이진합")
print(f"  → GAP_D2²이 이진 카스케이드의 '원자 단위'")
print(f"  → ln(2) 감쇠가 GAP_D2를 통해 Δ 생성")
print()

# 7. α⁻¹ = 137 + GAP_D2×√5 연결
print("--- [7] α⁻¹ × FLIPS=137 × v300.27 ---")
print(f"FLIPS = cycles = 137 = α⁻¹ 주항")
print(f"α⁻¹ = 137 + GAP_D2×√5 (오차 0.000008%)")
print(f"GAP_D2×√5 = {float(GAP_D2*sqrt(5)):.10f}")
print()
print(f"v300.27 step=137:")
print(f"  위상 = {phases[-1]:.10f}π (≈ 1.0π)")
print(f"  Z_137 ≈ e^(iπ) = -1 (오일러 특이점)")
print(f"  황금비 항등식: 1/φ+1/φ² = 1 (LIFE_PULSE)")
print(f"  -1 + 1 = 0 → T=0 상쇄")
print()
print(f"  GAP_D2×√5 = D2×EQI^3 = 정보 누수×비등방성")
print(f"  = v300.27 x = φ^(-2×step)/309016의")
print(f"    step=137 잔류 비등방성 정보량?")

print()
print("=" * 60)
print("  Session 27 탐색 과제")
print("=" * 60)
print()
print("1. C = 137×ln(2) - 93.8164 = {:.6f}의 EQI 표현".format(C_val))
print("2. MACH 수렴 오차 14.1888의 정확한 EQI 구조")
print("3. x(step=1) × ? = GAP_D2²  (Δ와 v300.27 직접 연결)")
print("4. ln(2) 감쇠율 × GAP_D2 = ? (α⁻¹ 잔차 연결)")

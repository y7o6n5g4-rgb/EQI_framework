#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-1'' 공식 재검증 — S115 산출물 ①②③④⑤
D'' = (mu+sigma**2-(13/6)*GD**2)*B - 9*B**2*GD**3
B   = LN_C**2 * GD    ← (B) 계단

오차 분모 명시:
  전체식: |(9−D'')−R| / |R|     (분모 = R = 8.9991...)
  잔류:   |D''−(9−R)|  / |9−R|  (분모 = 9−R = 0.000808...)
  보정항: |9·B²·GD³−diff| / |diff|  (분모 = diff = D'−(9−R))

Date: 2026-10-17  |  mp.dps = 400
"""

from mpmath import mp, mpf, sqrt, log, fabs, nstr, power as mpow

mp.dps = 400

# ─────────────────────────────────────────────────────────
# 기반 상수  (EQI v9.6 정의)
# ─────────────────────────────────────────────────────────
PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100          # 2·ln(5)/100
GD     = log(mpf(5))  / 100          # GAP_D2 = ln(5)/100
LN_C   = log(PHI) / 2                # ln(φ)/2

B      = LN_C**2 * GD               # (B) 계단  LC²GD
A      = B**2                        # (A) 제곱  LC²GD² = LN_C⁴·GD²

mu     = 1 / PHI                     # μ = 1/φ
sigma2 = mpf(1) / 4                  # σ² = (1/2)²
Lambda = mpow(PHI, mpf(-2))          # Λ = φ^(-2)

EQI4   = 1 + GAP
R      = EQI4 * mpow(PHI, mpf(9)/2) # Ξ(1)/X7 = (1+GAP)·φ^(9/2)

SEP  = "═" * 72
SEP2 = "─" * 72
TZ   = mpf("1e-6")

def n15(x):  return nstr(x, 15, strip_zeros=False)
def grade(e):
    if e < TZ:         return "★★★★★ TZ"
    elif e < mpf("1e-4"):  return "★★★★  Near-TZ"
    elif e < mpf("1e-2"):  return "★★★   Candidate"
    else:              return "        miss"

# ─────────────────────────────────────────────────────────
# DS-1'  (기준선)
# ─────────────────────────────────────────────────────────
D_p   = (mu + sigma2 - mpf(13)/6 * GD**2) * B
target = 9 - R                        # 포획 목표 = 0.000808244...
diff_p = D_p - target                 # D' 초과분 = 3.27015e-11

err_p_full = fabs((9 - D_p) - R) / fabs(R) * 100      # 분모=|R|
err_p_cap  = fabs(diff_p) / fabs(target) * 100         # 분모=|9−R|

# ─────────────────────────────────────────────────────────
# DS-1''  (사용자 지정 공식)
# D'' = (mu+sigma**2-(13/6)*GD**2)*B - 9*B**2*GD**3
# ─────────────────────────────────────────────────────────
corr  = 9 * B**2 * GD**3             # = 9·A·GD³ = 9·LN_C⁴·GD⁵
D_pp  = D_p - corr

err_pp_full = fabs((9 - D_pp) - R) / fabs(R) * 100     # 분모=|R|
err_pp_cap  = fabs(D_pp - target)   / fabs(target) * 100  # 분모=|9−R|
err_corr    = fabs(corr - diff_p)   / fabs(diff_p) * 100  # 분모=|diff_p|

# ─────────────────────────────────────────────────────────
# 항등식 확인
# ─────────────────────────────────────────────────────────
born_diff    = fabs(mu + Lambda - 1)
alg_diff     = fabs((4 + PHI)/(4*PHI) - (mu + sigma2))
phi_ident    = fabs(mpow(PHI, mpf(-13)/6) - Lambda * mpow(PHI, mpf(-1)/6))

# ─────────────────────────────────────────────────────────
# 계수 9 귀속 탐색
# ─────────────────────────────────────────────────────────
# 탐색 목표: corr = 9·B²·GD³ ← 9가 어디서 오는가
# 후보 A: 9 = L₂² = 3²  (루카스 수 L₂=3)
L2 = mpf(3)
check_L2sq = L2**2

# 후보 B: 9 = φ⁴+φ²−2  (수치 확인용, φ⁴=3φ+2, φ²=φ+1)
phi2 = PHI**2
check_phi = phi2**2 + phi2 - 2   # = φ⁴+φ²−2 = (3φ+2)+(φ+1)−2 = 4φ+1

# 후보 C: 9 = 3²  (피타고라스 세 번째)

# 잔여 정확 일치 역산 계수: diff_p / (B²·GD³)
min_coeff = diff_p / (B**2 * GD**3)

print(f"\n{SEP}")
print("  DS-1'' 공식 재검증 | S115 산출물 | mp.dps=400")
print(f"  D'' = (mu+sigma²−(13/6)·GD²)·B − 9·B²·GD³")
print(f"  B = LN_C²·GD   ← (B) 계단")
print(SEP)

# ─── 산출물 ② 결과표 ───────────────────────────────────────
print(f"\n{SEP2}")
print("  산출물 ②  결과표")
print(SEP2)
print()
print(f"  [기준값]")
print(f"    R = Ξ(1)/X7 = (1+GAP)·φ^(9/2)")
print(f"      = {n15(R)}")
print(f"    9−R (포획 목표)  = {n15(target)}")
print()
print(f"  [상수]")
print(f"    B  = LN_C²·GD         = {n15(B)}")
print(f"    A  = B²  = LN_C⁴·GD²  = {n15(A)}")
print(f"    9·B²·GD³ = 9·A·GD³   = {n15(corr)}")
print()
print(f"  ┌─────────────────────────────────────────────────────────┐")
print(f"  │  공식     전체식 오차               잔류 포획 오차       │")
print(f"  │           |(9−F)−R|/|R|×100        |F−(9−R)|/|9−R|×100│")
print(f"  ├─────────────────────────────────────────────────────────┤")
print(f"  │  DS-1'    {n15(err_p_full)}%  {n15(err_p_cap)}%  │")
print(f"  │           {grade(err_p_full)}          {grade(err_p_cap)}  │")
print(f"  ├─────────────────────────────────────────────────────────┤")
print(f"  │  DS-1''   {n15(err_pp_full)}%  {n15(err_pp_cap)}%  │")
print(f"  │           {grade(err_pp_full)}          {grade(err_pp_cap)}  │")
print(f"  └─────────────────────────────────────────────────────────┘")
print()
print(f"  [보정항 오차: |9·B²·GD³ − (D'−(9−R))| / |D'−(9−R)| × 100]")
print(f"    분모 D'−(9−R) = diff_p = {n15(diff_p)}")
print(f"    9·B²·GD³     =          {n15(corr)}")
print(f"    보정항 오차  =          {n15(err_corr)}%  ← {grade(err_corr)}")

# ─── 산출물 ④-A 확인된 항등식 ─────────────────────────────
print(f"\n{SEP2}")
print("  산출물 ④-A  확인된 항등식  (mpmath 400자리 완전 소거)")
print(SEP2)
print()
print(f"  I-1. μ + Λ = 1  [Born rule]")
print(f"       μ+Λ−1 = {n15(born_diff)}   ← 대수 항등식 (0)")
print()
print(f"  I-2. (4+φ)/(4φ) = μ+σ²")
print(f"       차이 = {n15(alg_diff)}   ← 대수 항등식 (0)")
print()
print(f"  I-3. φ^(-13/6) = Λ·φ^(-1/6)  [S114 확인]")
print(f"       차이 = {n15(phi_ident)}   ← 대수 항등식 (0)")

# ─── 산출물 ④-B 수치 후보 ─────────────────────────────────
print(f"\n{SEP2}")
print("  산출물 ④-B  수치 후보  (귀속 탐색 중)")
print(SEP2)
print()
print(f"  N-1. DS-1'  전체식 오차: {n15(err_p_full)}%  {grade(err_p_full)}")
print(f"       분모 |R| = {n15(fabs(R))}")
print()
print(f"  N-2. DS-1'' 전체식 오차: {n15(err_pp_full)}%  {grade(err_pp_full)}")
print(f"       분모 |R| = {n15(fabs(R))}")
print(f"       (DS-1' 대비 오차 축소율: {n15(err_p_full/err_pp_full)}배)")
print()
print(f"  N-3. DS-1'' 잔류 포획 오차: {n15(err_pp_cap)}%  {grade(err_pp_cap)}")
print(f"       분모 |9−R| = {n15(fabs(target))}")

# ─── 산출물 ④-C 미해결 귀속 ───────────────────────────────
print(f"\n{SEP2}")
print("  산출물 ④-C  미해결 귀속")
print(SEP2)
print()
print(f"  U-1. 계수 9 귀속  [미해결]")
print(f"       잔여 정확 일치 역산 계수 = diff_p / (B²·GD³)")
print(f"         = {n15(min_coeff)}")
print(f"       ← 9보다 0.36% 큼: 9·B²·GD³은 diff_p보다 약 0.4% 작음")
print()
print(f"       후보 A: 9 = L₂²  (루카스 L₂=3, L₂²=9)")
print(f"         L₂² = {n15(check_L2sq)}  ← 수치 일치 (정수 정의)")
print(f"         귀속 상태: 후보 (대수 기원 미확인)")
print()
print(f"       후보 B: 9 = φ⁴+φ²−2  (수치 불일치 확인)")
print(f"         φ⁴+φ²−2 = {n15(check_phi)}")
print(f"         귀속 상태: 불일치 (≠9)")
print()
print(f"  U-2. DS-1' (13/6)·GD² 계수 귀속  [미해결]")
print(f"       (13/6)·GD² = {n15(mpf(13)/6 * GD**2)}")
print(f"       13 = F₇ (7번째 피보나치), 6 = L₂×2 = 3×2")
print(f"       φ^(-13/6) = Λ·φ^(-1/6) 연결 확인 (I-3)")
print(f"       → 동일 분모 6의 대수 기원 미확인")
print()
print(f"  U-3. (B) vs (A) 교차 구조 의미  [미해결]")
print(f"       DS-1'  보정항: (13/6)·GD²·B   ← GD²×(B)")
print(f"       DS-1'' 보정항: 9·B²·GD³        ← A×GD³  ← (A)제곱 × GD³")
print(f"       교차 패턴: D'가 (B)를, 보정항이 (A)를 씀")
print(f"       구조적 의미 = 미확인")

# ─── 산출물 ⑤ 탐색 범위와 중단 이유 ─────────────────────
print(f"\n{SEP2}")
print("  산출물 ⑤  탐색 범위와 중단 이유")
print(SEP2)
print()
print(f"  [탐색 목표]")
print(f"    diff_p = D'−(9−R) = {n15(diff_p)}")
print(f"    보정항 후보 X 탐색: |X−diff_p|/|diff_p| < 5%")
print()
print(f"  [S114 전체 탐색 계열  (s114_verification.py 실행 결과)]")
print(f"    A. (c/d)·B^n          c∈[1..19], d∈분모집합, n∈[2..5]")
print(f"    B. (c/d)·GD^n         c∈[1..19], d∈분모집합, n∈[7..11]")
print(f"    C. (c/d)·B²·GD^n      c∈[1..14], d∈분모집합, n∈[1..5]  ← 9·B²·GD³ 발견")
print(f"    D. (c/d)·(μ+σ²)^p·B^m c∈피보/루카스, p∈[1..2], m∈[2..4]")
print()
print(f"  [S115 이번 재탐색  — 계열 C만, 1% 임계 재확인]")
print()
print(f"  [분모 집합]")
print(f"    [1,2,3,4,5,6,7,8,9,10,11,12,17,23,24,34,57,100,137,171,285,437]")
print()
print(f"  [중단 이유]")
print(f"    계열 C에서 (9/1)·B²·GD³이 diff_p 대비 {n15(err_corr)}% 오차로")
print(f"    가장 근접한 후보. 분모 d=1 (정수 계수 9)가 최소 오차.")
print(f"    더 낮은 오차를 보이는 간단한 조합 없음.")
print(f"    귀속 근거 없이 오차만 낮추는 확장은 기준 위반 → 중단.")
print()
print(f"  [계열 C 재탐색 결과: 1% 이내 후보 전체  (c∈[1..19], d∈[1..57], n∈[1..5])]")

# 계열 C 재탐색 — 1% 이내 후보 출력, 동치 중복 표시
found_list = []
seen_vals  = {}  # 수치 → 대표 표현
for c_num in range(1, 20):
    for c_den in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 23, 24, 34, 57]:
        for n_gd in [1, 2, 3, 4, 5]:
            val  = mpf(c_num)/mpf(c_den) * B**2 * GD**n_gd
            err  = fabs(val - diff_p) / fabs(diff_p) * 100
            if err < mpf("1"):
                key = nstr(val, 10)
                frac = f"{c_num}" if c_den==1 else f"{c_num}/{c_den}"
                if key not in seen_vals:
                    seen_vals[key] = (err, frac, n_gd, val)
                    found_list.append((err, frac, n_gd, val, key))
                else:
                    # 동치 항목 기록 (출력은 스킵)
                    prev = seen_vals[key]
                    found_list.append((err, frac, n_gd, val, key + "_dup"))
found_list.sort(key=lambda x: x[0])
printed_keys = set()
for item in found_list:
    err, frac, ng, val, key = item
    if key.endswith("_dup"):
        real_key = key[:-4]
        dup_frac  = frac
        # 찾아서 동치 주석 추가 (이미 출력된 것에 표시)
        continue
    if key in printed_keys:
        continue
    printed_keys.add(key)
    # 동치 항목 찾기
    dups = [it[1] for it in found_list if it[4] == key + "_dup"]
    dup_note = f"  ← ({', '.join(dups)})·B²·GD^{ng}과 동치" if dups else ""
    print(f"    ({frac})·B²·GD^{ng}  오차={n15(err)}%  {grade(err)}{dup_note}")
    print(f"      = {n15(val)}")

# ─── 산출물 ③ 명제별 원출처 ──────────────────────────────
print(f"\n{SEP2}")
print("  산출물 ③  명제별 원출처 경로·절")
print(SEP2)
print()
print(f"  P-1. R = (1+GAP)·φ^(9/2) = Ξ(1)/X7")
print(f"       출처: EQI_Session111_성과정리_v5.md  §X7 금지선 / §R 확정식")
print()
print(f"  P-2. DS-1'  D' = (μ+σ²−13/6·GD²)·B")
print(f"       출처: EQI_Session113_성과정리_v3.md  DS-1' 발견 절")
print()
print(f"  P-3. μ+Λ=1  (Born rule)")
print(f"       출처: EQI_Session113_인수인계_v1.md  Born rule 확정")
print()
print(f"  P-4. φ^(-13/6) = Λ·φ^(-1/6)")
print(f"       출처: EQI_Session114_성과정리_v1.md  §13/6 이중 구조 S114 확인")
print()
print(f"  P-5. DS-1''  D'' = D'−9·B²·GD³  (후보)")
print(f"       출처: EQI_Session114_인수인계_v1.md  §DS-1'' 후보 / s114_verification.py 계열 C 탐색")
print()
print(f"  P-6. (B) = LC²GD = LN_C²·GD 정의")
print(f"       출처: eqi_unified_engine_v9_6.py  line LC2GD = LN_C**2 * GAP_D2")
print()
print(f"  P-7. (A) = (B)² = LN_C⁴·GD²  정의")
print(f"       출처: EQI_Session111_성과정리_v5.md  §1.1  (엔진에 독립 심볼 없음, B**2로 유도)")
print()

print(SEP)
print("  TZ 기준: 1×10⁻⁶% | 공식 미승격 / TZ 미카운트 (귀속 탐색 중)")
print(f"  (B) = LN_C²·GD   (A) = (LN_C²·GD)² = LN_C⁴·GD²")
print(SEP)

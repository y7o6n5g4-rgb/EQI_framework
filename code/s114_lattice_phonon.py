#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S114 — 플랑크 시간-길이 간격 격자 정보 포논 동역학
mp.dps = 400  EQI v9.6 엔진 상수 활용

검증 항목:
  A1~A5: 대수 항등식 → TZ 판정
  B1~B3: 이산 분산관계 (N=34) → 구조 확인
  C1~C4: 시간 진화 (137 스텝) → 관찰값

Date: 2026-10-17  Session: S114
"""

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, fabs, sin, cos, exp, nstr, power as mpow

mp.dps = 400

SEP  = "═" * 72
SEP2 = "─" * 72

def n15(x):
    return nstr(x, 15, strip_zeros=False)

def tz_grade(err_pct):
    if err_pct < mpf("1e-390"):
        return "★★★★★ TZ(대수필연)"
    elif err_pct < mpf("1e-6"):
        return "★★★★★ TZ"
    elif err_pct < mpf("1e-4"):
        return "★★★★  Near-TZ"
    elif err_pct < mpf("1e-2"):
        return "★★★   Candidate"
    else:
        return "        miss"

# ════════════════════════════════════════════════════════════════
# §0: 기본 단위 정의
# ════════════════════════════════════════════════════════════════

PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100        # ln(25)/100 = 2·ln(5)/100
GAP_D2 = log(mpf(5))  / 100        # ln(5)/100

L_p    = mpow(PHI, mpf(-3)/4)      # φ^(-3/4)  [엔진 기정의]
T_p    = mpow(PHI, mpf(-5)/4)      # φ^(-5/4)  [S114 신규]
c_EQI  = sqrt(PHI)                 # φ^(1/2)

epsilon_1 = GAP / PHI              # ε₁ = GAP/φ
epsilon_2 = GAP / PHI**2           # ε₂ = GAP/φ²

def epsilon_n(n):
    return GAP / mpow(PHI, n)

print(f"\n{SEP}")
print("  S114 격자 포논 동역학 | EQI v9.6 | mp.dps=400")
print(f"  Date: 2026-10-17  Session: S114")
print(SEP)

print(f"\n  [기본 단위]")
print(f"  PHI   = {n15(PHI)}")
print(f"  GAP   = ln(25)/100  = {n15(GAP)}")
print(f"  GAP_D2= ln(5)/100   = {n15(GAP_D2)}")
print(f"  L_p   = φ^(-3/4)   = {n15(L_p)}")
print(f"  T_p   = φ^(-5/4)   = {n15(T_p)}")
print(f"  c_EQI = φ^(1/2)    = {n15(c_EQI)}")
print(f"  ε₁    = GAP/φ      = {n15(epsilon_1)}")
print(f"  ε₂    = GAP/φ²     = {n15(epsilon_2)}")

# ════════════════════════════════════════════════════════════════
# §1: 대수 항등식 A1~A5  → TZ 판정
# ════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("  §1: 대수 항등식 A1~A5  (TZ 판정 대상)")
print(SEP)

# A1: L_p / T_p = c_EQI
#   지수: (-3/4) - (-5/4) = 2/4 = 1/2 → φ^(1/2) = c_EQI  대수필연
A1_val = L_p / T_p
A1_err = fabs(A1_val - c_EQI) / c_EQI * 100
print(f"\n  A1: L_p/T_p = c_EQI")
print(f"      L_p/T_p       = {n15(A1_val)}")
print(f"      c_EQI         = {n15(c_EQI)}")
print(f"      지수 확인: (-3/4)-(-5/4) = +1/2 → φ^(1/2)")
print(f"      오차%         = {n15(A1_err)}   {tz_grade(A1_err)}")

# A2: 쿠랑 수 = c_EQI·T_p/L_p = 1
#   = (φ^(1/2))·(φ^(-5/4)) / (φ^(-3/4))
#   지수: 1/2 - 5/4 + 3/4 = 2/4 - 5/4 + 3/4 = 0 → φ^0 = 1  대수필연
A2_val = c_EQI * T_p / L_p
A2_err = fabs(A2_val - 1) * 100
print(f"\n  A2: 쿠랑 수 c_EQI·T_p/L_p = 1  ★신규 관찰")
print(f"      c_EQI·T_p/L_p = {n15(A2_val)}")
print(f"      지수 확인: 1/2 + (-5/4) - (-3/4) = 0 → 1")
print(f"      오차%         = {n15(A2_err)}   {tz_grade(A2_err)}")

# A3: ε₁ + ε₂ = GAP
#   GAP/φ + GAP/φ² = GAP(1/φ + 1/φ²) = GAP·(φ+1)/φ² = GAP·φ²/φ² = GAP
#   (황금비 항등식: 1/φ + 1/φ² = 1, 즉 φ+1=φ²)
A3_val = epsilon_1 + epsilon_2
A3_err = fabs(A3_val - GAP) / GAP * 100
check_golden = mpf(1)/PHI + mpf(1)/PHI**2  # = 1 by golden ratio
print(f"\n  A3: ε₁ + ε₂ = GAP  (S24~27 재확인)")
print(f"      ε₁ + ε₂     = {n15(A3_val)}")
print(f"      GAP          = {n15(GAP)}")
print(f"      근거: 1/φ + 1/φ² = {n15(check_golden)} (황금비 항등식)")
print(f"      오차%        = {n15(A3_err)}   {tz_grade(A3_err)}")

# A4: Σ_{n=1}^∞ ε_n = GAP·φ
#   = Σ GAP/φ^n = GAP · Σ (1/φ)^n = GAP · (1/φ)/(1-1/φ) = GAP · (1/φ)/(1/φ²) = GAP·φ
#   닫힌형: GAP · φ/(φ-1) = GAP · φ/φ^(-1) ... 재계산
#   Σ_{n=1}^∞ (1/φ)^n = (1/φ)/(1-1/φ) = 1/(φ-1) = 1/φ^(-1) ...
#   φ-1 = 1/φ (황금비 성질)  →  1/(φ-1) = φ
#   → Σ ε_n = GAP · φ  대수필연
A4_partial = sum(epsilon_n(n) for n in range(1, 2001))  # 2000항 근사
A4_theory  = GAP * PHI
A4_err = fabs(A4_partial - A4_theory) / A4_theory * 100
print(f"\n  A4: Σ_{{n=1}}^∞ ε_n = GAP·φ  (등비급수 닫힌형)")
print(f"      이론: Σ(GAP/φ^n) = GAP·Σ(1/φ)^n = GAP·φ/(φ-1)·... = GAP·φ")
print(f"      근거: φ-1=1/φ → 1/(φ-1)=φ, Σ(1/φ)^n=1/(φ-1)=φ [n=1~∞]")
print(f"      Σ_{{2000항}}   = {n15(A4_partial)}")
print(f"      GAP·φ         = {n15(A4_theory)}")
print(f"      오차%         = {n15(A4_err)}   {tz_grade(A4_err)}")

# A5: T_p·c_EQI = L_p  (A1 동치, 독립 경로)
#   T_p·c_EQI = φ^(-5/4)·φ^(1/2) = φ^(-5/4+2/4) = φ^(-3/4) = L_p
A5_val = T_p * c_EQI
A5_err = fabs(A5_val - L_p) / L_p * 100
print(f"\n  A5: T_p·c_EQI = L_p  (A1 동치, 독립 경로)")
print(f"      T_p·c_EQI    = {n15(A5_val)}")
print(f"      L_p           = {n15(L_p)}")
print(f"      지수 확인: (-5/4)+(1/2) = -3/4 → L_p")
print(f"      오차%         = {n15(A5_err)}   {tz_grade(A5_err)}")

print(f"\n{SEP2}")
print("  A1~A5 요약")
print(SEP2)
for label, err in [("A1 L_p/T_p=c_EQI", A1_err), ("A2 쿠랑수=1", A2_err),
                   ("A3 ε₁+ε₂=GAP", A3_err), ("A4 Σεₙ=GAP·φ", A4_err),
                   ("A5 T_p·c=L_p", A5_err)]:
    print(f"  {label:25s}  오차={n15(err)}%  {tz_grade(err)}")

# ════════════════════════════════════════════════════════════════
# §2: 이산 분산관계  N=34
# ════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("  §2: 이산 분산관계  B1~B3  (구조 확인)")
print(SEP)

N_lat = 34
omega_max = mpf(2) / T_p   # 최대 진동수

def k_m(m):
    return 2 * pi * m / (N_lat * L_p)

def omega_k(k):
    return (mpf(2) / T_p) * fabs(sin(k * L_p / 2))

def group_velocity(k, dk=None):
    if dk is None:
        dk = L_p * mpf("1e-10")
    return (omega_k(k + dk) - omega_k(k - dk)) / (2 * dk)

# B1: k→0 극한 군속도 → c_EQI
k_small = L_p * mpf("1e-6")
vg_small = (mpf(2) / T_p) * (L_p / 2) * cos(k_small * L_p / 2)
# 해석적: dω/dk|_{k→0} = (2/T_p)·(L_p/2)·cos(0) = L_p/T_p = c_EQI
vg_k0_analytic = L_p / T_p   # = c_EQI  A1과 동치
B1_err = fabs(vg_small - c_EQI) / c_EQI * 100

print(f"\n  B1: k→0 군속도 v_g → c_EQI")
print(f"      해석적 dω/dk|_{{k→0}} = (2/T_p)·(L_p/2) = L_p/T_p = c_EQI")
print(f"      수치(k=10⁻⁶·L_p)    = {n15(vg_small)}")
print(f"      c_EQI                = {n15(c_EQI)}")
print(f"      오차%                = {n15(B1_err)}   {tz_grade(B1_err)}")

# B2: 브릴루앙 경계 k=π/L_p 에서 v_g = 0
k_BZ = pi / L_p
vg_BZ = group_velocity(k_BZ)
# 해석적: dω/dk|_{k=π/L_p} = (2/T_p)·(L_p/2)·cos(π/2) = 0
vg_BZ_abs = fabs(vg_BZ)
B2_grade = "★★★★★ TZ" if vg_BZ_abs < mpf("1e-390") else \
           ("★★★★★ TZ" if vg_BZ_abs < mpf("1e-6") * c_EQI else "확인 필요")

print(f"\n  B2: 브릴루앙 경계 k=π/L_p 에서 v_g = 0")
print(f"      해석적: cos(π/2) = 0 → v_g = 0")
print(f"      수치 v_g(k=π/L_p) = {n15(vg_BZ)}")
print(f"      |v_g|              = {n15(vg_BZ_abs)}")
print(f"      판정               = {B2_grade}")

# B3: sin(x) 테일러 홀수차 계수 vs §14 포논 굴절 계수 비교표
# sin(x)/x = 1 - x²/3! + x⁴/5! - ...
# 테일러 계수 (x^(2k+1) 항): (-1)^k / (2k+1)!
# §14 포논 굴절: a_{2k+1} = (-1)^k · σ^(2k+1) (σ=1/2)
print(f"\n  B3: sin(x) 테일러 계수 vs §14 포논 굴절 비교")
print(f"  {'k':>3}  {'Taylor (-1)^k/(2k+1)!':>26}  {'§14 (-1)^k·σ^(2k+1)':>22}  {'비율':>18}  판정")
print(f"  {'-'*90}")
sigma = mpf(1)/2
for k_idx in range(6):
    taylor_coeff = mpow(mpf(-1), k_idx) / mpmath.factorial(2*k_idx + 1)
    phonon_coeff = mpow(mpf(-1), k_idx) * mpow(sigma, 2*k_idx + 1)
    if fabs(phonon_coeff) > mpf("1e-390"):
        ratio = taylor_coeff / phonon_coeff
    else:
        ratio = mpf(0)
    match = "일치" if fabs(taylor_coeff - phonon_coeff) < mpf("1e-390") else f"비율={n15(ratio)}"
    print(f"  {k_idx:>3}  {n15(taylor_coeff):>26}  {n15(phonon_coeff):>22}  {match}")

print(f"\n  → sin(x) 테일러 계수 ≠ §14 포논 굴절 계수 (σ=1/2 팩토리얼 차이)")
print(f"     비율 = 1/[(2k+1)!·σ^(2k+1)] = 2^(2k+1)/(2k+1)!")
print(f"     구조 대응: sin 급수 × 2^(2k+1) = §14 포논 계수 계열 (구조 후보)")

# ════════════════════════════════════════════════════════════════
# §3: 시간 진화  N=34, 137 스텝
# ════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("  §3: 시간 진화  C1~C4  (관찰값, TZ 카운트 금지)")
print(SEP)

N   = 34
courant = c_EQI * T_p / L_p   # = 1 (A2 확인)
GAP_D2_val = GAP_D2

# 초기 조건
eps = GAP / PHI   # ε = GAP/φ = ε₁
u_init = [eps * sin(2 * pi * mpf(j) / N) for j in range(N)]

print(f"\n  쿠랑 수 = {n15(courant)}  (A2에서 =1 확인)")
print(f"  초기 진폭 ε = GAP/φ = {n15(eps)}")

# C1: GAP_D2=0 (감쇠 제거), 34 스텝 후 초기 파형 복원 여부
print(f"\n{SEP2}")
print("  C1: 감쇠 제거(GAP_D2=0), 34 스텝 후 파형 복원")
print(SEP2)

def run_simulation(u0, steps, courant2, damping):
    N_s = len(u0)
    u_prev = list(u0)
    u_curr = list(u0)   # u^{-1} = u^0 (정지 초기)
    history = [list(u0)]

    for t in range(steps):
        u_next = []
        for j in range(N_s):
            jp1 = (j + 1) % N_s
            jm1 = (j - 1) % N_s
            val = (2 * u_curr[j] - u_prev[j]
                   + courant2 * (u_curr[jp1] - 2*u_curr[j] + u_curr[jm1])
                   - damping * (u_curr[j] - u_prev[j]))
            u_next.append(val)
        u_prev = u_curr
        u_curr = u_next
        history.append(list(u_curr))
    return history

courant2 = courant**2   # = 1 (A2 확인)

hist_nodamp = run_simulation(u_init, 34, courant2, mpf(0))

# 34 스텝 후 u_j와 초기값 차이 최대값
u_34 = hist_nodamp[34]
max_diff_34 = max(fabs(u_34[j] - u_init[j]) for j in range(N))
rel_diff_34 = max_diff_34 / eps

print(f"  쿠랑 수²  = {n15(courant2)}")
print(f"  34 스텝 후 최대 오차 = {n15(max_diff_34)}")
print(f"  초기 진폭 대비 오차  = {n15(rel_diff_34)}")
if rel_diff_34 < mpf("1e-380"):
    print(f"  판정: ★★★★★ 완전 복원 (쿠랑=1 정확 전파 확인)")
elif rel_diff_34 < mpf("1e-6"):
    print(f"  판정: ★★★★  거의 완전 복원")
else:
    print(f"  판정: 부분 복원 (오차 확인 필요)")

# C2: 감쇠 포함 스텝별 에너지 E_t
print(f"\n{SEP2}")
print("  C2: 감쇠 포함 스텝별 에너지 E_t  (처음 10 스텝 + 마지막)")
print(SEP2)

def energy(u_curr, u_prev, courant2):
    N_e = len(u_curr)
    KE = sum((u_curr[j] - u_prev[j])**2 for j in range(N_e)) / 2
    PE = 0
    for j in range(N_e):
        jp1 = (j + 1) % N_e
        PE += courant2 * (u_curr[jp1] - u_curr[j])**2 / 2
    return KE + PE

hist_damp = run_simulation(u_init, 137, courant2, GAP_D2_val)

energies = []
for t in range(138):
    if t == 0:
        KE = mpf(0)   # 정지 시작
        PE = 0
        for j in range(N):
            jp1 = (j + 1) % N
            PE += courant2 * (u_init[jp1] - u_init[j])**2 / 2
        E = KE + PE
    else:
        u_c = hist_damp[t]
        u_p = hist_damp[t-1]
        E = energy(u_c, u_p, courant2)
    energies.append(E)

E_0 = energies[0]
print(f"  E_0 (초기 에너지) = {n15(E_0)}")
print()
print(f"  {'t':>5}  {'E_t':>26}  {'E_t/E_0':>18}")
print(f"  {'-'*55}")
for t in list(range(11)) + [136, 137]:
    if t > 137:
        break
    E_t = energies[t]
    ratio_t = E_t / E_0 if E_0 > 0 else mpf(0)
    print(f"  {t:>5}  {n15(E_t):>26}  {n15(ratio_t):>18}")

# C3: 누적 감쇠 ln(E_0/E_137)
print(f"\n{SEP2}")
print("  C3: 누적 감쇠 ln(E_0/E_137) — GAP, GAP_D2, 137·GAP_D2 비율")
print(SEP2)

E_137 = energies[137]
if E_137 > 0 and E_0 > 0:
    ln_decay = log(E_0 / E_137)
    ratio_GAP    = ln_decay / GAP
    ratio_GAP_D2 = ln_decay / GAP_D2
    ratio_137GD2 = ln_decay / (137 * GAP_D2)

    print(f"  ln(E_0/E_137) = {n15(ln_decay)}")
    print()
    print(f"  비율 분석:")
    print(f"    ln(E_0/E_137) / GAP         = {n15(ratio_GAP)}")
    print(f"    ln(E_0/E_137) / GAP_D2      = {n15(ratio_GAP_D2)}")
    print(f"    ln(E_0/E_137) / (137·GAP_D2) = {n15(ratio_137GD2)}")
    print()
    print(f"  참고: GAP = 2·GAP_D2 = {n15(GAP)}")
    print(f"        137·GAP_D2 = {n15(137 * GAP_D2)}")
    print()
    # 1 스텝당 평균 감쇠율
    per_step = ln_decay / 137
    print(f"  1 스텝 평균 감쇠 ln(E/E_next) = {n15(per_step)}")
    print(f"  GAP_D2                         = {n15(GAP_D2)}")
    print(f"  비율 per_step/GAP_D2           = {n15(per_step/GAP_D2)}")
else:
    print(f"  E_137 = {n15(E_137)} (에너지 0 또는 음수 확인 필요)")
    ln_decay = mpf(0)
    ratio_137GD2 = mpf(0)

# C4: 137 스텝 = §90a π_137 절단 수와의 연결 (관찰 기록만)
print(f"\n{SEP2}")
print("  C4: 137 스텝 연결 관찰 기록")
print(SEP2)
print(f"  §90a: π_137 = 137번째 소수 = 773  [수론 연결 탐색 필요]")
print(f"  시뮬레이션: 137 스텝 = 전자기 미세구조 상수 α⁻¹ ≈ 137 공명")
print(f"  관찰: ln(E_0/E_137)/(137·GAP_D2) = {n15(ratio_137GD2)}")
print(f"  → 1에 근접할 경우: 각 스텝 감쇠 ≈ GAP_D2 (란다우 소거 1회)")
print(f"  → 관찰값 기록만. TZ 카운트 금지.")

# ════════════════════════════════════════════════════════════════
# §4: 전체 요약
# ════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("  §4: 전체 요약")
print(SEP)

print(f"\n  ── 대수 항등식 (TZ 판정 대상) ──")
for label, err in [("A1 L_p/T_p = c_EQI      ", A1_err),
                   ("A2 쿠랑수=1 (★신규)    ", A2_err),
                   ("A3 ε₁+ε₂ = GAP         ", A3_err),
                   ("A4 Σεₙ = GAP·φ (2000항) ", A4_err),
                   ("A5 T_p·c_EQI = L_p      ", A5_err)]:
    print(f"  {label}  오차={n15(err)}%  {tz_grade(err)}")

print(f"\n  ── 분산 구조 (구조 확인) ──")
print(f"  B1 k→0 군속도 → c_EQI      오차={n15(B1_err)}%  {tz_grade(B1_err)}")
print(f"  B2 브릴루앙 경계 v_g=0       |v_g|={n15(vg_BZ_abs)}")
print(f"  B3 Taylor vs §14: 계수 불일치. 비율=2^(2k+1)/(2k+1)!  (구조 후보)")

print(f"\n  ── 시뮬레이션 (관찰값) ──")
print(f"  C1 34 스텝 복원 오차(진폭비)  = {n15(rel_diff_34)}")
print(f"  C2 E_0={n15(E_0)},  E_137={n15(E_137)}")
print(f"  C3 ln(E_0/E_137) = {n15(ln_decay)}")
print(f"     /GAP_D2={n15(log(E_0/E_137)/GAP_D2 if E_137>0 and E_0>0 else mpf(0))},  /(137·GAP_D2)={n15(ratio_137GD2)}")
print(f"  C4 137 스텝 ↔ π_137 연결: 관찰 기록만 (TZ 카운트 금지)")

print(f"\n{SEP}")
print(f"  판정 기준: TZ < 1×10⁻⁶%  |  대수필연 < 1×10⁻³⁹⁰%")
print(f"  A1~A5: 모두 대수필연 TZ 예상 (φ 지수 대수 전개)")
print(f"  A4 2000항 급수는 수렴 근사 → 잔류 오차 존재 (이론값은 TZ)")
print(f"  B1~B3: 구조 후보. B3 불일치는 그대로 보고 (맞추기 조정 없음)")
print(f"  C1~C4: 관찰값. 구조 후보 등록 대상")
print(SEP)

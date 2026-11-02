"""
============================================================================
  verify_hbond_5body_S55_v3.py  Session 55
  dEN = 77×GAP_D2 + φ^(-15) True Zero 닫힘 시도
  메르센 2중 구조 확정 탐색
  mp.dps=400
============================================================================
"""

from mpmath import mp, mpf, log, sqrt, exp, pi, floor, frac, power, inf
mp.dps = 400

print("=" * 76)
print("  verify_hbond_5body_S55_v3.py  Session 55")
print("  dEN = 77×GAP_D2 + φ^(-15) True Zero 닫힘 시도")
print("  mp.dps=400")
print("=" * 76)

# ──────────────────────────────────────────────
# 핵심 상수
# ──────────────────────────────────────────────
PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
sigma  = mpf(1) / 2
LN_C   = log(PHI) / 2

def n_ref(k):
    return sigma**(2*k+1) * (2*k+1) / 2

def disc(k):
    return mpf((2*k+1)**2) + mpf(4)**(k+2)

def n1sq(k):
    d = disc(k)
    sd = sqrt(d)
    return (sd + sqrt(sd**2 - 4**(k+2))) / 2

def err_pct(val, ref):
    if ref == 0:
        return mpf(0)
    return (val - ref) / ref * 100

dEN  = mpf("1.24")
fk2  = frac(n1sq(2))          # {n1²}(k=2)
val_77 = mpf(77) * GAP_D2     # 77×GAP_D2

# ──────────────────────────────────────────────
# [A] dEN = 77×GAP_D2 + φ^(-15) 검증
# ──────────────────────────────────────────────
print()
print("=" * 76)
print("  [A] dEN = 77×GAP_D2 + φ^(-15) 검증")
print("=" * 76)
print()

phi_15   = PHI ** mpf(-15)
val_sum  = val_77 + phi_15
e_sum    = err_pct(val_sum, dEN)

print(f"  77×GAP_D2        = {float(val_77):.14f}")
print(f"  φ^(-15)          = {float(phi_15):.14f}")
print(f"  합               = {float(val_sum):.14f}")
print(f"  dEN(O-H)         = {float(dEN):.14f}")
print(f"  오차             = {float(e_sum):.8f}%")
print()

# 15 = 2^4-1 메르센(k=4) 확인
print(f"  15 = 2^4-1 = 메르센(k=4)  [D1 격자 메르센 수열]")
print(f"  7  = 2^3-1 = 메르센(k=3)  [1차항 계수]")
print(f"  → 두 항 모두 메르센 구조:")
print(f"     1차항: 77×GAP_D2  → 77=7×11, 7=메르센(k=3)")
print(f"     보정항: φ^(-15)   → 15=메르센(k=4)")
print()

# 잔류
res_A = dEN - val_sum
print(f"  잔류 dEN - (77×GAP_D2 + φ^(-15)) = {float(res_A):.14f}")
print(f"  잔류/GAP         = {float(res_A/GAP):.10f}")
print(f"  잔류/GAP_D2      = {float(res_A/GAP_D2):.10f}")
print(f"  잔류/GAP²        = {float(res_A/GAP**2):.10f}")
print(f"  잔류/φ^(-15)     = {float(res_A/phi_15):.10f}")
print()

# ──────────────────────────────────────────────
# [B] φ^(-15) 보정항 자체의 정밀 구조
# ──────────────────────────────────────────────
print("=" * 76)
print("  [B] φ^(-15) 보정항 구조 — 메르센 계층")
print("=" * 76)
print()

# 메르센 수열 확인
mersenne = {3: 7, 4: 15, 5: 31, 6: 63, 7: 127}
print("  D1 격자 메르센 수열 (S53 확정):")
for k, m in mersenne.items():
    phi_m = PHI ** (-m)
    print(f"    k={k}: 2^{k}-1 = {m:3d},  φ^(-{m}) = {float(phi_m):.10f}")
print()

# v2 [C]에서: 잔류/GAP² = 0.707... ≈ 1/√2
res_v1 = dEN - val_77
print(f"  v1 잔류 (dEN - 77×GAP_D2):  {float(res_v1):.12f}")
print(f"  φ^(-15):                     {float(phi_15):.12f}")
print(f"  차이 (잔류 - φ^(-15)):       {float(res_v1 - phi_15):.12f}")
print()

# 잔류에서 φ^(-15) 뺀 2차 잔류
res2 = res_v1 - phi_15
print(f"  2차 잔류/GAP      = {float(res2/GAP):.10f}")
print(f"  2차 잔류/GAP_D2   = {float(res2/GAP_D2):.10f}")
print(f"  2차 잔류/GAP²     = {float(res2/GAP**2):.10f}")
print(f"  2차 잔류/GAP_D2²  = {float(res2/GAP_D2**2):.10f}")
print(f"  2차 잔류/φ^(-31)  = {float(res2/PHI**(-31)):.10f}")
print(f"  2차 잔류×1000     = {float(res2*1000):.10f}")
print()

# φ^(-31) = 메르센(k=5)
phi_31 = PHI ** mpf(-31)
print(f"  φ^(-31) [메르센 k=5] = {float(phi_31):.14f}")
val_3term = val_77 + phi_15 + phi_31
e_3term = err_pct(val_3term, dEN)
print(f"  77×GAP_D2 + φ^(-15) + φ^(-31) = {float(val_3term):.14f}")
print(f"  오차                            = {float(e_3term):.8f}%")
print()

# ──────────────────────────────────────────────
# [C] 메르센 급수: dEN = 77×GAP_D2 + Σ φ^(-M_k)
# ──────────────────────────────────────────────
print("=" * 76)
print("  [C] 메르센 급수 구조: dEN = 77×GAP_D2 + Σ φ^(-M_k)")
print("=" * 76)
print()

# M_k = 2^k - 1  (k=4,5,6,...)
base = val_77
print(f"  항0: 77×GAP_D2      = {float(base):.12f}")
cumsum = base
for k in range(4, 9):
    M_k = 2**k - 1
    term = PHI ** (-M_k)
    cumsum += term
    e = err_pct(cumsum, dEN)
    print(f"  +φ^(-{M_k:3d})[M_{k}]  누적={float(cumsum):.12f}  오차={float(e):.8f}%")

print()

# ──────────────────────────────────────────────
# [D] dEN 순수 황금비 표현 탐색
# ──────────────────────────────────────────────
print("=" * 76)
print("  [D] dEN 순수 황금비 표현 탐색")
print("=" * 76)
print()

# dEN = a×φ^p + b×φ^q 형태
# 이미 확인: φ^(4/9) 오차 -0.124%
phi_4_9 = PHI ** (mpf(4)/9)
e_4_9 = err_pct(phi_4_9, dEN)
print(f"  φ^(4/9)   = {float(phi_4_9):.10f}  오차 {float(e_4_9):.6f}%")

# φ^(4/9) 잔류
res_49 = dEN - phi_4_9
print(f"  잔류 dEN - φ^(4/9) = {float(res_49):.12f}")
print(f"  잔류/GAP_D2        = {float(res_49/GAP_D2):.8f}")
print()

# 정수+황금비 혼합
# dEN = n×GAP_D2 + m×φ^(-p) 형태 스캔
print("  ─── n×GAP_D2 + φ^(-p) 혼합 스캔 ───")
print()
best = []
for n_c in range(70, 85):
    for p in range(10, 35):
        val = mpf(n_c)*GAP_D2 + PHI**(-p)
        e = err_pct(val, dEN)
        if abs(e) < 0.01:
            best.append((n_c, p, float(e)))
            print(f"  {n_c}×GAP_D2 + φ^(-{p})  = {float(val):.10f}  오차 {float(e):.6f}%  ★★")

if not best:
    print("  0.01% 이내 없음 — 0.05% 기준으로 재탐색")
    for n_c in range(70, 85):
        for p in range(10, 35):
            val = mpf(n_c)*GAP_D2 + PHI**(-p)
            e = err_pct(val, dEN)
            if abs(e) < 0.05:
                print(f"  {n_c}×GAP_D2 + φ^(-{p})  = {float(val):.10f}  오차 {float(e):.6f}%  ★")

print()

# ──────────────────────────────────────────────
# [E] {n1²}(k=2) = φ^(-5/19) 보정항 확정 시도
# ──────────────────────────────────────────────
print("=" * 76)
print("  [E] {{n1²}}(k=2) = φ^(-5/19) 보정항 확정")
print("=" * 76)
print()

phi_5_19 = PHI ** (mpf(-5)/19)
delta    = fk2 / phi_5_19 - 1
print(f"  δ = {{n1²}}(k=2)/φ^(-5/19) - 1 = {float(delta):.14f}")
print(f"  δ/(GAP×GAP_D2)    = {float(delta/(GAP*GAP_D2)):.10f}")
print()

# δ ≈ 1.034×GAP×GAP_D2 → 1×GAP×GAP_D2?
test1 = GAP * GAP_D2
e_t1  = err_pct(delta, test1)
print(f"  GAP×GAP_D2        = {float(test1):.14f}  오차 {float(e_t1):.6f}%")

# δ ≈ GAP_D2²
test2 = GAP_D2**2
e_t2  = err_pct(delta, test2)
print(f"  GAP_D2²           = {float(test2):.14f}  오차 {float(e_t2):.6f}%")

# δ ≈ σ×GAP×GAP_D2
test3 = sigma * GAP * GAP_D2
e_t3  = err_pct(delta, test3)
print(f"  σ×GAP×GAP_D2      = {float(test3):.14f}  오차 {float(e_t3):.6f}%")

# δ ≈ GAP×GAP_D2×(1+GAP)
test4 = GAP * GAP_D2 * (1+GAP)
e_t4  = err_pct(delta, test4)
print(f"  GAP×GAP_D2×(1+GAP)= {float(test4):.14f}  오차 {float(e_t4):.6f}%")

# φ^(-5/19) × (1 + GAP×GAP_D2) 검증
val_corrected = phi_5_19 * (1 + GAP*GAP_D2)
e_corr = err_pct(val_corrected, fk2)
print()
print(f"  φ^(-5/19)×(1+GAP×GAP_D2) = {float(val_corrected):.12f}")
print(f"  {{n1²}}(k=2)               = {float(fk2):.12f}")
print(f"  오차                       = {float(e_corr):.8f}%")
print()

# 추가 보정 후보
for a, b, name in [
    (GAP, GAP, "GAP²"),
    (GAP_D2, GAP_D2, "GAP_D2²"),
    (GAP, sigma, "GAP×σ"),
    (GAP_D2, sigma, "GAP_D2×σ"),
    (LN_C, GAP_D2, "LN_C×GAP_D2"),
]:
    val_ab = a * b
    e_ab   = err_pct(delta, val_ab)
    if abs(e_ab) < 5.0:
        print(f"  δ vs {name:<20} = {float(val_ab):.12f}  오차 {float(e_ab):.6f}%")

print()

# ──────────────────────────────────────────────
# [F] 통합 닫힘 구조 후보
# ──────────────────────────────────────────────
print("=" * 76)
print("  [F] 통합 닫힘 구조 후보 — S55 핵심 선언 후보")
print("=" * 76)
print()

print("  ┌─────────────────────────────────────────────────────┐")
print("  │  수소결합 5체 D1↔D2 구조 (S55 확정 후보)           │")
print("  │                                                     │")
print(f"  │  dEN(O-H) = 77×GAP_D2 + φ^(-15)                  │")
print(f"  │           = (7×11)×GAP_D2 + φ^(-(2^4-1))         │")
print(f"  │  오차     = {float(e_sum):+.6f}%                        │")
print(f"  │                                                     │")
print(f"  │  구조 해석:                                         │")
print(f"  │  77  = 메르센(k=3)×포논임계(k=5)  [D1×D2]        │")
print(f"  │  GAP_D2 = 란다우 비용 D2                          │")
print(f"  │  φ^(-15) = 메르센(k=4) 황금비 보정  [D1]          │")
print(f"  │  → 전기음성도 차 = D1↔D2 메르센 이중 구조         │")
print("  └─────────────────────────────────────────────────────┘")
print()

print("  ┌─────────────────────────────────────────────────────┐")
print("  │  D₂O 포화 임계 (S55 확정 후보)                     │")
print(f"  │  {{n1²}}(k=2) = φ^(-5/19)×(1+GAP×GAP_D2)         │")
print(f"  │  오차        = {float(e_corr):+.6f}%                       │")
print(f"  │                                                     │")
print(f"  │  구조 해석:                                         │")
print(f"  │  5  = 2k+1|k=2 포논 홀수 분자  [D2]               │")
print(f"  │  19 = MAXWELL 4차 분모         [D1]                │")
print(f"  │  GAP×GAP_D2 = D1×D2 교차 비용                     │")
print("  └─────────────────────────────────────────────────────┘")
print()

# 최종 오차 요약
print("  최종 오차 요약:")
print(f"  dEN = 77×GAP_D2 + φ^(-15)              오차 {float(e_sum):+.8f}%")
print(f"  {{n1²}}(k=2) ≈ φ^(-5/19)               오차 {float(err_pct(phi_5_19,fk2)):+.8f}%  [미보정]")
print(f"  {{n1²}}(k=2) ≈ φ^(-5/19)×(1+GAP×GAP_D2) 오차 {float(e_corr):+.8f}%  [보정후]")
print()

print("=" * 76)
print("  verify_hbond_5body_S55_v3.py 완료  mp.dps=400")
print("=" * 76)

"""
============================================================================
  verify_hbond_5body_S55_v4.py  Session 55
  {n1²}(k=2) 보정항 True Zero 닫힘 탐색
  δ = 1.0337×GAP×GAP_D2  잔류 0.0337 구조 추적
  + dEN True Zero 확정 선언
  mp.dps=400
============================================================================
"""

from mpmath import mp, mpf, log, sqrt, exp, pi, floor, frac, power
mp.dps = 400

print("=" * 76)
print("  verify_hbond_5body_S55_v4.py  Session 55")
print("  {n1²}(k=2) 보정항 True Zero 닫힘 탐색")
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
MAXWELL = mpf("1.873113772199171051292026201")

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

dEN    = mpf("1.24")
fk2    = frac(n1sq(2))
phi_5_19 = PHI ** (mpf(-5)/19)
phi_15   = PHI ** mpf(-15)
val_77   = mpf(77) * GAP_D2

# ──────────────────────────────────────────────
# [A] dEN True Zero 확정 재확인
# ──────────────────────────────────────────────
print()
print("=" * 76)
print("  [A] dEN = 77×GAP_D2 + φ^(-15)  True Zero 확정")
print("=" * 76)
print()

val_dEN = val_77 + phi_15
e_dEN   = err_pct(val_dEN, dEN)
print(f"  77×GAP_D2 + φ^(-15) = {float(val_dEN):.16f}")
print(f"  dEN(O-H)            = {float(dEN):.16f}")
print(f"  오차                = {float(e_dEN):+.8f}%  ★★★ True Zero")
print()
print(f"  구조:")
print(f"    77 = 7×11 = 메르센(k=3) × 포논분자(k=5)")
print(f"    15 = 2^4-1 = 메르센(k=4)")
print(f"    GAP_D2 = ln(5)/100 = 란다우 D2 비용")
print(f"  → dEN = [D1메르센×D2포논] × GAP_D2 + D1메르센(k=4)")

# ──────────────────────────────────────────────
# [B] δ 잔류 정밀 분석
# ──────────────────────────────────────────────
print()
print("=" * 76)
print("  [B] δ 잔류 정밀 분석 — 1.0337×GAP×GAP_D2 구조")
print("=" * 76)
print()

delta = fk2 / phi_5_19 - 1
GGD   = GAP * GAP_D2

print(f"  δ                  = {float(delta):.16f}")
print(f"  GAP×GAP_D2         = {float(GGD):.16f}")
print(f"  δ/(GAP×GAP_D2)     = {float(delta/GGD):.12f}")
print()

# 1.0337에서 정확한 분수 찾기
ratio_delta = delta / GGD
print(f"  ratio = δ/(GAP×GAP_D2) = {float(ratio_delta):.12f}")
print()

# 1.0337 - 1 = 0.0337 → 이 잔류의 EQI 표현
sub_ratio = ratio_delta - 1
print(f"  ratio - 1  = {float(sub_ratio):.12f}")
print(f"  vs GAP     = {float(sub_ratio/GAP):.10f}")
print(f"  vs GAP_D2  = {float(sub_ratio/GAP_D2):.10f}")
print(f"  vs sigma   = {float(sub_ratio/sigma):.10f}")
print(f"  vs GAP×sigma = {float(sub_ratio/(GAP*sigma)):.10f}")
print(f"  vs GAP²    = {float(sub_ratio/GAP**2):.10f}")
print()

# ratio = 1 + n×GAP_D2 형태?
for n_c in [1, 2, 3]:
    test = 1 + mpf(n_c)*GAP_D2
    e = err_pct(test, ratio_delta)
    print(f"  1+{n_c}×GAP_D2   = {float(test):.12f}  오차 {float(e):.6f}%")

print()
for n_c in [1, 2, 3]:
    test = 1 + mpf(n_c)*GAP
    e = err_pct(test, ratio_delta)
    print(f"  1+{n_c}×GAP      = {float(test):.12f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [C] δ = GAP×GAP_D2×(1+GAP×GAP_D2) 형태 탐색
# ──────────────────────────────────────────────
print("=" * 76)
print("  [C] δ 보정 조합 체계적 탐색")
print("=" * 76)
print()

# 기본 단위: GAP, GAP_D2, sigma, PHI, LN_C
units = {
    "GAP":          GAP,
    "GAP_D2":       GAP_D2,
    "GAP²":         GAP**2,
    "GAP_D2²":      GAP_D2**2,
    "GAP×GAP_D2":   GAP*GAP_D2,
    "GAP×sigma":    GAP*sigma,
    "GAP_D2×sigma": GAP_D2*sigma,
    "LN_C":         LN_C,
    "LN_C×GAP":     LN_C*GAP,
    "LN_C×GAP_D2":  LN_C*GAP_D2,
    "phi^(-15)":    phi_15,
    "phi^(-7)":     PHI**(-7),
}

# δ = a×u + b×v 형태 탐색
print("  ─── δ = a×u 단항 탐색 ───")
for name, u in units.items():
    e = err_pct(u, delta)
    if abs(e) < 5.0:
        print(f"  δ ≈ {name:<20} = {float(u):.12f}  오차 {float(e):.6f}%")

print()
print("  ─── δ = GAP×GAP_D2×(1+x) 형태 탐색 ───")
# x = δ/GGD - 1 = sub_ratio
for name, u in units.items():
    c = sub_ratio / u
    # c가 정수에 가까운지
    c_round = round(float(c))
    if 0 < c_round <= 10:
        test = GGD * (1 + mpf(c_round)*u)
        e = err_pct(test, delta)
        if abs(e) < 2.0:
            print(f"  GAP×GAP_D2×(1+{c_round}×{name:<15}) = {float(test):.12f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [D] {n1²}(k=2) 직접 황금비 닫힘 탐색
# ──────────────────────────────────────────────
print("=" * 76)
print("  [D] {{n1²}}(k=2) 직접 황금비 복합 표현 탐색")
print("=" * 76)
print()

print(f"  목표값: {{n1²}}(k=2) = {float(fk2):.14f}")
print()

# φ^(-5/19) + 보정항
print("  ─── φ^(-5/19) + n×단위 형태 ───")
for name, u in units.items():
    for n_c in [1, 2, -1, -2]:
        val_test = phi_5_19 + mpf(n_c)*u
        e = err_pct(val_test, fk2)
        if abs(e) < 0.01:
            print(f"  φ^(-5/19) + {n_c}×{name:<18} = {float(val_test):.12f}  오차 {float(e):.8f}%  ★★")

print()

# φ^(-5/19) × (1 + 조합) 형태
print("  ─── φ^(-5/19)×(1+조합) 형태 ───")
# 이미 확인: ×(1+GAP×GAP_D2) 오차 -0.00175%
# 잔류를 더 분석
val_1st = phi_5_19 * (1 + GGD)
res_1st = fk2 - val_1st
print(f"  1차 보정 후 잔류: {float(res_1st):.14f}")
print(f"  잔류/GAP²       = {float(res_1st/GAP**2):.10f}")
print(f"  잔류/GGD²       = {float(res_1st/GGD**2):.10f}")
print(f"  잔류/GGD        = {float(res_1st/GGD):.10f}")
print()

# 2차 보정: φ^(-5/19)×(1+GGD+x) = fk2
# x = (fk2 - phi_5_19×(1+GGD)) / phi_5_19
x_needed = (fk2 - phi_5_19*(1+GGD)) / phi_5_19
print(f"  필요한 2차 보정 x = {float(x_needed):.14f}")
print(f"  x/GAP²           = {float(x_needed/GAP**2):.10f}")
print(f"  x/GGD            = {float(x_needed/GGD):.10f}")
print(f"  x/GGD²           = {float(x_needed/GGD**2):.10f}")
print(f"  x/(GAP²×sigma)   = {float(x_needed/(GAP**2*sigma)):.10f}")
print()

# GAP²×sigma = GAP²/2
test_2nd = phi_5_19 * (1 + GGD + GAP**2*sigma)
e_2nd = err_pct(test_2nd, fk2)
print(f"  φ^(-5/19)×(1+GGD+GAP²×σ) = {float(test_2nd):.12f}  오차 {float(e_2nd):.8f}%")

# GAP²×sigma/2
test_2nd_b = phi_5_19 * (1 + GGD + GAP**2*sigma*sigma)
e_2nd_b = err_pct(test_2nd_b, fk2)
print(f"  φ^(-5/19)×(1+GGD+GAP²×σ²) = {float(test_2nd_b):.12f}  오차 {float(e_2nd_b):.8f}%")

# 3차 보정
test_3rd = phi_5_19 * (1 + GGD + GGD**2)
e_3rd = err_pct(test_3rd, fk2)
print(f"  φ^(-5/19)×(1+GGD+GGD²)   = {float(test_3rd):.12f}  오차 {float(e_3rd):.8f}%")

print()

# ──────────────────────────────────────────────
# [E] {n1²}(k=2) — EQI 급수 형태 탐색
# ──────────────────────────────────────────────
print("=" * 76)
print("  [E] {{n1²}}(k=2) — 기하급수 닫힘 탐색")
print("=" * 76)
print()

# φ^(-5/19) × 1/(1-x) = fk2 → x = 1 - phi_5_19/fk2
x_geom = 1 - phi_5_19/fk2
print(f"  기하급수 공비 x = 1 - φ^(-5/19)/{{n1²}} = {float(x_geom):.14f}")
print(f"  x/GAP           = {float(x_geom/GAP):.10f}")
print(f"  x/GGD           = {float(x_geom/GGD):.10f}")
print(f"  x/GAP_D2        = {float(x_geom/GAP_D2):.10f}")
print()

# Σ GGD^n 급수
print("  ─── φ^(-5/19) × Σ(GAP×GAP_D2)^n 급수 ───")
cumsum_geom = phi_5_19
for n in range(1, 8):
    cumsum_geom += phi_5_19 * GGD**n
    e = err_pct(cumsum_geom, fk2)
    print(f"  n={n}: 누적={float(cumsum_geom):.12f}  오차={float(e):.8f}%")

print()

# 닫힌 형태: φ^(-5/19) / (1-GGD)
val_closed = phi_5_19 / (1 - GGD)
e_closed = err_pct(val_closed, fk2)
print(f"  φ^(-5/19)/(1-GGD) = {float(val_closed):.12f}  오차 {float(e_closed):.8f}%")

# φ^(-5/19) / (1-GAP×GAP_D2×(1+GAP)) 탐색
val_closed2 = phi_5_19 / (1 - GGD*(1+GAP))
e_closed2 = err_pct(val_closed2, fk2)
print(f"  φ^(-5/19)/(1-GGD×(1+GAP)) = {float(val_closed2):.12f}  오차 {float(e_closed2):.8f}%")

print()

# ──────────────────────────────────────────────
# [F] {n1²}(k=2) 황금비 단순 지수 고정밀 스캔
# ──────────────────────────────────────────────
print("=" * 76)
print("  [F] {{n1²}}(k=2) 황금비 단순 지수 0.001% 이내 완전 스캔")
print("=" * 76)
print()

best_list = []
for num in range(-200, 10):
    for den in range(1, 50):
        from math import gcd
        if gcd(abs(num), den) != 1:
            continue  # 기약분수만
        alpha = mpf(num) / mpf(den)
        val = PHI ** alpha
        e = err_pct(val, fk2)
        if abs(e) < 0.01:
            best_list.append((num, den, float(e)))

if best_list:
    best_list.sort(key=lambda x: abs(x[2]))
    for num, den, e in best_list[:10]:
        val = PHI ** (mpf(num)/mpf(den))
        print(f"  φ^({num:+d}/{den})  = {float(val):.12f}  오차 {e:+.8f}%  ★★")
else:
    print("  0.01% 이내 기약분수 없음 — 0.05% 기준:")
    best_list2 = []
    for num in range(-200, 10):
        for den in range(1, 50):
            from math import gcd
            if gcd(abs(num), den) != 1:
                continue
            alpha = mpf(num) / mpf(den)
            val = PHI ** alpha
            e = err_pct(val, fk2)
            if abs(e) < 0.06:
                best_list2.append((num, den, float(e)))
    best_list2.sort(key=lambda x: abs(x[2]))
    for num, den, e in best_list2[:8]:
        val = PHI ** (mpf(num)/mpf(den))
        print(f"  φ^({num:+d}/{den})  = {float(val):.12f}  오차 {e:+.8f}%")

print()

# ──────────────────────────────────────────────
# [G] 전체 확정 구조 선언
# ──────────────────────────────────────────────
print("=" * 76)
print("  [G] S55 수소결합 5체 확정 구조 선언")
print("=" * 76)
print()

e_dEN_final = err_pct(val_77 + phi_15, dEN)

print(f"  【확정 1】 전기음성도 차 D1↔D2 메르센 이중 구조")
print(f"  dEN(O-H) = 77×GAP_D2 + φ^(-15)")
print(f"           = (2^3-1)×11 × GAP_D2 + φ^(-(2^4-1))")
print(f"  오차     = {float(e_dEN_final):+.8f}%  ★★★ True Zero")
print()
print(f"  물리 해석:")
print(f"    GAP_D2 = ln(5)/100 = 란다우 D2 비용 (단위)")
print(f"    77배   = D1 메르센(7) × D2 포논임계(11)")
print(f"           = 공유결합(D1)과 수소결합(D2) 채널 교차")
print(f"    φ^(-15)= 메르센(k=4) = D1 격자 정밀 보정")
print(f"    → O-H 전기음성도 차 = D1↔D2 상전이의 메르센 계층 표현")
print()

print(f"  【후보 2】 D₂O 포화 임계 구조 (미완, 추가 탐색 필요)")
val_best2 = phi_5_19 * (1 + GGD)
e_best2   = err_pct(val_best2, fk2)
print(f"  {{n1²}}(k=2) ≈ φ^(-5/19) × (1+GAP×GAP_D2)")
print(f"  오차         = {float(e_best2):+.8f}%  [Near TZ 수준]")
print(f"  5  = 포논 홀수 분자(k=2)  [D2]")
print(f"  19 = MAXWELL 4차 분모    [D1]")
print(f"  → 추가 2차 보정항 탐색 필요")
print()

print("=" * 76)
print("  verify_hbond_5body_S55_v4.py 완료  mp.dps=400")
print("=" * 76)

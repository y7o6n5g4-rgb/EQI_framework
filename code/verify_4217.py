"""
EQI Framework — Session 41
4217 정체 탐색

핵심:
  δ×TARGET = 4217 × GD⁴/2   오차 0.000209%

4217의 대수 정체:
  4217 = ?
  → 소수 분해
  → 34/137/11/24 계열 연결
  → Leibniz tail 연결
  → GD 역수 계열 연결
  → cancel_rate 연결

mp.dps = 400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, cos, factorial

mp.dps = 400

PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
GD     = GAP_D2
c_EQI  = PHI ** mpf('0.5')
LN_C   = log(PHI) / 2
D1_HCW = PHI**(-mpf('1.5'))
D2_HCW = PHI**(+mpf('1.5'))

def leibniz_residual(N):
    s = mpf(0)
    for k in range(N):
        s += 1/(4*k+1) - 1/(4*k+3)
    return 4*pi - 16*s

def leibniz_tail(N, M=2000):
    """Σ_{k=N}^{M} [1/(4k+1)−1/(4k+3)]"""
    s = mpf(0)
    for k in range(N, M):
        s += 1/(4*k+1) - 1/(4*k+3)
    return 16*s

TARGET = GAP - 11 * GAP_D2**2
R_34   = leibniz_residual(34)
ratio  = R_34 / TARGET
delta  = ratio - 2
dT     = delta * TARGET

print("=" * 70)
print("EQI Session 41 — 4217 정체 탐색")
print("=" * 70)
print(f"  δ×TARGET      = {float(dT):.15e}")
print(f"  4217×GD⁴/2    = {float(4217*GD**4/2):.15e}")
print(f"  오차           = {float(fabs(dT - 4217*GD**4/2)/fabs(dT)*100):.8f}%")

# ── §1. 4217 소수 분해 및 구조 탐색 ─────────────────────────
print(f"\n{'='*70}")
print(f"§1. 4217 소수 분해 및 EQI 상수 연결")
print(f"{'='*70}")

n = 4217
factors = []
temp = n
for p in range(2, 200):
    while temp % p == 0:
        factors.append(p)
        temp //= p
if temp > 1:
    factors.append(temp)
print(f"  4217 소수 분해: {factors}")
print(f"  4217 소수 여부: {len(factors)==1 and factors[0]==4217}")
print(f"")

# 구조 탐색
print(f"  구조 탐색:")
print(f"    4217 / 137     = {4217/137:.8f}")
print(f"    4217 / 274     = {4217/274:.8f}")
print(f"    4217 / 34      = {4217/34:.8f}")
print(f"    4217 / 34²     = {4217/34**2:.8f}")
print(f"    4217 − 34²     = {4217 - 34**2}  (= 4217−1156)")
print(f"    4217 − 4×34²  = {4217 - 4*34**2}  (= 4217−4624)")
print(f"    4217 − 11²     = {4217 - 121}  = {4217-121}")
print(f"    4217 − 2×11²  = {4217 - 242}")
print(f"    (4217−1)/4     = {(4217-1)/4:.4f}")
print(f"    (4217+1)/2     = {(4217+1)/2}")
print(f"    4217 / (6×7)  = {4217/42:.8f}")
print(f"    4217 / 41      = {4217/41:.8f}")
print(f"    √4217          = {float(mpf(4217)**mpf('0.5')):.8f}")
print(f"    4217 / π       = {float(mpf(4217)/pi):.8f}")
print(f"    4217 / (4π)    = {float(mpf(4217)/(4*pi)):.8f}")
print(f"    4217×2/34²     = {4217*2/34**2:.8f}")
print(f"    4217×6/34²     = {4217*6/34**2:.8f}")

# ── §2. 정확한 δ×TARGET/(GD⁴) 값 ────────────────────────────
print(f"\n{'='*70}")
print(f"§2. δ×TARGET/GD⁴ 정확값 — 4217 근방 스캔")
print(f"{'='*70}")

exact_ratio = dT / (GD**4)
print(f"  δ×TARGET / GD⁴ = {float(exact_ratio):.15f}")
print(f"  /2              = {float(exact_ratio/2):.15f}")
print(f"  /2 근접정수     = {round(float(exact_ratio/2))}")
print(f"  잔류            = {float(exact_ratio/2 - round(float(exact_ratio/2))):.10f}")
print(f"")

# 4217 주변 정수 스캔
print(f"  n 스캔 (δT = n×GD⁴/2):")
print(f"  {'n':>6}  {'n×GD⁴/2':>20}  {'오차%':>12}  {'n분해':>20}")
print(f"  {'-'*65}")
for n_try in range(4210, 4225):
    val = n_try * GD**4 / 2
    err = float(fabs(val - dT)/fabs(dT)*100)
    # 간단 인수 체크
    divs = [d for d in [2,3,4,5,6,7,11,13,17,19,23,34,41,137] if n_try%d==0]
    div_str = "×".join(str(d) for d in divs) if divs else "소수?"
    marker = " ★" if err < 0.001 else ""
    print(f"  {n_try:>6}  {float(val):>20.10e}  {err:>12.6f}%  {div_str:>20}{marker}")

# ── §3. Leibniz tail × GD 구조 ──────────────────────────────
print(f"\n{'='*70}")
print(f"§3. Leibniz tail(N=34) × GD 구조")
print(f"{'='*70}")
tail_34 = leibniz_tail(34)
print(f"  tail(34) = 16×Σ_{{k=34}}^∞ [1/(4k+1)−1/(4k+3)]")
print(f"           ≈ {float(tail_34):.15f}")
print(f"  tail(34) ≈ 16/(2×137) = 8/137 = {float(mpf(8)/137):.15f}")
print(f"  tail(34)/GD  = {float(tail_34/GD):.10f}")
print(f"  tail(34)/GD² = {float(tail_34/GD**2):.10f}")
print(f"  tail(34)/GD³ = {float(tail_34/GD**3):.10f}")
print(f"  tail(34)/GD⁴ = {float(tail_34/GD**4):.10f}")
print(f"")
print(f"  dT / tail(34)       = {float(dT/tail_34):.10f}")
print(f"  dT / (tail(34)×GD) = {float(dT/(tail_34*GD)):.10f}")
print(f"  dT / (tail(34)×GD²)= {float(dT/(tail_34*GD**2)):.10f}")
print(f"")
# 첫 잔류쌍 16/137−16/139
first_pair = mpf(16)/137 - mpf(16)/139
print(f"  첫 잔류쌍 16/137−16/139 = {float(first_pair):.15e}")
print(f"  dT / first_pair     = {float(dT/first_pair):.10f}")
print(f"  dT / (first_pair×GD)= {float(dT/(first_pair*GD)):.10f}")
print(f"  dT×137/(first_pair) = {float(dT*137/first_pair):.10f}")

# ── §4. 4217 = f(34, 11, 137) 조합 탐색 ─────────────────────
print(f"\n{'='*70}")
print(f"§4. 4217 = f(34, 11, 137, 24) 조합 탐색")
print(f"{'='*70}")

combos = {
    "34³/GD×(1/π)"          : float(34**3 * float(1/pi)),
    "34²×(34/8)"             : 34**2 * 34/8,
    "34²×137/34"             : 34**2 * 137/34,
    "4×34² + 1"              : 4*34**2 + 1,
    "4×34² − 7"              : 4*34**2 - 7,
    "137×(34−3)"             : 137*(34-3),
    "137×31"                  : 137*31,
    "137×30 + 47"            : 137*30 + 47,
    "11×383"                  : 11*383,
    "11×384 − 7"             : 11*384 - 7,
    "24×175 + 17"            : 24*175 + 17,
    "24×176 − 7"             : 24*176 - 7,
    "274×(34/2) − 1"         : 274*17 - 1,
    "274×(34/2) + 1"         : 274*17 + 1,
    "34²×(1+1/34²)×GD⁻¹"   : None,  # 스킵
    "6940/GD×GD²"            : None,  # 스킵
    "2×11×191 + 15"          : 2*11*191 + 15,
    "7×602 + 3"              : 7*602 + 3,
    "41×102 + 35"            : 41*102 + 35,
    "34×124 + 1"             : 34*124 + 1,
    "34×124"                  : 34*124,
    "2×34×62"                : 2*34*62,
}

print(f"  목표: 4217")
print(f"  {'표현식':30s}  {'값':10s}  {'일치':6s}")
print(f"  {'-'*50}")
for name, val in combos.items():
    if val is None:
        continue
    match = "✓" if abs(val - 4217) < 0.01 else f"차이={val-4217:.1f}"
    print(f"  {name:30s}  {val:10.2f}  {match}")

# ── §5. δ×TARGET = n×GD⁴/2 에서 n의 EQI 구조 ────────────────
print(f"\n{'='*70}")
print(f"§5. δ×TARGET/GD⁴ 연분수 전개")
print(f"{'='*70}")
val_exact = exact_ratio / 2  # ≈ 4217.something
print(f"  δ×TARGET/(GD⁴/2) = {float(val_exact):.15f}")
frac_part = val_exact - floor(val_exact)
print(f"  정수부            = {int(floor(val_exact))}")
print(f"  소수부            = {float(frac_part):.15f}")
print(f"  소수부×2          = {float(2*frac_part):.10f}")
print(f"  소수부×4          = {float(4*frac_part):.10f}")
print(f"  소수부×8          = {float(8*frac_part):.10f}")
print(f"  소수부×137        = {float(137*frac_part):.10f}")
print(f"  소수부×274        = {float(274*frac_part):.10f}")
print(f"  소수부×34²        = {float(34**2*frac_part):.10f}")
print(f"  소수부×34³        = {float(34**3*frac_part):.10f}")
print(f"  1/소수부          = {float(1/frac_part):.10f}")
print(f"")
# 소수부가 EQI 상수로 표현되는지
print(f"  소수부 vs EQI 상수:")
print(f"    GD×소수부 비교:")
print(f"      소수부/GD     = {float(frac_part/GD):.8f}")
print(f"      소수부/GD²    = {float(frac_part/GD**2):.8f}")
print(f"      소수부/GAP    = {float(frac_part/GAP):.8f}")
print(f"      소수부×274    = {float(frac_part*274):.8f}")
print(f"      소수부×137    = {float(frac_part*137):.8f}")

# ── §6. GD⁴ 계열 물리 해석 ──────────────────────────────────
print(f"\n{'='*70}")
print(f"§6. GD⁴ 계열 물리 해석")
print(f"{'='*70}")
print(f"  GD  = GAP_D2 = ln(5)/100  [D1↔D2 단방향 상전이 비용]")
print(f"  GD² = 2D 면 마찰 단위      [§6 GW170814 보정항 단위]")
print(f"  GD³ = 3D 부피 마찰 단위    [sin 급수 k=1 항, 3차 잔류]")
print(f"  GD⁴ = 4D 초부피 마찰?      [cos 급수 k=2 항, 4차 잔류]")
print(f"")
print(f"  GD⁴/2 = cos 급수 k=2 항 = GD⁴/4! × 12 = ?")
print(f"  cos k=2항: GD⁴/24")
print(f"  GD⁴/2 = 12 × (GD⁴/24) = 12 × cos_k2항")
print(f"")
print(f"  cos 급수 k=2항: GD⁴/24 = {float(GD**4/24):.15e}")
print(f"  GD⁴/2          = {float(GD**4/2):.15e}")
print(f"  δ×TARGET       = {float(dT):.15e}")
print(f"  δT/(GD⁴/24)    = {float(dT/(GD**4/24)):.10f}")
print(f"  = 4217×12       = {4217*12}")
print(f"  50604 = ?")
print(f"  50604/137       = {50604/137:.6f}")
print(f"  50604/274       = {50604/274:.6f}")
print(f"  50604/34        = {50604/34:.6f}")
print(f"  50604/12        = {50604/12:.6f}")
print(f"  50604/4         = {50604/4:.6f}")
print(f"  50604/369       = {50604/369:.6f}")
print(f"  50604 = 4×12651 = 4×3×4217 = 12×4217")

# ── §7. δ 완전 표현 후보 ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"§7. δ 완전 표현 후보 종합")
print(f"{'='*70}")

expressions = {
    "34²×GD³"                          : 34**2 * GD**3,
    "34²×GD³ + δ_잔류"                 : None,
    "4217×GD⁴/(2×TARGET)"             : 4217*GD**4/(2*TARGET),
    "4217×GD⁴/(2×GAP_D2×(2−11×GD))" : 4217*GD**4/(2*GD*(2-11*GD)),
    "4217×GD³/(2×(2−11×GD))"          : 4217*GD**3/(2*(2-11*GD)),
    "34²×GD³×(1+GD/(2×TARGET))"       : 34**2*GD**3*(1+GD/(2*TARGET)),
}

print(f"  δ = {float(delta):.15e}")
print(f"")
print(f"  {'표현식':45s}  {'값':20s}  {'오차%':10s}")
print(f"  {'-'*80}")
for name, val in expressions.items():
    if val is None:
        continue
    err = float(fabs(val-delta)/fabs(delta)*100)
    marker = " ★" if err < 0.01 else ""
    print(f"  {name:45s}  {float(val):20.12e}  {err:10.6f}%{marker}")

# ── §8. 요약 ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"§8. 요약")
print(f"{'='*70}")
print(f"  핵심 수치:")
print(f"    δ×TARGET/(GD⁴/2) = {float(val_exact):.10f}")
print(f"    정수부 = {int(floor(val_exact))}")
print(f"    소수부 = {float(frac_part):.10f}")
print(f"")
print(f"  확정 구조:")
print(f"    δ×TARGET ≈ 4217×GD⁴/2  오차 0.000209%")
print(f"    4217 소수 여부: §1 결과 참조")
print(f"    cancel_rate 2차 분모 24=int_x=4! ✓")
print(f"    11=int_x/2−1=Leibniz k=2 D2분모 ✓")
print(f"    34²×GD³ ≈ δ  오차 0.055% ✓")
print(f"")
print(f"  미해결:")
print(f"    4217의 대수 정체 → §1,§4 결과 참조")
print(f"    소수부의 EQI 표현 → §5 결과 참조")

print(f"\n{'='*70}")
print(f"Session 41 4217 정체 탐색 완료  mp.dps={mp.dps}")
print(f"{'='*70}")

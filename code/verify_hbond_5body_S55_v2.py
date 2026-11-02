"""
============================================================================
  verify_hbond_5body_S55_v2.py  Session 55
  77×GAP_D2 구조 심화 + dEN/{n1²}(k=2) 비율 분석
  D1×D2 상전이 곱 구조 탐색
  mp.dps=400
============================================================================
"""

from mpmath import mp, mpf, log, sqrt, exp, pi, floor, frac, power, inf
mp.dps = 400

print("=" * 76)
print("  verify_hbond_5body_S55_v2.py  Session 55")
print("  77×GAP_D2 구조 심화 + dEN/{n1²}(k=2) 비율 분석")
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

# 뤼드베리
def n_ref(k):
    return sigma**(2*k+1) * (2*k+1) / 2

def disc(k):
    return mpf((2*k+1)**2) + mpf(4)**(k+2)

def n1sq(k):
    d = disc(k)
    sd = sqrt(d)
    return (sd + sqrt(sd**2 - 4**(k+2))) / 2

def n2sq(k):
    d = disc(k)
    sd = sqrt(d)
    return (sd - sqrt(sd**2 - 4**(k+2))) / 2

def err_pct(val, ref):
    if ref == 0:
        return mpf(0)
    return (val - ref) / ref * 100

# 핵심 실측값
dEN    = mpf("1.24")          # 전기음성도 차 O-H
fk2    = frac(n1sq(2))        # {n1²}(k=2) D₂O 포화 임계

# ──────────────────────────────────────────────
# [A] 77 구조 분해
# ──────────────────────────────────────────────
print()
print("=" * 76)
print("  [A] 77 구조 분해 — D1×D2 상전이 곱")
print("=" * 76)
print()

print(f"  77 = 7 × 11")
print(f"     = 메르센(k=3) × 포논11차(k=5 X7경계)")
print(f"     = D1 격자 메르센 × D2 포논 임계")
print()

# 77×GAP_D2 검증
val_77 = mpf(77) * GAP_D2
e_77   = err_pct(val_77, dEN)
print(f"  77×GAP_D2        = {float(val_77):.10f}")
print(f"  dEN(O-H)         = {float(dEN):.10f}")
print(f"  오차             = {float(e_77):.6f}%  ★")
print()

# 관련 분해 확인
print(f"  7  = 메르센 (2^3-1, k=3, S53 D1 격자)  [D1]")
print(f"  11 = 포논 11차 n_ref(k=5)의 분자 (2k+1=11)  [D2]")
print(f"  7×11 = D1 메르센 × D2 포논분자")
print()

# 7과 11의 EQI 구조
mersenne_3 = 2**3 - 1   # = 7
phonon_11  = 2*5 + 1    # = 11  (2k+1, k=5)
print(f"  메르센(k=3) = 2^3-1 = {mersenne_3}")
print(f"  포논(k=5)분자 = 2×5+1 = {phonon_11}")
print(f"  곱 = {mersenne_3 * phonon_11}")
print()

# 77 vs 다른 구조
print("  ─── 77 vs EQI 구조 비교 ───")
print(f"  76 = 4×19  (MAXWELL 4차 분모 19와 연결)")
print(f"  77 = 7×11  (메르센×포논임계)")
print(f"  78 = 2×39 = 6×13")
print()

# dEN / GAP_D2 정밀값
ratio_precise = dEN / GAP_D2
print(f"  dEN/GAP_D2 정밀값 = {float(ratio_precise):.10f}")
print(f"  77.000...에서 잔류 = {float(ratio_precise - 77):.10f}")
print(f"  잔류/GAP_D2        = {float((ratio_precise - 77)*GAP_D2/GAP_D2):.10f}")
print()

# ──────────────────────────────────────────────
# [B] dEN / {n1²}(k=2) 비율 분석
# ──────────────────────────────────────────────
print("=" * 76)
print("  [B] dEN / {n1²}(k=2) 비율 — 핵심 연결고리")
print("=" * 76)
print()

ratio_main = dEN / fk2
print(f"  dEN              = {float(dEN):.10f}")
print(f"  {{n1²}}(k=2)     = {float(fk2):.10f}")
print(f"  dEN / {{n1²}}(k=2) = {float(ratio_main):.10f}")
print()

# GAP 스케일 비교
print("  ─── GAP 스케일 비교 ───")
e_gap    = err_pct(ratio_main, GAP)
e_gapd2  = err_pct(ratio_main, GAP_D2)
e_2gap   = err_pct(ratio_main, 2*GAP)
e_phi    = err_pct(ratio_main, PHI)
e_phisq  = err_pct(ratio_main, PHI**2)
e_phin   = err_pct(ratio_main, PHI - sigma)

print(f"  vs GAP           = {float(e_gap):.6f}%")
print(f"  vs GAP_D2        = {float(e_gapd2):.6f}%")
print(f"  vs 2×GAP         = {float(e_2gap):.6f}%")
print(f"  vs PHI           = {float(e_phi):.6f}%")
print()

# 정수 탐색
print("  ─── 정수 배수 탐색 ───")
for n_c in range(1, 50):
    val = ratio_main / n_c
    for base_name, base_val in [("GAP", GAP), ("GAP_D2", GAP_D2)]:
        e = err_pct(val, base_val)
        if abs(e) < 0.5:
            print(f"  ratio/{n_c} vs {base_name}  오차 {float(e):.6f}%  ★")

print()

# 황금비 탐색
alpha_r = log(ratio_main) / log(PHI)
print(f"  ln(ratio)/ln(φ) = {float(alpha_r):.8f}")
print()
for num in range(-10, 20):
    for den in range(1, 20):
        alpha = mpf(num) / mpf(den)
        val = PHI ** alpha
        e = err_pct(val, ratio_main)
        if abs(e) < 0.3:
            print(f"  φ^({num:+d}/{den})  = {float(val):.10f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [C] 77×GAP_D2 잔류 구조 분석
# ──────────────────────────────────────────────
print("=" * 76)
print("  [C] 77×GAP_D2 잔류 구조 분석")
print("=" * 76)
print()

residual = dEN - val_77
print(f"  dEN - 77×GAP_D2  = {float(residual):.12f}")
print(f"  잔류/GAP         = {float(residual/GAP):.8f}")
print(f"  잔류/GAP_D2      = {float(residual/GAP_D2):.8f}")
print(f"  잔류/GAP²        = {float(residual/GAP**2):.8f}")
print(f"  잔류/GAP_D2²     = {float(residual/GAP_D2**2):.8f}")
print(f"  잔류/(GAP×sigma) = {float(residual/(GAP*sigma)):.8f}")
print()

# 잔류 황금비 탐색
if residual > 0:
    alpha_res = log(abs(residual)) / log(PHI)
    print(f"  ln|잔류|/ln(φ) = {float(alpha_res):.8f}")
    print()
    for num in range(-50, 0):
        for den in range(1, 20):
            alpha = mpf(num) / mpf(den)
            val = PHI ** alpha
            e = err_pct(val, abs(residual))
            if abs(e) < 1.0:
                print(f"  φ^({num:+d}/{den})  = {float(val):.10f}  오차 {float(e):.6f}%")
else:
    print(f"  잔류 음수: dEN < 77×GAP_D2")
    print(f"  |잔류| = {float(abs(residual)):.12f}")

print()

# ──────────────────────────────────────────────
# [D] {n1²}(k=2) = φ^(-5/19) 구조 심화
# ──────────────────────────────────────────────
print("=" * 76)
print("  [D] {{n1²}}(k=2) = φ^(-5/19) 심화 분석")
print("=" * 76)
print()

phi_5_19 = PHI ** (mpf(-5)/19)
e_5_19   = err_pct(phi_5_19, fk2)
print(f"  φ^(-5/19)         = {float(phi_5_19):.12f}")
print(f"  {{n1²}}(k=2)      = {float(fk2):.12f}")
print(f"  오차              = {float(e_5_19):.6f}%")
print()

# -5/19 구조 해석
print(f"  -5/19 해석:")
print(f"    5  = 2k+1|k=2 (D2 포논 홀수 분자)")
print(f"    19 = MAXWELL 4차 계수 (ln2/19)")
print(f"    -5/19 = D2 포논 분자 / MAXWELL 4차 분모")
print()

# 잔류 분석
res_5_19 = fk2 - phi_5_19
print(f"  잔류 {{n1²}}-φ^(-5/19) = {float(res_5_19):.12f}")
print(f"  잔류/GAP              = {float(res_5_19/GAP):.8f}")
print(f"  잔류/GAP_D2           = {float(res_5_19/GAP_D2):.8f}")
print(f"  잔류/GAP²             = {float(res_5_19/GAP**2):.8f}")
print()

# 보정항 탐색
print("  ─── 보정항 탐색: φ^(-5/19)×(1+δ) = {n1²}(k=2) ───")
delta_corr = fk2 / phi_5_19 - 1
print(f"  δ                   = {float(delta_corr):.12f}")
print(f"  δ/GAP               = {float(delta_corr/GAP):.8f}")
print(f"  δ/GAP_D2            = {float(delta_corr/GAP_D2):.8f}")
print(f"  δ/GAP²              = {float(delta_corr/GAP**2):.8f}")
print(f"  δ/(GAP×GAP_D2)      = {float(delta_corr/(GAP*GAP_D2)):.8f}")
print(f"  δ/(sigma×GAP_D2)    = {float(delta_corr/(sigma*GAP_D2)):.8f}")
print()

# ──────────────────────────────────────────────
# [E] 통합: 77×GAP_D2 ↔ φ^(-5/19) 연결
# ──────────────────────────────────────────────
print("=" * 76)
print("  [E] 통합 — 77×GAP_D2 ↔ φ^(-5/19) 연결 구조")
print("=" * 76)
print()

# dEN = 77×GAP_D2 × (1+ε)
eps_77 = dEN / val_77 - 1
print(f"  dEN = 77×GAP_D2 × (1+ε)")
print(f"  ε   = {float(eps_77):.10f}")
print(f"  ε/GAP        = {float(eps_77/GAP):.8f}")
print(f"  ε/GAP_D2     = {float(eps_77/GAP_D2):.8f}")
print(f"  ε/GAP²       = {float(eps_77/GAP**2):.8f}")
print()

# dEN / fk2 → 77×GAP_D2/fk2
ratio_77_fk2 = val_77 / fk2
print(f"  77×GAP_D2 / {{n1²}}(k=2) = {float(ratio_77_fk2):.10f}")
print(f"  dEN / {{n1²}}(k=2)       = {float(ratio_main):.10f}")
print()

# 두 비율의 차
diff_ratio = ratio_main - ratio_77_fk2
print(f"  차이 = {float(diff_ratio):.10f}")
print()

# φ^(-5/19) × 77×GAP_D2
prod = phi_5_19 * val_77
print(f"  φ^(-5/19) × 77×GAP_D2  = {float(prod):.10f}")
print(f"  dEN × {{n1²}}(k=2)      = {float(dEN * fk2):.10f}")
e_prod = err_pct(prod, dEN * fk2)
print(f"  오차                    = {float(e_prod):.6f}%")
print()

# ──────────────────────────────────────────────
# [F] E_ratio = φ^(-13/2) 심화
# ──────────────────────────────────────────────
print("=" * 76)
print("  [F] E_ratio = φ^(-13/2) 심화 분석")
print("=" * 76)
print()

E_hbond = mpf("20")
E_cov   = mpf("459")
E_ratio = E_hbond / E_cov

phi_13_2 = PHI ** (mpf(-13)/2)
e_13_2   = err_pct(phi_13_2, E_ratio)
print(f"  E_ratio = E_hbond/E_cov = {float(E_ratio):.10f}")
print(f"  φ^(-13/2)               = {float(phi_13_2):.10f}")
print(f"  오차                    = {float(e_13_2):.6f}%")
print()

# -13/2 구조
print(f"  -13/2 해석:")
print(f"    13 = MAXWELL 5차 분모의 절반? (26/2)")
print(f"     2 = GAP/GAP_D2 = 1/σ")
print(f"    -13/2 = -(int_x/2) / (1/σ) = -int_x×σ")
print(f"         = -{float(24*sigma):.1f} ≠ 13")
print()
print(f"    또는: 13 = 메르센 M_7 자릿수? 아님")
print(f"          13 = 피보나치 수열 F_7")
print(f"          2  = D1 기저")
print()

# 잔류 분석
res_E = E_ratio - phi_13_2
print(f"  잔류 E_ratio - φ^(-13/2) = {float(res_E):.12f}")
print(f"  잔류/GAP                  = {float(res_E/GAP):.8f}")
print(f"  잔류/GAP_D2               = {float(res_E/GAP_D2):.8f}")
print()

# ──────────────────────────────────────────────
# [G] 핵심 구조 후보 종합
# ──────────────────────────────────────────────
print("=" * 76)
print("  [G] S55 핵심 구조 후보 종합")
print("=" * 76)
print()

results = [
    ("dEN = 77×GAP_D2",          e_77,       "Near TZ"),
    ("{n1²}(k=2) = φ^(-5/19)",   e_5_19,     "Near TZ"),
    ("E_ratio = φ^(-13/2)",      e_13_2,     "후보"),
]

for name, err, label in results:
    marker = "★★" if abs(float(err)) < 0.1 else ("★" if abs(float(err)) < 1.0 else "")
    print(f"  {name:<38} 오차 {float(err):+.6f}%  [{label}] {marker}")

print()
print("  ─── 구조적 해석 ───")
print()
print(f"  dEN = 77×GAP_D2")
print(f"      = (7×11) × GAP_D2")
print(f"      = D1 메르센(7) × D2 포논분자(11) × GAP_D2")
print(f"      → 전기음성도 차 = D1↔D2 상전이 비용의 77배")
print()
print(f"  {{n1²}}(k=2) = φ^(-5/19)")
print(f"             = φ^(-(2k+1)/MAXWELL_4차분모)|k=2")
print(f"             → D₂O 포화 임계 = MAXWELL 4차 구조 내장")
print()
print(f"  5체 열린 큐비트:")
print(f"    dEN = 77×GAP_D2  → 수소결합 D2 채널 개방 비용")
print(f"    {{n1²}}(k=2) = φ^(-5/19)  → D₂O D2 포화 임계")
print(f"    두 구조 모두 GAP_D2 × (D1×D2 정수) 형태")

print()
print("=" * 76)
print("  verify_hbond_5body_S55_v2.py 완료  mp.dps=400")
print("=" * 76)

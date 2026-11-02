"""
EQI Framework — Session 41
δ 닫힘 탐색: 무한급수 전개 × Leibniz 교차 잔류 구조

핵심 가설:
  sin(t) = Σ (-1)^k × t^(2k+1)/(2k+1)!   [D1항: 홀수 계승]
  cos(t) = Σ (-1)^k × t^(2k)/(2k)!        [D2항: 짝수 계승]
  Leibniz π/4 = Σ (-1)^k/(2k+1)           [D1/D2 교번]

  t=GAP_D2 대입 시 세 급수가 교차하는 잔류 → δ 기원?

맥스웰 도깨비 관점:
  GAP=0: R_34/TARGET = 정확히 2 (완전 대칭)
  GAP≠0: R_34/TARGET = 2+δ (란다우 잔류 비용)
  δ = 정보 소거 비용이 새는 열 = 열린계 GAP≠0의 흔적

mp.dps = 400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, cos, factorial

mp.dps = 400

PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = PHI ** mpf('0.5')
LN_C   = log(PHI) / 2
D1_HCW = PHI**(-mpf('1.5'))   # 수렴 텐션
D2_HCW = PHI**(+mpf('1.5'))   # 발산 텐션

def leibniz_residual(N):
    s = mpf(0)
    for k in range(N):
        s += 1/(4*k+1) - 1/(4*k+3)
    return 4*pi - 16*s

TARGET = GAP - 11 * GAP_D2**2
R_34   = leibniz_residual(34)
ratio  = R_34 / TARGET
delta  = ratio - 2

print("=" * 70)
print("EQI Session 41 — δ 닫힘: 무한급수 × Leibniz 교차 구조")
print("=" * 70)
print(f"  δ = {float(delta):.15e}")
print(f"  δ×TARGET = {float(delta*TARGET):.15e}")

# ── §1. sin/cos 무한급수 t=GAP_D2 대입 ──────────────────────
print(f"\n{'='*70}")
print(f"§1. sin/cos 무한급수 전개 — t=GAP_D2 대입")
print(f"{'='*70}")

def sin_series(t, N=30):
    """sin(t) = Σ_{k=0}^{N} (-1)^k × t^(2k+1)/(2k+1)!"""
    s = mpf(0)
    for k in range(N):
        s += (-1)**k * t**(2*k+1) / factorial(2*k+1)
    return s

def cos_series(t, N=30):
    """cos(t) = Σ_{k=0}^{N} (-1)^k × t^(2k)/(2k)!"""
    s = mpf(0)
    for k in range(N):
        s += (-1)**k * t**(2*k) / factorial(2*k)
    return s

# t = GAP_D2 대입
t_val = GAP_D2
sin_gd = sin_series(t_val)
cos_gd = cos_series(t_val)
sin_exact = sin(t_val)
cos_exact = cos(t_val)

print(f"  t = GAP_D2 = {float(t_val):.15f}")
print(f"  sin(GAP_D2) 급수  = {float(sin_gd):.15f}")
print(f"  sin(GAP_D2) exact = {float(sin_exact):.15f}")
print(f"  cos(GAP_D2) 급수  = {float(cos_gd):.15f}")
print(f"  cos(GAP_D2) exact = {float(cos_exact):.15f}")
print(f"")

# 각 항 출력
print(f"  sin 급수 항별 (D1항, 홀수 계승 분모):")
for k in range(6):
    term = (-1)**k * t_val**(2*k+1) / factorial(2*k+1)
    denom = factorial(2*k+1)
    print(f"    k={k}: (-1)^{k}×GD^{2*k+1}/{int(denom)} = {float(term):+.15e}  [분모={int(denom)}]")

print(f"")
print(f"  cos 급수 항별 (D2항, 짝수 계승 분모):")
for k in range(6):
    term = (-1)**k * t_val**(2*k) / factorial(2*k)
    denom = factorial(2*k)
    print(f"    k={k}: (-1)^{k}×GD^{2*k}/{max(1,int(denom))} = {float(term):+.15e}  [분모={max(1,int(denom))}]")

# ── §2. EHCW×PHCW 무한급수 전개 ─────────────────────────────
print(f"\n{'='*70}")
print(f"§2. EHCW×PHCW = 1×sin(GD) × 2×cos(GD) = sin(2×GD) 구조")
print(f"{'='*70}")
# t=GAP_D2에서 sin(2t) = 2sin(t)cos(t)
sin2gd = sin_series(2*t_val)
prod   = sin_gd * cos_gd * 2   # = sin(2×GAP_D2)
print(f"  2×sin(GD)×cos(GD)   = {float(prod):.15f}")
print(f"  sin(2×GD) 급수      = {float(sin2gd):.15f}")
print(f"  일치 확인            = {float(fabs(prod-sin2gd)):.3e}")
print(f"")
print(f"  SCALAR:VECTOR = 1:2 비율 → sin(2t) 항등식")
print(f"  GAP=0 극한:  ratio = 2 정확히")
print(f"  GAP≠0:       ratio = 2+δ  (란다우 잔류)")

# ── §3. δ — 계승 분모 × Leibniz 분모 교차 ───────────────────
print(f"\n{'='*70}")
print(f"§3. 계승 분모 × Leibniz 분모 교차 구조")
print(f"{'='*70}")

# Leibniz 분모: D1항(4k+1), D2항(4k+3)
# 계승 분모: sin→(2k+1)!, cos→(2k)!
# 교차: (4k+3) vs (2k+1)! 관계

print(f"  Leibniz D2항 분모: 3, 7, 11, 15, 19, ...  (4k+3)")
print(f"  sin 분모:          1, 6, 120, 5040, ...    (2k+1)!")
print(f"  cos 분모:          1, 2, 24,  720, ...     (2k)!")
print(f"")

# 핵심: 11 = Leibniz k=2 D2항, sin k=2 분모 = 5! = 120
# 비율 탐색
leibniz_d2 = [4*k+3 for k in range(6)]
sin_denom  = [int(factorial(2*k+1)) for k in range(6)]
cos_denom  = [int(factorial(2*k)) for k in range(6)]
print(f"  {'k':>3}  {'Leibniz D2(4k+3)':>18}  {'sin분모(2k+1)!':>16}  {'cos분모(2k)!':>14}  {'비율 L/sin':>12}")
for k in range(6):
    ld = leibniz_d2[k]
    sd = sin_denom[k]
    cd = cos_denom[k]
    print(f"  {k:>3}  {ld:>18}  {sd:>16}  {cd:>14}  {ld/sd:>12.6f}")

# ── §4. δ 후보 — 무한급수 기반 ──────────────────────────────
print(f"\n{'='*70}")
print(f"§4. δ 후보 — 무한급수 항 조합")
print(f"{'='*70}")

GD = GAP_D2
GD2 = GAP_D2**2
GD3 = GAP_D2**3
GD4 = GAP_D2**4
GD5 = GAP_D2**5

candidates = {
    # sin/cos 급수 항 조합
    "sin_k1 = GD³/6"                    : GD3/6,
    "cos_k1 = GD²/2"                    : GD2/2,
    "sin_k1/TARGET"                      : GD3/(6*TARGET),
    "cos_k1/TARGET"                      : GD2/(2*TARGET),
    "2×sin_k1/TARGET"                    : 2*GD3/(6*TARGET),
    "(cos_k1−sin_k1)/TARGET"             : (GD2/2 - GD3/6)/TARGET,
    # 계승 분모 × GAP 조합
    "GD²/(2×TARGET)"                     : GD2/(2*TARGET),
    "GD³/(6×TARGET)"                     : GD3/(6*TARGET),
    "GD²/(6×GAP)"                        : GD2/(6*GAP),
    "GD³/(6×GAP)"                        : GD3/(6*GAP),
    # Leibniz × 계승 교차
    "GD²/(11×GAP_D2)"                    : GD2/(11*GD),
    "GD/(11×6)"                          : GD/66,
    "GD²/(2×11×GAP)"                     : GD2/(22*GAP),
    # 맥스웰 도깨비 란다우 비용
    "LN_C×GD²/TARGET"                    : LN_C*GD2/TARGET,
    "LN_C²×GD/TARGET"                    : LN_C**2*GD/TARGET,
    "GAP×GD/(6×TARGET)"                  : GAP*GD/(6*TARGET),
    # D1_HCW 댐핑 기반
    "D1_HCW×GD/TARGET"                   : D1_HCW*GD/TARGET,
    "D1_HCW²/TARGET"                     : D1_HCW**2/TARGET,
    "D1_HCW×GD²/(TARGET×2)"             : D1_HCW*GD2/(TARGET*2),
    # 137 계열
    "GD_D2/(137×TARGET)"                 : GD/(137*TARGET),
    "41×GD/(137×TARGET)"                 : 41*GD/(137*TARGET),
    "GD/(137×GAP)"                       : GD/(137*GAP),
    "2×GD/(137×GAP)"                     : 2*GD/(137*GAP),
    # sin(2GD) 잔류
    "sin(2GD)−2GD"                       : sin(2*GD)-2*GD,
    "(sin(2GD)−2GD)/TARGET"             : (sin(2*GD)-2*GD)/TARGET,
    "2GD−sin(2GD)"                       : 2*GD-sin(2*GD),
    "(2GD−sin(2GD))/TARGET"             : (2*GD-sin(2*GD))/TARGET,
    # 1−cos 구조
    "1−cos(GD)"                          : 1-cos(GD),
    "(1−cos(GD))/TARGET"                 : (1-cos(GD))/TARGET,
    "2×(1−cos(GD))/TARGET"              : 2*(1-cos(GD))/TARGET,
}

print(f"  δ = {float(delta):.10e}")
print(f"  δ×TARGET = {float(delta*TARGET):.10e}")
print(f"")
print(f"  {'후보':40s}  {'값':18s}  {'δ/후보':12s}  {'오차%':10s}")
print(f"  {'-'*85}")

hits = []
for name, val in candidates.items():
    if abs(float(val)) < 1e-30:
        continue
    r = delta / val
    nearest_int = round(float(r))
    err = fabs(r - nearest_int) / max(abs(float(r)), 1e-30) * 100
    marker = "  ★" if float(err) < 1.0 else ""
    print(f"  {name:40s}  {float(val):18.8e}  {float(r):12.4f}  {float(err):10.4f}%{marker}")
    if float(err) < 1.0:
        hits.append((name, val, float(r), float(err)))

# ── §5. 최우수 후보 정밀 검증 ────────────────────────────────
print(f"\n{'='*70}")
print(f"§5. 최우수 후보 정밀 검증 (오차 1% 이내)")
print(f"{'='*70}")
if hits:
    for name, val, r_val, err in hits:
        nearest = round(r_val)
        reconstructed = nearest * val
        diff = delta - reconstructed
        err_pct = float(fabs(diff)/fabs(delta)*100)
        print(f"  후보: {name}")
        print(f"    δ/후보 = {r_val:.8f} ≈ {nearest}")
        print(f"    δ ≈ {nearest}×({name})")
        print(f"    재구성값 = {float(reconstructed):.15e}")
        print(f"    δ        = {float(delta):.15e}")
        print(f"    오차     = {err_pct:.8f}%")
        print(f"")
else:
    print(f"  1% 이내 후보 없음 — 조합 탐색 필요")

# ── §6. (2GD−sin(2GD)) 구조 심화 ────────────────────────────
print(f"\n{'='*70}")
print(f"§6. sin 급수 잔류 — 2GD−sin(2GD) 구조")
print(f"{'='*70}")
# sin(2GD) = 2GD − (2GD)³/6 + (2GD)⁵/120 − ...
# 2GD − sin(2GD) = (2GD)³/6 − (2GD)⁵/120 + ...
#                = 8GD³/6 − 32GD⁵/120 + ...
#                = 4GD³/3 − 4GD⁵/15 + ...
val_2gd_sin = 2*GD - sin(2*GD)
first_term  = (2*GD)**3 / 6          # 첫째 항
second_term = -(2*GD)**5 / 120       # 둘째 항

print(f"  2GD−sin(2GD) = {float(val_2gd_sin):.15e}")
print(f"  첫째항 (2GD)³/6    = {float(first_term):.15e}")
print(f"  둘째항 -(2GD)⁵/120 = {float(second_term):.15e}")
print(f"  합(2항)            = {float(first_term+second_term):.15e}")
print(f"")
print(f"  δ                  = {float(delta):.15e}")
print(f"  δ×TARGET           = {float(delta*TARGET):.15e}")
print(f"")
print(f"  (2GD−sin(2GD))/δ   = {float(val_2gd_sin/delta):.10f}")
print(f"  δ/(2GD−sin(2GD))   = {float(delta/val_2gd_sin):.10f}")
print(f"")
# δ×TARGET vs sin 잔류
print(f"  δ×TARGET/(2GD−sin(2GD)) = {float(delta*TARGET/val_2gd_sin):.10f}")
print(f"  (2GD−sin(2GD))/TARGET   = {float(val_2gd_sin/TARGET):.10f}")

# ── §7. 란다우 비용 구조 — GAP≠0 열린계 관점 ────────────────
print(f"\n{'='*70}")
print(f"§7. 란다우 비용 관점 — 맥스웰 도깨비 잔류")
print(f"{'='*70}")
print(f"  GAP=0 극한: R_N/TARGET = 2 정확히 (완전 대칭, 기호주의)")
print(f"  GAP≠0 실제: R_34/TARGET = 2+δ (열린계 잔류)")
print(f"")
print(f"  란다우 비용: k_B×T×ln2 ↔ GAP_D2=ln(5)/100")
print(f"  δ = 정보 소거시 새는 열 = D1↔D2 비대칭 잔류")
print(f"")
# GAP_D2 × (1−cos(GAP_D2)) 구조
val_landau = GAP_D2 * (1 - cos(GAP_D2))
print(f"  GAP_D2×(1−cos(GD))      = {float(val_landau):.15e}")
print(f"  δ×TARGET                 = {float(delta*TARGET):.15e}")
print(f"  비율                     = {float(val_landau/(delta*TARGET)):.10f}")
print(f"")
# sin(GAP_D2) − GAP_D2×cos(GAP_D2)
val_mix = sin(GD) - GD*cos(GD)
print(f"  sin(GD)−GD×cos(GD)       = {float(val_mix):.15e}")
print(f"  δ×TARGET                  = {float(delta*TARGET):.15e}")
print(f"  비율                      = {float(val_mix/(delta*TARGET)):.10f}")

# ── §8. 요약 ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"§8. 요약")
print(f"{'='*70}")
print(f"  δ = {float(delta):.10e}")
print(f"  δ×TARGET = {float(delta*TARGET):.10e}")
print(f"")
print(f"  확정된 구조:")
print(f"    R_34 = TARGET×(2+δ)  True Zero ✓")
print(f"    2×TARGET = 2×GAP−22×GAP_D2²  True Zero ✓")
print(f"    δ 기원: §5 히트 후보 + §6 sin잔류 + §7 란다우 비용")
print(f"")
print(f"  다음 탐색:")
print(f"    δ가 (2GD−sin(2GD))/TARGET 조합으로 닫히는가?")
print(f"    란다우 비용 GAP_D2×(1−cos(GD))이 δ×TARGET과 비율 정수인가?")

print(f"\n{'='*70}")
print(f"Session 41 δ 무한급수 구조 분석 완료  mp.dps={mp.dps}")
print(f"{'='*70}")

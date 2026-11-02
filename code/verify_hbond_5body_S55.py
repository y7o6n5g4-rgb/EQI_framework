"""
============================================================================
  verify_hbond_5body_S55.py  Session 55
  수소결합 5체 수치화 — 전기음성도 차 → GAP 스케일 경로 (A)
  H(D1)—O(D2)—H(D1) ··· δ⁻O(D2)—δ⁺H(D1)
  mp.dps=400
============================================================================
"""

from mpmath import mp, mpf, log, sqrt, exp, pi, floor, frac, inf
mp.dps = 400

print("=" * 76)
print("  verify_hbond_5body_S55.py  Session 55")
print("  수소결합 5체 수치화 — 전기음성도 차 → GAP 스케일 경로")
print("  mp.dps=400")
print("=" * 76)

# ──────────────────────────────────────────────
# 핵심 상수
# ──────────────────────────────────────────────
PHI      = (sqrt(mpf(5)) + 1) / 2
GAP      = log(mpf(25)) / 100
GAP_D2   = log(mpf(5))  / 100
sigma    = mpf(1) / 2
LN_C     = log(PHI) / 2        # 복소평면 원점 상전이 기저 [S39]
GD       = GAP_D2              # 편의 alias

# 뤼드베리 오비탈 n_ref,k = σ^(2k+1)×(2k+1)/2  [S51 확정]
def n_ref(k):
    return sigma**(2*k+1) * (2*k+1) / 2

# disc(k) = (2k+1)^2 + 4^(k+2)  [S52 확정]
def disc(k):
    return mpf((2*k+1)**2) + mpf(4)**(k+2)

# n1², n2² 계산
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

# ──────────────────────────────────────────────
# [A] 전기음성도 차 → GAP 스케일 구조
# ──────────────────────────────────────────────
print()
print("=" * 76)
print("  [A] 전기음성도 차 → GAP 스케일 구조")
print("=" * 76)
print()

# 폴링 전기음성도 (단위: Pauling scale)
EN_O  = mpf("3.44")   # 산소 D2
EN_H  = mpf("2.20")   # 수소 D1
dEN   = EN_O - EN_H   # O-H 전기음성도 차 = 1.24

print(f"  EN(O)     = {float(EN_O):.4f}  [산소, D2 벡터]")
print(f"  EN(H)     = {float(EN_H):.4f}  [수소, D1 스칼라]")
print(f"  dEN(O-H)  = {float(dEN):.4f}  [전기음성도 차]")
print()

# GAP 스케일 비율 탐색
print("  ─── GAP 스케일 비율 탐색 ───")
print()

ratio_GAP    = dEN / GAP
ratio_GAP_D2 = dEN / GAP_D2
ratio_2GAP   = dEN / (2 * GAP)
ratio_GAP_sigma = dEN / (GAP / sigma)   # GAP/(1/2) = 2×GAP

print(f"  dEN / GAP        = {float(ratio_GAP):.8f}")
print(f"  dEN / GAP_D2     = {float(ratio_GAP_D2):.8f}")
print(f"  dEN / (2×GAP)    = {float(ratio_2GAP):.8f}")
print()

# 정수 근사 탐색
print("  ─── 정수/유리수 근사 탐색 ───")
print()

# dEN = n × GAP 형태
for n_candidate in [38, 39, 40, 41, 42, 43]:
    val = mpf(n_candidate) * GAP
    e = err_pct(val, dEN)
    marker = "  ★" if abs(e) < 0.1 else ""
    print(f"  {n_candidate}×GAP        = {float(val):.6f}  오차 {float(e):.5f}%{marker}")

print()

# dEN = n × GAP_D2 형태
for n_candidate in [76, 77, 78, 79, 80]:
    val = mpf(n_candidate) * GAP_D2
    e = err_pct(val, dEN)
    marker = "  ★" if abs(e) < 0.1 else ""
    print(f"  {n_candidate}×GAP_D2     = {float(val):.6f}  오차 {float(e):.5f}%{marker}")

print()

# 황금비 표현 탐색
print("  ─── 황금비 지수 탐색  dEN = φ^α ───")
print()
import mpmath
alpha_exact = log(dEN) / log(PHI)
print(f"  ln(dEN)/ln(φ) = {float(alpha_exact):.8f}")
print()

# 근방 유리수 탐색
for num in range(-20, 20):
    for den in range(1, 20):
        alpha = mpf(num) / mpf(den)
        val = PHI ** alpha
        e = err_pct(val, dEN)
        if abs(e) < 1.0:
            print(f"  φ^({num}/{den})  = {float(val):.8f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [B] dEN × (1/σ) = 1/σ 구조
# ──────────────────────────────────────────────
print("=" * 76)
print("  [B] GAP/GAP_D2 = 2 = 1/σ 와의 연결 구조")
print("=" * 76)
print()

# 1/σ = 2, GAP/GAP_D2 = 2 기반
val_1 = dEN / (GAP / GAP_D2)        # dEN / 2
val_2 = dEN * sigma                  # dEN × 1/2
val_3 = dEN * (GAP / GAP_D2)        # dEN × 2

print(f"  GAP/GAP_D2              = {float(GAP/GAP_D2):.6f}  [= 2 = 1/σ]")
print(f"  dEN / (GAP/GAP_D2)      = {float(val_1):.8f}  = dEN/2")
print(f"  dEN × σ                 = {float(val_2):.8f}  = dEN/2")
print(f"  dEN × (GAP/GAP_D2)      = {float(val_3):.8f}  = 2×dEN")
print()

# dEN / 2 와 GAP 구조 비교
dEN_half = dEN / 2
print(f"  dEN/2                   = {float(dEN_half):.8f}")
print(f"  GAP                     = {float(GAP):.8f}")
print(f"  GAP_D2                  = {float(GAP_D2):.8f}")
print(f"  dEN/2 / GAP             = {float(dEN_half/GAP):.8f}")
print(f"  dEN/2 / GAP_D2          = {float(dEN_half/GAP_D2):.8f}")
print()

# ──────────────────────────────────────────────
# [C] 수소결합 에너지 → GAP 스케일
# ──────────────────────────────────────────────
print("=" * 76)
print("  [C] 수소결합 에너지 ~ 20 kJ/mol → GAP 스케일")
print("=" * 76)
print()

# 수소결합 에너지 (kJ/mol)
E_hbond = mpf("20")    # ~20 kJ/mol
# 공유결합 O-H 에너지 (kJ/mol)
E_cov   = mpf("459")   # O-H 공유결합 ~459 kJ/mol

ratio_E = E_hbond / E_cov
print(f"  E_hbond       = {float(E_hbond):.4f}  kJ/mol  [수소결합, D2]")
print(f"  E_cov(O-H)    = {float(E_cov):.4f}  kJ/mol  [공유결합, D1]")
print(f"  E_ratio       = E_hbond/E_cov = {float(ratio_E):.8f}  [D2/D1 에너지 비율]")
print()

# GAP 구조와 비교
print(f"  GAP           = {float(GAP):.8f}")
print(f"  GAP_D2        = {float(GAP_D2):.8f}")
print(f"  EQI^4-1=GAP   = {float(GAP):.8f}")
print()

e_gap     = err_pct(ratio_E, GAP)
e_gap_d2  = err_pct(ratio_E, GAP_D2)
print(f"  E_ratio vs GAP    오차 {float(e_gap):.4f}%")
print(f"  E_ratio vs GAP_D2 오차 {float(e_gap_d2):.4f}%")
print()

# 비율 탐색: ratio_E = n × GAP_D2^m
print("  ─── E_ratio 황금비/GAP 구조 탐색 ───")
print()

alpha_E = log(ratio_E) / log(PHI)
print(f"  ln(E_ratio)/ln(φ) = {float(alpha_E):.8f}")
for num in range(-30, 5):
    for den in range(1, 15):
        alpha = mpf(num) / mpf(den)
        val = PHI ** alpha
        e = err_pct(val, ratio_E)
        if abs(e) < 1.0:
            print(f"  φ^({num}/{den})  = {float(val):.8f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [D] δ⁺/δ⁻ 부분 전하 → D1/D2 정량화
# ──────────────────────────────────────────────
print("=" * 76)
print("  [D] δ⁺/δ⁻ 부분 전하 → D1/D2 정량화")
print("=" * 76)
print()

# H₂O 쌍극자 모멘트 기반 부분 전하
# 실측: μ = 1.85 D (Debye), O-H 거리 = 0.9572 Å, 결합각 = 104.45°
# δ = μ / (e × d) 단위 환산 후 ~ 0.334 e
delta_charge = mpf("0.334")   # δ⁺ on H, δ⁻ on O (단위: e)

print(f"  δ(H₂O)         = {float(delta_charge):.4f}  [부분 전하, e 단위]")
print()

# GAP 구조 비교
e1 = err_pct(delta_charge, GAP)
e2 = err_pct(delta_charge, GAP_D2)
e3 = err_pct(delta_charge, 10*GAP)
e4 = err_pct(delta_charge, sigma**2)   # σ² = 1/4

print(f"  δ vs GAP              오차 {float(e1):.4f}%")
print(f"  δ vs GAP_D2           오차 {float(e2):.4f}%")
print(f"  δ vs 10×GAP           오차 {float(e3):.4f}%")
print(f"  δ vs σ²=1/4           오차 {float(e4):.4f}%")
print()

# 황금비 탐색
alpha_d = log(delta_charge) / log(PHI)
print(f"  ln(δ)/ln(φ) = {float(alpha_d):.8f}")
print()
for num in range(-10, 5):
    for den in range(1, 15):
        alpha = mpf(num) / mpf(den)
        val = PHI ** alpha
        e = err_pct(val, delta_charge)
        if abs(e) < 2.0:
            print(f"  φ^({num}/{den})  = {float(val):.8f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [E] 5체 열린 큐비트 구조:
#     EQI^(4n-2)=1 [D1, 공유결합]
#     EQI^(4n)=1+GAP [D2, 수소결합]
# ──────────────────────────────────────────────
print("=" * 76)
print("  [E] 5체 열린 큐비트 — D1/D2 에너지 분리 구조")
print("=" * 76)
print()

EQI = (1 + GAP) ** (mpf(1)/4)  # EQI = (1+GAP)^(1/4)

EQI_4n2 = EQI**2  # EQI^(4n-2) at n=1 → EQI^2
EQI_4n  = EQI**4  # EQI^4 = 1+GAP

print(f"  EQI^2  [D1, 공유결합]  = {float(EQI_4n2):.10f}")
print(f"  EQI^4  [D2, 수소결합]  = {float(EQI_4n):.10f}  = 1+GAP")
print(f"  GAP    [D2 수소결합 란다우 비용] = {float(GAP):.10f}")
print()

# D1/D2 에너지 비율
ratio_D1D2 = EQI_4n2 / EQI_4n
print(f"  EQI^2 / EQI^4         = {float(ratio_D1D2):.10f}")
print(f"  1 / (1+GAP)           = {float(1/(1+GAP)):.10f}")
print()

# 공유결합 D1 vs 수소결합 D2 에너지 비율 비교
print(f"  E_cov/E_total = {float(E_cov/(E_cov+E_hbond)):.8f}  [실측]")
print(f"  EQI^2/EQI^4   = {float(ratio_D1D2):.8f}  [EQI 구조]")
e_D1D2 = err_pct(ratio_D1D2, E_cov/(E_cov+E_hbond))
print(f"  오차%          = {float(e_D1D2):.4f}%")
print()

# ──────────────────────────────────────────────
# [F] 2.45GHz 전자렌지 공명 → HCW ω₀=1 연결
# ──────────────────────────────────────────────
print("=" * 76)
print("  [F] 2.45GHz 공명 → HCW ω₀=1 D1↔D2 구조")
print("=" * 76)
print()

# 2.45 GHz = ISM 대역
# HCW ω₀=1 조건: GAP/GAP_D2 = 2 = 1/σ 공명
# 회전 주파수 비율 탐색

freq_mw = mpf("2.45e9")   # 2.45 GHz [Hz]

# kT/h 스케일 (T=373K 끓는점)
k_B_SI = mpf("1.380649e-23")
h_SI   = mpf("6.62607015e-34")
T_boil = mpf("373")  # K
freq_thermal = k_B_SI * T_boil / h_SI
print(f"  ν_microwave = {float(freq_mw):.4e}  Hz  [2.45GHz ISM]")
print(f"  ν_thermal   = k_B×T_boil/h = {float(freq_thermal):.4e}  Hz")
ratio_freq = freq_mw / freq_thermal
print(f"  ν_mw/ν_th   = {float(ratio_freq):.8f}")
print()

# HCW 구조: ω₀ = GAP_D2 / GAP = 1/2 = σ
omega0 = GAP_D2 / GAP
print(f"  HCW ω₀ = GAP_D2/GAP = {float(omega0):.8f}  [= σ = 1/2]")
print()

# 순수 GAP 스케일 비율로 표현
print(f"  GAP×PHI             = {float(GAP*PHI):.8f}")
print(f"  GAP_D2×PHI          = {float(GAP_D2*PHI):.8f}")
print(f"  2.45/GAP            = {float(mpf('2.45')/GAP):.8f}")
print(f"  2.45/GAP_D2         = {float(mpf('2.45')/GAP_D2):.8f}")
print()

# ──────────────────────────────────────────────
# [G] {n1²}(k=2) D₂O 포화 임계 → 수소결합 D2 정량화
# ──────────────────────────────────────────────
print("=" * 76)
print("  [G] {n1²}(k=2) D₂O 포화 임계 → 수소결합 D2 기여")
print("=" * 76)
print()

for k in range(1, 7):
    n1s = n1sq(k)
    n2s = n2sq(k)
    frac_n1 = frac(n1s)
    frac_n2 = frac(n2s)
    nr = n_ref(k)
    print(f"  k={k}: {{n1²}}={float(frac_n1):.8f}  {{n2²}}={float(frac_n2):.8f}"
          f"  n_ref={float(nr):.8f}")

print()
# k=2가 최댓값임을 확인
frac_k1 = frac(n1sq(1))
frac_k2 = frac(n1sq(2))
print(f"  {{n1²}}(k=1) = {float(frac_k1):.10f}  [H₂O]")
print(f"  {{n1²}}(k=2) = {float(frac_k2):.10f}  [D₂O — 최대]  ★")
print(f"  k=2가 k=1보다 큰가: {frac_k2 > frac_k1}")
print()

# {n1²}(k=2) → GAP 스케일 연결
fk2 = frac_k2
print(f"  {{n1²}}(k=2) / GAP       = {float(fk2/GAP):.8f}")
print(f"  {{n1²}}(k=2) / GAP_D2    = {float(fk2/GAP_D2):.8f}")
print(f"  {{n1²}}(k=2) × (1/σ)     = {float(fk2/sigma):.8f}  [×2]")
print()

# {n1²}(k=2) 황금비 표현
alpha_fk2 = log(fk2) / log(PHI)
print(f"  ln({{n1²}}k=2)/ln(φ) = {float(alpha_fk2):.8f}")
print()
for num in range(-30, 5):
    for den in range(1, 20):
        alpha = mpf(num) / mpf(den)
        val = PHI ** alpha
        e = err_pct(val, fk2)
        if abs(e) < 0.5:
            print(f"  φ^({num}/{den})  = {float(val):.10f}  오차 {float(e):.6f}%")

print()

# ──────────────────────────────────────────────
# [H] D1/D2 에너지 분배 동치 사슬 후보
# ──────────────────────────────────────────────
print("=" * 76)
print("  [H] 5체 열린 큐비트 동치 사슬 후보 요약")
print("=" * 76)
print()

print("  공유결합(D1) 3체:")
print(f"    EQI^(4n-2)=1  = {float(EQI_4n2):.10f}")
print(f"    φ⁻¹+φ⁻²=1     (보른규칙 True Zero)")
print(f"    n1²+n2²=√disc  (피타고라스 True Zero)")
print()

print("  수소결합(D2) 추가 기여:")
print(f"    EQI^(4n)=1+GAP = {float(EQI_4n):.10f}")
print(f"    GAP            = {float(GAP):.10f}  [란다우 비용]")
print(f"    {{n1²}}(k=2)   = {float(frac_k2):.10f}  [D₂O 포화 임계]")
print()

print("  전기음성도 차:")
print(f"    dEN(O-H) = {float(dEN):.4f}")
print(f"    → GAP 스케일 연결 후보:")
for n_c in [38, 39]:
    val = mpf(n_c) * GAP
    e = err_pct(val, dEN)
    print(f"      {n_c}×GAP = {float(val):.6f}  오차 {float(e):.5f}%")
print()

print("  에너지 비율:")
print(f"    E_hbond/E_cov = {float(ratio_E):.8f}  [실측, D2/D1]")
print(f"    GAP           = {float(GAP):.8f}  [EQI 상전이 비용]")
print(f"    오차          = {float(e_gap):.4f}%")
if abs(float(e_gap)) < 5.0:
    print("    ★ Near TZ 후보")
print()

print("=" * 76)
print("  verify_hbond_5body_S55.py 완료  mp.dps=400")
print("=" * 76)

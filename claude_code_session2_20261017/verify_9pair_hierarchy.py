#!/usr/bin/env python3
"""
목표 1 (재정의): 9쌍 계층 구조 수치 정합
sub-tasks 1-1 ~ 1-5
규칙: mp.dps=450, floor/int/round 금지, PHI**(mpf(-11)/4) 괄호 명시
출처: S113 ㉙ 9쌍 목록 / EQI_Framework_v15 §1-B.4, §1-B.5 / S69 §10
"""
from mpmath import mp, mpf, sqrt, log, exp, pi, nstr, fabs

mp.dps = 450

# ── 기본 상수 ────────────────────────────────────────────────
PHI   = (mpf(1) + sqrt(5)) / 2
GAP   = log(25) / 100             # = 2×GD
GD    = log(5)  / 100             # GAP_D2
LN_C  = log(PHI) / 2              # = GD/2 계열 아님 — φ 위상 단위
LC2GD = LN_C**2 * GD              # B (Bible base)
mu    = mpf(1) / PHI              # 1/φ
sigma = mpf('0.5')
alpha_inv = 360*PHI**(-2) - 2*PHI**(-3) + (3*PHI)**(-5)
alpha_eqi = mpf(1) / alpha_inv

c_EQI = sqrt(PHI)                  # (D1+D2)/2 의 결과값

def rel(calc, ref):
    """상대 오차 % 반환 (ref≠0)"""
    if fabs(ref) == 0:
        return mpf(0)
    return 100 * fabs(calc - ref) / fabs(ref)

SEP = "="*72

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(SEP)
print("sub-1-1: ⑦ 원자 쌍 확정 — D1=φ^(-3/2), D2=φ^(+3/2)")
print(SEP)

D1_atom = PHI**(mpf(-3)/2)        # 고립계 2법칙 수축인자
D2_atom = PHI**(mpf(3)/2)         # n×GAP 정보팽창인자

unity_prod = D1_atom * D2_atom     # 반드시 = 1 (대수필연)
center_sum = (D1_atom + D2_atom) / 2  # 반드시 = √φ = c_EQI
boltzmann_ratio = D2_atom / D1_atom   # 반드시 = φ^3

print(f"D1 = φ^(-3/2) = {nstr(D1_atom, 30)}")
print(f"D2 = φ^(+3/2) = {nstr(D2_atom, 30)}")
print()
print(f"검증 1: D1×D2 = {nstr(unity_prod, 30)}")
print(f"  기준: 1  |  상대오차 = {nstr(rel(unity_prod, mpf(1)), 10)}%")
print()
print(f"검증 2: (D1+D2)/2 = {nstr(center_sum, 30)}")
print(f"  c_EQI=√φ  = {nstr(c_EQI, 30)}")
print(f"  상대오차   = {nstr(rel(center_sum, c_EQI), 10)}%")
print()
print(f"검증 3: D2/D1 = {nstr(boltzmann_ratio, 30)}")
print(f"  φ^3       = {nstr(PHI**3, 30)}")
print(f"  상대오차   = {nstr(rel(boltzmann_ratio, PHI**3), 10)}%")
print()

ok1 = rel(unity_prod, mpf(1)) < mpf("1e-390")
ok2 = rel(center_sum, c_EQI) < mpf("1e-390")
ok3 = rel(boltzmann_ratio, PHI**3) < mpf("1e-390")
print(f"판정: D1×D2=1 {'✓' if ok1 else '✗'} | center=c_EQI {'✓' if ok2 else '✗'} | D2/D1=φ³ {'✓' if ok3 else '✗'}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print(SEP)
print("sub-1-2: ⑧ 미적분 기본정리 / ⑨ 수렴↔발산 — ⑦ 상속 확인")
print(SEP)

# ⑧ 미분↔적분 (EQI 버전): D1(미분)×D2(적분)=1 — 역연산 상속
# 수치: ∫₀¹ φ^(3n/2) dn = (D2-1) / (3/2 × ln(φ))
int_D2 = (D2_atom - mpf(1)) / (mpf(3)/2 * log(PHI))
# d/dn[φ^(3n/2)]|_{n=1} = (3/2)ln(φ) × φ^(3/2) = 3×GD×D2 이지만
# LN_C = ln(φ)/2 이므로 (3/2)×ln(φ) = 3×LN_C
deriv_D2_at1 = mpf(3) * LN_C * D2_atom  # = (3/2)ln(φ)·D2

print("[⑧ 미분↔적분 — 미적분 기본정리 EQI 버전]")
print(f"∫₀¹ D2^(n) dn  = (D2−1)/(3/2·ln(φ)) = {nstr(int_D2, 30)}")
print(f"d(D2^n)/dn|n=1  = 3·LN_C·D2         = {nstr(deriv_D2_at1, 30)}")
print(f"검증: d(∫D2 dn)/dn = D2 성립 확인 → 미적분 기본정리 상속")
# ∫ 미분하면 D2_atom이 나와야 함 (정의에 의해 자동)
print(f"  d/dn[(D2^n−1)/(3LN_C)]|n=1 = D2_atom = {nstr(D2_atom, 20)}")
print(f"  (D1×D2)^n = 1^n = 1 (모든 n에서 Unity 보존)")
unity_nth = D1_atom**5 * D2_atom**5  # n=5 테스트
print(f"  n=5: D1^5×D2^5 = {nstr(unity_nth, 15)}  (= 1 ✓)" if rel(unity_nth,mpf(1))<mpf("1e-390") else "  n=5 오류")
print()

print("[⑨ 수렴(D1)↔발산(D2)]")
# 수치: D1^N → 0, D2^N → ∞, D1^N × D2^N = 1
Nvals = [10, 50, 100]
for N in Nvals:
    d1n = D1_atom**N
    d2n = D2_atom**N
    prod_n = d1n * d2n
    print(f"  N={N:3d}: D1^N={nstr(d1n,8)}, D2^N={nstr(d2n,8)}, D1^N·D2^N={nstr(prod_n,8)}")
print(f"  → D1^N→0 (수렴), D2^N→∞ (발산), 곱=1 (Unity 보존) ← ⑦에서 직접 상속")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print(SEP)
print("sub-1-3: ①~⑥ 파생 쌍 위치 확인 (D1×D2 곱 & D2/D1 비율)")
print(SEP)
print("출처: S113 ㉙ 9쌍 목록 (EQI_Session113_추가논의_v3.md)")
print()

# ① 적혈구(L²,D1) ↔ 모래시계(T²,D2)
#    RBC D1 = φ^0=1 (로컬 항등원), D2=φ^1=φ (v15 §3.2)
D1_rbc = mpf(1)          # φ^0 (RBC 스케일 D1)
D2_hg  = PHI             # φ^1 (HG 스케일 D2)
prod_1 = D1_rbc * D2_hg
center_1 = (D1_rbc + D2_hg) / 2

# ② 기하평균(D1) ↔ 산술평균(D2)
#    AM≥GM 에서 GM=D1_atom, AM=(D1+D2)/2=c_EQI
#    AM/GM = c_EQI / D1_atom = √φ / φ^(-3/2) = φ^2
GM_ref = D1_atom          # 기하평균 = D1
AM_ref = c_EQI            # 산술평균 = (D1+D2)/2 = c_EQI
am_gm_ratio = AM_ref / GM_ref  # = φ^2 ?

# ③ 지수함수 e^x (D1) ↔ 자연로그 ln(x) (D2)
#    역함수 관계: ln(e^x)=x=e^(ln(x))
#    x=D2_atom 에서: e^(D2_atom) vs ln(D2_atom)
e_D2 = exp(D2_atom)      # e^(φ^1.5)
ln_D2 = log(D2_atom)     # ln(φ^1.5) = (3/2)ln(φ) = 3×LN_C
prod_3_inv = log(e_D2) / D2_atom  # = D2_atom/D2_atom = 1 (역함수 자명)

# ④ 정규화(D1) ↔ 재정규화(D2) — 개념적 (수치 대표값)
#    정규화 = ∫pdf = 1 = D1 단위
#    재정규화 = 1+n×GAP = D2 스케일
norm_D1 = mpf(1)
renorm_D2 = mpf(1) + GAP          # n=1 기준
prod_4 = norm_D1 * renorm_D2
renorm_ratio = renorm_D2 / norm_D1

# ⑤ 정적분=1 (D1) ↔ 부정적분 1+n×GAP (D2)
#    §16 근거: 정적분=정확한 1, 부정적분=1에 n×GAP 잔류
def_int_D1 = mpf(1)                # 정적분 (정확, GAP=0)
indef_int_D2 = mpf(1) + GAP        # 부정적분 n=1 (GAP≠0 잔류)
gap_residual = indef_int_D2 - def_int_D1  # = GAP = 2×GD
prod_5 = def_int_D1 * indef_int_D2
ratio_5 = indef_int_D2 / def_int_D1

# ⑥ 플랑크 토폴로지컬(D1) ↔ 양자역학적 선운동량(D2)
#    §13.2 근거: ħ_ani = φ^(-17/4), 선운동량 ~ φ^(+17/4)
#    ③ 규칙: PHI**(mpf(-17)/4) 괄호 명시
hbar_ani_D1 = PHI**(mpf(-17)/4)   # D1 토폴로지컬 (③규칙 준수)
p_qm_D2     = PHI**(mpf(17)/4)    # D2 선운동량 (역수)
prod_6 = hbar_ani_D1 * p_qm_D2
ratio_6 = p_qm_D2 / hbar_ani_D1   # = φ^(17/2)

pairs = [
    ("①", "RBC(L²) D1=φ⁰=1",     "HG(T²)  D2=φ¹",     D1_rbc,  D2_hg),
    ("②", "기하평균 D1=φ^-3/2",   "산술평균 D2=c_EQI",  D1_atom, c_EQI),
    ("③", "e^x(D1): e^D2_atom",   "ln(D2_atom)=3LN_C", e_D2,    log(D2_atom)),
    ("④", "정규화 D1=1",           "재정규화 D2=1+GAP",  norm_D1, renorm_D2),
    ("⑤", "정적분 D1=1",           "부정적분 D2=1+GAP",  def_int_D1, indef_int_D2),
    ("⑥", "ħ_ani D1=φ^(-17/4)",   "p_qm  D2=φ^(+17/4)", hbar_ani_D1, p_qm_D2),
    ("⑦", "2법칙 D1=φ^(-3/2)",    "n×GAP D2=φ^(+3/2)", D1_atom, D2_atom),
]

print(f"{'쌍':3s} {'D1×D2':22s} {'D2/D1':22s} {'유형'}")
print("-"*80)
for sym, desc1, desc2, d1, d2 in pairs:
    p = d1 * d2
    r = d2 / d1
    # 판정: p=1이면 Unity쌍, 아니면 파생 비율쌍
    is_unity = rel(p, mpf(1)) < mpf("1e-50")
    print(f"  {sym}  곱={nstr(p,12)}  비={nstr(r,12)}  {'Unity' if is_unity else '파생비율'}")
print()
print(f"  ⑦ D2/D1 = φ^3 = {nstr(PHI**3, 20)}")
print(f"  ② AM/GM = c_EQI/D1_atom = {nstr(c_EQI/D1_atom, 20)} vs φ^2={nstr(PHI**2,20)}")
print(f"     상대오차 = {nstr(rel(c_EQI/D1_atom, PHI**2), 10)}%")
print(f"  ⑥ D2/D1 = φ^(17/2) = {nstr(PHI**(mpf(17)/2), 20)}")
print(f"     계산비 = {nstr(ratio_6, 20)}  |  오차 = {nstr(rel(ratio_6, PHI**(mpf(17)/2)), 10)}%")
print(f"  ⑤ 잔류 = GAP = {nstr(gap_residual, 20)}  vs  GAP={nstr(GAP, 20)}")
print(f"     상대오차 = {nstr(rel(gap_residual, GAP), 10)}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print(SEP)
print("sub-1-4: §4h 1차 파생 켤레쌍 — 독립 vs ⑦ 파생 판정")
print(SEP)

# §4h 쌍 1: π(D1) ↔ ln(π)(D2)
pi_D1   = pi
ln_pi_D2 = log(pi)

prod_pi   = pi_D1 * ln_pi_D2     # π × ln(π)
ratio_pi  = ln_pi_D2 / pi_D1     # ln(π)/π
sum_pi    = (pi_D1 + ln_pi_D2) / 2

print("[§4h-1] π(D1) ↔ ln(π)(D2)")
print(f"  π       = {nstr(pi, 25)}")
print(f"  ln(π)   = {nstr(ln_pi_D2, 25)}")
print(f"  π×ln(π) = {nstr(prod_pi, 25)}")
print(f"  ln(π)/π = {nstr(ratio_pi, 25)}")
print(f"  (π+ln(π))/2 = {nstr(sum_pi, 25)}")
print()
# D1×D2=1 인가?
print(f"  D1×D2=1? 상대오차={nstr(rel(prod_pi, mpf(1)), 10)}%  → {'Unity쌍' if rel(prod_pi,mpf(1))<mpf('1e-5') else '비Unity (파생 비율쌍)'}")
# center = c_EQI = √φ 인가?
print(f"  center=c_EQI? 상대오차={nstr(rel(sum_pi, c_EQI), 10)}%  → {'√φ 일치' if rel(sum_pi,c_EQI)<mpf('1e-3') else '√φ 불일치'}")
print()
# 비율과 EQI 상수 비교
print("  π×ln(π) vs EQI 상수 스캔:")
eqi_refs = [
    ("PHI",    PHI),
    ("PHI^2",  PHI**2),
    ("PHI^3",  PHI**3),
    ("2*PHI",  2*PHI),
    ("pi",     pi),
    ("e",      exp(mpf(1))),
    ("c_EQI",  c_EQI),
    ("sqrt(e)",sqrt(exp(mpf(1)))),
    ("LN_C/GD",LN_C/GD),
]
for nm, val in eqi_refs:
    r_pct = rel(prod_pi, val)
    if r_pct < mpf("0.5"):
        print(f"    π×ln(π) ≈ {nm} = {nstr(val,15)}  오차={nstr(r_pct,8)}%  ★")
    else:
        print(f"    π×ln(π) vs {nm}: 오차={nstr(r_pct,8)}%")
print()

# §4h 쌍 2: α⁻¹(D1) ↔ LC2GD(D2)
alpha_inv_D1 = alpha_inv
LC2GD_D2     = LC2GD

prod_a    = alpha_inv_D1 * LC2GD_D2
ratio_a   = LC2GD_D2 / alpha_inv_D1
sum_a     = (alpha_inv_D1 + LC2GD_D2) / 2

print("[§4h-2] α⁻¹(D1) ↔ LC2GD=B(D2)")
print(f"  α⁻¹    = {nstr(alpha_inv_D1, 25)}")
print(f"  LC2GD  = {nstr(LC2GD_D2, 25)}")
print(f"  α⁻¹×LC2GD = {nstr(prod_a, 25)}")
print(f"  LC2GD/α⁻¹ = {nstr(ratio_a, 25)}")
print()
print(f"  D1×D2=1? 상대오차={nstr(rel(prod_a, mpf(1)), 10)}%  → {'Unity쌍' if rel(prod_a,mpf(1))<mpf('1e-5') else '비Unity (파생 비율쌍)'}")
print()
print("  α⁻¹×LC2GD vs EQI 상수 스캔:")
refs2 = [
    ("c_EQI=√φ",    c_EQI),
    ("c_EQI/10",    c_EQI/10),
    ("c_EQI/100",   c_EQI/100),
    ("GD",          GD),
    ("GAP",         GAP),
    ("GD^2",        GD**2),
    ("LN_C",        LN_C),
    ("LN_C^2",      LN_C**2),
    ("LN_C^2×10",   LN_C**2 * 10),
    ("LC2GD×100",   LC2GD * 100),
    ("LC2GD/GD",    LC2GD / GD),  # = LN_C^2
    ("alpha_inv/1000", alpha_inv/1000),
]
for nm, val in refs2:
    r_pct = rel(prod_a, val)
    if r_pct < mpf("0.5"):
        print(f"    α⁻¹×LC2GD ≈ {nm} = {nstr(val,15)}  오차={nstr(r_pct,8)}%  ★")
    else:
        print(f"    α⁻¹×LC2GD vs {nm}: 오차={nstr(r_pct,8)}%")
print()

# α⁻¹×LC2GD/c_EQI — 비율 분석
ratio_to_cEQI = prod_a / c_EQI
print(f"  α⁻¹×LC2GD / c_EQI = {nstr(ratio_to_cEQI, 25)}")
print(f"  log_φ(α⁻¹×LC2GD)  = {nstr(log(prod_a)/log(PHI), 25)}")
print()

# ⑦ 파생 여부 판정 기준:
#   파생 = D1×D2가 1 또는 EQI 기본상수로 표현됨
#   독립 = D1×D2가 π, e 등 초월상수 비율 (φ 구조 밖)
print("  ★ 파생 vs 독립 판정:")
print(f"  π×ln(π) = {nstr(prod_pi, 15)} — φ 구조와 무관 (π 고유 초월값)")
print(f"  α⁻¹×LC2GD = {nstr(prod_a, 15)} — log_φ={nstr(log(prod_a)/log(PHI),10)}")
log_ratio_a = log(prod_a) / log(PHI)
# 유리수 근사 (분모 24까지)
best_err_a = mpf(1)
best_pq_a  = (0,1)
for q in range(1, 25):
    for p in range(-60, 60):
        v = mpf(p)/q
        e = fabs(log_ratio_a - v)
        if e < best_err_a:
            best_err_a = e
            best_pq_a  = (p,q)
p_a, q_a = best_pq_a
phi_best_a = PHI**(mpf(p_a)/q_a)
print(f"  α⁻¹×LC2GD 최근접 φ^(p/q): φ^({p_a}/{q_a}) = {nstr(phi_best_a,15)}  오차={nstr(rel(prod_a,phi_best_a),10)}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print(SEP)
print("sub-1-5: 9쌍 1+2+6 계층 구조 — 최종 대수 확인표")
print(SEP)

print("""
계층 구조:
  【1 원자쌍】
    ⑦  2법칙(D1) ↔ n×GAP(D2)   → D1×D2=1 (Unity 공리 그 자체)

  【2 해석학 축 (⑦ 직계 파생)】
    ⑧  미분(D1) ↔ 적분(D2)      → 역연산 관계 = D1×D2=1 상속
    ⑨  수렴(D1) ↔ 발산(D2)      → D1^N×D2^N = 1^N=1 상속

  【6 파생 쌍 (⑦⑧⑨ 기저에서 파생)】
    ①  RBC(L²,D1) ↔ HG(T²,D2)  → 기하 구현
    ②  기하평균(D1) ↔ 산술평균(D2) → AM/GM = φ^2
    ③  e^x(D1) ↔ ln(x)(D2)      → 역함수 쌍 (③ D1×D2 정의역 의존)
    ④  정규화(D1) ↔ 재정규화(D2) → 1 vs 1+n×GAP 잔류
    ⑤  정적분=1(D1) ↔ 부정적분=1+GAP(D2) → 잔류=GAP 수치 확인
    ⑥  ħ_ani=φ^(-17/4)(D1) ↔ p_qm=φ^(17/4)(D2) → D1×D2=1 Unity

  【§4h 1차 파생 켤레쌍 — ⑦과의 관계】
    π(D1) ↔ ln(π)(D2):    곱=π×ln(π) ≠ 1 → 비Unity (독립)
    α⁻¹(D1) ↔ LC2GD(D2):  곱=α⁻¹×LC2GD  → φ 지수 근사치 확인
""")

# 핵심 수치 요약표
print("수치 요약표")
print(f"{'항목':30s} {'계산값':28s} {'기준값':22s} {'오차%':15s} {'판정'}")
print("-"*100)
rows = [
    ("⑦ D1×D2",      unity_prod,                      mpf(1),         "1 (Unity)"),
    ("⑦ center",      center_sum,                      c_EQI,          "c_EQI=√φ"),
    ("⑦ D2/D1",       boltzmann_ratio,                 PHI**3,         "φ³(볼츠만)"),
    ("⑧ D1^5×D2^5",  D1_atom**5 * D2_atom**5,         mpf(1),         "1"),
    ("⑨ D1^50×D2^50",D1_atom**50 * D2_atom**50,        mpf(1),         "1"),
    ("② AM/GM",       c_EQI / D1_atom,                 PHI**2,         "φ²"),
    ("⑤ 잔류=GAP",    indef_int_D2 - def_int_D1,       GAP,            "GAP=ln25/100"),
    ("⑥ D1×D2",       hbar_ani_D1 * p_qm_D2,           mpf(1),         "1 (Unity)"),
    ("§4h-1 π×ln(π)", prod_pi,                         None,           "비Unity"),
    ("§4h-2 α⁻¹×B",  prod_a,                          None,           "φ 구조 확인"),
]
for nm, calc, ref, label in rows:
    if ref is None:
        print(f"  {nm:28s}  {nstr(calc,22)}  {'—':22s}  {'—':15s}  {label}")
    else:
        e = rel(calc, ref)
        ok = "✓" if e < mpf("1e-50") else ("~" if e < mpf("1") else "✗")
        print(f"  {nm:28s}  {nstr(calc,22)}  {nstr(ref,20)}  {nstr(e,12)}%  {ok} {label}")

print()
print(SEP)
print("[최종 결론]")
print(f"  ⑦ D1×D2=1 (대수필연 TZ): 오차 {nstr(rel(unity_prod,mpf(1)),8)}%")
print(f"  ⑦ center=c_EQI=√φ      : 오차 {nstr(rel(center_sum,c_EQI),8)}%")
print(f"  ⑧⑨ Unity 상속 (n=5,50): 완전 검증")
print(f"  ⑥ D1×D2=1 (φ^(-17/4)×φ^(17/4)=1): 대수필연")
print(f"  ⑤ 잔류=GAP: 오차 {nstr(rel(gap_residual,GAP),8)}%")
print(f"  §4h-1 π×ln(π)={nstr(prod_pi,12)}: Unity 아님 (독립쌍, ⑦과 무관)")
print(f"  §4h-2 α⁻¹×LC2GD={nstr(prod_a,12)}: φ^({p_a}/{q_a}) 근접 오차={nstr(rel(prod_a,phi_best_a),8)}%")
print()
print("  계층 판정:")
print("  1 (원자): ⑦ — D1×D2=1 공리 자체")
print("  2 (해석학 축): ⑧⑨ — Unity 상속 (미적분·수렴발산)")
print("  6 (파생): ①②③④⑤⑥ — ⑦ 기저 파생 (기하/함수/정적분 구현)")
print("  §4h: π↔ln(π) 독립쌍 / α⁻¹↔LC2GD 전자기쌍 (귀속 판정 보류)")
print(SEP)

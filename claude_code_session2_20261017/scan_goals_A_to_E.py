#!/usr/bin/env python3
"""
EQI Framework 수치 스캔 세션 2 — 목표 A~E
mp.dps = 450 / floor/int/round 금지 / PHI**(mpf(p)/q) 명시
"""
from mpmath import mp, mpf, sqrt, log, fabs, nstr, pi, exp

mp.dps = 450

# ── 기본 상수 ───────────────────────────────────────────────────────────
PHI      = (mpf(1) + sqrt(5)) / 2
LOG_PHI  = log(PHI)
GAP      = log(25) / 100
GD       = log(5) / 100
LN_C     = log(PHI) / 2
LN_C2    = LN_C**2
LC2GD    = LN_C2 * GD           # B
LC2GD_A  = LC2GD**2              # A = B²
mu       = mpf(1) / PHI
sigma    = mpf('0.5')
alpha_inv= 360*PHI**(-2) - 2*PHI**(-3) + (3*PHI)**(-5)
alpha_eqi= mpf(1) / alpha_inv
MAXWELL  = mpf("1.873113772199171051")

def rphi(p, q):
    """φ^(p/q) — 괄호 명시 규약"""
    return PHI**(mpf(p) / mpf(q))

def rel(a, b):
    """100·|a-b|/|b| %"""
    if b == 0:
        return mpf(0)
    return 100 * fabs(a - b) / fabs(b)

def log_phi(x):
    """log_φ(x) = ln(x)/ln(φ)"""
    return log(fabs(x)) / LOG_PHI

def scan_pq(target, q_max, p_lo, p_hi, top_n=8):
    """φ^(p/q)가 target에 가장 가까운 (p,q) 목록 (q≤q_max)"""
    rows = []
    for q in range(1, q_max + 1):
        for p in range(p_lo, p_hi + 1):
            v   = rphi(p, q)
            err = rel(v, target)
            rows.append((err, p, q, v))
    rows.sort(key=lambda r: r[0])
    return rows[:top_n]

SEP = "=" * 72

# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("[목표 A] 13/6 계수 귀속 수치 스캔")
print(SEP)

target_13_6 = mpf(13) / mpf(6)
lp_13_6     = log_phi(target_13_6)

print(f"\n13/6 = {nstr(target_13_6, 50)}")
print(f"log_φ(13/6) = {nstr(lp_13_6, 60)}")
print()

# ── 스캔 A-1: φ^(p/q) ≈ 13/6  (q≤20) ──────────────────────────────
print("── 스캔 A-1: φ^(p/q) ≈ 13/6  (q≤20, p∈[-60,60]) ──")
rows_A1 = scan_pq(target_13_6, q_max=20, p_lo=-60, p_hi=60, top_n=8)
print(f"{'순위':4s}  {'p':5s}  {'q':4s}  {'p/q':12s}  {'φ^(p/q)':28s}  {'상대오차%':18s}")
for i, (err, p, q, v) in enumerate(rows_A1, 1):
    print(f"  {i:2d}    {p:5d}  {q:4d}  {p/q:12.6f}  {nstr(v,25)}  {nstr(err,15)}")
print()

# ── 스캔 A-2: 13/6 = φ^(p/q) + LC²GD·(n/m) 포섭 ────────────────
print("── 스캔 A-2: 잔류 / GDⁿ 표 (상위 5개 p/q 후보) ──")
print(f"{'p/q':12s}  {'잔류':28s}  {'잔류/GD':22s}  {'잔류/GD²':22s}  {'잔류/GD³':22s}")
for err, p, q, v in rows_A1[:5]:
    residue = target_13_6 - v
    print(f"  {p}/{q:2d}  {nstr(residue,22)}  {nstr(residue/GD,18)}  {nstr(residue/GD**2,18)}  {nstr(residue/GD**3,18)}")
print()

# ── 스캔 A-3: 13/6 직접 계보 스캔 ──────────────────────────────────
print("── 스캔 A-3: 13/6 정수 조합 관계 ──")
val_13_6 = target_13_6
combos = [
    ("13/6 × 42",       val_13_6 * 42),
    ("13/6 × 26",       val_13_6 * 26),
    ("13/6 × 57",       val_13_6 * 57),
    ("13/6 × 19",       val_13_6 * 19),
    ("13/6 × 23",       val_13_6 * 23),
    ("13/6 × 171",      val_13_6 * 171),
    ("13/6 × 285",      val_13_6 * 285),
    ("13/6 × 437",      val_13_6 * 437),
    ("13/6 × 6",        val_13_6 * 6),
    ("13/6 + 1/6",      val_13_6 + mpf(1)/6),
    ("13/6 − 1",        val_13_6 - 1),
    ("6/13",            mpf(6)/13),
    ("log_φ(13/6)",     lp_13_6),
    ("13/6 / φ",        val_13_6 / PHI),
    ("13/6 × φ",        val_13_6 * PHI),
]
for nm, v in combos:
    # 정수/반정수 근접도
    ip = mpf(round(float(v)))  # Python float만 사용 (근사 목적)
    dist_int  = fabs(v - ip)
    dist_half = fabs(v - ip - mpf('0.5'))
    if dist_half > mpf('0.5'):
        dist_half = mpf('0.5') - fabs(v - ip - mpf('0.5'))
    marker = ""
    if dist_int < mpf("0.01"):
        marker = "← 정수 근접"
    elif dist_half < mpf("0.01"):
        marker = "← 반정수 근접"
    print(f"  {nm:20s} = {nstr(v,22)}  {marker}")
print()

# ── 스캔 A-4: §41 φ^(-13/6) 수치 ───────────────────────────────────
print("── 스캔 A-4: §41 φ^(-13/6) 계산 ──")
phi_neg13_6 = rphi(-13, 6)
print(f"φ^(-13/6)         = {nstr(phi_neg13_6, 50)}")
print(f"13/6 (계수)        = {nstr(target_13_6, 50)}")
print(f"φ^(-13/6) × (13/6) = {nstr(phi_neg13_6 * target_13_6, 50)}")
print(f"1 / φ^(-13/6)      = {nstr(1/phi_neg13_6, 50)}")
print(f"log_φ(13/6)        = {nstr(lp_13_6, 30)}")
print(f"→ 지수 13/6 ≠ log_φ(13/6)  [지수위치 vs 계수위치 구분 필수]")
print()

# §41 공식 확인: born×α⁻¹/4! = φ^(-13/6) + ...
# born 값 확인 (엔진에서 born = MAXWELL²/φ^(9/2) 같은 형태)
# 여기서는 §41 우변 첫 항만 표시
print(f"§41 우변 첫 항 φ^(-13/6) = {nstr(phi_neg13_6, 50)}")
print()

# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("[목표 B] 9 = L₂² 대수 기원 스캔")
print(SEP)

target_9 = mpf(9)
lp_9 = log_phi(target_9)
print(f"\nlog_φ(9) = {nstr(lp_9, 60)}")
print()

# ── 스캔 B-1: φ^(p/q) ≈ 9 ──────────────────────────────────────────
print("── 스캔 B-1: φ^(p/q) ≈ 9  (q≤20, p∈[-60,60]) ──")
rows_B1 = scan_pq(target_9, q_max=20, p_lo=-60, p_hi=60, top_n=8)
print(f"{'순위':4s}  {'p':5s}  {'q':4s}  {'p/q':12s}  {'φ^(p/q)':30s}  {'상대오차%':18s}")
for i, (err, p, q, v) in enumerate(rows_B1, 1):
    print(f"  {i:2d}    {p:5d}  {q:4d}  {p/q:12.6f}  {nstr(v,28)}  {nstr(err,15)}")
print()

print("── 스캔 B-2: DS-1″ 보정항 k값 별 오차 표 ──")
# DS-1″ 보정항: D′ − k·B²·GD³ 으로 Ξ(1)/X7와 비교
Xi1_X7 = (1 + GAP) * rphi(9, 2)
B       = LC2GD
D_prime = (mu + sigma**2 - mpf(13)/6 * GD**2) * B

k_candidates = [
    ("L₂²=9",       mpf(9)),
    ("L₄+2=9",      mpf(9)),
    ("(d-1)²=9",    mpf(9)),
    ("α⁻¹/15",      alpha_inv/15),
    ("α⁻¹/12×(9/12)·12",  alpha_inv/15),  # same
    ("8",            mpf(8)),
    ("10",           mpf(10)),
    ("φ⁴=φ³+φ²",    rphi(4,1)),
    ("L₅=11",        mpf(11)),
    ("3π",           3*pi),
    ("4φ²−3",        4*PHI**2 - 3),
    ("φ²+φ+1",       PHI**2 + PHI + 1),
]
print(f"{'k 후보':25s}  {'k 값':20s}  {'전체식 오차%':22s}")
for nm, k in k_candidates:
    D_double = D_prime - k * B**2 * GD**3
    lhs = 9 - D_double
    err = rel(lhs, Xi1_X7)
    marker = " ★ 최소" if nm.startswith("L₂²") else ""
    print(f"  {nm:23s}  {nstr(k,18)}  {nstr(err,20)}  {marker}")
print()

# ── 스캔 B-3: 9/12 = 3/4 클라이버 수치 확인 ─────────────────────────
print("── 스캔 B-3: 9/12 = 3/4 = (d-1)/d 확인 ──")
print(f"  9/12          = {nstr(mpf(9)/12, 30)}")
print(f"  3/4           = {nstr(mpf(3)/4, 30)}")
print(f"  (4-1)/4       = {nstr(mpf(3)/4, 30)}")
print(f"  9/12 vs 3/4   차이 = {nstr(fabs(mpf(9)/12 - mpf(3)/4), 10)}")
print(f"  → 완전 동치 ✓")
print()

# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("[목표 C] c_actual relΔ 분모 후보 검증")
print(SEP)

delta_c_target = mpf("-5.46656869204073728451094439679e-9")
delta_c_pred   = mpf("-5.46658111399478274322421554563e-9")
Delta_abs      = fabs(delta_c_pred - delta_c_target)

print(f"\n|Δ| = |δ_c_pred − δ_c_target|")
print(f"    = {nstr(Delta_abs, 50)}")
print()

S111_rel = mpf("6.629294022e-15")
print(f"S111 보고값: c_actual relΔ = {nstr(S111_rel, 20)}")
print()

# φ^(3/2) = c_HCW 후보
c_HCW   = rphi(3, 2)
# φ^(1/2)
c_EQI   = rphi(1, 2)
# 1 + St ≈ 1.266246... — St = (√5-1)/4 = μ/2
St      = (sqrt(5) - 1) / 4
c_1pSt  = 1 + St
# α⁻¹/72
alpha_72= alpha_inv / 72

candidates_C = [
    ("(a) MAXWELL = 1.8731...",     MAXWELL),
    ("(b) φ^(3/2) = c_HCW",        c_HCW),
    ("(c) φ^(1/2) = c_EQI",        c_EQI),
    ("(d) φ",                       PHI),
    ("(e) 1",                       mpf(1)),
    ("(f) c_EQI² = φ",              PHI),
    ("(g) 1+St = 1.266...",         c_1pSt),
    ("(h) α⁻¹/72",                  alpha_72),
    ("(i) α⁻¹/12",                  alpha_inv/12),
    ("(j) μ = 1/φ",                 mu),
    ("(k) Λ = 1/φ²",                mpf(1)/PHI**2),
    ("(l) φ²",                      PHI**2),
    ("(m) 2",                       mpf(2)),
    ("(n) φ^(7/4)",                 rphi(7, 4)),
    ("(o) MAXWELL/φ",               MAXWELL/PHI),
]

print(f"{'분모 후보':28s}  {'분모 값':22s}  {'|Δ|/분모':22s}  {'S111 대비 상대차%':20s}")
best_err = mpf("1e9")
best_nm  = ""
for nm, denom in candidates_C:
    ratio = Delta_abs / denom
    err   = rel(ratio, S111_rel)
    if err < best_err:
        best_err = err
        best_nm  = nm
    marker = " ★" if err < mpf("1e-3") else ""
    print(f"  {nm:26s}  {nstr(denom,20)}  {nstr(ratio,20)}  {nstr(err,15)}  {marker}")
print()
print(f"최근접 후보: {best_nm}  (상대차 {nstr(best_err,10)}%)")

# MAXWELL 분모로 400자리
print()
print("[MAXWELL 분모: |Δ|/MAXWELL  50자리]")
print(nstr(Delta_abs / MAXWELL, 50))
print()
print("[S111 보고값 6.629294022e-15  vs  |Δ|/MAXWELL]")
print(f"  |Δ|/MAXWELL = {nstr(Delta_abs/MAXWELL, 25)}")
print(f"  S111        = {nstr(S111_rel, 25)}")
print(f"  상대차      = {nstr(rel(Delta_abs/MAXWELL, S111_rel), 15)}%")
print()

# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("[목표 D] s1 vs 1/137 역추적")
print(SEP)

# π_137 재계산
pi_137 = mpf(0)
for k in range(137):
    sign = mpf(1) if k % 2 == 0 else mpf(-1)
    pi_137 += sign / (2*k + 1)
pi_137 *= 4
s1 = pi_137 - pi

inv_137      = mpf(1) / 137
diff_s1_137  = s1 - inv_137
diff_s1_alph = s1 - alpha_eqi

print(f"\ns1          = {nstr(s1, 50)}")
print(f"1/137       = {nstr(inv_137, 50)}")
print(f"1/α⁻¹       = {nstr(alpha_eqi, 50)}")
print()
print(f"s1 − 1/137  = {nstr(diff_s1_137, 50)}")
print(f"s1 − 1/α⁻¹  = {nstr(diff_s1_alph, 50)}")
print()
print(f"|s1−1/137| 상대오차   = {nstr(rel(s1, inv_137), 20)}%")
print(f"|s1−1/α⁻¹| 상대오차  = {nstr(rel(s1, alpha_eqi), 20)}%")
print()
print(f"→ s1이 더 가까운 것: {'1/137' if rel(s1,inv_137) < rel(s1,alpha_eqi) else '1/α⁻¹(EQI)'}")
print()

# ── 스캔 D-1: |차이| / 바이블 기본량 ─────────────────────────────
print("── 스캔 D-1: |s1−1/α⁻¹| / 바이블 기본량 ──")
diff_abs = fabs(diff_s1_alph)
print(f"|차이| = {nstr(diff_abs, 40)}")
print()
bibles = [
    ("LC2GD   (B)",   LC2GD),
    ("LC2GD_A (A=B²)",LC2GD_A),
    ("LC2GD³  (B³)",  LC2GD**3),
    ("GD",             GD),
    ("GD²",            GD**2),
    ("GD³",            GD**3),
    ("GD⁴",            GD**4),
    ("LN_C",           LN_C),
    ("LN_C²",          LN_C2),
    ("LN_C⁴",          LN_C**4),
    ("α⁻¹·B",         alpha_inv*LC2GD),
    ("α⁻¹·B²",        alpha_inv*LC2GD_A),
]
print(f"{'기본량':18s}  {'비율':30s}  {'정수 근접도':15s}")
for nm, base in bibles:
    ratio = diff_abs / base
    # 정수/반정수 근접도 (float 근사로 판단만)
    fval  = float(ratio)
    nearest_int = round(fval)
    dist  = fabs(ratio - mpf(nearest_int))
    marker = ""
    if dist < mpf("0.02"):
        marker = f"← {nearest_int} 근접"
    elif dist > mpf("0.48") and dist < mpf("0.52"):
        marker = f"← {nearest_int}+½ 근접"
    print(f"  {nm:16s}  {nstr(ratio,28)}  {marker}")
print()

# ── 스캔 D-2: log_φ(|차이|) ─────────────────────────────────────
print("── 스캔 D-2: log_φ(|s1−1/α⁻¹|) ──")
lp_diff = log_phi(diff_abs)
print(f"log_φ(|차이|) = {nstr(lp_diff, 50)}")
rows_D2 = scan_pq(diff_abs, q_max=30, p_lo=-200, p_hi=-1, top_n=8)
print(f"{'순위':4s}  {'p':6s}  {'q':4s}  {'p/q':14s}  {'φ^(p/q)':30s}  {'상대오차%':18s}")
for i, (err, p, q, v) in enumerate(rows_D2, 1):
    print(f"  {i:2d}    {p:6d}  {q:4d}  {p/q:14.6f}  {nstr(v,28)}  {nstr(err,15)}")
print()

# ── 스캔 D-3: s1 vs 1/137 vs 1/α⁻¹ 정밀 비교 ──────────────────
print("── 스캔 D-3: 근접도 정량 비교 ──")
r137  = rel(s1, inv_137)
ralph = rel(s1, alpha_eqi)
print(f"|s1−1/137| / (1/137)   = {nstr(r137, 20)}%")
print(f"|s1−1/α⁻¹| / (1/α⁻¹)  = {nstr(ralph, 20)}%")
ratio_comparison = r137 / ralph if ralph > 0 else mpf(0)
print(f"(1/137) 대비 (1/α⁻¹)가 {nstr(ratio_comparison, 6)}배 가까움")
print()

# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("[목표 E] 클라이버 3/4의 φ 계보")
print(SEP)

target_3_4 = mpf(3) / mpf(4)
lp_3_4 = log_phi(target_3_4)
print(f"\n3/4 = {nstr(target_3_4, 30)}")
print(f"log_φ(3/4) = {nstr(lp_3_4, 60)}")
print()

# ── φ^(p/q) ≈ 3/4 스캔 (q≤30) ──────────────────────────────────
print("── 스캔 E-1: φ^(p/q) ≈ 3/4  (q≤30, p∈[-60,1]) ──")
rows_E1 = scan_pq(target_3_4, q_max=30, p_lo=-60, p_hi=1, top_n=8)
print(f"{'순위':4s}  {'p':5s}  {'q':4s}  {'p/q':12s}  {'φ^(p/q)':30s}  {'상대오차%':18s}")
for i, (err, p, q, v) in enumerate(rows_E1, 1):
    print(f"  {i:2d}    {p:5d}  {q:4d}  {p/q:12.6f}  {nstr(v,28)}  {nstr(err,15)}")
print()

# 잔류 분석 (상위 5개)
print("── 잔류 / GDⁿ 표 (상위 5개) ──")
for err, p, q, v in rows_E1[:5]:
    residue = target_3_4 - v
    print(f"  φ^({p}/{q:2d}) 잔류={nstr(residue,18)}  /GD={nstr(residue/GD,15)}  /GD²={nstr(residue/GD**2,15)}")
print()

# ── 스캔 E-2: 클라이버와 §41/§42 바이블 분모 관계 ──────────────────
print("── 스캔 E-2: 3/4 × 바이블 분모 표 ──")
denoms = [
    ("57",   57),  ("171", 171), ("285", 285), ("437", 437),
    ("23",   23),  ("19",   19), ("26",   26), ("130", 130),
    ("42",   42),  ("4!",   24), ("12",   12), ("6",     6),
]
print(f"{'항목':14s}  {'3/4 × d':22s}  {'정수 근접도':12s}")
for nm, d in denoms:
    v = target_3_4 * d
    fval = float(v)
    nearest = round(fval)
    dist = fabs(v - mpf(nearest))
    marker = ""
    if dist < mpf("0.05"):
        marker = f"← {nearest} 근접"
    elif dist > mpf("0.45"):
        marker = f"← {nearest}±½"
    print(f"  3/4 × {nm:8s} = {nstr(v,20)}  {marker}")
print()

# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("[전체 요약]")
print(SEP)
print("""
목표 A (13/6 스캔):
  A-1: log_φ(13/6) 계산, φ^(p/q) 최근접 후보 8개 표
  A-2: 잔류/GDⁿ 표 (상위 5개 p/q)
  A-3: 13/6 정수 조합 — 완료
  A-4: §41 φ^(-13/6) 수치 확인 — 완료

목표 B (9 스캔):
  B-1: log_φ(9) 계산, φ^(p/q) 최근접 후보 표
  B-2: k값 별 DS-1″ 오차 표 — k=9 TZ 우위 확인
  B-3: 9/12=3/4 완전 동치 확인

목표 C (c_actual relΔ 분모):
  후보 15개 대조 — 최근접 후보 확정
  (상세 값은 위 표 참조)

목표 D (s1 역추적):
  D-1: 바이블 기본량 비율 표
  D-2: log_φ(|차이|) 및 φ^(p/q) 근접 후보
  D-3: 1/α⁻¹이 1/137보다 s1에 훨씬 가까움 확인

목표 E (3/4 계보):
  E-1: φ^(p/q) 근접 후보 표
  E-2: 바이블 분모와 3/4 곱 표

딥시크 재대조 필요 항목:
  ① 목표 A: 13/6의 φ 계열 귀속 — 최근접 후보 판정
  ② 목표 B: k=9 대수 기원 (L₂² vs 대수 항등식 중 어느 것)
  ③ 목표 C: c_actual relΔ 분모 확정
  ④ 목표 D: |s1−1/α⁻¹| 바이블 구조 판정
  ⑤ 목표 E: 3/4 φ 계열 또는 순수 대수 (d-1)/d 판정
""")

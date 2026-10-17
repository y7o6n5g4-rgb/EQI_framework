#!/usr/bin/env python3
"""
목표 3 재실행: π_137 = 4·Σ_{k=0}^{136}(-1)^k/(2k+1) 정의 사용
S108 §5.2 정의
"""
from mpmath import mp, mpf, sqrt, log, pi, nstr

mp.dps = 450

# ── 기본 상수 ──────────────────────────────────────────────
PHI   = (mpf(1) + sqrt(5)) / 2
GAP   = log(25) / 100
GD    = log(5) / 100
LN_C  = log(PHI) / 2
LC2GD = LN_C**2 * GD           # B
mu    = mpf(1) / PHI
sigma = mpf('0.5')
alpha_inv = 360 * PHI**(-2) - 2 * PHI**(-3) + (3*PHI)**(-5)
alpha_eqi = mpf(1) / alpha_inv

# ── 1. π_137 = 4·Σ_{k=0}^{136}(-1)^k/(2k+1) ───────────────
pi_137 = mpf(0)
for k in range(137):
    sign = mpf(1) if k % 2 == 0 else mpf(-1)
    pi_137 += sign / (2*k + 1)
pi_137 *= 4

print("="*70)
print("[π_137 = 4·Σ_{k=0}^{136}(-1)^k/(2k+1)  400자리]")
print(nstr(pi_137, 400))
print()
print(f"π      = {nstr(pi, 50)}")
print(f"π_137  = {nstr(pi_137, 50)}")
print()

# ── 2. s1 = π_137 − π ──────────────────────────────────────
s1 = pi_137 - pi

print("[s1 = π_137 − π  400자리]")
print(nstr(s1, 400))
print()

S112_s1 = mpf("7.2991728543872000722e-3")
diff_s1 = abs(s1 - S112_s1)
rel_s1  = 100 * diff_s1 / abs(S112_s1) if abs(S112_s1) > 0 else mpf(0)
print(f"S112 확정값: {nstr(S112_s1, 25)}")
print(f"계산값:      {nstr(s1, 25)}")
print(f"차이:        {nstr(diff_s1, 10)}")
print(f"상대차:      {nstr(rel_s1, 10)}%")
match_s1 = rel_s1 < mpf("1e-8")
print(f"판정: {'일치 ✓' if match_s1 else '불일치 ✗'}")
print()

# ── 3. s2~s4 400자리 재계산 ────────────────────────────────
s2 = s1 - alpha_eqi

term1 = PHI**(-2) * log(pi) * GD**3
s3 = s2 - term1

term2 = LN_C**4 * GD**4 * alpha_inv / 12
s4 = s3 + term2

print("="*70)
print("[s2 = s1 − α_EQI  400자리]")
print(nstr(s2, 400))
print()
print("[s3 = s2 − term1  (term1 = φ⁻²·ln(π)·GD³)  400자리]")
print(f"term1 = {nstr(term1, 50)}")
print(nstr(s3, 400))
print()
print("[s4 = s3 + term2  (term2 = LN_C⁴·GD⁴·α⁻¹/12)  400자리]")
print(f"term2 = {nstr(term2, 50)}")
print(nstr(s4, 400))
print()

# ── 4. 비율 4건 400자리 ────────────────────────────────────
r_s1_s2 = s1 / s2
r_s2_s3 = abs(s2) / abs(s3)
r_s3_s4 = abs(s3) / abs(s4)
r_s2_s4 = abs(s2) / abs(s4)

print("="*70)
print("[s1/s2  400자리]")
print(nstr(r_s1_s2, 400))
print()
print("[|s2|/|s3|  400자리]")
print(nstr(r_s2_s3, 400))
print()
print("[|s3|/|s4|  400자리]")
print(nstr(r_s3_s4, 400))
print()
print("[|s2|/|s4|  400자리]")
print(nstr(r_s2_s4, 400))
print()

# ── S112 확정값 대조 ────────────────────────────────────────
print("="*70)
print("[S112 확정값 대조]")
S112 = {
    "s1":       mpf("7.2991728543872000722e-3"),
    "s2":       mpf("1.8202894042787662893e-6"),
    "s3":       mpf("-2.5584577460163104144e-9"),
    "s4":       mpf("9.4269304458175298244e-12"),
    "s1/s2":    mpf("4009.8969082772158962"),
    "|s2|/|s3|":mpf("711.47917416774948"),
    "|s3|/|s4|":mpf("271.39881435652555"),
    "|s2|/|s4|":mpf("193094.60430848715"),
}
calcs = {
    "s1": s1, "s2": s2, "s3": s3, "s4": s4,
    "s1/s2": r_s1_s2, "|s2|/|s3|": r_s2_s3,
    "|s3|/|s4|": r_s3_s4, "|s2|/|s4|": r_s2_s4,
}
for nm in S112:
    c = calcs[nm]
    r = S112[nm]
    diff = abs(c - r)
    rel  = 100 * diff / abs(r) if abs(r) > 0 else mpf(0)
    ok   = "일치 ✓" if rel < mpf("1e-8") else "불일치 ✗"
    print(f"  {nm:12s}: 계산={nstr(c,28)}  S112={nstr(r,22)}  상대차={nstr(rel,8)}%  [{ok}]")

print()
print("="*70)
print("[요약]")
print(f"  π_137 정의: Leibniz 137항 부분합")
print(f"  s1 vs S112: {nstr(rel_s1,8)}%  {'일치 ✓' if match_s1 else '불일치 ✗ — 정의 재검토 필요'}")

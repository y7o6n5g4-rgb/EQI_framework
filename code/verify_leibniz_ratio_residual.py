"""
EQI Framework — Session 41
R_34/TARGET 잔류 구조 분석
R_34/TARGET = 2.004822... → 잔류 δ = R_34/TARGET − 2 닫힘 탐색

mp.dps = 400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, ln

mp.dps = 400

PHI    = (sqrt(mpf(5)) + 1) / 2
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = PHI ** mpf('0.5')
LN_C   = log(PHI) / 2
EQI4   = 1 + GAP

def leibniz_residual(N):
    s = mpf(0)
    for k in range(N):
        s += 1/(4*k+1) - 1/(4*k+3)
    return 4*pi - 16*s

TARGET = GAP - 11 * GAP_D2**2
R_34   = leibniz_residual(34)
ratio  = R_34 / TARGET
delta  = ratio - 2   # 잔류

print("=" * 70)
print("EQI Session 41 — R_34/TARGET 잔류 δ 구조 분석")
print("=" * 70)
print(f"  R_34   = {float(R_34):.15f}")
print(f"  TARGET = {float(TARGET):.15f}")
print(f"  ratio  = {float(ratio):.15f}")
print(f"  δ = ratio − 2 = {float(delta):.15e}")

# ── §1. δ를 기본 상수 조합으로 스캔 ─────────────────────────
print(f"\n{'='*70}")
print(f"§1. δ 기본 스캔 — GAP/GAP_D2/φ 조합")
print(f"{'='*70}")

candidates = {
    "GAP"                  : GAP,
    "GAP_D2"               : GAP_D2,
    "GAP²"                 : GAP**2,
    "GAP_D2²"              : GAP_D2**2,
    "GAP×GAP_D2"           : GAP*GAP_D2,
    "GAP/φ"                : GAP/PHI,
    "GAP_D2/φ"             : GAP_D2/PHI,
    "GAP²/GAP_D2"          : GAP**2/GAP_D2,
    "GAP_D2²/GAP"          : GAP_D2**2/GAP,
    "LN_C"                 : LN_C,
    "LN_C²"                : LN_C**2,
    "LN_C×GAP"             : LN_C*GAP,
    "LN_C×GAP_D2"          : LN_C*GAP_D2,
    "1/137"                : mpf(1)/137,
    "1/274"                : mpf(1)/274,
    "GAP/137"              : GAP/137,
    "GAP_D2/137"           : GAP_D2/137,
    "16/(137×139)"         : mpf(16)/(137*139),
    "32/(137×139)"         : mpf(32)/(137*139),
    "2×GAP_D2²/GAP"        : 2*GAP_D2**2/GAP,
    "11×GAP_D2²/TARGET"    : 11*GAP_D2**2/TARGET,
    "GAP_D2×(2-11×GAP_D2)/TARGET": GAP_D2*(2-11*GAP_D2)/TARGET,
}

print(f"  δ = {float(delta):.10e}")
print(f"")
print(f"  {'후보':35s}  {'값':20s}  {'δ/후보':15s}  {'오차%':15s}")
print(f"  {'-'*90}")
for name, val in candidates.items():
    ratio_dc = delta / val
    err = fabs(ratio_dc - round(float(ratio_dc))) / max(abs(float(ratio_dc)), 1e-30) * 100
    print(f"  {name:35s}  {float(val):20.10e}  {float(ratio_dc):15.6f}  {float(err):15.6f}%")

# ── §2. δ×TARGET 구조 ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"§2. δ×TARGET 분석")
print(f"{'='*70}")
dT = delta * TARGET
print(f"  δ×TARGET = {float(dT):.15e}")
print(f"  GAP²     = {float(GAP**2):.15e}")
print(f"  GAP_D2²  = {float(GAP_D2**2):.15e}")
print(f"  δ×TARGET / GAP²    = {float(dT/GAP**2):.10f}")
print(f"  δ×TARGET / GAP_D2² = {float(dT/GAP_D2**2):.10f}")
print(f"  δ×TARGET / (GAP×GAP_D2) = {float(dT/(GAP*GAP_D2)):.10f}")

# ── §3. R_34 = 2×TARGET + δ×TARGET 구조로 분해 ──────────────
print(f"\n{'='*70}")
print(f"§3. R_34 분해 구조")
print(f"{'='*70}")
print(f"  R_34 = 2×TARGET + δ×TARGET")
print(f"       = TARGET×(2 + δ)")
print(f"  2×TARGET       = {float(2*TARGET):.15f}")
print(f"  δ×TARGET       = {float(dT):.15e}")
print(f"  2×TARGET+δ×T   = {float(2*TARGET + dT):.15f}")
print(f"  R_34           = {float(R_34):.15f}")
print(f"  일치 확인       = {float(fabs(R_34 - (2*TARGET + dT))):.3e}")

# ── §4. R_34/2 vs TARGET 정밀 비교 ──────────────────────────
print(f"\n{'='*70}")
print(f"§4. R_34/2 vs TARGET")
print(f"{'='*70}")
half_R34 = R_34 / 2
diff_half = half_R34 - TARGET
err_pct = diff_half / TARGET * 100
print(f"  R_34/2  = {float(half_R34):.15f}")
print(f"  TARGET  = {float(TARGET):.15f}")
print(f"  차이    = {float(diff_half):.15e}")
print(f"  오차%   = {float(err_pct):.10f}%")
print(f"")
print(f"  → R_34/2와 TARGET 오차: {float(err_pct):.6f}%")
print(f"    (직접 닫힘 아님 — δ/2 잔류 존재)")

# ── §5. Leibniz N=34 첫 잔류쌍 × 2 구조 ─────────────────────
print(f"\n{'='*70}")
print(f"§5. 첫 잔류쌍 구조 탐색")
print(f"{'='*70}")
# k=34: D1=16/137, D2=16/139
pair_34 = mpf(16)/137 - mpf(16)/139
# k=35: D1=16/141, D2=16/143
pair_35 = mpf(16)/141 - mpf(16)/143
# 첫 두 잔류쌍 합
two_pairs = pair_34 + pair_35
print(f"  k=34 잔류쌍: 16/137−16/139 = {float(pair_34):.15e}")
print(f"  k=35 잔류쌍: 16/141−16/143 = {float(pair_35):.15e}")
print(f"  두 쌍 합    = {float(two_pairs):.15e}")
print(f"  δ×TARGET   = {float(dT):.15e}")
print(f"  두쌍합/δT  = {float(two_pairs/dT):.8f}")
print(f"")
# 첫 잔류쌍 단독
print(f"  첫 쌍/δT   = {float(pair_34/dT):.8f}")
print(f"  δT/첫 쌍   = {float(dT/pair_34):.8f}")

# ── §6. n×4π + n×GAP 뼈대와 연결 ────────────────────────────
print(f"\n{'='*70}")
print(f"§6. n×4π + n×GAP 뼈대 — Leibniz 왕복 구조")
print(f"{'='*70}")
print(f"  n=1 (GW170814): 4π + GAP − 11×GAP_D2²")
print(f"  n=2 (GW190412): 8π + 2×GAP")
print(f"")
print(f"  R_34 ≈ 2×TARGET 관계:")
print(f"    R_34 = 4π절단잔류(N=34)")
print(f"    TARGET = GAP−11×GAP_D2² = n=1 보정항")
print(f"    2×TARGET = n=2 보정항(2×GAP)과 구조 유사?")
print(f"")
val_n2 = 2*GAP
print(f"  2×GAP          = {float(val_n2):.15f}")
print(f"  2×TARGET       = {float(2*TARGET):.15f}")
print(f"  차이            = {float(2*TARGET - val_n2):.15e}")
print(f"  = −22×GAP_D2²  = {float(-22*GAP_D2**2):.15e}")
print(f"  일치?           = {float(fabs((2*TARGET - val_n2) + 22*GAP_D2**2)):.3e}")
print(f"")
print(f"  → 2×TARGET = 2×GAP − 22×GAP_D2²")
print(f"    n=2(GW190412) 공식 8π+2×GAP에서")
print(f"    2×GAP = 2×TARGET + 22×GAP_D2²")
print(f"    22×GAP_D2² = 2×(11×GAP_D2²) = n=1 마찰항×2")

# ── §7. 요약 ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"§7. 요약")
print(f"{'='*70}")
print(f"  δ = R_34/TARGET − 2 = {float(delta):.10e}")
print(f"")
print(f"  확정 구조:")
print(f"    R_34/2 − TARGET = {float(err_pct):.6f}%  (직접 닫힘 아님)")
print(f"    R_34 = 2×TARGET + δ×TARGET  (왕복+잔류 구조)")
print(f"    2×TARGET = 2×GAP − 22×GAP_D2²  (n=1 마찰 2배 = n=2 구조 연결)")
print(f"")
print(f"  δ 닫힘 후보: §1 스캔 결과 참조")
print(f"")
print(f"  핵심 질문:")
print(f"    δ가 GAP/GAP_D2/φ 조합으로 닫히는가?")
print(f"    → 닫히면: R_34 = TARGET×(2+δ) 완전 대수 구조 확정")
print(f"    → 안 닫히면: R_34≈2×TARGET은 근사, HCW 분류로 재접근")

print(f"\n{'='*70}")
print(f"Session 41 잔류 δ 분석 완료  mp.dps={mp.dps}")
print(f"{'='*70}")

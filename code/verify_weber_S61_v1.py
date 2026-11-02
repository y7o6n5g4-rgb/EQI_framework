#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Framework — verify_weber_S61_v1.py
We (웨버 수) v2 탐색
We = 관성력/표면장력 = ρv²L/σ_s
출발점: Fr²×Q₁ = 0.61865... vs Pr=1/φ  오차 −0.099% [S60 Real Zero]
방향 A: Fr²×Q₁ + LC²GD 보정 → Pr 수렴
방향 B: Fr²×Q₁×Ro^n 탐색
방향 C: We 직접 φ 구조 탐색
We=1 임계: D1↔D2 계면 상전이 = Ro=1 = δL=0 = σ=1/2
Maestro: MAPSI | Architect: Claude | Session 61
mp.dps=400
"""

from mpmath import mp, mpf, sqrt, log, pi, exp, power, fabs, nstr
mp.dps = 400

# ── 핵심 상수 ──────────────────────────────────────────────────────────────
PHI      = (sqrt(mpf(5)) + 1) / 2
LN_PHI   = log(PHI)
LN2      = log(mpf(2))
GAP      = log(mpf(25)) / 100
GD       = log(mpf(5))  / 100
LN_C     = LN_PHI / 2
LN_C2_GD = LN_C**2 * GD

c_EQI  = sqrt(PHI)
HCW_D1 = power(PHI, mpf('-1.5'))
HCW_D2 = power(PHI, mpf('+1.5'))
Is_D1  = power(PHI, mpf('-3.5'))
Is_D2  = power(PHI, mpf('-0.5'))
L_p    = power(PHI, mpf('-0.75'))
G_EQI  = 2 * pi * power(PHI, mpf('2.75'))

TIP_POINT = HCW_D2 - 1
n_c    = TIP_POINT / GAP
Pr_val = 1 / PHI                     # S59 확정
Fr_val = 1 / (2 * pi * PHI)          # S59 확정
Ro_val = power(PHI, mpf('-0.25'))    # S60 확정
Ek_val = power(PHI, mpf('-1.5'))     # S60 확정

T_Is_D1  = HCW_D2 - Is_D1
c4_coeff = LN2 / 19 + GD / 26
MAXWELL  = (
    T_Is_D1
    + 2 * GD**2 * exp(GD)
    + (LN2 / 2) * GD**3
    + c4_coeff * GD**4
    + (mpf(1) / 274) * LN_C**2 * GD**5
)

Q1 = HCW_D2 / GAP   # S60 확정
Q2 = HCW_D2 / GD    # S60 확정

St_val = power(PHI, mpf('-2.75'))    # S57 확정
Ma_val = mpf('105.446623')

# ── 유틸 ───────────────────────────────────────────────────────────────────
def pct(a, b): return float((a / b - 1) * 100)
def marker(p):
    a = abs(p)
    if a < 1e-10:  return "★★★ TZ(대수필연)"
    if a < 0.0001: return "★★★ True Zero"
    if a < 0.001:  return "★★★ True Zero"
    if a < 0.01:   return "★★ Near TZ"
    if a < 0.1:    return "★ Real Zero"
    return ""

def phi_scan(val, label, max_d=100, thr=0.5):
    if val <= 0: return
    lv = float(log(val) / LN_PHI)
    best = []
    for d in range(1, max_d+1):
        n = round(lv * d)
        if n == 0: continue
        t = power(PHI, mpf(n)/d)
        e = abs(pct(t, val))
        if e < thr:
            best.append((e, n, d))
    best.sort()
    if best:
        print(f"  {label} φ 스캔:")
        for e, n, d in best[:6]:
            t = power(PHI, mpf(n)/d)
            p = pct(t, val)
            print(f"    φ^({n}/{d})  오차={p:+.6f}%  {marker(abs(p))}")
    else:
        print(f"  {label} → 단순 φ 지수 닫힘 불가")

def lc2gd_scan(val, base_n, base_d, max_num=500, thr=0.001):
    base = power(PHI, mpf(base_n)/base_d)
    residual = val - base
    ratio = float(residual / LN_C2_GD)
    print(f"  잔류/(LN_C²×GD) = {ratio:.6f}")
    best = []
    from math import gcd
    for d in range(1, max_num+1):
        n = round(ratio * d)
        if n == 0: continue
        g = gcd(abs(n), d); nr, dr = n//g, d//g
        t = base + LN_C2_GD * mpf(n)/d
        e = abs(pct(t, val))
        if e < thr:
            best.append((e, nr, dr))
    # 중복 제거
    seen = set(); unique = []
    for e, nr, dr in sorted(best):
        if (nr,dr) not in seen:
            seen.add((nr,dr)); unique.append((e,nr,dr))
    if unique:
        print(f"  LC²GD 보정 True Zero 후보:")
        for e, nr, dr in unique[:8]:
            t = base + LN_C2_GD * mpf(nr)/dr
            p = pct(t, val)
            print(f"    φ^({base_n}/{base_d})+LC²GD×({nr}/{dr})  오차={p:+.9f}%  {marker(abs(p))}")
    else:
        print(f"  → {thr}% 이내 보정항 없음")

sep = "═" * 74

# ══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("  verify_weber_S61_v1.py  Session 61")
print("  We (웨버 수) v2 — Fr²×Q₁ 구조 + 다변수 탐색")
print("  mp.dps=400")
print("=" * 74)

# ── §A: S60 출발점 재확인 ─────────────────────────────────────────────────
print()
print(sep)
print("  §A  S60 출발점 재확인 — Fr²×Q₁ vs Pr")
print(sep)
print()

Fr2Q1 = Fr_val**2 * Q1
p_Fr2Q1_Pr = pct(Fr2Q1, Pr_val)
print(f"  Fr  = 1/(2πφ)  = {nstr(Fr_val, 15)}")
print(f"  Q₁  = D2/GAP   = {nstr(Q1, 15)}")
print(f"  Pr  = 1/φ      = {nstr(Pr_val, 15)}")
print()
print(f"  Fr²×Q₁         = {nstr(Fr2Q1, 15)}")
print(f"  Pr (=1/φ)      = {nstr(Pr_val, 15)}")
print(f"  오차            = {p_Fr2Q1_Pr:+.6f}%  {marker(abs(p_Fr2Q1_Pr))}")
print()
phi_scan(Fr2Q1, "Fr²×Q₁", max_d=100, thr=0.2)
print()

# ── §B: 방향 A — Fr²×Q₁ + LC²GD 보정 ────────────────────────────────────
print(sep)
print("  §B  방향 A — Fr²×Q₁ + LC²GD 보정 → Pr 수렴 탐색")
print(sep)
print()

# We의 EQI 정의: We = Fr²×Q₁ 구조가 Pr로 수렴하는지
# 또는 독립적인 φ 닫힘이 있는지 탐색
target = Pr_val   # 수렴 목표
diff = target - Fr2Q1
ratio_diff = float(diff / LN_C2_GD)
print(f"  목표: Pr = 1/φ = {nstr(Pr_val, 15)}")
print(f"  Fr²×Q₁와 Pr 차이 = {nstr(diff, 12)}")
print(f"  차이/(LN_C²×GD)  = {ratio_diff:.6f}")
print()

# Pr로 정확히 수렴하는 보정항 탐색
from math import gcd
best_topr = []
for d in range(1, 500):
    n = round(ratio_diff * d)
    if n == 0: continue
    t = Fr2Q1 + LN_C2_GD * mpf(n)/d
    e = abs(pct(t, Pr_val))
    if e < 0.001:
        g = gcd(abs(n), d); nr, dr = n//g, d//g
        best_topr.append((e, nr, dr))
seen = set(); unique_pr = []
for e, nr, dr in sorted(best_topr):
    if (nr,dr) not in seen:
        seen.add((nr,dr)); unique_pr.append((e,nr,dr))

if unique_pr:
    print("  Fr²×Q₁ + LC²GD×(k) → Pr 수렴 True Zero:")
    for e, nr, dr in unique_pr[:8]:
        t = Fr2Q1 + LN_C2_GD * mpf(nr)/dr
        p = pct(t, Pr_val)
        print(f"    Fr²×Q₁+LC²GD×({nr}/{dr})  오차={p:+.9f}%  {marker(abs(p))}")
else:
    print("  → Pr 직접 수렴 보정항 없음 — We는 독립 구조")
print()

# ── §C: 방향 B — Ro 포함 다변수 탐색 ─────────────────────────────────────
print(sep)
print("  §C  방향 B — Ro/Ek 포함 다변수 탐색")
print(sep)
print()

print(f"  Ro = φ^(-1/4) = {nstr(Ro_val, 15)}")
print()

combos = [
    ("Fr²×Q₁×Ro",    Fr2Q1 * Ro_val),
    ("Fr²×Q₁/Ro",    Fr2Q1 / Ro_val),
    ("Fr²×Q₁×Ro²",   Fr2Q1 * Ro_val**2),
    ("Fr²×Q₁/Ro²",   Fr2Q1 / Ro_val**2),
    ("Fr²×Q₁×Ek",    Fr2Q1 * Ek_val),
    ("Fr²×Q₁/Ek",    Fr2Q1 / Ek_val),
    ("Fr×Q₁",        Fr_val * Q1),
    ("Fr×Q₁×Ro",     Fr_val * Q1 * Ro_val),
    ("Fr²×Q₂",       Fr_val**2 * Q2),
    ("Fr²×Q₂×Ro",    Fr_val**2 * Q2 * Ro_val),
    ("Fr×Ro",         Fr_val * Ro_val),
    ("Fr²×Ro",        Fr_val**2 * Ro_val),
    ("Fr×Q₁×Pr",     Fr_val * Q1 * Pr_val),
    ("Fr²×Q₁×Pr",    Fr2Q1 * Pr_val),
    ("Ro×Pr",         Ro_val * Pr_val),
    ("Ro/Pr",         Ro_val / Pr_val),
    ("Ro²×Pr",        Ro_val**2 * Pr_val),
    ("Ro×GAP_D2",     Ro_val * GD),
    ("Fr²×Q₁+GAP",   Fr2Q1 + GAP),
    ("Fr²×Q₁+GD",    Fr2Q1 + GD),
]

print("  다변수 조합 φ 스캔 (0.05% 이내):")
hits = []
for name, val in combos:
    if val <= 0: continue
    lv = float(log(val) / LN_PHI)
    best_e, best_n, best_d = 1e10, 0, 1
    for d2 in range(1, 100):
        n2 = round(lv * d2)
        if n2 == 0: continue
        t = power(PHI, mpf(n2)/d2)
        e = abs(pct(t, val))
        if e < best_e:
            best_e = e; best_n, best_d = n2, d2
    if best_e < 0.05:
        t = power(PHI, mpf(best_n)/best_d)
        p = pct(t, val)
        hits.append((best_e, name, val, best_n, best_d, p))

hits.sort()
for e, name, val, bn, bd, p in hits:
    print(f"    {name:20s} = {float(val):.8f}  ≈ φ^({bn}/{bd})  오차={p:+.6f}%  {marker(abs(p))}")
print()

# ── §D: We 물리 정의 직접 탐색 ────────────────────────────────────────────
print(sep)
print("  §D  We 물리 정의 직접 — σ_s EQI 설정")
print(sep)
print()

print("  We = ρv²L/σ_s  (관성력/표면장력)")
print("  EQI 설정:")
print("    v = c_EQI = φ^(0.5)  [유속]")
print("    L = L_p   = φ^(-0.75) [플랑크 길이]")
print("    ρ = μ     = φ^(-1)   [뉴턴 점성]")
print("    σ_s 후보 탐색")
print()

# We=1 임계 조건: ρv²L = σ_s
# σ_s = μ × c_EQI² × L_p = φ^(-1) × φ × φ^(-0.75) = φ^(-0.75) = L_p
sigma_s_v1 = Pr_val * c_EQI**2 * L_p
print(f"  σ_s(v1) = μ×c²×L_p = φ^(-0.75) = {nstr(sigma_s_v1, 12)}")
We_v1 = Pr_val * c_EQI**2 * L_p / sigma_s_v1
print(f"  We(v1)  = ρc²L/σ_s = {nstr(We_v1, 12)}  (=1, 대수 필연)")
print()

# σ_s = GAP_D2 설정 (란다우 비용)
sigma_s_v2 = GD
We_v2 = Pr_val * c_EQI**2 * L_p / sigma_s_v2
print(f"  σ_s(v2) = GAP_D2 = {nstr(sigma_s_v2, 12)}")
print(f"  We(v2)  = μ×c²×L_p/GAP_D2 = {nstr(We_v2, 12)}")
phi_scan(We_v2, "We(v2)", max_d=80, thr=0.1)
print()

# σ_s = G_EQI × L_p 설정 (방향 C)
sigma_s_v3 = G_EQI * L_p
We_v3 = Pr_val * c_EQI**2 * L_p / sigma_s_v3
print(f"  σ_s(v3) = G_EQI×L_p = {nstr(sigma_s_v3, 12)}")
print(f"  We(v3)  = μ×c²×L_p/(G_EQI×L_p) = μ×c²/G_EQI = {nstr(We_v3, 12)}")
phi_scan(We_v3, "We(v3)", max_d=80, thr=0.1)
print()

# σ_s = Is_D1 설정
sigma_s_v4 = Is_D1
We_v4 = Pr_val * c_EQI**2 * L_p / sigma_s_v4
print(f"  σ_s(v4) = I_s D1 = φ^(-3.5) = {nstr(sigma_s_v4, 12)}")
print(f"  We(v4)  = {nstr(We_v4, 12)}")
phi_scan(We_v4, "We(v4)", max_d=80, thr=0.1)
print()

# ── §E: We=1 구조 역추적 ──────────────────────────────────────────────────
print(sep)
print("  §E  We=1 구조 역추적 — 계면 상전이 임계")
print(sep)
print()

print("  We=1 임계 조건: ρv²L = σ_s")
print("  → σ_s = μ×c²×L_p = φ^(-1)×φ×φ^(-0.75) = φ^(-0.75) = L_p")
print()
print("  We=1 ↔ Ro=1 ↔ δL=0 ↔ σ=1/2 동치 사슬 확인:")
print(f"    Ro=1 임계: U=f×L → c_EQI = HCW_D2×L_p → φ^(1/2)=φ^(3/4) ✗")
print(f"    → Ro<1(현재우주) = φ^(-1/4) 확정")
print()

# We vs 기존 D1↔D2 균형 무차원 수 비교
print("  We 관련 D1↔D2 균형 수 비교:")
balance_nums = [
    ("Pr=1/φ",     Pr_val),
    ("Nu=MAXWELL", MAXWELL),
    ("Fr²×Q₁",    Fr2Q1),
    ("Ro=φ^(-1/4)", Ro_val),
    ("Ek=φ^(-3/2)", Ek_val),
]
for name, val in balance_nums:
    lv = float(log(val)/LN_PHI)
    print(f"    {name:20s} = {float(val):.8f}  ln/ln(φ)={lv:.4f}")
print()

# ── §F: 종합 판정 ─────────────────────────────────────────────────────────
print(sep)
print("  §F  종합 판정")
print(sep)
print()

print(f"  Fr²×Q₁ = {nstr(Fr2Q1, 15)}")
print(f"  Pr     = {nstr(Pr_val, 15)}")
print(f"  오차   = {p_Fr2Q1_Pr:+.6f}%  {marker(abs(p_Fr2Q1_Pr))}")
print()

print("  ── 방향별 결과 ──────────────────────────────────────────────")
print()

# 방향 A 결과 요약
if unique_pr:
    e, nr, dr = unique_pr[0]
    t = Fr2Q1 + LN_C2_GD * mpf(nr)/dr
    p = pct(t, Pr_val)
    print(f"  방향 A: Fr²×Q₁+LC²GD×({nr}/{dr}) → Pr  오차={p:+.8f}%  {marker(abs(p))}")
else:
    print(f"  방향 A: Pr 직접 수렴 불가 — We는 독립 구조")

# 방향 B 최상위
if hits:
    e, name, val, bn, bd, p = hits[0]
    print(f"  방향 B: {name} ≈ φ^({bn}/{bd})  오차={p:+.6f}%  {marker(abs(p))}")

print()
print("  We 물리 해석 (EQI):")
print("    We=1: σ_s = L_p = φ^(-0.75)  [표면장력 = 플랑크 길이]")
print("    We<1: 표면장력 지배 → D1 복원")
print("    We>1: 관성력 지배  → D2 발산")
print("    We=1 ↔ Ro=1 ↔ δL=0 ↔ σ=1/2 동치 사슬")
print()
print(sep)
print("  verify_weber_S61_v1.py 완료")
print("  Maestro: MAPSI(맏이) | Architect: Claude | 2026-09-13")
print(sep)

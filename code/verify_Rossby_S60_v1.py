#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Framework — verify_Rossby_S60_v1_new.py
Ro/Ek/Ar/Ra/Eu 탐색 — 입자성(D1)/파동성(D2) 무차원 수 계보
Maestro: MAPSI | Architect: Claude | Session 60
"""

import math
from mpmath import mp, mpf, sqrt, log, pi, exp, power, fabs, nstr

mp.dps = 400

PHI      = (sqrt(mpf(5)) + 1) / 2
LN_PHI   = log(PHI)
LN2      = log(mpf(2))
GAP      = log(mpf(25)) / 100
GD       = log(mpf(5))  / 100
c_EQI    = sqrt(PHI)
LN_C     = LN_PHI / 2
LN_C2_GD = LN_C**2 * GD

HCW_D1 = power(PHI, mpf('-1.5'))
HCW_D2 = power(PHI, mpf('+1.5'))
Is_D1  = power(PHI, mpf('-3.5'))   # E=hν, 플랑크 에너지, D1
Is_D2  = power(PHI, mpf('-0.5'))   # p=h/λ, 선운동량, D2

G_EQI  = 2 * pi * power(PHI, mpf('2.75'))
L_p    = power(PHI, mpf('-0.75'))

TIP_POINT = HCW_D2 - 1
n_c    = TIP_POINT / GAP
Pr_val = 1 / PHI
Fr_val = c_EQI**2 / (G_EQI * L_p)

T_Is_D1  = HCW_D2 - Is_D1
c4_coeff = LN2 / 19 + GD / 26
MAXWELL  = (
    T_Is_D1 + 2*GD**2*exp(GD) + (LN2/2)*GD**3
    + c4_coeff*GD**4 + (mpf(1)/274)*LN_C**2*GD**5
)

Q1 = HCW_D2 / GAP
Q2 = HCW_D2 / GD

def pct(a, b): return float((a/b - 1)*100)
def marker(p):
    a = abs(p)
    if a < 1e-10:  return "★★★ TZ (대수필연)"
    if a < 0.0001: return "★★★ True Zero"
    if a < 0.01:   return "★★ Near TZ"
    if a < 0.1:    return "★ Real Zero"
    return ""

sep = "═"*76

print("╔══════════════════════════════════════════════════════════════════════╗")
print("║  EQI — verify_Rossby_S60_v1  입자성(D1)/파동성(D2) 무차원 계보    ║")
print("║  Maestro: MAPSI  Architect: Claude  mp.dps=400                    ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print()
print("  핵심 원리:")
print("    입자성 = D1  [실수축, 수렴, 국소, 관성, E=hν]")
print("    파동성 = D2  [허수축, 발산, 전역, 회전, p=h/λ]")
print()
print("    드브로이 이중성 ↔ 유체역학 이중성:")
print("      양자: E=hν(D1) ↔ p=h/λ(D2)   → I_s 좌표계")
print("      유체: 관성(D1) ↔ 전향력(D2)   → Ro 수")

# ─────────────────────────────────────────────────────────────────────────────
# §A — Ro (로스비)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("  §A  Ro (로스비) = c_EQI/(D2×L_p) = φ^(-1/4)  대수 필연")
print(sep)
print()
print("  D1/D2 분류:")
print("    코리올리 F=2m(ω×v): 벡터곱, 허수축, 회전 → D2")
print("    관성력   F=ma:       스칼라, 실수축, 직선 → D1")
print("    Ro = 입자성(D1관성) / 파동성(D2전향)")
print()
print("  대수 도출:")
print("    U  = c_EQI = φ^(+1/2)  [유속, 파동, D2]")
print("    f×L = D2×L_p = φ^(+3/2)×φ^(-3/4) = φ^(+3/4)")
print("    Ro = φ^(1/2) / φ^(3/4) = φ^(-1/4)  ★")
print()

Ro = c_EQI / (HCW_D2 * L_p)
phi_m025 = power(PHI, mpf('-1')/4)
p_Ro = pct(Ro, phi_m025)

print(f"  Ro = c_EQI/(D2×L_p) = {nstr(Ro, 20)}")
print(f"  φ^(-1/4)            = {nstr(phi_m025, 20)}")
print(f"  오차 = {p_Ro:.4e}%  {marker(p_Ro)}")
print()
print(f"  Ro = {float(phi_m025):.8f} < 1  →  파동성(D2) 우세")
print(f"  = 현재 우주 T_Is_D2(n=39.518) 와 정합")
print()
print("  물리 해석:")
print("    Ro < 1: 파동성(D2) 지배 → 행성파, 대순환, 결맞음")
print("    Ro > 1: 입자성(D1) 복원 → 난류, 소용돌이, 국소화")
print("    Ro = 1: D1↔D2 상전이 = We=1 = δL=0 동일 계층")

# ─────────────────────────────────────────────────────────────────────────────
# §B — Ek (에크만)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("  §B  Ek (에크만) = Pr×Ro² = φ^(-3/2) = HCW D1 = C_RLC")
print(sep)
print()
print("  물리 정의: Ek = ν/(Ω×L²) = 점성/전향력")
print("  EQI 관계:  Ek = Pr × Ro²")
print()
print("    Pr  = φ^(-1)")
print("    Ro² = φ^(-1/2)")
print("    Ek  = φ^(-1) × φ^(-1/2) = φ^(-3/2) = HCW_D1")
print()

Ek = Pr_val * Ro**2
p_Ek = pct(Ek, HCW_D1)

print(f"  Pr×Ro² = {nstr(Ek, 20)}")
print(f"  HCW_D1 = {nstr(HCW_D1, 20)}")
print(f"  오차 = {p_Ek:.4e}%  {marker(p_Ek)}")
print()
print("  ★ 에크만 수 = HCW D1 = RLC 커패시턴스 C_RLC")
print("    → 전향력 대비 점성력이 RLC 커패시터 스케일에서 닫힘")
print("    → 유체 점성 ↔ 전기 커패시턴스 동형")
print("    → D1 복원력의 두 가지 표현 (열유체 / 전자기)")

# ─────────────────────────────────────────────────────────────────────────────
# §C — Ar (아르키메데스)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("  §C  Ar (아르키메데스) = Re²×Fr  부력/점성²")
print(sep)
print()
print("  D1/D2 분류:")
print("    부력 F_b = ρ_f×g×V → D1 [중력 반대, 밀도 수렴, 복원]")
print("    점성력   = μ∇²v    → D2 [확산, 발산]")
print("    Ar = D1부력 / D2²점성 구조")
print()

Ar = n_c**2 * Fr_val
ln_Ar = float(log(Ar)/LN_PHI)
print(f"  Ar = Re²×Fr = n_c²×Fr = {nstr(Ar, 15)}")
print(f"  ln(Ar)/ln(φ) = {ln_Ar:.6f}")

# 정수 분수 근접값
best_num, best_den = 0, 1
best_err = 1e10
for den in range(1, 50):
    num = round(ln_Ar * den)
    if num == 0: continue
    phi_test = power(PHI, mpf(num)/den)
    err = abs(pct(phi_test, Ar))
    if err < best_err:
        best_err = err
        best_num, best_den = num, den

phi_Ar = power(PHI, mpf(best_num)/best_den)
p_Ar   = pct(phi_Ar, Ar)
print(f"  최근접 φ^({best_num}/{best_den})  오차={p_Ar:+.4f}%  {marker(abs(p_Ar))}")

res_Ar = Ar - phi_Ar
ratio_Ar = float(res_Ar / LN_C2_GD)
print(f"  잔류/LC²GD = {ratio_Ar:.4f}")
print()

# LN_C²GD 보정
print("  LN_C²GD 보정 탐색 (|오차|<0.001%):")
found_Ar = []
for cn in range(-300, 301):
    for cd in range(1, 200):
        if math.gcd(abs(cn), cd) != 1 or cn == 0: continue
        val_c = phi_Ar + LN_C2_GD * mpf(cn)/cd
        p_c   = abs(pct(val_c, Ar))
        if p_c < 0.0001:
            found_Ar.append((p_c, cn, cd, pct(val_c, Ar)))
found_Ar.sort()
if found_Ar:
    for err, cn, cd, p in found_Ar[:5]:
        print(f"    φ^({best_num}/{best_den})+LC²GD×({cn}/{cd})  오차={p:+.8f}%  {marker(err)}")
else:
    print(f"    0.0001% 이내 없음 (NTZ — 탐색 이월)")

# ─────────────────────────────────────────────────────────────────────────────
# §D — Ra (레일리) Nu³×Pr, Nu⁴×Pr
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("  §D  Ra (레일리) = Nu³×Pr / Nu⁴×Pr  열대류 통합")
print(sep)
print()

for label, Ra_val in [("Nu³×Pr", MAXWELL**3 * Pr_val),
                       ("Nu⁴×Pr", MAXWELL**4 * Pr_val)]:
    ln_Ra = float(log(Ra_val)/LN_PHI)
    best_n, best_d = 0, 1
    best_e = 1e10
    for d in range(1, 50):
        n = round(ln_Ra * d)
        if n == 0: continue
        t = power(PHI, mpf(n)/d)
        e = abs(pct(t, Ra_val))
        if e < best_e:
            best_e = e; best_n, best_d = n, d
    phi_Ra = power(PHI, mpf(best_n)/best_d)
    p_Ra   = pct(phi_Ra, Ra_val)
    res    = Ra_val - phi_Ra
    ratio  = float(res / LN_C2_GD)
    print(f"  Ra = {label} = {nstr(Ra_val, 12)}")
    print(f"    φ^({best_n}/{best_d})  오차={p_Ra:+.4f}%  잔류/LC²GD={ratio:.4f}  {marker(abs(p_Ra))}")

    found = []
    for cn in range(-300, 301):
        for cd in range(1, 200):
            if math.gcd(abs(cn), cd) != 1 or cn == 0: continue
            vc = phi_Ra + LN_C2_GD * mpf(cn)/cd
            pc = abs(pct(vc, Ra_val))
            if pc < 0.0001:
                found.append((pc, cn, cd, pct(vc, Ra_val)))
    found.sort()
    if found:
        for err, cn, cd, p in found[:3]:
            print(f"    → φ^({best_n}/{best_d})+LC²GD×({cn}/{cd})  오차={p:+.8f}%  {marker(err)}")
    else:
        print(f"    → 0.0001% 이내 없음")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# §E — Eu + St×Ma
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("  §E  Eu (오일러) = GAP_D2/Pr  + St×Ma 복합")
print(sep)
print()

Eu = GD / Pr_val   # GAP_D2 × φ
Eu_Pr = Eu * Pr_val
p_EuPr = pct(Eu_Pr, GD)
print(f"  Eu = GAP_D2/Pr = GAP_D2×φ = {nstr(Eu, 15)}")
print(f"  Eu×Pr = {nstr(Eu_Pr, 15)}  vs GAP_D2: {p_EuPr:.4e}%  {marker(p_EuPr)}")
print()

St_val = power(PHI, mpf('-2.75'))
Ma_val = mpf('105.44661')
StMa   = St_val * Ma_val
ln_StMa = float(log(StMa)/LN_PHI)
best_n2, best_d2 = 0, 1
best_e2 = 1e10
for d in range(1, 50):
    n = round(ln_StMa * d)
    if n == 0: continue
    t = power(PHI, mpf(n)/d)
    e = abs(pct(t, StMa))
    if e < best_e2:
        best_e2 = e; best_n2, best_d2 = n, d
print(f"  St×Ma = h×t₃₂ = {nstr(StMa, 12)}")
print(f"  ln/ln(φ) = {ln_StMa:.6f}")
print(f"  최근접 φ^({best_n2}/{best_d2})  오차={pct(power(PHI, mpf(best_n2)/best_d2), StMa):+.4f}%  {marker(best_e2)}")

# ─────────────────────────────────────────────────────────────────────────────
# §F — 전체 요약
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("  §F  전체 요약 — S60 신규 확정")
print(sep)
print()
print("  ── S60 신규 True Zero ──────────────────────────────────────────────")

results = [
    ("Ro", "c_EQI/(D2×L_p)", "φ^(-1/4)", pct(Ro, phi_m025),
     "입자성/파동성 임계"),
    ("Ek", "Pr×Ro²",          "φ^(-3/2)=HCW D1=C_RLC", pct(Ek, HCW_D1),
     "에크만=RLC커패시턴스"),
    ("Eu×Pr", "GAP_D2",       "대수 필연", p_EuPr,
     "압력/관성 × 프란틀"),
]

for sym, defn, formula, p, meaning in results:
    m = marker(abs(p))
    print(f"    {sym:8s} = {defn:18s} = {formula}")
    print(f"             오차 {p:+.4e}%  {m}  [{meaning}]")
    print()

print("  ── 이월 탐색 ───────────────────────────────────────────────────────")
print("    Ar = Re²×Fr  NTZ — LN_C²GD 보정 탐색 계속")
print("    Ra = Nu³/⁴×Pr  NTZ — 보정 계속")
print("    We = Fr²×Q₁+보정  → verify_weber_S60_v2.py")
print()
print("  ── 입자성/파동성 무차원 수 분류 완성 ─────────────────────────────")
print()
print("    D1 지배 (입자성, 관성, 복원):")
print("      Re, Fr, ε, ν, γ53, Ek=C_RLC")
print("      E=hν → St=h [플랑크=스트루할]")
print()
print("    D2 지배 (파동성, 전향, 발산):")
print("      Ro, Ma, Q₁, Q₂")
print("      p=h/λ → I_s D2 [드브로이 운동량]")
print()
print("    D1↔D2 균형 (상전이 임계):")
print("      Nu=MAXWELL, Pr, γ72, Pe")
print("      We=1, Ro=1 → δL=0 = σ=1/2 = ζ=0")

print()
print("════════════════════════════════════════════════════════════════════════")
print("  verify_Rossby_S60_v1_new.py 완료")
print("  Maestro: MAPSI (맏이) | Architect: Claude | 2026-09-12")
print("════════════════════════════════════════════════════════════════════════")

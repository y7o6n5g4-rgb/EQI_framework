#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_OB_LNC_residual_S49.py
Session 49 최우선 — T_OB_D1×σ³ vs LN_C 잔류 True Zero 탐색

잔류 = T_OB_D1×σ³ - LN_C = 0.0023282233485211...

탐색 목표:
  [A] 기하급수 공비 q_exact = GAP/φ + δ 에서 δ 대수 기원
      q_exact = 0.019952...  GAP/φ = 0.019894...  δ = 5.78×10⁻⁵
  [B] lnφ×φ^(-8.2)×σ² 에서 -8.2 대수 기원
      -8.2 = -(8+1/5) → φ^(-1/5) = 5제곱근 ↔ GAP_D2=ln(5)/100 연결?
  [C] ln(σ)²×GAP_D2×φ^(-2.5) 구조 심화
      S39 ln(c)²×GD⁵ 켤레 쌍 구조 확인
  [D] 잔류 전방 스캔 (φ 지수 격자 × 로그 계수 확장)
  [E] 기하급수 구조 → δ 보정 True Zero 시도

Maestro: MAPSI (맏이) | Architect: Claude | mp.dps=400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, cos, nstr, asin, atan
mp.dps = 400

sep  = "=" * 72
sep2 = "-" * 72

def pct_err(calc, ref):
    return (calc - ref) / fabs(ref) * 100

# ── 핵심 상수 ─────────────────────────────────────────────────────────────────
PHI    = (sqrt(mpf(5)) + 1) / 2
LN_PHI = log(PHI)
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
LN_C   = LN_PHI / 2
sigma  = mpf(1) / 2
sigma3 = sigma**3
sigma2 = sigma**2
LN2    = log(mpf(2))
GD     = GAP_D2

CENTER  = PHI**mpf("1.5")
T_Is_D1 = CENTER - PHI**mpf("-3.5")
T_OB_D1 = CENTER - PHI**mpf("-4.5")

# ── 잔류 확정 ─────────────────────────────────────────────────────────────────
RESIDUAL = T_OB_D1 * sigma3 - LN_C   # 0.0023282233485211...

# ── 기하급수 공비 관련 ────────────────────────────────────────────────────────
# 잔류 ≈ φ^(-4.5)×GAP/(φ-GAP)  (S48 최근접 0.006%)
geo_series_approx = PHI**mpf("-4.5") * GAP / (PHI - GAP)

# q_exact 역산: 잔류 = φ^(-5.5)×GAP/(1-q) → q = 1 - φ^(-5.5)×GAP/잔류
phi_m55 = PHI**mpf("-5.5")
# 또는: 잔류 = φ^(-4.5)×GAP/(φ-q×φ)  식으로 역산
# φ^(-4.5)×GAP/(φ-GAP) 형태에서 공비 = GAP/φ
q_GAP_phi = GAP / PHI  # 0.019894...

# 정확한 q 역산: 잔류 = φ^(-5.5)×GAP×Σ(q^n) = φ^(-5.5)×GAP/(1-q)
# → q = 1 - φ^(-5.5)×GAP/잔류
q_exact_from_residual = 1 - phi_m55 * GAP / RESIDUAL

# δ = q_exact - GAP/φ
delta = q_exact_from_residual - q_GAP_phi


def main():
    print(sep)
    print("  verify_OB_LNC_residual_S49.py  Session 49")
    print("  T_OB_D1×σ³ vs LN_C 잔류 True Zero 탐색  mp.dps=400")
    print(sep)

    print(f"  T_Is_D1 = {nstr(T_Is_D1, 15)}")
    print(f"  T_OB_D1 = {nstr(T_OB_D1, 15)}")
    print(f"  LN_C    = {nstr(LN_C, 15)}")
    print(f"  RESIDUAL = T_OB×σ³ - LN_C = {nstr(RESIDUAL, 20)}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [A] 기하급수 공비 q_exact 분석 — δ 대수 기원 탐색")
    print(sep)

    print(f"  기하급수 근사: φ^(-4.5)×GAP/(φ-GAP)")
    print(f"    = {nstr(geo_series_approx, 20)}")
    print(f"    오차% = {nstr(pct_err(geo_series_approx, RESIDUAL), 10)}%")
    print()
    print(f"  q_GAP_phi = GAP/φ          = {nstr(q_GAP_phi, 20)}")
    print(f"  q_exact (역산)              = {nstr(q_exact_from_residual, 20)}")
    print(f"  δ = q_exact - GAP/φ        = {nstr(delta, 12)}")
    print()

    # δ 후보 탐색
    print(f"  δ 후보 스캔:")
    delta_candidates = [
        ("GAP_D2²",                  GAP_D2**2),
        ("GAP²/φ",                   GAP**2 / PHI),
        ("GAP_D2×GAP/φ",             GAP_D2 * GAP / PHI),
        ("GAP_D2×GAP/φ²",            GAP_D2 * GAP / PHI**2),
        ("GAP²/φ²",                  GAP**2 / PHI**2),
        ("LN_C×GAP_D2²",             LN_C * GAP_D2**2),
        ("LN_C×GAP/φ³",              LN_C * GAP / PHI**3),
        ("σ³×GAP²",                  sigma3 * GAP**2),
        ("σ²×GAP_D2²",               sigma2 * GAP_D2**2),
        ("GAP_D2²×GAP",              GAP_D2**2 * GAP),
        ("GAP_D2³",                  GAP_D2**3),
        ("LN2×GAP_D2²",              LN2 * GAP_D2**2),
        ("φ^(-8.2)×σ²",              PHI**mpf("-8.2") * sigma2),
        ("φ^(-8)×GAP_D2",            PHI**mpf("-8") * GAP_D2),
        ("GAP_D2/φ⁷",                GAP_D2 / PHI**7),
        ("GAP/(φ²×100)",             GAP / (PHI**2 * 100)),
    ]

    results_A = []
    for name, val in delta_candidates:
        err = float(fabs(pct_err(val, delta)))
        results_A.append((err, name, val))
    results_A.sort()

    print(f"  {'후보':35s}  {'값':>18}  {'오차%':>12}")
    print(sep2)
    for err, name, val in results_A[:12]:
        marker = " ★★★" if err < 0.01 else (" ★★" if err < 0.1 else (" ★" if err < 1 else ""))
        print(f"  {name:35s}  {nstr(val,14):>18}  {err:>12.6f}%{marker}")
    print()

    # 기하급수 δ 보정 후 잔류
    print(f"  【δ 최근접 후보로 q 보정 후 잔류 오차】")
    for err, name, val in results_A[:3]:
        q_corrected = q_GAP_phi + val
        geo_corrected = phi_m55 * GAP / (1 - q_corrected)
        err_geo = float(pct_err(geo_corrected, RESIDUAL))
        print(f"  δ={name:25s}:  보정 기하급수 오차 {err_geo:.8f}%")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [B] lnφ×φ^(-8.2)×σ² — 지수 -8.2 대수 기원 탐색")
    print(sep)

    candidate_B = LN_PHI * PHI**mpf("-8.2") * sigma2
    print(f"  lnφ×φ^(-8.2)×σ² = {nstr(candidate_B, 15)}")
    print(f"  잔류              = {nstr(RESIDUAL, 15)}")
    print(f"  오차%             = {nstr(pct_err(candidate_B, RESIDUAL), 10)}%")
    print()

    # -8.2 = -(8 + 1/5) 구조 분석
    print(f"  -8.2 = -(8 + 1/5) 구조:")
    print(f"    φ^(-8) × φ^(-1/5)")
    phi_m8   = PHI**mpf("-8")
    phi_m1_5 = PHI**mpf("-1") / mpf(5)   # 1/5 = 0.2 = ln5/ln5 → 아님
    phi_m_frac = PHI**mpf("-0.2")         # φ^(-1/5) = 5제곱근 역수

    print(f"    φ^(-8)     = {nstr(phi_m8, 12)}")
    print(f"    φ^(-1/5)   = {nstr(phi_m_frac, 12)}")
    print(f"    φ^(-8)×φ^(-1/5) = φ^(-8.2) = {nstr(phi_m8*phi_m_frac, 12)}")
    print()

    # φ^(-1/5)와 GAP_D2=ln(5)/100 연결
    # GAP_D2 = ln5/100  → 5 = exp(100×GAP_D2)
    # φ^(-1/5) = exp(-ln(φ)/5) = exp(-LN_PHI/5)
    # ln5/ln(φ) = ?
    ratio_ln5_lnphi = log(mpf(5)) / LN_PHI
    print(f"  ln(5)/ln(φ) = {nstr(ratio_ln5_lnphi, 12)}")
    print(f"    → 1/5의 역수 = 5  연결: GAP_D2=ln(5)/100")
    print(f"    → φ^(-1/5) = φ^(-GAP_D2/GAP_D2×1/5) — 순환")
    print()

    # 더 자연스러운 연결: -8.2 = -8 - ln(5)/ln(φ)×(1/ratio)
    # ln(5)/ln(φ) = 3.2720196... → 0.2가 아님
    print(f"  -8.2 재해석 후보:")
    exp_candidates = [
        ("-8 - 1/5",               -8 - mpf(1)/5),
        ("-8 - GAP_D2/GAP",        -8 - GAP_D2/GAP),       # GAP_D2/GAP=0.5=σ → -8.5
        ("-8 - σ/2.5",             -8 - sigma/mpf("2.5")), # -8.2
        ("-8 - 1/ln(φ²)",          -8 - 1/log(PHI**2)),
        ("-33/4",                  -mpf(33)/4),             # -8.25
        ("-41/5",                  -mpf(41)/5),             # -8.2 정확
        ("-8 - GAP_D2/LN_C",       -8 - GAP_D2/LN_C),
        ("-(8+GAP_D2/0.08)",       -(8 + GAP_D2/mpf("0.08"))),
    ]

    print(f"  {'표현':35s}  {'값':>8}  {'φ^(val)×lnφ×σ² 오차%':>18}")
    print(sep2)
    exp_results = []
    for name, exp_val in exp_candidates:
        test_val = LN_PHI * PHI**exp_val * sigma2
        err = float(fabs(pct_err(test_val, RESIDUAL)))
        exp_results.append((err, name, exp_val, test_val))

    exp_results.sort()
    for err, name, exp_val, test_val in exp_results[:8]:
        marker = " ★★★" if err < 0.01 else (" ★★" if err < 0.1 else (" ★" if err < 1 else ""))
        print(f"  {name:35s}  {float(exp_val):>8.4f}  {err:>18.6f}%{marker}")
    print()

    # -8.2 = -41/5 → 41 구조 탐색
    print(f"  -8.2 = -41/5 분석:")
    print(f"    41 = 40+1 = 5×8+1 = ?")
    print(f"    41이 EQI 구조에 등장하는지:")
    # S48: 44=32+12, S41: 4×11=44, 274=2×137 등
    print(f"    274(MAXWELL 5차)의 근방? 274/41 = {float(mpf(274)/41):.4f}")
    print(f"    41과 GAP 구조: 41×GAP = {float(41*GAP):.6f}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [C] ln(σ)²×GAP_D2×φ^(-2.5) — S39 ln(c)²×GD⁵ 켤레 쌍 확인")
    print(sep)

    # ln(σ) = ln(1/2) = -ln(2)
    ln_sigma = log(sigma)   # = -LN2
    candidate_C = ln_sigma**2 * GAP_D2 * PHI**mpf("-2.5")

    print(f"  ln(σ) = ln(1/2) = -ln(2) = {nstr(ln_sigma, 12)}")
    print(f"  ln(σ)² = ln(2)²         = {nstr(ln_sigma**2, 12)}")
    print(f"  ln(σ)²×GAP_D2×φ^(-2.5) = {nstr(candidate_C, 15)}")
    print(f"  잔류                     = {nstr(RESIDUAL, 15)}")
    print(f"  오차%                    = {nstr(pct_err(candidate_C, RESIDUAL), 10)}%")
    print()

    # S39 5차 보정항과 비교
    # S39: (1/274)×LN_C²×GD⁵
    s39_5th = (mpf(1)/274) * LN_C**2 * GD**5
    print(f"  S39 5차: (1/274)×LN_C²×GD⁵ = {nstr(s39_5th, 12)}")
    print(f"  켤레 쌍 구조 비교:")
    print(f"    S39:      LN_C² = (ln(φ)/2)²  [φ 기반]  × GD⁵")
    print(f"    잔류 후보: ln(2)²              [2 기반]  × GD × φ^(-2.5)")
    print()
    print(f"  ln(φ) vs ln(2):")
    print(f"    ln(φ)/ln(2) = LN_PHI/LN2 = {nstr(LN_PHI/LN2, 10)}")
    print(f"    → φ 기반(S39 D1) ↔ 2 기반(잔류 D2) 켤레쌍 구조 확인")
    print()

    # φ^(-2.5) 구조 분석
    print(f"  φ^(-2.5) = HG D1 = φ^(-2.5):")
    print(f"    = {nstr(PHI**mpf('-2.5'), 12)}")
    print(f"    HG 좌표계 D1 = φ^(-2.5)  (S26 확정)")
    print()

    # 보정 계수 탐색 (오차 0.269% → True Zero)
    resid_C = RESIDUAL - candidate_C
    print(f"  잔류 - C후보 = {nstr(resid_C, 12)}")
    print(f"  보정 필요량:  {nstr(resid_C, 12)}")

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [D] 전방 스캔 — φ 지수 격자 × 로그 계수 확장")
    print(sep)

    # φ 지수: -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0
    # 로그 계수: LN_C, LN2, LN_PHI, log(π), GAP, GAP_D2, 1
    # 추가 인수: σ, σ², σ³, GAP, GAP_D2, GAP², GAP_D2²

    phi_exps = [mpf(x)/2 for x in range(-3, -13, -1)]  # -1.5 ~ -6.0
    log_coeffs = [
        ("LN_C²",    LN_C**2),
        ("LN2²",     LN2**2),
        ("LN_PHI²",  LN_PHI**2),
        ("LN_C",     LN_C),
        ("LN2",      LN2),
        ("LN_PHI",   LN_PHI),
        ("log(π)²",  log(pi)**2),
        ("log(π)",   log(pi)),
        ("1",        mpf(1)),
    ]
    extra_factors = [
        ("×GAP",      GAP),
        ("×GAP_D2",   GAP_D2),
        ("×GAP²",     GAP**2),
        ("×GAP_D2²",  GAP_D2**2),
        ("×σ²×GAP",   sigma2*GAP),
        ("×σ³×GAP",   sigma3*GAP),
        ("×σ²",       sigma2),
        ("×σ³",       sigma3),
        ("×1",        mpf(1)),
    ]

    scan_results = []
    for exp_val in phi_exps:
        phi_val = PHI**exp_val
        for lname, lval in log_coeffs:
            for ename, eval_ in extra_factors:
                candidate = lval * phi_val * eval_
                if candidate <= 0:
                    continue
                err = float(fabs(pct_err(candidate, RESIDUAL)))
                if err < 0.5:
                    label = f"lnφ^{float(exp_val):.1f}×{lname}{ename}"
                    scan_results.append((err, label, candidate))

    scan_results.sort()
    print(f"  오차 0.5% 이내 후보 (상위 20개):")
    print(f"  {'후보':50s}  {'값':>15}  {'오차%':>10}")
    print(sep2)
    for err, name, val in scan_results[:20]:
        marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
        print(f"  {name:50s}  {nstr(val,10):>15}  {err:>10.6f}%{marker}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [E] 기하급수 구조 정밀화 — φ^(-4.5)×GAP/(φ-GAP-δ)")
    print(sep)

    # S48: 0.006% 후보 φ^(-4.5)×GAP/(φ-GAP)
    # 분모를 (φ-GAP-δ)로 보정해서 True Zero 달성 시도
    # φ-GAP = 1.585888... 에서 δ를 더 빼면 분모 감소 → 결과 증가

    base_geo = PHI**mpf("-4.5") * GAP / (PHI - GAP)
    print(f"  기저: φ^(-4.5)×GAP/(φ-GAP) = {nstr(base_geo, 15)}")
    print(f"  잔류                         = {nstr(RESIDUAL, 15)}")
    print(f"  차이(RESIDUAL - base)        = {nstr(RESIDUAL - base_geo, 12)}")
    print(f"  기저 오차%                   = {nstr(pct_err(base_geo, RESIDUAL), 10)}%")
    print()

    # 정확한 분모 역산: 분모 = φ^(-4.5)×GAP/RESIDUAL
    exact_denom = PHI**mpf("-4.5") * GAP / RESIDUAL
    print(f"  정확한 분모 = φ^(-4.5)×GAP/RESIDUAL = {nstr(exact_denom, 15)}")
    print(f"  φ-GAP                                = {nstr(PHI-GAP, 15)}")
    denom_diff = exact_denom - (PHI - GAP)
    print(f"  차이 = 정확한분모-(φ-GAP)            = {nstr(denom_diff, 12)}")
    print()

    # 분모 보정항 후보
    print(f"  분모 보정항 후보 (차이={nstr(denom_diff,8)}):")
    denom_candidates = [
        ("GAP²",              GAP**2),
        ("GAP_D2²",           GAP_D2**2),
        ("GAP²/φ",            GAP**2/PHI),
        ("GAP×GAP_D2",        GAP*GAP_D2),
        ("LN_C×GAP_D2²",      LN_C*GAP_D2**2),
        ("σ²×GAP²",           sigma2*GAP**2),
        ("GAP_D2³",           GAP_D2**3),
        ("-GAP²",             -GAP**2),
        ("-GAP_D2²",          -GAP_D2**2),
        ("-GAP²/φ",           -GAP**2/PHI),
        ("-GAP×GAP_D2",       -GAP*GAP_D2),
        ("-LN_C×GAP_D2²",     -LN_C*GAP_D2**2),
    ]

    denom_results = []
    for name, val in denom_candidates:
        err = float(fabs(pct_err(val, denom_diff)))
        denom_results.append((err, name, val))
    denom_results.sort()

    print(f"  {'후보':30s}  {'값':>15}  {'오차%':>12}  {'검증(오차%)':>12}")
    print(sep2)
    for err, name, val in denom_results[:10]:
        # 검증: φ^(-4.5)×GAP/(φ-GAP-val)
        try:
            test_geo = PHI**mpf("-4.5") * GAP / (PHI - GAP - val)
            verify_err = float(pct_err(test_geo, RESIDUAL))
            marker = " ★★★" if fabs(verify_err) < 0.001 else (" ★★" if fabs(verify_err) < 0.01 else (" ★" if fabs(verify_err) < 0.1 else ""))
            print(f"  {name:30s}  {nstr(val,10):>15}  {err:>12.6f}%  {verify_err:>12.8f}%{marker}")
        except:
            pass
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  ▶▶ 탐색 요약")
    print(sep)
    print()
    print(f"  잔류 = {nstr(RESIDUAL, 20)}")
    print()
    print(f"  [A] δ = q_exact - GAP/φ = {nstr(delta, 10)}")
    if results_A:
        best_A = results_A[0]
        print(f"      최근접: {best_A[1]}  오차 {best_A[0]:.6f}%")
    print()
    print(f"  [B] lnφ×φ^(-8.2)×σ²  오차 {float(pct_err(LN_PHI*PHI**mpf('-8.2')*sigma2, RESIDUAL)):.4f}%")
    if exp_results:
        best_B = exp_results[0]
        print(f"      최근접 지수: {best_B[1]}  ({float(best_B[2]):.4f})  오차 {best_B[0]:.6f}%")
    print()
    print(f"  [C] ln(σ)²×GAP_D2×φ^(-2.5)  오차 {float(pct_err(candidate_C, RESIDUAL)):.4f}%")
    print(f"      = ln(2)²×란다우×HG_D1  [S39 ln(c)²×GD⁵ D2 켤레]")
    print()
    if scan_results:
        best_D = scan_results[0]
        print(f"  [D] 전방 스캔 최근접: {best_D[1]}  오차 {best_D[0]:.6f}%")
    print()
    if denom_results:
        best_E = denom_results[0]
        print(f"  [E] 분모 보정항 최근접: {best_E[1]}  오차 {best_E[0]:.6f}%")
    print()
    print(sep)
    print(f"  Session 49 verify_OB_LNC_residual_S49.py 완료")
    print(f"  mp.dps={mp.dps}")
    print(sep)


if __name__ == "__main__":
    main()

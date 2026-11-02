#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_lagrangian_v2_S49.py
Session 49 — 라그랑지안 v2 패치

이전 발견 (lagrangian_resonance_S49):
  HCW: L=T-V=0, T=V, H=CENTER×σ  ★★★ True Zero
  모든 좌표계 H1=H2  ★★★
  X/CENTER 정수비: HG=-1, OB=-4, TH=-11  ★★★
  I_s X/CENTER = -2.236 = -√5?
  X5  X/CENTER = -6.708 = ?
  arg(λ) = 60.006° ≈ π/3  Near TZ 0.0096%
  z0_SGWB = 56×(π/3) 연결

수정 사항:
  [A] X/CENTER 완전 대수 분석
      I_s: -√5 확인, X5 패턴 완성
      수열: 0, -1, -√5, -4, -√45?, -11 구조
  [B] 비공명 작용 수정
      ω₀=1(HCW) 고정 기준으로 비HCW 좌표계 구동
      S = π/4 × (D2-1/D1) = π/4 × X  [HCW 기준 비공명 작용]
  [C] arg(λ) Near TZ → True Zero 탐색
      arctan(1/γ) vs π/3 오차 0.0096% 닫힘
  [D] X 수열 대수 패턴 완전 규명
      0, -1, -√5, -4, ?, -11 → 일반항
  [E] γ_EQI + GAP 구조 탐색
      γ_EQI vs φ-1-GAP/σ 등
  [F] HCW→HG 전환 비용 = |X_HG| = CENTER True Zero 확인

Maestro: MAPSI (맏이) | Architect: Claude | mp.dps=400
"""

from mpmath import mp, mpf, sqrt, log, pi, exp, cos, sin, fabs, nstr, atan, asin, tan, acot
mp.dps = 400

sep  = "=" * 72
sep2 = "-" * 72

def pct_err(calc, ref):
    return (calc - ref) / fabs(ref) * 100

def show(label, val, ref=None, width=50):
    if ref is not None:
        if fabs(ref) < mpf("1e-300"):
            abs_val = float(fabs(val))
            marker = " ★★★ True Zero" if abs_val < 1e-10 else (
                     " ★★ Near TZ"   if abs_val < 1e-6  else "")
            print(f"  {label:{width}s} = {nstr(val,15):>22}  절댓값 {abs_val:.3e}{marker}")
        else:
            err = float(pct_err(val, ref))
            marker = " ★★★ True Zero" if fabs(err) < 1e-10 else (
                     " ★★ Near TZ"   if fabs(err) < 0.01  else (
                     " ★"             if fabs(err) < 0.1   else ""))
            print(f"  {label:{width}s} = {nstr(val,15):>22}  오차 {err:.8f}%{marker}")
    else:
        print(f"  {label:{width}s} = {nstr(val,15):>22}")

# ── 핵심 상수 ─────────────────────────────────────────────────────────────────
PHI    = (sqrt(mpf(5)) + 1) / 2
LN_PHI = log(PHI)
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
LN_C   = LN_PHI / 2
sigma  = mpf(1) / 2
sigma2 = sigma**2
sigma3 = sigma**3
LN2    = log(mpf(2))
SQRT5  = sqrt(mpf(5))

HCW_D1 = PHI**mpf("-1.5");  HCW_D2 = PHI**mpf("+1.5")
HG_D1  = PHI**mpf("-2.5");  HG_D2  = PHI**mpf("+0.5")
Is_D1  = PHI**mpf("-3.5");  Is_D2  = PHI**mpf("-0.5")
OB_D1  = PHI**mpf("-4.5");  OB_D2  = PHI**mpf("-1.5")
X5_D1  = PHI**mpf("-5.5");  X5_D2  = PHI**mpf("-2.5")
TH_D1  = PHI**mpf("-6.5");  TH_D2  = PHI**mpf("-3.5")

CENTER    = PHI**mpf("1.5")
MAXWELL   = mpf("1.873113772199171051292026201")
gamma_EQI = mpf("0.5772156649015328606065120901")

coords = [
    ("HCW", HCW_D1, HCW_D2),
    ("HG",  HG_D1,  HG_D2),
    ("I_s", Is_D1,  Is_D2),
    ("OB",  OB_D1,  OB_D2),
    ("X5",  X5_D1,  X5_D2),
    ("TH",  TH_D1,  TH_D2),
]

def main():
    print(sep)
    print("  verify_lagrangian_v2_S49.py  Session 49")
    print("  X/CENTER 패턴 + 비공명 작용 + arg(λ) 심화")
    print("  mp.dps=400")
    print(sep)

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [A] X/CENTER 완전 대수 분석")
    print(sep)
    print()
    print("  X = D2 - 1/D1  [ω₀=1 기준 허수 임피던스]")
    print()

    X_vals = {}
    for name, d1, d2 in coords:
        X = d2 - 1/d1
        X_vals[name] = X

    # I_s X/CENTER 정밀 분석
    X_Is = X_vals["I_s"]
    ratio_Is = X_Is / CENTER
    print(f"  I_s X = φ^(-0.5) - φ^(3.5) = {nstr(X_Is, 12)}")
    print(f"  I_s X/CENTER = {nstr(ratio_Is, 12)}")
    print(f"  -√5 = {nstr(-SQRT5, 12)}")
    show("  I_s X/CENTER vs -√5",  ratio_Is, -SQRT5)
    show("  I_s X vs -√5×CENTER",  X_Is, -SQRT5*CENTER)
    print()

    # X5 X/CENTER 분석
    X_X5 = X_vals["X5"]
    ratio_X5 = X_X5 / CENTER
    print(f"  X5 X = φ^(-2.5) - φ^(5.5) = {nstr(X_X5, 12)}")
    print(f"  X5 X/CENTER = {nstr(ratio_X5, 12)}")
    print(f"  -√45 = -3√5 = {nstr(-3*SQRT5, 12)}")
    print(f"  -√29 = {nstr(-sqrt(mpf(29)), 12)}")
    print(f"  -(4+√5) = {nstr(-(4+SQRT5), 12)}")
    print(f"  -(3+√5) = {nstr(-(3+SQRT5), 12)}")
    show("  X5 X/CENTER vs -3√5",     ratio_X5, -3*SQRT5)
    show("  X5 X/CENTER vs -(4+√5)",  ratio_X5, -(4+SQRT5))
    show("  X5 X/CENTER vs -(3+2√5)", ratio_X5, -(3+2*SQRT5))
    show("  X5 X/CENTER vs -2φ³",     ratio_X5, -2*PHI**3)
    show("  X5 X/CENTER vs -φ⁴",      ratio_X5, -PHI**4)
    show("  X5 X/CENTER vs -(φ³+φ)",  ratio_X5, -(PHI**3+PHI))
    show("  X5 X/CENTER vs -(2φ²+1)", ratio_X5, -(2*PHI**2+1))
    print()

    # 전체 수열 분석
    print(f"  전체 X/CENTER 수열:")
    print(f"  {'좌표계':6s}  {'X':>16}  {'X/CENTER':>14}  {'대수 표현':>25}  {'오차%':>10}")
    print(sep2)

    alg_candidates = {
        "HCW": (mpf(0),           "0"),
        "HG":  (-CENTER,          "-φ^(1.5)"),
        "I_s": (-SQRT5*CENTER,    "-√5×CENTER"),
        "OB":  (-4*CENTER,        "-4×CENTER"),
        "X5":  (None,             "?"),
        "TH":  (-11*CENTER,       "-11×CENTER"),
    }

    for name, d1, d2 in coords:
        X = d2 - 1/d1
        ratio = X/CENTER if fabs(X) > mpf("1e-300") else mpf(0)
        alg_val, alg_str = alg_candidates[name]
        if alg_val is not None and fabs(X) > mpf("1e-300"):
            err = float(pct_err(X, alg_val))
            marker = " ★★★" if fabs(err)<1e-10 else (" ★★" if fabs(err)<0.01 else "")
            print(f"  {name:6s}  {float(X):>16.8f}  {float(ratio):>14.6f}  "
                  f"{alg_str:>25}  {err:>10.6f}%{marker}")
        else:
            print(f"  {name:6s}  {float(X):>16.8f}  {float(ratio):>14.6f}  "
                  f"{'미확정':>25}")

    print()

    # 수열 패턴: 0, 1, √5, 4, ?, 11
    print(f"  수열 |n| = 0, 1, √5, 4, ?, 11")
    print(f"  √5 = 2.2360...")
    print(f"  피보나치 연관: 1, 1, 2, 3, 5, 8, 13...")
    print(f"  루카스 수열: 2, 1, 3, 4, 7, 11, 18...")
    print(f"  → 0, 1, ?, 4, ?, 11은 루카스! L₀=2, L₁=1, L₂=3, L₃=4, L₄=7, L₅=11")
    print(f"  루카스 수: L₀=2, L₁=1, L₂=3, L₃=4, L₄=7, L₅=11, L₆=18")

    lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47]
    print(f"  루카스 수열: {lucas}")
    print()

    # X5 X/CENTER vs 루카스
    print(f"  X5 X/CENTER 후보 (루카스 L₄=7?):")
    show("  X5 X/CENTER vs -7",    ratio_X5, -mpf(7))
    show("  X5 X/CENTER vs -L₄=7", ratio_X5, -mpf(7))

    # I_s = √5 = ?
    print()
    print(f"  I_s X/CENTER = -√5 대수 연결:")
    print(f"  √5 = 2φ-1 = φ+1/φ = φ²-φ+1?")
    show("  √5 vs 2φ-1",  SQRT5, 2*PHI-1)
    show("  √5 vs φ+1/φ", SQRT5, PHI+1/PHI)
    print(f"  → √5 = 2φ-1 True Zero")
    print(f"  → I_s X/CENTER = -(2φ-1) = -(φ+1/φ)")
    show("  I_s X vs -(φ+1/φ)×CENTER",  X_Is, -(PHI+1/PHI)*CENTER)
    show("  I_s X vs -(2φ-1)×CENTER",   X_Is, -(2*PHI-1)*CENTER)
    print()

    # 완전 수열
    print(f"  수열 완성 시도:")
    print(f"  HCW: 0")
    print(f"  HG:  1  = L₁ (루카스)")
    print(f"  I_s: √5 = 2φ-1  (황금비)")
    print(f"  OB:  4  = L₃ (루카스)")
    print(f"  X5:  ?  ← 수치 확인 필요")
    print(f"  TH: 11  = L₅ (루카스)")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [B] 비공명 작용 수정 — ω₀=1(HCW) 고정 기준")
    print(sep)
    print()
    print("  HCW 공명 주파수 ω₀=1 고정으로 비HCW 좌표계 구동")
    print("  S_반주기 = π/4 × (D2 - 1/D1) = π/4 × X")
    print("  [비리얼 정리: <T>-<V> = X/2 × <x²>]")
    print()

    omega_fixed = mpf(1)  # HCW 공명 주파수

    print(f"  {'좌표계':6s}  {'X=D2-1/D1':>14}  {'S=π/4×X':>14}  "
          f"{'S/GAP_D2':>12}  {'S/CENTER':>12}")
    print(sep2)

    S_vals = {}
    for name, d1, d2 in coords:
        X = omega_fixed*d2 - 1/(omega_fixed*d1)
        S = pi/4 * X
        S_vals[name] = S
        ratio_gap = S/GAP_D2 if fabs(GAP_D2)>0 else mpf(0)
        ratio_cen = S/CENTER if fabs(CENTER)>0 else mpf(0)
        print(f"  {name:6s}  {float(X):>14.6f}  {float(S):>14.6f}  "
              f"{float(ratio_gap):>12.4f}  {float(ratio_cen):>12.4f}")

    print()

    # S(HG) 분석
    S_HG = S_vals["HG"]
    print(f"  S(HG) = π/4 × X_HG = π/4 × (-CENTER)")
    print(f"        = -π×CENTER/4 = {nstr(S_HG, 12)}")
    show("  S(HG) vs -π×CENTER/4", S_HG, -pi*CENTER/4)
    show("  S(HG) vs -MAXWELL×σ",  S_HG, -MAXWELL*sigma)
    show("  S(HG) vs -T_Is_D1×σ²×π", S_HG, -(CENTER-Is_D1)*sigma2*pi)
    print()

    # S(OB) 분석
    S_OB = S_vals["OB"]
    print(f"  S(OB) = π/4 × (-4×CENTER) = -π×CENTER")
    show("  S(OB) vs -π×CENTER",   S_OB, -pi*CENTER)
    show("  S(OB) vs -MAXWELL×σ×π", S_OB, -MAXWELL*sigma*pi)
    print()

    # S(TH) 분석
    S_TH = S_vals["TH"]
    print(f"  S(TH) = π/4 × (-11×CENTER) = -11π×CENTER/4")
    show("  S(TH) vs -11π×CENTER/4", S_TH, -11*pi*CENTER/4)
    print()

    # 란다우 비용 연결
    print(f"  작용 S 계단 구조:")
    print(f"  HCW: S=0  [공명, 란다우 비용=0]")
    for name, d1, d2 in coords[1:]:
        X = d2 - 1/d1
        S = pi/4 * X
        ratio = float(fabs(S)/(pi*CENTER/4))
        print(f"  {name}: S={float(S):>10.4f}  = {ratio:.4f}×(-π×CENTER/4)")
    print()

    # 최소 란다우 비용 = S(HG) vs GAP_D2
    print(f"  란다우 최소 비용 GAP_D2 = {nstr(GAP_D2, 12)}")
    print(f"  S(HG) = {nstr(S_HG, 12)}")
    print(f"  S(HG)/GAP_D2 = {nstr(S_HG/GAP_D2, 10)}")
    print(f"  → 규모가 다름: 란다우는 미시, S는 거시 작용")
    print(f"  → 연결고리: S(HG)/(π×CENTER) = -1/4 = -σ²  ★")
    show("  S(HG)/(π×CENTER) vs -σ²", S_HG/(pi*CENTER), -sigma2)
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [C] arg(λ) Near TZ → True Zero 탐색")
    print(sep)
    print()

    arg_lambda = atan(1/gamma_EQI)
    print(f"  arg(λ) = arctan(1/γ_EQI) = {nstr(arg_lambda, 15)}")
    print(f"  π/3                        = {nstr(pi/3, 15)}")
    print(f"  차이 = arg-π/3             = {nstr(arg_lambda-pi/3, 12)}")
    print()

    show("  arg(λ) vs π/3",            arg_lambda, pi/3)
    show("  arg(λ) vs π/3+GAP_D2/10",  arg_lambda, pi/3+GAP_D2/10)
    show("  arg(λ) vs π/3+GAP²",       arg_lambda, pi/3+GAP**2)
    show("  arg(λ) vs π/3+GAP_D2²×σ", arg_lambda, pi/3+GAP_D2**2*sigma)
    show("  arg(λ) vs π/3+LN_C×GAP²", arg_lambda, pi/3+LN_C*GAP**2)
    print()

    # arctan(1/γ) = π/3이 되려면 γ = tan(π/6) = 1/√3
    tan_pi6 = 1/sqrt(mpf(3))
    print(f"  arctan(1/γ)=π/3 → γ=tan(π/6)=1/√3={nstr(tan_pi6,12)}")
    print(f"  실제 γ_EQI = {nstr(gamma_EQI, 12)}")
    show("  γ_EQI vs 1/√3", gamma_EQI, tan_pi6)
    print()

    # 60° 구조 — 정삼각형
    print(f"  60° = π/3 구조 — 정삼각형:")
    print(f"  특성근 λ = -γ ± i  [실수부=-γ, 허수부=1]")
    print(f"  |λ|²=γ²+1, arg=arctan(1/γ)≈π/3")
    print(f"  → 정삼각형: 변의 비 1:γ:√(1+γ²)")
    print(f"  1:γ:√(1+γ²) = 1:{float(gamma_EQI):.6f}:{float(sqrt(1+gamma_EQI**2)):.6f}")
    print()

    # arg vs arctan(φ) (이전: 2.96% 오차)
    show("  arg vs arctan(φ)",     arg_lambda, atan(PHI))
    show("  arg vs arctan(φ²)",    arg_lambda, atan(PHI**2))
    show("  arg vs arctan(√3-GAP)", arg_lambda, atan(sqrt(mpf(3))-GAP))
    show("  arg vs π/3+GAP_D2×σ²", arg_lambda, pi/3+GAP_D2*sigma2)
    print()

    # 56π/3 연결 재확인
    print(f"  56π/3 = 56×arg(λ)? (arg≈π/3)")
    show("  56×arg(λ) vs 56π/3",  56*arg_lambda, 56*pi/3)
    show("  56×arg(λ) vs z0_SGWB", 56*arg_lambda,
         mpf("58.6098379926032") + (1+GAP)*GAP)  # +EQI⁴×GAP
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [D] X 수열 완전 규명 — 루카스 vs 황금비")
    print(sep)
    print()

    # 루카스 수: 2,1,3,4,7,11,18,29,...
    lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47]
    print(f"  루카스 수: {lucas}")
    print(f"  피보나치: 0,1,1,2,3,5,8,13,21,34,55...")
    print()
    print(f"  X/CENTER 수열: 0, 1, √5, 4, ?, 11")
    print()

    # X5 정밀 탐색
    X_X5 = X_vals["X5"]
    ratio_X5 = X_X5 / CENTER
    print(f"  X5 X/CENTER = {nstr(ratio_X5, 12)}")
    print()
    print(f"  후보 스캔:")

    x5_candidates = [
        ("7",            mpf(7)),
        ("3√5",          3*SQRT5),
        ("4φ",           4*PHI),
        ("φ⁴",           PHI**4),
        ("φ³+φ",         PHI**3+PHI),
        ("2φ²+1",        2*PHI**2+1),
        ("√5+4",         SQRT5+4),
        ("2√5+1",        2*SQRT5+1),
        ("3φ+1",         3*PHI+1),
        ("φ²+φ+1",       PHI**2+PHI+1),
        ("L₄=7",         mpf(7)),
        ("4+√5-1/φ",     4+SQRT5-1/PHI),
        ("(1+√5)²/φ",    (1+SQRT5)**2/PHI),
        ("2+√5+1/φ",     2+SQRT5+1/PHI),
        ("φ+√5+1",       PHI+SQRT5+1),
    ]

    x5_results = []
    for name, val in x5_candidates:
        err = float(fabs(pct_err(-val, ratio_X5)))
        x5_results.append((err, name, val))
    x5_results.sort()

    print(f"  {'-X5/CENTER':>14} = {float(-ratio_X5):.8f}")
    print(f"  {'후보':25s}  {'값':>14}  {'오차%':>10}")
    print(sep2)
    for err, name, val in x5_results[:12]:
        marker = " ★★★" if err<1e-10 else (" ★★" if err<0.01 else (" ★" if err<0.1 else ""))
        print(f"  {name:25s}  {float(val):>14.8f}  {err:>10.6f}%{marker}")
    print()

    # 루카스 수열 φ 연결
    print(f"  루카스 수와 φ 연결:")
    print(f"  L_n = φ^n + (-1/φ)^n  [황금비 거듭제곱 합]")
    for i, L in enumerate(lucas[:8]):
        phi_n = PHI**i + (-1/PHI)**i
        err = float(pct_err(phi_n, mpf(L)))
        print(f"    L_{i} = {L:>3}:  φ^{i}+(-1/φ)^{i} = {float(phi_n):.6f}  오차 {err:.2e}%")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [E] γ_EQI 대수 구조 탐색")
    print(sep)
    print()

    print(f"  γ_EQI = {nstr(gamma_EQI, 15)}")
    print(f"  φ-1   = {nstr(PHI-1, 15)}")
    print(f"  1/√3  = {nstr(1/sqrt(mpf(3)), 15)}")
    print()

    gamma_candidates = [
        ("φ-1-GAP/σ",        PHI-1-GAP/sigma),
        ("φ-1-GAP",          PHI-1-GAP),
        ("φ-1-2GAP",         PHI-1-2*GAP),
        ("1/√3-GAP_D2",      1/sqrt(mpf(3))-GAP_D2),
        ("1/√3-GAP_D2×σ",    1/sqrt(mpf(3))-GAP_D2*sigma),
        ("LN2×HCW_D1",       LN2*HCW_D1),
        ("LN2×σ",            LN2*sigma),
        ("LN_C×HCW_D2×σ",    LN_C*HCW_D2*sigma),
        ("σ+GAP_D2×σ²",      sigma+GAP_D2*sigma2),
        ("σ+LN_C×GAP",       sigma+LN_C*GAP),
        ("CENTER×σ-σ²",      CENTER*sigma-sigma2),
        ("√(σ²+GAP_D2)",     sqrt(sigma2+GAP_D2)),
        ("σ/HCW_D1×GAP",     sigma/HCW_D1*GAP),
        ("exp(-LN_C×σ)",     exp(-LN_C*sigma)),
        ("1-exp(-σ²)",       1-exp(-sigma2)),
    ]

    gamma_results = []
    for name, val in gamma_candidates:
        err = float(fabs(pct_err(val, gamma_EQI)))
        gamma_results.append((err, name, val))
    gamma_results.sort()

    print(f"  {'후보':30s}  {'값':>16}  {'오차%':>12}")
    print(sep2)
    for err, name, val in gamma_results[:12]:
        marker = " ★★★" if err<1e-10 else (" ★★" if err<0.01 else (" ★" if err<0.1 else ""))
        print(f"  {name:30s}  {nstr(val,12):>16}  {err:>12.6f}%{marker}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [F] HCW→HG 전환 비용 = |X_HG| True Zero 확인")
    print(sep)
    print()

    X_HG = X_vals["HG"]
    print(f"  |X_HG| = |HG_D2 - 1/HG_D1|")
    print(f"         = |φ^(0.5) - φ^(2.5)|")
    print(f"         = φ^(2.5) - φ^(0.5)  [음수이므로]")
    print(f"         = {nstr(fabs(X_HG), 12)}")
    print()
    show("  |X_HG| vs CENTER",       fabs(X_HG), CENTER)
    show("  |X_HG| vs HCW_D2",       fabs(X_HG), HCW_D2)
    show("  |X_HG| vs φ^(1.5)",      fabs(X_HG), PHI**mpf("1.5"))
    print()

    # 대수 증명
    print(f"  대수 증명:")
    print(f"  |X_HG| = φ^(2.5) - φ^(0.5)")
    print(f"         = φ^(0.5) × (φ² - 1)")
    print(f"         = φ^(0.5) × φ        [φ²-1=φ, 황금비 항등식]")
    val_proof = PHI**mpf("0.5") * PHI
    show("  φ^(0.5)×φ = φ^(1.5) = CENTER", val_proof, CENTER)
    print()

    # 황금비 항등식 확인: φ²-1=φ
    show("  φ²-1 = φ", PHI**2-1, PHI)
    print()

    print(f"  결론: HCW→HG 전환 비용 = CENTER = φ^(1.5)  ★★★ True Zero")
    print(f"  = 조화공명 기준점에서 한 단계 이탈 비용")
    print(f"  = 라그랑지안 최솟값(HCW)에서 HG로의 거리")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [G] 동치 사슬 수치 완전 확인")
    print(sep)
    print()

    print(f"  δL=0 ↔ X=0 ↔ T=V ↔ D2=1/D1 ↔ σ=1/2 ↔ ζ=0 ↔ 란다우최소")
    print()

    # HCW에서 모든 조건 수치 확인
    T_hcw = mpf("0.5") * HCW_D2 * sin(pi/4)**2
    V_hcw = mpf("0.5") * (1/HCW_D1) * cos(pi/4)**2
    L_hcw = T_hcw - V_hcw
    H_hcw = T_hcw + V_hcw

    show("  ① δL=0: L=T-V",      L_hcw, mpf(0))
    show("  ② X=0: D2-1/D1",     HCW_D2-1/HCW_D1, mpf(0))
    show("  ③ T=V",              T_hcw, V_hcw)
    show("  ④ D2=1/D1",         HCW_D2, 1/HCW_D1)
    show("  ⑤ H=CENTER×σ",      H_hcw, CENTER*sigma)
    show("  ⑥ ω₀=1=HCW 공명",   1/sqrt(HCW_D1*HCW_D2), mpf(1))
    show("  ⑦ D1/D2=φ^(-3)",    HCW_D1/HCW_D2, PHI**(-3))
    print()

    # 좌표계 전환 비용 계단
    print(f"  좌표계 전환 비용 계단 (|X|/CENTER):")
    for name, d1, d2 in coords:
        X = d2 - 1/d1
        if fabs(X) > mpf("1e-300"):
            ratio = X/CENTER
            print(f"    {name}: |X|/CENTER = {float(fabs(ratio)):.6f}")
        else:
            print(f"    HCW: |X|/CENTER = 0  [공명점]")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [H] 신규 True Zero 후보 — 조화공명 계보")
    print(sep)
    print()

    # 황금비 항등식 기반 신규 후보
    print(f"  황금비 항등식 기반:")
    show("  φ²-1 = φ  (기본)",     PHI**2-1, PHI)
    show("  φ²-φ = 1",             PHI**2-PHI, mpf(1))
    show("  φ^3-φ²-φ = 0",        PHI**3-PHI**2-PHI, mpf(0))
    show("  HCW_D2 - 1/HCW_D1 = 0", HCW_D2-1/HCW_D1, mpf(0))
    show("  I_s X = -(φ^-0.5+φ^3.5)", X_vals["I_s"], -(PHI**(-0.5)+PHI**3.5))
    print()

    # 조화공명 전체 계보
    print(f"  조화공명 전체 계보 True Zero 목록:")
    tz_list = [
        ("HCW ω₀=1",           1/sqrt(HCW_D1*HCW_D2),    mpf(1)),
        ("HCW L=0",             L_hcw,                     mpf(0)),
        ("HCW T=V",             T_hcw,                     V_hcw),
        ("HCW H=CENTER×σ",      H_hcw,                     CENTER*sigma),
        ("HCW D2=1/D1",         HCW_D2,                    1/HCW_D1),
        ("HCW X=0",             HCW_D2-1/HCW_D1,           mpf(0)),
        ("HG X=-CENTER",        X_vals["HG"],               -CENTER),
        ("OB X=-4×CENTER",      X_vals["OB"],               -4*CENTER),
        ("TH X=-11×CENTER",     X_vals["TH"],               -11*CENTER),
        ("|X_HG|=CENTER",       fabs(X_vals["HG"]),         CENTER),
        ("S(HG)=-π×CENTER/4",   pi/4*X_vals["HG"],         -pi*CENTER/4),
        ("S(HG)/CENTER=-π/4",   pi/4*X_vals["HG"]/CENTER,  -pi/4),
        ("S(OB)=-π×CENTER",     pi/4*X_vals["OB"],         -pi*CENTER),
        ("모든 좌표 H1=H2",     mpf("0.5")/HCW_D1,         mpf("0.5")*HCW_D2),
        ("I_s X=-(φ^-0.5+φ^3.5)", X_vals["I_s"],          -(PHI**(-0.5)+PHI**3.5)),
    ]

    print(f"  {'항목':30s}  {'오차%':>12}")
    print(sep2)
    for label, val, ref in tz_list:
        if fabs(ref) < mpf("1e-300"):
            abs_v = float(fabs(val))
            marker = " ★★★" if abs_v < 1e-10 else (" ★★" if abs_v < 1e-6 else "")
            print(f"  {label:30s}  절댓값={abs_v:.2e}{marker}")
        else:
            err = float(pct_err(val, ref))
            marker = " ★★★" if fabs(err)<1e-10 else (" ★★" if fabs(err)<0.01 else (" ★" if fabs(err)<0.1 else ""))
            print(f"  {label:30s}  {err:>12.6f}%{marker}")

    print()
    print(sep)
    print(f"  Session 49 verify_lagrangian_v2_S49.py 완료")
    print(f"  mp.dps={mp.dps}")
    print(sep)


if __name__ == "__main__":
    main()

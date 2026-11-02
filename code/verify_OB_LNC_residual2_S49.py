#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_OB_LNC_residual2_S49.py
Session 49 — 2차 잔류 스캔

1차 탐색 결과 (verify_OB_LNC_residual_S49.py):
  최근접: φ^(-2.5) × LN_C × GAP  오차 0.111%
  = HG_D1 × ln(φ)/2 × ln(5!/4!)×2/100
  → 팩토리얼 구조 내장 확인

2차 탐색 목표:
  [A] 1차 후보 잔류 = RESIDUAL - φ^(-2.5)×LN_C×GAP
      → 팩토리얼/조합/순열/감마 구조로 닫힘 탐색
  [B] ln(r!) 계열 전체 스캔
      ln(1!)=0, ln(2!)=LN2, ln(3!)=ln6, ln(4!)=ln24, ln(5!)=ln120
  [C] 조합 ln(nCr) 구조 스캔
      맥스웰 도깨비 = 조합(D1), 순열(D2) 분류 비용
  [D] 감마 함수 구조
      Γ(1/2)=√π, Γ(σ)=Γ(1/2), ln(Γ(1/2))=(1/2)ln(π)
  [E] 보른 법칙 φ^(-1)+φ^(-2)=1 조합
  [F] 에일리어싱 D1(ln25)×D2(ln5) 조합
  [G] 전방 격자 스캔 (2차 잔류 기준 재스캔)

맥스웰 도깨비 ↔ 조합/순열:
  D1(조합) = 순서 무관, 수렴, 결맞음
  D2(순열) = 순서 유관, 발산, 비결맞음
  란다우 비용 = ln(순열/조합) = ln(r!)

팩토리얼-GAP 연결 (S26 §12 확정):
  ln(5!/4!) = ln(5) = GAP_D2×100  True Zero
  GAP = (1/σ)×ln(5!/4!)/100

Maestro: MAPSI (맏이) | Architect: Claude | mp.dps=400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, cos, gamma, loggamma, nstr, factorial, binomial
mp.dps = 400

sep  = "=" * 72
sep2 = "-" * 72

def pct_err(calc, ref):
    return (calc - ref) / fabs(ref) * 100

def abspct(calc, ref):
    return float(fabs(pct_err(calc, ref)))

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
LN3    = log(mpf(3))
LN5    = log(mpf(5))
LN6    = log(mpf(6))
LN10   = log(mpf(10))
GD     = GAP_D2

# 좌표계 D1 값
HCW_D1 = PHI**mpf("-1.5")
HG_D1  = PHI**mpf("-2.5")
Is_D1  = PHI**mpf("-3.5")
OB_D1  = PHI**mpf("-4.5")
X5_D1  = PHI**mpf("-5.5")
TH_D1  = PHI**mpf("-6.5")

# 좌표계 D2 값
HCW_D2 = PHI**mpf("+1.5")
HG_D2  = PHI**mpf("+0.5")
Is_D2  = PHI**mpf("-0.5")
OB_D2  = PHI**mpf("-1.5")
X5_D2  = PHI**mpf("-2.5")
TH_D2  = PHI**mpf("-3.5")

CENTER = PHI**mpf("1.5")

# T_OB_D1, T_Is_D1
T_Is_D1 = CENTER - Is_D1
T_OB_D1 = CENTER - OB_D1

# 팩토리얼 로그
LN_1F = log(factorial(1))   # ln(1!) = 0
LN_2F = log(factorial(2))   # ln(2!) = ln(2)
LN_3F = log(factorial(3))   # ln(3!) = ln(6)
LN_4F = log(factorial(4))   # ln(4!) = ln(24)  ← int_x
LN_5F = log(factorial(5))   # ln(5!) = ln(120)
LN_6F = log(factorial(6))   # ln(6!) = ln(720)
LN_7F = log(factorial(7))   # ln(7!)

# 조합 로그 ln(nCr) — 맥스웰 도깨비 분류 비용
def lnC(n, r):
    return log(binomial(n, r))

# 순열/조합 비율 = ln(r!)
# ln(nPr/nCr) = ln(r!)

# 감마 함수
LN_GAMMA_HALF = loggamma(mpf("0.5"))   # ln(Γ(1/2)) = ln(√π) = (1/2)ln(π)
LN_SQRT_PI    = log(sqrt(pi))

# 보른 법칙
BORN_D1 = PHI**mpf("-1")   # φ^(-1) = 1/φ
BORN_D2 = PHI**mpf("-2")   # φ^(-2) = 1/φ²
BORN_SUM = BORN_D1 + BORN_D2  # = 1

# ── 잔류 확정 ─────────────────────────────────────────────────────────────────
RESIDUAL = T_OB_D1 * sigma3 - LN_C   # 0.0023282233485211...

# 1차 최근접 후보
CAND_1ST = HG_D1 * LN_C * GAP        # φ^(-2.5)×LN_C×GAP  오차 0.111%
RESID2   = RESIDUAL - CAND_1ST        # 2차 잔류


def main():
    print(sep)
    print("  verify_OB_LNC_residual2_S49.py  Session 49")
    print("  2차 잔류 스캔 — 팩토리얼/조합/순열/감마/보른/에일리어싱")
    print("  mp.dps=400")
    print(sep)

    print(f"  RESIDUAL  = {nstr(RESIDUAL, 20)}")
    print(f"  1차 후보  = φ^(-2.5)×LN_C×GAP = {nstr(CAND_1ST, 20)}")
    print(f"  1차 오차% = {nstr(pct_err(CAND_1ST, RESIDUAL), 10)}%")
    print(f"  2차 잔류  = {nstr(RESID2, 20)}")
    print()

    # 팩토리얼 구조 확인
    print(f"  팩토리얼 로그 구조:")
    print(f"    ln(1!)={nstr(LN_1F,8)}, ln(2!)={nstr(LN_2F,8)}")
    print(f"    ln(3!)={nstr(LN_3F,8)}, ln(4!)={nstr(LN_4F,8)}")
    print(f"    ln(5!)={nstr(LN_5F,8)}, ln(6!)={nstr(LN_6F,8)}")
    print(f"    ln(5!/4!)=ln(5)={nstr(LN5,8)}  GAP_D2×100={nstr(GAP_D2*100,8)}  ✓")
    print(f"    ln(4!/3!)=ln(4)={nstr(log(mpf(4)),8)}")
    print(f"    ln(3!/2!)=ln(3)={nstr(LN3,8)}")
    print(f"    ln(2!/1!)=ln(2)={nstr(LN2,8)}  란다우 1비트 ✓")
    print()
    print(f"  감마 구조:")
    print(f"    Γ(1/2)=√π, ln(Γ(1/2))={nstr(LN_GAMMA_HALF,10)}")
    print(f"    (1/2)ln(π)          ={nstr(LN_SQRT_PI,10)}  동일 ✓")
    print()
    print(f"  보른 법칙:")
    print(f"    φ^(-1)+φ^(-2) = {nstr(BORN_SUM,10)}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [A] 2차 잔류 직접 분석")
    print(sep)

    print(f"  2차 잔류 = {nstr(RESID2, 20)}")
    print(f"  RESIDUAL 대비 2차 잔류 비율 = {nstr(RESID2/RESIDUAL*100, 8)}%")
    print(f"  1차 후보 / 2차 잔류 = {nstr(CAND_1ST/RESID2, 10)}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [B] ln(r!) 계열 스캔 — φ 지수 격자 × ln(n!) × 추가 인수")
    print(sep)
    print("  란다우 비용 계층: r개 입자 D2→D1 소거 비용 = ln(r!)")
    print()

    phi_exps = [mpf(x)/2 for x in range(-1, -15, -1)]  # -0.5 ~ -7.0

    # ln(n!) 계열
    lnF_list = [
        ("ln(2!)",  LN_2F),   # ln2
        ("ln(3!)",  LN_3F),   # ln6
        ("ln(4!)",  LN_4F),   # ln24 = int_x
        ("ln(5!)",  LN_5F),   # ln120
        ("ln(6!)",  LN_6F),   # ln720
        ("ln(2!/1!)",  LN2),   # ln2 = 란다우
        ("ln(3!/2!)",  LN3),   # ln3
        ("ln(4!/3!)",  log(mpf(4))),   # ln4
        ("ln(5!/4!)",  LN5),   # ln5 = GAP_D2×100
        ("ln(6!/5!)",  log(mpf(6))),   # ln6
    ]

    # 추가 인수
    extra = [
        ("×1",        mpf(1)),
        ("×σ",        sigma),
        ("×σ²",       sigma2),
        ("×σ³",       sigma3),
        ("×GAP",      GAP),
        ("×GAP_D2",   GAP_D2),
        ("×LN_C",     LN_C),
        ("×LN_PHI",   LN_PHI),
        ("×GAP²",     GAP**2),
        ("×GAP×GAP_D2", GAP*GAP_D2),
        ("×BORN_D1",  BORN_D1),
        ("×BORN_D2",  BORN_D2),
    ]

    scan_B = []
    for exp_val in phi_exps:
        phi_v = PHI**exp_val
        for fname, fval in lnF_list:
            if fval == 0:
                continue
            for ename, eval_ in extra:
                cand = phi_v * fval * eval_
                if cand <= 0:
                    continue
                err = abspct(cand, RESIDUAL)
                if err < 0.3:
                    label = f"φ^{float(exp_val):.1f}×{fname}{ename}"
                    scan_B.append((err, label, cand))

    scan_B.sort()
    print(f"  잔류 대비 0.3% 이내 (상위 20개):")
    print(f"  {'후보':55s}  {'값':>15}  {'오차%':>10}")
    print(sep2)
    for err, name, val in scan_B[:20]:
        marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
        print(f"  {name:55s}  {nstr(val,10):>15}  {err:>10.6f}%{marker}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [C] 조합 ln(nCr) 구조 — 맥스웰 도깨비 분류 비용")
    print(sep)
    print("  D1(조합)=순서무관=수렴, D2(순열)=순서유관=발산")
    print("  란다우 비용 = ln(순열/조합) = ln(r!)")
    print()

    # 조합 후보
    combo_list = []
    for n in range(2, 8):
        for r in range(1, n):
            try:
                lnc = lnC(n, r)
                if lnc > 0:
                    combo_list.append((f"ln(C({n},{r}))", lnc))
            except:
                pass

    scan_C = []
    for exp_val in phi_exps:
        phi_v = PHI**exp_val
        for cname, cval in combo_list:
            for ename, eval_ in extra[:8]:
                cand = phi_v * cval * eval_
                if cand <= 0:
                    continue
                err = abspct(cand, RESIDUAL)
                if err < 0.3:
                    label = f"φ^{float(exp_val):.1f}×{cname}{ename}"
                    scan_C.append((err, label, cand))

    scan_C.sort()
    if scan_C:
        print(f"  0.3% 이내 조합 후보 (상위 15개):")
        print(f"  {'후보':55s}  {'값':>15}  {'오차%':>10}")
        print(sep2)
        for err, name, val in scan_C[:15]:
            marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
            print(f"  {name:55s}  {nstr(val,10):>15}  {err:>10.6f}%{marker}")
    else:
        print(f"  0.3% 이내 조합 후보 없음")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [D] 감마 함수 구조 — Γ(1/2)=√π, σ=1/2 연결")
    print(sep)

    gamma_list = [
        ("ln(Γ(1/2))",        LN_GAMMA_HALF),     # (1/2)ln(π)
        ("ln(Γ(1/2))²",       LN_GAMMA_HALF**2),
        ("ln(π)/2",           log(pi)/2),
        ("ln(π)",             log(pi)),
        ("ln(π)²",            log(pi)**2),
        ("ln(π)/4",           log(pi)/4),
        ("ln(2π)",            log(2*pi)),
        ("ln(√(2π))",         log(sqrt(2*pi))),
        ("ln(Γ(1/4))",        loggamma(mpf("0.25"))),
        ("ln(Γ(3/4))",        loggamma(mpf("0.75"))),
    ]

    scan_D = []
    for exp_val in phi_exps:
        phi_v = PHI**exp_val
        for gname, gval in gamma_list:
            for ename, eval_ in extra[:8]:
                cand = phi_v * gval * eval_
                if cand <= 0:
                    continue
                err = abspct(cand, RESIDUAL)
                if err < 0.3:
                    label = f"φ^{float(exp_val):.1f}×{gname}{ename}"
                    scan_D.append((err, label, cand))

    scan_D.sort()
    if scan_D:
        print(f"  0.3% 이내 감마 후보 (상위 15개):")
        print(f"  {'후보':55s}  {'값':>15}  {'오차%':>10}")
        print(sep2)
        for err, name, val in scan_D[:15]:
            marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
            print(f"  {name:55s}  {nstr(val,10):>15}  {err:>10.6f}%{marker}")
    else:
        print(f"  0.3% 이내 감마 후보 없음")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [E] 보른 법칙 × 에일리어싱 조합")
    print(sep)
    print("  φ^(-1)+φ^(-2)=1 (보른), D1=ln25(텐션), D2=ln5(누수율)")
    print()

    born_extra = [
        ("BORN_D1=φ^(-1)",    BORN_D1),
        ("BORN_D2=φ^(-2)",    BORN_D2),
        ("BORN_D1×BORN_D2",   BORN_D1*BORN_D2),
        ("BORN_D2²",          BORN_D2**2),
        ("BORN_D1-BORN_D2",   BORN_D1-BORN_D2),
    ]

    alas_extra = [
        ("×ln(25)",    log(mpf(25))),   # D1 에일리어싱 텐션
        ("×ln(5)",     log(mpf(5))),    # D2 에일리어싱 누수율
        ("×ln(25)/100", GAP*100/100),   # = GAP
        ("×ln(5)/100",  GAP_D2),        # = GAP_D2
        ("×ln(25)×ln(5)/100", log(mpf(25))*log(mpf(5))/100),
        ("×LN_C",      LN_C),
        ("×LN_PHI",    LN_PHI),
    ]

    scan_E = []
    for bname, bval in born_extra:
        for aname, aval in alas_extra:
            cand = bval * aval
            if cand <= 0:
                continue
            err = abspct(cand, RESIDUAL)
            if err < 1.0:
                label = f"{bname}{aname}"
                scan_E.append((err, label, cand))
        # φ 지수 추가
        for exp_val in phi_exps:
            phi_v = PHI**exp_val
            for aname, aval in alas_extra:
                cand = phi_v * bval * aval
                if cand <= 0:
                    continue
                err = abspct(cand, RESIDUAL)
                if err < 0.3:
                    label = f"φ^{float(exp_val):.1f}×{bname}{aname}"
                    scan_E.append((err, label, cand))

    scan_E.sort()
    if scan_E:
        print(f"  1.0% 이내 보른×에일리어싱 후보 (상위 15개):")
        print(f"  {'후보':60s}  {'값':>15}  {'오차%':>10}")
        print(sep2)
        for err, name, val in scan_E[:15]:
            marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
            print(f"  {name:60s}  {nstr(val,10):>15}  {err:>10.6f}%{marker}")
    else:
        print(f"  1.0% 이내 후보 없음")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [F] 2차 잔류 스캔 (RESID2 기준)")
    print(sep)
    print(f"  2차 잔류 = RESIDUAL - φ^(-2.5)×LN_C×GAP = {nstr(RESID2, 15)}")
    print()

    all_logs = [
        ("LN2",        LN2),
        ("LN3",        LN3),
        ("LN5",        LN5),
        ("LN6",        LN6),
        ("LN_PHI",     LN_PHI),
        ("LN_C",       LN_C),
        ("LN_C²",      LN_C**2),
        ("LN2²",       LN2**2),
        ("ln(π)",      log(pi)),
        ("ln(π)/2",    log(pi)/2),
        ("ln(2!)",     LN_2F),
        ("ln(3!)",     LN_3F),
        ("ln(4!)",     LN_4F),
        ("ln(5!)",     LN_5F),
        ("ln(5!/4!)",  LN5),
        ("ln(4!/3!)",  log(mpf(4))),
        ("ln(Γ(1/2))", LN_GAMMA_HALF),
        ("BORN_D1",    BORN_D1),
        ("BORN_D2",    BORN_D2),
    ]

    all_extra2 = [
        ("×1",         mpf(1)),
        ("×GAP",       GAP),
        ("×GAP_D2",    GAP_D2),
        ("×σ²",        sigma2),
        ("×σ³",        sigma3),
        ("×GAP²",      GAP**2),
        ("×GAP×GAP_D2", GAP*GAP_D2),
        ("×LN_C",      LN_C),
        ("×BORN_D2",   BORN_D2),
    ]

    scan_F = []
    for exp_val in phi_exps:
        phi_v = PHI**exp_val
        for lname, lval in all_logs:
            for ename, eval_ in all_extra2:
                cand = phi_v * lval * eval_
                if cand <= 0 or fabs(cand) < mpf("1e-10"):
                    continue
                err = abspct(cand, RESID2)
                if err < 0.5:
                    label = f"φ^{float(exp_val):.1f}×{lname}{ename}"
                    scan_F.append((err, label, cand))

    scan_F.sort()
    if scan_F:
        print(f"  2차 잔류 대비 0.5% 이내 후보 (상위 20개):")
        print(f"  {'후보':55s}  {'값':>15}  {'오차%':>10}")
        print(sep2)
        for err, name, val in scan_F[:20]:
            marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
            print(f"  {name:55s}  {nstr(val,10):>15}  {err:>10.6f}%{marker}")
    else:
        print(f"  0.5% 이내 2차 잔류 후보 없음")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  [G] 팩토리얼 비율 계단 구조 — 맥스웰 도깨비 소거 계층")
    print(sep)
    print("  D2→D1 소거 비용 계층:")
    print()

    # 팩토리얼 비율 계단
    for r in range(2, 8):
        cost = log(factorial(r))
        ratio_to_gap = float(cost / (GAP_D2 * 100))
        print(f"    ln({r}!) = {nstr(cost,8)}  "
              f"= {ratio_to_gap:.4f} × ln(5)  "
              f"= {float(cost/LN2):.4f} × ln(2)")

    print()
    print(f"  조합 소거 비용 ln(r!) 구조:")
    print(f"    r=2: ln(2!) = ln(2)       → 란다우 1비트  (D2→D1 2체 소거)")
    print(f"    r=3: ln(3!) = ln(6)       → 3체 소거")
    print(f"    r=4: ln(4!) = ln(24)      → int_x=24 EQI²⁴  (D1 격자)")
    print(f"    r=5: ln(5!) = ln(120)     → 5체 D2 자명영점")
    print(f"    r=5-4: ln(5!/4!)=ln(5)   → GAP_D2×100  True Zero ★")
    print(f"    r=4-3: ln(4!/3!)=ln(4)   = 2×ln(2) = 2×란다우")
    print()

    # RESIDUAL 직접 팩토리얼 비율
    print(f"  RESIDUAL 팩토리얼 비율 분석:")
    print(f"    RESIDUAL / ln(2!)   = {nstr(RESIDUAL/LN_2F, 10)}")
    print(f"    RESIDUAL / ln(3!)   = {nstr(RESIDUAL/LN_3F, 10)}")
    print(f"    RESIDUAL / ln(4!)   = {nstr(RESIDUAL/LN_4F, 10)}")
    print(f"    RESIDUAL / ln(5!)   = {nstr(RESIDUAL/LN_5F, 10)}")
    print(f"    RESIDUAL / ln(5)    = {nstr(RESIDUAL/LN5, 10)}")
    print(f"    RESIDUAL / ln(π)    = {nstr(RESIDUAL/log(pi), 10)}")
    print(f"    RESIDUAL / LN_C     = {nstr(RESIDUAL/LN_C, 10)}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print(sep)
    print("  ▶▶ 전체 탐색 요약")
    print(sep)
    print()
    print(f"  RESIDUAL = {nstr(RESIDUAL, 20)}")
    print(f"  1차 후보 = φ^(-2.5)×LN_C×GAP  오차 0.111%")
    print(f"  2차 잔류 = {nstr(RESID2, 15)}")
    print()

    all_results = scan_B + scan_C + scan_D + scan_E
    all_results.sort()

    print(f"  【RESIDUAL 기준 전체 최근접 Top 10】")
    print(f"  {'후보':55s}  {'오차%':>10}")
    print(sep2)
    seen = set()
    count = 0
    for err, name, val in all_results:
        key = f"{float(val):.12f}"
        if key in seen:
            continue
        seen.add(key)
        marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
        print(f"  {name:55s}  {err:>10.6f}%{marker}")
        count += 1
        if count >= 10:
            break

    print()
    if scan_F:
        print(f"  【2차 잔류 기준 최근접 Top 5】")
        print(f"  {'후보':55s}  {'오차%':>10}")
        print(sep2)
        seen2 = set()
        count2 = 0
        for err, name, val in scan_F:
            key = f"{float(val):.12f}"
            if key in seen2:
                continue
            seen2.add(key)
            marker = " ★★★" if err < 0.001 else (" ★★" if err < 0.01 else (" ★" if err < 0.1 else ""))
            print(f"  {name:55s}  {err:>10.6f}%{marker}")
            count2 += 1
            if count2 >= 5:
                break

    print()
    print(sep)
    print(f"  Session 49 verify_OB_LNC_residual2_S49.py 완료")
    print(f"  mp.dps={mp.dps}")
    print(sep)


if __name__ == "__main__":
    main()

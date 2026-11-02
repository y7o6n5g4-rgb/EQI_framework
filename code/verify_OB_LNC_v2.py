#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_OB_LNC_v2.py
Session 48 — T_OB_D1×σ³ vs LN_C 대수 기원 탐색 v2

v1 수정사항:
  T_OB_D1 > T_Is_D1 (부호 수정)
  T_OB_D1 = T_Is_D1 + φ^(-6.5)  [φ-1=1/φ 황금비 항등식]

탐색:
  [A] 수정된 분해: T_OB_D1×σ³ = T_Is_D1×σ³ + φ^(-6.5)×σ³
  [B] φ^(-6.5) 추가 후 새 잔류 스캔
  [C] GAP_D2/16 후보 정밀 확인
  [D] S39 ln(c)² 구조와의 연결
  [E] 전체 좌표계 분해 계단 구조

Maestro: MAPSI (맏이) | Architect: Claude | mp.dps=400
"""

from mpmath import mp, mpf, sqrt, log, pi, floor, fabs, exp, sin, nstr
mp.dps = 400

sep  = "=" * 70
sep2 = "-" * 70

def pct_err(calc, ref):
    return (calc - ref) / fabs(ref) * 100

PHI    = (sqrt(mpf(5)) + 1) / 2
LN_PHI = log(PHI)
GAP    = log(mpf(25)) / 100
GAP_D2 = log(mpf(5))  / 100
c_EQI  = sqrt(PHI)
LN_C   = LN_PHI / 2
sigma  = mpf(1) / 2
sigma3 = sigma**3
LN2    = log(mpf(2))

CENTER  = PHI**mpf("1.5")
T_Is_D1 = CENTER - PHI**mpf("-3.5")
T_OB_D1 = CENTER - PHI**mpf("-4.5")

# S39: MAXWELL 5차 공식
GD = GAP_D2
MAXWELL_5th = (T_Is_D1
    + 2*exp(GD)*GD**2
    + (LN2/2)*GD**3
    + (LN2/19 + GD/26)*GD**4
    + (mpf(1)/274)*LN_C**2*GD**5)

def main():
    print(sep)
    print("  verify_OB_LNC_v2.py  Session 48")
    print("  T_OB_D1×σ³ vs LN_C 탐색 v2 (부호 수정)  mp.dps=400")
    print(sep)

    print(f"  T_Is_D1 = φ^1.5-φ^(-3.5) = {nstr(T_Is_D1, 15)}")
    print(f"  T_OB_D1 = φ^1.5-φ^(-4.5) = {nstr(T_OB_D1, 15)}")
    print(f"  T_OB > T_Is: {T_OB_D1 > T_Is_D1}")
    print(f"  차이 = T_OB - T_Is = {nstr(T_OB_D1-T_Is_D1, 12)}")
    print()

    # ── [A] 수정된 분해 ───────────────────────────────────────────────────────
    print("[A] 수정된 분해: T_OB_D1 = T_Is_D1 + φ^(-6.5)")
    print(sep2)

    # φ-1 = 1/φ (황금비 항등식)
    # φ^(-3.5) - φ^(-4.5) = φ^(-4.5)×(φ-1) = φ^(-4.5)×φ^(-1) = φ^(-5.5)
    # 그런데 T_OB - T_Is = -(φ^(-4.5)) + φ^(-3.5)
    #                     = φ^(-3.5) - φ^(-4.5)
    #                     = φ^(-4.5)×(φ-1)
    #                     = φ^(-4.5)×(1/φ)
    #                     = φ^(-5.5)

    diff_exact = T_OB_D1 - T_Is_D1
    phi_m55 = PHI**mpf("-5.5")
    phi_m65 = PHI**mpf("-6.5")

    print(f"  T_OB - T_Is = {nstr(diff_exact, 15)}")
    print(f"  φ^(-5.5)    = {nstr(phi_m55, 15)}")
    print(f"  오차%        = {nstr(pct_err(diff_exact, phi_m55), 10)}%")
    print()

    # T_OB_D1×σ³ = T_Is_D1×σ³ + φ^(-5.5)×σ³
    val_Is_sigma3  = T_Is_D1 * sigma3
    val_phi55_s3   = phi_m55 * sigma3
    val_recon      = val_Is_sigma3 + val_phi55_s3

    print(f"  T_Is_D1×σ³           = {nstr(val_Is_sigma3, 15)}")
    print(f"  φ^(-5.5)×σ³          = {nstr(val_phi55_s3, 15)}")
    print(f"  합                   = {nstr(val_recon, 15)}")
    print(f"  T_OB_D1×σ³ 실측      = {nstr(T_OB_D1*sigma3, 15)}")
    print(f"  재구성 오차%          = {nstr(pct_err(val_recon, T_OB_D1*sigma3), 10)}%")
    print()

    # 새 잔류: T_OB_D1×σ³ - LN_C
    # = (T_Is_D1 + φ^(-5.5))×σ³ - LN_C
    residual_v2 = T_OB_D1*sigma3 - LN_C
    residual_Is = val_Is_sigma3 - LN_C      # I_s 잔류 (음수)
    residual_phi55 = val_phi55_s3            # φ^(-5.5) 기여

    print(f"  LN_C                 = {nstr(LN_C, 15)}")
    print(f"  T_OB_D1×σ³ - LN_C   = {nstr(residual_v2, 15)}")
    print(f"  T_Is_D1×σ³ - LN_C   = {nstr(residual_Is, 15)}  (음수)")
    print(f"  φ^(-5.5)×σ³          = {nstr(residual_phi55, 15)}")
    print()
    print(f"  → φ^(-5.5)×σ³가 I_s 잔류(음수)를 초과 보정:")
    print(f"    초과분 = φ^(-5.5)×σ³ + (T_Is_D1×σ³ - LN_C)")
    print(f"           = {nstr(residual_phi55 + residual_Is, 12)}")
    print(f"           = 최종 잔류 {nstr(residual_v2, 12)}")

    # ── [B] 새 잔류 스캔 ──────────────────────────────────────────────────────
    print()
    print("[B] 새 잔류 정밀 스캔")
    print(sep2)
    print(f"  잔류 = T_OB×σ³ - LN_C = {nstr(residual_v2, 15)}")
    print()

    # I_s 잔류와 φ^(-5.5) 기여 분리
    print(f"  구조 분리:")
    print(f"  LN_C - T_Is_D1×σ³    = {nstr(-residual_Is, 12)}  (I_s 부족분)")
    print(f"  φ^(-5.5)×σ³           = {nstr(residual_phi55, 12)}")
    print(f"  OB 잔류 = φ(-5.5)×σ³ - (LN_C-T_Is×σ³)")
    print(f"          = {nstr(residual_phi55, 12)} - {nstr(-residual_Is, 12)}")
    print(f"          = {nstr(residual_v2, 12)}")
    print()

    # 잔류 후보 스캔 (확장)
    candidates = [
        ("GAP_D2/16",             GAP_D2/16),
        ("GAP_D2/12",             GAP_D2/12),
        ("GAP_D2/8",              GAP_D2/8),
        ("GAP/16",                GAP/16),
        ("LN_C×GAP_D2²",          LN_C*GAP_D2**2),
        ("LN_C²×GAP_D2",          LN_C**2*GAP_D2),
        ("LN_C³",                 LN_C**3),
        ("σ³×GAP_D2",             sigma3*GAP_D2),
        ("σ²×GAP_D2",             sigma**2*GAP_D2),
        ("σ×GAP_D2",              sigma*GAP_D2),
        ("φ^(-5.5)×GAP",          phi_m55*GAP),
        ("φ^(-5.5)×GAP_D2",       phi_m55*GAP_D2),
        ("φ^(-6.5)×σ³",           phi_m65*sigma3),
        ("φ^(-6.5)/8",            phi_m65/8),
        ("φ^(-5.5)×LN_C",         phi_m55*LN_C),
        ("φ^(-4.5)×GAP_D2",       PHI**mpf("-4.5")*GAP_D2),
        ("GAP_D2²/GAP",           GAP_D2**2/GAP),
        ("GAP_D2³/LN_C",          GAP_D2**3/LN_C),
        ("LN_C×GAP²",             LN_C*GAP**2),
        ("σ³×GAP×(1-GAP)",        sigma3*GAP*(1-GAP)),
        # S39 ln(c)² 계열
        ("LN_C²/GAP",             LN_C**2/GAP),
        ("LN_C²/GAP_D2",          LN_C**2/GAP_D2),
        ("LN_C²×σ³",              LN_C**2*sigma3),
        ("(1/274)×LN_C²×GD",      LN_C**2*GAP_D2/274),
        ("(1/137)×LN_C²×GD",      LN_C**2*GAP_D2/137),
        ("(1/26)×LN_C×GD²",       LN_C*GAP_D2**2/26),
        ("(1/19)×LN_C×GD²",       LN_C*GAP_D2**2/19),
        # 복합
        ("φ^(-5.5)×σ³×GAP_D2",    phi_m55*sigma3*GAP_D2),
        ("φ^(-5.5)×LN_C×σ³",      phi_m55*LN_C*sigma3),
        ("GAP_D2/(2π²)",           GAP_D2/(2*pi**2)),
        ("LN_C/pi³",               LN_C/pi**3),
        ("LN2×GAP_D2/pi",          LN2*GAP_D2/pi),
        ("GAP_D2×LN_C/pi",         GAP_D2*LN_C/pi),
        ("σ³×LN_C×GAP",            sigma3*LN_C*GAP),
        ("σ³×LN_C×GAP_D2",         sigma3*LN_C*GAP_D2),
    ]

    results = [(float(fabs(pct_err(v, residual_v2))), n, float(v))
               for n, v in candidates if v > 0]
    results.sort()

    print(f"  {'후보':35s}  {'값':14}  {'오차%':12}")
    print(sep2)
    for err, name, val in results[:20]:
        marker = "★★★ True Zero" if err < 0.01 else \
                 ("★★ Near TZ" if err < 1 else \
                 ("★" if err < 10 else ""))
        print(f"  {name:35s}  {val:14.8f}  {err:12.6f}%  {marker}")

    # ── [C] GAP_D2/16 정밀 확인 ──────────────────────────────────────────────
    print()
    print("[C] 최근접 후보 정밀 분석")
    print(sep2)

    top5 = results[:5]
    for err, name, val in top5:
        mpval = None
        for nm, mv in candidates:
            if nm == name:
                mpval = mv
                break
        if mpval is not None:
            print(f"  {name}:")
            print(f"    값   = {nstr(mpval, 15)}")
            print(f"    잔류 = {nstr(residual_v2, 15)}")
            print(f"    오차 = {nstr(pct_err(mpval, residual_v2), 10)}%")
            print(f"    비율 = {nstr(residual_v2/mpval, 10)}")
            print()

    # ── [D] S39 ln(c)² 구조 연결 ─────────────────────────────────────────────
    print("[D] S39 MAXWELL 5차 항 구조와의 연결")
    print(sep2)
    print(f"  S39 §5: (1/274)×LN_C²×GD⁵")
    print(f"  LN_C²×GD⁵ = {nstr(LN_C**2*GAP_D2**5, 12)}")
    print(f"  /274       = {nstr(LN_C**2*GAP_D2**5/274, 12)}")
    print()
    print(f"  S39 MAXWELL 항4/항5 비율:")
    maxwell_4th_coeff = LN2/19 + GAP_D2/26
    maxwell_5th_coeff = LN_C**2/274
    print(f"  항4 계수: ln2/19+GD/26 = {nstr(maxwell_4th_coeff, 10)}")
    print(f"  항5 계수: LN_C²/274    = {nstr(maxwell_5th_coeff, 10)}")
    print(f"  항5/항4               = {nstr(maxwell_5th_coeff/maxwell_4th_coeff, 10)}")
    print()

    # 잔류와 MAXWELL 항 스케일 비교
    print(f"  잔류 vs MAXWELL 항 스케일:")
    print(f"  잔류            = {nstr(residual_v2, 12)}")
    print(f"  MAXWELL 항3 크기 = {nstr((LN2/2)*GAP_D2**3, 12)}")
    print(f"  MAXWELL 항4 크기 = {nstr(maxwell_4th_coeff*GAP_D2**4, 12)}")
    print(f"  MAXWELL 항5 크기 = {nstr(maxwell_5th_coeff*GAP_D2**5, 12)}")
    print()
    print(f"  잔류/항3 = {nstr(residual_v2/((LN2/2)*GAP_D2**3), 8)}")
    print(f"  잔류/항4 = {nstr(residual_v2/(maxwell_4th_coeff*GAP_D2**4), 8)}")
    print()

    # ── [E] 전체 좌표계 계단 구조 ────────────────────────────────────────────
    print("[E] 6좌표계 T_D1×σ³ 계단 — LN_C 기준")
    print(sep2)
    coords = [
        ("HCW/RBC", mpf("-1.5")),
        ("HG",      mpf("-2.5")),
        ("I_s",     mpf("-3.5")),
        ("OB",      mpf("-4.5")),
        ("X5",      mpf("-5.5")),
        ("TH",      mpf("-6.5")),
    ]
    print(f"  {'좌표계':10}  {'T_D1':12}  {'T_D1×σ³':12}  {'vs LN_C':10}  {'관계':20}")
    print(sep2)
    prev_err = None
    for name, d1e in coords:
        T_D1 = CENTER - PHI**d1e
        cost = T_D1 * sigma3
        err  = float(pct_err(cost, LN_C))
        rel  = "LN_C 아래" if err < 0 else "LN_C 위"
        marker = "★ LN_C 협공" if (prev_err is not None and
                  prev_err < 0 and err > 0) else ""
        print(f"  {name:10}  {nstr(T_D1,8):12}  {nstr(cost,8):12}  "
              f"{err:10.4f}%  {rel}  {marker}")
        prev_err = err

    print()
    print(f"  → I_s: LN_C 아래 2.715%")
    print(f"  → OB:  LN_C 위  0.968%")
    print(f"  → 두 좌표계가 LN_C를 협공")
    print(f"  → LN_C는 I_s와 OB 사이에 정확히 위치")
    print()

    # I_s와 OB 사이 내분점
    # LN_C = T_Is_D1×σ³ + t×(T_OB_D1×σ³ - T_Is_D1×σ³)
    # t = (LN_C - T_Is_D1×σ³) / (T_OB_D1×σ³ - T_Is_D1×σ³)
    t_interp = (LN_C - val_Is_sigma3) / (T_OB_D1*sigma3 - val_Is_sigma3)
    print(f"  LN_C 내분 비율 t = (LN_C-T_Is×σ³)/(T_OB×σ³-T_Is×σ³)")
    print(f"                   = {nstr(t_interp, 15)}")
    print(f"  1-t               = {nstr(1-t_interp, 15)}")
    print()
    print(f"  t 후보 탐색:")
    t_candidates = [
        ("σ² = 1/4",            sigma**2),
        ("σ = 1/2",             sigma),
        ("1-σ²",                1-sigma**2),
        ("φ^(-2)",              PHI**mpf("-2")),
        ("φ^(-1)",              PHI**mpf("-1")),
        ("GAP/LN_C",            GAP/LN_C),
        ("GAP_D2/LN_C",         GAP_D2/LN_C),
        ("1/(1+φ²)",            1/(1+PHI**2)),
        ("D1_HCW",              PHI**mpf("-1.5")),
        ("1/φ²",                1/PHI**2),
        ("lnφ/π",               LN_PHI/pi),
        ("LN_C/LN_PHI",         LN_C/LN_PHI),
        ("T_Is/(T_Is+T_OB)",    T_Is_D1/(T_Is_D1+T_OB_D1)),
    ]
    for name, val in t_candidates:
        err = float(fabs(pct_err(val, t_interp)))
        marker = "★★★" if err < 0.01 else ("★★" if err < 1 else ("★" if err < 5 else ""))
        if err < 20:
            print(f"  {name:25s}  {float(val):12.8f}  오차 {err:10.6f}%  {marker}")

    # ── 통합 요약 ─────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  ▶▶ T_OB_D1×σ³ vs LN_C 탐색 v2 요약")
    print(sep)
    print()
    print(f"  【부호 수정 확정】")
    print(f"  T_OB_D1 = T_Is_D1 + φ^(-5.5)  True Zero")
    print(f"  T_OB_D1×σ³ = T_Is_D1×σ³ + φ^(-5.5)×σ³")
    print(f"    T_Is_D1×σ³  = {nstr(val_Is_sigma3, 12)}  (LN_C 아래 2.715%)")
    print(f"    φ^(-5.5)×σ³ = {nstr(residual_phi55, 12)}")
    print(f"    합            = {nstr(T_OB_D1*sigma3, 12)}  (LN_C 위 0.968%)")
    print()
    print(f"  【LN_C 협공 구조】")
    print(f"  I_s(2.715% 아래) ← LN_C → OB(0.968% 위)")
    print(f"  LN_C 내분 비율 t = {nstr(t_interp, 10)}")
    print()
    print(f"  【잔류 분석】")
    print(f"  잔류 = {nstr(residual_v2, 12)}")
    print(f"  최근접 후보 (스캔):")
    for err, name, val in results[:3]:
        print(f"    {name:35s}  오차 {err:.6f}%")
    print()
    print(sep)
    print(f"  Session 48 — verify_OB_LNC_v2.py 완료")
    print(sep)


if __name__ == "__main__":
    main()

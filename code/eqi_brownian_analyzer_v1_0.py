#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 EQI Brownian Motion JSON Analyzer v1.2
H↔O 이중성 → D1/D2 벡터곱 연결 분석기
"""

import json, sys, os, glob
import numpy as np

PHI     = (np.sqrt(5) + 1) / 2
GAP     = np.log(25) / 100
GAP_D2  = np.log(5) / 100
eps     = GAP / PHI
EQI4    = 1 + GAP
A_EQI   = np.pi * (1 + GAP)
EMF_base= GAP_D2 * A_EQI * eps
N_BBH   = 0.25
TIP_POINT = PHI**1.5 - 1

def to_arr(val):
    """어떤 타입이든 1-D numpy 배열로 변환"""
    if val is None:
        return np.array([])
    a = np.asarray(val, dtype=float)
    if a.ndim == 0:          # 스칼라
        return a.reshape(1)
    return a.ravel()

def main():
    print("=" * 70)
    print(" 📊 EQI Brownian Motion JSON Analyzer v1.2")
    print("=" * 70)

    if len(sys.argv) > 1:
        target = sys.argv[1].strip("'\"")
    else:
        files = glob.glob("eqi_water_brownian_motion_*.json")
        if not files:
            print("❌ JSON 파일 없음"); sys.exit(1)
        target = max(files, key=os.path.getmtime)

    print(f"\n📂 로드: {target}")
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── 키 확인 ───────────────────────────────────────────────────
    print(f"[키] 최상위: {list(data.keys())}")
    pollen = data.get("pollen_brownian", {})
    print(f"[키] pollen: {list(pollen.keys())}")

    # ── 데이터 추출 (모두 to_arr 통과) ───────────────────────────
    analysis   = data.get("analysis", {})
    zigzag_val    = float(analysis.get("zigzag_pattern_intensity", 0))
    exploit_ratio = float(analysis.get("exploitation_exploration_ratio", 0))
    flux_max      = float(analysis.get("information_flux_maximum", 0))

    px        = to_arr(pollen.get("pollen_trajectory", {}).get("x"))
    py        = to_arr(pollen.get("pollen_trajectory", {}).get("y"))
    exploit_f = to_arr(pollen.get("exploitation_force"))
    explore_f = to_arr(pollen.get("exploration_force"))
    info_flux = to_arr(pollen.get("information_flux"))
    zigzag_ts = to_arr(pollen.get("zigzag_pattern"))

    print(f"[길이] 궤적={len(px)}, exploit={len(exploit_f)}, "
          f"flux={len(info_flux)}, zigzag={len(zigzag_ts)}")

    # ── 1. 핵심 통계 vs EQI 상수 ─────────────────────────────────
    print(f"\n[1] 핵심 통계 vs EQI 상수")
    print(f"  지그재그 강도        = {zigzag_val:.6f}")
    print(f"  1/φ² (W_H)          = {1/PHI**2:.6f}  차이: {abs(zigzag_val-1/PHI**2):.4f}")
    print(f"  지그재그 × φ²       = {zigzag_val*PHI**2:.6f}")
    print(f"  Exploit/Explore 비율= {exploit_ratio:.6f}")
    print(f"  1+GAP/8             = {1+GAP/8:.6f}  차이: {abs(exploit_ratio-(1+GAP/8)):.6f}")
    print(f"  플럭스 최대값       = {flux_max:.6f}")
    print(f"  N_BBH               = {N_BBH:.4f}  "
          f"{'✓ 일치!' if abs(flux_max-N_BBH)<0.01 else 'diff='+str(round(abs(flux_max-N_BBH),4))}")

    # ── 2. Re_EQI 아날로그 ───────────────────────────────────────
    if len(exploit_f) > 1 and len(zigzag_ts) > 1:
        print(f"\n[2] Re_EQI 아날로그 (지그재그/Exploit)")
        n = min(len(exploit_f), len(zigzag_ts))
        re_analog = zigzag_ts[:n] / (np.abs(exploit_f[:n]) + 1e-10)
        print(f"  Re_EQI 아날로그 평균: {np.mean(re_analog):.6f}")
        print(f"  Re_EQI 아날로그 최대: {np.max(re_analog):.6f}")
        print(f"  v9.4 GW150914 Re_EQI: 0.001203")
        print(f"  미시/거시 스케일 비율: {np.mean(re_analog)/0.001203:.1f}배")
        if len(info_flux) > 1:
            m = min(len(info_flux), len(zigzag_ts))
            corr = np.corrcoef(info_flux[:m], zigzag_ts[:m])[0, 1]
            print(f"  플럭스↔지그재그 상관계수: {corr:.4f}")

    # ── 3. H↔O 교대 → EMF 연결 ───────────────────────────────────
    if len(exploit_f) > 1 and len(explore_f) > 1:
        print(f"\n[3] H↔O 교대 → EMF 연결")
        n = min(len(exploit_f), len(explore_f))
        diff = exploit_f[:n] - explore_f[:n]
        cross_mean = np.mean(np.abs(diff))
        sign_ch = int(np.sum(np.diff(np.sign(diff)) != 0))
        print(f"  교대 강도 평균 |H-O|: {cross_mean:.6f}")
        print(f"  EMF_base:             {EMF_base:.6e}")
        print(f"  cross/EMF_base:       {cross_mean/EMF_base:.2f}")
        print(f"  H↔O 교차 횟수:       {sign_ch}회")
        print(f"  교차 주기:            {n/max(sign_ch,1):.1f} 스텝")

    # ── 4. 꽃가루 궤적 → unity_rd 동형 ──────────────────────────
    if len(px) > 1 and len(py) > 1:
        print(f"\n[4] 꽃가루 궤적 → unity_rd 동형")
        disp = np.sqrt(px**2 + py**2)
        print(f"  평균 변위:          {np.mean(disp):.6f}")
        print(f"  최대 변위:          {np.max(disp):.6f}")
        print(f"  n_ouro 아날로그:    {np.mean(disp)/GAP:.4f}  (n_c={TIP_POINT/GAP:.2f})")
        pct = np.sum(disp > TIP_POINT) / len(disp) * 100
        print(f"  TIP_POINT 초과:     {pct:.1f}%  (ZONE-2+ 아날로그)")
        if len(px) > 2:
            dx, dy = np.diff(px), np.diff(py)
            dot = dx[:-1]*dx[1:] + dy[:-1]*dy[1:]
            dch = int(np.sum(dot < 0))
            print(f"  방향 전환:          {dch}회 / {len(px)}스텝 = {dch/len(px)*100:.1f}%")

    # ── 5. 플럭스 → A_EQI 면적벡터 ───────────────────────────────
    if len(info_flux) > 1:
        print(f"\n[5] 정보 플럭스 → A_EQI 면적벡터")
        flux_mean = np.mean(info_flux)
        print(f"  플럭스 평균:          {flux_mean:.6f}")
        print(f"  플럭스×A_EQI:         {flux_mean*A_EQI:.6f}")
        print(f"  EMF_base:             {EMF_base:.6e}")
        print(f"  (플럭스×A_EQI)/EMF:   {flux_mean*A_EQI/EMF_base:.4f}")
        if len(zigzag_ts) > 1:
            m = min(len(info_flux), len(zigzag_ts))
            emf_a = np.mean(info_flux[:m] * zigzag_ts[:m])
            print(f"  EMF 아날로그(플럭스×지그재그): {emf_a:.6e}")
            print(f"  EMF_base:                       {EMF_base:.6e}")
            print(f"  비율:                           {emf_a/EMF_base:.4f}")

    # ── 종합 ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" 종합: 물분자(작년) ↔ EQI v9.4(오늘) 연결")
    print(f"{'='*70}")
    print(f"  H (Unity)       → D1 스칼라 = tau_true")
    print(f"  O (Multiplicity)→ D2 벡터  = tau_nl")
    print(f"  꽃가루          → unity_rd 추적")
    print(f"  지그재그 {zigzag_val:.4f} → Re_EQI 미시 원형")
    print(f"  비율 {exploit_ratio:.4f}    → 1+GAP/8 = {1+GAP/8:.6f}")
    print(f"  플럭스max {flux_max:.4f}→ N_BBH = {N_BBH}")
    print(f"\n  벡터곱 연결 가설:")
    print(f"  F_마찰 = mu_ratio × A_EQI × X")
    print(f"  X의 미시 기원 = 지그재그 강도 = H↔O 교대의 비선형 응답")

if __name__ == "__main__":
    main()
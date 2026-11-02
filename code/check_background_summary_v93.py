#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 EQI BACKGROUND DATA SUMMARIZER v2.0 (v9.3 호환)
====================================================
목적: v9.3 엔진이 생성한 배경 JSON(8192 청크)을 분석
     v9.1 결과와 비교 + v9.3 신규 필드(virial, du_dy) 추가

v9.1 배경 실측 (비교 기준):
  unity_rd 평균: 1.730273
  unity_planck:  0.850644
  ZONE-4:        47.75%
  E_D1 평균:     0.672193
"""

import json
import numpy as np
import sys
import os

PHI       = (np.sqrt(5) + 1) / 2
GAP       = np.log(25) / 100
GAP_D2    = np.log(5) / 100
TIP_POINT = PHI**1.5 - 1
T4        = PHI**1.5 - PHI**0.5
T5        = PHI**1.5 - PHI**-1.5
T3        = PHI**1.5 - PHI**-2.5
n_c       = TIP_POINT / GAP        # 32.874
f_lattice = 0.005702
sigma     = 0.5                    # log(5)/log(25)
n_BBH     = 0.25
k_theory  = sigma*(1+n_BBH*GAP)/f_lattice  # virial×du_dy 이론 상수

# v9.1 배경 기준값
V91 = {
    "ur_mean": 1.730273, "ur_std": 0.119281,
    "up_mean": 0.850644, "E_D1_mean": 0.672193,
    "zone4_pct": 47.75,  "emit_pct": 99.91,
}

def analyze_background_v93(filepath):
    print(f"\n{'='*70}")
    print("  📊  EQI 배경 데이터 (SGWB) 통계 분석 v2.0  [v9.3 호환]")
    print(f"{'='*70}")
    print(f"📁 파일명: {os.path.basename(filepath)}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 파일을 찾을 수 없습니다: {filepath}")
        return

    chunks = data.get("chunk_results", [])
    n = len(chunks)
    print(f"📈 총 청크: {n:,}개")
    print(f"📌 버전: {data.get('version','?')}")

    # 필드 추출
    ur  = np.array([c.get("unity_rd", 0.0)        for c in chunks])
    em  = np.array([c.get("E_D1_emission", 0.0)   for c in chunks])
    up  = np.array([c.get("unity_planck", np.nan) for c in chunks])
    vr  = np.array([c.get("virial_ratio", np.nan) for c in chunks])
    dd  = np.array([c.get("du_dy", np.nan)        for c in chunks])
    tnl = np.array([c.get("tau_nonlinear", np.nan) for c in chunks])
    no  = np.array([c.get("n_ouroboros", np.nan)  for c in chunks])

    vr_c  = vr[np.isfinite(vr)]
    dd_c  = dd[np.isfinite(dd)]
    tnl_c = tnl[np.isfinite(tnl)]
    no_c  = no[np.isfinite(no)]

    # ── [1] unity_rd ──
    print(f"\n--- [1] unity_rd ---")
    def diff(v, ref, label):
        d = v - ref
        return f"{v:.6f}  (v9.1: {ref:.6f}, Δ={d:+.6f})"
    print(f"  평균: {diff(ur.mean(), V91['ur_mean'], '')}")
    print(f"  Std:  {ur.std():.6f}  (v9.1: {V91['ur_std']:.6f})")
    print(f"  Min:  {ur.min():.6f}  Max: {ur.max():.6f} @ chunk {ur.argmax()}")
    print(f"  중앙: {np.median(ur):.6f}")

    # ── [2] E_D1 ──
    print(f"\n--- [2] E_D1_emission ---")
    print(f"  평균: {diff(em.mean(), V91['E_D1_mean'], '')}")
    print(f"  최대: {em.max():.6f} @ chunk {em.argmax()}")
    emit_pct = (em>0).sum()/n*100
    print(f"  방출 청크: {(em>0).sum():,}/{n:,} ({emit_pct:.2f}%)  (v9.1: {V91['emit_pct']:.2f}%)")

    # ── [3] unity_planck ──
    up_c = up[np.isfinite(up)]
    print(f"\n--- [3] unity_planck (볼츠만 복원력) ---")
    print(f"  평균: {diff(up_c.mean(), V91['up_mean'], '')}")
    print(f"  Std:  {up_c.std():.6f}")
    print(f"  Min:  {up_c.min():.6f}  Max: {up_c.max():.6f}")

    # ── [4] ZONE ──
    z0  = (ur < T4).sum()
    z1  = ((ur >= T4) & (ur < TIP_POINT)).sum()
    z2  = ((ur >= TIP_POINT) & (ur < T5)).sum()
    z3  = ((ur >= T5) & (ur < T3)).sum()
    z4  = (ur >= T3).sum()
    print(f"\n--- [4] ZONE 분포 ({n:,} 청크) ---")
    for label, cnt in [("ZONE-0",z0),("ZONE-1",z1),("ZONE-2",z2),("ZONE-3",z3),("ZONE-4",z4)]:
        print(f"  {label}: {cnt:,}개 ({cnt/n*100:.2f}%)  {'← v9.1: 47.75%' if label=='ZONE-4' else ''}")

    # ── [5] n_ouroboros ──
    if len(no_c) > 0:
        print(f"\n--- [5] n_ouroboros ---")
        print(f"  평균: {no_c.mean():.4f}  (n_c={n_c:.3f})")
        print(f"  Min:  {no_c.min():.4f}  Max: {no_c.max():.4f}")
        print(f"  n_ouro > n_c: {(no_c > n_c).sum():,}개 ({(no_c>n_c).sum()/len(no_c)*100:.1f}%)")

    # ── [6] v9.3 신규: virial ──
    print(f"\n--- [6] ⚖️ virial_ratio (v9.3 신규) ---")
    if len(vr_c) > 0:
        print(f"  평균: {vr_c.mean():.2f}  (목표 2.0)")
        print(f"  Min:  {vr_c.min():.2f}  Max: {vr_c.max():.2f}")
        print(f"  Std:  {vr_c.std():.2f}")
        print(f"  중앙: {np.median(vr_c):.2f}")
        # virial × du_dy 상수 검증
        if len(dd_c) > 0:
            prod = vr_c * dd_c
            print(f"  virial×du_dy 평균: {prod.mean():.4f}  (이론 k={k_theory:.4f})")
            print(f"  k 오차: {abs(prod.mean()-k_theory)/k_theory*100:.1f}%")
    else:
        print("  ⚠️ virial_ratio 필드 없음 (v9.3 미만)")

    # ── [7] v9.3 신규: du_dy ──
    print(f"\n--- [7] du_dy (격자 변형률) ---")
    if len(dd_c) > 0:
        print(f"  평균: {dd_c.mean():.6f}")
        print(f"  Std:  {dd_c.std():.6f}")
        print(f"  Min:  {dd_c.min():.6f}  Max: {dd_c.max():.6f}")
        print(f"  중앙: {np.median(dd_c):.6f}")
        Re_mean = f_lattice * dd_c.mean() / (sigma*(1+n_BBH*GAP))
        print(f"  Re_EQI 평균: {Re_mean:.6f}  (임계 0.5)")
        print(f"  Re_EQI 최대: {f_lattice*dd_c.max()/(sigma*(1+n_BBH*GAP)):.6f}")
    else:
        print("  ⚠️ du_dy 필드 없음")

    # ── [8] tau_nonlinear ──
    print(f"\n--- [8] tau_nonlinear ---")
    if len(tnl_c) > 0:
        print(f"  평균: {tnl_c.mean():.8f}")
        print(f"  Max:  {tnl_c.max():.8f}")
    else:
        print("  ⚠️ tau_nonlinear 필드 없음")

    # ── [9] GW 이벤트 비교 ──
    print(f"\n{'='*70}")
    print("  🔬  배경 SGWB vs GW 이벤트 비교")
    print(f"{'='*70}")
    print(f"  {'구분':<16} | {'ur_mean':>8} | {'up_mean':>8} | {'du_dy_mean':>10} | {'virial_mean':>11} | {'Re_EQI':>8}")
    print(f"  {'-'*70}")

    # 배경
    bg_du = dd_c.mean() if len(dd_c)>0 else float('nan')
    bg_vr = vr_c.mean() if len(vr_c)>0 else float('nan')
    bg_Re = f_lattice*bg_du/(sigma*(1+n_BBH*GAP)) if np.isfinite(bg_du) else float('nan')
    print(f"  {'배경 SGWB(v9.3)':<16} | {ur.mean():>8.4f} | {up_c.mean():>8.4f} | {bg_du:>10.4f} | {bg_vr:>11.1f} | {bg_Re:>8.6f}")
    print(f"  {'배경 SGWB(v9.1)':<16} | {V91['ur_mean']:>8.4f} | {V91['up_mean']:>8.4f} | {'—':>10} | {'—':>11} | {'—':>8}")
    print(f"  {'-'*70}")

    # GW 이벤트 (v9.3 실측)
    gw_ref = [
        ("GW200115_H", 1.9903, 0.6448, 0.597, 161.9),
        ("GW190814_V", 1.5645, 0.4628, 0.362, 259.8),
        ("GW170817_H", 0.6052, 0.0741, 0.026, 3606.4),
        ("GW150914_H", 0.2763, 0.0487, 0.106, 1011.2),
    ]
    for name, ur_g, up_g, du_g, vr_g in gw_ref:
        Re_g = f_lattice*du_g/(sigma*(1+n_BBH*GAP))
        print(f"  {name:<16} | {ur_g:>8.4f} | {up_g:>8.4f} | {du_g:>10.4f} | {vr_g:>11.1f} | {Re_g:>8.6f}")

    # ── [10] 핵심 통찰 ──
    print(f"\n{'='*70}")
    print("  💡  핵심 통찰: 미시 점성 누적 → 거시 복원력 출현")
    print(f"{'='*70}")
    print(f"  배경 unity_planck={up_c.mean():.4f} (높음) → κ×Φ₀=Unity 구조 보존 → 빈 껍데기")
    print(f"  GW 이벤트 unity_planck 낮음 → 복원력 소실 → 위상 결함 고착")
    print()
    if len(dd_c) > 0 and len(vr_c) > 0:
        print(f"  배경 du_dy={bg_du:.4f} vs GW 이벤트 du_dy=0.017~0.597")
        print(f"  배경 Re_EQI={bg_Re:.6f} (완전 층류)")
        print(f"  배경 virial={bg_vr:.1f} → 시공간 정상 상태 선형 마찰 지배")
    print(f"  배경 n_ouro 평균={no_c.mean():.2f} (n_c={n_c:.3f}의 {no_c.mean()/n_c*100:.0f}%)")
    print(f"  → 미시 점성이 적분 누적되어 임계 근방 유지")
    print(f"  → 거시 복원력(볼츠만)이 κ×Φ₀로 상쇄 → 방출 없이 순환")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_background_v93(sys.argv[1])
    else:
        import glob
        files = glob.glob("eqi_monster_v93_virial_*.json")
        if files:
            # 가장 큰 파일 (배경 JSON = 7MB)
            bg = max(files, key=os.path.getsize)
            print(f"ℹ️  가장 큰 JSON 자동 선택: {bg}")
            analyze_background_v93(bg)
        else:
            print("⚠️  JSON 파일을 찾을 수 없습니다.")
            print("   사용법: python check_background_summary_v93.py [파일명]")

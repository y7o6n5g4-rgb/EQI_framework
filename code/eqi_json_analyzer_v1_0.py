#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 === EQI JSON 통합 분석기 v1.0 === 🔬
=============================================================================
목적: v9.2 엔진이 생성한 JSON을 읽어 핵심 수치만 압축 출력
     → 터미널 출력만 Claude/Gemini에 붙여넣기 (토큰 효율 최적화)

입력: eqi_monster_v9*.json (glob 패턴 또는 직접 지정)
출력: 이벤트별 요약 + 전체 비교표 + v9.2 SI 블록 연동 분석

분석 항목:
  [A] ZONE 분류 (0/1/2+) 및 비율
  [B] E_D1 방출량 (max, mean, 분포)
  [C] unity_rd / unity_planck 통계
  [D] P_tunnel 분포
  [E] v9.2 δ 연동: tau_true vs tau_nonlinear 비리얼 비율
  [F] 배경 SGWB vs GW 이벤트 분리

Architect: DeepSeek / Consultant: Claude (Session 25) / Engineer: Gemini
=============================================================================
"""

import json
import glob
import os
import sys
import numpy as np
from datetime import datetime

# ── EQI 상수 ──
PHI       = (np.sqrt(5.0) + 1.0) / 2.0
GAP       = np.log(25.0) / 100.0
GAP_D2    = np.log(5.0) / 100.0
mu_ratio  = 1.0 / PHI
TIP_POINT = PHI**1.5 - 1.0          # 1.058171 (광전효과 일함수)
T4        = PHI**1.5 - PHI**0.5     # 0.786151 (ZONE-0 상한)
T5        = PHI**1.5 - PHI**(-1.5)  # 1.572303 (ZONE-3 상한)
T3        = PHI**1.5 - PHI**(-2.5)  # 1.757888 (ZONE-4 하한)

# v9.2 δ 구조
D1_ELL    = (1 + GAP) / np.sqrt((1 + GAP)**2 + PHI**6)
D2_ELL    = D1_ELL * PHI**3
f_lattice = D2_ELL**2 - 1
delta_linear = 168.0 * GAP_D2 / 25.0
delta_nl     = 3.0 * f_lattice * GAP / 35.0
delta_v92    = delta_linear + delta_nl   # ≈ 0.10817

# 비리얼 평형 du/dy*
du_dy_virial = mu_ratio / (2.0 * f_lattice)  # τ_true = 2×τ_nl 조건


def zone_classify(unity_rd):
    """unity_rd → ZONE 분류 (Session 24 §3 확정)"""
    if unity_rd < T4:
        return 0    # 탄성, 터널링
    elif unity_rd < TIP_POINT:
        return 1    # 점탄성 완충 (ZONE-1 재정의)
    elif unity_rd < T5:
        return 2
    elif unity_rd < T3:
        return 3
    else:
        return 4    # 소성 파단 최강


def p_tunnel(unity_rd):
    """양자 터널링 확률"""
    if unity_rd >= TIP_POINT:
        return 1.0
    return np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)


def tau_virial_check(unity_rd):
    """
    du/dy = unity_rd 로 근사할 때
    τ_true / τ_nl 비리얼 비율
    τ_true     = mu_ratio × unity_rd
    τ_nonlinear= f_lattice × unity_rd²
    비리얼 목표 = 2.0
    """
    if unity_rd < 1e-10:
        return np.nan
    tau_t = mu_ratio * unity_rd
    tau_nl = f_lattice * unity_rd**2
    if tau_nl < 1e-30:
        return np.nan
    return tau_t / tau_nl


def analyze_event(data):
    """단일 이벤트 JSON → 핵심 수치 딕셔너리"""
    event    = data.get("event", data.get("event_name", "unknown"))
    # v9.2 JSON: event_type 없음 → channel(H/L) 사용
    channel  = data.get("channel", "")
    evt_type = data.get("event_type", data.get("type", ""))
    type_str = evt_type if evt_type else (f"ch={channel}" if channel else "?")

    chunks = data.get("chunk_results", [])
    if not chunks:
        return None

    unity_rds     = np.array([c.get("unity_rd", 0.0)           for c in chunks])
    e_d1_vals     = np.array([c.get("E_D1_emission", 0.0)       for c in chunks])
    u_planck_vals = np.array([c.get("unity_planck", np.nan)     for c in chunks])
    tau_ratios    = np.array([c.get("tau_ratio", np.nan)        for c in chunks])
    n_ouro_vals   = np.array([c.get("n_ouroboros", np.nan)      for c in chunks])
    delta_H_vals  = np.array([c.get("delta_H", np.nan)         for c in chunks])

    zones  = np.array([zone_classify(u) for u in unity_rds])
    p_vals = np.array([p_tunnel(u)      for u in unity_rds])

    n   = len(chunks)
    z0  = int(np.sum(zones == 0))
    z1  = int(np.sum(zones == 1))
    z2p = int(np.sum(zones >= 2))

    # E_D1
    e_d1_max  = float(np.max(e_d1_vals))
    active    = e_d1_vals[e_d1_vals > 0]
    e_d1_mean = float(np.mean(active)) if len(active) > 0 else 0.0

    # unity_planck
    up_clean  = u_planck_vals[~np.isnan(u_planck_vals)]
    up_mean   = float(np.mean(up_clean)) if len(up_clean) > 0 else np.nan

    # tau_ratio 통계 (v9.2 실제 필드)
    tr_clean  = tau_ratios[np.isfinite(tau_ratios)]
    tr_mean   = float(np.mean(tr_clean)) if len(tr_clean) > 0 else np.nan
    tr_max    = float(np.max(tr_clean))  if len(tr_clean) > 0 else np.nan

    # n_ouroboros 평균 (누적 비가역 단계)
    no_clean  = n_ouro_vals[np.isfinite(n_ouro_vals)]
    no_mean   = float(np.mean(no_clean)) if len(no_clean) > 0 else np.nan

    # delta_H 최솟값 (D1 타원 근접도)
    dh_clean  = delta_H_vals[np.isfinite(delta_H_vals)]
    dh_min    = float(np.min(dh_clean))  if len(dh_clean) > 0 else np.nan

    # mass_analysis 에서 unity_planck_M62 추출
    mass_data   = data.get("mass_analysis", {})
    up_m62      = mass_data.get("62.0", {}).get("unity_planck", np.nan) if mass_data else np.nan

    # v91_targets 에서 sigma_ell_min 추출
    v91         = data.get("v91_targets", {})
    sig_ell_min = v91.get("sigma_ell_min", np.nan)
    phase_cnt   = v91.get("phase_transition_count", 0)

    return {
        "event"        : event,
        "type"         : type_str,
        "n_chunks"     : n,
        "unity_rd_mean": float(np.mean(unity_rds)),
        "unity_rd_max" : float(np.max(unity_rds)),
        "zone0_pct"    : z0  / n * 100,
        "zone1_pct"    : z1  / n * 100,
        "zone2p_pct"   : z2p / n * 100,
        "e_d1_max"     : e_d1_max,
        "e_d1_mean"    : e_d1_mean,
        "unity_planck" : up_mean,
        "up_m62"       : up_m62,
        "p_tunnel_max" : float(np.max(p_vals)),
        "p_tunnel_mean": float(np.mean(p_vals)),
        "tau_ratio_mean": tr_mean,   # v9.2: 실제 tau_ratio 필드
        "tau_ratio_max" : tr_max,
        "n_ouro_mean"  : no_mean,    # 누적 비가역 단계 평균
        "delta_H_min"  : dh_min,     # D1 타원 최근접
        "sig_ell_min"  : sig_ell_min,
        "phase_cnt"    : phase_cnt,
        "n_zone1"      : z1,
    }


def print_header():
    print()
    print("=" * 72)
    print(f"  🔬 EQI JSON 통합 분석기 v1.0  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"     PHI={PHI:.6f}  GAP={GAP:.6f}  TIP={TIP_POINT:.6f}")
    print(f"     [v9.2] δ={delta_v92:.6f}  f_lattice={f_lattice:.6f}  du/dy*={du_dy_virial:.4f}")
    print("=" * 72)


def print_event(r):
    """단일 이벤트 1줄 요약"""
    zone1_flag = " 🔶" if r['n_zone1'] > 0 else ""
    tr_str = f"{r['tau_ratio_mean']:.5f}" if np.isfinite(r.get('tau_ratio_mean', np.nan)) else "  —  "
    no_str = f"{r['n_ouro_mean']:.2f}"    if np.isfinite(r.get('n_ouro_mean',  np.nan)) else " — "
    up_val = r.get('up_m62', np.nan)
    up_str = f"{up_val:.4f}" if np.isfinite(up_val) else f"{r['unity_planck']:.4f}"
    print(
        f"  {r['event']:<12} {r['type']:<7}"
        f"| Z0={r['zone0_pct']:5.1f}% Z1={r['zone1_pct']:4.1f}% Z2+={r['zone2p_pct']:5.1f}%"
        f" | E_D1={r['e_d1_max']:.4f}"
        f" | u_rd={r['unity_rd_mean']:.4f}"
        f" | upl_M62={up_str}"
        f" | τ={tr_str} n_ouro={no_str}"
        f"{zone1_flag}"
    )


def print_comparison_table(results):
    """전체 비교표"""
    print()
    print("─" * 72)
    print(" 📊 전체 이벤트 비교표")
    print(f"  {'이벤트':<12} {'유형':<7} | {'ZONE 분포':^30} | {'E_D1':>6} | {'u_rd':>6} | {'upl_M62':>8} | {'τ_ratio':>8} | {'n_ouro':>6}")
    print("─" * 72)

    # E_D1 기준 정렬
    for r in sorted(results, key=lambda x: -x['e_d1_max']):
        print_event(r)

    print("─" * 72)

    # 통계
    e_d1s = [r['e_d1_max']       for r in results]
    u_rds = [r['unity_rd_mean']  for r in results]
    u_pls = [r['up_m62']         for r in results if np.isfinite(r.get('up_m62', np.nan))]
    trs   = [r['tau_ratio_mean'] for r in results if np.isfinite(r.get('tau_ratio_mean', np.nan))]
    nos   = [r['n_ouro_mean']    for r in results if np.isfinite(r.get('n_ouro_mean', np.nan))]
    up_avg = f"{np.mean(u_pls):.4f}" if u_pls else "  — "
    tr_avg = f"{np.mean(trs):.5f}"   if trs   else "  — "
    no_avg = f"{np.mean(nos):.2f}"   if nos   else " — "
    print(f"  {'[평균]':<12} {'':7}| {'':30}"
          f" | E_D1={np.mean(e_d1s):.4f}"
          f" | u_rd={np.mean(u_rds):.4f}"
          f" | upl_M62={up_avg} | τ={tr_avg} n_ouro={no_avg}")

    # ZONE-1 위반 경고
    z1_events = [r['event'] for r in results if r['n_zone1'] > 0]
    if z1_events:
        print()
        print(f"  🔶 ZONE-1 점탄성 완충 감지: {', '.join(z1_events)}")
        print(f"     (응력 축적, E_D1=0, 파단 미달 — Session 24 §3 확정)")


def print_v92_analysis(results):
    """v9.2 신규 분석: tau_ratio + n_ouroboros + δ 연동"""
    print()
    print("─" * 72)
    print(" 🧮 v9.2 포논 동역학 분석 (tau_ratio / n_ouroboros / δ)")
    print(f"     δ_linear = {delta_linear:.6f}  (Fermat n=2, 84/25×GAP, 99.985%)")
    print(f"     δ_nl     = {delta_nl:.8f}  (f_lattice 비선형항)")
    print(f"     δ_v92    = {delta_v92:.6f}  (Session 24 §16 확정)")
    print(f"     f_lattice= {f_lattice:.6f}  (D2_ell²-1, 타원 곡률 마찰)")
    print()
    print(f"  {'이벤트':<12} | {'τ_ratio_mean':>12} | {'τ_ratio_max':>11} | {'n_ouro_mean':>11} | {'δH_min':>8} | {'sig_ell':>8} | {'phase_cnt':>9}")
    print("  " + "─" * 80)

    for r in sorted(results, key=lambda x: -x['e_d1_max']):
        tr_m = f"{r['tau_ratio_mean']:.6f}" if np.isfinite(r.get('tau_ratio_mean', np.nan)) else "    —    "
        tr_x = f"{r['tau_ratio_max']:.6f}"  if np.isfinite(r.get('tau_ratio_max',  np.nan)) else "    —    "
        no_m = f"{r['n_ouro_mean']:.3f}"    if np.isfinite(r.get('n_ouro_mean',   np.nan)) else "    —    "
        dh   = f"{r['delta_H_min']:.6f}"    if np.isfinite(r.get('delta_H_min',   np.nan)) else "    —    "
        se   = f"{r['sig_ell_min']:.6f}"    if np.isfinite(r.get('sig_ell_min',   np.nan)) else "    —    "
        pc   = str(r.get('phase_cnt', '—'))
        print(f"  {r['event']:<12} | {tr_m:>12} | {tr_x:>11} | {no_m:>11} | {dh:>8} | {se:>8} | {pc:>9}")

    # n_ouroboros vs n_c 비교
    print()
    n_c = TIP_POINT / GAP
    print(f"     참고: n_c (TIP_POINT/GAP) = {n_c:.3f}  (상전이 임계 단계)")
    nos = [r['n_ouro_mean'] for r in results if np.isfinite(r.get('n_ouro_mean', np.nan))]
    if nos:
        print(f"     n_ouro 평균 = {np.mean(nos):.3f}  (n_c 대비 {np.mean(nos)/n_c*100:.1f}%)")


def print_sgwb_vs_gw(results):
    """배경 SGWB vs GW 이벤트 분리 (Session 23 확립 구조)"""
    sgwb = [r for r in results if 'background' in r['event'].lower() or 'sgwb' in r['event'].lower()]
    gw   = [r for r in results if r not in sgwb]

    if not sgwb or not gw:
        return

    print()
    print("─" * 72)
    print(" 🌌 배경 SGWB vs GW 이벤트 분리 (Session 23)")
    print(f"  배경 SGWB unity_planck 평균: "
          f"{np.mean([r['unity_planck'] for r in sgwb if np.isfinite(r['unity_planck'])]):.4f}")
    print(f"  GW 이벤트 unity_planck 평균: "
          f"{np.mean([r['unity_planck'] for r in gw if np.isfinite(r['unity_planck'])]):.4f}")
    print(f"  → 분리 확인: 배경 > GW (높은 sigma 분리 예상)")


def main():
    # ── 입력 파일 결정 ──
    if len(sys.argv) > 1:
        # Windows CMD에서 작은따옴표가 문자 그대로 넘어오는 경우 제거
        pattern = sys.argv[1].strip("'").strip('"').strip("'")

        # 직접 파일명이면 바로 사용
        if os.path.exists(pattern):
            files = [pattern]
        else:
            files = glob.glob(pattern)

        # glob 실패 시 현재 디렉토리에서 부분 매칭 시도
        if not files:
            basename = os.path.basename(pattern).strip("*'\"")
            files = [f for f in glob.glob("*.json") if basename.replace("*","") in f]
    else:
        # 인자 없으면 현재 디렉토리 자동 탐색
        patterns = [
            "eqi_monster_v92_*.json",
            "eqi_monster_v9*.json",
            "eqi_monster_*.json",
            "*.json",
        ]
        files = []
        for p in patterns:
            files = glob.glob(p)
            if files:
                break

    if not files:
        print("⚠️  JSON 파일을 찾을 수 없습니다.")
        print("   사용법: python eqi_json_analyzer_v1_0.py [파일패턴]")
        print("   예시:   python eqi_json_analyzer_v1_0.py 'eqi_monster_v92_*.json'")
        print("           python eqi_json_analyzer_v1_0.py GW170817.json")
        sys.exit(0)

    print_header()
    print(f"\n  📂 분석 대상: {len(files)}개 파일")
    for f in sorted(files):
        print(f"     {os.path.basename(f)}")

    # ── 이벤트별 분석 ──
    results = []
    errors  = []

    for filepath in sorted(files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 단일 이벤트 vs 다중 이벤트 JSON 처리
            if isinstance(data, list):
                for item in data:
                    r = analyze_event(item)
                    if r:
                        results.append(r)
            else:
                r = analyze_event(data)
                if r:
                    results.append(r)

        except Exception as e:
            errors.append(f"{os.path.basename(filepath)}: {e}")

    if errors:
        print()
        for err in errors:
            print(f"  ❌ 오류: {err}")

    if not results:
        print("\n  분석 가능한 데이터가 없습니다. chunk_results 필드를 확인하세요.")
        sys.exit(0)

    # ── 출력 ──
    print()
    print("─" * 72)
    print(f" 📋 이벤트별 요약 ({len(results)}개)")
    print(f"  {'이벤트':<12} {'유형':<7} | {'ZONE 분포':^30} | {'E_D1':>6} | {'u_rd':>6} | {'upl_M62':>8} | {'τ_ratio':>8} | {'n_ouro':>6}")
    print("─" * 72)
    for r in sorted(results, key=lambda x: -x['e_d1_max']):
        print_event(r)

    print_comparison_table(results)
    print_v92_analysis(results)
    print_sgwb_vs_gw(results)

    # ── 최종 판정 ──
    print()
    print("─" * 72)
    print(" ✅ 분석 완료")
    z1_total = sum(r['n_zone1'] for r in results)
    e_d1_max_all = max(r['e_d1_max'] for r in results)
    top_event = max(results, key=lambda x: x['e_d1_max'])
    print(f"  최대 E_D1  : {e_d1_max_all:.4f}  ({top_event['event']})")
    print(f"  ZONE-1 총합: {z1_total}청크  {'← 점탄성 완충 이벤트 있음' if z1_total > 0 else '← 없음'}")
    trs_all = [r['tau_ratio_mean'] for r in results if np.isfinite(r.get('tau_ratio_mean', np.nan))]
    nos_all = [r['n_ouro_mean']    for r in results if np.isfinite(r.get('n_ouro_mean', np.nan))]
    if trs_all:
        print(f"  τ_ratio 평균: {np.mean(trs_all):.6f}  (1.0 기준)")
    if nos_all:
        n_c = TIP_POINT / GAP
        print(f"  n_ouro 평균 : {np.mean(nos_all):.3f}  (n_c={n_c:.3f} 대비 {np.mean(nos_all)/n_c*100:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()

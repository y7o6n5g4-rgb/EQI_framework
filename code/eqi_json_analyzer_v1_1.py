#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 === EQI JSON 통합 분석기 v1.1 === 🔬
=============================================================================
목적: v9.2/v9.3 엔진이 생성한 JSON을 읽어 핵심 수치만 압축 출력
     → 터미널 출력만 Claude/Gemini에 붙여넣기 (토큰 효율 최적화)

[v1.1 추가 항목]
  - virial_ratio (v9.3): mean, min, max + 2.0 수렴도 확인
  - du_dy, tau_nonlinear: mean, max 확인
  - sigma_ell_min 복원 반영 (v91_targets에서 로드)
  - print_v93_virial_analysis() 신규 섹션 추가

Architect: DeepSeek / Consultant: Claude (Session 25-26) / Engineer: Gemini
=============================================================================
"""

import json
import glob
import os
import sys
import numpy as np
from datetime import datetime

# ── EQI 상수 ──
PHI = (np.sqrt(5.0) + 1.0) / 2.0
GAP = np.log(25.0) / 100.0
GAP_D2 = np.log(5.0) / 100.0
mu_ratio = 1.0 / PHI
TIP_POINT = PHI**1.5 - 1.0  # 1.058171
T4 = PHI**1.5 - PHI**0.5  # 0.786151
T5 = PHI**1.5 - PHI ** (-1.5)  # 1.572303
T3 = PHI**1.5 - PHI ** (-2.5)  # 1.757888

# v9.2/v9.3 δ 구조
D1_ELL = (1 + GAP) / np.sqrt((1 + GAP) ** 2 + PHI**6)
D2_ELL = D1_ELL * PHI**3
f_lattice = D2_ELL**2 - 1
delta_linear = 168.0 * GAP_D2 / 25.0
delta_nl = 3.0 * f_lattice * GAP / 35.0
delta_v92 = delta_linear + delta_nl  # ≈ 0.10817
du_dy_virial = mu_ratio / (2.0 * f_lattice)  # τ_true = 2×τ_nl 조건


def zone_classify(unity_rd):
    if unity_rd < T4:
        return 0
    elif unity_rd < TIP_POINT:
        return 1
    elif unity_rd < T5:
        return 2
    elif unity_rd < T3:
        return 3
    else:
        return 4


def analyze_event(data):
    event = data.get("event", data.get("event_name", "unknown"))
    channel = data.get("channel", "")
    evt_type = data.get("event_type", data.get("type", ""))
    type_str = evt_type if evt_type else (f"ch={channel}" if channel else "?")

    chunks = data.get("chunk_results", [])
    if not chunks:
        return None

    unity_rds = np.array([c.get("unity_rd", 0.0) for c in chunks])
    e_d1_vals = np.array([c.get("E_D1_emission", 0.0) for c in chunks])
    u_planck_vals = np.array([c.get("unity_planck", np.nan) for c in chunks])
    tau_ratios = np.array([c.get("tau_ratio", np.nan) for c in chunks])
    n_ouro_vals = np.array([c.get("n_ouroboros", np.nan) for c in chunks])

    # v9.3 추가 필드 파싱
    vr_vals = np.array([c.get("virial_ratio", np.nan) for c in chunks])
    du_dy_vals = np.array([c.get("du_dy", np.nan) for c in chunks])
    tau_nl_vals = np.array([c.get("tau_nonlinear", np.nan) for c in chunks])

    zones = np.array([zone_classify(u) for u in unity_rds])
    n = len(chunks)
    z0 = int(np.sum(zones == 0))
    z1 = int(np.sum(zones == 1))
    z2p = int(np.sum(zones >= 2))

    e_d1_max = float(np.max(e_d1_vals))
    up_m62 = data.get("v91_targets", {}).get("unity_planck_M62", np.nan)
    sig_ell_min = data.get("v91_targets", {}).get("sigma_ell_min", np.nan)

    # v9.3 비리얼 통계 산출
    vr_clean = vr_vals[np.isfinite(vr_vals)]
    du_dy_clean = du_dy_vals[np.isfinite(du_dy_vals)]
    tau_nl_clean = tau_nl_vals[np.isfinite(tau_nl_vals)]

    tr_clean = tau_ratios[np.isfinite(tau_ratios)]
    no_clean = n_ouro_vals[np.isfinite(n_ouro_vals)]

    return {
        "event": event,
        "type": type_str,
        "n_chunks": n,
        "unity_rd_mean": float(np.mean(unity_rds)),
        "zone0_pct": z0 / n * 100,
        "zone1_pct": z1 / n * 100,
        "zone2p_pct": z2p / n * 100,
        "e_d1_max": e_d1_max,
        "up_m62": up_m62,
        "tau_ratio_mean": float(np.mean(tr_clean)) if len(tr_clean) > 0 else np.nan,
        "tau_ratio_max": float(np.max(tr_clean)) if len(tr_clean) > 0 else np.nan,
        "n_ouro_mean": float(np.mean(no_clean)) if len(no_clean) > 0 else np.nan,
        "sig_ell_min": sig_ell_min,
        "n_zone1": z1,
        # v9.3 지표
        "vr_mean": float(np.mean(vr_clean)) if len(vr_clean) > 0 else np.nan,
        "vr_min": float(np.min(vr_clean)) if len(vr_clean) > 0 else np.nan,
        "vr_max": float(np.max(vr_clean)) if len(vr_clean) > 0 else np.nan,
        "du_dy_mean": float(np.mean(du_dy_clean)) if len(du_dy_clean) > 0 else np.nan,
        "du_dy_max": float(np.max(du_dy_clean)) if len(du_dy_clean) > 0 else np.nan,
        "tau_nl_mean": (
            float(np.mean(tau_nl_clean)) if len(tau_nl_clean) > 0 else np.nan
        ),
    }


def print_header():
    print()
    print("=" * 80)
    print(
        f"  🔬 EQI JSON 통합 분석기 v1.1  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"     PHI={PHI:.6f}  GAP={GAP:.6f}  TIP={TIP_POINT:.6f}")
    print(
        f"     [v9.3] δ={delta_v92:.6f}  f_lattice={f_lattice:.6f}  du/dy*={du_dy_virial:.4f}"
    )
    print("=" * 80)


def print_event(r):
    zone1_flag = " 🔶" if r["n_zone1"] > 0 else ""
    tr_str = (
        f"{r['tau_ratio_mean']:.5f}"
        if np.isfinite(r.get("tau_ratio_mean", np.nan))
        else "  —  "
    )
    no_str = (
        f"{r['n_ouro_mean']:.2f}"
        if np.isfinite(r.get("n_ouro_mean", np.nan))
        else " — "
    )
    up_str = f"{r['up_m62']:.4f}" if np.isfinite(r.get("up_m62", np.nan)) else "  —  "
    print(
        f"  {r['event']:<18} {r['type']:<5}"
        f"| Z0={r['zone0_pct']:5.1f}% Z1={r['zone1_pct']:4.1f}% Z2+={r['zone2p_pct']:5.1f}%"
        f" | E_D1={r['e_d1_max']:.4f}"
        f" | u_rd={r['unity_rd_mean']:.4f}"
        f" | upl_M62={up_str}"
        f" | τ={tr_str} n_ouro={no_str}"
        f"{zone1_flag}"
    )


def print_comparison_table(results):
    print()
    print("─" * 80)
    print(" 📊 전체 이벤트 비교표")
    print(
        f"  {'이벤트':<18} {'유형':<5} | {'ZONE 분포':^30} | {'E_D1':>6} | {'u_rd':>6} | {'upl_M62':>8} | {'τ_ratio':>8} | {'n_ouro':>6}"
    )
    print("─" * 80)

    for r in sorted(results, key=lambda x: -x["e_d1_max"]):
        print_event(r)


def print_v92_analysis(results):
    print()
    print("─" * 80)
    print(" 🧮 v9.2 포논 동역학 분석 (tau_ratio / n_ouroboros)")
    print(
        f"  {'이벤트':<18} | {'τ_ratio_mean':>12} | {'τ_ratio_max':>11} | {'n_ouro_mean':>11} | {'sig_ell':>8}"
    )
    print("  " + "─" * 68)

    for r in sorted(results, key=lambda x: -x["e_d1_max"]):
        tr_m = (
            f"{r['tau_ratio_mean']:.6f}"
            if np.isfinite(r.get("tau_ratio_mean", np.nan))
            else "    —    "
        )
        tr_x = (
            f"{r['tau_ratio_max']:.6f}"
            if np.isfinite(r.get("tau_ratio_max", np.nan))
            else "    —    "
        )
        no_m = (
            f"{r['n_ouro_mean']:.3f}"
            if np.isfinite(r.get("n_ouro_mean", np.nan))
            else "    —    "
        )
        se = (
            f"{r['sig_ell_min']:.6f}"
            if np.isfinite(r.get("sig_ell_min", np.nan))
            else "    —    "
        )
        print(f"  {r['event']:<18} | {tr_m:>12} | {tr_x:>11} | {no_m:>11} | {se:>8}")


def print_v93_virial_analysis(results):
    print()
    print("─" * 80)
    print(" ⚖️ v9.3 비리얼 열역학 평형 분석 (Virial Equilibrium Tracking)")
    print(f"     목표 평형비율(Virial Ratio) = 2.0  (τ_true / τ_nl)")
    print(
        f"     * 2.0 수렴은 격자가 낙하산 종단속도처럼 선형/비선형 마찰 평형을 이룸을 뜻함"
    )
    print()
    print(
        f"  {'이벤트':<18} | {'VR_mean':>8} | {'VR_min':>8} | {'VR_max':>8} | {'du_dy_mean':>10} | {'τ_nl_mean':>10}"
    )
    print("  " + "─" * 72)

    vr_means = []
    for r in sorted(results, key=lambda x: -x["e_d1_max"]):
        # v9.3 데이터가 없는 경우를 대비한 처리
        if np.isnan(r.get("vr_mean", np.nan)):
            continue

        vr_m = f"{r['vr_mean']:.4f}"
        vr_i = f"{r['vr_min']:.4f}"
        vr_x = f"{r['vr_max']:.4f}"
        dd_m = f"{r['du_dy_mean']:.4f}"
        tn_m = f"{r['tau_nl_mean']:.4f}"
        print(
            f"  {r['event']:<18} | {vr_m:>8} | {vr_i:>8} | {vr_x:>8} | {dd_m:>10} | {tn_m:>10}"
        )
        vr_means.append(r["vr_mean"])

    if vr_means:
        print("  " + "─" * 72)
        print(
            f"  [전체 평균] Virial Ratio (τ_true / τ_nl) = {np.mean(vr_means):.4f}  (목표 2.0)"
        )


def main():
    if len(sys.argv) > 1:
        pattern = sys.argv[1].strip("'").strip('"').strip("'")
        if os.path.exists(pattern):
            files = [pattern]
        else:
            files = glob.glob(pattern)
        if not files:
            basename = os.path.basename(pattern).strip("*'\"")
            files = [f for f in glob.glob("*.json") if basename.replace("*", "") in f]
    else:
        patterns = ["eqi_monster_v93_virial_*.json", "eqi_monster_v92_ellipse_*.json"]
        files = []
        for p in patterns:
            files = glob.glob(p)
            if files:
                break

    if not files:
        print("⚠️  JSON 파일을 찾을 수 없습니다.")
        sys.exit(0)

    print_header()
    print(f"\n  📂 분석 대상: {len(files)}개 파일")

    results, errors = [], []

    for filepath in sorted(files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
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
        for err in errors:
            print(f"  ❌ 오류: {err}")
    if not results:
        print("\n  분석 가능한 데이터가 없습니다.")
        sys.exit(0)

    print_comparison_table(results)
    print_v92_analysis(results)
    print_v93_virial_analysis(results)

    print()
    print("=" * 80)
    print(" ✅ 분석 완료")


if __name__ == "__main__":
    main()

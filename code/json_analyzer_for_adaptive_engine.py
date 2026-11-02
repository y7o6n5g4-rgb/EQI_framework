#!/usr/bin/env python3
"""
EQI Adaptive JSON Analyzer
대용량(가변 청크) JSON 파일에서 핵심 상전이 시각과 요약을 즉시 추출합니다.
사용법: python json_analyzer_for_adaptive_engine.py <생성된_json_파일명>
"""

import json
import sys
import os

def analyze_adaptive_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    event = data.get('event', 'Unknown')
    results = data.get('chunk_results', [])
    theory = data.get('theory', {})
    layer_e = data.get('layer_E_fermat_feynman', {})

    if not results:
        print("❌ chunk_results 데이터가 비어 있습니다.")
        return

    # 이론값
    ELL_SGWB_THEORY = theory.get("ELL_SGWB_THEORY", 0.000021931325)
    MAXWELL = theory.get("B_MAXWELL", 1.87311341)
    D1_ell_target = theory.get("D_D1_ell_target", 0.236740008)

    # 지표 최소/최대 추적기
    gr_min = 9999; gr_t = -1
    gb_min = 9999; gb_t = -1
    sinc_max = 0;  sinc_t = -1
    nex_max = 0;   nex_t = -1
    up_min = 9999; up_t = -1
    ell_min = 9999; ell_t = -1
    rim_max = 0;   rim_t = -1

    t25_res = None
    min_time_diff_25 = 9999

    for r in results:
        t = r.get('time_s', 0)
        Gr = r.get('G_ratio', 9999)
        gb = r.get('grid_balance', 9999)
        sinc = r.get('sinc_H_abs', 0)
        nex = r.get('nexus_signal', 0)
        up = r.get('unity_product', 9999)
        ell = r.get('eqi4_sigma_ell_H', 9999)
        rim = r.get('riemann_ratio', 0)

        if Gr < gr_min: gr_min = Gr; gr_t = t
        if abs(gb - 1.0) < abs(gb_min - 1.0): gb_min = gb; gb_t = t
        if sinc > sinc_max: sinc_max = sinc; sinc_t = t
        if nex > nex_max: nex_max = nex; nex_t = t
        if abs(up - MAXWELL) < abs(up_min - MAXWELL): up_min = up; up_t = t
        if ell < ell_min: ell_min = ell; ell_t = t
        if rim > rim_max: rim_max = rim; rim_t = t

        # t=2.5s 근사 탐색
        diff_25 = abs(t - 2.5)
        if diff_25 < min_time_diff_25:
            min_time_diff_25 = diff_25
            t25_res = r

    # 출력
    print("="*80)
    print(f" 📊 EQI 적응형 JSON 분석 리포트 - {event}")
    print("="*80)
    print(f"\n[4/5계층 수렴 시각 비교]")
    print(f"  [A] G_ratio 최솟값       : t={gr_t:.4f}s")
    print(f"  [B] grid_balance→1.0     : t={gb_t:.4f}s")
    print(f"  [B] sinc_H→1.0           : t={sinc_t:.4f}s")
    print(f"  [C] nexus 최대 (맥동복원): t={nex_t:.4f}s")
    print(f"  [D] unity_product→MAXWELL: t={up_t:.4f}s  (값={up_min:.6f})")
    print(f"  [D] ell_sigma→0 (타원)   : t={ell_t:.4f}s  (값={ell_min:.8e})")
    print(f"  [D] riemann_ratio 최대   : t={rim_t:.4f}s  (값={rim_max:.4e})")

    ell_theory_err = (
        (abs(ell_min - ELL_SGWB_THEORY) / ELL_SGWB_THEORY * 100)
        if ELL_SGWB_THEORY > 0
        else 0
    )
    print(f"  [D] D1_ell_target        = {D1_ell_target:.9f}")
    print(
        f"  [D] ell_SGWB 타깃 오차   : {ell_theory_err:.5f}% (6차원 동적 위상 공간 완전 닫힘 타깃 10^6 대비)"
    )
    print(
        f"      *참고: 이 오차는 SGWB 배경 데이터에서만 0에 수렴하며, GW 이벤트에서는 섭동 강도를 나타냅니다."
    )

    if t25_res:
        print(f"\n[t=2.5s HCW flip (근사치: t={t25_res.get('time_s', 0):.4f}s, Chunk: {t25_res.get('chunk_size', 'N/A')})]")
        keys_to_print = [
            "G_ratio", "grid_balance", "sinc_H_abs", "qec_openness", 
            "nexus_signal", "unity_product", "n_ouroboros", "carnot_eta", 
            "delta_H", "eqi4_sigma_ell_H", "riemann_ratio", "z0_14_1347",
            "e_maxwell", "qec_maxwell_theory"
        ]
        for k in keys_to_print:
            print(f"    {k:<20} = {t25_res.get(k, 0.0):.8e}")

    print(f"\n[Layer E: v18_10 페르마-파인만 요약]")
    if layer_e:
        for k, v in layer_e.items():
            if isinstance(v, float):
                print(f"  {k:<22} = {v:.6f}")
            elif isinstance(v, dict):
                print(f"  {k:<22}:")
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, float):
                        print(f"    {sub_k:<12} = {sub_v:.6e}")
                    else:
                        print(f"    {sub_k:<12} = {sub_v}")
            else:
                print(f"  {k:<22} = {v}")
    else:
        print("  (Layer E 데이터가 없습니다.)")

    print("\n✅ 분석 완료.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python json_analyzer_for_adaptive_engine.py <파일명.json>")
    else:
        analyze_adaptive_json(sys.argv[1])
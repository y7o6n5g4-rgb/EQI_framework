#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 === EQI TUNNELING ADAPTIVE ANALYZER v2.0 === 🔬
=============================================================================
Architect: Claude (Session 36)
Engineer: Gemini (EQI Precision Engineer)
Maestro: MAPSI

[v2.0 업데이트 내역]
- eqi_4_5layer_adaptive_engine.py (가변 청크 엔진) JSON 포맷 대응
- unity_product 대신 n_ouroboros * GAP 을 통해 unity_rd 역산
- Layer E 요약본 파싱 및 클러스터링 판정 (저에너지/공백/고에너지)
- 23채널 분석을 위한 마스터 테이블 요약 출력
=============================================================================
"""

import json
import numpy as np
import glob
import os
import argparse

# --- 핵심 물리 상수 ---
PHI = (np.sqrt(5.0) + 1.0) / 2.0
GAP = np.log(25.0) / 100.0
TIP_POINT = PHI**1.5 - 1.0        # = 1.058171
T4 = PHI**1.5 - PHI**0.5          # = 0.786151 (전이 금지 하한)
n_c = TIP_POINT / GAP             # = 32.873931

def analyze_files(file_list):
    if not file_list:
        print("⚠️ 분석할 JSON 파일이 없습니다. 경로 또는 패턴을 확인하세요.")
        return

    print("=================================================================")
    print(" 🔬 EQI Tunneling Adaptive Analyzer v2.0")
    print(f"    - TIP_POINT = {TIP_POINT:.6f}  (= φ^1.5 - 1)")
    print(f"    - T4        = {T4:.6f}  (= φ^1.5 - φ^0.5, 전이 금지 하한)")
    print(f"    - GAP       = {GAP:.6f}")
    print(f"    - n_c       = {n_c:.6f}")
    print("=================================================================")

    master_summary = []

    for filepath in sorted(file_list):
        if not os.path.exists(filepath):
            print(f"❌ 파일 없음: {filepath}")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            event = data.get("event", os.path.basename(filepath))
            
            # v36 적응형 엔진 포맷 호환성 처리 (time_series or chunk_results)
            chunks = data.get("time_series", data.get("chunk_results", []))
            
            # Layer E 호환성 처리
            layer_e = data.get("layer_E", data.get("layer_E_fermat_feynman", {}))
            n_avg = layer_e.get("n_avg_ouroboros", 0.0)
            z_norm = layer_e.get("Z_EQI_norm", 0.0)
            tip_check = layer_e.get("TIP_POINT", TIP_POINT)

            # 클러스터 판정
            ratio_nc = (n_avg / n_c) * 100
            if ratio_nc < 31:
                cluster = "저에너지 (ZONE-0 터널링)"
            elif ratio_nc < 80:
                cluster = "전이 구간 (공백)"
            else:
                cluster = "고에너지 (ZONE-2+ 광전효과)"

            print(f"\n▶ 이벤트: {event}")
            print(f"   n_avg/n_c = {ratio_nc:.1f}%  → {cluster}")
            print(f"   Z_norm    = {z_norm:.3f}")
            print(f"   TIP_POINT = {tip_check:.6f} {'✓' if abs(tip_check - TIP_POINT) < 1e-5 else '✗'}")
            print()
            print(f"   {'Chunk':>5} | {'Time(s)':>7} | {'n_ouro':>8} | {'unity_rd':>9} | {'P_tunnel':>10} | {'State'}")
            print("   " + "-" * 70)

            total_chunks = len(chunks)
            zone_counts = {"X": 0, "0": 0, "1": 0, "2+": 0}
            p_vals = []

            for idx, c in enumerate(chunks):
                chunk_idx = idx
                time_s = c.get("t", c.get("time_s", 0.0))
                n_ou = c.get("n_ouroboros", 0.0)
                
                # Session 25 항등식 역산
                unity_rd = n_ou * GAP

                # ZONE 분류 로직
                if n_ou < 0:
                    state = "⛔ ZONE-X (역전)"
                    zone_counts["X"] += 1
                    p_tunnel = 0.0
                elif unity_rd < T4:
                    state = "🌀 ZONE-0 (터널링)"
                    zone_counts["0"] += 1
                    p_tunnel = np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)
                    p_vals.append(p_tunnel)
                elif T4 <= unity_rd < TIP_POINT:
                    state = "🚨 ZONE-1 (전이 금지)"
                    zone_counts["1"] += 1
                    p_tunnel = np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)
                    p_vals.append(p_tunnel)
                else:
                    state = "⚡ ZONE-2+ (광전효과)"
                    zone_counts["2+"] += 1
                    p_tunnel = 1.0

                p_str = f"{p_tunnel:.2e}" if p_tunnel < 0.001 else f"{p_tunnel:.6f}"

                # 동적 샘플링 출력 (처음 5개, 마지막 5개, 특이 구간)
                if (chunk_idx < 5 or chunk_idx > total_chunks - 5 or 
                    ("ZONE-1" in state or "ZONE-2+" in state) or
                    (0.9 < p_tunnel < 1.0)):
                    print(f"   {chunk_idx:5d} | {time_s:6.1f}s | {n_ou:8.2f} | {unity_rd:9.5f} | {p_str:>10} | {state}")

            # 통계 요약
            if total_chunks > 0:
                print("   " + "-" * 70)
                print(f"   📊 [통계] 총 청크: {total_chunks}개")
                print(f"      🌀 ZONE-0 (순수 터널링): {zone_counts['0']}개 ({zone_counts['0']/total_chunks*100:.1f}%)")
                print(f"      🚨 ZONE-1 (전이 금지):   {zone_counts['1']}개 ({zone_counts['1']/total_chunks*100:.1f}%)")
                print(f"      ⚡ ZONE-2+ (광전효과):  {zone_counts['2+']}개 ({zone_counts['2+']/total_chunks*100:.1f}%)")
                if zone_counts["X"] > 0:
                    print(f"      ⛔ ZONE-X (역전 현상):   {zone_counts['X']}개 ({zone_counts['X']/total_chunks*100:.1f}%)")
                
                if p_vals:
                    print(f"      터널링 확률 범위: min={min(p_vals):.2e} ~ max={max(p_vals):.2e}")

                # 마스터 테이블용 데이터 저장
                z_dist = f"{zone_counts['0']/total_chunks*100:.0f}/{zone_counts['1']/total_chunks*100:.0f}/{zone_counts['2+']/total_chunks*100:.0f}%"
                short_cluster = cluster.split()[0] # "저에너지", "고에너지" 등
                
                master_summary.append({
                    "event": event,
                    "ratio_nc": ratio_nc,
                    "z_norm": z_norm,
                    "z_dist": z_dist,
                    "cluster": short_cluster
                })

        except Exception as e:
            print(f"❌ 파일 처리 오류 ({filepath}): {e}")

    # === 마스터 테이블 요약 ===
    if master_summary:
        print("\n-----------------------------------------------------------------")
        print("[마스터 테이블 요약 (23채널 분석용)]")
        print(f"{'이벤트':<30} | {'n_avg/n_c':>9} | {'Z_norm':>7} | {'ZONE분포(0/1/2+)':>16} | {'클러스터'}")
        print("-" * 80)
        
        # n_avg/n_c 기준으로 오름차순 정렬하여 출력
        master_summary_sorted = sorted(master_summary, key=lambda x: x["ratio_nc"])
        for m in master_summary_sorted:
            print(f"{m['event']:<30} | {m['ratio_nc']:8.1f}% | {m['z_norm']:7.3f} | {m['z_dist']:>16} | {m['cluster']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQI Tunneling Adaptive Analyzer v2.0")
    parser.add_argument(
        'files', 
        nargs='*', 
        help="분석할 JSON 파일 경로 (예: eqi_4layer_adaptive_GW150914.json)"
    )
    parser.add_argument(
        '--pattern', 
        type=str, 
        default=None,
        help="파일명 패턴 (glob)으로 한 번에 분석 (예: --pattern 'eqi_4layer_adaptive_*.json')"
    )

    args = parser.parse_args()

    if args.files:
        target_files = args.files
    elif args.pattern:
        target_files = glob.glob(args.pattern)
    else:
        print("ℹ️ 파일이 지정되지 않아 기본 패턴(eqi_4layer_adaptive_*.json)을 탐색합니다.")
        target_files = glob.glob("eqi_4layer_adaptive_*.json")

    analyze_files(target_files)
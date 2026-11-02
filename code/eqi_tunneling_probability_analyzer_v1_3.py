#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 === EQI QUANTUM TUNNELING PROBABILITY ANALYZER v1.3 === 🔬
=============================================================================
Architect: DeepSeek (Supreme Architect) & Claude (Advisor)
Engineer: Gemini (EQI Precision Engineer)
Date: 2026-07-22

[v1.3 업데이트]
- argparse 도입: 수동으로 특정 JSON 파일을 직접 지정 가능
- --pattern 옵션 추가: 기존 glob 패턴 지원 (하위 호환)
- 파일 미지정 시 기본값(eqi_monster_v91_ellipse_*.json)으로 폴백
=============================================================================
"""

import json
import numpy as np
import glob
import os
import argparse
import sys

PHI = (np.sqrt(5.0) + 1.0) / 2.0
GAP = np.log(25.0) / 100.0
TIP_POINT = PHI**1.5 - 1.0
T4 = PHI**1.5 - PHI**0.5

def analyze_files(file_list):
    """분석할 파일 리스트를 받아서 처리하는 코어 함수"""
    if not file_list:
        print("⚠️ 분석할 파일이 없습니다.")
        return

    print("=================================================================")
    print(" 🔬 [EQI] 거시적 양자 터널링 분석기 v1.3 (수동 파일 지정)")
    print(f"    - 장벽 에너지 (TIP_POINT) : {TIP_POINT:.6f}")
    print(f"    - 장벽 두께 (GAP)         : {GAP:.6f}")
    print(f"    - 전이 금지 구간 (ZONE-1) : {T4:.6f} ~ {TIP_POINT:.6f}")
    print("=================================================================")

    for filepath in sorted(file_list):
        if not os.path.exists(filepath):
            print(f"❌ 파일 없음: {filepath}")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            event = data.get("event", os.path.basename(filepath))
            chunks = data.get("chunk_results", [])
            
            print(f"\n▶ 이벤트: {event} (출처: {os.path.basename(filepath)})")
            print(f"{'Chunk':>5} | {'Time(s)':>7} | {'Unity_rd (E)':>10} | {'P_tunnel (확률)':>15} | {'상태 (State)':<20}")
            print("-" * 75)

            total_chunks = len(chunks)
            warning_count = 0

            for c in chunks:
                chunk_idx = c.get("chunk", 0)
                time_s = c.get("time_s", 0.0)
                unity_rd = c.get("unity_rd", 0.0)

                # --- ZONE 분류 및 터널링 확률 ---
                if unity_rd < T4:
                    p_tunnel = np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)
                    state = "🌀 ZONE-0: 터널링"
                    p_str = f"{p_tunnel:.6f}" if p_tunnel >= 0.0001 else f"{p_tunnel:.2e}"
                elif T4 <= unity_rd < TIP_POINT:
                    p_tunnel = np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)
                    state = "🚨 ZONE-1: 이론 위반 (경고)"
                    p_str = f"{p_tunnel:.6f}" if p_tunnel >= 0.0001 else f"{p_tunnel:.2e}"
                    warning_count += 1
                else:
                    p_tunnel = 1.0
                    p_str = "1.000000"
                    state = "⚡ ZONE-2/3/4: 광전효과"

                # --- 동적 샘플링 출력 ---
                if (chunk_idx < 5 or chunk_idx > total_chunks - 5 or 
                    (0.9 < p_tunnel < 1.0) or unity_rd > TIP_POINT * 0.9 or
                    (T4 <= unity_rd < TIP_POINT)):
                    print(f"{chunk_idx:5d} | {time_s:6.1f}s | {unity_rd:10.6f} | {p_str:>15} | {state}")

            # --- 통계 요약 ---
            total = len(chunks)
            if total > 0:
                zone0 = sum(1 for c in chunks if c.get("unity_rd", 0.0) < T4)
                zone1 = sum(1 for c in chunks if T4 <= c.get("unity_rd", 0.0) < TIP_POINT)
                zone_pe = total - zone0 - zone1
                
                p_vals = [np.exp(-2.0 * (TIP_POINT - c.get("unity_rd", 0.0)) / GAP) 
                          for c in chunks if c.get("unity_rd", 0.0) < TIP_POINT]
                
                print("-" * 75)
                print(f"📊 [통계] 총 청크: {total}개")
                print(f"   - 🌀 ZONE-0 (순수 터널링): {zone0}개 ({zone0/total*100:.1f}%)")
                print(f"   - 🚨 ZONE-1 (이론 위반):   {zone1}개 ({zone1/total*100:.1f}%)")
                print(f"   - ⚡ ZONE-2/3/4 (광전효과): {zone_pe}개 ({zone_pe/total*100:.1f}%)")
                if p_vals:
                    print(f"   - 터널링 확률 범위: min={min(p_vals):.2e} ~ max={max(p_vals):.6f}")
                
                if zone1 > 0:
                    print("   🚨🚨 [치명적 경고] ZONE-1 데이터 발견! EQI 공리계 예측 위반.")

        except Exception as e:
            print(f"❌ 파일 처리 오류 ({filepath}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EQI 터널링 확률 분석기 v1.3 - 특정 JSON 파일을 직접 지정하여 분석합니다."
    )
    parser.add_argument(
        'files', 
        nargs='*', 
        help="분석할 JSON 파일 경로 (예: GW200115.json GW200129.json)"
    )
    parser.add_argument(
        '--pattern', 
        type=str, 
        default=None,
        help="파일명 패턴 (glob)으로 분석할 경우 사용 (예: --pattern 'GW200*.json')"
    )

    args = parser.parse_args()

    # 1. 수동 파일 목록이 있으면 우선 사용
    if args.files:
        target_files = args.files
    # 2. --pattern 옵션이 있으면 glob 확장
    elif args.pattern:
        target_files = glob.glob(args.pattern)
    # 3. 둘 다 없으면 기본 패턴 (하위 호환)
    else:
        print("ℹ️ 파일이 지정되지 않아 기본 패턴(eqi_monster_v91_ellipse_*.json)을 탐색합니다.")
        target_files = glob.glob("eqi_monster_v91_ellipse_*.json")

    analyze_files(target_files)
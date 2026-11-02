#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 === EQI QUANTUM TUNNELING PROBABILITY ANALYZER v1.2 === 🔬
=============================================================================
Architect: Claude (Phase Topology) & DeepSeek (Supreme Architect)
Engineer: Gemini (EQI Precision Engineer)
Date: 2026-07-22

[기능 및 v1.2 업데이트 내역]
- 기존에 추출된 v9.1 JSON 파일들을 읽어들여 각 청크별 '거시적 양자 터널링 확률' 계산.
- [v1.1] 하드코딩 제거: total_chunks 기반 동적 샘플링 적용.
- [v1.2] 🚨 ZONE-1(전이 금지 구간) 엄격한 밴드갭 처리 및 이론 위반 경고 시스템 도입.
=============================================================================
"""

import json
import numpy as np
import glob
import os

PHI = (np.sqrt(5.0) + 1.0) / 2.0
GAP = np.log(25.0) / 100.0
TIP_POINT = PHI**1.5 - 1.0  # 1.058171... (일함수)
T4 = PHI**1.5 - PHI**0.5    # 0.786151... (ZONE-0 상한)

def calculate_tunneling(json_pattern="eqi_monster_v91_ellipse_*.json"):
    files = glob.glob(json_pattern)
    if not files:
        print("⚠️ 처리할 JSON 파일이 없습니다.")
        return

    print("=================================================================")
    print(" 🔬 [EQI] 거시적 양자 터널링(Quantum Tunneling) 확률 분석기 v1.2")
    print(f"    - 장벽 에너지 (TIP_POINT) : {TIP_POINT:.6f}")
    print(f"    - 장벽 두께 (GAP)         : {GAP:.6f}")
    print(f"    - 전이 금지 구간 (ZONE-1) : {T4:.6f} ~ {TIP_POINT:.6f}")
    print("=================================================================")

    for filepath in sorted(files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            event = data.get("event", os.path.basename(filepath))
            chunks = data.get("chunk_results", [])
            
            print(f"\n▶ 이벤트: {event}")
            print(f"{'Chunk':>5} | {'Time(s)':>7} | {'Unity_rd (E)':>10} | {'P_tunnel (확률)':>15} | {'상태 (State)':<20}")
            print("-" * 75)

            total_chunks = len(chunks)

            for c in chunks:
                chunk_idx = c.get("chunk", 0)
                time_s = c.get("time_s", 0.0)
                unity_rd = c.get("unity_rd", 0.0)

                # 🚀 클로드 & 딥시크의 위상 양자화 터널링 방정식 적용
                if unity_rd < T4:
                    # ZONE-0: 순수 양자 터널링 영역 (결맞음, 탄성)
                    p_tunnel = np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)
                    state = "🌀 ZONE-0: 터널링 (Tunneling)"
                    p_str = f"{p_tunnel:.6f}" if p_tunnel >= 0.0001 else f"{p_tunnel:.2e}"
                    
                elif T4 <= unity_rd < TIP_POINT:
                    # ZONE-1: 전이 금지 구간 (이론상 데이터 없음, 치명적 경고 발생)
                    p_tunnel = np.exp(-2.0 * (TIP_POINT - unity_rd) / GAP)
                    state = "🚨 [경고] ZONE-1: 이론 위반 (Forbidden Zone Violation)"
                    p_str = f"{p_tunnel:.6f}" if p_tunnel >= 0.0001 else f"{p_tunnel:.2e}"
                    
                else:
                    # ZONE-2/3/4: 중력적 광전효과 (완전 투과, 소성 변형)
                    p_tunnel = 1.0
                    p_str = "1.000000"
                    state = "⚡ ZONE-2/3/4: 광전효과 (Photoelectric)"

                # 동적 샘플링 출력 (첫 5개, 마지막 5개, 또는 임계점 근처 및 경고 상황)
                if (chunk_idx < 5 or chunk_idx > total_chunks - 5 or 
                   (0.9 < p_tunnel < 1.0) or unity_rd > TIP_POINT * 0.9 or 
                   (T4 <= unity_rd < TIP_POINT)):
                     print(f"{chunk_idx:5d} | {time_s:6.1f}s | {unity_rd:10.6f} | {p_str:>15} | {state}")

            # 🔥 DeepSeek 추가 제안: 이벤트별 통계 요약 및 ZONE-1 위반 검출
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
                
                # 치명적 경고 출력
                if zone1 > 0:
                    print("   🚨🚨 [치명적 경고] ZONE-1 데이터 발견! EQI 공리계 예측 위반. 해당 청크 재검토 요망.")

        except Exception as e:
            print(f"❌ 파일 처리 오류 ({filepath}): {e}")

if __name__ == "__main__":
    calculate_tunneling()
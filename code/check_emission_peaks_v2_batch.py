#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 === EQI EMISSION PEAK & ZONE-4 BATCH SCANNER v2.0 === 🔍
=============================================================================
Usage:
    python check_emission_peaks_v2_batch.py eqi_monster_v91_ellipse_*.json
    python check_emission_peaks_v2_batch.py GW200*.json
    python check_emission_peaks_v2_batch.py (기본값: 모든 v91 JSON)
=============================================================================
"""

import json
import numpy as np
import glob
import sys
import os

PHI = (np.sqrt(5.0) + 1.0) / 2.0
TIP_POINT = PHI**1.5 - 1.0
T5 = PHI**1.5 - PHI**-1.5
THR_T3 = PHI**1.5 - PHI**-2.5

def analyze_files(file_list):
    if not file_list:
        print("⚠️ 분석할 JSON 파일이 없습니다.")
        return

    print("=" * 80)
    print("🔍 [EQI] ZONE-4 돌파율 및 방출량 피크 일괄 분석기 (Batch Scanner)")
    print(f"   TIP_POINT (T1) = {TIP_POINT:.6f} | T5 = {T5:.6f} | THR_T3 = {THR_T3:.6f}")
    print("=" * 80)
    print(f"{'Event Name':<20} | {'Max Unity_rd':>12} | {'Max E_D1':>12} | {'Zone (Peak)':<12} | {'Hit T3?'}")
    print("-" * 80)

    results = []

    for filepath in sorted(file_list):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            event = data.get("event", os.path.basename(filepath))
            chunks = data.get("chunk_results", [])
            
            if not chunks:
                continue

            # 데이터 추출
            unity_vals = [c.get("unity_rd", 0.0) for c in chunks]
            em_vals = [c.get("E_D1_emission", 0.0) for c in chunks]
            
            max_unity = max(unity_vals) if unity_vals else 0.0
            max_em = max(em_vals) if em_vals else 0.0
            idx_max = np.argmax(unity_vals) if unity_vals else -1
            
            # ZONE 분류 (피크 기준)
            if max_unity >= THR_T3:
                zone = "ZONE-4"
                hit_t3 = "✅ YES"
            elif max_unity >= T5:
                zone = "ZONE-3"
                hit_t3 = "❌ NO"
            elif max_unity >= TIP_POINT:
                zone = "ZONE-2"
                hit_t3 = "❌ NO"
            elif max_unity >= PHI**1.5 - PHI**0.5:
                zone = "ZONE-1"
                hit_t3 = "❌ NO"
            else:
                zone = "ZONE-0"
                hit_t3 = "❌ NO"

            results.append({
                "event": event,
                "max_unity": max_unity,
                "max_em": max_em,
                "zone": zone,
                "hit_t3": hit_t3,
                "file": os.path.basename(filepath),
                "count": len(chunks)
            })

        except Exception as e:
            print(f"❌ 오류 ({filepath}): {e}")

    # 정렬: 방출량 내림차순 (신기록 순)
    results.sort(key=lambda x: x["max_em"], reverse=True)

    for r in results:
        print(f"{r['event']:<20} | {r['max_unity']:12.6f} | {r['max_em']:12.6f} | {r['zone']:<12} | {r['hit_t3']}")
    
    print("-" * 80)
    print(f"✅ 총 {len(results)}개 이벤트 분석 완료.")
    
    # 주요 통계
    zone4_count = sum(1 for r in results if r['hit_t3'] == "✅ YES")
    print(f"📊 ZONE-4 (인과율 경계) 돌파 이벤트: {zone4_count}개")
    if results:
        print(f"🏆 최고 방출량: {results[0]['event']} ({results[0]['max_em']:.6f})")

if __name__ == "__main__":
    # 1. 커맨드 인자로 파일 목록이 들어오면 사용
    if len(sys.argv) > 1:
        # * 와일드카드 확장을 위해 glob 처리 (shell에서 안 될 경우 대비)
        files = []
        for pattern in sys.argv[1:]:
            files.extend(glob.glob(pattern))
    else:
        # 2. 아무것도 없으면 현재 디렉토리의 모든 v91 JSON 탐색
        print("ℹ️ 파일이 지정되지 않아 기본 패턴(eqi_monster_v91_ellipse_*.json)을 탐색합니다.")
        files = glob.glob("eqi_monster_v91_ellipse_*.json")
    
    analyze_files(files)
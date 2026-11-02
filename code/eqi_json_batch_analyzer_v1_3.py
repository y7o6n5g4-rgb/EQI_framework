#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 === EQI JSON BATCH ANALYZER v1.3 (MULTI-EVENT MASTER TABLE) === 📊
=============================================================================
[v9.4 / v2009.x 다중 JSON 일괄 분석기]
1. 폴더 내 모든 v9.4/v2009 json 파일을 자동 탐색
2. 20개 이벤트의 Re_EQI, Virial 평형, 상전이 구역(ZONE) 분포 일괄 추출
3. 이벤트별 마스터 요약 테이블(Master Table) 출력
=============================================================================
"""

import json
import glob
import os
import numpy as np

def main():
    print("=" * 115)
    print(" 📊 EQI Framework JSON Batch Analyzer v1.3 (Multi-Event Master Table)")
    print("=" * 115)

    # v9.4 및 v2009 관련 JSON 파일 모두 찾기
    files = glob.glob("eqi_monster_v94_vector_*.json") + glob.glob("eqi_monster_v2009_*.json")
    
    if not files:
        print("❌ 오류: 분석할 JSON 파일을 찾을 수 없습니다.")
        print("   현재 폴더에 'eqi_monster_v94_...json' 파일들이 있는지 확인해 주세요.")
        return

    print(f"📂 총 {len(files)}개의 이벤트 JSON 파일을 발견했습니다. 일괄 분석을 시작합니다...\n")

    # 테이블 헤더 출력
    header_fmt = "{:<16} | {:>6} | {:>6} | {:>6} | {:>6} | {:>8} | {:>9} | {:>9} | {:>10} | {:>10}"
    print("-" * 115)
    print(header_fmt.format(
        "Event Name", "Chunks", "Z-0(%)", "Z-1(%)", "Z-2(%)", 
        "Avg u_rd", "Avg n_ou", "Avg Re_EQ", "Avg Virial", "Maxw_Res"
    ))
    print("-" * 115)

    TIP_POINT = 1.058171
    summary_data = []

    for filepath in sorted(files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"파일 읽기 오류 ({filepath}): {e}")
            continue

        event = data.get("event_name", data.get("event", "Unknown"))
        chunks = data.get("chunk_results", [])
        if not chunks:
            continue

        # 통계 변수
        u_rds = []
        n_ouros = []
        re_eqis = []
        virials = []
        maxw_res = 0.0
        
        zones = {"Z-0": 0, "Z-1": 0, "Z-2+": 0}

        for c in chunks:
            u_rd = c.get("unity_rd", 0.0)
            n_ou = c.get("n_ouroboros", 0.0)
            re_eqi = c.get("re_eqi", 0.0)
            virial = c.get("virial_ratio", 0.0)
            
            # Maxw_Res는 구조상 거의 고정이므로 첫 번째 청크 값 사용
            if maxw_res == 0.0:
                maxw_res = c.get("maxwell_residual", c.get("euler_residual", 0.0))

            u_rds.append(u_rd)
            n_ouros.append(n_ou)
            re_eqis.append(re_eqi)
            virials.append(virial)

            # Zone 판별
            if u_rd >= TIP_POINT:
                zones["Z-2+"] += 1
            elif u_rd >= 0.5:
                zones["Z-1"] += 1
            else:
                zones["Z-0"] += 1

        # 통계 산출
        total = len(chunks)
        z0_pct = (zones["Z-0"] / total) * 100
        z1_pct = (zones["Z-1"] / total) * 100
        z2_pct = (zones["Z-2+"] / total) * 100
        
        mean_u_rd = np.mean(u_rds)
        mean_n_ou = np.mean(n_ouros)
        mean_re_eqi = np.mean(re_eqis)
        mean_virial = np.mean(virials)

        # 테이블 행 출력
        print(header_fmt.format(
            event[:16], 
            total, 
            f"{z0_pct:5.1f}%", 
            f"{z1_pct:5.1f}%", 
            f"{z2_pct:5.1f}%", 
            f"{mean_u_rd:8.4f}", 
            f"{mean_n_ou:9.2f}", 
            f"{mean_re_eqi:9.6f}", 
            f"{mean_virial:10.1f}", 
            f"{maxw_res:10.3e}"
        ))

        # 전체 종합을 위해 데이터 저장
        summary_data.append({
            "event": event,
            "z0_pct": z0_pct,
            "z1_pct": z1_pct,
            "z2_pct": z2_pct,
            "mean_re_eqi": mean_re_eqi,
            "mean_virial": mean_virial
        })

    print("-" * 115)
    
    if summary_data:
        avg_all_re_eqi = np.mean([d["mean_re_eqi"] for d in summary_data])
        avg_all_virial = np.mean([d["mean_virial"] for d in summary_data])
        
        print("\n[🚀 20개 이벤트 우주적 스케일 종합 진단]")
        print(f"  • 분석된 총 이벤트 수 : {len(summary_data)} 개")
        print(f"  • 전체 평균 Re_EQI    : {avg_all_re_eqi:.6f} " + 
              ("(Fermat n=2 선형 층류 지배)" if avg_all_re_eqi < 0.5 else "(비리얼 돌파)"))
        print(f"  • 전체 평균 비리얼 비 : {avg_all_virial:.1f} (tau_true / tau_nl)")
        
        bns_candidates = [d["event"] for d in summary_data if d["z1_pct"] > 0 or d["z2_pct"] > 0]
        if bns_candidates:
            print(f"  • 상전이(Z-1 이상) 관측 이벤트: {', '.join(bns_candidates)} (중성자별 충돌 등 광전효과 동반 유력)")
        else:
            print("  • 상전이 관측 이벤트: 없음 (전체 진공 채널 순수 중력파)")

    print("===================================================================================================================")

if __name__ == "__main__":
    main()
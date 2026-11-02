#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 === EQI JSON ANALYZER v1.2 (VECTOR & RE_EQI) === 📊
=============================================================================
[v9.4 / v2009.x JSON 전용 분석기]
1. Re_EQI 및 Virial Ratio 비리얼 평형 실시간 추적
2. D2 Vector Product (Area Vector, EMF) 및 로렌츠 아날로그 추출
3. Maxwell Residual 및 n_ouroboros 상태 출력
4. Zone 자동 분류 (ZONE-0, ZONE-1, ZONE-2+)
=============================================================================
"""

import json
import sys
import os
import glob
import numpy as np

def main():
    print("=" * 90)
    print(" 📊 EQI Framework JSON Analyzer v1.2 (Vector & Re_EQI)")
    print("=" * 90)

    # 1. 파일 자동 탐색 또는 인자 받기
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # 현재 폴더에서 v94 또는 v2009 관련 json 찾기
        files = glob.glob("eqi_monster_v94_*.json") + glob.glob("eqi_monster_v2009_*.json")
        if not files:
            print("❌ 오류: 분석할 JSON 파일을 찾을 수 없습니다.")
            print("   사용법: python eqi_json_analyzer_v1_2_vector.py [파일명.json]")
            sys.exit(1)
        # 최신 파일 선택
        target_file = max(files, key=os.path.getmtime)
    
    target_file = target_file.strip("'\"")
    
    print(f"\n📂 파일 로드 중: {target_file}")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        sys.exit(1)

    # 2. 메타데이터 파싱
    version = data.get("version", "Unknown")
    event = data.get("event_name", data.get("event", "Unknown"))
    channel = data.get("channel", "Unknown")
    
    # 3. v2009 이론 블록 (없으면 기본값)
    v2009 = data.get("v2009_theory", {})
    a_eqi = v2009.get("A_EQI", 0.0)
    emf_base = v2009.get("EMF_base", 0.0)
    r_s_eqi = v2009.get("r_s_EQI", 0.0)

    print(f"\n[기본 정보]")
    print(f"  • Event:   {event} ({channel} Channel)")
    print(f"  • Version: {version}")
    if a_eqi:
        print(f"  • A_EQI (단위 타원 면적): {a_eqi:.6f}")
        print(f"  • EMF_base (정보 기전력): {emf_base:.6e}")
        print(f"  • r_s (슈바르츠실트):   {r_s_eqi:.6f}")

    # 4. 청크 데이터 파싱
    chunks = data.get("chunk_results", [])
    if not chunks:
        print("\n❌ 오류: chunk_results 데이터가 없습니다.")
        sys.exit(1)

    # 통계용 리스트
    u_rds = []
    n_ouros = []
    re_eqis = []
    virials = []
    emfs = []
    maxw_res = []
    zones = {"ZONE-0": 0, "ZONE-1": 0, "ZONE-2+": 0}

    TIP_POINT = 1.058171

    # 테이블 헤더 출력
    print("\n[청크별 정밀 벡터 & 마찰 지표 (M=1.4M☉ 기준)]")
    print("-" * 105)
    print(f"{'Chunk':>5} | {'Time(s)':>7} | {'ZONE':>6} | {'u_rd':>7} | {'n_ouro':>7} | {'Re_EQI':>9} | {'Virial':>8} | {'EMF_scale':>11} | {'Maxw_Res':>11}")
    print("-" * 105)

    for c in chunks:
        idx = c.get("chunk", 0)
        t_s = c.get("time_s", 0.0)
        u_rd = c.get("unity_rd", 0.0)
        n_ou = c.get("n_ouroboros", 0.0)
        re_eqi = c.get("re_eqi", 0.0)
        virial = c.get("virial_ratio", 0.0)
        emf = c.get("emf_quantum_scale", c.get("EMF_EQI", 0.0))
        maxw = c.get("maxwell_residual", c.get("euler_residual", 0.0))
        
        u_rds.append(u_rd)
        n_ouros.append(n_ou)
        re_eqis.append(re_eqi)
        virials.append(virial)
        emfs.append(emf)
        maxw_res.append(maxw)

        # Zone 판별 (대략적인 기준: TIP_POINT 초과면 2+, 0.5이상이면 1)
        if u_rd >= TIP_POINT:
            zone_str = "Z-2+ "
            zones["ZONE-2+"] += 1
        elif u_rd >= 0.5:
            zone_str = "Z-1 🔶"
            zones["ZONE-1"] += 1
        else:
            zone_str = "Z-0  "
            zones["ZONE-0"] += 1

        # 64개 이하일 때는 모두 출력, 아니면 건너뛰기 렌더링 (GW150914는 64개이므로 모두 출력됨)
        if len(chunks) <= 100 or idx % (len(chunks)//20) == 0:
            print(f"{idx:5d} | {t_s:7.3f} | {zone_str} | {u_rd:7.4f} | {n_ou:7.2f} | {re_eqi:9.6f} | {virial:8.1f} | {emf:11.3e} | {maxw:11.3e}")

    print("-" * 105)

    # 5. 종합 요약 통계
    total_chunks = len(chunks)
    print("\n[📊 분석 종합 요약]")
    print(f"  • 상전이 구역 분포: ZONE-0: {zones['ZONE-0']/total_chunks*100:.1f}% | ZONE-1: {zones['ZONE-1']/total_chunks*100:.1f}% | ZONE-2+: {zones['ZONE-2+']/total_chunks*100:.1f}%")
    print(f"  • 평균 Unity_rd (에일리어싱 텐션): {np.mean(u_rds):.4f}")
    print(f"  • 평균 n_ouroboros (순환 단계):  {np.mean(n_ouros):.2f}")
    print(f"  • 평균 Re_EQI (비선형/선형 마찰):  {np.mean(re_eqis):.6f} " + ("(완전 층류, Fermat n=2)" if np.mean(re_eqis) < 0.5 else "(비리얼 평형 돌파)"))
    print(f"  • 평균 Virial Ratio (열역학 평형): {np.mean(virials):.1f}")
    print(f"  • 최대 패러데이 정보 기전력(EMF):  {np.max(emfs):.3e}")
    
    print("\n=========================================================================================")
    print(" 💡 [Maestro 진단 포인트]")
    print(" 1. GW150914의 경우 진공 채널(BBH) 특성상 100% ZONE-0에 수렴하는지 확인하십시오.")
    print(" 2. Re_EQI 값이 0.5에 한참 못 미치는 0.00x 대라면, 현재 우주의 페르마 n=2 층류 영역임을 뜻합니다.")
    print(" 3. EMF_scale이 D2 면적벡터를 통한 패러데이 정보 마찰의 변동성을 보여줍니다.")
    print("=========================================================================================\n")

if __name__ == "__main__":
    main()
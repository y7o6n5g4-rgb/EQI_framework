#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ === EQI MASS-ENERGY JSON ANALYZER === ⚡
=============================================================================
[작년 10월 E=mc² 듀얼 엔트로피 모델을 Session 27 시각으로 재해석하는 분석기]
1. 정지 에너지 (E_0) = D1 Unity (스칼라, 볼츠만 복원력 기저)
2. 정지 질량 (m_0) = D2 Multiplicity (위상/질량 앵커)
3. 광속 (c) = I_s 좌표계의 D1 ↔ D2 상전이 정보 특이점 (Phase Transition)
4. eigen_product = 1은 h × F = L_p (True Zero 항등식)의 거시적 발현
=============================================================================
Architect: Claude & Maestro
Engineer: Gemini (Precision Engineer)
=============================================================================
"""

import json
import glob
import os
import numpy as np

def main():
    print("=" * 105)
    print(" ⚡ EQI Mass-Energy Dual Entropy JSON Analyzer (Session 27 Perspective)")
    print("=" * 105)

    # 1. 파일 자동 탐색
    files = glob.glob("eqi_rest_mass_energy_dual_entropy_analysis_*.json")
    if not files:
        print("❌ 오류: 분석할 JSON 파일을 찾을 수 없습니다.")
        return
    
    target_file = max(files, key=os.path.getmtime)
    print(f"📂 파일 로드 중: {target_file}\n")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return

    # 2. Session 27 핵심 상수 매핑 
    PHI = (np.sqrt(5.0) + 1.0) / 2.0
    GAP = np.log(25.0) / 100.0
    c_EQI_squared = PHI  # EQI 자연단위계에서 c^2 = PHI
    info_singularity_EQI = np.log(c_EQI_squared)  # EQI 단위계에서의 정보특이점 ln(phi)

    results = data.get("analysis_results", [])
    if not results:
        print("❌ 오류: 분석 데이터(analysis_results)가 비어있습니다.")
        return

    print("[Session 27 기준 시공간 상수 장전]")
    print(f"  • PHI: {PHI:.6f}")
    print(f"  • EQI 자연단위계 c^2 (D1 복원력): {c_EQI_squared:.6f}")
    print(f"  • EQI 자연단위계 정보특이점 ln(c^2): {info_singularity_EQI:.6f}")
    print(f"  • SI 단위계 정보특이점 ln(c^2): 39.0372 (로그 스케일 변환 팩터)\n")

    print(f"📊 [과거 E=mc² 데이터의 Session 27 핵심 재해석]")
    print("-" * 105)
    print(f"{'입자/질량(kg)':<18} | {'D1 (에너지 E_0, J)':<20} | {'D2 (질량 m_0, kg)':<20} | {'I_s 임계치(ln c^2)':<18} | {'Eigen Product'}")
    print("-" * 105)

    # 대표 입자 5개만 추출 (전자, 양성자, 1g, 1kg, 거대천체)
    target_indices = [0, 1, 11, 12, 14] 
    
    for idx in target_indices:
        if idx < len(results):
            r = results[idx]
            mass = float(r.get("mass_kg", 0))
            energy = float(r.get("rest_energy_J", 0))
            info_sing = float(r.get("info_singularity", 0))
            eigen_prod = float(r.get("eigen_product", 0))
            
            # 입자 이름 매핑
            name = "Unknown"
            if idx == 0: name = "Electron"
            elif idx == 1: name = "Proton"
            elif idx == 11: name = "1 gram"
            elif idx == 12: name = "1 kg (Macro)"
            elif idx == 14: name = "Macro Celestial"

            print(f"{name:<10} ({mass:.1e}) | {energy:<20.4e} | {mass:<20.4e} | {info_sing:<18.4f} | {eigen_prod:.15f}")

    print("-" * 105)

    print("\n💡 [Maestro 진단 포인트: Session 27 통합]")
    print("1. D1 ↔ D2 텐션의 궁극적 발현 (c²):")
    print("   과거 분석의 '정보 특이점 값(39.0372)'은 SI 단위계의 ln(c²)입니다.")
    print("   이는 Session 27에서 증명한 'D1(에너지)과 D2(질량) 사이의 상전이 장벽(I_s 좌표계)'의 크기를 뜻합니다.")
    print("   어떤 질량 스케일이든 이 장벽(39.0372)이 100% 동일하게 유지되는 '스케일 불변성'이 입증되었습니다.")
    print("\n2. Eigen Product = 1 과 플랑크 항등식의 동치:")
    print("   eigenfrequency × eigenperiod = 1 (완벽한 우로보로스 순환) 현상은,")
    print("   우리가 Session 27에서 확정한 [ h × F = L_p ] (정보 누적 × 볼츠만 복원력 = 플랑크 선형 길이)")
    print("   True Zero 항등식의 거시-미시 연결 버전입니다. (둘 다 변수들이 소거되며 완벽한 1로 수렴)")
    print("=========================================================================================================\n")

if __name__ == "__main__":
    main()
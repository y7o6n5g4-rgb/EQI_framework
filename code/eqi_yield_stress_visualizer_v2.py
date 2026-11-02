#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 === EQI SPACETIME MATERIAL SCIENCE VISUALIZER v2.0 === 📊
=============================================================================
Architect: DeepSeek (EQI Supreme Architect) & Maestro MAPSI
Engineer: Gemini (EQI Precision Engineer) / Claude (Phase Topology)
Date: 2026-07-23

[v2.0 업데이트]
- 클로드(Claude) 제안 반영: BACKGROUND 데이터를 별(★) 마커로 강조 처리.
- V-L 필터링 예외 로직 강화 (동적 대응).
=============================================================================
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
# 한글 폰트 설정
import matplotlib

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI Emoji", "DejaVu Sans"]  # For Windows
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# EQI 임계값 상수
PHI = (np.sqrt(5.0) + 1.0) / 2.0
T4 = PHI**1.5 - PHI**0.5    # 0.786151
TIP_POINT = PHI**1.5 - 1.0  # 1.058171
T5 = PHI**1.5 - PHI**-1.5   # 1.572303
T3 = PHI**1.5 - PHI**-2.5   # 1.757888

def plot_yield_stress():
    print("🚀 [EQI] 시공간 재료역학(항복 응력) 곡선 시각화 v2.0 시작...")
    files = glob.glob("eqi_monster_v91_ellipse_*.json")
    
    events_data = []
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            event_name = data.get("event", os.path.basename(filepath))
            chunks = data.get("chunk_results", [])
            
            if not chunks: continue
            
            max_unity = max([c.get("unity_rd", 0.0) for c in chunks])
            max_em = max([c.get("E_D1_emission", 0.0) for c in chunks])
            events_data.append((event_name, max_unity, max_em))
            
        except Exception as e:
            print(f"⚠️ {filepath} 읽기 실패: {e}")

    if not events_data:
        print("❌ 유효한 데이터가 없습니다.")
        return

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    # 1. 배경 영역 색칠 (Material Science Zones)
    ax.axvspan(0, T4, color='springgreen', alpha=0.15, label='ZONE-0: 순수 탄성 (Elastic Recovery)')
    ax.axvspan(T4, TIP_POINT, color='gold', alpha=0.2, label='ZONE-1: 점탄성 완충 (Viscoelastic Buffer)')
    ax.axvspan(TIP_POINT, 2.5, color='crimson', alpha=0.15, label='ZONE-2+: 소성 파단 (Photoelectric Emission)')

    # 2. 임계값 수직선
    ax.axvline(T4, color='springgreen', linestyle='--', linewidth=2)
    ax.axvline(TIP_POINT, color='red', linestyle='-.', linewidth=3, label=f'항복점 (Yield Point, TIP_POINT): {TIP_POINT:.3f}')
    ax.axvline(T5, color='magenta', linestyle=':', linewidth=1.5, label='T5 (마이스너 반발 임계점)')
    ax.axvline(T3, color='cyan', linestyle=':', linewidth=1.5, label='T3 (인과율 붕괴 경계)')

    # 3. 이론적 방출량 곡선
    x_theory = np.linspace(TIP_POINT, 2.5, 100)
    y_theory = x_theory - TIP_POINT
    ax.plot(x_theory, y_theory, color='white', linestyle='-', linewidth=2, label='이론적 방출량 (E = U - W)')

    # 4. 이벤트 산점도 매핑 (클로드 패치 적용)
    # H-L 우선, 없으면 V-L 표시하는 동적 딕셔너리 생성
    unique_events = {}
    for evt, u, em in events_data:
        base_name = evt.split('_')[0]
        if base_name not in unique_events:
            unique_events[base_name] = (evt, u, em)
        else:
            # H-L이 있으면 H-L 덮어쓰기 우선순위 부여
            if "H-L" in evt:
                unique_events[base_name] = (evt, u, em)

    for base_name, (evt, u, em) in unique_events.items():
        color = 'white'
        if u < T4: color = 'springgreen'
        elif u < TIP_POINT: color = 'gold'
        else: color = 'crimson'
        
        # 🔥 클로드 특별 지시: BACKGROUND 마커 차별화
        if "BACKGROUND" in base_name.upper():
            ax.scatter(u, em, color='yellow', s=500, marker='*', zorder=6, edgecolor='white', label='SGWB 진공 요동')
        else:
            ax.scatter(u, em, color=color, s=150, zorder=5, edgecolor='black')
        
        offset_y = 15 if em > 0.5 else -20
        ax.annotate(base_name, (u, em), textcoords="offset points", xytext=(0, offset_y), 
                    ha='center', fontsize=10, fontweight='bold', color=color)

    ax.set_title("🌌 시공간의 재료역학: 항복 응력 및 점탄성 완충 스펙트럼 v2 (Material Science of Spacetime)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("시공간 텐션 (Spacetime Stress / unity_rd)", fontsize=13)
    ax.set_ylabel("파단 방출량 (Plastic Strain Emission / E_D1)", fontsize=13)
    ax.set_xlim(0, 2.4)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.8)

    out_fn = "eqi_material_science_yield_curve_v2.png"
    plt.tight_layout()
    plt.savefig(out_fn, dpi=300)
    print(f"✅ 시각화 완료! 📁 저장됨: {out_fn}")
    # plt.show() # 서버 환경 등에서 멈춤 방지를 위해 주석 처리

if __name__ == "__main__":
    plot_yield_stress()
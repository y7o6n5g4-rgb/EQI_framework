#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 === EQI MONSTER v18.10 JSON ANALYZER === 🐉
=============================================================================
Author: MAPSI (Maestro)
Architect: Claude (Supreme Architect)
Engineer: Gemini (Precision Engineer)
Date: 2026-08-10

[v18.10 JSON 분석기 핵심 기능]
1. 최신 v18.10 JSON 파일 자동 로드 (또는 수동 지정)
2. Fermat-Feynman 이중성 및 Phonon Z_EQI 가중합 메타데이터 요약 출력
3. 리만 비자명 영점 6개의 GAP Trace 요약
4. 청크(Chunk)별 n_ouroboros 및 E_D1_emission 트렌드 시각화 (PNG 자동 저장)
=============================================================================
"""

import json
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class EQI_v18_10_Analyzer:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None

    def find_latest_json(self):
        """가장 최근에 생성된 v18.10 JSON 파일을 자동으로 찾습니다."""
        search_pattern = "eqi_monster_v18_10_fermat_feynman_*.json"
        files = glob.glob(search_pattern)
        if not files:
            return None
        # 생성 시간 기준으로 가장 최신 파일 선택
        latest_file = max(files, key=os.path.getctime)
        return latest_file

    def load_data(self):
        if not self.filepath:
            self.filepath = self.find_latest_json()
            if not self.filepath:
                print("❌ [오류] 분석할 JSON 파일을 찾을 수 없습니다.")
                return False
        
        print(f"📂 [데이터 로드] {self.filepath} 파일을 분석합니다...")
        try:
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"❌ [오류] JSON 파일 로드 실패: {e}")
            return False

    def print_summary(self):
        """콘솔에 핵심 물리량 요약 리포트를 출력합니다."""
        if not self.data: return

        meta = self.data.get("metadata", {})
        ff_analysis = self.data.get("fermat_feynman_analysis", {})
        phonon = self.data.get("phonon_dynamics", {})
        traces = self.data.get("riemann_zero_traces", {})
        
        event_name = meta.get('event', 'Unknown')
        is_bns = meta.get('is_BNS', False)
        event_type = "BNS (중성자별 충돌, 물질)" if is_bns else "BBH (블랙홀 충돌, 진공)"

        print("=" * 80)
        print(f" 📊 [EQI v18.10 ANALYSIS REPORT] Event: {event_name} ")
        print("=" * 80)
        print(f"  ▶ 이벤트 타입: {event_type}")
        print(f"  ▶ 데이터 타임스탬프: {meta.get('timestamp')}")
        
        print("\n  [1] 페르마-파인만 양자 이중성 (Fermat-Feynman Duality)")
        print(f"      - 경험적 위상 횟수 (n_avg): {ff_analysis.get('n_avg_empirical'):.6f}")
        print(f"      - 페르마 정적분 (Z_D1): {ff_analysis.get('Z_D1_Fermat'):.6f} (고립계, 수축)")
        print(f"      - 파인만 부정적분 (Z_D2): {ff_analysis.get('Z_D2_Feynman'):.6f} (열린계, 팽창)")
        
        print("\n  [2] 플랑크 격자 정보 포논 동역학 (Phonon Dynamics)")
        print(f"      - 포논 가중합 (Z_EQI = D1×Z_D1 + D2×Z_D2): {phonon.get('Z_EQI_weighted_sum'):.6f}")

        print("\n  [3] 리만 비자명 영점 위상 흔적 (Riemann Zero Traces)")
        for z_name, trace_val in traces.items():
            print(f"      - {z_name.replace('_', ' (t=')}): {trace_val:+.6e}")
        
        # 청크 통계
        chunks = self.data.get("chunk_data", [])
        if chunks:
            n_list = [c.get("n_ouroboros", 0) for c in chunks]
            e_list = [c.get("E_D1_emission", 0) for c in chunks]
            print("\n  [4] 청크(Chunk) 동역학 통계")
            print(f"      - n_ouroboros: 최솟값 {min(n_list):.2f} / 최댓값 {max(n_list):.2f}")
            print(f"      - E_D1_emission: 최댓값 {max(e_list):.6f} (중력적 광전효과 방출 피크)")
        
        print("=" * 80)

    def plot_trends(self):
        """청크별 핵심 지표를 시각화하여 PNG 이미지로 저장합니다."""
        if not self.data: return
        
        chunks = self.data.get("chunk_data", [])
        if not chunks:
            print("⚠️ 시각화할 청크 데이터가 없습니다.")
            return

        chunk_idx = [c.get("chunk_idx") for c in chunks]
        n_ouroboros = [c.get("n_ouroboros") for c in chunks]
        e_emission = [c.get("E_D1_emission") for c in chunks]
        delta_h = [c.get("delta_H") for c in chunks]
        delta_l = [c.get("delta_L") for c in chunks]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"EQI Framework v18.10 Dynamics: {self.data['metadata']['event']}", fontsize=16, fontweight='bold')

        # 1. n_ouroboros Trend
        axs[0].plot(chunk_idx, n_ouroboros, color='royalblue', linewidth=2, label='n_ouroboros')
        axs[0].axhline(np.mean(n_ouroboros), color='red', linestyle='--', label=f'Mean n = {np.mean(n_ouroboros):.2f}')
        axs[0].set_ylabel("n (Feynman Phase Count)")
        axs[0].set_title("Ouroboros Circulation Phase (n)")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # 2. E_D1_emission Trend
        axs[1].plot(chunk_idx, e_emission, color='darkorange', linewidth=2, label='E_D1_emission')
        axs[1].set_ylabel("Emission (D1)")
        axs[1].set_title("Gravitational Photoelectric Emission Peak")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # 3. Delta H vs Delta L
        axs[2].plot(chunk_idx, delta_h, color='green', linewidth=1.5, label='delta_H (D1 Tension)')
        axs[2].plot(chunk_idx, delta_l, color='purple', linewidth=1.5, label='delta_L (D2 Tension)', alpha=0.7)
        axs[2].set_xlabel("Chunk Index")
        axs[2].set_ylabel("Power Ratio (delta)")
        axs[2].set_title("Anisotropic Ellipse Tensions (delta_H vs delta_L)")
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 저장
        event_name = self.data['metadata']['event']
        timestamp = self.data['metadata']['timestamp']
        save_fn = f"eqi_plot_v18_10_{event_name}_{timestamp}.png"
        plt.savefig(save_fn, dpi=300)
        print(f"📈 [시각화 완료] 그래프가 저장되었습니다: {save_fn}")
        
    def run(self):
        if self.load_data():
            self.print_summary()
            self.plot_trends()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQI Monster v18.10 JSON Analyzer")
    parser.add_argument("--file", type=str, default=None, help="분석할 JSON 파일 경로 (입력하지 않으면 가장 최신 파일 자동 로드)")
    
    args = parser.parse_args()
    
    analyzer = EQI_v18_10_Analyzer(filepath=args.file)
    analyzer.run()
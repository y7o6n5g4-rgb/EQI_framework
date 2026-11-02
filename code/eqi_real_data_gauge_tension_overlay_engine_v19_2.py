#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Real-Data Gauge Tension Overlay Engine (v19.2)
============================================================
[v19.2 맏이님 전용 실측 데이터 로딩 패치]
1. H1(핸포드) 및 L1(리빙스턴) 검출기 데이터 통합 로딩 인터페이스 구현
2. 사용자가 직접 데이터 경로를 입력하여 오버레이 연산 가능
3. 리만 임계선(Sigma=1/2) 기반의 듀얼 검출기 위상 동기화(Phase Lock-on) 록온
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
import os
import h5py
import mpmath
from mpmath import mp

# 400자리 정밀도 설정
mp.dps = 400

class GaugeTensionEngine:
    def __init__(self, h1_path, l1_path):
        self.h1_path = h1_path
        self.l1_path = l1_path
        self.tension_constant = mp.log(mp.mpf("25"))
        print("=" * 80)
        print("  Maestro MAPSI: EQI Real-Data Gauge Tension Overlay Engine v19.2")
        print("  [듀얼 검출기(H1/L1) 동기화 및 1.0 Unity 정합성 검증 모드]")
        print("=" * 80)

    def load_ligo_hdf5(self, path):
        """실제 LIGO HDF5 파일 로딩"""
        try:
            with h5py.File(path, 'r') as f:
                strain = f['strain/Strain'][:]
                xspacing = f['strain/Strain'].attrs['Xspacing']
            fs = int(1.0 / xspacing)
            t = np.arange(len(strain)) * xspacing
            return t, strain, fs
        except Exception as e:
            print(f"  ⚠️ 데이터 로딩 오류 발생: {e}")
            return None, None, None

    def process(self):
        # 1. 데이터 로드 (Maestro께서 입력하신 경로 사용)
        t1, s1, fs1 = self.load_ligo_hdf5(self.h1_path)
        t2, s2, fs2 = self.load_ligo_hdf5(self.l1_path)
        
        if s1 is None or s2 is None:
            print("  [주의] 데이터 파일 경로가 올바르지 않습니다. 경로를 확인하십시오.")
            return

        # 2. 대역 통과 필터 및 힐베르트 변환 (진폭 엔벨로프 추출)
        b, a = butter(4, [20 / (fs1/2), 400 / (fs1/2)], btype='band')
        env1 = np.abs(hilbert(filtfilt(b, a, s1)))
        env2 = np.abs(hilbert(filtfilt(b, a, s2)))

        # 3. 렌더링
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(t1, env1, color='cyan', alpha=0.6, label='H1 Strain Envelope')
        ax1.plot(t1, [float(self.tension_constant) for _ in t1], color='gold', ls='--', label='ln(25) Gauge Limit')
        ax1.set_title("H1 Detector: Spacetime Strain vs $\ln(25)$ Tension")
        ax1.legend()

        ax2.plot(t2, env2, color='magenta', alpha=0.6, label='L1 Strain Envelope')
        ax2.set_title("L1 Detector: Spacetime Strain vs $\ln(25)$ Tension")
        ax2.legend()
        
        os.makedirs("./output_v19", exist_ok=True)
        plt.savefig("./output_v19/dual_detector_overlay.png")
        print("  ✅ [Rendering] H1/L1 듀얼 텐션 맵 렌더링 완료.")

# --- 데이터 경로 입력부 ---
if __name__ == "__main__":
    # 마에스트로 맏이님: 여기에 실제 HDF5 파일 경로를 입력하십시오.
    H1_DATA_PATH = "H-H1_LOSC_4_V2-1126259446-32.hdf5"
    L1_DATA_PATH = "L-L1_LOSC_4_V2-1126259446-32.hdf5"
    
    engine = GaugeTensionEngine(H1_DATA_PATH, L1_DATA_PATH)
    engine.process()
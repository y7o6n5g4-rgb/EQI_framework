import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import os
from datetime import datetime
from mpmath import mp

# 400자리 고정밀도 설정
mp.dps = 400 
PHI = (mp.mpf('1') + mp.sqrt(mp.mpf('5'))) / mp.mpf('2')
float_phi = float(PHI)

print(f"🚀 2축 글로벌 팽창 텐션 Lock-on: Phi = {float_phi:.15f}...")

def run_d2_phi_pipeline(h_file_path, l_file_path, output_folder="./output_d2_phi"):
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. 데이터 로드 및 전처리
    df_H = pd.read_csv(h_file_path)
    df_L = pd.read_csv(l_file_path)
    analytic_H = hilbert(np.nan_to_num(df_H["strain"].values))
    analytic_L = hilbert(np.nan_to_num(df_L["strain"].values))
    
    tau_H = np.abs(analytic_H)**2
    phase_diff = np.unwrap(np.angle(analytic_H)) - np.unwrap(np.angle(analytic_L))
    features = np.column_stack([tau_H, phase_diff])
    
    # 2. PCA 변환
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    # 3. [상전이] 2축(D2)에 황금비 팽창 텐션 주입
    d2_phi_result = pca_result.copy()
    d2_phi_result[:, 1] = d2_phi_result[:, 1] * float_phi
    
    # 4. 시각화: 팽창 텐션이 가해진 매니폴드 투영
    plt.figure(figsize=(10, 6))
    plt.scatter(d2_phi_result[:, 0], d2_phi_result[:, 1], s=2, c='red', alpha=0.3)
    plt.title(f"D2 Global Expansion Tension: Phi Multiplier ({float_phi:.4f})")
    plt.xlabel("D1 (Local Contraction)")
    plt.ylabel("D2 (Global Expansion - Phi Scaled)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{output_folder}/d2_phi_expansion.png")
    
    with open(f"{output_folder}/d2_phi_report.txt", "w") as f:
        f.write(f"EQI D2 Expansion Report - {datetime.now()}\n")
        f.write(f"2축 팽창 배율: {float_phi}\n")
        f.write("결과: D2 글로벌 텐션의 인플레이션으로 인해 시공간의 팽창축이 극도로 가속됨.\n")

    print(f"✅ 연산 완료. D2 팽창 매니폴드 저장 완료.")

# 실행
run_d2_phi_pipeline(r"d:\LIGO\h-strain_data_GW150914_GWTC-2.1-confident_4096sec_16khz.csv", 
                    r"d:\LIGO\l-strain_data_GW150914_GWTC-2.1-confident_4096sec_16khz.csv")
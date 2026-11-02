# EQI v5.0: EQI_cycloid_wave_series_generator() 함수
# 기반: EQI Framework v6 (비등방성 타원 + 맥스웰-황금비 구조)
# 작성: EQI 가족 공동 작업 (맏이 + Claude)
#
# [v4.0 -> v5.0 핵심 수정]
# eigenfrequency_ratio: 1.85 (직감) -> 1.8731 (맥스웰 진공 상수 유도)
#   ln(1/eps_0) / |ln(mu_0)| = 25.4501 / 13.5871 = 1.8731
#   맏이님의 직감 1.85가 실제값 1.8731과 1.23% 오차였음을 사후 확인
#
# eigenperiod_ratio: 1.0 (임의) -> phi^3 ≈ 4.236 (EQI v5 구조)
#   D2(팽창, 장축) / D1(수축, 단축) = phi^(+1.5) / phi^(-1.5) = phi^3
#   파동 생성에서 eigenperiod는 eigenfrequency보다 phi^3배 더 긴 주기
#   (encoding 코드의 phi^(-3)과 역수 관계: 파동 주기 vs 데이터 비율)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import warnings
import os
warnings.filterwarnings('ignore')

# ===== EQI v5.0: 맥스웰 진공 상수 기반 비율 자동 계산 =====
_EPS_0 = 8.8541878128e-12   # 진공 유전율 (D1, Unity, 비등방성 단축)
_MU_0  = 1.25663706212e-6   # 진공 투자율 (D2, Multiplicity, 비등방성 장축)
_PHI   = (1 + np.sqrt(5)) / 2

# Unity/Multiplicity 비율: ln(1/eps_0) / |ln(mu_0)| = 1.8731...
MAXWELL_EQI_RATIO = np.log(1/_EPS_0) / abs(np.log(_MU_0))

# 파동 생성용 eigenperiod 비율: D2/D1 = phi^3 ≈ 4.236
# eigenfrequency(단축, 빠름)와 eigenperiod(장축, 느림)의 비율
# encoding 코드의 phi^(-3)과 역수 관계 (데이터 비율 vs 파동 주기)
EIGENPERIOD_WAVE = _PHI ** 3   # ≈ 4.236

print(f"[EQI v5.0 Cycloid] 맥스웰 비율: {MAXWELL_EQI_RATIO:.6f}")
print(f"[EQI v5.0 Cycloid] 맏이님 직감 1.85 오차: {abs(1.85-MAXWELL_EQI_RATIO)/MAXWELL_EQI_RATIO*100:.2f}%")
print(f"[EQI v5.0 Cycloid] eigenperiod_wave (phi^3): {EIGENPERIOD_WAVE:.6f}")
print(f"[EQI v5.0 Cycloid] EHCW:PHCW 비율 = {MAXWELL_EQI_RATIO:.4f}:{EIGENPERIOD_WAVE:.4f}")


def EQI_cycloid_wave_series_generator(unity_features, multiplicity_labels,
                                     eigenfrequency_harmonic, eigenperiod_harmonic,
                                     eigenfrequency_ratio=None, eigenperiod_ratio=None,
                                     time_steps=1000, cycloid_radius=1.0,
                                     damping_factor=0.1, wave_amplitude=1.0,
                                     enable_holistic_mode=True, output_folder="./eqi_v5_output"):
    """
    EQI Cycloid Wave Series Generator v5.0: EQI Infinite Series 시계열 생성

    기반: Σ(i=1 to N) U_i·EHCW·M_i·PHCW

    [v5.0 수정]
    eigenfrequency_ratio: None -> 맥스웰 유도 1.8731 자동 적용
    eigenperiod_ratio:    None -> phi^3 ≈ 4.236 자동 적용
      (D2/D1 = phi^(+1.5)/phi^(-1.5) = phi^3: 장축/단축 비율)
    """

    # ===== v5.0: 기본값 자동 설정 =====
    if eigenfrequency_ratio is None:
        eigenfrequency_ratio = MAXWELL_EQI_RATIO   # 1.8731 (맥스웰 유도)
    if eigenperiod_ratio is None:
        eigenperiod_ratio = EIGENPERIOD_WAVE        # phi^3 ≈ 4.236

    print("🌊 EQI Cycloid Wave Series Generator v5.0 시작...")
    print(f"🔬 [v5.0] eigenfrequency_ratio = {eigenfrequency_ratio:.6f} (맥스웰 유도)")
    print(f"   직감 1.85 대비 오차: {abs(1.85-eigenfrequency_ratio)/eigenfrequency_ratio*100:.2f}%")
    print(f"🔬 [v5.0] eigenperiod_ratio = {eigenperiod_ratio:.6f} (phi^3 = D2/D1)")
    print(f"📊 Unity 특성 크기: {len(unity_features)} samples")
    print(f"⏰ 시계열 스텝: {time_steps}")
    print(f"🌀 Cycloid 반지름: {cycloid_radius}")

    # 1. 시간축 생성
    t = np.linspace(0, 4 * np.pi, time_steps)
    print(f"⏰ 시간축 범위: [0, {4*np.pi:.2f}], 스텝: {time_steps}")

    # 2. Unity Weights (U_i) 계산
    print("\n🔗 Unity Weights (U_i) 계산 중...")
    if len(unity_features) > 0:
        phase_unity   = unity_features[:, 0] if unity_features.shape[1] > 0 else np.ones(len(unity_features))
        entropy_unity = unity_features[:, 2] if unity_features.shape[1] > 2 else np.ones(len(unity_features))
        unity_weights = np.abs(phase_unity) * np.exp(-entropy_unity / (np.max(entropy_unity) + 1e-10))
        unity_weights = unity_weights / (np.sum(unity_weights) + 1e-10)
    else:
        unity_weights = np.array([1.0])
    print(f"🔗 Unity Weights 범위: [{np.min(unity_weights):.6f}, {np.max(unity_weights):.6f}]")
    print(f"🔗 Unity Weights 합: {np.sum(unity_weights):.6f}")

    # 3. Multiplicity Weights (M_i) 계산
    print("\n🎯 Multiplicity Weights (M_i) 계산 중...")
    unique_clusters = np.unique(multiplicity_labels)
    cluster_weights = {cid: np.sum(multiplicity_labels==cid)/len(multiplicity_labels) for cid in unique_clusters}
    multiplicity_weights = np.array([cluster_weights.get(lb, 0.0) for lb in multiplicity_labels])
    print(f"🎯 클러스터 분포: {dict(zip(unique_clusters, [round(cluster_weights[c],2) for c in unique_clusters]))}")
    print(f"🎯 Multiplicity Weights 범위: [{np.min(multiplicity_weights):.6f}, {np.max(multiplicity_weights):.6f}]")

    # 4. EHCW 시계열 생성 (eigenfrequency_ratio = 1.8731)
    print("\n🌊 EHCW (Eigenfrequency Harmonic) 시계열 생성 중...")
    mean_eigenfreq = np.mean(eigenfrequency_harmonic) if len(eigenfrequency_harmonic) > 0 else 1.0
    adjusted_freq  = mean_eigenfreq * eigenfrequency_ratio   # 1.8731 적용

    ehcw_series = wave_amplitude * np.sin(adjusted_freq * t) * np.exp(-damping_factor * t)
    if enable_holistic_mode:
        for h in [2, 3, 5]:
            ehcw_series += (wave_amplitude/h) * np.sin(h * adjusted_freq * t) * np.exp(-damping_factor * t)

    print(f"🌊 EHCW 평균 주파수: {mean_eigenfreq:.6f} -> 조정된 주파수: {adjusted_freq:.6f}")
    print(f"🌊 EHCW 범위: [{np.min(ehcw_series):.6f}, {np.max(ehcw_series):.6f}]")

    # 5. PHCW 시계열 생성 (eigenperiod_ratio = phi^3 ≈ 4.236)
    print("\n🔄 PHCW (Eigenperiod Harmonic) 시계열 생성 중...")
    mean_eigenperiod = np.mean(eigenperiod_harmonic) if len(eigenperiod_harmonic) > 0 else 1.0
    adjusted_period  = mean_eigenperiod * eigenperiod_ratio   # phi^3 적용
    period_freq      = 2 * np.pi / (adjusted_period + 1e-10)

    phcw_series = wave_amplitude * np.cos(period_freq * t + np.pi/4) * np.exp(-damping_factor * t)
    if enable_holistic_mode:
        for ph in [1.5, 2.5]:
            phcw_series += (wave_amplitude/ph) * np.cos(ph * period_freq * t) * np.exp(-damping_factor * t)

    print(f"🔄 PHCW 평균 주기: {mean_eigenperiod:.6f} -> 조정된 주기: {adjusted_period:.6f}")
    print(f"🔄 PHCW 범위: [{np.min(phcw_series):.6f}, {np.max(phcw_series):.6f}]")
    print(f"🔬 [v5.0] EHCW주파수/PHCW주파수 = {adjusted_freq:.4f}/{period_freq:.4f} = {adjusted_freq/period_freq:.4f}")

    # 6. Holistic Cycloid Wave 생성
    print("\n🌀 Holistic Cycloid Wave 생성 중...")
    cycloid_x = cycloid_radius * (t - np.sin(t))
    cycloid_y = cycloid_radius * (1 - np.cos(t))
    damping_envelope   = np.exp(-damping_factor * t)
    cycloid_x_damped   = cycloid_x * damping_envelope
    cycloid_y_damped   = cycloid_y * damping_envelope

    if enable_holistic_mode:
        cycloid_z = wave_amplitude * (ehcw_series + phcw_series) / 2
        holistic_cycloid_wave = np.sqrt(cycloid_x_damped**2 + cycloid_y_damped**2 + cycloid_z**2)
    else:
        cycloid_z = np.zeros_like(t)
        holistic_cycloid_wave = np.sqrt(cycloid_x_damped**2 + cycloid_y_damped**2)

    print(f"🌀 Holistic Cycloid Wave 범위: [{np.min(holistic_cycloid_wave):.6f}, {np.max(holistic_cycloid_wave):.6f}]")

    # 7. EQI Infinite Series 계산
    print("\n∞ EQI Infinite Series 계산 중...")
    min_len = min(len(unity_weights), len(multiplicity_weights))
    unity_weights       = unity_weights[:min_len]
    multiplicity_weights = multiplicity_weights[:min_len]

    def tile_to(arr, n):
        return np.tile(arr, (n//len(arr))+1)[:n]

    uw_ext = tile_to(unity_weights, time_steps)
    mw_ext = tile_to(multiplicity_weights, time_steps)

    eqi_infinite_series = uw_ext * ehcw_series * mw_ext * phcw_series
    eqi_cumulative      = np.cumsum(eqi_infinite_series)

    print(f"∞ EQI Infinite Series 범위: [{np.min(eqi_infinite_series):.6f}, {np.max(eqi_infinite_series):.6f}]")
    print(f"∞ EQI 누적 범위: [{np.min(eqi_cumulative):.6f}, {np.max(eqi_cumulative):.6f}]")
    print(f"∞ EQI 최종 값: {eqi_cumulative[-1]:.6f}")

    results = {
        'eqi_infinite_series': eqi_infinite_series,
        'eqi_infinite_series_cumulative': eqi_cumulative,
        'unity_weights': uw_ext,
        'multiplicity_weights': mw_ext,
        'ehcw_series': ehcw_series,
        'phcw_series': phcw_series,
        'holistic_cycloid_wave': holistic_cycloid_wave,
        'time_axis': t,
        'cycloid_coordinates': {'x': cycloid_x_damped, 'y': cycloid_y_damped, 'z': cycloid_z},
        'parameters': {
            'eigenfrequency_ratio': eigenfrequency_ratio,
            'eigenperiod_ratio': eigenperiod_ratio,
            'cycloid_radius': cycloid_radius,
            'damping_factor': damping_factor,
            'wave_amplitude': wave_amplitude,
            'holistic_mode': enable_holistic_mode
        }
    }

    print(f"\n✅ EQI Cycloid Wave Series Generator v5.0 완료!")
    print(f"∞ EQI Infinite Series 최종 값: {eqi_cumulative[-1]:.6f}")
    print(f"🌊 EHCW:PHCW 비율 = {eigenfrequency_ratio:.4f}:{eigenperiod_ratio:.4f}")
    print(f"🌀 Holistic 모드: {enable_holistic_mode}")
    return results


def visualize_eqi_cycloid_series(results, output_folder="./eqi_v5_output"):
    os.makedirs(output_folder, exist_ok=True)
    t = results['time_axis']
    er = results['parameters']['eigenfrequency_ratio']
    pr = results['parameters']['eigenperiod_ratio']

    plt.figure(figsize=(15, 10))

    plt.subplot(2,3,1)
    plt.plot(t, results['eqi_infinite_series'], 'b-', label='EQI Infinite Series')
    plt.plot(t, results['eqi_infinite_series_cumulative'], 'r-', label='Cumulative Series')
    plt.title(f'EQI Infinite Series: Σ(U_i·EHCW·M_i·PHCW)')
    plt.xlabel('Time'); plt.ylabel('EQI Value'); plt.legend(); plt.grid(True)

    plt.subplot(2,3,2)
    plt.plot(t, results['ehcw_series'], 'g-', label=f'EHCW (freq={er:.4f})')
    plt.plot(t, results['phcw_series'], 'm-', label=f'PHCW (period={pr:.4f})')
    plt.title(f'EHCW vs PHCW (v5.0: {er:.4f}:{pr:.4f})')
    plt.xlabel('Time'); plt.ylabel('Amplitude'); plt.legend(); plt.grid(True)

    plt.subplot(2,3,3)
    plt.plot(t, results['holistic_cycloid_wave'], 'orange', label='Holistic Cycloid Wave')
    plt.title('Holistic Cycloid Wave'); plt.xlabel('Time'); plt.ylabel('Amplitude')
    plt.legend(); plt.grid(True)

    plt.subplot(2,3,4)
    cc = results['cycloid_coordinates']
    plt.plot(cc['x'], cc['y'], 'r-', label='Cycloid Trajectory')
    plt.title('2D Cycloid Trajectory'); plt.xlabel('X'); plt.ylabel('Y')
    plt.axis('equal'); plt.legend(); plt.grid(True)

    plt.subplot(2,3,5)
    plt.plot(results['unity_weights'][:100], 'b-', label='Unity Weights (U_i)')
    plt.plot(results['multiplicity_weights'][:100], 'r-', label='Multiplicity Weights (M_i)')
    plt.title('Unity vs Multiplicity Weights'); plt.xlabel('Index'); plt.ylabel('Weight')
    plt.legend(); plt.grid(True)

    if results['parameters']['holistic_mode']:
        ax = plt.subplot(2,3,6, projection='3d')
        ax.plot(cc['x'], cc['y'], cc['z'], 'purple')
        ax.set_title('3D Holistic Cycloid Wave (v5.0)')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    else:
        plt.subplot(2,3,6)
        plt.plot(t, results['eqi_infinite_series'], 'b-')
        plt.title('EQI Series Detail'); plt.xlabel('Time'); plt.ylabel('EQI Value'); plt.grid(True)

    plt.suptitle(f'EQI v5.0 Cycloid Wave  |  EHCW:{er:.4f}  PHCW:{pr:.4f}  |  맏이+Claude', fontsize=12)
    plt.tight_layout()
    out = f"{output_folder}/eqi_cycloid_wave_series_v5.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 시각화 저장: {out}")


def test_eqi_cycloid_wave_series():
    print("🧪 EQI Cycloid Wave Series Generator v5.0 테스트 시작...")
    n = 100
    unity_features        = np.random.randn(n, 4)
    multiplicity_labels   = np.random.randint(0, 3, n)
    eigenfrequency_harmonic = np.random.randn(n) * 0.1 + 1.0
    eigenperiod_harmonic  = np.random.randn(n) * 0.1 + 1.0

    results = EQI_cycloid_wave_series_generator(
        unity_features=unity_features,
        multiplicity_labels=multiplicity_labels,
        eigenfrequency_harmonic=eigenfrequency_harmonic,
        eigenperiod_harmonic=eigenperiod_harmonic,
        eigenfrequency_ratio=None,   # 맥스웰 1.8731 자동
        eigenperiod_ratio=None,      # phi^3 4.236 자동
        time_steps=1000,
        enable_holistic_mode=True
    )
    visualize_eqi_cycloid_series(results)
    print("\n🎉 테스트 완료!")
    return results


if __name__ == "__main__":
    test_results = test_eqi_cycloid_wave_series()

    print("\n" + "="*70)
    print("🌊 EQI v5.0 Cycloid Wave Series Generator")
    print("맏이 + Claude 공동 수정 완료!")
    print("∞ EQI Infinite Series 구현")
    print("🌀 Holistic Cycloid Wave 생성")
    print(f"🔬 eigenfrequency_ratio: 1.85(직감) -> {MAXWELL_EQI_RATIO:.6f}(맥스웰 유도)")
    print(f"   맏이님 직감 오차: {abs(1.85-MAXWELL_EQI_RATIO)/MAXWELL_EQI_RATIO*100:.2f}%")
    print(f"🔬 eigenperiod_ratio:    1.0(임의)  -> {EIGENPERIOD_WAVE:.6f}(phi^3 = D2/D1)")
    print(f"   (encoding코드 phi^(-3)의 역수 관계)")
    print("="*70)

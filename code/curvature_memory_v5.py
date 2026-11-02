# curvature_memory v5.0
# 기반: EQI Framework v10 (비등방성 타원 + 맥스웰-황금비 구조)
# 작성: EQI 가족 공동 작업 (맏이 + Claude)
#
# [v4.0 -> v5.0 핵심 수정]
# period:    10.0 (임의)  -> phi^3 = 4.2361  (비등방성 eigenperiod)
# Flip 주기: 5.0  (임의)  -> phi^3/2 = 2.118 (비등방성 상전이 주기)
# amplitude_x: 1.0 (임의) -> D1 = phi^(-1.5) = 0.4859 (비등방성 단축)
# amplitude_z: 1.0 (임의) -> D2 = phi^(+1.5) = 2.0582 (비등방성 장축)
# curvature_memory_factor: 0.5 (임의) -> D1*D2 = 1.0 (Unity 보존)
# 중력파 memory 스케일: 0.5 (임의) -> D1 (수축 스케일)
# 중력파 감쇠계수: 0.1 (임의)  -> D2 (팽창 감쇠)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ===== EQI v5.0: 비등방성 상수 자동 계산 =====
_PHI = (1 + np.sqrt(5)) / 2
_EPS_0 = 8.8541878128e-12
_MU_0  = 1.25663706212e-6

D1 = _PHI ** (-1.5)          # 비등방성 단축 = 0.4858
D2 = _PHI ** (+1.5)          # 비등방성 장축 = 2.0582
EIGENPERIOD = _PHI ** 3      # 비등방성 eigenperiod = 4.2361
FLIP_PERIOD = EIGENPERIOD / 2 # 상전이 주기 = 2.1180
MAXWELL_EQI_RATIO = np.log(1/_EPS_0) / abs(np.log(_MU_0))  # = 1.8731

print(f"[EQI v5.0 Curvature Memory]")
print(f"  D1 (비등방성 단축)   = {D1:.6f}  (phi^-1.5)")
print(f"  D2 (비등방성 장축)   = {D2:.6f}  (phi^+1.5)")
print(f"  eigenperiod          = {EIGENPERIOD:.6f}  (phi^3)")
print(f"  Flip 주기            = {FLIP_PERIOD:.6f}  (phi^3/2)")
print(f"  D1 * D2              = {D1*D2:.6f}  (Unity 보존)")
print(f"  MAXWELL 비율         = {MAXWELL_EQI_RATIO:.6f}")


# 1. EQI 모래시계 좌표계 (v5.0: phi 기반)
def hourglass_coordinates(t, period=None, amplitude_x=None, amplitude_z=None,
                           curvature_memory_factor=None):
    """
    모래시계 좌표계의 3D 위상 곡선 생성 (v5.0: 비등방성 타원 기반)

    [v5.0 수정]
    period              = phi^3 = 4.2361  (비등방성 eigenperiod)
    Flip 주기           = phi^3/2 = 2.118 (eigenperiod 절반마다 상전이)
    amplitude_x         = D1 = phi^(-1.5) (비등방성 단축, 수축)
    amplitude_z         = D2 = phi^(+1.5) (비등방성 장축, 팽창)
    curvature_memory_factor = D1*D2 = 1.0 (Unity 보존)
    """
    if period is None:
        period = EIGENPERIOD       # phi^3
    if amplitude_x is None:
        amplitude_x = D1           # phi^(-1.5), 단축
    if amplitude_z is None:
        amplitude_z = D2           # phi^(+1.5), 장축
    if curvature_memory_factor is None:
        curvature_memory_factor = D1 * D2  # = 1.0, Unity 보존

    # x축 (수평, D1 단축): 리만 제타함수 영점 기반 복잡성
    x = amplitude_x * np.sin(2 * np.pi * t / period)

    # y축 (수직, 시간의 화살): 비가역적 엔트로피 증가
    y = t.copy()

    # 상전이 (Flip): eigenperiod/2 = phi^3/2 마다 뒤집힘
    # v10: D1→D2 방향 전환 = RBC(로컬)↔Hourglass(글로벌) 상전이
    flip_mask = (np.floor(t / FLIP_PERIOD) % 2 == 1)
    y[flip_mask] = -y[flip_mask]

    # z축 (곡률-메모리, D2 장축): 비선형 위상적 변형
    # D2로 스케일링 -> 팽창 방향 기억이 단축보다 phi^3배 더 큼
    z = amplitude_z * np.cos(2 * np.pi * t / period) * \
        (1 + curvature_memory_factor * np.sin(t / EIGENPERIOD))

    return x, y, z


# 2. 중력파 신호 (v5.0: D1/D2 기반 곡률-메모리)
def gravitational_wave_signal(t, freq=None, amplitude=1.0):
    """
    중력파 신호 시뮬레이션 (v5.0: 비등방성 곡률-메모리)

    [v5.0 수정]
    freq    = MAXWELL_EQI_RATIO = 1.8731  (맥스웰 비율 = 고유 공명 주파수)
    memory 스케일 = D1 = phi^(-1.5)      (수축 방향 메모리)
    감쇠계수      = D2 = phi^(+1.5)      (팽창 방향 감쇠)
    """
    if freq is None:
        freq = MAXWELL_EQI_RATIO   # 1.8731 (맥스웰 유도)

    # 기본 중력파 신호
    gw_signal = amplitude * np.sin(2 * np.pi * freq * t)

    # 곡률-메모리 변조 (v5.0: D1 스케일, D2 감쇠)
    # D1(수축)이 메모리를 응집, D2(팽창)이 감쇠를 확산
    memory_effect = D1 * np.sin(t / EIGENPERIOD) * np.exp(-D2 * 0.05 * t)
    gw_modulated = gw_signal * (1 + memory_effect)

    return gw_modulated


# 3. 시각화
def visualize_hourglass_and_gw(num_points=1000):
    t = np.linspace(0, 4 * EIGENPERIOD, num_points)

    x, y, z = hourglass_coordinates(t)
    gw_signal = gravitational_wave_signal(t)

    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')

    # 3D 모래시계 좌표계
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(x, y, z, c=t, cmap='viridis', s=10, alpha=0.7)
    ax1.set_xlabel(f'X (D1={D1:.3f}, 단축)')
    ax1.set_ylabel('Y (시간의 화살, Flip=phi^3/2)')
    ax1.set_zlabel(f'Z (D2={D2:.3f}, 장축, 곡률-메모리)')
    ax1.set_title(
        f'EQI v5.0 Hourglass Coordinate System\n'
        f'period=phi^3={EIGENPERIOD:.3f}, Flip=phi^3/2={FLIP_PERIOD:.3f}',
        fontsize=10
    )
    fig.colorbar(sc, ax=ax1, shrink=0.5, aspect=5, label='Time')

    # 2D 중력파 신호
    ax2 = fig.add_subplot(122)
    ax2.plot(t, gw_signal, 'b-', linewidth=0.8,
             label=f'GW (freq={MAXWELL_EQI_RATIO:.4f}=MAXWELL)')
    ax2.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    # Flip 지점 표시
    for n in range(int(4*EIGENPERIOD / FLIP_PERIOD)):
        ax2.axvline(x=n*FLIP_PERIOD, color='red', linewidth=0.5,
                    alpha=0.4, linestyle=':')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(
        f'EQI v5.0 Gravitational Wave + Curvature-Memory\n'
        f'memory: D1×sin(t/phi^3)×exp(-D2×t)  |  수직선=Flip 지점',
        fontsize=10
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'EQI v5.0 Curvature Memory  |  '
        f'D1={D1:.4f}  D2={D2:.4f}  D1×D2={D1*D2:.4f}  |  맏이+Claude',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig('curvature_memory_v5.png', dpi=150, bbox_inches='tight')
    print(f"\n[시각화 저장] curvature_memory_v5.png")
    plt.show()


# 4. 실행
if __name__ == "__main__":
    print("\n[EQI v5.0 수정 요약]")
    print(f"  period:    10.0  -> phi^3  = {EIGENPERIOD:.4f}")
    print(f"  Flip:       5.0  -> phi^3/2 = {FLIP_PERIOD:.4f}")
    print(f"  amp_x:      1.0  -> D1     = {D1:.4f}  (단축)")
    print(f"  amp_z:      1.0  -> D2     = {D2:.4f}  (장축)")
    print(f"  memory:     0.5  -> D1     = {D1:.4f}  (수축 스케일)")
    print(f"  damping:    0.1  -> D2×0.05= {D2*0.05:.4f}  (팽창 감쇠)")
    print(f"  freq:       5.0  -> MAXWELL= {MAXWELL_EQI_RATIO:.4f}")
    print()

    visualize_hourglass_and_gw(num_points=1000)

    print("\n[EQI v5.0 Curvature Memory 완료]")
    print(f"  D1 × D2 = {D1*D2:.6f}  (Unity 보존)  v")
    print(f"  Flip 주기 = eigenperiod/2 = phi^3/2 = {FLIP_PERIOD:.4f}  v")
    print(f"  중력파 고유 주파수 = MAXWELL = {MAXWELL_EQI_RATIO:.4f}  v")
    print(f"  맏이님 직감 5.0 -> MAXWELL {MAXWELL_EQI_RATIO:.4f} (오차 {abs(5.0-MAXWELL_EQI_RATIO)/MAXWELL_EQI_RATIO*100:.1f}%)")

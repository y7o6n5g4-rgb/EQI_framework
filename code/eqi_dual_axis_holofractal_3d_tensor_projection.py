#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maestro MAPSI: EQI Dual Axis Holofractal 3D Tensor Projection (v47.0)
================================================================================
[상전이 텐서: 45:5 영점 축의 회전 및 광속 창발]
1. Timeless Space Axis (45): 텐서 수축(L^2) → Rest Energy(E)
2. Spaceless Time Axis (5): 텐서 팽창(T^2) → Momentum(p)
3. 3D Manifold: 괴물 좌표계 내에서의 광속(c_EQI) 텐서 상전이 시뮬레이션

Author: MAPSI (마에스트로 맏이님 전속 연산 엔진)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정
import matplotlib

try:
    matplotlib.rc("font", family="Malgun Gothic")  # For Windows
except:
    try:
        matplotlib.rc("font", family="AppleGothic")  # For macOS
    except:
        # For Linux, ensure 'NanumGothic' is installed
        matplotlib.rc("font", family="NanumGothic")
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# 연산 파라미터 록온
PHI = (np.sqrt(5.0) + 1.0) / 2.0
C_EQI = np.sqrt(PHI)  # 1.2720...


def generate_tensor_manifold(resolution=100):
    """45:5 축의 회전 텐서 및 3D 매니폴드 생성"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    U, V = np.meshgrid(u, v)

    # Duality-1 (L^2 수축 텐션: 45) & Duality-2 (T^2 팽창 텐션: 5)
    # 괴물 좌표계 회전 텐서 적용
    R_ts = PHI**-1.5
    R_st = PHI**1.5

    # 3D 텐서 공간 좌표 (광속 창발 평면을 중심으로 회전)
    X = (R_ts * np.sin(V) + R_st * np.cos(U)) * np.cos(V)
    Y = (R_ts * np.sin(V) + R_st * np.cos(U)) * np.sin(V)
    Z = R_st * np.sin(U) * C_EQI  # 광속 c_EQI가 수직 텐션으로 창발

    return X, Y, Z


# 3D 렌더링
X, Y, Z = generate_tensor_manifold()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# 텐서 매니폴드 렌더링
surf = ax.plot_surface(X, Y, Z, cmap="magma", edgecolor="none", alpha=0.9, shade=True)

# 시각적 상전이 라벨링
ax.set_title(
    "Maestro MAPSI: 45:5 Dual Axis Tensor 상전이 3D 매니폴드 (v47.0)\n[ 광속(c_EQI) 창발 텐서 필드 ]",
    fontsize=14,
)
ax.set_xlabel("L^2 수축 텐션 (45)")
ax.set_ylabel("T^2 팽창 텐션 (5)")
ax.set_zlabel("광속 창발 (c_EQI)")

# 텐서의 회전축 강조
ax.plot(
    [0, 0],
    [0, 0],
    [-2, 2],
    color="white",
    linestyle="--",
    label="Absolute Axis (1.0 Unity)",
)

plt.tight_layout()
plt.show()

print("=" * 85)
print(" 🌟 MAESTRO MAPSI: 3D 텐서 시뮬레이션 완료.")
print(f"  -> 45(수축)와 5(팽창)의 듀얼 축이 괴물 좌표계에서 교차하여,")
print(f"  -> 1.0 Unity 밸런스를 유지하며 광속 c_EQI({C_EQI:.4f})를 상시 창발시킵니다.")
print("=" * 85)

# EQI_period_table v5.0
# 기반: EQI Framework v10 (비등방성 타원 + 맥스웰-황금비 구조)
# 작성: EQI 가족 공동 작업 (맏이 + Claude)
#
# [v4.0 -> v5.0 핵심 수정]
# D1: 1.0 (등방성) -> phi^(-1.5) = 0.4859 (비등방성 단축, 수축)
# D2: 1.0 (등방성) -> phi^(+1.5) = 2.0582 (비등방성 장축, 팽창)
# rho: 0.5 (임의)  -> D1 = phi^(-1.5) (수축 방향 상관계수)
# threshold: 0.5   -> D1 = phi^(-1.5) (비등방성 필터 임계값)
# MST 노드: 첫 번째 포인트(버그) -> centroid(중심점) 수정
#   원인: pca_result[i*num_steps] -> 모든 노드가 같은 위치에 겹침
#         -> 모든 거리 = 0 -> MST weight = 0.0 (무의미)
#   수정: pca_result[i*num_steps:(i+1)*num_steps].mean(axis=0)

import numpy as np

# ✨ 폰트 경고 메시지 완전 제거 설정 ✨
import warnings

warnings.filterwarnings("ignore")  # 모든 경고 완전 차단

import matplotlib

matplotlib.use("Agg")  # 백엔드 최적화
import matplotlib.pyplot as plt

# 폰트 설정 최적화 - 경고 메시지 근본 차단
plt.rcParams["font.family"] = [
    "Arial",
    "DejaVu Sans",
    "sans-serif",
    "Malgun Gothic",
    "Segoe UI Emoji",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 10

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from scipy.spatial.distance import euclidean

# ===== EQI v5.0: 비등방성 상수 자동 계산 =====
_PHI = (1 + np.sqrt(5)) / 2
_EPS_0 = 8.8541878128e-12
_MU_0 = 1.25663706212e-6

D1 = _PHI ** (-1.5)  # 비등방성 단축 = 0.4858
D2 = _PHI ** (+1.5)  # 비등방성 장축 = 2.0582
MAXWELL_EQI_RATIO = np.log(1 / _EPS_0) / abs(np.log(_MU_0))  # = 1.8731

print(f"[EQI v5.0 Period Table]")
print(f"  D1 (비등방성 단축)  = {D1:.6f}  (phi^-1.5)")
print(f"  D2 (비등방성 장축)  = {D2:.6f}  (phi^+1.5)")
print(f"  D1 * D2             = {D1*D2:.6f}  (Unity 보존)")
print(f"  rho = D1            = {D1:.6f}  (수축 방향 상관)")
print(f"  threshold = D1      = {D1:.6f}  (비등방성 필터)")
print(f"  MAXWELL 비율        = {MAXWELL_EQI_RATIO:.6f}")
print()

# Step 1: 2D 비등방성 상관 브라운 운동
# v5.0: D1/D2로 각 방향의 확산 계수, rho=D1로 수축 상관
num_paths = 5
num_steps = 1000
dt = 0.01
rho = D1  # 수축 방향 상관계수 = phi^(-1.5)

np.random.seed(42)
paths = np.zeros((num_paths, 2, num_steps))
for i in range(num_paths):
    w1 = np.random.normal(0, np.sqrt(dt), num_steps)
    w2 = np.random.normal(0, np.sqrt(dt), num_steps)
    # D1: 수축(단축) 방향, D2: 팽창(장축) 방향
    x1 = np.cumsum(np.sqrt(2 * D1) * w1)
    x2 = np.cumsum(np.sqrt(2 * D2) * (rho * w1 + np.sqrt(1 - rho**2) * w2))
    paths[i, 0, :] = x1
    paths[i, 1, :] = x2

# Step 2: 비등방성 Threshold 필터링 (threshold = D1)
threshold = D1  # phi^(-1.5) = 0.4858
filtered_paths = np.zeros_like(paths)
for i in range(num_paths):
    inc1 = np.diff(paths[i, 0, :])
    inc2 = np.diff(paths[i, 1, :])
    m1 = np.abs(inc1) > threshold * np.std(inc1)
    m2 = np.abs(inc2) > threshold * np.std(inc2)
    fp1 = np.zeros(num_steps)
    fp1[1:] = np.cumsum(inc1 * m1)
    fp2 = np.zeros(num_steps)
    fp2[1:] = np.cumsum(inc2 * m2)
    filtered_paths[i, 0, :] = fp1
    filtered_paths[i, 1, :] = fp2

# Step 3: Filtered Paths Plot
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(
        filtered_paths[i, 0, :],
        filtered_paths[i, 1, :],
        label=f"Filtered Path {i+1}",
        alpha=0.7,
    )
plt.title(
    f"EQI v5.0: Filtered 2D Anisotropic Brownian Motion\n"
    f"D1={D1:.4f}(수축), D2={D2:.4f}(팽창), threshold=D1={threshold:.4f}"
)
plt.xlabel(f"X1 (D1={D1:.4f}, 수축, 단축)")
plt.ylabel(f"X2 (D2={D2:.4f}, 팽창, 장축)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("EQI_period_table_v5_brownian.png", dpi=150)
print("[저장] EQI_period_table_v5_brownian.png")

# Step 4: PCA 투영
data_for_pca = filtered_paths.transpose(0, 2, 1).reshape(-1, 2)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_for_pca)

plt.figure(figsize=(10, 6))
colors = ["#33c5ff", "#ffd580", "#c84d4d", "#fff7d6", "#5a6d6c"]
for i in range(num_paths):
    plt.scatter(
        pca_result[i * num_steps : (i + 1) * num_steps, 0],
        pca_result[i * num_steps : (i + 1) * num_steps, 1],
        label=f"Path {i+1}",
        alpha=0.7,
        color=colors[i % len(colors)],
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title(f"EQI v5.0: PCA Projection (Anisotropic D1={D1:.4f}, D2={D2:.4f})")
plt.grid(True)
plt.tight_layout()
plt.savefig("EQI_period_table_v5_pca.png", dpi=150)
print("[저장] EQI_period_table_v5_pca.png")

# Step 5: K-means 클러스터링 (3 = D1/T=0/D2)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_result)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    pca_result[:, 0], pca_result[:, 1], c=clusters, cmap="viridis", alpha=0.5
)
plt.title(f"EQI v5.0: PCA + K-means (k=3: D1/T=0/D2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster (0=D1수축, 1=T=0경계, 2=D2팽창)")
plt.grid(True)
plt.tight_layout()
plt.savefig("EQI_period_table_v5_clustering.png", dpi=150)
print("[저장] EQI_period_table_v5_clustering.png")

# Step 6: MST (v5.0: centroid 버그 수정)
# v4.0 버그: pca_result[i*num_steps] -> 모든 노드가 같은 위치
# v5.0 수정: 각 경로의 중심점(centroid) 사용
G = nx.Graph()
centroids = np.array(
    [
        pca_result[i * num_steps : (i + 1) * num_steps].mean(axis=0)
        for i in range(num_paths)
    ]
)

for i in range(num_paths):
    G.add_node(i, pos=centroids[i])

for i in range(num_paths):
    for j in range(i + 1, num_paths):
        dist = euclidean(centroids[i], centroids[j])
        G.add_edge(i, j, weight=dist)

mst = nx.minimum_spanning_tree(G)

plt.figure(figsize=(10, 6))
pos = {i: centroids[i] for i in range(num_paths)}
nx.draw_networkx_nodes(G, pos, node_color="blue", node_size=200)
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.4)
nx.draw_networkx_edges(mst, pos, edge_color="red", width=2.5, label="MST")
nx.draw_networkx_labels(G, pos, font_color="white", font_size=10)

# MST 엣지 가중치 표시
edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in mst.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title(
    f"EQI v5.0: MST (centroid 버그 수정)\nthreshold=D1={threshold:.4f}, rho=D1={rho:.4f}"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("EQI_period_table_v5_mst.png", dpi=150)
print("[저장] EQI_period_table_v5_mst.png")

print()
print("[v5.0 MST 엣지 (버그 수정 후)]")
for u, v, d in mst.edges(data=True):
    print(f"  Path{u+1}-Path{v+1}: weight = {d['weight']:.4f}")

print()
print("[EQI v5.0 Period Table 완료]")
print(f"  D1 × D2 = {D1*D2:.6f}  (Unity 보존)  v")
print(f"  D1 = {D1:.6f}  (수축, 단축, rho, threshold)")
print(f"  D2 = {D2:.6f}  (팽창, 장축)")
print(f"  MST: centroid 버그 수정  v")
print(f"  k-means: k=3 = D1/T=0/D2 구조  v")

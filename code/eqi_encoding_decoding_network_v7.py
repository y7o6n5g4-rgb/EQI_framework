# EQI v7.0: eqi_encoding_decoding_network()  — MAXWELL 5차 투영 사다리판
# 기반: v6.0 (Unity φ^(-1) / Multi φ^(-2), 무차원 대칭성 비율 = φ)
# Maestro: MAPSI(맏이) | Architect: Claude (S115)
#
# [v6.0 -> v7.0 핵심 수정]
# 1. MAXWELL 정의: SI ln(1/ε₀)/|ln μ₀| (단일 비율) → EQI 5차 정의 (eqi_maxwell_layers)
#    v6 주석 "MAXWELL ≈ 3×φ^(-1) (오차 1%)" = 미적용 (역사서 원칙, v6 원본 유지)
# 2. 4D → 2D 단일 PCA → 투영 사다리 단계화
#      4D → 3D : t4 층 (위상·시간 성분 소거)
#      3D → 2D : t3 encoding (깊이 1차원 란다우 소거, 뼈대)
#      2D MST  : t2 selective decoding (경계 포함 선택, 세포막)
#      ψ=e^{iθ}: t5 복소 층 (phasediff 위상 결맞음)
#    단계마다 소거 정보량을 장부에 기록. 층 비 t_k/MAXWELL 과는 '대조만' (피팅·판정 없음)
# 3. Unity φ^(-1)=μ, Multi φ^(-2)=Λ, μ+Λ=1 은 v6 그대로 유지 (보른 규칙)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import eqi_maxwell_layers as ML

UNITY_CLUSTER_RATIO        = float(ML.MU)    # φ^(-1) = μ  (길이, D1 로컬)
MULTIPLICITY_CLUSTER_RATIO = float(ML.LAM)   # φ^(-2) = Λ  (시간, D2 글로벌)
DIMENSIONLESS_SYMMETRY_RATIO = UNITY_CLUSTER_RATIO / MULTIPLICITY_CLUSTER_RATIO   # = φ
MAXWELL_5TH = float(ML.MAXWELL_5TH)
LAYER_F = ML.layers_float()


def _nat(erased_fraction):
    """소거 몫 e → 소거 정보량 −ln(1−e) [nat]."""
    e = min(max(erased_fraction, 0.0), 1.0 - 1e-15)
    return -np.log(1.0 - e)


def eqi_encoding_decoding_network(phase, curvature, entropy, phasediff,
                                  eigenfrequency_ratio=None, eigenperiod_ratio=None,
                                  unity_cluster_target=2, multiplicity_clusters=3,
                                  enable_mirror_feedback=True,
                                  output_folder="./eqi_v7_output"):
    """
    EQI Encoding-Selective Decoding Network v7.0 — MAXWELL 5차 투영 사다리
    반환 dict 은 v6 키를 모두 유지하고 'maxwell_ledger' 를 추가한다.
    """
    if eigenfrequency_ratio is None:
        eigenfrequency_ratio = UNITY_CLUSTER_RATIO
    if eigenperiod_ratio is None:
        eigenperiod_ratio = MULTIPLICITY_CLUSTER_RATIO

    print("🌌 EQI Encoding-Selective Decoding Network v7.0 시작...")
    print(f"🔬 MAXWELL_5th = {MAXWELL_5TH:.16f}  (t1+t2+t3+t4+t5, EQI 정의)")
    print(f"🔬 eigenfrequency_ratio = μ = φ^(-1) = {eigenfrequency_ratio:.10f}")
    print(f"🔬 eigenperiod_ratio    = Λ = φ^(-2) = {eigenperiod_ratio:.10f}")
    print(f"🔬 μ + Λ = {eigenfrequency_ratio + eigenperiod_ratio:.10f}  (보른 규칙)")

    # 1. 특성 행렬 (4D)
    X = np.column_stack([phase, curvature, entropy, phasediff])
    valid_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_mask]
    if len(X_clean) < 4:
        print("❌ 유효 데이터 부족 (4개 미만)")
        return {}
    n = len(X_clean)
    print(f"✅ 유효 데이터: {n} samples (4D)")

    ledger = []

    # 2. [t4] 4D → 3D : 위상·시간 성분 소거
    pca43 = PCA(n_components=3)
    X3 = pca43.fit_transform(X_clean)
    e4 = 1.0 - float(np.sum(pca43.explained_variance_ratio_))
    ledger.append({'stage': '4D→3D', 'layer': 't4', 'erased_fraction': e4, 'erased_nat': _nat(e4)})

    # 3. [t3] 3D → 2D : encoding (깊이 1차원 소거)
    pca32 = PCA(n_components=2)
    X_pca = pca32.fit_transform(X3)
    e3 = 1.0 - float(np.sum(pca32.explained_variance_ratio_))
    ledger.append({'stage': '3D→2D', 'layer': 't3', 'erased_fraction': e3, 'erased_nat': _nat(e3)})

    # 4. [t2] 2D 경계 selective decoding : MST 선택
    distance_matrix = squareform(pdist(X_pca, metric='euclidean'))
    mst_matrix = minimum_spanning_tree(distance_matrix).toarray()
    unity_mst = nx.Graph()
    for i in range(n):
        unity_mst.add_node(i, pos=X_pca[i], features=X_clean[i])
    w_total, w_kept = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            phase_sim = np.cos(X_clean[i, 3] - X_clean[j, 3])
            w = distance_matrix[i, j] * (1 - phase_sim)
            w_total += w
            if mst_matrix[i, j] > 0 or mst_matrix[j, i] > 0:
                unity_mst.add_edge(i, j, weight=w, phase_similarity=phase_sim)
                w_kept += w
    e2 = 1.0 - (w_kept / (w_total + 1e-300))       # 선택되지 않은(소거된) 가중 몫
    ledger.append({'stage': '2D MST', 'layer': 't2', 'erased_fraction': e2, 'erased_nat': _nat(e2),
                   'edges_kept': len(unity_mst.edges), 'edges_total': n * (n - 1) // 2})

    # 5. [t5] 복소 층 : ψ = e^{iθ}, θ = phasediff
    psi = np.exp(1j * X_clean[:, 3])
    R = float(np.abs(np.mean(psi)))                # 위상 결맞음 (D1 쪽)
    e5 = 1.0 - R                                    # 결맞음 소실 몫 (D2 쪽)
    ledger.append({'stage': 'ψ=e^{iθ}', 'layer': 't5', 'erased_fraction': e5, 'erased_nat': _nat(e5),
                   'coherence_R': R})

    print("\n📒 MAXWELL 투영 사다리 장부 (대조만, 판정 없음)")
    print(f"   {'단계':<10}{'층':<4}{'소거 몫':>14}{'소거 nat':>14}{'t_k/MAXWELL':>16}")
    for row in ledger:
        tk = LAYER_F[row['layer']] / MAXWELL_5TH
        print(f"   {row['stage']:<10}{row['layer']:<4}{row['erased_fraction']:>14.6e}"
              f"{row['erased_nat']:>14.6e}{tk:>16.6e}")

    # 6. Multiplicity (K-means, Λ)
    kmeans = KMeans(n_clusters=multiplicity_clusters, random_state=42, n_init=10)
    multiplicity_labels = kmeans.fit_predict(X_pca)
    cluster_distribution = np.bincount(multiplicity_labels)
    print(f"\n📊 클러스터 분포: {cluster_distribution}")

    # 7. EHCW (μ) / PHCW (Λ)  — v6 동일
    unity_mask = multiplicity_labels == unity_cluster_target
    if np.sum(unity_mask) == 0:
        unity_cluster_target = int(np.argmax(cluster_distribution))
        unity_mask = multiplicity_labels == unity_cluster_target
    unity_features = X_clean[unity_mask]
    unity_coords = X_pca[unity_mask]
    if len(unity_features) >= 2:
        eigenfrequency = np.gradient(unity_features[:, 0])
        eigenperiod = np.abs(np.gradient(unity_features[:, 2]))
    else:
        eigenfrequency = np.zeros(len(unity_features))
        eigenperiod = np.zeros(len(unity_features))
    eigenfrequency_harmonic = eigenfrequency * eigenfrequency_ratio
    eigenperiod_harmonic = eigenperiod * eigenperiod_ratio

    # 8. EQI Infinite Series — v6 동일
    eqi_infinite_series = np.sum(eigenfrequency_harmonic) + np.sum(eigenperiod_harmonic)
    print(f"∞ EQI Infinite Series: {eqi_infinite_series:.6f}")

    # 9. Mirror Feedback Tensor — v6 동일 (μ: MST 에지, Λ: 같은 클러스터)
    mirror_feedback_tensor = np.zeros((n, n))
    if enable_mirror_feedback:
        for i, j in unity_mst.edges():
            mirror_feedback_tensor[i, j] = eigenfrequency_ratio
            mirror_feedback_tensor[j, i] = eigenfrequency_ratio
        same = multiplicity_labels[:, None] == multiplicity_labels[None, :]
        np.fill_diagonal(same, False)
        mirror_feedback_tensor += same * eigenperiod_ratio

    collective_eqi_label_map = {
        'unity_cluster': unity_cluster_target,
        'multiplicity_labels': multiplicity_labels,
        'unity_indices': np.where(unity_mask)[0],
        'cluster_distribution': cluster_distribution,
        'pca_coords': X_pca,
        'original_features': X_clean,
        'dimensionless_symmetry_ratio': eigenfrequency_ratio / eigenperiod_ratio,
    }
    results = {
        'mirror_feedback_tensor': mirror_feedback_tensor,
        'collective_eqi_label_map': collective_eqi_label_map,
        'unity_mst': unity_mst,
        'multiplicity_clusters': multiplicity_labels,
        'eigenfrequency_harmonic': eigenfrequency_harmonic,
        'eigenperiod_harmonic': eigenperiod_harmonic,
        'eqi_infinite_series': eqi_infinite_series,
        'unity_features': unity_features,
        'unity_coords': unity_coords,
        'pca': pca32, 'pca_4to3': pca43,
        'kmeans': kmeans,
        'eigenfrequency_ratio': eigenfrequency_ratio,
        'eigenperiod_ratio': eigenperiod_ratio,
        'dimensionless_symmetry_ratio': eigenfrequency_ratio / eigenperiod_ratio,
        'maxwell_5th': MAXWELL_5TH,
        'maxwell_ledger': ledger,
    }
    print(f"\n✅ v7.0 완료 — Unity {unity_cluster_target} ({len(unity_features)} samples)")
    return results


def test_eqi_encoding_decoding_network_v7(n=200, seed=0):
    rng = np.random.default_rng(seed)       # 재현 가능한 시험 입력
    t = np.linspace(0, 8 * np.pi, n)
    phase = np.sin(t) + 0.1 * rng.standard_normal(n)
    curvature = np.cos(2 * t) + 0.1 * rng.standard_normal(n)
    entropy = np.abs(np.sin(t / 2)) + 0.05 * rng.standard_normal(n)
    phasediff = 0.3 * np.sin(3 * t) + 0.1 * rng.standard_normal(n)
    return eqi_encoding_decoding_network(phase, curvature, entropy, phasediff)


if __name__ == "__main__":
    test_eqi_encoding_decoding_network_v7()

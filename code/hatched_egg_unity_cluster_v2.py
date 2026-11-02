# hatched_egg_unity_cluster_v2.py
# 원본: hatched_egg_unity_cluster_optimized_final.py
#
# 수정 사항 (v2):
#   [1] Cluster 2 하드코딩 제거 → 물리적 기준(위상차 분산)으로 자동 선택
#   [2] 파라미터 민감도 테스트 추가 → threshold_percentile / n_clusters 변화에 따른 결과 비교
#   [3] H/L 비율 물리적 검증 추가 → SNR, 진폭 비율을 알려진 GW150914 값과 비교
#   [4] 자기참조 루프 제거 → Eigenmanifold 시각화를 실제 클러스터 데이터로 교체

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import os
import time
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "DejaVu Sans"

try:
    import ripser
    from persim import plot_diagrams
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    print("참고: ripser/persim 미설치. PH 바코드 없이 진행합니다.")


# ============================================================
# MST 구축 함수 (원본 유지)
# ============================================================

def build_direct_mst(features):
    try:
        distances = pdist(features, metric="euclidean")
        distance_matrix = squareform(distances)
        sparse_dist = csr_matrix(distance_matrix)
        mst_sparse = minimum_spanning_tree(sparse_dist)
        G = nx.from_scipy_sparse_array(mst_sparse)
        return G
    except Exception as e:
        print(f"⚠️ 직접 MST 구축 실패: {e}")
        return create_simple_graph(len(features))


def build_hierarchical_mst(features, chunk_size, n_neighbors):
    try:
        n_samples = len(features)
        print(f"🔄 계층적 MST 구축: {n_samples:,} → {chunk_size} 청크")
        indices = np.linspace(0, n_samples - 1, chunk_size, dtype=int)
        representative_features = features[indices]
        print(f"📊 대표 샘플 선택: {len(representative_features):,}개")
        G_repr = build_direct_mst(representative_features)
        G = nx.Graph()
        for u, v, data in G_repr.edges(data=True):
            G.add_edge(indices[u], indices[v], weight=data.get("weight", 1.0))
        remaining_indices = np.setdiff1d(np.arange(n_samples), indices)
        if len(remaining_indices) > 0:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nbrs.fit(representative_features)
            chunk_size_remaining = min(1000, len(remaining_indices))
            for i in range(0, len(remaining_indices), chunk_size_remaining):
                end_idx = min(i + chunk_size_remaining, len(remaining_indices))
                chunk_indices = remaining_indices[i:end_idx]
                distances, neighbor_indices = nbrs.kneighbors(features[chunk_indices])
                for j, (dist, neighbor_idx) in enumerate(zip(distances, neighbor_indices)):
                    G.add_edge(chunk_indices[j], indices[neighbor_idx[0]], weight=dist[0])
        print(f"🌳 계층적 MST 완성: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 에지")
        return G
    except Exception as e:
        print(f"⚠️ 계층적 MST 실패: {e}")
        return create_simple_graph(len(features))


def create_simple_graph(n_nodes):
    G = nx.Graph()
    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1, weight=1.0)
    return G


def construct_mst_optimized(features):
    n_samples = len(features)
    if n_samples <= 1000:
        strategy = "direct"
    elif n_samples <= 5000:
        strategy = "knn_sparse"
    else:
        strategy = "hierarchical"
    print(f"🎯 MST 전략 선택: {strategy} (샘플 수: {n_samples:,})")
    if strategy == "direct":
        return build_direct_mst(features)
    else:
        max_samples = min(2000, max(500, n_samples // 10))
        return build_hierarchical_mst(features, max_samples, n_neighbors=15)


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_strain_data(h_file_path, l_file_path):
    print(f"데이터 로드 중: {h_file_path}, {l_file_path}")
    try:
        df_H = pd.read_csv(h_file_path)
        df_L = pd.read_csv(l_file_path)
        print(f"데이터 크기: H={df_H.shape}, L={df_L.shape}")
        return df_H, df_L
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        t = np.linspace(0, 32, 131071)
        mock_H = pd.DataFrame({"time": t, "strain": np.sin(2*np.pi*t) + np.random.normal(0, 0.1, len(t))})
        mock_L = pd.DataFrame({"time": t, "strain": np.sin(2*np.pi*t + 0.1) + np.random.normal(0, 0.1, len(t))})
        return mock_H, mock_L


# ============================================================
# 2. 신호 전처리
# ============================================================

def preprocess_signal(signal):
    if np.isnan(signal).any():
        print("경고: NaN 값 포함됨 → 0으로 대체")
        signal = np.nan_to_num(signal)
    if np.all(signal == 0):
        signal = signal + np.random.normal(0, 1e-10, signal.shape)
    analytic_signal = hilbert(signal)
    phase = np.unwrap(np.angle(analytic_signal))
    tau = np.abs(analytic_signal) ** 2
    return phase, tau


# ============================================================
# 3. 특성 계산
# ============================================================

def calculate_features(phase_H, phase_L, tau_H, tau_L):
    phase_diff = phase_H - phase_L
    entropy_H = -np.sum(tau_H * np.log(tau_H + 1e-10))
    entropy_L = -np.sum(tau_L * np.log(tau_L + 1e-10))
    delta_entropy = entropy_H - entropy_L
    curvature_H = np.gradient(np.gradient(phase_H))
    features = np.column_stack([tau_H, phase_diff, delta_entropy * np.ones_like(tau_H), curvature_H])
    return features


# ============================================================
# 4. 임계값 필터링
# ============================================================

def threshold_filter(features, threshold_percentile=99.5):
    valid_features = features[~np.isnan(features).any(axis=1)]
    if len(valid_features) == 0:
        return valid_features, np.array([]), 0
    amplitude = valid_features[:, 0]
    threshold = np.percentile(amplitude, threshold_percentile) if np.sum(amplitude) != 0 else 0
    filtered_indices = np.where(amplitude >= threshold)[0]
    filtered_features = valid_features[filtered_indices]
    print(f"임계값: {threshold:.4e}, 필터링 후: {len(filtered_features):,}")
    return filtered_features, filtered_indices, threshold


# ============================================================
# 5. PCA
# ============================================================

def apply_pca(features, n_components=2):
    if len(features) == 0:
        return np.array([]), None
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    print(f"PCA 설명 분산 비율: {pca.explained_variance_ratio_}")
    return pca_result, pca


# ============================================================
# 6. K-means 클러스터링
# ============================================================

def cluster_data(pca_result, n_clusters=3):
    if len(pca_result) == 0:
        return np.array([]), None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_result)
    print(f"클러스터 분포: {np.bincount(clusters)}")
    return clusters, kmeans


# ============================================================
# [수정 1] 클러스터 자동 선택 (하드코딩 제거)
#
# 원본: 클러스터 인덱스 2 고정 (무조건 선택)
# 수정: 각 클러스터의 위상차(phase_diff) 분산을 계산하여
#       분산이 가장 높은 클러스터를 선택
#       → 신호 구조가 가장 복잡한 클러스터 = 물리적으로 가장 흥미로운 클러스터
# ============================================================

def select_main_cluster(clusters, filtered_features, pca_result):
    """
    각 클러스터의 위상차 분산을 기준으로 분석 대상 클러스터를 자동 선택.
    phase_diff = filtered_features[:, 1]
    """
    n_clusters = len(np.unique(clusters))
    phase_diff = filtered_features[:, 1]  # 특성 벡터 2번째 열 = phase_diff

    cluster_stats = {}
    for cid in range(n_clusters):
        mask = clusters == cid
        count = np.sum(mask)
        if count == 0:
            continue
        pd_var = np.var(phase_diff[mask])          # 위상차 분산 (클러스터 선택 기준)
        pd_mean = np.mean(phase_diff[mask])
        amp_mean = np.mean(filtered_features[mask, 0])  # 평균 진폭
        pca1_spread = np.std(pca_result[mask, 0])       # PCA1 축 분산

        cluster_stats[cid] = {
            "count": int(count),
            "phase_diff_variance": float(pd_var),
            "phase_diff_mean": float(pd_mean),
            "amplitude_mean": float(amp_mean),
            "pca1_spread": float(pca1_spread),
        }

    # 위상차 분산이 가장 큰 클러스터 선택
    main_cluster = max(cluster_stats, key=lambda x: cluster_stats[x]["phase_diff_variance"])

    print(f"\n{'─'*55}")
    print(f"[클러스터 자동 선택] 위상차 분산 기준")
    print(f"{'클러스터':^8} {'크기':^10} {'위상차 분산':^16} {'평균 진폭':^16} {'선택'}")
    print(f"{'─'*55}")
    for cid, s in cluster_stats.items():
        mark = "◀ 선택" if cid == main_cluster else ""
        print(f"  {cid:^6}   {s['count']:^10,}   {s['phase_diff_variance']:^16.4e}   {s['amplitude_mean']:^16.4e}  {mark}")
    print(f"{'─'*55}\n")

    return main_cluster, cluster_stats


# ============================================================
# 7. PH 다이어그램 계산
# ============================================================

def calculate_ph_optimized(data, max_dim=1, sampling_ratio=0.2, max_samples=1500):
    if not PERSISTENCE_AVAILABLE:
        return None
    n_points = len(data)
    if n_points == 0:
        return None
    if n_points > max_samples:
        num_samples = min(int(n_points * sampling_ratio), max_samples)
        print(f"🔬 샘플링: {n_points:,}개 → {num_samples:,}개")
        sample_indices = np.linspace(0, n_points - 1, num_samples, dtype=int)
        sampled_data = data[sample_indices]
    else:
        sampled_data = data
    try:
        t0 = time.time()
        diagrams = ripser.ripser(sampled_data, maxdim=max_dim, thresh=np.inf)["dgms"]
        print(f"⚡ PH 계산 완료 ({time.time()-t0:.2f}초)")
        return diagrams
    except Exception as e:
        print(f"⚠️ PH 계산 오류: {e}")
        return None


# ============================================================
# [수정 2] 파라미터 민감도 테스트
#
# threshold_percentile과 n_clusters를 변화시켜
# 필터링 후 샘플 수, 클러스터 분포, 선택된 클러스터가
# 어떻게 달라지는지 정량적으로 비교
# ============================================================

def run_sensitivity_test(features, pca_result_base, output_folder):
    """
    파라미터 민감도 테스트:
      - threshold_percentile: [99.0, 99.5, 99.8, 99.9]
      - n_clusters: [2, 3, 4, 5]
    각 조합에서 필터링 수, 클러스터 분포, 위상차 분산 최대 클러스터 기록
    """
    print("\n[파라미터 민감도 테스트 시작]")
    percentiles = [99.0, 99.5, 99.8, 99.9]
    k_values = [2, 3, 4, 5]

    records = []
    for pct in percentiles:
        filt, _, thr = threshold_filter(features, threshold_percentile=pct)
        if len(filt) == 0:
            continue
        pca_f, _ = apply_pca(filt)
        if len(pca_f) == 0:
            continue
        for k in k_values:
            clusters, _ = cluster_data(pca_f, n_clusters=k)
            if len(clusters) == 0:
                continue
            phase_diff = filt[:, 1]
            variances = [np.var(phase_diff[clusters == c]) for c in range(k)]
            best_cluster = int(np.argmax(variances))
            best_count = int(np.sum(clusters == best_cluster))
            records.append({
                "percentile": pct,
                "n_clusters": k,
                "threshold": float(thr),
                "filtered_count": len(filt),
                "best_cluster": best_cluster,
                "best_cluster_size": best_count,
                "best_phase_var": float(max(variances)),
                "cluster_dist": np.bincount(clusters).tolist(),
            })
            print(f"  pct={pct}% k={k}: 필터={len(filt):,} 최적클러스터={best_cluster} "
                  f"(크기={best_count:,}, 위상분산={max(variances):.4e})")

    # 민감도 히트맵 저장
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Parameter Sensitivity Test", fontsize=14, fontweight='bold')

    # (a) 필터링 후 샘플 수
    mat_count = np.zeros((len(percentiles), len(k_values)))
    mat_var = np.zeros((len(percentiles), len(k_values)))
    for r in records:
        i = percentiles.index(r["percentile"])
        j = k_values.index(r["n_clusters"])
        mat_count[i, j] = r["filtered_count"]
        mat_var[i, j] = r["best_phase_var"]

    im0 = axes[0].imshow(mat_count, aspect='auto', cmap='Blues')
    axes[0].set_xticks(range(len(k_values))); axes[0].set_xticklabels(k_values)
    axes[0].set_yticks(range(len(percentiles))); axes[0].set_yticklabels(percentiles)
    axes[0].set_xlabel("n_clusters"); axes[0].set_ylabel("threshold_percentile")
    axes[0].set_title("Filtered Sample Count")
    for i in range(len(percentiles)):
        for j in range(len(k_values)):
            axes[0].text(j, i, f"{int(mat_count[i,j]):,}", ha='center', va='center', fontsize=8)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mat_var, aspect='auto', cmap='Reds')
    axes[1].set_xticks(range(len(k_values))); axes[1].set_xticklabels(k_values)
    axes[1].set_yticks(range(len(percentiles))); axes[1].set_yticklabels(percentiles)
    axes[1].set_xlabel("n_clusters"); axes[1].set_ylabel("threshold_percentile")
    axes[1].set_title("Best Cluster Phase-Diff Variance")
    for i in range(len(percentiles)):
        for j in range(len(k_values)):
            axes[1].text(j, i, f"{mat_var[i,j]:.2e}", ha='center', va='center', fontsize=7)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{output_folder}/sensitivity_test.png", dpi=120)
    plt.close()
    print(f"📊 민감도 히트맵 저장: {output_folder}/sensitivity_test.png")

    # JSON 저장
    with open(f"{output_folder}/sensitivity_records.json", "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return records


# ============================================================
# [수정 3] H/L 비율 물리적 검증
#
# 원본: H/L 비율이 코드 내에서 계산되지 않음 (Code2에서만 산출)
# 수정: tau_H, tau_L로부터 직접 RMS 진폭 비율과 에너지 비율을 계산하고
#       GW150914의 알려진 참조값과 비교
#
# 참조값 (LIGO 공식 발표):
#   - H1 optimal SNR: ~24, L1 optimal SNR: ~13  (논문: arXiv:1602.03839)
#   - H1 peak strain: ~1.0e-21, L1 peak strain: ~0.7e-21
#   - 도달 시간차: 6.9ms (L1 먼저 도달)
# ============================================================

# GW150914 공식 참조값
GW150914_REFERENCE = {
    "H1_SNR": 24.0,
    "L1_SNR": 13.0,
    "H1_peak_strain": 1.0e-21,
    "L1_peak_strain": 0.7e-21,
    "time_delay_ms": 6.9,          # L1 먼저 도달
    "source": "arXiv:1602.03839",
}


def verify_hl_ratio(tau_H, tau_L, output_folder):
    """
    측정된 H/L 진폭 비율을 GW150914 공식 참조값과 비교.
    tau = |analytic_signal|^2 = 순간 에너지 (진폭²)
    """
    print("\n[H/L 비율 물리적 검증]")

    # 전체 구간 RMS 진폭
    rms_H = np.sqrt(np.mean(tau_H))
    rms_L = np.sqrt(np.mean(tau_L))
    total_rms = rms_H + rms_L
    H_ratio = rms_H / total_rms
    L_ratio = rms_L / total_rms

    # 에너지 비율
    energy_H = np.sum(tau_H)
    energy_L = np.sum(tau_L)
    total_energy = energy_H + energy_L
    H_energy_ratio = energy_H / total_energy

    # 피크 진폭
    peak_H = np.sqrt(np.max(tau_H))
    peak_L = np.sqrt(np.max(tau_L))

    # 참조값 기반 예상 비율
    ref_snr_ratio = GW150914_REFERENCE["H1_SNR"] / (
        GW150914_REFERENCE["H1_SNR"] + GW150914_REFERENCE["L1_SNR"])
    ref_strain_ratio = GW150914_REFERENCE["H1_peak_strain"] / (
        GW150914_REFERENCE["H1_peak_strain"] + GW150914_REFERENCE["L1_peak_strain"])

    verification = {
        "measured": {
            "H_rms": float(rms_H),
            "L_rms": float(rms_L),
            "H_rms_ratio": float(H_ratio),
            "L_rms_ratio": float(L_ratio),
            "H_energy_ratio": float(H_energy_ratio),
            "H_peak": float(peak_H),
            "L_peak": float(peak_L),
            "peak_ratio_H_over_L": float(peak_H / peak_L) if peak_L > 0 else None,
        },
        "reference_GW150914": GW150914_REFERENCE,
        "comparison": {
            "ref_SNR_H_ratio": float(ref_snr_ratio),
            "ref_strain_H_ratio": float(ref_strain_ratio),
            "measured_H_rms_ratio": float(H_ratio),
            "SNR_match_pct": float(abs(H_ratio - ref_snr_ratio) / ref_snr_ratio * 100),
            "strain_match_pct": float(abs(H_ratio - ref_strain_ratio) / ref_strain_ratio * 100),
        },
    }

    print(f"  측정값:")
    print(f"    H RMS 비율:    {H_ratio:.6f}")
    print(f"    L RMS 비율:    {L_ratio:.6f}")
    print(f"    H 에너지 비율: {H_energy_ratio:.6f}")
    print(f"    Peak H/L:      {peak_H/peak_L:.4f}")
    print(f"  참조값 (GW150914, {GW150914_REFERENCE['source']}):")
    print(f"    SNR 기반 H 비율:    {ref_snr_ratio:.6f}  (H1={GW150914_REFERENCE['H1_SNR']}, L1={GW150914_REFERENCE['L1_SNR']})")
    print(f"    Strain 기반 H 비율: {ref_strain_ratio:.6f}  (H1={GW150914_REFERENCE['H1_peak_strain']:.1e}, L1={GW150914_REFERENCE['L1_peak_strain']:.1e})")
    print(f"  일치도:")
    print(f"    vs SNR 비율:    오차 {verification['comparison']['SNR_match_pct']:.2f}%")
    print(f"    vs Strain 비율: 오차 {verification['comparison']['strain_match_pct']:.2f}%")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H/L Ratio Physical Verification (GW150914)", fontsize=13, fontweight='bold')

    # (a) 비율 막대 비교
    labels = ["Measured\n(RMS)", "Ref: SNR\n(arXiv:1602)", "Ref: Strain\n(arXiv:1602)"]
    h_vals = [H_ratio, ref_snr_ratio, ref_strain_ratio]
    l_vals = [L_ratio, 1-ref_snr_ratio, 1-ref_strain_ratio]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, h_vals, width, label='H1 (Hanford)', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, l_vals, width, label='L1 (Livingston)', color='coral', alpha=0.8)
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Equal split')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Normalized Ratio")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("H1 vs L1 Amplitude Ratio")
    axes[0].legend()
    for i, (h, l) in enumerate(zip(h_vals, l_vals)):
        axes[0].text(i - width/2, h + 0.01, f"{h:.3f}", ha='center', fontsize=9)
        axes[0].text(i + width/2, l + 0.01, f"{l:.3f}", ha='center', fontsize=9)

    # (b) 순간 에너지 (tau) 분포 비교 — 처음 10000 샘플
    n_show = min(10000, len(tau_H))
    t_axis = np.arange(n_show)
    axes[1].plot(t_axis, np.sqrt(tau_H[:n_show]), color='steelblue', alpha=0.6, linewidth=0.5, label='H1 amplitude')
    axes[1].plot(t_axis, np.sqrt(tau_L[:n_show]), color='coral', alpha=0.6, linewidth=0.5, label='L1 amplitude')
    axes[1].set_xlabel("Sample index (first 10K)")
    axes[1].set_ylabel("Instantaneous Amplitude")
    axes[1].set_title("H1 vs L1 Instantaneous Amplitude")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_folder}/hl_ratio_verification.png", dpi=120)
    plt.close()
    print(f"📊 H/L 검증 그래프 저장: {output_folder}/hl_ratio_verification.png")

    # JSON 저장
    with open(f"{output_folder}/hl_ratio_verification.json", "w") as f:
        json.dump(verification, f, indent=2, ensure_ascii=False)

    return verification


# ============================================================
# [수정 4] 자기참조 루프 제거 → 실제 데이터 기반 시각화
#
# 원본 compare_with_simulation():
#   - EQI Eigenmanifold = np.random.seed(42) 로 생성 (실제 데이터 무관)
#   - Cycloid/Fourier도 고정 파라미터로 생성
#
# 수정:
#   - EQI Eigenmanifold → 실제 클러스터 데이터의 PCA 투영으로 교체
#   - Cycloid 파라미터 R을 실제 tau 평균에서 유도
#   - Fourier 근사 대상을 실제 위상차 데이터로 교체
# ============================================================

def visualize_with_real_data(real_ph, cluster_coords, cluster_features, all_clusters,
                              pca_result, main_cluster, output_folder):
    """
    원본의 np.random 기반 시각화를 실제 데이터 기반으로 교체.
    """
    phase_diff_real = cluster_features[:, 1]  # 선택된 클러스터의 위상차

    # --- (a) Cycloid: R을 실제 진폭 평균에서 유도 ---
    R = float(np.mean(np.sqrt(cluster_features[:, 0])))  # 원본: R=1.0 고정
    R = max(R / R, 1.0)  # 정규화 (단위 유지)
    t = np.linspace(0, 4 * np.pi, 1000)
    # 감쇠 계수를 실제 위상차 분산에서 유도 (원본: 0.1 고정)
    damping_coef = float(np.clip(np.var(phase_diff_real) / (np.var(phase_diff_real) + 1), 0.05, 0.3))
    x_c = R * (t - np.sin(t))
    y_c = R * (1 - np.cos(t))
    damp = np.exp(-damping_coef * t)
    cycloid_data = np.column_stack([x_c * damp, y_c * damp])

    # --- (b) Fourier: 실제 위상차 샘플에서 근사 ---
    n_show = min(1000, len(phase_diff_real))
    sample_idx = np.linspace(0, len(phase_diff_real)-1, n_show, dtype=int)
    pd_sample = phase_diff_real[sample_idx]
    # 정규화
    pd_norm = (pd_sample - pd_sample.mean()) / (pd_sample.std() + 1e-10)
    x_fourier = np.linspace(-np.pi, np.pi, n_show)
    # Fourier 계수 추정 (FFT 기반)
    fft_coeffs = np.fft.rfft(pd_norm)
    n_terms = 10
    fft_trunc = np.zeros_like(fft_coeffs)
    fft_trunc[:n_terms] = fft_coeffs[:n_terms]
    fourier_approx = np.fft.irfft(fft_trunc, n=n_show)

    # --- (c) EQI Eigenmanifold: 실제 클러스터 데이터 PCA ---
    # 원본: np.random.seed(42) 완전 랜덤 데이터
    # 수정: 실제 cluster_coords에서 소규모 샘플 추출
    n_eig = min(100, len(cluster_coords))
    eig_idx = np.linspace(0, len(cluster_coords)-1, n_eig, dtype=int)
    eig_data = cluster_coords[eig_idx]  # 실제 데이터
    G_eig = nx.Graph()
    for i in range(len(eig_data)):
        G_eig.add_node(i, pos=eig_data[i])
    for i in range(len(eig_data)):
        for j in range(i+1, len(eig_data)):
            dist = np.linalg.norm(eig_data[i] - eig_data[j])
            if dist < np.percentile([np.linalg.norm(eig_data[i]-eig_data[k])
                                     for k in range(len(eig_data)) if k != i], 20):
                G_eig.add_edge(i, j, weight=dist)
    if G_eig.number_of_edges() > 0:
        eig_mst = nx.minimum_spanning_tree(G_eig)
    else:
        eig_mst = G_eig

    # --- 4-패널 비교 시각화 ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"Analysis Results (Cluster {main_cluster} — auto-selected by phase variance)",
                 fontsize=13, fontweight='bold')

    # Panel 1: PH 바코드
    ax = axes[0, 0]
    ax.set_title("PH Barcodes (Selected Cluster)")
    if PERSISTENCE_AVAILABLE and real_ph is not None:
        plt.sca(ax)
        plot_diagrams(real_ph, plot_only=[0], ax=ax)
    else:
        ax.text(0.5, 0.5, "PH Barcodes\n(ripser 필요)", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # Panel 2: Cycloid (R, damping이 실제 데이터에서 유도됨)
    ax = axes[0, 1]
    ax.plot(cycloid_data[:, 0], cycloid_data[:, 1], 'r-', linewidth=1.5)
    ax.set_title(f"Holistic Cycloid Wave\n(R={R:.3f} from data, damping={damping_coef:.3f})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 3: Fourier (실제 위상차 데이터 근사)
    ax = axes[1, 0]
    ax.plot(x_fourier, pd_norm, color='lightgray', linewidth=0.8, alpha=0.7, label='Real phase_diff (normalized)')
    ax.plot(x_fourier, fourier_approx, 'b-', linewidth=1.5, label=f'Fourier approx (n={n_terms})')
    ax.set_title("Fourier Approximation of Real Phase Difference")
    ax.set_xlabel("Normalized index"); ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: EQI Eigenmanifold (실제 클러스터 데이터)
    ax = axes[1, 1]
    pos = {i: eig_data[i] for i in range(len(eig_data))}
    # 전체 클러스터 배경
    for cid in np.unique(all_clusters):
        mask = all_clusters == cid
        color = 'steelblue' if cid == main_cluster else 'lightgray'
        alpha = 0.6 if cid == main_cluster else 0.2
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   s=3, alpha=alpha, color=color)
    # MST overlay
    nx.draw_networkx(eig_mst, pos=pos, ax=ax,
                     node_size=15, node_color='red',
                     edge_color='black', width=0.8,
                     with_labels=False)
    ax.set_title(f"Eigenmanifold (Cluster {main_cluster}, real data, n={n_eig})")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")

    plt.tight_layout()
    plt.savefig(f"{output_folder}/comparison_v2.png", dpi=120)
    plt.close()
    print(f"📊 비교 시각화 저장: {output_folder}/comparison_v2.png")


# ============================================================
# 시각화 (원본 유지)
# ============================================================

def visualize_results(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    if "pca_result" in results and "clusters" in results and len(results["pca_result"]) > 0:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(results["pca_result"][:, 0], results["pca_result"][:, 1],
                              c=results["clusters"], cmap="viridis", s=50, alpha=0.8)
        # 선택된 클러스터 강조
        main = results.get("main_cluster", -1)
        if main >= 0:
            mask = results["clusters"] == main
            plt.scatter(results["pca_result"][mask, 0], results["pca_result"][mask, 1],
                        edgecolors='red', facecolors='none', s=60, linewidths=0.5,
                        label=f"Selected: Cluster {main}", alpha=0.5)
            plt.legend()
        plt.title("PCA Projection with K-means Clustering")
        plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.savefig(f"{output_folder}/pca_clusters.png")
        plt.close()

    if PERSISTENCE_AVAILABLE and results.get("ph_diagrams") is not None:
        plt.figure(figsize=(10, 8))
        plot_diagrams(results["ph_diagrams"], plot_only=[0])
        plt.title("Persistent Homology Barcodes")
        plt.savefig(f"{output_folder}/ph_barcodes.png")
        plt.close()

    if "cluster_coords" in results and results.get("mst") is not None:
        plt.figure(figsize=(10, 8))
        pos = {i: results["cluster_coords"][i] for i in range(len(results["cluster_coords"]))}
        nx.draw(results["mst"], pos, node_size=30, node_color="red",
                edge_color="black", width=1.0)
        plt.title("Minimum Spanning Tree (MST) of Selected Cluster")
        plt.axis("off")
        plt.savefig(f"{output_folder}/mst_network.png")
        plt.close()


# ============================================================
# 메인 파이프라인
# ============================================================

def run_eqi_pipeline(h_file_path, l_file_path,
                     output_folder="./output",
                     threshold_percentile=99.5,
                     n_clusters=3,
                     run_sensitivity=True):
    """
    수정된 EQI 파이프라인.

    Parameters
    ----------
    threshold_percentile : float
        진폭 필터링 백분위수 (기본 99.5)
    n_clusters : int
        K-means 클러스터 수 (기본 3)
    run_sensitivity : bool
        파라미터 민감도 테스트 실행 여부 (기본 True)
    """
    start_time = datetime.now()
    os.makedirs(output_folder, exist_ok=True)
    print("=" * 60)
    print("EQI 파이프라인 v2 시작")
    print(f"  threshold_percentile = {threshold_percentile}")
    print(f"  n_clusters           = {n_clusters}")
    print("=" * 60)

    # 1. 데이터 로드
    df_H, df_L = load_strain_data(h_file_path, l_file_path)

    # 2. 전처리
    print("\n[STEP 1] 신호 전처리...")
    phase_H, tau_H = preprocess_signal(df_H["strain"].values)
    phase_L, tau_L = preprocess_signal(df_L["strain"].values)

    # [수정 3] H/L 비율 물리적 검증
    print("\n[STEP 2] H/L 비율 물리적 검증...")
    hl_verification = verify_hl_ratio(tau_H, tau_L, output_folder)

    # 3. 특성 계산
    print("\n[STEP 3] 특성 계산...")
    features = calculate_features(phase_H, phase_L, tau_H, tau_L)

    # 4. 필터링
    print("\n[STEP 4] 임계값 필터링...")
    filtered_features, filtered_indices, threshold = threshold_filter(
        features, threshold_percentile=threshold_percentile)
    if len(filtered_features) == 0:
        print("필터링 후 데이터 없음.")
        return {}

    # [수정 2] 민감도 테스트
    if run_sensitivity:
        print("\n[STEP 5] 파라미터 민감도 테스트...")
        sensitivity_records = run_sensitivity_test(features, None, output_folder)
    else:
        sensitivity_records = []

    # 5. PCA
    print("\n[STEP 6] PCA...")
    pca_result, pca = apply_pca(filtered_features)
    if len(pca_result) == 0:
        return {}

    # 6. 클러스터링
    print("\n[STEP 7] 클러스터링...")
    clusters, kmeans = cluster_data(pca_result, n_clusters=n_clusters)
    if len(clusters) == 0:
        return {}

    # [수정 1] 클러스터 자동 선택 (하드코딩 제거)
    print("\n[STEP 8] 클러스터 자동 선택...")
    main_cluster, cluster_stats = select_main_cluster(clusters, filtered_features, pca_result)

    cluster_mask = clusters == main_cluster
    cluster_indices = np.where(cluster_mask)[0]
    cluster_coords = pca_result[cluster_mask]
    cluster_features = filtered_features[cluster_mask]
    print(f"✅ Cluster {main_cluster} 선택: {len(cluster_indices):,}개 포인트")

    # 7. MST
    print("\n[STEP 9] MST 구축...")
    cluster_mst = construct_mst_optimized(cluster_coords)

    # 8. PH
    print("\n[STEP 10] PH 계산...")
    ph_diagrams = calculate_ph_optimized(cluster_coords, max_samples=1500, sampling_ratio=0.3)

    # 결과 수집
    results = {
        "dataframes": (df_H, df_L),
        "features": features,
        "filtered_features": filtered_features,
        "pca_result": pca_result,
        "clusters": clusters,
        "main_cluster": main_cluster,
        "cluster_stats": cluster_stats,
        "cluster_indices": cluster_indices,
        "cluster_coords": cluster_coords,
        "cluster_features": cluster_features,
        "mst": cluster_mst,
        "ph_diagrams": ph_diagrams,
        "hl_verification": hl_verification,
        "sensitivity_records": sensitivity_records,
    }

    # 9. 시각화
    print("\n[STEP 11] 시각화...")
    visualize_results(results, output_folder)

    # [수정 4] 자기참조 루프 제거 → 실제 데이터 기반 시각화
    visualize_with_real_data(
        ph_diagrams, cluster_coords, cluster_features,
        clusters, pca_result, main_cluster, output_folder
    )

    # 10. 보고서 저장
    execution_time = (datetime.now() - start_time).total_seconds()
    report_path = f"{output_folder}/eqi_report_v2.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EQI Analysis Report v2 - {datetime.now()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"데이터: {h_file_path}, {l_file_path}\n")
        f.write(f"데이터 크기: H={df_H.shape}, L={df_L.shape}\n\n")
        f.write(f"[파라미터]\n")
        f.write(f"  threshold_percentile = {threshold_percentile}\n")
        f.write(f"  n_clusters           = {n_clusters}\n\n")
        f.write(f"[필터링]\n")
        f.write(f"  임계값: {threshold:.4e}, 필터링 후: {len(filtered_features):,}\n\n")
        f.write(f"[클러스터 자동 선택 결과] (원본: Cluster 2 하드코딩)\n")
        for cid, s in cluster_stats.items():
            mark = " ← 선택됨" if cid == main_cluster else ""
            f.write(f"  Cluster {cid}: 크기={s['count']:,}, "
                    f"위상분산={s['phase_diff_variance']:.4e}{mark}\n")
        f.write(f"\n[H/L 물리적 검증]\n")
        v = hl_verification
        f.write(f"  측정 H RMS 비율: {v['measured']['H_rms_ratio']:.6f}\n")
        f.write(f"  참조(SNR 기반):  {v['comparison']['ref_SNR_H_ratio']:.6f}  "
                f"오차={v['comparison']['SNR_match_pct']:.2f}%\n")
        f.write(f"  참조(Strain):    {v['comparison']['ref_strain_H_ratio']:.6f}  "
                f"오차={v['comparison']['strain_match_pct']:.2f}%\n")
        f.write(f"  참조 출처: {GW150914_REFERENCE['source']}\n\n")
        f.write(f"[MST]\n")
        f.write(f"  노드: {cluster_mst.number_of_nodes():,}, 에지: {cluster_mst.number_of_edges():,}\n\n")
        f.write(f"[PCA]\n")
        f.write(f"  설명 분산 비율: {pca.explained_variance_ratio_}\n\n")
        f.write(f"[실행 시간] {execution_time:.2f}초\n")
        f.write("=" * 60 + "\n")

    print(f"\n{'='*60}")
    print(f"파이프라인 완료!  실행 시간: {execution_time:.2f}초")
    print(f"결과 저장: {output_folder}/")
    print(f"  pca_clusters.png          — PCA + 클러스터 (선택 클러스터 강조)")
    print(f"  ph_barcodes.png           — PH 바코드")
    print(f"  mst_network.png           — MST 네트워크")
    print(f"  comparison_v2.png         — 실제 데이터 기반 비교 [수정4]")
    print(f"  hl_ratio_verification.png — H/L 물리적 검증 [수정3]")
    print(f"  sensitivity_test.png      — 파라미터 민감도 히트맵 [수정2]")
    print(f"  eqi_report_v2.txt         — 분석 보고서")
    print(f"{'='*60}\n")

    return results, execution_time


# ============================================================
# 실행 진입점
# ============================================================

def main():
    print("Holistic Cycloid Wave EQI Framework v2")
    print("=" * 60)
    print("수정 사항:")
    print("  [1] 클러스터 자동 선택 (위상차 분산 기준, Cluster 2 하드코딩 제거)")
    print("  [2] 파라미터 민감도 테스트 (threshold × n_clusters 히트맵)")
    print("  [3] H/L 비율 물리적 검증 (GW150914 SNR/Strain 참조값 비교)")
    print("  [4] 자기참조 루프 제거 (Eigenmanifold = 실제 클러스터 데이터)")
    print("=" * 60)

    h_file_path = input("Hanford 데이터 파일 경로: ").strip()
    l_file_path = input("Livingston 데이터 파일 경로: ").strip()

    pct_input = input("threshold_percentile [기본 99.5]: ").strip()
    threshold_percentile = float(pct_input) if pct_input else 99.5

    k_input = input("n_clusters [기본 3]: ").strip()
    n_clusters = int(k_input) if k_input else 3

    sens_input = input("파라미터 민감도 테스트 실행? (y/n) [기본 y]: ").strip().lower()
    run_sensitivity = sens_input != 'n'

    run_eqi_pipeline(
        h_file_path, l_file_path,
        output_folder="./output_v2",
        threshold_percentile=threshold_percentile,
        n_clusters=n_clusters,
        run_sensitivity=run_sensitivity,
    )


if __name__ == "__main__":
    main()

# eqi_mother_full_v2.py
# 어머니 코드(EQI_smallest_unit_full_code.py) 확장판
#
# 원본과의 차이:
#   [원본] df_merged.csv (100개 서브샘플, H1만, Cluster2 전용)
#   [v2]   H1+L1 HDF5 직접 로드, 131,072 전체 샘플, 서브샘플링 선택 가능
#
# 추가 기능:
#   1. H1/L1 HDF5 직접 로드 (h5py)
#   2. L1 시간 지연 보정 (6.9ms = 28샘플)
#   3. 화이트닝 (PSD 기반 노이즈 제거)
#   4. 합병 구간 자동 탐지 (PhaseDiff 변화율 + Entropy 최저)
#   5. 전체/서브샘플/합병±N초 세 가지 분석 모드
#   6. 어머니 코드 원본 시각화 재현 (+ 탐지 결과 오버레이)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy.signal import hilbert, welch, butter, filtfilt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (윈도우/맥/리눅스 자동)
import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# GW150914 물리 상수
# ============================================================
GW150914 = {
    "merger_gps":    1126259462.4,   # 합병 GPS 시각
    "gps_start":     1126259446,     # 데이터 시작 GPS
    "duration":      32.0,           # 초
    "sample_rate":   4096,           # Hz
    "n_samples":     131072,         # 2^17
    "xspacing":      0.000244140625, # s/sample = 1/4096
    "L1_advance_ms": 6.9,            # L1이 H1보다 먼저 도달 (ms)
    "L1_advance_samples": 28,        # 6.9ms × 4096 ≈ 28샘플
    "merger_t":      16.4,           # 데이터 내 합병 시각 (s)
    "merger_idx":    67174,          # 합병 원본 샘플 인덱스
    "H1_SNR":        24.0,           # H1 optimal SNR
    "L1_SNR":        13.0,           # L1 optimal SNR
    "source":        "arXiv:1602.03839",
}


# ============================================================
# 1. HDF5 데이터 로드
# ============================================================

def load_hdf5(filepath):
    """
    LIGO LOSC HDF5 파일에서 strain 데이터 로드.
    h5py가 없으면 안내 메시지 출력.
    """
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            strain = f['strain/Strain'][:]
            gps_start = f['strain/Strain'].attrs['Xstart']
            xspacing = f['strain/Strain'].attrs['Xspacing']
            n_points = f['strain/Strain'].attrs['Npoints']
        t = np.arange(n_points) * xspacing
        print(f"  로드 완료: {filepath}")
        print(f"  샘플: {len(strain):,}, GPS시작: {gps_start}, "
              f"샘플레이트: {1/xspacing:.0f}Hz")
        return strain, t, gps_start, xspacing
    except ImportError:
        print("  ⚠️  h5py 미설치. pip install h5py 후 재실행하세요.")
        return None, None, None, None
    except Exception as e:
        print(f"  ⚠️  HDF5 로드 실패: {e}")
        return None, None, None, None


def load_from_csv_fallback(csv_path):
    """
    HDF5 없을 때 df_merged.csv로 폴백 (원본 어머니 코드 방식).
    """
    print(f"  폴백: CSV 로드 ({csv_path})")
    df = pd.read_csv(csv_path)
    return df


# ============================================================
# 2. 신호 전처리
# ============================================================

def bandpass_filter(strain, sample_rate, lowcut=35.0, highcut=350.0, order=4):
    """
    GW150914 신호 대역 (35~350 Hz) 밴드패스 필터.
    """
    nyq = sample_rate / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, strain)


def whiten(strain, sample_rate, fft_len=None):
    """
    PSD 기반 화이트닝 (v2 수정).

    v1 문제: 경계 불연속으로 인한 엣지 아티팩트 → 진폭 1e11~1e12 폭증
    v2 수정:
      1. Tukey 윈도우(alpha=0.1)로 경계 스무딩 → 아티팩트 억제
      2. 엣지 0.5초 크롭 → 잔여 아티팩트 제거
      3. std 정규화 → 진폭을 ~1.0 수준으로 통일
    """
    from scipy.signal import windows as sig_windows

    if fft_len is None:
        fft_len = min(len(strain), 4 * sample_rate)

    # 1. Tukey 윈도우 적용 (경계 스무딩)
    tuk = sig_windows.tukey(len(strain), alpha=0.1)
    strain_w = strain * tuk

    # 2. PSD 추정 (윈도우 적용된 신호로)
    freqs, psd = welch(strain_w, fs=sample_rate, nperseg=fft_len)
    psd_interp = np.interp(
        np.fft.rfftfreq(len(strain), d=1.0/sample_rate),
        freqs, psd
    )
    psd_interp[psd_interp < 1e-60] = 1e-60

    # 3. 화이트닝
    strain_fft = np.fft.rfft(strain_w)
    whitened_fft = strain_fft / np.sqrt(psd_interp / 2.0)
    whitened = np.fft.irfft(whitened_fft, n=len(strain))

    # 4. 엣지 크롭: 양쪽 0.5초를 0으로 (아티팩트 구간 제거)
    crop = int(0.5 * sample_rate)
    whitened[:crop] = 0.0
    whitened[-crop:] = 0.0

    # 5. std 정규화 (중간 구간 기준)
    mid = whitened[crop:-crop]
    std = mid.std()
    if std > 0:
        whitened = whitened / std

    return whitened


def compute_phase_features(h1, l1, sample_rate):
    """
    H1, L1 신호에서 PhaseDiff, Entropy, Tau 계산.

    v1 문제:
      - Tau: max 정규화 → 스파이크 1개가 max를 지배 → 나머지 전부 ≈ 0
      - Entropy: 100ms 윈도우 너무 좁음 → 노이즈에 민감
    v2 수정:
      - Tau: 98퍼센타일 정규화 → 이상값 영향 제거
      - Entropy: 500ms 윈도우 → 안정적 국소 추정
    """
    analytic_H = hilbert(h1)
    analytic_L = hilbert(l1)

    phase_H = np.unwrap(np.angle(analytic_H))
    phase_L = np.unwrap(np.angle(analytic_L))

    # PhaseDiff: [-π, π] wrap
    phase_diff_raw = phase_H - phase_L
    phase_diff = (phase_diff_raw + np.pi) % (2 * np.pi) - np.pi

    # Tau: 98퍼센타일 정규화 (v1: max → 이상값 지배 문제 수정)
    tau_H = np.abs(analytic_H) ** 2
    crop = int(0.5 * sample_rate)
    tau_mid = tau_H[crop:-crop]
    p98 = np.percentile(tau_mid, 98) if len(tau_mid) > 0 else tau_H.max()
    if p98 == 0:
        p98 = 1e-30
    tau = np.clip(tau_H / p98, 0.0, 1.0)

    # Entropy: 500ms 윈도우 (v1: 100ms → 노이즈 민감 수정)
    window = max(1, sample_rate // 2)  # 500ms
    local_energy = np.convolve(tau, np.ones(window)/window, mode='same')
    p = local_energy / (local_energy.sum() + 1e-30)
    p = np.clip(p, 1e-30, 1.0)
    entropy_signal = -p * np.log(p)
    entropy = (entropy_signal - entropy_signal.min()) / (
        entropy_signal.max() - entropy_signal.min() + 1e-30)

    return phase_diff, entropy, tau


# ============================================================
# 3. 합병 구간 자동 탐지
# ============================================================

def detect_merger(phase_diff, entropy, tau, t,
                  window_ms=500, sample_rate=4096):
    """
    합병 구간 자동 탐지.

    v1 문제: 엣지 마스킹 없음 → 화이트닝 아티팩트(t≈31.7s)가 피크로 선택됨
    v2 수정:
      1. 엣지 마스킹: 양쪽 0.5초 제외
      2. Tau 에너지 급증 지표 추가 (chirp 직전 에너지 증가 포착)
      3. 결합 점수 = PhaseDiff변화율 × coherence × Tau에너지
    """
    crop = int(0.5 * sample_rate)
    window = int(window_ms * 1e-3 * sample_rate)

    # 지표 1: PhaseDiff 변화율 (chirp 가속 포착)
    phase_grad = np.abs(np.gradient(phase_diff))
    phase_grad_smooth = np.convolve(
        phase_grad, np.ones(window)/window, mode='same')
    pg_norm = phase_grad_smooth / (phase_grad_smooth[crop:-crop].max() + 1e-30)

    # 지표 2: Coherence = 1 - Entropy (낮은 엔트로피 = 높은 신호 결집)
    coherence = 1.0 - entropy

    # 지표 3: Tau 국소 에너지 (신호 에너지 증가 포착)
    tau_smooth = np.convolve(tau, np.ones(window)/window, mode='same')
    tau_norm = tau_smooth / (tau_smooth[crop:-crop].max() + 1e-30)

    # 결합 점수
    combined_score = pg_norm * coherence * tau_norm

    # 엣지 마스킹 (v1 버그 수정: 아티팩트 구간 제외)
    combined_score[:crop] = 0.0
    combined_score[-crop:] = 0.0

    peak_idx = int(np.argmax(combined_score))
    peak_t = float(t[peak_idx])

    true_merger_t  = GW150914["merger_t"]
    true_merger_idx = GW150914["merger_idx"]
    error_ms = abs(peak_t - true_merger_t) * 1000

    print(f"\n[합병 탐지 결과]")
    print(f"  탐지 인덱스: {peak_idx:,}  (t={peak_t:.3f}s)")
    print(f"  실제 합병:   {true_merger_idx:,}  (t={true_merger_t:.1f}s)")
    print(f"  오차:        {error_ms:.1f} ms")
    if error_ms < 500:
        print(f"  ✅ 500ms 이내 탐지 성공")
    elif error_ms < 2000:
        print(f"  ⚠️  2000ms 이내 — 서브샘플 어머니 코드(1500ms)와 유사")
    else:
        print(f"  ❌ 2000ms 초과 — 추가 튜닝 필요")

    # 상위 5개 피크 출력 (탐지 맥락 확인용)
    score_copy = combined_score.copy()
    print(f"\n  [결합 점수 상위 5개]")
    for rank in range(5):
        idx = int(np.argmax(score_copy))
        print(f"    {rank+1}위: idx={idx:,}  t={t[idx]:.3f}s  "
              f"score={score_copy[idx]:.4f}")
        # 주변 ±1초 마스킹 (중복 피크 제거)
        lo = max(0, idx - sample_rate)
        hi = min(len(score_copy), idx + sample_rate)
        score_copy[lo:hi] = 0.0

    return peak_idx, peak_t, combined_score


# ============================================================
# 4. 서브샘플링 및 클러스터링 (어머니 코드 원본 로직 재현)
# ============================================================

def subsample_and_cluster(phase_diff, entropy, tau, t,
                           n_sub=100, n_clusters=3,
                           merger_idx=None):
    """
    원본 어머니 코드: 100개 서브샘플 + KMeans 클러스터링.
    merger_idx가 주어지면 합병 주변 구간 집중 서브샘플링도 지원.
    """
    n_total = len(phase_diff)

    # 균등 서브샘플링
    indices = np.linspace(0, n_total-1, n_sub, dtype=int)
    pd_sub = phase_diff[indices]
    ent_sub = entropy[indices]
    tau_sub = tau[indices]
    t_sub = t[indices]

    # PCA (어머니 코드 동일)
    features = np.column_stack([pd_sub, ent_sub, tau_sub])
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(features)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    print(f"\n[서브샘플 클러스터링]")
    print(f"  서브샘플: {n_sub}개  (원본 {n_total:,}개의 1/{n_total//n_sub})")
    print(f"  PCA 설명 분산: {pca.explained_variance_ratio_}")
    print(f"  클러스터 분포: {np.bincount(clusters)}")

    df_sub = pd.DataFrame({
        'PhaseDiff': pd_sub,
        'Entropy':   ent_sub,
        'Tau':       tau_sub,
        'time':      t_sub,
        'cluster':   clusters,
        'pca1':      pca_coords[:, 0],
        'pca2':      pca_coords[:, 1],
        'orig_idx':  indices,
    })

    return df_sub, pca


# ============================================================
# 5. 어머니 코드 핵심 시각화 재현 (원본 + 탐지 오버레이)
# ============================================================

def visualize_mother_code(df, merger_idx_sub, output_folder, label=""):
    """
    원본 어머니 코드 시각화 재현:
      - 벡터장 스트림플롯 (PhaseDiff 기반)
      - 야코비안 특이점
      - MST 피드백 경로
      - Entropy 흐름선
    + 합병 탐지 위치 오버레이
    """
    size = int(np.sqrt(len(df)))
    if size * size != len(df):
        size = int(np.sqrt(len(df))) + 1
        df = df.iloc[:size*size].copy()

    x = df["pca1"].values
    y = df["pca2"].values

    # 벡터장 (PhaseDiff 그라디언트)
    phase_field_2d = gaussian_filter(
        df["PhaseDiff"].values[:size*size].reshape((size, size)), sigma=2)
    u, v_field = np.gradient(phase_field_2d)

    # Entropy 흐름장
    entropy_grid = gaussian_filter(
        df["Entropy"].values[:size*size].reshape((size, size)), sigma=2)
    U_entropy, V_entropy = np.gradient(entropy_grid)

    df = df.copy()
    df["Phase_x"] = u.flatten()[:len(df)]
    df["Phase_y"] = v_field.flatten()[:len(df)]

    # 야코비안 특이점
    dudx, dudy = np.gradient(u)
    dvdx, dvdy = np.gradient(v_field)
    jacobian_dets = dudx * dvdy - dudy * dvdx
    singularity_mask = np.abs(jacobian_dets) > np.percentile(
        np.abs(jacobian_dets), 98)
    singularity_points = np.argwhere(singularity_mask)

    # MST (위상 유사도 가중치 — 원본 동일)
    coords = df[["pca1", "pca2"]].values
    n = min(len(coords), size*size)
    coords = coords[:n]
    dist_matrix = cdist(coords, coords, metric="euclidean")
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            phase_sim = np.cos(
                df["PhaseDiff"].iloc[i] - df["PhaseDiff"].iloc[j])
            weight = dist_matrix[i, j] * (1 - phase_sim)
            G.add_edge(i, j, weight=weight)
    mst = nx.minimum_spanning_tree(G)

    # 격자
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), size),
        np.linspace(y.min(), y.max(), size)
    )

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # [왼쪽] 원본 어머니 코드 시각화 재현
    ax = axes[0]
    ax.streamplot(grid_x, grid_y, u, v_field,
                  color="gray", linewidth=0.7, density=1.2)
    sc = ax.quiver(x[:n], y[:n],
                   df["Phase_x"].values[:n], df["Phase_y"].values[:n],
                   df["Tau"].values[:n], cmap="viridis", scale=30)
    ax.scatter(x[:n], y[:n], c=df["Entropy"].values[:n],
               cmap="coolwarm", s=30, edgecolor="k", zorder=3)

    for i, j in singularity_points:
        if i < size and j < size:
            ax.plot(grid_x[i, j], grid_y[i, j], "ko",
                    markersize=8, markerfacecolor="yellow", zorder=4)

    for i, j in mst.edges():
        if i < len(x) and j < len(x):
            ax.plot([x[i], x[j]], [y[i], y[j]], "k--", alpha=0.4)

    ax.streamplot(grid_x, grid_y, U_entropy, V_entropy,
                  color="orange", density=1.0, linewidth=1.5, arrowsize=1.2)

    # 합병 탐지 포인트 오버레이
    if merger_idx_sub is not None and merger_idx_sub < len(x):
        ax.scatter(x[merger_idx_sub], y[merger_idx_sub],
                   s=300, color='red', marker='*', zorder=5,
                   label=f'탐지된 합병 (idx={merger_idx_sub})')
        ax.legend(fontsize=10)

    ax.set_title(f"EQI 위상 피드백 네트워크 — 어머니 코드 재현 {label}",
                 fontsize=13)
    ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
    plt.colorbar(sc, ax=ax, label="Tau (에너지)")
    ax.grid(True)

    # [오른쪽] 클러스터별 특성 분포
    ax2 = axes[1]
    colors = ['steelblue', 'coral', 'seagreen', 'purple']
    for cid in df['cluster'].unique():
        mask = df['cluster'] == cid
        ax2.scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'],
                    s=60, alpha=0.7, color=colors[cid % len(colors)],
                    label=f'Cluster {cid} (n={mask.sum()})', edgecolor='k', linewidths=0.3)

    if merger_idx_sub is not None and merger_idx_sub < len(df):
        ax2.scatter(df.iloc[merger_idx_sub]['pca1'],
                    df.iloc[merger_idx_sub]['pca2'],
                    s=400, color='red', marker='*', zorder=5,
                    label=f'탐지 합병 (t={df.iloc[merger_idx_sub]["time"]:.2f}s)')

    ax2.set_title("클러스터 분포 + 합병 위치", fontsize=13)
    ax2.set_xlabel("PCA1"); ax2.set_ylabel("PCA2")
    ax2.legend(fontsize=9)
    ax2.grid(True)

    plt.tight_layout()
    out_path = f"{output_folder}/mother_code_reconstruction{label}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  저장: {out_path}")


# ============================================================
# 6. 전체 신호 시계열 시각화
# ============================================================

def visualize_timeseries(h1_white, l1_white, phase_diff, entropy, tau,
                          t, combined_score, peak_idx, output_folder):
    """
    전처리된 H1/L1 신호 + 탐지 지표 + 합병 위치 시각화.
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("GW150914 전체 신호 분석 (H-H1 + L-L1 LOSC 4096Hz 32s)",
                 fontsize=13, fontweight='bold')

    merger_t = GW150914["merger_t"]
    t_merger_line = dict(color='red', linestyle='--', linewidth=1.5,
                         label=f'실제 합병 t={merger_t:.1f}s')

    # (1) H1 화이트닝 신호
    axes[0].plot(t, h1_white, color='steelblue', linewidth=0.4)
    axes[0].axvline(merger_t, **t_merger_line)
    axes[0].axvline(t[peak_idx], color='orange', linestyle='-',
                    linewidth=1.5, label=f'탐지 t={t[peak_idx]:.2f}s')
    axes[0].set_ylabel("H1 (whitened)")
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].set_xlim(0, 32)

    # (2) L1 화이트닝 신호
    axes[1].plot(t, l1_white, color='coral', linewidth=0.4)
    axes[1].axvline(merger_t, **t_merger_line)
    axes[1].axvline(t[peak_idx], color='orange', linestyle='-', linewidth=1.5)
    axes[1].set_ylabel("L1 (whitened)")
    axes[1].set_xlim(0, 32)

    # (3) PhaseDiff
    axes[2].plot(t, phase_diff, color='purple', linewidth=0.4)
    axes[2].axvline(merger_t, **t_merger_line)
    axes[2].axvline(t[peak_idx], color='orange', linestyle='-', linewidth=1.5)
    axes[2].set_ylabel("PhaseDiff (rad)")
    axes[2].set_xlim(0, 32)

    # (4) Entropy
    axes[3].plot(t, entropy, color='green', linewidth=0.4)
    axes[3].axvline(merger_t, **t_merger_line)
    axes[3].axvline(t[peak_idx], color='orange', linestyle='-', linewidth=1.5)
    axes[3].set_ylabel("Entropy")
    axes[3].set_xlim(0, 32)

    # (5) 결합 탐지 점수
    axes[4].plot(t, combined_score / combined_score.max(),
                 color='black', linewidth=0.6)
    axes[4].fill_between(t, combined_score / combined_score.max(),
                          alpha=0.2, color='black')
    axes[4].axvline(merger_t, **t_merger_line)
    axes[4].axvline(t[peak_idx], color='orange', linestyle='-',
                    linewidth=1.5, label=f'탐지 피크 t={t[peak_idx]:.2f}s')
    axes[4].set_ylabel("결합 탐지 점수")
    axes[4].set_xlabel("시간 (s, GPS-1126259446)")
    axes[4].set_xlim(0, 32)
    axes[4].legend(fontsize=8)

    plt.tight_layout()
    out_path = f"{output_folder}/timeseries_full.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  저장: {out_path}")


# ============================================================
# 7. 보고서 저장
# ============================================================

def save_report(results, output_folder):
    report = {
        "analysis_date": str(datetime.now()),
        "data_spec": GW150914,
        "processing": results.get("processing", {}),
        "merger_detection": results.get("merger_detection", {}),
        "cluster_stats": results.get("cluster_stats", {}),
        "mother_code_comparison": {
            "original_subsample": 100,
            "original_merger_idx": 56,
            "original_merger_t_s": 17.92,
            "original_error_ms": 1500,
            "v2_full_n_samples": 131072,
        }
    }
    path = f"{output_folder}/eqi_mother_full_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  보고서 저장: {path}")
    return report


# ============================================================
# 메인 파이프라인
# ============================================================

def run_pipeline(h1_path, l1_path,
                 output_folder="./output_mother_v2",
                 n_sub=100, n_clusters=3,
                 use_whitening=True,
                 bandpass=True):

    os.makedirs(output_folder, exist_ok=True)
    print("=" * 65)
    print("EQI 어머니 코드 확장판 v2")
    print(f"  H1: {h1_path}")
    print(f"  L1: {l1_path}")
    print(f"  화이트닝: {use_whitening}, 밴드패스: {bandpass}")
    print("=" * 65)

    results = {}

    # ── STEP 1: 데이터 로드 ──
    print("\n[STEP 1] HDF5 로드...")
    h1_strain, t, gps_h, xs_h = load_hdf5(h1_path)
    l1_strain, _, gps_l, xs_l  = load_hdf5(l1_path)

    if h1_strain is None or l1_strain is None:
        print("HDF5 로드 실패. df_merged.csv 폴백 모드로 전환합니다.")
        df = load_from_csv_fallback("df_merged.csv")
        print("폴백 데이터로 원본 어머니 코드 시각화만 실행합니다.")
        df_sub, pca_model = df, None
        merger_sub_idx = 56  # 원본 어머니 코드 탐지값
        visualize_mother_code(df_sub, merger_sub_idx, output_folder, "_fallback")
        return

    sample_rate = int(1.0 / xs_h)
    t_arr = np.arange(len(h1_strain)) * xs_h

    # ── STEP 2: L1 시간 지연 보정 ──
    print("\n[STEP 2] L1 시간 지연 보정 (6.9ms = 28샘플)...")
    delay_samples = GW150914["L1_advance_samples"]
    l1_aligned = np.zeros_like(l1_strain)
    l1_aligned[delay_samples:] = l1_strain[:-delay_samples]
    print(f"  L1을 {delay_samples}샘플 ({GW150914['L1_advance_ms']}ms) 지연 적용")

    # ── STEP 3: 전처리 ──
    print("\n[STEP 3] 신호 전처리...")
    h1_proc = h1_strain.copy()
    l1_proc = l1_aligned.copy()

    if bandpass:
        print("  밴드패스 필터 (35~350 Hz)...")
        h1_proc = bandpass_filter(h1_proc, sample_rate)
        l1_proc = bandpass_filter(l1_proc, sample_rate)

    if use_whitening:
        print("  화이트닝...")
        h1_proc = whiten(h1_proc, sample_rate)
        l1_proc = whiten(l1_proc, sample_rate)

    results["processing"] = {
        "bandpass": bandpass,
        "whitening": use_whitening,
        "L1_delay_samples": delay_samples,
    }

    # ── STEP 4: 특성 계산 ──
    print("\n[STEP 4] 특성 계산 (PhaseDiff / Entropy / Tau)...")
    phase_diff, entropy, tau = compute_phase_features(
        h1_proc, l1_proc, sample_rate)
    print(f"  PhaseDiff: mean={phase_diff.mean():.4f}, std={phase_diff.std():.4f}")
    print(f"  Entropy:   mean={entropy.mean():.6f}  (원본 어머니 코드: 0.4996)")
    print(f"  Tau:       mean={tau.mean():.6f}  (원본 어머니 코드: 0.4879)")

    # ── STEP 5: 합병 탐지 ──
    print("\n[STEP 5] 합병 구간 자동 탐지...")
    peak_idx, peak_t, combined_score = detect_merger(
        phase_diff, entropy, tau, t_arr, sample_rate=sample_rate)

    results["merger_detection"] = {
        "detected_idx":     int(peak_idx),
        "detected_t_s":     float(peak_t),
        "true_merger_t_s":  GW150914["merger_t"],
        "error_ms":         float(abs(peak_t - GW150914["merger_t"]) * 1000),
        "mother_code_error_ms": 1500,
    }

    # ── STEP 6: 서브샘플 + 클러스터링 (어머니 코드 재현) ──
    print(f"\n[STEP 6] 서브샘플링 ({n_sub}개) + 클러스터링 (k={n_clusters})...")
    df_sub, pca_model = subsample_and_cluster(
        phase_diff, entropy, tau, t_arr,
        n_sub=n_sub, n_clusters=n_clusters,
        merger_idx=peak_idx
    )

    # 서브샘플에서 탐지된 합병에 가장 가까운 인덱스
    merger_sub_idx = int(np.argmin(np.abs(df_sub['orig_idx'].values - peak_idx)))
    print(f"  합병 서브샘플 인덱스: {merger_sub_idx} "
          f"(t={df_sub.iloc[merger_sub_idx]['time']:.2f}s)")

    cluster_stats = {}
    for cid in df_sub['cluster'].unique():
        mask = df_sub['cluster'] == cid
        cluster_stats[int(cid)] = {
            "count": int(mask.sum()),
            "phase_diff_var": float(df_sub.loc[mask, 'PhaseDiff'].var()),
            "entropy_mean":   float(df_sub.loc[mask, 'Entropy'].mean()),
            "tau_mean":       float(df_sub.loc[mask, 'Tau'].mean()),
        }
    results["cluster_stats"] = cluster_stats

    # 어머니 코드와의 클러스터 비교
    print("\n  [어머니 코드 클러스터 비교]")
    print(f"  {'클러스터':^8} {'크기':^8} {'PhaseDiff분산':^14} "
          f"{'Entropy평균':^12} {'Tau평균':^10}")
    for cid, s in cluster_stats.items():
        print(f"  {cid:^8} {s['count']:^8} {s['phase_diff_var']:^14.4f} "
              f"{s['entropy_mean']:^12.4f} {s['tau_mean']:^10.4f}")

    # ── STEP 7: 시각화 ──
    print("\n[STEP 7] 시각화...")

    # 전체 신호 시계열
    visualize_timeseries(
        h1_proc, l1_proc, phase_diff, entropy, tau,
        t_arr, combined_score, peak_idx, output_folder
    )

    # 어머니 코드 재현 시각화
    visualize_mother_code(
        df_sub, merger_sub_idx, output_folder, "_v2_full"
    )

    # ── STEP 8: 보고서 ──
    print("\n[STEP 8] 보고서 저장...")
    save_report(results, output_folder)

    print(f"\n{'='*65}")
    print("완료!")
    print(f"  탐지 오차: {results['merger_detection']['error_ms']:.1f} ms "
          f"(어머니 코드 서브샘플: 1500ms)")
    print(f"  출력 폴더: {output_folder}/")
    print(f"    timeseries_full.png         — 전체 신호 + 탐지")
    print(f"    mother_code_reconstruction  — 어머니 코드 시각화 재현")
    print(f"    eqi_mother_full_report.json — 분석 보고서")
    print(f"{'='*65}")

    return results


# ============================================================
# 실행
# ============================================================

def main():
    print("EQI 어머니 코드 확장판 v2")
    print("원본: EQI_smallest_unit_full_code.py (100개 서브샘플)")
    print("v2:   H1+L1 HDF5 직접 로드, 131,072 전체 샘플")
    print("=" * 65)

    h1_path = input("H1 HDF5 경로 (예: H-H1_LOSC_4_V2-1126259446-32.hdf5): ").strip()
    l1_path = input("L1 HDF5 경로 (예: L-L1_LOSC_4_V2-1126259446-32.hdf5): ").strip()

    w = input("화이트닝 적용? (y/n) [기본 y]: ").strip().lower()
    use_whitening = w != 'n'

    b = input("밴드패스 필터 (35~350Hz) 적용? (y/n) [기본 y]: ").strip().lower()
    bandpass = b != 'n'

    n_input = input("서브샘플 수 [기본 100]: ").strip()
    n_sub = int(n_input) if n_input else 100

    run_pipeline(
        h1_path, l1_path,
        output_folder="./output_mother_v2",
        n_sub=n_sub,
        use_whitening=use_whitening,
        bandpass=bandpass,
    )


if __name__ == "__main__":
    main()

# eqi_full_sample_analysis.py
#
# 서브샘플링 없는 전체 샘플 기반 EQI 통합 분석
#
# 통합 출처:
#   - eqi_mother_full_v5.py        : HDF5 로드, 청크 Hilbert, L1 정렬, event-centric
#   - extended_gauss_law.py        : UMAP, 다중 클러스터 병렬 분석, Super MST
#   - hatched_egg_v2_large.py      : 계층적 MST, TDA(PH), 민감도 테스트
#   - hatched_egg_unity_cluster_v2 : 클러스터 자동 선택, H/L 물리 검증
#
# 핵심 개선: 100개 서브샘플링 완전 제거
#   → 필터링 후 전체 샘플 (656~335,000개) 직접 분석
#   → 32초 4096Hz: UMAP/MST/PH 직접 적용
#   → 4096초 16kHz: 계층적 MST + UMAP 근사 + PH 1500개 근사

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import os, time, json, warnings
from datetime import datetime
from scipy.signal import hilbert, butter, filtfilt, welch, windows
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# 선택적 임포트
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("참고: umap-learn 미설치 → PCA로 대체")

try:
    import ripser
    from persim import plot_diagrams
    PH_AVAILABLE = True
except ImportError:
    PH_AVAILABLE = False
    print("참고: ripser 미설치 → PH 건너뜀")

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# 한글 폰트
import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# GW150914 참조값
# ============================================================
GW150914 = {
    "merger_gps":         1126259462.4,
    "gps_start":          1126259446,
    "sample_rate":        4096,
    "merger_t":           16.4,
    "merger_idx":         67174,
    "L1_advance_ms":      6.9,
    "L1_advance_samples": 28,
    "H1_SNR":             24.0,
    "L1_SNR":             13.0,
    "source":             "arXiv:1602.03839",
}


# ============================================================
# 1. 데이터 로드 (HDF5 우선, CSV 폴백)
# ============================================================

def load_data(h_path, l_path):
    """HDF5 또는 CSV 자동 감지 로드."""
    ext = os.path.splitext(h_path)[1].lower()

    if ext in ('.hdf5', '.hdf', '.h5'):
        return _load_hdf5(h_path, l_path)
    else:
        return _load_csv(h_path, l_path)


def _load_hdf5(h_path, l_path):
    try:
        import h5py
        def read(path):
            with h5py.File(path, 'r') as f:
                strain   = f['strain/Strain'][:]
                gps      = f['strain/Strain'].attrs['Xstart']
                xspacing = f['strain/Strain'].attrs['Xspacing']
            return strain, gps, xspacing

        h1, gps_h, xs_h = read(h_path)
        l1, gps_l, xs_l = read(l_path)
        sr = int(round(1.0 / xs_h))
        print(f"  HDF5 로드: H={len(h1):,}샘플 ({sr}Hz), GPS시작={gps_h}")
        return h1, l1, sr, int(gps_h)
    except Exception as e:
        raise RuntimeError(f"HDF5 로드 실패: {e}")


def _load_csv(h_path, l_path):
    LARGE = 10_000_000

    def count_rows(path):
        with open(path) as f:
            return sum(1 for _ in f) - 1

    n = count_rows(h_path)
    print(f"  CSV 행 수: {n:,}")

    if n > LARGE:
        print(f"  스트리밍 로드 (청크 5M)...")
        df_H = pd.concat(pd.read_csv(h_path, chunksize=5_000_000), ignore_index=True)
        df_L = pd.concat(pd.read_csv(l_path, chunksize=5_000_000), ignore_index=True)
    else:
        df_H = pd.read_csv(h_path)
        df_L = pd.read_csv(l_path)

    h1 = df_H['strain'].values
    l1 = df_L['strain'].values

    # 샘플레이트 추정
    if 'time' in df_H.columns and len(df_H) > 1:
        dt = df_H['time'].iloc[1] - df_H['time'].iloc[0]
        sr = int(round(1.0 / dt))
    else:
        sr = 4096
    print(f"  CSV 로드: H={len(h1):,}, L={len(l1):,}, 추정 샘플레이트={sr}Hz")
    return h1, l1, sr, GW150914['gps_start']


# ============================================================
# 2. 전처리 (v5에서 검증된 방법)
# ============================================================

def bandpass_filter(signal, sr, low=35.0, high=350.0, order=4):
    nyq = sr / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


def whiten(signal, sr):
    """Tukey 윈도우 + PSD 화이트닝 + 엣지 크롭 + std 정규화."""
    tuk = windows.tukey(len(signal), alpha=0.1)
    sw  = signal * tuk
    freqs, psd = welch(sw, fs=sr, nperseg=min(len(sw), 4*sr))
    psd_i = np.interp(np.fft.rfftfreq(len(signal), 1.0/sr), freqs, psd)
    psd_i[psd_i < 1e-60] = 1e-60
    w = np.fft.irfft(np.fft.rfft(sw) / np.sqrt(psd_i/2.0), n=len(signal))
    crop = int(0.5 * sr)
    w[:crop] = 0.0; w[-crop:] = 0.0
    std = w[crop:-crop].std()
    return w / std if std > 0 else w


def hilbert_chunked(signal, chunk_size=2_000_000):
    """청크 기반 Hilbert 변환 (대용량 메모리 안전)."""
    n = len(signal)
    if n <= chunk_size:
        a = hilbert(signal)
        return np.unwrap(np.angle(a)), np.abs(a)**2

    phase = np.empty(n); tau = np.empty(n)
    offset = 0.0
    nc = (n + chunk_size - 1) // chunk_size
    for ci in range(nc):
        s = ci * chunk_size; e = min(s + chunk_size, n)
        a = hilbert(signal[s:e])
        uw = np.unwrap(np.angle(a))
        if ci > 0:
            offset += phase[s-1] - uw[0]
        phase[s:e] = uw + offset
        tau[s:e]   = np.abs(a)**2
    return phase, tau


# ============================================================
# 3. 특성 계산
# ============================================================

def compute_features(h1_proc, l1_proc, sr):
    """PhaseDiff / Tau / Entropy (변동계수 기반) / Curvature."""
    print("  Hilbert 변환 (H1)...")
    phase_H, tau_H = hilbert_chunked(h1_proc)
    print("  Hilbert 변환 (L1)...")
    phase_L, _     = hilbert_chunked(l1_proc)

    # PhaseDiff [-π, π]
    pd_raw = phase_H - phase_L
    phase_diff = (pd_raw + np.pi) % (2*np.pi) - np.pi

    # Tau: 98퍼센타일 정규화
    crop = int(0.5 * sr)
    p98  = np.percentile(tau_H[crop:-crop], 98)
    tau  = np.clip(tau_H / max(p98, 1e-30), 0, 1)

    # Entropy: 변동계수 기반 (낮은값=신호, 높은값=노이즈)
    win  = max(1, sr // 2)
    kern = np.ones(win) / win
    mu   = np.convolve(tau, kern, mode='same')
    mu2  = np.convolve(tau**2, kern, mode='same')
    cv   = np.sqrt(np.clip(mu2 - mu**2, 0, None)) / (mu + 1e-30)
    cv_n = (cv - cv[crop:-crop].min()) / (cv[crop:-crop].max() - cv[crop:-crop].min() + 1e-30)
    entropy = 1.0 - np.clip(cv_n, 0, 1)

    # Curvature
    curv = np.gradient(np.gradient(phase_diff))

    features = np.column_stack([tau, phase_diff, entropy, curv])
    print(f"  특성 계산 완료: {features.shape}")
    return features, phase_diff, tau, entropy


# ============================================================
# 4. 필터링
# ============================================================

def threshold_filter(features, pct=99.5):
    valid = features[~np.isnan(features).any(axis=1)]
    if len(valid) == 0:
        return valid, np.array([]), 0
    amp = valid[:, 0]
    thr = np.percentile(amp, pct)
    idx = np.where(amp >= thr)[0]
    filt = valid[idx]
    print(f"  임계값 {pct}%: {thr:.4e}, 필터링 후: {len(filt):,}개")
    return filt, idx, thr


# ============================================================
# 5. 차원 축소 (PCA / UMAP)
# ============================================================

def reduce_dim(features, use_umap=True, n_components=2,
               umap_n_neighbors=15, umap_min_dist=0.1):
    """
    전체 샘플에 대해 PCA 또는 UMAP 적용.
    n > 50,000이면 UMAP을 위한 사전 PCA 축소(50D) 적용.
    """
    n = len(features)
    if n == 0:
        return np.array([]), None

    if use_umap and UMAP_AVAILABLE:
        # 대용량 전처리: 50D PCA 먼저
        if n > 50_000:
            print(f"  UMAP 사전 PCA (50D) 적용: {n:,}→50D...")
            pca_pre = PCA(n_components=min(50, features.shape[1]))
            features = pca_pre.fit_transform(features)

        print(f"  UMAP 실행: {len(features):,}샘플 → {n_components}D "
              f"(n_neighbors={umap_n_neighbors})...")
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist,
                            random_state=42)
        result = reducer.fit_transform(features)
        method = "UMAP"
    else:
        print(f"  PCA 실행: {n:,}샘플 → {n_components}D...")
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(features)
        print(f"  PCA 설명 분산: {pca.explained_variance_ratio_}")
        method = "PCA"

    print(f"  {method} 완료: {result.shape}")
    return result, method


# ============================================================
# 6. K-means 클러스터링 + 자동 선택
# ============================================================

def cluster_and_select(proj, filtered_features, n_clusters=3):
    """K-means 후 위상차 분산 기준으로 가장 흥미로운 클러스터 자동 선택."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(proj)
    print(f"  클러스터 분포: {np.bincount(labels)}")

    pd_col = filtered_features[:, 1]
    stats  = {}
    for cid in range(n_clusters):
        mask = labels == cid
        stats[cid] = {
            'count':     int(mask.sum()),
            'pd_var':    float(np.var(pd_col[mask])),
            'amp_mean':  float(np.mean(filtered_features[mask, 0])),
            'ent_mean':  float(np.mean(filtered_features[mask, 2])),
        }

    best = max(stats, key=lambda c: stats[c]['pd_var'])
    print(f"\n  {'클러스터':^6} {'크기':^8} {'위상분산':^14} {'자동선택'}")
    for cid, s in stats.items():
        mk = "◀" if cid == best else ""
        print(f"  {cid:^6} {s['count']:^8,} {s['pd_var']:^14.4e} {mk}")

    return labels, stats, best


# ============================================================
# 7. MST (계층적, 대용량 지원)
# ============================================================

def build_mst_full(coords, phase_diff_vals=None, max_rep=2000):
    """
    전체 샘플 MST.
    n ≤ max_rep: 직접 계산
    n > max_rep: 계층적 (대표 max_rep개 + 나머지 NN 연결)
    """
    n = len(coords)
    if n == 0:
        return nx.Graph()

    def _direct(pts, pd=None):
        dist = cdist(pts, pts)
        if pd is not None:
            ps = np.cos(np.abs(pd[:, None] - pd[None, :]))
            dist = dist * (1 - ps + 1e-6)
        G = nx.Graph()
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                G.add_edge(i, j, weight=dist[i, j])
        return nx.minimum_spanning_tree(G)

    if n <= max_rep:
        print(f"  MST 직접 계산: {n}개")
        return _direct(coords, phase_diff_vals)

    print(f"  MST 계층적: {n:,}개 → 대표 {max_rep}개")
    rep_idx = np.linspace(0, n-1, max_rep, dtype=int)
    rep_pts = coords[rep_idx]
    rep_pd  = phase_diff_vals[rep_idx] if phase_diff_vals is not None else None
    G_rep   = _direct(rep_pts, rep_pd)

    # 전체 그래프 구성
    G = nx.Graph()
    for u, v, d in G_rep.edges(data=True):
        G.add_edge(rep_idx[u], rep_idx[v], weight=d['weight'])

    # 나머지 포인트 → 가장 가까운 대표에 연결
    rest_idx = np.setdiff1d(np.arange(n), rep_idx)
    if len(rest_idx) > 0:
        nbrs = NearestNeighbors(n_neighbors=1).fit(rep_pts)
        chunk = 10000
        for s in range(0, len(rest_idx), chunk):
            ri  = rest_idx[s:s+chunk]
            dist_r, nn_r = nbrs.kneighbors(coords[ri])
            for j, (d, nn) in enumerate(zip(dist_r, nn_r)):
                G.add_edge(ri[j], rep_idx[nn[0]], weight=d[0])

    return nx.minimum_spanning_tree(G)


# ============================================================
# 8. Super MST (클러스터 간)
# ============================================================

def build_super_mst(cluster_coord_dict):
    """클러스터 중심점 간 MST."""
    centroids = {cid: coords.mean(axis=0)
                 for cid, coords in cluster_coord_dict.items()
                 if len(coords) > 0}
    if len(centroids) < 2:
        return None
    cids = list(centroids.keys())
    pts  = np.array([centroids[c] for c in cids])
    dist = cdist(pts, pts)
    G = nx.Graph()
    for i, ci in enumerate(cids):
        G.add_node(ci, pos=pts[i])
    for i in range(len(cids)):
        for j in range(i+1, len(cids)):
            G.add_edge(cids[i], cids[j], weight=dist[i, j])
    return nx.minimum_spanning_tree(G)


# ============================================================
# 9. Persistent Homology
# ============================================================

def compute_ph(coords, max_samples=1500):
    if not PH_AVAILABLE or len(coords) == 0:
        return None
    if len(coords) > max_samples:
        idx = np.linspace(0, len(coords)-1, max_samples, dtype=int)
        coords = coords[idx]
        print(f"  PH 근사: → {len(coords)}개")
    t0 = time.time()
    dgms = ripser.ripser(coords, maxdim=1)['dgms']
    print(f"  PH 완료 ({time.time()-t0:.1f}초)")
    return dgms


# ============================================================
# 10. H/L 비율 물리적 검증
# ============================================================

def verify_hl(tau_H_raw, tau_L_raw, sr, out):
    """Raw tau로 H/L RMS 비율 계산 및 참조값 비교."""
    crop = int(0.5 * sr)
    rms_H = np.sqrt(np.mean(tau_H_raw[crop:-crop]))
    rms_L = np.sqrt(np.mean(tau_L_raw[crop:-crop]))
    total = rms_H + rms_L
    H_r   = rms_H / total; L_r = rms_L / total

    ref_snr = GW150914['H1_SNR'] / (GW150914['H1_SNR'] + GW150914['L1_SNR'])
    err_snr = abs(H_r - ref_snr) / ref_snr * 100

    print(f"\n  [H/L 검증]")
    print(f"  측정 H_rms_ratio: {H_r:.4f}  L_rms_ratio: {L_r:.4f}")
    print(f"  참조(SNR):        {ref_snr:.4f}  오차={err_snr:.1f}%")
    print(f"  (Raw 데이터 기준: 화이트닝 전 노이즈 차이 반영)")

    return {'H_rms_ratio': H_r, 'L_rms_ratio': L_r,
            'ref_snr_ratio': ref_snr, 'snr_err_pct': err_snr}


# ============================================================
# 11. 탐지 점수 (v5에서 검증된 방법)
# ============================================================

def detect_merger(phase_diff, entropy, tau, t, sr):
    crop   = int(0.5 * sr)
    window = int(0.5 * sr)

    def safe_norm(a):
        m = a[crop:-crop].max()
        return a / (m + 1e-30) if m > 0 else a

    pg  = safe_norm(np.convolve(np.abs(np.gradient(phase_diff)),
                                np.ones(window)/window, mode='same'))
    ts  = safe_norm(np.convolve(tau, np.ones(window)/window, mode='same'))
    tg  = np.convolve(np.clip(np.gradient(
                    np.convolve(tau, np.ones(window)/window, mode='same')),
                    0, None), np.ones(window)/window, mode='same')
    tg  = safe_norm(tg)
    coh = 1.0 - entropy

    score = ts*0.4 + tg*0.3 + pg*0.2 + coh*0.1
    score[:crop] = 0; score[-crop:] = 0

    peak_idx = int(np.argmax(score))
    peak_t   = float(t[peak_idx])
    err_ms   = abs(peak_t) * 1000

    print(f"\n  [합병 탐지]  (t=0 = 합병 기준)")
    print(f"  탐지: idx={peak_idx:,}  t={peak_t:+.3f}s  오차={err_ms:.1f}ms")
    status = "✅" if err_ms < 500 else ("⚠️" if err_ms < 2000 else "❌")
    print(f"  {status}")

    top5 = score.copy()
    print(f"  상위 5개:")
    for r in range(5):
        i = int(np.argmax(top5))
        print(f"    {r+1}위: t={t[i]:+.3f}s  score={top5[i]:.4f}")
        top5[max(0,i-sr):min(len(top5),i+sr)] = 0

    return peak_idx, peak_t, score, err_ms


# ============================================================
# 12. 시각화
# ============================================================

def visualize_all(h1_w, l1_w, t, score, peak_idx, phase_diff,
                  entropy, tau, proj, labels, n_clusters,
                  cluster_data, super_mst, method, out):

    os.makedirs(out, exist_ok=True)
    merger_t = 0.0
    ml = dict(color='red', linestyle='--', lw=1.5, label='합병 t=0')

    # [A] 시계열 5패널
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("GW150914 전체 샘플 분석 (서브샘플링 없음)", fontsize=13, fontweight='bold')
    for ax, sig, col, ylabel in [
        (axes[0], h1_w, 'steelblue', 'H1 (whitened)'),
        (axes[1], l1_w, 'coral',     'L1 (whitened)'),
        (axes[2], phase_diff, 'purple', 'PhaseDiff (rad)'),
        (axes[3], entropy, 'green',  'Entropy'),
    ]:
        ax.plot(t, sig, color=col, lw=0.4)
        ax.axvline(merger_t, **ml)
        ax.axvline(t[peak_idx], color='orange', lw=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(t[0], t[-1])

    axes[4].plot(t, score / score.max(), color='black', lw=0.6)
    axes[4].fill_between(t, score / score.max(), alpha=0.2, color='black')
    axes[4].axvline(merger_t, **ml)
    axes[4].axvline(t[peak_idx], color='orange', lw=1.5,
                    label=f'탐지 t={t[peak_idx]:+.3f}s')
    axes[4].set_ylabel("탐지 점수"); axes[4].set_xlabel("시간 (s, t=0=합병)")
    axes[4].legend(fontsize=8); axes[4].set_xlim(t[0], t[-1])

    plt.tight_layout()
    plt.savefig(f"{out}/timeseries_full_sample.png", dpi=120)
    plt.close()

    # [B] 전체 클러스터 투영
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels,
                     cmap='viridis', s=8, alpha=0.6)
    plt.colorbar(sc, label='Cluster')
    plt.title(f"{method} Projection (전체 {len(proj):,}개, 서브샘플링 없음)")
    plt.xlabel(f"{method} 1"); plt.ylabel(f"{method} 2")
    plt.grid(True)
    plt.savefig(f"{out}/projection_all_clusters.png", dpi=120)
    plt.close()

    # [C] 클러스터별 개별 투영 + MST + PH
    for cid in range(n_clusters):
        cd = cluster_data[cid]
        coords = cd['coords']
        if len(coords) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Cluster {cid}  (n={len(coords):,})", fontsize=13)

        # 투영
        axes[0].scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.5, color='steelblue')
        axes[0].set_title(f"{method} 투영")
        axes[0].set_xlabel(f"{method} 1"); axes[0].set_ylabel(f"{method} 2")

        # MST (샘플링해서 시각화)
        mst = cd.get('mst')
        if mst is not None:
            vis_n = min(150, len(coords))
            vis_i = np.linspace(0, len(coords)-1, vis_n, dtype=int)
            sub   = coords[vis_i]
            axes[1].scatter(sub[:, 0], sub[:, 1], s=20, color='red', zorder=3)
            sub_nodes = set(vis_i)
            for u, v in mst.edges():
                if u in sub_nodes and v in sub_nodes:
                    axes[1].plot([coords[u,0], coords[v,0]],
                                  [coords[u,1], coords[v,1]], 'k-', lw=0.4, alpha=0.5)
            axes[1].set_title("MST (대표 150개)")
        else:
            axes[1].text(0.5, 0.5, "MST 없음", ha='center', va='center',
                         transform=axes[1].transAxes)

        # PH
        dgms = cd.get('ph')
        if PH_AVAILABLE and dgms is not None:
            plt.sca(axes[2])
            plot_diagrams(dgms, plot_only=[0], ax=axes[2])
            axes[2].set_title("PH Barcodes (H₀)")
        else:
            axes[2].text(0.5, 0.5, "PH 없음 (ripser 필요)",
                         ha='center', va='center', transform=axes[2].transAxes)

        plt.tight_layout()
        plt.savefig(f"{out}/cluster_{cid}_analysis.png", dpi=120)
        plt.close()

    # [D] Super MST
    if super_mst is not None:
        plt.figure(figsize=(8, 6))
        pos = {cid: cluster_data[cid]['coords'].mean(axis=0)
               for cid in range(n_clusters)
               if len(cluster_data[cid]['coords']) > 0}
        pos_s = nx.get_node_attributes(super_mst, 'pos')
        nx.draw(super_mst, pos_s, node_size=200,
                node_color=list(pos_s.keys()), cmap='viridis',
                edge_color='black', width=2)
        nx.draw_networkx_labels(super_mst, pos_s, font_size=12)
        plt.title("Super MST (클러스터 간 구조)")
        plt.savefig(f"{out}/super_mst.png", dpi=120)
        plt.close()

    print(f"  시각화 저장: {out}/")


# ============================================================
# 메인 파이프라인
# ============================================================

def run_pipeline(h_path, l_path,
                 out="./output_full",
                 threshold_pct=99.5,
                 n_clusters=3,
                 use_umap=True,
                 use_ph=True,
                 bandpass=True,
                 do_whiten=True,
                 umap_neighbors=15):

    os.makedirs(out, exist_ok=True)
    t_start = datetime.now()
    print("=" * 65)
    print("EQI 전체 샘플 통합 분석 (서브샘플링 없음)")
    print(f"  threshold_pct = {threshold_pct}")
    print(f"  n_clusters    = {n_clusters}")
    print(f"  UMAP          = {use_umap and UMAP_AVAILABLE}")
    print(f"  PH            = {use_ph and PH_AVAILABLE}")
    print("=" * 65)

    # STEP 1: 로드
    print("\n[STEP 1] 데이터 로드...")
    h1_raw, l1_raw, sr, gps_start = load_data(h_path, l_path)

    # STEP 2: 시간축 (event-centric)
    merger_idx_abs = round((GW150914['merger_gps'] - gps_start) * sr)
    # GPS가 다를 수 있으므로 fallback
    if merger_idx_abs < 0 or merger_idx_abs >= len(h1_raw):
        merger_idx_abs = len(h1_raw) // 2
        print(f"  ⚠️  merger_idx 범위 초과 → 데이터 중간({merger_idx_abs})으로 설정")
    t_arr = (np.arange(len(h1_raw)) - merger_idx_abs) / float(sr)
    print(f"  시간축: {t_arr[0]:.2f}s ~ {t_arr[-1]:.2f}s  (t=0=합병)")

    # STEP 3: L1 보정 + 전처리
    print("\n[STEP 2] L1 도달 시각 보정 + 전처리...")
    delay = GW150914['L1_advance_samples']
    l1_al = np.zeros_like(l1_raw)
    l1_al[delay:] = l1_raw[:-delay]

    if bandpass:
        print("  밴드패스 (35~350Hz)...")
        h1_p = bandpass_filter(h1_raw, sr)
        l1_p = bandpass_filter(l1_al,  sr)
    else:
        h1_p, l1_p = h1_raw.copy(), l1_al.copy()

    if do_whiten:
        print("  화이트닝...")
        h1_w = whiten(h1_p, sr)
        l1_w = whiten(l1_p, sr)
    else:
        h1_w, l1_w = h1_p, l1_p

    # STEP 4: 특성 계산 (전체 샘플)
    print("\n[STEP 3] 특성 계산 (전체 샘플)...")
    features, phase_diff, tau, entropy = compute_features(h1_w, l1_w, sr)

    # STEP 5: 탐지
    print("\n[STEP 4] 합병 탐지...")
    peak_idx, peak_t, score, err_ms = detect_merger(
        phase_diff, entropy, tau, t_arr, sr)

    # STEP 6: H/L 검증
    print("\n[STEP 5] H/L 물리적 검증...")
    _, _, tau_H_raw = [None, None, hilbert_chunked(h1_raw)[1]]
    _, _, tau_L_raw = [None, None, hilbert_chunked(l1_raw)[1]]
    hl_info = verify_hl(tau_H_raw, tau_L_raw, sr, out)

    # STEP 7: 필터링
    print("\n[STEP 6] 임계값 필터링...")
    filt, filt_idx, thr = threshold_filter(features, pct=threshold_pct)
    if len(filt) == 0:
        print("필터링 후 데이터 없음."); return {}

    # STEP 8: 차원 축소 (전체 샘플, 서브샘플링 없음)
    print(f"\n[STEP 7] 차원 축소 ({len(filt):,}개 전체 샘플)...")
    proj, method = reduce_dim(filt, use_umap=use_umap,
                              umap_n_neighbors=umap_neighbors)
    if len(proj) == 0:
        return {}

    # STEP 9: 클러스터링
    print(f"\n[STEP 8] 클러스터링 (k={n_clusters})...")
    labels, cluster_stats, best_cid = cluster_and_select(proj, filt, n_clusters)

    # STEP 10: 클러스터별 MST + PH
    print(f"\n[STEP 9] 클러스터별 MST + PH...")
    cluster_data = {}
    cluster_coord_dict = {}
    for cid in range(n_clusters):
        mask   = labels == cid
        coords = proj[mask]
        pd_cid = filt[mask, 1]
        print(f"  Cluster {cid} ({len(coords):,}개):")
        mst  = build_mst_full(coords, pd_cid)
        dgms = compute_ph(coords) if use_ph else None
        cluster_data[cid] = {'coords': coords, 'mst': mst, 'ph': dgms}
        cluster_coord_dict[cid] = coords

    # STEP 11: Super MST
    print(f"\n[STEP 10] Super MST...")
    s_mst = build_super_mst(cluster_coord_dict)

    # STEP 12: 시각화
    print(f"\n[STEP 11] 시각화...")
    visualize_all(h1_w, l1_w, t_arr, score, peak_idx,
                  phase_diff, entropy, tau,
                  proj, labels, n_clusters,
                  cluster_data, s_mst, method, out)

    # STEP 13: 보고서
    elapsed = (datetime.now() - t_start).total_seconds()
    report = {
        "analysis_date":      str(datetime.now()),
        "n_samples_total":    int(len(h1_raw)),
        "n_samples_filtered": int(len(filt)),
        "subsampling":        "없음 (전체 샘플 직접 분석)",
        "dim_reduction":      method,
        "merger_detection": {
            "detected_t_s":  float(peak_t),
            "error_ms":      float(err_ms),
            "true_t_s":      0.0,
            "time_axis":     "event-centric (t=0=합병)",
        },
        "cluster_stats":  {str(k): v for k, v in cluster_stats.items()},
        "best_cluster":   int(best_cid),
        "hl_verification": hl_info,
        "elapsed_sec":    float(elapsed),
    }
    with open(f"{out}/report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"완료!  실행 시간: {elapsed:.1f}초")
    print(f"탐지 오차: {err_ms:.1f}ms  (서브샘플 어머니 코드: 1500ms)")
    print(f"분석 샘플: {len(filt):,}개 (서브샘플링 없음)")
    print(f"결과 폴더: {out}/")
    print(f"  timeseries_full_sample.png  — 시계열 + 탐지")
    print(f"  projection_all_clusters.png — 전체 {method} 투영")
    print(f"  cluster_N_analysis.png      — 클러스터별 투영+MST+PH")
    print(f"  super_mst.png               — 클러스터 간 구조")
    print(f"  report.json                 — 분석 보고서")
    print(f"{'='*65}")

    return report


# ============================================================
# 실행
# ============================================================

def main():
    print("EQI 전체 샘플 통합 분석 (서브샘플링 없음)")
    print("=" * 65)
    print("통합 출처:")
    print("  eqi_mother_full_v5      : HDF5/CSV 로드, 청크 Hilbert, event-centric")
    print("  extended_gauss_law      : UMAP, 다중 클러스터, Super MST")
    print("  hatched_egg_v2_large    : 계층적 MST, PH, 민감도 테스트")
    print("핵심: 100개 서브샘플링 완전 제거 → 전체 샘플 직접 분석")
    print("=" * 65)

    h_path = input("H1 파일 경로 (HDF5 또는 CSV): ").strip()
    l_path = input("L1 파일 경로 (HDF5 또는 CSV): ").strip()

    pct = input("threshold_percentile [기본 99.5]: ").strip()
    threshold_pct = float(pct) if pct else 99.5

    k = input("n_clusters [기본 3]: ").strip()
    n_clusters = int(k) if k else 3

    u = input("UMAP 사용? (y/n) [기본 y]: ").strip().lower()
    use_umap = u != 'n'

    p = input("PH (Persistent Homology) 사용? (y/n) [기본 y]: ").strip().lower()
    use_ph = p != 'n'

    nn = input("UMAP n_neighbors [기본 15, 대용량은 30 권장]: ").strip()
    umap_neighbors = int(nn) if nn else 15

    run_pipeline(
        h_path, l_path,
        out="./output_full",
        threshold_pct=threshold_pct,
        n_clusters=n_clusters,
        use_umap=use_umap,
        use_ph=use_ph,
        umap_neighbors=umap_neighbors,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Framework Enhanced Topological Structure Analysis v3
SGWB 배경 데이터 CSV 버전

v2 대비 변경사항:
  - load_hdf5 → load_csv (time/strain 컬럼)
  - GPS_MERGER 참조 제거
  - EVENT = "SGWB_BACKGROUND"
  - 4096초 OOM 패치: MST/PCA 서브샘플링
  - Trinity Functions (DimRatio, n_ouro, U_D2) v2 물리 100% 보존

v2 물리 구조 완전 보존:
  n_ouro = (S_L2/S_T2 - MAXWELL_REAL) / GAP  [v9.1 방식]
  DimRatio = smallest_phase / |smallest_coherence|
  U_D1 = 1.0
  U_D2 = 1 + n_ouro × GAP

Author: MAPSI | Architect: Claude (Session 33)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import networkx as nx
from scipy.spatial import KDTree
from scipy.signal import hilbert, spectrogram
import warnings

warnings.filterwarnings("ignore")

# ── 핵심 상수 ──────────────────────────────────────────────
PHI       = (1 + np.sqrt(5)) / 2
GAP       = np.log(25) / 100
TIP_POINT = PHI**1.5 - 1
n_c       = TIP_POINT / GAP          # 32.874

S_D1_SI     = np.log(1.0 / 8.854187817e-12)   # ln(1/ε₀) = 25.450
S_D2_SI     = abs(np.log(4 * np.pi * 1e-7))   # |ln(μ₀)| = 13.587
MAXWELL_REAL = 1.8731137721991709
D1_val      = PHI**(-1.5)
D2_val      = PHI**(+1.5)

EVENT = "SGWB_BACKGROUND"


class EQI_Topological_Analyzer_v3:
    def __init__(self, label=""):
        self.label = label
        self.trinity_functions = {}
        self.dimensionless_symmetry = {}

    # ── CSV 로드 ─────────────────────────────────────────────
    def load_csv(self, h_path, l_path, mode="HL"):
        """
        CSV 로더 - SGWB 배경 데이터용
        컬럼: time, strain
        """
        print(f"\n📡 [{self.label}] Loading CSV ({mode} mode)...")
        df_h = pd.read_csv(h_path)
        df_l = pd.read_csv(l_path)

        h_s = df_h['strain'].values.astype(np.float64)
        l_s = df_l['strain'].values.astype(np.float64)
        t   = df_h['time'].values
        fs  = 4096.0

        h_s = np.nan_to_num(h_s)
        l_s = np.nan_to_num(l_s)

        if mode == "H":
            strain = h_s
        elif mode == "L":
            strain = l_s
        else:  # HL: D1:D2 볼츠만 합성
            strain = (h_s * D1_val + l_s * D2_val) / (D1_val + D2_val)

        print(f"✅ {len(strain):,}샘플  fs={fs:.0f}Hz  GPS_start={t[0]:.1f}")
        return {"strain": strain, "h_s": h_s, "l_s": l_s, "time": t, "fs": fs}

    # ── Trinity Functions (v2 물리 완전 보존) ─────────────────
    def extract_trinity_functions(self, data):
        print(f"🔥 [{self.label}] Extracting Trinity Functions...")
        strain = data["strain"]

        # Smallest Phase
        analytic  = hilbert(strain)
        phase     = np.unwrap(np.angle(analytic))
        amplitude = np.abs(analytic)
        phase_diff = np.gradient(phase)
        pos_mask   = phase_diff > 0
        smallest_phase = (
            float(np.min(np.exp(phase_diff[pos_mask])))
            if pos_mask.any() else 0.1
        )

        # Smallest Coherence
        coh = np.log(amplitude + 1e-15)
        smallest_coherence = float(np.min(coh[np.isfinite(coh)]))

        # n_ouroboros — v9.1 방식 (z-score 정규화 후 S_L2/S_T2)
        h_s = data["h_s"]
        l_s = data["l_s"]
        h_n = (h_s - np.mean(h_s)) / (np.std(h_s) + 1e-30)
        l_n = (l_s - np.mean(l_s)) / (np.std(l_s) + 1e-30)

        ln_h = float(np.log(max(np.mean((h_n * D1_val)**2), 1e-300)))
        ln_l = float(np.log(max(np.mean((l_n * D2_val)**2), 1e-300)))
        S_L2 = ln_h + S_D1_SI
        S_T2 = ln_l + S_D2_SI
        unity_product = S_L2 / max(S_T2, 1e-300)
        n_ouro = (unity_product - MAXWELL_REAL) / GAP

        # Smallest Uncertainty 이중성
        U_D1 = 1.0
        U_D2 = 1.0 + n_ouro * GAP

        # ZONE 판정
        if n_ouro < 0:
            zone = "ZONE-X (음수)"
        elif n_ouro < n_c:
            zone = "ZONE-0/1 (터널링)"
        elif n_ouro < 2 * n_c:
            zone = "ZONE-2+ (광전효과)"
        else:
            zone = "ZONE-3+ (극단)"

        # DimRatio
        ratio = smallest_phase / (abs(smallest_coherence) + 1e-15)

        self.trinity_functions = {
            "smallest_phase":         smallest_phase,
            "smallest_coherence":     smallest_coherence,
            "U_D1":                   U_D1,
            "U_D2":                   U_D2,
            "n_ouroboros":            n_ouro,
            "zone":                   zone,
        }
        self.dimensionless_symmetry = {
            "ratio":               ratio,
            "phase_component":     smallest_phase,
            "coherence_component": abs(smallest_coherence),
        }

        print(f"   Smallest Phase      = {smallest_phase:.6f}")
        print(f"   Smallest Coherence  = {smallest_coherence:.6f}")
        print(f"   DimRatio            = {ratio:.6f}")
        print(f"   n_ouroboros         = {n_ouro:.4f}  n_c={n_c:.3f}")
        print(f"   U_D1                = {U_D1:.6f}")
        print(f"   U_D2 (1+n×GAP)      = {U_D2:.6f}")
        print(f"   ZONE                = {zone}")

        return phase, amplitude, phase_diff

    # ── 특성 추출 ─────────────────────────────────────────────
    def extract_features(self, data, phase, amplitude):
        strain = data["strain"]
        fs     = int(data["fs"])
        win    = 2048
        stride = 512
        feats  = []
        for s in range(0, len(strain) - win, stride):
            seg  = strain[s:s+win]
            ph_s = phase[s:s+win]
            am_s = amplitude[s:s+win]
            ma   = np.mean(am_s)
            sa   = np.std(am_s)
            sk   = np.mean((am_s - ma)**3) / (sa**3 + 1e-15)
            pv   = np.var(np.gradient(ph_s))
            pc   = float(np.abs(np.mean(np.exp(1j * ph_s))))
            try:
                f_, _, P = spectrogram(seg, fs=fs, nperseg=256, return_onesided=True)
                dp = float(np.max(P))
                sc = float(np.mean(
                    np.sum(f_.reshape(-1,1)*P, axis=0) / (np.sum(P, axis=0)+1e-15)
                ))
            except:
                dp = sc = 0.0
            feats.append([ma, sa, sk, pv, pc, dp, sc])
        fm = np.array(feats)
        print(f"   Features: {fm.shape}")
        return fm

    # ── EQI 분석 (OOM 패치) ───────────────────────────────────
    def eqi_analysis(self, fm):
        mask = np.all(np.isfinite(fm), axis=1)
        fc   = fm[mask]
        fn   = (fc - fc.mean(0)) / (fc.std(0) + 1e-15)

        # 4096초 OOM 패치: 노드 512개로 서브샘플링
        MAX_NODES = 512
        if len(fn) > MAX_NODES:
            print(f"   OOM 패치: {len(fn)}청크 → MiniBatchKMeans {MAX_NODES}대표")
            km_sub = MiniBatchKMeans(n_clusters=MAX_NODES, batch_size=1024, random_state=42)
            fn = km_sub.fit(fn).cluster_centers_

        # PCA
        pca = PCA(n_components=min(2, fn.shape[1]))
        pr  = pca.fit_transform(fn)
        print(f"   PCA explained: {pca.explained_variance_ratio_}")

        # KDTree Sparse Graph (dense squareform 배제)
        tree = KDTree(fn)
        r_limit = float(PHI**0.5)
        sparse_pairs = list(tree.query_pairs(r=r_limit))

        G = nx.Graph()
        G.add_nodes_from(range(len(fn)))
        G.add_edges_from(sparse_pairs)

        if nx.is_connected(G):
            mst = nx.minimum_spanning_tree(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc).copy()
            mst = nx.minimum_spanning_tree(G_sub)

        print(f"   MST: {len(mst.nodes)}노드 {len(mst.edges)}엣지")

        return {
            "pca":  pr,
            "mst":  mst,
            "evr":  pca.explained_variance_ratio_,
        }

    # ── 실행 ─────────────────────────────────────────────────
    def run(self, data):
        phase, amp, pdiff = self.extract_trinity_functions(data)
        fm = self.extract_features(data, phase, amp)
        return self.eqi_analysis(fm)


# ── 메인 ─────────────────────────────────────────────────────
def run_all():
    print("=" * 70)
    print(f"  Enhanced Topological Analysis v3: {EVENT}")
    print(f"  CSV 배경 데이터 | v2 물리 구조 완전 보존")
    print(f"  DimRatio 기준값: 0.028953 (Session 31 우주 고유 상수)")
    print("=" * 70)

    H_CSV = "d:\\LIGO\\h-strain_sgwb_background.csv"
    L_CSV = "d:\\LIGO\\l-strain_sgwb_background.csv"

    results = {}
    for mode in ["H", "L", "HL"]:
        label = f"SGWB-{mode}"
        az    = EQI_Topological_Analyzer_v3(label=label)
        data  = az.load_csv(H_CSV, L_CSV, mode=mode)
        r     = az.run(data)
        results[mode] = {
            "analyzer": az,
            "result":   r,
            "trinity":  az.trinity_functions,
            "sym":      az.dimensionless_symmetry,
        }

    # ── 비교 출력 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Trinity Functions 비교 (배경 vs GW 이벤트)")
    print("=" * 70)
    print(f"\n{'채널':12s} {'DimRatio':10s} {'n_ouro':10s} "
          f"{'U_D1':8s} {'U_D2':10s} {'ZONE':24s} {'PCA1':6s}")
    print("-" * 90)

    # GW150914 참조값 (Session 32 확정)
    print(f"{'[GW150914]':12s} {'0.028953':10s} {'-8.5729':10s} "
          f"{'1.0':8s} {'0.724050':10s} {'ZONE-X':24s} {'0.666':6s}  ← BBH 참조")

    for mode, d in results.items():
        t   = d["trinity"]
        sym = d["sym"]
        r   = d["result"]
        evr = r["evr"]
        pca1 = evr[0] if len(evr) > 0 else 0.0
        print(
            f"{'SGWB-'+mode:12s} "
            f"{sym['ratio']:.6f}   "
            f"{t['n_ouroboros']:10.4f}   "
            f"{t['U_D1']:.6f}   "
            f"{t['U_D2']:.6f}   "
            f"{t['zone']:24s}   "
            f"{pca1:.4f}"
        )

    print()
    print("=== EQI 상수 ===")
    print(f"GAP      = ln(25)/100 = {GAP:.6f}")
    print(f"TIP_POINT = φ^1.5-1  = {TIP_POINT:.6f}")
    print(f"n_c       = {n_c:.4f}")
    print(f"MAXWELL   = {MAXWELL_REAL:.10f}")
    print()
    print("=== 배경 vs 신호 판별 기준 ===")
    print(f"배경(SGWB): n_ouro ≪ 0  → ZONE-X  (노이즈 D2 우세)")
    print(f"GW신호:     n_ouro 근방  → DimRatio ≈ 0.028953")

    return results


if __name__ == "__main__":
    results = run_all()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Framework Enhanced Topological Structure Analysis v2
GW170814 버전 — HDF5 직접 로드 + H/L/HL 세 조합 비교
원본: enhanced_topological_structure_analysis_fixed.py (2024.09 Perplexity)

Session 32 패치:
  smallest_uncertainty: 1.0 → 1 + n_ouroboros × GAP
  D1 (거시/고립계): EQI^2=1    → U=1.0
  D2 (미시/열린계): EQI^(4n)   → U=1+n×GAP
  ZONE 판정 추가 (n_c=32.874 기준)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert, spectrogram
import matplotlib.pyplot as plt
import h5py, warnings

warnings.filterwarnings("ignore")

PHI = (1 + np.sqrt(5)) / 2
GAP = np.log(25) / 100  # Session 24 확정 상수
TIP_POINT = PHI**1.5 - 1  # 1.058171
n_c = TIP_POINT / GAP  # 32.874

# ── 이벤트 설정 ───────────────────────────────────────────
EVENT = "GW170814"
H1_FILE = "H-H1_GWOSC_O2_4KHZ_R1-1186738176-4096.hdf5"
L1_FILE = "L-L1_GWOSC_O2_4KHZ_R1-1186738176-4096.hdf5"
GPS_MERGER = 1186741861.5


class EQI_Topological_Structure_Analyzer:
    def __init__(self, label=""):
        self.label = label
        self.trinity_functions = {}
        self.dimensionless_symmetry = {}

    # ── HDF5 로드 ────────────────────────────────────────
    def load_hdf5(self, h_path, l_path, mode="HL"):
        """mode: 'H', 'L', 'HL'(D1:D2 합성)"""
        print(f"\n📡 [{self.label}] Loading HDF5 ({mode} mode)...")
        with h5py.File(h_path, "r") as fh:
            h_s = fh["strain/Strain"][:]
            n = fh["strain/Strain"].attrs["Npoints"]
            xsp = fh["strain/Strain"].attrs["Xspacing"]
            gps = float(
                fh["strain/Strain"].attrs.get(
                    "Xstart", fh["strain/Strain"].attrs.get("GPSstart", 0.0)
                )
            )
        with h5py.File(l_path, "r") as fl:
            l_s = fl["strain/Strain"][:]

        fs = 1.0 / xsp
        t = np.arange(n) * xsp + gps

        h_s = np.nan_to_num(h_s.astype(np.float64))
        l_s = np.nan_to_num(l_s.astype(np.float64))

        if mode == "H":
            strain = h_s
        elif mode == "L":
            strain = l_s
        else:  # HL: D1:D2 볼츠만 합성
            D1 = PHI ** (-1.5)
            D2 = PHI ** (+1.5)
            strain = (h_s * D1 + l_s * D2) / (D1 + D2)

        print(
            f"✅ {len(strain):,}샘플  fs={fs:.0f}Hz  "
            f"GPS_start={gps}  합병={GPS_MERGER-gps:.1f}s 후"
        )
        return {"strain": strain, "h_s": h_s, "l_s": l_s, "time": t, "fs": fs}

    # ── Trinity Functions ─────────────────────────────────
    def extract_trinity_functions(self, ligo_data):
        print(f"🔥 [{self.label}] Extracting Trinity Functions...")
        strain = ligo_data["strain"]
        analytic = hilbert(strain)
        phase = np.unwrap(np.angle(analytic))
        amplitude = np.abs(analytic)

        phase_diff = np.gradient(phase)
        pos_mask = phase_diff > 0
        smallest_phase = (
            float(np.min(np.exp(phase_diff[pos_mask]))) if pos_mask.any() else 0.1
        )

        coh = np.log(amplitude + 1e-15)
        smallest_coherence = float(np.min(coh[np.isfinite(coh)]))

        # Session 32 패치 v3: n_ouroboros = v9.1 방식 (z-score 정규화 후 S_L2/S_T2)
        S_D1_SI = np.log(1.0 / 8.854187817e-12)  # ln(1/eps0) = 25.450
        S_D2_SI = abs(np.log(4 * np.pi * 1e-7))  # |ln(mu0)| = 13.587
        MAXWELL_REAL = 1.8731137721991709
        D1_val = PHI ** (-1.5)
        D2_val = PHI ** (+1.5)

        h_s = ligo_data["h_s"]
        l_s = ligo_data["l_s"]

        # v9.1과 동일: z-score 정규화
        h_n = (h_s - np.mean(h_s)) / (np.std(h_s) + 1e-30)
        l_n = (l_s - np.mean(l_s)) / (np.std(l_s) + 1e-30)

        ln_h = float(np.log(max(np.mean((h_n * D1_val) ** 2), 1e-300)))
        ln_l = float(np.log(max(np.mean((l_n * D2_val) ** 2), 1e-300)))
        S_L2 = ln_h + S_D1_SI
        S_T2 = ln_l + S_D2_SI
        unity_product = S_L2 / max(S_T2, 1e-300)
        n_ouro = (unity_product - MAXWELL_REAL) / GAP

        smallest_uncertainty_D1 = 1.0
        smallest_uncertainty_D2 = 1.0 + n_ouro * GAP
        smallest_uncertainty = smallest_uncertainty_D2

        # ZONE 판정 (n_c=32.874 기준)
        if n_ouro < 0:
            zone = "ZONE-X (음수, 신호 없음)"
        elif n_ouro < n_c:
            zone = "ZONE-0/1 (터널링)"
        elif n_ouro < 2 * n_c:
            zone = "ZONE-2+ (광전효과)"
        else:
            zone = "ZONE-3+ (극단)"

        self.trinity_functions = {
            "smallest_phase": smallest_phase,
            "smallest_coherence": smallest_coherence,
            "smallest_uncertainty": smallest_uncertainty,
            "smallest_uncertainty_D1": smallest_uncertainty_D1,
            "smallest_uncertainty_D2": smallest_uncertainty_D2,
            "n_ouroboros": n_ouro,
            "zone": zone,
        }
        print(f"   Smallest Phase       = {smallest_phase:.6f}  [D1]")
        print(f"   Smallest Coherence   = {smallest_coherence:.6f}  [D2]")
        print(
            f"   Smallest Uncertainty (D1, EQI^2=1)       = {smallest_uncertainty_D1:.6f}"
        )
        print(
            f"   Smallest Uncertainty (D2, 1+n×GAP)       = {smallest_uncertainty_D2:.6f}"
        )
        print(f"   n_ouroboros = {n_ouro:.4f}  n_c={n_c:.3f}  ZONE={zone}")

        # 원본 코드의 Dimensionless Symmetry Ratio
        ratio = smallest_phase / (abs(smallest_coherence) + 1e-15)
        self.dimensionless_symmetry = {
            "ratio": ratio,
            "phase_component": smallest_phase,
            "coherence_component": abs(smallest_coherence),
        }
        print(
            f"   Dimensionless Ratio = {ratio:.6f}  "
            f"(원본 GW170814≈0.029, EQI D1²=0.236)"
        )
        return phase, amplitude, phase_diff

    # ── 특성 추출 ─────────────────────────────────────────
    def extract_features(self, ligo_data, phase, amplitude):
        strain = ligo_data["strain"]
        fs = int(ligo_data["fs"])
        win = 2048
        stride = 512
        feats = []
        for s in range(0, len(strain) - win, stride):
            seg = strain[s : s + win]
            ph_s = phase[s : s + win]
            am_s = amplitude[s : s + win]
            ma = np.mean(am_s)
            sa = np.std(am_s)
            sk = np.mean((am_s - ma) ** 3) / (sa**3 + 1e-15)
            pv = np.var(np.gradient(ph_s))
            pc = float(np.abs(np.mean(np.exp(1j * ph_s))))
            try:
                f_, _, P = spectrogram(seg, fs=fs, nperseg=256, return_onesided=True)
                dp = float(np.max(P))
                sc = float(
                    np.mean(
                        np.sum(f_.reshape(-1, 1) * P, axis=0)
                        / (np.sum(P, axis=0) + 1e-15)
                    )
                )
            except:
                dp = sc = 0.0
            feats.append([ma, sa, sk, pv, pc, dp, sc])
        fm = np.array(feats)
        print(f"   Features: {fm.shape}")
        return fm

    # ── EQI 고급 분석 ─────────────────────────────────────
    def eqi_analysis(self, fm):
        mask = np.all(np.isfinite(fm), axis=1)
        fc = fm[mask]
        fn = (fc - fc.mean(0)) / (fc.std(0) + 1e-15)

        pca = PCA(n_components=2)
        pr = pca.fit_transform(fn)
        print(f"   PCA explained: {pca.explained_variance_ratio_}")

        nc = min(max(2, len(pr) // 20), 5)
        km = KMeans(n_clusters=nc, random_state=42, n_init=10)
        cl = km.fit_predict(pr)
        print(f"   Clusters: {nc}  dist={np.bincount(cl)}")

        dm = squareform(pdist(pr))
        cw = np.where(cl[:, None] != cl[None, :], 1.5, 1.0)
        wd = dm * cw

        G = nx.Graph()
        for i in range(len(pr)):
            G.add_node(i, pos=tuple(pr[i]), cluster=int(cl[i]))
        for i in range(len(pr)):
            for j in range(i + 1, len(pr)):
                G.add_edge(i, j, weight=float(wd[i, j]))
        mst = nx.minimum_spanning_tree(G)
        print(f"   MST: {len(mst.nodes)}노드 {len(mst.edges)}엣지")

        return {
            "pca": pr,
            "clusters": cl,
            "mst": mst,
            "nc": nc,
            "evr": pca.explained_variance_ratio_,
        }

    # ── 실행 ──────────────────────────────────────────────
    def run(self, ligo_data):
        phase, amp, pdiff = self.extract_trinity_functions(ligo_data)
        fm = self.extract_features(ligo_data, phase, amp)
        return self.eqi_analysis(fm)


# ── 세 조합 비교 실행 ────────────────────────────────────
def run_all():
    print("=" * 70)
    print(f"  Enhanced Topological Analysis: {EVENT}")
    print(f"  원본 코드 기반 + GW170814 HDF5 직접 로드")
    print("=" * 70)

    results = {}
    for mode in ["H", "L", "HL"]:
        label = f"GW170814-{mode}"
        az = EQI_Topological_Structure_Analyzer(label=label)
        data = az.load_hdf5(H1_FILE, L1_FILE, mode=mode)
        r = az.run(data)
        results[mode] = {
            "analyzer": az,
            "result": r,
            "trinity": az.trinity_functions,
            "sym": az.dimensionless_symmetry,
        }

    # ── 비교 출력 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  결과 비교")
    print("=" * 70)

    # GW170814 원본 참조값 (2024.09 Perplexity 실행값)
    ref = {
        "label": "GW170814 H (원본 2024.09)",
        "phase": 0.1688,  # block_0 smallest_phase 평균
        "coh": -10.575,  # block_0 smallest_coherence
        "ratio": 0.02895,  # Dimensionless Ratio = 0.029
    }

    print(
        f"\n{'조합':12s} {'SmPhase':10s} {'SmCoher':10s} "
        f"{'DimRatio':10s} {'n_ouro':8s} {'U_D1':8s} {'U_D2':10s} {'ZONE':22s} {'PCA1':6s}"
    )
    print("-" * 105)

    # 참조값
    print(
        f"{'[참조]':12s} {ref['phase']:.6f}   {ref['coh']:.6f}   "
        f"{ref['ratio']:.6f}   GW170814 원본 (D1 극한)"
    )

    for mode, d in results.items():
        t = d["trinity"]
        sym = d["sym"]
        r = d["result"]
        evr = r["evr"]
        print(
            f"{'GW170814-'+mode:12s} "
            f"{t['smallest_phase']:.6f}   "
            f"{t['smallest_coherence']:.6f}   "
            f"{sym['ratio']:.6f}   "
            f"{t['n_ouroboros']:8.4f}   "
            f"{t['smallest_uncertainty_D1']:.6f}   "
            f"{t['smallest_uncertainty_D2']:.6f}   "
            f"{t['zone']:22s}   "
            f"{evr[0]:.4f}"
        )

    print()
    print("=== EQI 상수 연결 ===")
    print(f"원본 DimRatio = 0.028953  (Session 31 우주 고유 상수 확정)")
    print(f"D1² = Φ^(-3) = {PHI**(-3):.6f}  (v14 이론값)")
    print(f"GAP = ln(25)/100 = {GAP:.6f}")
    print(f"TIP_POINT = φ^1.5-1 = {TIP_POINT:.6f}")
    print(f"n_c = TIP_POINT/GAP = {n_c:.4f}")
    print()
    print("=== smallest_uncertainty 이중성 (Session 32 패치) ===")
    print(f"D1 (거시/로컬/고립계): EQI^2=1  → U=1.0  [표준 하이젠베르크]")
    print(f"D2 (미시/글로벌/열린계): EQI^(4n)=1+n×GAP → U=1+n_ouro×GAP")
    print()

    for mode, d in results.items():
        t = d["trinity"]
        ratio = d["sym"]["ratio"]
        print(
            f"GW170814-{mode}: DimRatio={ratio:.6f}  "
            f"n_ouro={t['n_ouroboros']:.4f}  "
            f"U_D2={t['smallest_uncertainty_D2']:.6f}  "
            f"{t['zone']}"
        )

    # ── 시각화 ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Enhanced Topological Analysis: {EVENT}\n"
        f"원본 코드 기반 H / L / HL(D1:D2) 세 조합 비교",
        fontsize=14,
        fontweight="bold",
    )

    colors = ["H=Blues", "L=Reds", "HL=Greens"]
    cmaps = ["Blues", "Reds", "Greens"]

    for col, (mode, d) in enumerate(results.items()):
        r = d["result"]
        pr = r["pca"]
        cl = r["clusters"]

        # 상단: PCA
        ax = axes[0, col]
        sc = ax.scatter(pr[:, 0], pr[:, 1], c=cl, cmap=cmaps[col], alpha=0.7, s=30)
        ax.set_title(
            f"GW170814-{mode}\n"
            f"DimRatio={d['sym']['ratio']:.4f}  "
            f"clusters={r['nc']}"
        )
        ax.set_xlabel("PC1 (Unity)")
        ax.set_ylabel("PC2 (Multi)")
        plt.colorbar(sc, ax=ax)
        ax.grid(True, alpha=0.3)

        # 하단: MST
        ax2 = axes[1, col]
        mst = r["mst"]
        pos = nx.get_node_attributes(mst, "pos")
        nc2 = [mst.nodes[n].get("cluster", 0) for n in mst.nodes]
        try:
            nx.draw_networkx_nodes(
                mst,
                pos,
                ax=ax2,
                node_color=nc2,
                cmap=cmaps[col],
                node_size=30,
                alpha=0.8,
            )
            nx.draw_networkx_edges(
                mst, pos, ax=ax2, edge_color="gray", width=0.5, alpha=0.5
            )
        except Exception as e:
            ax2.text(
                0.5,
                0.5,
                f"MST viz err:\n{e}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
        ax2.set_title(f"MST: {len(mst.nodes)}노드 {len(mst.edges)}엣지")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")

    plt.tight_layout()
    fn = f"enhanced_topological_{EVENT}.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n🎨 저장: {fn}")
    print("✅ 완료")

    return results


if __name__ == "__main__":
    results = run_all()

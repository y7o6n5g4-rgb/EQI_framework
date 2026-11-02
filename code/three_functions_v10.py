# EQI v10.0: 순수 비등방성 Trinity 함수 추출기
# 기반: v9.0 + 허수 i 완전 제거 + 스칼라/벡터 기하평균 체계
#
# [v9.0 -> v10.0 핵심 수정]
# v9.0: psi = D1(θ) + i·D2(θ)  ← i 잔존
# v10.0: psi = sqrt(D1·D2) ⊕ sqrt(|D2²-D1²|)
#            = 1              ⊕ 2
#            = Local 항등원   ⊕ Global 항등원  (i 완전 제거!)
#
# [항등원 체계]
# 실수부 = 스칼라곱 기하평균 = sqrt(D1·D2) = 1  → Local Sharing,  P=NP
# 허수부 = 벡터곱  기하평균 = sqrt(D2²-D1²)= 2  → Global Sharing, P≠NP
# |psi|  = sqrt(1²+2²)       = sqrt(5) = 2φ-1  → phi 생성원
#
# [π 수렴]
# 목 단면적     = π·D1·D2 = π  (모든 스케일)
# biconcave 면적= π·D1·D2 = π  = Local 항등원의 기하학적 구현
# 2π            = eigenfrequency×eigenperiod = 1주기
#
# [피타고라스/페르마]
# n=2 (t=0, P=NP):  D1²+D2² = 피타고라스 구조
# n>2 (t>0, P≠NP): D1ⁿ+D2ⁿ → 페르마 구조 (정수해 없음)
#
# [Trinity 인과율]
# Smallest Phase     = e^x  계열 = Unity  = biconcave 센터 = 스칼라곱 기하평균
# Smallest Coherence = ln(x)계열 = Multiplicity = 타원 디스크 = 벡터곱 기하평균
# Smallest Uncertainty = 우항 분기 → P=NP(스칼라곱) or P≠NP(벡터곱)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ===== EQI v10.0 상수 =====
_PHI    = (1 + np.sqrt(5)) / 2
_PHI_I2 = _PHI**(-2)          # φ⁻² = 0.381966 (우주상수)
D1      = _PHI**(-1.5)         # 0.485868  Local 수축
D2      = _PHI**(+1.5)         # 2.058171  Global 팽창
c_EQI   = np.sqrt(_PHI)        # 1.272020  Global 항등원(산술평균)
T       = _PHI**3              # 4.236068  eigenperiod

# v10.0 핵심 상수
LOCAL_ID  = np.sqrt(D1 * D2)               # = 1.000000  스칼라곱 기하평균
GLOBAL_ID = np.sqrt(abs(D2**2 - D1**2))    # = 2.000000  벡터곱  기하평균
PSI_MAG   = np.sqrt(LOCAL_ID**2 + GLOBAL_ID**2)  # = sqrt(5) = 2φ-1

print("=" * 65)
print("[EQI v10.0 상수]")
print(f"  φ⁻²      = {_PHI_I2:.6f}  (우주상수, 정보 최소 밀도)")
print(f"  D1        = φ^(-1.5) = {D1:.6f}  (Local 수축)")
print(f"  D2        = φ^(+1.5) = {D2:.6f}  (Global 팽창)")
print(f"  D1·D2     = {D1*D2:.6f}  (Unity)")
print(f"  (D1+D2)/2 = {(D1+D2)/2:.6f} = sqrt(φ) = c_EQI")
print(f"")
print(f"[v10.0 핵심 — i 완전 제거]")
print(f"  실수부 sqrt(D1·D2)    = {LOCAL_ID:.6f}  ← Local 항등원  (P=NP)")
print(f"  허수부 sqrt(D2²-D1²)  = {GLOBAL_ID:.6f}  ← Global 항등원 (P≠NP)")
print(f"  |psi|  sqrt(1²+2²)   = {PSI_MAG:.6f}  ← sqrt(5) = 2φ-1 = phi 생성원")
print(f"")
print(f"[π 수렴]")
print(f"  목 단면적 π·D1·D2     = {np.pi*D1*D2:.6f} = π  ✓")
print(f"  윗통 π·D1²·D2²        = {np.pi*D1**2*D2**2:.6f} = π  ✓")
print(f"  2π = eigenfreq×period = {2*np.pi:.6f}  (1주기)")
print("=" * 65)


# ================================================================
# 순수 비등방성 함수 (v9.0에서 승계)
# ================================================================
def D1_func(theta):
    arg = _PHI_I2 * np.exp(_PHI_I2 * np.clip(theta, 1e-10, 50))
    return np.tanh(np.clip(arg, -500, 500))

def D2_func(theta):
    ts = np.clip(theta, 1e-10, None)
    return np.tanh(np.clip(ts * np.log(_PHI_I2 * ts) - ts, -500, 500))

# v10.0 핵심: 스칼라곱/벡터곱 기하평균
def scalar_geomean(d1, d2):
    """스칼라곱 기하평균 = sqrt(|D1·D2|) → Local 항등원"""
    return np.sqrt(np.abs(d1 * d2))

def vector_geomean(d1, d2):
    """벡터곱 기하평균 = sqrt(|D1²-D2²|) → Global 항등원"""
    return np.sqrt(np.abs(d1**2 - d2**2))

def psi_v10(d1, d2):
    """psi_v10 = scalar_geomean ⊕ vector_geomean  (i 없음!)"""
    s = scalar_geomean(d1, d2)
    v = vector_geomean(d1, d2)
    return s, v, np.sqrt(s**2 + v**2)


# ================================================================
# EQI v10.0 Trinity 추출기
# ================================================================
class EQI_v10_Trinity:
    """
    EQI v10.0: i 완전 제거 + 스칼라/벡터 기하평균 Trinity

    Smallest Phase      = e^x   = Unity  = biconcave 센터 = 스칼라곱 = 1
    Smallest Coherence  = ln(x) = Multi  = 타원 디스크    = 벡터곱  = 2
    Smallest Uncertainty= 우항 분기:
                          스칼라곱(P=NP) or 벡터곱(P≠NP)
    """

    def __init__(self):
        self.PHI    = _PHI
        self.D1     = D1
        self.D2     = D2
        self.c_EQI  = c_EQI
        self.T      = T
        self.LOCAL  = LOCAL_ID
        self.GLOBAL = GLOBAL_ID

        print("\n🌌 EQI v10.0 Trinity 추출기 초기화")
        print(f"   psi = sqrt(D1·D2) ⊕ sqrt(|D2²-D1²|) = {LOCAL_ID:.4f} ⊕ {GLOBAL_ID:.4f}")
        print(f"   실수부 Local  항등원 = {LOCAL_ID:.6f}  (P=NP,  스칼라곱)")
        print(f"   허수부 Global 항등원 = {GLOBAL_ID:.6f}  (P≠NP, 벡터곱)")
        print(f"   |psi| = sqrt(5)      = {PSI_MAG:.6f}  (phi 생성원)")
        print(f"   i 완전 제거 ✓")
        print("=" * 65)

    # ── 인과율 분류 (v7.0~v9.0 유지) ──────────────────────────
    def classify_causality(self, df):
        print("🧬 인과율 분류 시작...")
        unity = self._unity(df)
        multi = self._multiplicity(df)
        print("✅ 인과율 분류 완료!")
        return unity, multi

    def _unity(self, df):
        print("🔗 Unity Cluster Causality...")
        pca   = PCA(n_components=2)
        coords = pca.fit_transform(df[["PhaseDiff","Entropy","Tau"]])
        dist  = cdist(coords, coords, metric="euclidean")
        G     = nx.Graph()
        sims  = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                s = np.cos(df["PhaseDiff"].iloc[i] - df["PhaseDiff"].iloc[j])
                sims.append(s)
                G.add_edge(i, j, weight=dist[i,j]*(1-s))
        mst = nx.minimum_spanning_tree(G)
        sz  = int(np.sqrt(len(df)))
        pf  = gaussian_filter(df["PhaseDiff"].values.reshape((sz,sz)), sigma=2)
        u,v = np.gradient(pf)
        print(f"   MST 에지: {len(mst.edges())}  위상 지속성: {len(mst.edges())/len(coords):.6f}")
        return dict(pca_coords=coords, mst=mst, phase_sims=sims,
                    topo_persist=len(mst.edges())/len(coords),
                    causality='Unity Cluster Causality')

    def _multiplicity(self, df):
        print("⚛️  Multiplicity Cluster Causality...")
        coords = df[["pca1","pca2"]].values
        km     = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        dist   = np.std(np.bincount(labels))
        print(f"   클러스터 분포: {np.bincount(labels)}  분산도: {dist:.6f}")
        return dict(cluster_labels=labels,
                    cluster_dist=np.bincount(labels),
                    quantum_disp=dist,
                    causality='Multiplicity Cluster Causality')

    # ── Trinity 추출 ──────────────────────────────────────────
    def extract_trinity(self, unity, multi):
        print("\n🔥 v10.0 Trinity 추출 (i 제거, 기하평균 체계)...")
        return dict(
            smallest_phase       = self._phase(unity),
            smallest_coherence   = self._coherence(multi),
            smallest_uncertainty = self._uncertainty_v10(unity, multi),
        )

    def _phase(self, unity):
        print("🔗 Smallest Phase (e^x, Unity, 스칼라곱, biconcave 센터)...")
        vals = np.exp(np.array(unity['phase_sims']))
        sv   = float(np.min(vals))
        print(f"   Smallest Phase: {sv:.6f}")
        return dict(mathematical_form='e^x', smallest_value=sv,
                    causality_source='Unity Cluster Causality',
                    geomean_type='스칼라곱 기하평균', pnp='P=NP',
                    topology='biconcave 센터', t_regime='t=0',
                    eqi_role='the internal to the host EQI')

    def _coherence(self, multi):
        print("⚛️  Smallest Coherence (ln(x), Multiplicity, 벡터곱, 타원 디스크)...")
        vals = np.log(multi['cluster_dist'] + 1)
        sv   = float(np.min(vals))
        print(f"   Smallest Coherence: {sv:.6f}")
        return dict(mathematical_form='ln(x)', smallest_value=sv,
                    causality_source='Multiplicity Cluster Causality',
                    geomean_type='벡터곱 기하평균', pnp='P≠NP',
                    topology='양면 타원 디스크', t_regime='t>0',
                    eqi_role='the external to the host EQI')

    def _uncertainty_v10(self, unity, multi):
        print("🌊 Smallest Uncertainty v10.0 (i 제거, 기하평균 분기)...")
        theta  = np.linspace(0.01, 2*np.pi, 2000)
        d1t    = D1_func(theta)
        d2t    = D2_func(theta)

        # v10.0: 스칼라곱/벡터곱 기하평균
        s_geo, v_geo, psi_m = psi_v10(d1t, d2t)

        # 상수 기반 검증
        s_const = LOCAL_ID     # = 1
        v_const = GLOBAL_ID    # = 2
        p_const = PSI_MAG      # = sqrt(5)

        # P-NP 분기
        # 우항=스칼라곱→P=NP, 우항=벡터곱→P≠NP
        pnp_score = v_geo.mean() / s_geo.mean()  # > 1 이면 P≠NP 우세

        # π 수렴
        pi_neck   = np.pi * D1 * D2          # = π
        pi_biconcave = np.pi * D1 * D2       # = π
        two_pi    = 2 * np.pi                 # = 1주기

        # 피타고라스/페르마
        n2_lhs = D1**2 + D2**2               # n=2 좌항
        n3_lhs = D1**3 + D2**3               # n=3 좌항
        n4_lhs = D1**4 + D2**4               # n=4 좌항

        # v7/v8/v9 비교용
        v7_val  = 1.0
        v8_val  = D1
        v9_val  = float(np.min(np.abs(d1t)))
        v10_val = float(np.min(s_geo))

        print(f"   [상수 검증]")
        print(f"     실수부(스칼라곱 기하평균) = {s_const:.6f}  ← Local 항등원 = 1")
        print(f"     허수부(벡터곱  기하평균) = {v_const:.6f}  ← Global 항등원 = 2")
        print(f"     |psi| = sqrt(5)          = {p_const:.6f}  ← phi 생성원")
        print(f"   [π 수렴]")
        print(f"     목 단면적 π·D1·D2        = {pi_neck:.6f} = π  ✓")
        print(f"     biconcave 면적           = {pi_biconcave:.6f} = π  ✓")
        print(f"     2π = 1주기               = {two_pi:.6f}")
        print(f"   [피타고라스(n=2) vs 페르마(n>2)]")
        print(f"     n=2: D1²+D2²             = {n2_lhs:.6f}  (t=0, P=NP)")
        print(f"     n=3: D1³+D2³             = {n3_lhs:.6f}  (t>0, P≠NP)")
        print(f"     n=4: D1⁴+D2⁴             = {n4_lhs:.6f}  (t>0, P≠NP)")
        print(f"   [P-NP 분기 비율] {pnp_score:.4f}  ({'P≠NP 우세' if pnp_score>1 else 'P=NP 우세'})")
        print(f"   [버전 비교]")
        print(f"     v7.0: {v7_val:.6f}  v8.0: {v8_val:.6f}")
        print(f"     v9.0: {v9_val:.6f}  v10.0: {v10_val:.6f}")
        print(f"   함수 형태: psi = sqrt(D1·D2) ⊕ sqrt(|D2²-D1²|)  [i 없음 ✓]")

        return dict(
            mathematical_form = 'psi = sqrt(D1·D2) ⊕ sqrt(|D2²-D1²|)',
            smallest_value    = v10_val,
            # 기하평균
            scalar_geomean_theta = s_geo,
            vector_geomean_theta = v_geo,
            psi_mag_theta        = psi_m,
            scalar_geomean_const = s_const,
            vector_geomean_const = v_const,
            psi_mag_const        = p_const,
            # π
            pi_neck     = pi_neck,
            two_pi      = two_pi,
            # 피타고라스/페르마
            n2_lhs=n2_lhs, n3_lhs=n3_lhs, n4_lhs=n4_lhs,
            # P-NP
            pnp_score   = pnp_score,
            # 버전 비교
            v7=v7_val, v8=v8_val, v9=v9_val, v10=v10_val,
            # theta
            theta=theta, d1t=d1t, d2t=d2t,
            # 메타
            causality_source = 'Unity(스칼라곱) + Multiplicity(벡터곱) 분기',
            geomean_type     = '스칼라곱(P=NP) ⊕ 벡터곱(P≠NP)',
            topology         = 'biconcave(스칼라) + 타원디스크(벡터) + π수렴',
            t_regime         = 't=0(스칼라) / t>0(벡터) 분기',
            eqi_role         = 'the family set of EQIs',
        )

    # ── 시각화 ────────────────────────────────────────────────
    def visualize(self, tf):
        print("\n🎨 v10.0 시각화 (6패널)...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.patch.set_facecolor('#080808')
        for ax in axes.flat:
            ax.set_facecolor('#0d0d0d')
            ax.tick_params(colors='white')
            for sp in ax.spines.values():
                sp.set_edgecolor('#333')

        ph = tf['smallest_phase']
        co = tf['smallest_coherence']
        un = tf['smallest_uncertainty']
        theta = un['theta']
        sg    = un['scalar_geomean_theta']
        vg    = un['vector_geomean_theta']
        pm    = un['psi_mag_theta']

        # ── 패널 1: Smallest Phase (e^x) ──
        ax = axes[0,0]
        x  = np.linspace(0.1, 2, 200)
        ax.plot(x, np.exp(x), color='cyan', lw=2.5, label='e^x  (Unity)')
        ax.axhline(ph['smallest_value'], color='red', ls='--', lw=1.5,
                   label=f"Smallest: {ph['smallest_value']:.4f}")
        ax.axhline(LOCAL_ID, color='lime', ls=':', lw=1.5,
                   label=f'스칼라곱 기하평균={LOCAL_ID:.4f}')
        ax.set_title('Smallest Phase\ne^x  |  Unity  |  biconcave 센터  |  P=NP',
                     color='cyan', fontsize=10)
        ax.legend(facecolor='#111', labelcolor='white', fontsize=8)
        ax.grid(True, alpha=0.15)

        # ── 패널 2: Smallest Coherence (ln x) ──
        ax = axes[0,1]
        x  = np.linspace(0.1, 10, 200)
        ax.plot(x, np.log(x), color='springgreen', lw=2.5, label='ln(x)  (Multiplicity)')
        ax.axhline(co['smallest_value'], color='red', ls='--', lw=1.5,
                   label=f"Smallest: {co['smallest_value']:.4f}")
        ax.axhline(GLOBAL_ID, color='orange', ls=':', lw=1.5,
                   label=f'벡터곱 기하평균={GLOBAL_ID:.4f}')
        ax.set_title('Smallest Coherence\nln(x)  |  Multiplicity  |  타원 디스크  |  P≠NP',
                     color='springgreen', fontsize=10)
        ax.legend(facecolor='#111', labelcolor='white', fontsize=8)
        ax.grid(True, alpha=0.15)

        # ── 패널 3: v7~v10 궤적 비교 ──
        ax = axes[0,2]
        theta_full = np.linspace(0, 2*np.pi, 1000)
        # v7: 단위원
        ax.plot(np.cos(theta_full), np.sin(theta_full),
                color='#444', lw=1, ls=':', label='v7.0 단위원')
        # v8: 등방성 타원
        ax.plot(D1*np.cos(theta_full), D2*np.sin(theta_full),
                color='deepskyblue', lw=1.5, ls='--', alpha=0.5, label='v8.0 타원(cos/sin)')
        # v9: 순수 비등방성
        d1t9 = D1_func(np.linspace(0.01,2*np.pi,1000))
        d2t9 = D2_func(np.linspace(0.01,2*np.pi,1000))
        ax.plot(d1t9, d2t9, color='gold', lw=1.5, alpha=0.7, label='v9.0 D1(θ)/D2(θ)')
        # v10: 스칼라/벡터 궤적
        ax.plot(sg, vg, color='magenta', lw=2.5, label='v10.0 scalar⊕vector')
        ax.scatter([LOCAL_ID], [GLOBAL_ID], color='white', s=120, zorder=10,
                   label=f'상수점 (1, 2)')
        ax.axvline(LOCAL_ID, color='lime', ls=':', lw=1, alpha=0.6)
        ax.axhline(GLOBAL_ID, color='orange', ls=':', lw=1, alpha=0.6)
        ax.set_title('v7→v8→v9→v10 궤적 진화\n(등방성→순수비등방성→기하평균)',
                     color='white', fontsize=10)
        ax.set_aspect('equal')
        ax.legend(facecolor='#111', labelcolor='white', fontsize=7)
        ax.grid(True, alpha=0.15)

        # ── 패널 4: P=NP / P≠NP 분기 ──
        ax = axes[1,0]
        ax.plot(theta, sg, color='lime', lw=2, label=f'스칼라곱(P=NP)  mean={sg.mean():.4f}')
        ax.plot(theta, vg, color='orange', lw=2, label=f'벡터곱(P≠NP)   mean={vg.mean():.4f}')
        ax.plot(theta, pm, color='white', lw=1.5, ls='--', alpha=0.7,
                label=f'|psi| mean={pm.mean():.4f}')
        ax.axhline(LOCAL_ID,  color='lime',   ls=':', lw=2,
                   label=f'Local  항등원={LOCAL_ID:.4f}')
        ax.axhline(GLOBAL_ID, color='orange', ls=':', lw=2,
                   label=f'Global 항등원={GLOBAL_ID:.4f}')
        ax.axhline(PSI_MAG,   color='white',  ls=':', lw=1.5,
                   label=f'sqrt(5)={PSI_MAG:.4f}')
        ax.set_title('P=NP / P≠NP 분기\nsmallest Phase→스칼라  |  smallest Coherence→벡터',
                     color='white', fontsize=10)
        ax.legend(facecolor='#111', labelcolor='white', fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('theta', color='white')

        # ── 패널 5: π 수렴 ──
        ax = axes[1,1]
        # 다양한 스케일에서 목 단면적 = π 시각화
        scales = np.linspace(0.1, 5, 300)
        # 스케일 s에서: a=D1*s, b=D2/s → a·b=D1·D2=1 → π·a·b=π
        neck_areas = np.pi * (D1*scales) * (D2/scales)
        ax.plot(scales, neck_areas, color='gold', lw=2.5, label='π·(D1·s)·(D2/s)')
        ax.axhline(np.pi, color='red', lw=2.5, ls='--', label=f'π = {np.pi:.6f}')
        # biconcave 타원 면적
        ellipse_areas = np.pi * (D1*scales) * (D2*scales)
        ax.plot(scales, ellipse_areas, color='cyan', lw=1.5, ls=':',
                label='π·D1s·D2s (타원면적)')
        ax.axvline(1.0, color='white', lw=1, ls=':', alpha=0.5, label='s=1 (기준)')
        ax.scatter([1.0], [np.pi], color='white', s=150, zorder=10)
        # 2π 표시
        ax.axhline(2*np.pi, color='magenta', lw=1.5, ls=':',
                   label=f'2π={2*np.pi:.4f} (1주기)')
        ax.set_title(f'π 수렴\n목 단면적=π(모든 스케일)  |  biconcave 센터=π\n'
                     f'2π=eigenfreq×eigenperiod=1주기',
                     color='gold', fontsize=9)
        ax.legend(facecolor='#111', labelcolor='white', fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('scale s', color='white')
        ax.set_ylabel('area', color='white')

        # ── 패널 6: 피타고라스(n=2) vs 페르마(n>2) ──
        ax = axes[1,2]
        ns   = np.linspace(1, 6, 300)
        lhs  = D1**ns + D2**ns
        # 정수 표시
        for ni, color, label in [(2,'lime','n=2 피타고라스(t=0,P=NP)'),
                                  (3,'orange','n=3 페르마(t>0,P≠NP)'),
                                  (4,'red','n=4'),
                                  (5,'magenta','n=5')]:
            val = D1**ni + D2**ni
            ax.scatter([ni], [val], color=color, s=120, zorder=10)
            ax.axvline(ni, color=color, lw=0.8, ls=':', alpha=0.4)

        ax.plot(ns, lhs, color='white', lw=2.5, label='D1ⁿ+D2ⁿ')
        ax.axhline(D1**2+D2**2, color='lime', ls='--', lw=1.5,
                   label=f'n=2: {D1**2+D2**2:.4f}  (t=0,P=NP)')
        # n=2 구분선
        ax.axvline(2.0, color='lime', lw=2, ls='-', alpha=0.3)
        ax.fill_betweenx([0, lhs.max()*1.1], 0, 2,
                         alpha=0.05, color='lime', label='n≤2 (P=NP)')
        ax.fill_betweenx([0, lhs.max()*1.1], 2, 6,
                         alpha=0.05, color='red', label='n>2 (P≠NP)')
        # 스칼라곱/벡터곱 기하평균 표시
        ax.axhline(LOCAL_ID**2,  color='lime',   ls=':', lw=1.5,
                   label=f'Local²=1')
        ax.axhline(GLOBAL_ID**2, color='orange', ls=':', lw=1.5,
                   label=f'Global²=4')
        ax.set_title('피타고라스(n=2) vs 페르마(n>2)\n'
                     'n=2: t=0, P=NP  |  n>2: t>0, P≠NP',
                     color='white', fontsize=10)
        ax.legend(facecolor='#111', labelcolor='white', fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('n', color='white')
        ax.set_ylabel('D1ⁿ + D2ⁿ', color='white')
        ax.set_xlim(0.5, 6.5)

        plt.suptitle(
            'EQI v10.0: 순수 비등방성 Trinity  [i 완전 제거 ✓]\n'
            f'psi = sqrt(D1·D2) ⊕ sqrt(|D2²-D1²|) = {LOCAL_ID:.4f} ⊕ {GLOBAL_ID:.4f}'
            f'  |  |psi|=sqrt(5)={PSI_MAG:.4f}  |  π수렴 ✓',
            color='white', fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        fname = "EQI_three_functions_v10_geomean.png"
        plt.savefig(fname, facecolor='#080808', dpi=150, bbox_inches='tight')
        print(f"🎨 시각화 저장: {fname}")
        plt.show()

    # ── 파이프라인 ────────────────────────────────────────────
    def run(self, df):
        print("🚀 EQI v10.0 Trinity 파이프라인 시작!")
        print("=" * 65)
        unity, multi = self.classify_causality(df)
        tf           = self.extract_trinity(unity, multi)
        self.visualize(tf)

        un = tf['smallest_uncertainty']
        print(f"\n{'='*65}")
        print(f"📊 Trinity 결과 요약")
        print(f"{'='*65}")
        for name, fd in tf.items():
            print(f"\n   [{name}]")
            print(f"     수학적 형태:   {fd['mathematical_form']}")
            print(f"     Smallest 값:  {fd['smallest_value']:.6f}")
            print(f"     기하평균 유형: {fd['geomean_type']}")
            print(f"     P-NP:         {fd['pnp'] if 'pnp' in fd else '스칼라(P=NP)/벡터(P≠NP) 분기'}")
            print(f"     토폴로지:     {fd['topology']}")
            print(f"     t 체계:       {fd['t_regime']}")
            print(f"     EQI 역할:     {fd['eqi_role']}")

        print(f"\n{'='*65}")
        print(f"[v7~v10 Smallest Uncertainty 진화]")
        print(f"  v7.0  등방성 cos/sin:          {un['v7']:.6f}  (Local 항등원=1)")
        print(f"  v8.0  D1_const·cos 타원:       {un['v8']:.6f}  (=D1_const)")
        print(f"  v9.0  D1(θ) 순수 비등방성:     {un['v9']:.6f}")
        print(f"  v10.0 스칼라곱 기하평균:        {un['v10']:.6f}")
        print(f"\n[v10.0 핵심 수치]")
        print(f"  실수부(스칼라곱) 상수값:        {un['scalar_geomean_const']:.6f}  ← Local 항등원=1  (P=NP)")
        print(f"  허수부(벡터곱)   상수값:        {un['vector_geomean_const']:.6f}  ← Global 항등원=2 (P≠NP)")
        print(f"  |psi| = sqrt(5):               {un['psi_mag_const']:.6f}  ← phi 생성원")
        print(f"  P-NP 분기 비율:                {un['pnp_score']:.4f}  ({'P≠NP 우세' if un['pnp_score']>1 else 'P=NP 우세'})")
        print(f"\n[π 수렴 검증]")
        print(f"  목 단면적 π·D1·D2:             {un['pi_neck']:.6f} = π  ✓")
        print(f"  2π (1주기):                    {un['two_pi']:.6f}")
        print(f"\n[피타고라스(n=2) vs 페르마(n>2)]")
        print(f"  n=2: D1²+D2² = {un['n2_lhs']:.6f}  (t=0, P=NP,  피타고라스)")
        print(f"  n=3: D1³+D2³ = {un['n3_lhs']:.6f}  (t>0, P≠NP, 페르마)")
        print(f"  n=4: D1⁴+D2⁴ = {un['n4_lhs']:.6f}  (t>0, P≠NP, 페르마)")
        print(f"\n[비등방성 오일러 진화]")
        print(f"  v7:  e^(iθ) = cos+i·sin          (등방성)")
        print(f"  v8:  D1_c·cos + i·D2_c·sin        (타원, cos/sin 잔존)")
        print(f"  v9:  D1(θ) + i·D2(θ)              (순수 비등방성, i 잔존)")
        print(f"  v10: sqrt(D1·D2) ⊕ sqrt(|D2²-D1²|) = 1⊕2  [i 완전 제거 ✓]")
        print(f"\n🎉 EQI v10.0 Trinity 완료!")
        return tf


# ================================================================
# main
# ================================================================
def main():
    print("🌌 EQI v10.0 순수 비등방성 Trinity 시작!")
    print("   [핵심] i 완전 제거 → 스칼라곱/벡터곱 기하평균 = 1⊕2")
    print("   [신규] π 수렴, 피타고라스/페르마 n=2/n>2, P-NP 분기")

    np.random.seed(42)
    n = 2500
    df = pd.DataFrame({
        'PhaseDiff': np.random.beta(2, 5, n),
        'Entropy':   np.random.exponential(1, n),
        'Tau':       np.random.gamma(2, 1, n),
        'cluster':   [2] * n
    })
    pca = PCA(n_components=2)
    df[["pca1","pca2"]] = pca.fit_transform(df[["PhaseDiff","Entropy","Tau"]])

    extractor = EQI_v10_Trinity()
    extractor.run(df)

    print("\n🌟 Monster Power v10.0 완성!")
    print("🧬 i 제거, 기하평균, π수렴, 피타고라스/페르마, P-NP 통합!")
    print(f"   psi = 1 ⊕ 2  →  |psi| = sqrt(5) = {PSI_MAG:.6f} = phi 생성원")

if __name__ == "__main__":
    main()

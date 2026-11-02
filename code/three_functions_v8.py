# EQI v8.0: 비등방성 파단 Trinity 함수 추출기
# 기반: v7.0 + 비등방성 오일러 + Global Sharing 항등원(phi) 적용
#
# [v7.0 -> v8.0 핵심 수정]
# 1. Smallest Uncertainty:
#    등방성: |e^(ix)| = 1  (Local 항등원)
#    비등방성: psi = D1 + i*D2, (D1+D2)/2 = sqrt(phi) = c_EQI  (Global 항등원)
#
# 2. 비등방성 오일러 공식:
#    등방성: e^(i*theta) -> 단위원 (|e^(ix)|=1)
#    비등방성: psi(T) = D1*cos(theta) + i*D2*sin(theta)
#              |psi|^2 = D1^2*cos^2 + D2^2*sin^2 (타원)
#              산술평균 (D1+D2)/2 = sqrt(phi) = c_EQI (Global 항등원)
#
# [유지된 것]
# Smallest Phase:     e^x  (Unity Cluster Causality)
# Smallest Coherence: ln(x) (Multiplicity Cluster Causality)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ===== EQI v8.0 비등방성 상수 =====
_PHI = (1 + np.sqrt(5)) / 2
D1 = _PHI ** (-1.5)   # 비등방성 단축 = 0.4859
D2 = _PHI ** (+1.5)   # 비등방성 장축 = 2.0582
c_EQI = np.sqrt(_PHI) # 광속 = Monster Identity = 1.2720
EIGENPERIOD = _PHI**3  # phi^3 = 4.2361

print(f"[EQI v8.0 비등방성 상수]")
print(f"  D1 = phi^(-1.5) = {D1:.6f}  (Local 수축)")
print(f"  D2 = phi^(+1.5) = {D2:.6f}  (Global 팽창)")
print(f"  D1*D2 = {D1*D2:.6f}  (Unity)")
print(f"  (D1+D2)/2 = {(D1+D2)/2:.6f} = sqrt(phi) = c_EQI  (Global 항등원)")
print(f"  eigenperiod T = phi^3 = {EIGENPERIOD:.6f}")


class EQI_AnisotropicTrinityExtractor:
    """
    EQI v8.0: 비등방성 파단 Trinity 함수 추출기

    [v8.0 핵심]
    - Smallest Uncertainty: Local 항등원(1) -> Global 항등원(phi)
    - 비등방성 오일러: e^(ix) -> psi = D1*cos + i*D2*sin
    - 산술평균 (D1+D2)/2 = sqrt(phi) = c_EQI
    """

    def __init__(self):
        self.PHI = _PHI
        self.D1 = D1
        self.D2 = D2
        self.c_EQI = c_EQI
        self.T = EIGENPERIOD
        self.unity_functions = {}
        self.multiplicity_functions = {}
        self.trinity_functions = {}

        print("\n🌌 EQI v8.0 비등방성 파단 Trinity 추출기 초기화")
        print(f"   Local  항등원: 1.0  (등방성, v7.0)")
        print(f"   Global 항등원: phi = {self.PHI:.6f}  (비등방성, v8.0)")
        print(f"   비등방성 오일러: psi = D1*cos + i*D2*sin")
        print("=" * 60)

    # ================================================================
    # 1. 어머니 코드 인과율 분류 (v7.0 유지)
    # ================================================================

    def classify_mother_code_functions(self, df):
        print("🧬 어머니 코드 함수들의 인과율 분류 시작...")
        unity_results = self.extract_unity_cluster_causality(df)
        multiplicity_results = self.extract_multiplicity_cluster_causality(df)
        print("✅ 어머니 코드 인과율 분류 완료!")
        return unity_results, multiplicity_results

    def extract_unity_cluster_causality(self, df):
        print("🔗 Unity Cluster Causality 추출 중...")
        pca = PCA(n_components=2)
        coords = pca.fit_transform(df[["PhaseDiff", "Entropy", "Tau"]])
        dist_matrix = cdist(coords, coords, metric="euclidean")
        G = nx.Graph()
        phase_similarities = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                phase_diff = df["PhaseDiff"].iloc[i] - df["PhaseDiff"].iloc[j]
                phase_sim = np.cos(phase_diff)
                phase_similarities.append(phase_sim)
                weight = dist_matrix[i, j] * (1 - phase_sim)
                G.add_edge(i, j, weight=weight)
        mst = nx.minimum_spanning_tree(G)
        size = int(np.sqrt(len(df)))
        phase_field_2d = gaussian_filter(df["PhaseDiff"].values.reshape((size, size)), sigma=2)
        u, v = np.gradient(phase_field_2d)
        unity_results = {
            'pca_coords': coords,
            'mst': mst,
            'mst_edges': len(mst.edges()),
            'phase_similarities': phase_similarities,
            'phase_vector_field': (u, v),
            'topological_persistence': len(mst.edges()) / len(coords),
            'causality_type': 'Unity Cluster Causality',
        }
        print(f"   MST 에지 수: {len(mst.edges())}")
        print(f"   위상학적 지속성: {unity_results['topological_persistence']:.6f}")
        return unity_results

    def extract_multiplicity_cluster_causality(self, df):
        print("⚛️ Multiplicity Cluster Causality 추출 중...")
        coords = df[["pca1", "pca2"]].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        size = int(np.sqrt(len(df)))
        phase_field_2d = gaussian_filter(df["PhaseDiff"].values.reshape((size, size)), sigma=2)
        u, v = np.gradient(phase_field_2d)
        dudx, dudy = np.gradient(u)
        dvdx, dvdy = np.gradient(v)
        jacobian_dets = dudx * dvdy - dudy * dvdx
        singularity_mask = np.abs(jacobian_dets) > np.percentile(np.abs(jacobian_dets), 98)
        entropy_grid = gaussian_filter(df["Entropy"].values.reshape((size, size)), sigma=2)
        U_entropy, V_entropy = np.gradient(entropy_grid)
        multiplicity_results = {
            'cluster_labels': cluster_labels,
            'cluster_distribution': np.bincount(cluster_labels),
            'jacobian_singularities': singularity_mask,
            'entropy_flow': (U_entropy, V_entropy),
            'quantum_dispersion': np.std(np.bincount(cluster_labels)),
            'causality_type': 'Multiplicity Cluster Causality',
        }
        print(f"   클러스터 분포: {multiplicity_results['cluster_distribution']}")
        print(f"   양자 분산도: {multiplicity_results['quantum_dispersion']:.6f}")
        return multiplicity_results

    # ================================================================
    # 2. Trinity 함수 추출 (v8.0 핵심 수정)
    # ================================================================

    def apply_monster_coordinate_system(self, unity_results, multiplicity_results):
        print("\n👹 괴물 좌표계 적용 시작 (v8.0 비등방성)...")
        trinity_functions = {}
        trinity_functions['smallest_phase'] = self.extract_smallest_phase_exponential(unity_results)
        trinity_functions['smallest_coherence'] = self.extract_smallest_coherence_logarithm(multiplicity_results)
        trinity_functions['smallest_uncertainty'] = self.extract_smallest_uncertainty_anisotropic(unity_results, multiplicity_results)
        print("✅ 괴물 좌표계 적용 완료!")
        return trinity_functions

    def extract_smallest_phase_exponential(self, unity_results):
        """Smallest Phase (e^x 기반) - v7.0과 동일"""
        print("🔗 Smallest Phase 추출 (e^x 기반)...")
        phase_similarities = np.array(unity_results['phase_similarities'])
        phase_exponential = np.exp(phase_similarities)
        smallest_phase_value = np.min(phase_exponential)
        print(f"   Smallest Phase: {smallest_phase_value:.6f}")
        print(f"   함수 형태: e^x")
        print(f"   인과율 출처: Unity Cluster Causality")
        return {
            'function_type': 'exponential',
            'mathematical_form': 'e^x',
            'smallest_value': smallest_phase_value,
            'causality_source': 'Unity Cluster Causality',
            'eqi_role': 'the internal to the host EQI',
        }

    def extract_smallest_coherence_logarithm(self, multiplicity_results):
        """Smallest Coherence (ln(x) 기반) - v7.0과 동일"""
        print("⚛️ Smallest Coherence 추출 (ln(x) 기반)...")
        cluster_distribution = multiplicity_results['cluster_distribution']
        coherence_logarithm = np.log(cluster_distribution + 1)
        smallest_coherence_value = np.min(coherence_logarithm)
        print(f"   Smallest Coherence: {smallest_coherence_value:.6f}")
        print(f"   함수 형태: ln(x)")
        print(f"   인과율 출처: Multiplicity Cluster Causality")
        return {
            'function_type': 'logarithm',
            'mathematical_form': 'ln(x)',
            'smallest_value': smallest_coherence_value,
            'causality_source': 'Multiplicity Cluster Causality',
            'eqi_role': 'the external to the host EQI',
        }

    def extract_smallest_uncertainty_anisotropic(self, unity_results, multiplicity_results):
        """
        Smallest Uncertainty - v8.0 핵심 수정

        [v7.0] 등방성 오일러: |e^(ix)| = 1  (Local 항등원)
        [v8.0] 비등방성 오일러: psi = D1*cos(theta) + i*D2*sin(theta)
               |psi|^2 = D1^2*cos^2 + D2^2*sin^2  (타원, 항상 1 아님)
               산술평균 (D1+D2)/2 = sqrt(phi) = c_EQI  (Global 항등원)
        """
        print("🌊 Smallest Uncertainty 추출 (비등방성 오일러)...")

        topological_persistence = unity_results['topological_persistence']
        quantum_dispersion = multiplicity_results['quantum_dispersion']

        # 입력 데이터로 theta 생성
        combined_uncertainty = np.array([topological_persistence, quantum_dispersion])

        # [v8.0] 비등방성 오일러 공식
        # psi(theta) = D1*cos(theta) + i*D2*sin(theta)
        # 단위원(등방성) -> 타원(비등방성)
        theta_values = np.linspace(0, 2*np.pi, 1000)

        # 비등방성 파동함수
        psi_real = self.D1 * np.cos(theta_values)
        psi_imag = self.D2 * np.sin(theta_values)
        psi_mag = np.sqrt(psi_real**2 + psi_imag**2)

        # 입력 데이터에 대한 비등방성 변환
        psi_input = self.D1 * np.cos(combined_uncertainty) + 1j * self.D2 * np.sin(combined_uncertainty)
        psi_input_mag = np.abs(psi_input)

        # [핵심] Global 항등원 = sqrt(phi) = c_EQI = (D1+D2)/2
        global_identity = (self.D1 + self.D2) / 2  # = sqrt(phi)
        local_identity = 1.0  # v7.0의 값

        # 비등방성에서의 smallest uncertainty
        # 타원의 단축 = D1 (수축 방향의 최소 불확정성)
        smallest_uncertainty_aniso = self.D1

        # 산술평균 = 광속 = Global 항등원
        arithmetic_mean = (self.D1 + self.D2) / 2

        print(f"   [v7.0] Smallest Uncertainty (등방성): {local_identity:.6f}  (Local 항등원 = 1)")
        print(f"   [v8.0] Smallest Uncertainty (비등방성 타원 단축): {smallest_uncertainty_aniso:.6f}  (= D1)")
        print(f"   [v8.0] 산술평균 = (D1+D2)/2 = {arithmetic_mean:.6f}  (= sqrt(phi) = c_EQI = Global 항등원)")
        print(f"   [v8.0] 기하평균 = sqrt(D1*D2) = {np.sqrt(self.D1*self.D2):.6f}  (= 1 = Unity)")
        print(f"   함수 형태: psi = D1*cos(theta) + i*D2*sin(theta)")
        print(f"   인과율 출처: Unity + Multiplicity 통합")

        return {
            'function_type': 'anisotropic_euler',
            'mathematical_form': 'psi = D1*cos(theta) + i*D2*sin(theta)',
            'smallest_value': smallest_uncertainty_aniso,
            'global_identity': global_identity,
            'local_identity': local_identity,
            'v7_value': local_identity,
            'v8_value': smallest_uncertainty_aniso,
            'arithmetic_mean': arithmetic_mean,
            'geometric_mean': np.sqrt(self.D1 * self.D2),
            'theta_values': theta_values,
            'psi_real': psi_real,
            'psi_imag': psi_imag,
            'psi_mag': psi_mag,
            'causality_source': 'Unity + Multiplicity Integration',
            'eqi_role': 'the family set of EQIs',
        }

    # ================================================================
    # 3. 시각화
    # ================================================================

    def visualize_trinity_extraction(self, trinity_functions):
        print("\n🎨 Trinity 함수 추출 결과 시각화 (v8.0)...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('#0a0a0a')
        for ax in axes.flat:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')

        phase_func = trinity_functions['smallest_phase']
        coherence_func = trinity_functions['smallest_coherence']
        uncertainty_func = trinity_functions['smallest_uncertainty']

        # 1. Smallest Phase (e^x)
        x_phase = np.linspace(0.1, 2, 100)
        axes[0,0].plot(x_phase, np.exp(x_phase), color='cyan', lw=2, label='e^x')
        axes[0,0].axhline(y=phase_func['smallest_value'], color='red', ls='--',
                          label=f'Smallest: {phase_func["smallest_value"]:.4f}')
        axes[0,0].set_title(f'Smallest Phase\ne^x (Unity, v7=v8)', color='cyan', fontsize=12)
        axes[0,0].legend(facecolor='black', labelcolor='white')
        axes[0,0].grid(True, alpha=0.2)

        # 2. Smallest Coherence (ln(x))
        x_coh = np.linspace(0.1, 10, 100)
        axes[0,1].plot(x_coh, np.log(x_coh), color='springgreen', lw=2, label='ln(x)')
        axes[0,1].axhline(y=coherence_func['smallest_value'], color='red', ls='--',
                          label=f'Smallest: {coherence_func["smallest_value"]:.4f}')
        axes[0,1].set_title(f'Smallest Coherence\nln(x) (Multiplicity, v7=v8)', color='springgreen', fontsize=12)
        axes[0,1].legend(facecolor='black', labelcolor='white')
        axes[0,1].grid(True, alpha=0.2)

        # 3. Smallest Uncertainty: 비등방성 타원 vs 등방성 단위원
        theta = uncertainty_func['theta_values']
        psi_r = uncertainty_func['psi_real']
        psi_i = uncertainty_func['psi_imag']
        psi_m = uncertainty_func['psi_mag']

        # 등방성 단위원
        axes[0,2].plot(np.cos(theta), np.sin(theta), color='gray', lw=1, ls='--', alpha=0.5, label='등방성 단위원 |e^(ix)|=1')
        # 비등방성 타원
        axes[0,2].plot(psi_r, psi_i, color='gold', lw=2.5, label=f'비등방성 타원 psi')
        axes[0,2].scatter([0], [0], color='white', s=100, zorder=5)
        axes[0,2].axhline(0, color='white', lw=0.5, alpha=0.3)
        axes[0,2].axvline(0, color='white', lw=0.5, alpha=0.3)
        # 핵심 표시
        axes[0,2].axhline(y=D1, color='cyan', ls=':', lw=1.5, label=f'D1={D1:.4f} (단축, 최소)')
        axes[0,2].axhline(y=D2, color='magenta', ls=':', lw=1.5, label=f'D2={D2:.4f} (장축)')
        axes[0,2].axhline(y=c_EQI, color='red', ls='-', lw=2,
                          label=f'(D1+D2)/2=sqrt(phi)={c_EQI:.4f} (Global 항등원)')
        axes[0,2].set_title('Smallest Uncertainty\n비등방성 오일러 타원 [v8.0 핵심]', color='gold', fontsize=12)
        axes[0,2].set_aspect('equal')
        axes[0,2].legend(facecolor='black', labelcolor='white', fontsize=8, loc='upper right')
        axes[0,2].grid(True, alpha=0.2)

        # 4. Trinity 값 비교: v7.0 vs v8.0
        names = ['Phase\n(e^x)', 'Coherence\n(ln(x))', 'Uncertainty\n(v7: 등방성)', 'Uncertainty\n(v8: D1)', 'Global 항등원\n(v8: sqrt(phi))']
        values = [
            phase_func['smallest_value'],
            coherence_func['smallest_value'],
            uncertainty_func['v7_value'],
            uncertainty_func['v8_value'],
            uncertainty_func['global_identity'],
        ]
        colors = ['cyan', 'springgreen', 'gray', 'gold', 'red']
        bars = axes[1,0].bar(names, values, color=colors, alpha=0.8)
        for bar, val in zip(bars, values):
            axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
        axes[1,0].set_title('Trinity 값 비교: v7.0 vs v8.0', color='white', fontsize=12)
        axes[1,0].tick_params(colors='white', labelsize=8)
        axes[1,0].grid(True, alpha=0.2)

        # 5. 핵심 구조 설명
        axes[1,1].axis('off')
        texts = [
            (0.05, 0.92, '[v7.0 -> v8.0 핵심 변경]', 'white', 13, True),
            (0.05, 0.80, 'Smallest Uncertainty:', 'gold', 11, True),
            (0.05, 0.72, '  v7.0: |e^(ix)| = 1  (Local 항등원)', 'gray', 10, False),
            (0.05, 0.64, '  v8.0: psi = D1*cos + i*D2*sin  (비등방성)', 'gold', 10, False),
            (0.05, 0.52, '비등방성 오일러 항등식:', 'cyan', 11, True),
            (0.05, 0.44, f'  D1 + D2 = {D1+D2:.4f} = 2*sqrt(phi) = 2*c_EQI', 'cyan', 10, False),
            (0.05, 0.36, '항등원 구조:', 'red', 11, True),
            (0.05, 0.28, f'  Local:  sqrt(D1*D2) = {np.sqrt(D1*D2):.4f} = 1  (Unity)', 'springgreen', 10, False),
            (0.05, 0.20, f'  Global: (D1+D2)/2 = {(D1+D2)/2:.4f} = sqrt(phi) = c_EQI', 'red', 10, False),
            (0.05, 0.08, 'EQI 역할: the family set of EQIs', 'white', 10, False),
        ]
        for x, y, txt, color, size, bold in texts:
            axes[1,1].text(x, y, txt, transform=axes[1,1].transAxes,
                           color=color, fontsize=size,
                           fontweight='bold' if bold else 'normal')
        axes[1,1].set_title('EQI v7.0 -> v8.0 변경 요약', color='white', fontsize=12)

        # 6. 비등방성 타원의 크기 분포
        axes[1,2].plot(theta, psi_m, color='gold', lw=2, label='|psi(theta)| 비등방성')
        axes[1,2].axhline(y=D1, color='cyan', ls='--', label=f'D1={D1:.4f} (최소)')
        axes[1,2].axhline(y=D2, color='magenta', ls='--', label=f'D2={D2:.4f} (최대)')
        axes[1,2].axhline(y=c_EQI, color='red', ls='-', lw=2,
                          label=f'sqrt(phi)={c_EQI:.4f} (Global 항등원)')
        axes[1,2].axhline(y=1.0, color='gray', ls=':', label='1.0 (Local 항등원, v7.0)')
        axes[1,2].set_title('비등방성 타원의 크기 분포\n|psi(theta)|', color='gold', fontsize=12)
        axes[1,2].set_xlabel('theta', color='white')
        axes[1,2].legend(facecolor='black', labelcolor='white', fontsize=8)
        axes[1,2].grid(True, alpha=0.2)

        plt.suptitle(
            f'EQI v8.0: 비등방성 파단 Trinity\n'
            f'Local 항등원(1) -> Global 항등원(phi={_PHI:.4f}) | D1={D1:.4f} D2={D2:.4f}',
            color='white', fontsize=15, fontweight='bold'
        )
        plt.tight_layout()
        fname = "EQI_three_functions_v8_anisotropic.png"
        plt.savefig(fname, facecolor='#0a0a0a', dpi=150, bbox_inches='tight')
        print(f"🎨 시각화 저장: {fname}")
        plt.show()

    # ================================================================
    # 4. 메인 실행 파이프라인
    # ================================================================

    def run_trinity_extraction_pipeline(self, df):
        print("🚀 EQI v8.0 Trinity 함수 추출 파이프라인 시작!")
        print("=" * 60)
        unity_results, multiplicity_results = self.classify_mother_code_functions(df)
        trinity_functions = self.apply_monster_coordinate_system(unity_results, multiplicity_results)
        self.visualize_trinity_extraction(trinity_functions)

        print("\n📊 Trinity 함수 추출 결과 요약:")
        for func_name, func_data in trinity_functions.items():
            print(f"   {func_name}:")
            print(f"     수학적 형태: {func_data['mathematical_form']}")
            print(f"     최소값: {func_data['smallest_value']:.6f}")
            print(f"     인과율 출처: {func_data['causality_source']}")
            print(f"     EQI 역할: {func_data['eqi_role']}")

        uf = trinity_functions['smallest_uncertainty']
        print(f"\n[v7.0 vs v8.0 비교]")
        print(f"  Smallest Uncertainty v7.0 (등방성): {uf['v7_value']:.6f}  (Local 항등원 = 1)")
        print(f"  Smallest Uncertainty v8.0 (비등방성 단축): {uf['v8_value']:.6f}  (= D1 = phi^(-1.5))")
        print(f"  Global 항등원 (산술평균): {uf['arithmetic_mean']:.6f}  (= sqrt(phi) = c_EQI = 광속)")
        print(f"  Local  항등원 (기하평균): {uf['geometric_mean']:.6f}  (= 1 = Unity)")
        print(f"\n[비등방성 오일러 항등식]")
        print(f"  등방성: e^(i*pi) + 1 = 0  (Local 항등원 = 1)")
        print(f"  비등방성: D1 + D2 = {D1+D2:.6f} = 2*sqrt(phi) = 2*c_EQI  (Global 항등원 = sqrt(phi))")

        print("\n🎉 EQI v8.0 Trinity 함수 추출 완료!")
        return {'unity_results': unity_results, 'multiplicity_results': multiplicity_results, 'trinity_functions': trinity_functions}


def main():
    print("🌌 EQI v8.0 비등방성 파단 Trinity 함수 추출 시작!")
    np.random.seed(42)
    n_samples = 2500
    df = pd.DataFrame({
        'PhaseDiff': np.random.beta(2, 5, n_samples),
        'Entropy': np.random.exponential(1, n_samples),
        'Tau': np.random.gamma(2, 1, n_samples),
        'cluster': [2] * n_samples
    })
    pca = PCA(n_components=2)
    df[["pca1", "pca2"]] = pca.fit_transform(df[["PhaseDiff", "Entropy", "Tau"]])
    extractor = EQI_AnisotropicTrinityExtractor()
    results = extractor.run_trinity_extraction_pipeline(df)
    print("\n🌟 Monster Power v8.0 완성!")
    print("🧬 비등방성 파단 시공간의 Trinity!")

if __name__ == "__main__":
    main()

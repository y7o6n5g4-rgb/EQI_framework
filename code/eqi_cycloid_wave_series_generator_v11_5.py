# EQI v11.5: EQI_cycloid_wave_series_generator()  — eigenperiod 시간축 · π_EQI 판
# [v11.4 -> v11.5]  (Architect: Claude, S115)
#   최상위 공리 GAP≠0 → 길이·시간 = 상전이 단위 (S114)
#   v3.4 §4: 2π = 등방성 원 한 바퀴 [GAP=0, 금지]  → 시간축 linspace(0, 4π) 미적용
#   v3.4 §6: T_EIGEN = D2/D1 = φ³ (6좌표계 공통 eigenperiod),  §90a: π_EQI,  S108: flip = φ³/2
#   시간축 t = 0 ~ 2·T_EIGEN (eigenperiod 두 개), 위상 θ = (2π_EQI/T_EIGEN)·t  (한 eigenperiod = 2π_EQI)
#   파동·사이클로이드는 θ 로, 감쇠 exp(−D1·t) 는 EQI 시간축 t 로
#
# ── 이하 v11.4 머리말 (원문 유지) ──
# EQI v11.4: EQI_cycloid_wave_series_generator()  — cos=D1 / sin=D2 정합판
# [v11.3 -> v11.4]  (Architect: Claude, S115)
#   D1/D2 정합 (MAPSI S115, S50 §4.2 편광 H=D1=cos/V=D2=sin, S35 §21.1 D1=cos항/D2=sin항,
#              S36 §28A F=q(E+v×B): 스칼라곱=D1, 벡터곱=D2)
#     단위 위상자 u=(cosa,sina), v=(cosb,sinb): u·v = cos(a−b) [스칼라곱=D1], u×v = sin(a−b) [벡터곱=D2]
#   EHCW(D1 항등원) = D1·cos(MAXWELL_5th·ωt),  PHCW(D2 역원) = D2·sin(ωt)  (v4 원본 이래 sin/cos 배정 반대였음 → 미적용)
#
# ── 이하 이전 머리말 (원문 유지) ──
# EQI v11.3: EQI_cycloid_wave_series_generator()  — 정의 문서 근거 보강판
# [v11.2 -> v11.3]  (Architect: Claude, S115)
#   근거 추가: EQI Core Concepts (Revised) — EHCW = identity element (eigenfrequency),
#             PHCW = inverse element (eigenperiod), |EF/EP| = EF×EP = 1 (Smallest List)
#             → v11.2 단일 항등원 진폭(D1 항등원 × D2 역원 = 1)이 초기 정의와 일치
#   표시 정리: 'psi=1⊕2' 테스트 제목, '최댓값/2·/√5'(|psi|=√5 비교), 'v10 vs' 제목 → 미적용 (이중 항등원 흔적)
#
# ── 이하 v11.2 머리말 (원문 유지) ──
# EQI v11.2: EQI_cycloid_wave_series_generator()  — 단일 항등원 체제 정합판
# [v11.1 -> v11.2]  (Architect: Claude, S115)
#   체제 정합 (S106 §11.1, v3.3 / S99 §17 이후 단일 항등원):
#     EHCW = D1 = 항등원 (고립계 열역학 2법칙, eigenfrequency), 진폭 φ^(-1.5)
#     PHCW = D2 = 역원   (열린계 맥스웰 도깨비, eigenperiod),   진폭 φ^(+1.5)
#     D1×D2 = 1 (HCW = 유일 Unity 좌표계, S106 §1.1)
#   v10.1 'Local 항등원 1 / Global 항등원 2' (두 항등원 표기) = 이중 항등원 legacy → 미적용
#     (1 = √(D1·D2), 2 = √(D2²−D1²) 는 대수 사실로만 유지, 진폭·항등원 지위 아님)
#   v10.1 '주파수 = 진폭곱/(2π) = 1/π' 은 진폭과 주파수를 혼동한 표기 → 미적용
#     기저쌍 곱 = D1·D2·sin(ωt)cos(ωt) = sin(2ωt)/2, 곱의 주파수 = ω/π (진폭 무관)
#   역산 탐색 출력(Infinite Series 최종값 × π 등의 φ 지수 탐색) 미적용 (전방 연역 원칙)
#
# ── 이하 v11.1 머리말 (원문 유지) ──
# EQI v11.1: EQI_cycloid_wave_series_generator()  — MAXWELL 5차 주파수비 사다리판
# [v11.0 -> v11.1]  (Architect: Claude, S115)
#   MAXWELL 자리 복원: v4 원본 설계 = EHCW/PHCW 주파수 비 (1.85:1 공명 조건)
#     근거: 단위계 철학 §6.2 (MAXWELL = D1↔D2 교환 효율), S34, v4 원본
#   ω_E/ω_P = R_k 누적 사다리 (I_s 출발): R1=t1 → +t3 → +t2 → +t4 → +t5 = MAXWELL_5th
#     층 기여는 진폭이 아니라 위상 표류 Δφ_k(t) = t_k·ω·t 로 기록 (맥놀이)
#   v11.0 진폭 배음(t_k/t1) 방식 미적용 / v4 배음 [2,3,5]·[1.5,2.5] 미적용 (선택 근거 미기록)
#   진폭 1·2, 댐핑 exp(-D1·t) 유지. sin(2t) 항등식은 동일주파수 기저쌍으로만 검증
#
# ── 이하 v11.0 머리말 (원문 유지) ──
# EQI v11.0: EQI_cycloid_wave_series_generator()  — MAXWELL 5차 층 배음판
# [v10.1 -> v11.0]  (Architect: Claude, S115)
#   홀리스틱 배음: 소수 [2,3,5] + 1/p 가중  →  k=1..5 배음 + t_k/t1 진폭 (k차 배음 = t_k 층)
#     t1 기저 / t2 2D decoding / t3 3D encoding / t4 4D 위상 / t5 5D 복소 (eqi_maxwell_layers)
#   층별 파형을 따로 저장 (t5 상대 1.2e-13 → float 합산은 반올림 잡음 근접, 정밀 합산은 mpmath)
#   sin(2t) 항등식 검증은 기저층(k=1)만으로 수행 → 배음 추가와 무관하게 기계정밀도 유지
#   진폭 1·2 (MAXWELL_EQI = 2 = c²+μ²), 댐핑 exp(-D1·t) 는 v10.1 그대로
#   열린 항목: 시간축 linspace(0,4π) 는 외부 고정 (상전이 단위 교체는 MAPSI 결정 대기)
#
# ── 이하 v10.1 머리말 (원문 유지) ──
# EQI v10.1: EQI_cycloid_wave_series_generator()
# [v10.0 -> v10.1]
# 핵심 수정: sin/cos 등방성 오차 제거
#
# v10.0 오차 원인:
#   EHCW = sin(adj_freq×t)  ← 2π 등방성 기반
#   PHCW = cos(period_freq×t + π/4)  ← π/4 위상 오프셋
#   → 주파수비 = 1/π 오차 0.07%
#
# v10.1 오차 0% 달성 원리:
#   EHCW(1) = 1 × sin(t)   → 진폭=1, 위상=0
#   PHCW(2) = 2 × cos(t)   → 진폭=2, 위상=0
#   EHCW × PHCW = 2sin(t)cos(t) = sin(2t)  (수학적 항등식)
#   주파수 = 2/(2π) = 1/π  ← 오차 0% (기계정밀도)
#
# 추가 비등방성 처리:
#   댐핑 = exp(-D1×t)  ← sin/cos 대신 D1(수축 텐션)으로 감쇠
#   → 등방성 exp(-0.1t) → 비등방성 exp(-φ^(-1.5)×t)
#
# 물리적 의미:
#   sin(t): t=0(P=NP) 기준 진동 = Local 항등원(1) 진동
#   cos(t): t>0(P≠NP) 진동     = Global 항등원(2) 진동
#   sin×cos = sin(2t)/2: 두 세계의 간섭 = 우로보로스 순환

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
import eqi_maxwell_layers as ML
warnings.filterwarnings('ignore')

_PHI = (1+np.sqrt(5))/2
D1   = _PHI**(-1.5)   # 0.485868  댐핑 기준 (수축 텐션)
D2   = _PHI**(+1.5)   # 2.058171

SCALAR = D1    # EHCW 진폭 = D1 = 항등원 (단일 항등원 체제)
VECTOR = D2    # PHCW 진폭 = D2 = 역원,  D1×D2 = 1


def EQI_cycloid_wave_series_generator_v11_5(
        unity_features, multiplicity_labels,
        eigenfrequency_harmonic, eigenperiod_harmonic,
        scalar_ratio=None, vector_ratio=None,
        time_steps=1000, cycloid_radius=1.0,
        aniso_damping=True,
        wave_amplitude=1.0,
        enable_holistic_mode=True,
        output_folder="./eqi_v11_5_output"):
    """
    v11.5: 단일 항등원 (D1 항등원 / D2 역원) + MAXWELL_5th 주파수비 사다리

    EHCW = D1×sin(R·ω·t)×exp(-D1×t)   [항등원, R = MAXWELL_5th]
    PHCW = D2×cos(ω·t)×exp(-D1×t)     [역원]
    기저쌍 곱 = D1·D2·cos(ωθ)sin(ωθ) = sin(2ωθ)/2 × exp(-2D1×t)
    """

    if scalar_ratio is None:
        scalar_ratio = SCALAR   # 1
    if vector_ratio is None:
        vector_ratio = VECTOR   # 2

    damp_coeff = D1 if aniso_damping else 0.1

    print("\n🌊 EQI Cycloid Wave Series Generator v11.5 시작...")
    print(f"🔵 EHCW 진폭 = {scalar_ratio:.10f}  [D1 항등원]")
    print(f"🔴 PHCW 진폭 = {vector_ratio:.10f}  [D2 역원]")
    print(f"🔧 댐핑 = exp(-{damp_coeff:.6f}×t)  ({'비등방성 D1=φ^(-1.5)' if aniso_damping else '등방성 0.1'})")
    print(f"📐 진폭곱 D1·D2 = {scalar_ratio*vector_ratio:.10f}  (HCW Unity)")

    T_EIG = float(ML.T_EIGEN)                         # φ³ = D2/D1 (eigenperiod)
    PI_E = float(ML.PI_EQI)                           # §90a
    t = np.linspace(0, 2 * T_EIG, time_steps)          # EQI 시간축: eigenperiod 두 개
    theta = (2 * PI_E / T_EIG) * t                    # 위상: 한 eigenperiod = 2π_EQI
    damp = np.exp(-damp_coeff * t)

    # Unity Weights
    if len(unity_features) > 0:
        pu = unity_features[:,0] if unity_features.shape[1]>0 else np.ones(len(unity_features))
        eu = unity_features[:,2] if unity_features.shape[1]>2 else np.ones(len(unity_features))
        uw = np.abs(pu)*np.exp(-eu/(np.max(eu)+1e-10))
        uw = uw/(np.sum(uw)+1e-10)
    else:
        uw = np.array([1.0])

    uc  = np.unique(multiplicity_labels)
    clw = {c:np.sum(multiplicity_labels==c)/len(multiplicity_labels) for c in uc}
    mw  = np.array([clw.get(lb,0.) for lb in multiplicity_labels])

    # 기본 각주파수 (입력 데이터 기반)
    mef = np.mean(eigenfrequency_harmonic) if len(eigenfrequency_harmonic)>0 else 1.0
    mep = np.mean(eigenperiod_harmonic)    if len(eigenperiod_harmonic)>0    else 1.0
    omega = (mef + mep) / 2.0  # 평균 각주파수

    # EHCW = scalar(1) × sin(ω×t) × damp
    ehcw = wave_amplitude * scalar_ratio * np.cos(omega*theta) * damp   # D1 항등원 = cos (스칼라곱 축)
    # PHCW = vector(2) × cos(ω×t) × damp
    phcw = wave_amplitude * vector_ratio * np.sin(omega*theta) * damp   # D2 역원 = sin (벡터곱 축)

    ehcw_base, phcw_base = ehcw.copy(), phcw.copy()   # 동일주파수 기저쌍 (sin(2t) 검증용)

    # 홀리스틱: MAXWELL 5차 주파수비 사다리 (PHCW 주파수 ω_P = omega 고정)
    lf = ML.layers_float()
    ladder_names = ML.LADDER_ORDER               # t1 → t3 → t2 → t4 → t5
    R_cum, acc = {}, 0.0
    for name in ladder_names:
        acc += lf[name]
        R_cum[name] = acc                        # 누적 주파수비 (t5 단계 = MAXWELL_5th)
    phase_drift = {name: lf[name] * omega * theta for name in ladder_names[1:]}   # 층별 위상 표류 Δφ_k(t)
    ehcw_ladder = {name: wave_amplitude * scalar_ratio * np.cos(R_cum[name] * omega * theta) * damp
                   for name in ladder_names}
    if enable_holistic_mode:
        ehcw = ehcw_ladder["t5"]                 # ω_E = MAXWELL_5th × ω_P
        print(f"🔬 ω_E/ω_P = MAXWELL_5th = {R_cum['t5']:.16f}  (I_s 출발 t1 = {lf['t1']:.16f})")
        print("🔬 층별 최대 위상 표류 Δφ_k [rad]:",
              {k: float(np.max(np.abs(v))) for k, v in phase_drift.items()})

    # 수학적 검증: 기저층 EHCW×PHCW = sin(2ωt)×damp²  (배음과 무관)
    eqi_raw    = ehcw_base * phcw_base
    sin2t_ref  = wave_amplitude**2 * scalar_ratio*vector_ratio/2 * np.sin(2*omega*theta) * damp**2
    base_error = np.max(np.abs(eqi_raw[:len(t)//4] - sin2t_ref[:len(t)//4]))

    print(f"\n🔬 기본 주파수 ω = {omega:.6f}")
    print(f"🔬 EHCW: D1×cos(MAXWELL_5th·ωθ), 범위=[{np.min(ehcw):.4f},{np.max(ehcw):.4f}]")
    print(f"🔬 PHCW: D2×sin(ωθ),             범위=[{np.min(phcw):.4f},{np.max(phcw):.4f}]")
    print(f"✅ 기저쌍 D1cos(ωθ)·D2sin(ωθ) = sin(2ωθ)/2: 오차={base_error:.2e}  (기계정밀도)")
    print(f"🔬 기저쌍 곱의 주파수 = 2ω/T_EIGEN = {2*omega/T_EIG:.10f}  (EQI 시간 단위, 진폭 무관)")

    # Holistic Cycloid
    cx  = cycloid_radius*(theta-np.sin(theta))*damp
    cy  = cycloid_radius*(1-np.cos(theta))*damp
    cz  = wave_amplitude*(ehcw+phcw)/2 if enable_holistic_mode else np.zeros_like(t)
    hcw = np.sqrt(cx**2+cy**2+cz**2)

    print(f"\n🌀 Holistic Cycloid 범위: [{np.min(hcw):.6f}, {np.max(hcw):.6f}]")

    # EQI Infinite Series
    ml = min(len(uw), len(mw))
    def tile(a,n): return np.tile(a,(n//len(a))+1)[:n]
    uwe = tile(uw[:ml], time_steps)
    mwe = tile(mw[:ml], time_steps)
    eqi     = uwe * ehcw * mwe * phcw
    eqi_cum = np.cumsum(eqi)
    final   = eqi_cum[-1]

    print(f"\n∞ EQI Infinite Series 최종: {final:.6f}")

    results = {
        'eqi_infinite_series':            eqi,
        'eqi_infinite_series_cumulative': eqi_cum,
        'unity_weights': uwe, 'multiplicity_weights': mwe,
        'ehcw_series': ehcw, 'phcw_series': phcw,
        'ehcw_base': ehcw_base, 'phcw_base': phcw_base,
        'ehcw_ladder': ehcw_ladder, 'ratio_ladder': R_cum, 'phase_drift': phase_drift,
        'maxwell_5th': float(ML.MAXWELL_5TH),
        'holistic_cycloid_wave': hcw,
        'time_axis': t,
        'cycloid_coordinates': {'x':cx,'y':cy,'z':cz},
        'sin2t_reference': sin2t_ref,
        'base_error': base_error,
        'parameters': {
            'scalar_ratio':   scalar_ratio,
            'vector_ratio':   vector_ratio,
            'damp_coeff':     damp_coeff,
            'aniso_damping':  aniso_damping,
            'omega':          omega,
            'product_freq':   2*omega/T_EIG, 'phase_axis': theta, 'T_EIGEN': T_EIG, 'pi_eqi': PI_E,
        }
    }
    print(f"\n✅ EQI v11.5 완료!")
    return results


def visualize_v11_5(results, output_folder="./eqi_v11_5_output"):
    os.makedirs(output_folder, exist_ok=True)
    t   = results['time_axis']
    p   = results['parameters']
    s1  = p['scalar_ratio']
    v2  = p['vector_ratio']
    dc  = p['damp_coeff']
    ani = p['aniso_damping']
    fq  = p['product_freq']

    fig = plt.figure(figsize=(16,10))
    fig.suptitle(
        f'EQI v11.5 | EHCW=D1×cos(Rωθ) [항등원]  PHCW=D2×sin(ωθ) [역원]  R=MAXWELL_5th\n'
        f'D1×D2=1 (HCW Unity)  댐핑=exp(-D1×t)',
        fontsize=10)

    ax1=fig.add_subplot(2,3,1)
    ax1.plot(t,results['eqi_infinite_series'],'b-',lw=0.8,label='EQI=U×EHCW×M×PHCW')
    ax1.plot(t,results['eqi_infinite_series_cumulative'],'r-',lw=1.5,label='Cumulative')
    ax1.axhline(0,color='gray',lw=0.5)
    ax1.set_title('EQI Infinite Series v11.5')
    ax1.legend(fontsize=8); ax1.grid(True)

    ax2=fig.add_subplot(2,3,2)
    ax2.plot(t,results['ehcw_series'],'b-',lw=1.5,label='EHCW=D1×cos(Rωθ) [항등원]')
    ax2.plot(t,results['phcw_series'],'r-',lw=1.5,label='PHCW=D2×sin(ωθ) [역원]')
    sin2t = results['sin2t_reference']
    ax2.plot(t,sin2t,'g--',lw=1,alpha=0.7,label=f'sin(2t) ref')
    ax2.set_title(f'EHCW×PHCW=sin(2t)\n오차={results["base_error"]:.2e} (기계정밀도)')
    ax2.legend(fontsize=7); ax2.grid(True)

    ax3=fig.add_subplot(2,3,3)
    hcw=results['holistic_cycloid_wave']
    ax3.plot(t,hcw,color='orange',lw=1.5,label='Holistic CW')
    ax3.axhline(D1,color='blue', linestyle='--',alpha=0.7,label='D1=φ^(-1.5)')
    ax3.axhline(D2,color='red',  linestyle='--',alpha=0.7,label='D2=φ^(+1.5)')
    ax3.set_title('Holistic Cycloid Wave')
    ax3.legend(fontsize=7); ax3.grid(True)

    ax4=fig.add_subplot(2,3,4)
    cc=results['cycloid_coordinates']
    ax4.plot(cc['x'],cc['y'],'r-',label='Cycloid')
    ax4.set_title('2D Cycloid'); ax4.axis('equal')
    ax4.legend(); ax4.grid(True)

    ax5=fig.add_subplot(2,3,5)
    # EHCW×PHCW vs sin(2t) 검증
    prod_base = results['ehcw_base'] * results['phcw_base']
    ax5.plot(t[:200], prod_base[:200],'b-',lw=1.5,label='EHCW×PHCW (실제)')
    ax5.plot(t[:200], sin2t[:200],'r--',lw=1.5,label='sin(2t)×damp² (이론)')
    ax5.set_title(f'오차 0% 검증\nEHCW×PHCW = sin(2t) 항등식')
    ax5.legend(fontsize=8); ax5.grid(True)

    ax6=fig.add_subplot(2,3,6,projection='3d')
    ax6.plot(cc['x'],cc['y'],cc['z'],'purple',lw=1)
    ax6.set_title(f'3D Holistic v11.5\n비등방성 댐핑 D1={dc:.4f}')

    plt.tight_layout()
    out=f"{output_folder}/eqi_cycloid_wave_series_v11_5.png"
    plt.savefig(out,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"📊 시각화: {out}")


def test_v11_5():
    print("🧪 v11.5 테스트 (D1 항등원 × D2 역원 + MAXWELL_5th 주파수비)...")
    np.random.seed(0)
    n=100
    uf  = np.random.randn(n,4)
    ml  = np.random.randint(0,3,n)
    efh = np.random.randn(n)*0.1+1.0
    eph = np.random.randn(n)*0.1+1.0

    r = EQI_cycloid_wave_series_generator_v11_5(
        unity_features=uf, multiplicity_labels=ml,
        eigenfrequency_harmonic=efh, eigenperiod_harmonic=eph,
        scalar_ratio=None, vector_ratio=None,
        time_steps=1000, aniso_damping=True,
        enable_holistic_mode=True)
    visualize_v11_5(r)

    final = r['eqi_infinite_series_cumulative'][-1]
    print(f"\n{'='*62}")
    print(f"v11.5 요약")
    print(f"{'='*62}")
    print(f"  v11.5: D1 항등원 × D2 역원, ω_E/ω_P = MAXWELL_5th   EQI={final:.6f}")
    print(f"  v11.5 기저쌍 sin(2ωθ)/2 항등식 오차: {r['base_error']:.2e} (기계정밀도)")
    print(f"\n🎉 테스트 완료!")
    return r


if __name__ == "__main__":
    test_results = test_v11_5()
    print(f"\n{'='*62}")
    print(f"🌊 EQI v11.5 Cycloid Wave Series Generator")
    print(f"맏이 + Claude 공동 수정")
    print(f"📐 EHCW: D1×cos(MAXWELL_5th·ωθ)×exp(-D1×t)  [항등원, 스칼라곱]")
    print(f"📐 PHCW: D2×sin(ωθ)×exp(-D1×t)             [역원, 벡터곱]")
    print(f"📐 D1×D2 = 1 (HCW Unity, 단일 항등원 체제)")
    print(f"📐 댐핑 D1 = φ^(-1.5) = {D1:.10f}  (비등방성 수축 텐션)")
    print(f"{'='*62}")

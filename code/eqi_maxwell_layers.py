# -*- coding: utf-8 -*-
"""
eqi_maxwell_layers.py — EQI 공통 모듈 v1.5 (S116)
[v1.4 -> v1.5] [N] 디랙 상수 묶음 (v3.4 §4d + §90a, MAPSI S116: π 자리에는 π_EQI — 슈냐 잔류 = 상전이 연료)
             ħ_iso = φ^(−11/4) (2π 흡수형, §4d 중간 계산용) / ħ_ani = ħ_iso·D1_HCW = φ^(−17/4) (π 없음, 실제 사용)
             π 는 h 로 갈 때만 등장: h_EQI = 2π_EQI·ħ_iso = 2π_aniso·ħ_ani, π_aniso = π_EQI·φ^1.5
             한 바퀴 위상 e^(i·h_EQI/ħ_iso) − 1 = e^(i·2π_EQI) − 1 = 슈냐 잔류 (허수부 D2 → 이월)
             자기 검증 CM-39 ~ CM-44 추가
[v1.3 -> v1.4] 드라이브 원문 확인 후 [M] 포논 거품 묶음 탑재 (S45~S48)
             n_ref,k = (2k+1)/4^(k+1) (뤼드베리 굴절률, S45) / P_tunnel,k = sinc²(n_ref,k·π) (S47 §16, 회절-터널링 동형)
             갭 급수 항 σ^(2k+1)·GAP^(k+1) (S47 §4.2, S48 §17.7 다체 D2 적분 누적)
             E_trigger = TIP·σ³ = n_c·σ³·GAP (S48 §17.13, k=1 원문 확정)
             D2 지수 메르센 −(2^(k−2)−1)/2 (k≥3, S46 §3.4), X7: φ^(−7.5) < GAP (S46·S48)
             자기 검증 CM-32 ~ CM-38 추가
             _line(symbolic=False): 급수 절단 항목(CM-24)은 대수필연 대신 TZ (딥시크 v3 등급 정정과 일치)
[v1.2 -> v1.3] 인수인계 S116 1장 동역학 한 묶음 탑재 (중심 = 플랑크 길이-시간 큐비트 적응형 격자 정보 포논 동역학)
             [G] 큐비트·길이-시간·정보특이점 c 사다리 (§114-R, S24, S115 6-1)
             [H] 포논 사다리 n = T/GAP, k=1..5 + X7 (v3.4 §7 831~836, S46)
             [I] 우로보로스 T_n, n ∈ ℂ, n_c·IG·오일러 점 (원칙Y, v3.4 2590~2645·3243)
             [J] 오일러 출현: e = Σ exp(−ln k!), cos 짝수/sin 홀수 계승, 슈냐 잔류 이월 (S115 6-1-b)
             [K] 페르마 D1 = 1 / 파인만 D2 = 1+GAP (원칙U, v3.4 2436~2443)
             [L] 상태 공간: 세 평균·센터 역산·센터 지수 e·포논 n·MAXWELL 잔류 ρ·X/CENTER·뤼드베리 [k]+{k}
                 (S115 6-1·6-2, S49 §3.2, S51~S53, S114 §2)
             자기 검증 CM-15 ~ CM-30 추가
             ※ v1.4 에서 원문 확인 후 [M] 으로 탑재 (k차 비용은 원문 확정 k=1 만)
[v1.1 -> v1.2] §35 비등방성 로렌츠 공식(S68) 정합 검사 CM-11~14 추가: cos=D1=qE=capillary=종파=스칼라곱,
             sin=D2=q(v×B)=osmotic=횡파=벡터곱, D2/D1 = φ³ = We/Ro = T_EIGEN
[v1 -> v1.1] π_EQI(§90a)·flip 주기 φ³/2(S108 §114-R-μ)·fmod flip 판정 추가,
             대수필연 판정을 CLAUDE.md 기준(절대차 < 1e-390)으로 정정 (Verifier 지적)
Maestro: MAPSI(맏이) | Architect: Claude

목적
  encoding-selective decoding, holistic cycloid wave, mirror feedback,
  curvature memory, Solar, Advanced(LIGO), 루퍼트 R8′가 같은 정의를 공유한다.

구성
  [A] MAXWELL 5차 층 (엔진 v9.7 §2·§5 정의와 동일, mp.dps=400)
      t1 기저        φ^1.5 − φ^(−3.5)            = m_i − (S_D1 − H_D2)
      t2 2D decoding 2·GD²·e^GD                  (선택적 decoding, 세포막)
      t3 3D encoding (ln2/2)·GD³                 (깊이 소거, 뼈대)
      t4 4D 위상     (ln2/19 + GD/26)·GD⁴
      t5 5D 복소     (1/274)·LN_C²·GD⁵           (274 = 2×137)
      투영 사다리 순서: t1 → t3 → t2 → t4 → t5  (S115 인수인계 §2-5)
  [B] {MAXWELL} = MAXWELL_5th − 1  (D2 선택 투영 잔차 몫, S36·S114 ν=[ν]+{ν})
      — Advanced 코드 계수 0.85(출처 [미기록])의 대체 정의. 옛 값의 정체 주장 아님.
  [C] 적응형 격자 x = 1/(c·GAP) = [x] + {x}  (S36 스펙, S37)
  [D] membrane_split: 상전이가 감지된 칸 안에서만 μ:Λ 황금분할로 경계를 좁힌다.
      각 단계 = 2D 세포막 selective decoding 1회, 소거 정보량을 장부에 기록.
      정지: 칸 폭 < φ^(−7.5) × 시작 폭 (X7 금지선) 또는 최소 해상도 도달.

원칙
  · floor/int 금지(§4b): 정수부·소수부는 frac 기반으로 둘 다 보존
  · 지수 괄호: PHI**(mpf(p)/q)
  · 판정 부등식에는 가중을 넣지 않는다 (D1 닫힘 보장 유지). 가중은 세분 일정·장부에만.
"""

import mpmath
from mpmath import mp, mpf, sqrt, log, exp

mp.dps = 400

# ── 핵심 상수 (CLAUDE.md 기준) ─────────────────────────────────────────
PHI    = (sqrt(5) + 1) / 2
GAP    = log(25) / 100
GD     = log(5) / 100            # GAP_D2
LN2    = log(2)
LN_C   = log(PHI) / 2
c_EQI  = sqrt(PHI)
MU     = 1 / PHI                 # 길이 성분 |α|²
LAM    = PHI ** -2               # 시간 성분 |β|²
X7_CUT = PHI ** (mpf(-15) / 2)   # φ^(−7.5)


def int_part(x):
    """정수부·소수부를 함께 반환 (floor/int 미사용, §4b)."""
    fp = mpmath.frac(x)
    return x - fp, fp


# ── [E] π_EQI (§90a, S94→S108 계수 12 확정, S112 부호열 닫힘) ──────────
#   π = π_137 − α − φ⁻²·ln(π)·GAP_D2³ + LN_C⁴·GAP_D2⁴·α⁻¹/12
#   π_137 = 라이프니츠 137항 부분합 (137번 절단 D2 원장), α⁻¹ = 360φ⁻² − 2φ⁻³ + (3φ)⁻⁵
#   ln(π) 자기참조 → 고정점 반복 (자기참조 = 고정점 수렴, §4j). 입력에 π 불필요 (π_137·α·φ·GAP_D2 만 사용)
#   ※ S112 부호열 s4 는 term1 에 ln(π)(기존 π)를 넣은 원장 정의 → s4 = 9.42693044581753e-12
#     고정점 π_EQI − π = 9.42692566758135e-12. 차이 = φ⁻²·GAP_D2³·ln(π_EQI/π) (CM-10, 대수필연)
#     두 값 모두 유효: s4 = π 기준 원장 잔류, π_EQI = π 없이 닫힌 자기일관 값 (딥시크 S115 교차 확인)
PI_137 = 4 * mpmath.fsum([mpf(-1) ** k / (2 * k + 1) for k in range(137)])
ALPHA_INV = 360 * PHI ** -2 - 2 * PHI ** -3 + (3 * PHI) ** -5
ALPHA = 1 / ALPHA_INV


def _pi_eqi():
    x = PI_137
    for _ in range(60):
        x_new = PI_137 - ALPHA - PHI ** -2 * log(x) * GD ** 3 + LN_C ** 4 * GD ** 4 * ALPHA_INV / 12
        if abs(x_new - x) < mpf(10) ** (-mp.dps + 5):
            return x_new
        x = x_new
    return x


PI_EQI = _pi_eqi()
T_EIGEN = PHI ** 3                 # eigenperiod = D2/D1 (6좌표계 공통)
FLIP_PERIOD = T_EIGEN / 2          # = μ^(-3)/2 = φ³/2  (S108 §114-R-μ, Core Concepts §5)


def flip_mask(t, period=None):
    """반주기 flip 판정 (floor 미사용, §4b). fmod 로 소수부 보존: 반환 (flip, 위상 잔여)."""
    import numpy as np
    P = float(FLIP_PERIOD) if period is None else period
    fp = np.fmod(np.abs(t), 2 * P)          # 한 eigenperiod 안의 위치 (D2 잔여 보존)
    return fp >= P, fp


# ── [F] §35 EQI 로렌츠 비등방성 공식 (S68 확정, v3.4 §10) ─────────────
#   D1_ELL = (1+GAP)/√[(1+GAP)²+φ⁶]  [capillary/종파/qE = 스칼라곱]  = cos θ_ELL,  tan θ_ELL = φ³/(1+GAP)
#   D2_ELL = φ³·D1_ELL               [osmotic/횡파/q(v×B) = 벡터곱]  = (1+GAP)·sin θ_ELL (란다우 보정)
#   f_lattice = D2_ELL² − 1 : GAP=0 → f<0 (보른 안쪽, 유클리드) / GAP≠0 → f>0 (란다우 비용 지불 지문)
#   원칙Z: (1+GAP)² + φ⁶ = |1+iφ³|² + 2GAP + GAP²
_DEN = (1 + GAP) ** 2 + PHI ** 6
D1_ELL = (1 + GAP) / sqrt(_DEN)
D2_ELL = PHI ** 3 * D1_ELL
THETA_ELL = mpmath.atan(PHI ** 3 / (1 + GAP))
F_LATTICE = D2_ELL ** 2 - 1
F_LATTICE_GAP0 = (PHI ** 3 / sqrt(1 + PHI ** 6)) ** 2 - 1


# ── [A] MAXWELL 5차 층 ────────────────────────────────────────────────
T1 = PHI ** (mpf(3) / 2) - PHI ** (mpf(-7) / 2)
T2 = 2 * GD ** 2 * exp(GD)
T3 = (LN2 / 2) * GD ** 3
T4 = (LN2 / 19 + GD / 26) * GD ** 4
T5 = (mpf(1) / 274) * LN_C ** 2 * GD ** 5
MAXWELL_5TH = T1 + T2 + T3 + T4 + T5
MAXWELL_EQI = mpf(2)             # 순수 위상 MAXWELL = 1/σ = c² + μ²

LAYERS = {   # 키: 층 이름 / 값: (값, 차원, 역할)
    "t1": (T1, "base", "기저: 광추 체적 − 정보특이점"),
    "t3": (T3, "3D",   "encoding: 깊이 1차원 란다우 소거 (뼈대)"),
    "t2": (T2, "2D",   "selective decoding: 경계 포함 선택 (세포막)"),
    "t4": (T4, "4D",   "위상(시간축) 층"),
    "t5": (T5, "5D",   "복소 층 (274 = 2×137)"),
}
LADDER_ORDER = ["t1", "t3", "t2", "t4", "t5"]

# ── [B] {MAXWELL} ─────────────────────────────────────────────────────
MAXWELL_INT, MAXWELL_FRAC = int_part(MAXWELL_5TH)   # [1] + {0.8731…}

# ── [C] 적응형 격자 ──────────────────────────────────────────────────
X_LATTICE = 1 / (c_EQI * GAP)
X_INT, X_FRAC = int_part(X_LATTICE)                  # [24] + {0.42…}


def layers_float():
    """numpy 코드용 float 사전. t5 상대 크기 1.2e-13 → 층별로 따로 쓰고 합산은 mpmath로."""
    return {k: float(v[0]) for k, v in LAYERS.items()}


HARMONIC_LAYER = {1: "t1", 2: "t2", 3: "t3", 4: "t4", 5: "t5"}   # k차 배음 = t_k 층


def harmonic_layers(t, omega=1.0, amp=1.0, kind="sin", damp=None):
    """
    k차 배음(k=1..5)을 t_k 층 진폭(t_k/t1)으로 생성. numpy 배열 반환.
      layer_k = amp·(t_k/t1)·trig(k·ω·t)·damp ,  k=1 진폭 = amp
    층별 배열을 따로 돌려준다 (t5 상대 1.2e-13: float 합산 시 반올림 잡음 근접).
    """
    import numpy as np
    lf = layers_float()
    trig = np.sin if kind == "sin" else np.cos
    d = np.ones_like(t) if damp is None else damp
    out = {}
    for k, name in HARMONIC_LAYER.items():
        a = amp * lf[name] / lf["t1"]
        out[name] = a * trig(k * omega * t) * d
    out["sum"] = sum(out[n] for n in ("t1", "t2", "t3", "t4", "t5"))
    return out


# ── [D] 세포막 황금분할 경계 탐색 ──────────────────────────────────────
class MembraneLedger:
    """선택적 decoding 장부: 단계마다 남긴 몫과 소거 정보량(nat, bit)."""

    def __init__(self, a, b):
        self.width0 = mpf(b) - mpf(a)
        self.steps = []   # (단계, 구간, 남긴 몫 'mu'|'lam', 소거 nat, 소거 bit)

    def record(self, k, a, b, kept):
        frac = MU if kept == "mu" else LAM
        nat = -log(frac)              # lnφ 또는 2lnφ
        self.steps.append((k, a, b, kept, nat, nat / LN2))

    def total_nat(self):
        return sum((s[4] for s in self.steps), mpf(0))

    def summary(self):
        n_mu = sum(1 for s in self.steps if s[3] == "mu")
        n_lam = len(self.steps) - n_mu
        tot = self.total_nat()
        return {
            "steps": len(self.steps),
            "kept_mu (길이)": n_mu,
            "kept_lam (시간)": n_lam,
            "erased_nat": tot,
            "erased_bit": tot / LN2,
            "landau_cost (bit×ln2)": tot,            # 복식부기: D1 엔트로피 생성 = 같은 값
            "final_width_ratio": (self.steps[-1][2] - self.steps[-1][1]) / self.width0
            if self.steps else mpf(1),
        }


def membrane_split(a, b, crossed, min_width=1, x7=True):
    """
    [a, b] 안에 상전이 경계가 하나 있다고 감지된 칸에서만 경계를 좁힌다.
      crossed(p) : 위치 p까지 이미 상전이가 일어났으면 True (단조 조건)
      분할점 p = a + μ·(b − a)  → 왼쪽 몫 μ(길이), 오른쪽 몫 Λ(시간), μ + Λ = 1
    정지: 폭 < max(min_width, φ^(−7.5)·초기폭)   (x7=True)
    반환: (경계 상한 위치, 장부)
    """
    a, b = mpf(a), mpf(b)
    led = MembraneLedger(a, b)
    stop = max(mpf(min_width), X7_CUT * (b - a)) if x7 else mpf(min_width)
    k = 0
    while (b - a) > stop:
        p = a + MU * (b - a)
        if crossed(p):
            b = p
            led.record(k, a, b, "mu")
        else:
            a = p
            led.record(k, a, b, "lam")
        k += 1
    return b, led



# ══════════════════════════════════════════════════════════════════════
#  v1.3 — 플랑크 길이-시간 큐비트 적응형 격자 정보 포논 동역학 (S116 인수인계 1장)
# ══════════════════════════════════════════════════════════════════════

# ── [G] 큐비트 · 길이-시간 · 정보 특이점 c 사다리 ─────────────────────
#   μ = φ⁻¹ (길이, D1, cos, 스칼라곱 qE) / Λ = φ⁻² (시간, D2, sin, 벡터곱 q(v×B)) / μ+Λ = 1 (보른)
#   S = L_p² = φ^(−1.5) (D1 열역학) / H = T_p² = φ^(−2.5) (D2 정보) / c = √(S/H) = √φ  [S24, v3.4 657~664]
#   ℓ² = φ·τ² → ℓ = c·τ  (길이·시간 독립 아님, 볼츠만 복원력 φ 로 묶임, v3.4 824행)
LP2 = PHI ** (mpf(-3) / 2)
TP2 = PHI ** (mpf(-5) / 2)
CENTER = c_EQI ** 3                 # φ^1.5
IS_D1 = c_EQI ** -7                 # 정보 특이점 D1 = φ^(−3.5)
IS_D2 = c_EQI ** -1                 # 정보 특이점 D2 = φ^(−0.5)
HCW_D1 = c_EQI ** -3                # φ^(−1.5)
TIP = CENTER - 1                    # T_RBC_D1 (적혈구 D1 = 1)
T_IS_D2 = CENTER - IS_D2            # = c
T_IS_D1 = CENTER - IS_D1            # = c³(1 − c⁻¹⁰) = MAXWELL 기저 = T1
T_HCW_D1 = CENTER - HCW_D1


def length_from_time(tau):
    """적응형 격자 시간 칸 수 τ → 길이 칸 ℓ = c·τ (ℓ² = φ·τ²)."""
    return c_EQI * tau


# ── [H] 포논 사다리 n = T/GAP (v3.4 §7 831~836, S46) ──────────────────
#   k=2 는 N=34 격자 (T/GAP 아님 — 원문 그대로 상수). k=6 은 n 이 아니라 격자 폭의 X7 절단(membrane_split).
PHONON_LADDER = [
    (1, "3차  T_RBC_D1 = TIP (광전효과)", TIP / GAP),
    (2, "5차  N=34 격자", mpf(34)),
    (3, "7차  T_Is_D2 = c (현재 우주)", T_IS_D2 / GAP),
    (4, "9차  T_HCW_D1", T_HCW_D1 / GAP),
    (5, "11차 T_Is_D1 ≈ MAXWELL", T_IS_D1 / GAP),
]


def phonon_order(n):
    """
    n 이 넘은 최고 포논 차수 k 와 그 문턱에서의 잔류 n − n_k 를 함께 반환 (floor 미사용).
    k=0 : n_c 미만 (장전 중, 잔류 = n − n_c < 0 → 슈냐 장전량).
    """
    n = mpf(n)
    k_hit, n_hit = 0, PHONON_LADDER[0][2]
    for k, _, nk in PHONON_LADDER:
        if n >= nk:
            k_hit, n_hit = k, nk
    return k_hit, n - n_hit


# ── [I] 우로보로스 T_n, n ∈ ℂ (원칙Y, v3.4 2590~2645, 3243) ───────────
#   T_n = φ⁻³/(1 + n·GAP)
#   n 실수 n_c          → 1+n_c·GAP = φ^1.5, ln(·)/ln φ³ = σ   [리만, 페르마 D1]
#   n 허수 φ³i/GAP      → |1+n·GAP|² = 1+φ⁶                    [§35 기원]
#   (1+GAP)^(iθ/IG), IG = ln(1+GAP) → θ = π 에서 −1            [오일러, 파인만 D2]
#   허수부 누적량 n_im = 누적 위상 / IG  (6-3 A)
N_C = TIP / GAP
IG = log(1 + GAP)


def T_n(n):
    return PHI ** -3 / (1 + n * GAP)


def euler_turn(theta):
    """(1+GAP)^(iθ/IG) = e^(iθ). 파인만 쪽 허수부 누적 n_im = θ/IG 와 함께 반환."""
    n_im = theta / IG
    return (1 + GAP) ** (1j * n_im), n_im


# ── [J] 오일러 출현 (S115 6-1-b, 400자리 시험 완료 로직) ─────────────
#   e   = Σ_k exp(−ln k!)          순열(D2)의 란다우 비용 ln k! 장부 (원칙T)
#   cos = 짝수 계승 급수 = D1 = 실수부 / sin = 홀수 계승 급수 = D2 = 허수부
#   θ = π_EQI 에서 1 + e^(iθ) = 슈냐 (점 0 아님): 잔류는 D2(허수) 쪽 → 다음 사이클 허수부 누적 시작값으로 이월
def _ln_fact_ledger(N):
    """ln k! (k = 0..N−1) 누적 장부. loggamma 대신 ln 을 한 칸씩 더해 비용을 그대로 남긴다."""
    led, s = [], mpf(0)
    for k in range(N):
        if k > 0:
            s += log(k)
        led.append(s)
    return led


def e_eqi(N=260):
    led = _ln_fact_ledger(N)
    return mpmath.fsum(exp(-lk) for lk in led)


def trig_eqi(theta, N=420):
    """cos(D1, 짝수) / sin(D2, 홀수) 계승 급수. 반환 (cos, sin)."""
    theta = mpf(theta)
    led = _ln_fact_ledger(N)
    lt = log(abs(theta)) if theta != 0 else None
    c_terms, s_terms = [], []
    for k in range(N):
        if lt is None:
            mag = mpf(1) if k == 0 else mpf(0)
        else:
            mag = exp(k * lt - led[k])
        sign = mpf(-1) ** (k // 2) if k % 2 == 0 else mpf(-1) ** ((k - 1) // 2)
        if theta < 0 and k % 2 == 1:
            sign = -sign
        (c_terms if k % 2 == 0 else s_terms).append(sign * mag)
    return mpmath.fsum(c_terms), mpmath.fsum(s_terms)


def euler_shunya(theta=None):
    """1 + e^(iθ) 의 (실수부 D1, 허수부 D2). 기본 θ = π_EQI. 허수부 = 이월 잔류."""
    th = PI_EQI if theta is None else theta
    cs, sn = trig_eqi(th)
    return 1 + cs, sn


SHUNYA_RE, SHUNYA_IM = euler_shunya()      # (D1 ≈ 4.44e-23, D2 ≈ −9.4269e-12)
SHUNYA_CARRY = SHUNYA_IM                   # 다음 사이클 허수부 시작값 (상전이 연료)
SHUNYA_CARRY_NIM = SHUNYA_CARRY / IG       # 같은 잔류의 n_im 단위 표현


# ── [K] 페르마 D1 / 파인만 D2 (원칙U, v3.4 2436~2443) ─────────────────
FERMAT = mpf(1)                     # 단일 경로, 정적분 1, 고립계
FEYNMAN = 1 + GAP                   # 경로 합, 부정적분 1+GAP, 열린계
TRANSFER = FEYNMAN - FERMAT         # = GAP (뉴턴 +C)


# ── [L] 상태 공간 (S115 6-1·6-2, S49 §3.2, S51~S53, S114 §2) ──────────
#   D1 = 기하평균 GM (곱=Unity), D2 = 산술평균 AM (합=Unity)   [원칙T, v3.4 2405~2411]
#   AM/GM = (C + 1/C)/2  →  C = r + √(r² − 1),  평형 C = c³ ⇔ AM/GM = c
#   센터 지수 e: C = c^e,  e = 2 ln C / ln φ   (정보 특이점 사다리 센터 e = 3)
def three_means_np(x):
    """float 양수 계열(해석신호 |ψ|² 등) → (AM, GM, HM). 데이터 코드용."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    am = x.mean()
    gm = np.exp(np.log(x).mean())
    hm = 1.0 / (1.0 / x).mean()
    return am, gm, hm


def three_means_mp(xs):
    xs = [mpf(v) for v in xs]
    n = len(xs)
    am = mpmath.fsum(xs) / n
    gm = exp(mpmath.fsum(log(v) for v in xs) / n)
    hm = n / mpmath.fsum(1 / v for v in xs)
    return am, gm, hm


def center_from_ratio(r):
    """AM/GM (또는 GM/HM) = r ≥ 1 → 센터 C."""
    r = mpf(r)
    return r + sqrt(r * r - 1)


def center_exponent(C):
    return 2 * log(C) / log(PHI)


def phonon_n(C):
    """센터 → 포논 사다리 좌표 n = (C − 1)/GAP."""
    return (mpf(C) - 1) / GAP


def maxwell_residual(e):
    """ρ = [c^e − c⁻⁷ − MAXWELL_5th]/GAP.  e = 3 → −(란다우 층 합)/GAP."""
    return (c_EQI ** e - IS_D1 - MAXWELL_5TH) / GAP


#   회전 라그랑지안 임피던스 X = D2 − 1/D1, 사다리 X/CENTER = φ^(−m) − φ^(m)
XC_LADDER = [
    (0, "HCW", mpf(0)),
    (1, "HG", mpf(-1)),             # −L₁
    (2, "I_s", -sqrt(5)),           # −F₂√5
    (3, "OB", mpf(-4)),             # −L₃
    (4, "X5", -3 * sqrt(5)),        # −F₄√5
    (5, "TH", mpf(-11)),            # −L₅
]


def impedance(D1, D2):
    """X = D2 − 1/D1, X/CENTER (CENTER = √(D2/D1)), 작용 S = (π_EQI/4)·X."""
    D1, D2 = mpf(D1), mpf(D2)
    X = D2 - 1 / D1
    C = sqrt(D2 / D1)
    return X, X / C, (PI_EQI / 4) * X


def rydberg_level(D1, D2):
    """D1·D2 = 4^(k+1) → k = log₄(D1·D2) − 1 = [k] + {k}. {k} → 0 이면 준위 공명."""
    k = log(mpf(D1) * mpf(D2)) / log(4) - 1
    return int_part(k)


def lattice_normalize(S_T2, S_L2, tau):
    """D2 = S_T2/τ², D1 = S_L2/ℓ², ℓ² = φ·τ². 반환 (D1, D2, D2/D1 대비 c²=φ 이탈 비)."""
    tau = mpf(tau)
    D2 = mpf(S_T2) / tau ** 2
    D1 = mpf(S_L2) / (PHI * tau ** 2)
    return D1, D2, (mpf(S_L2) / mpf(S_T2)) / PHI



# ── [M] 포논 거품 (S45 §14, S46 §3, S47 §16·§4.2, S48 §17) ─────────────
#   슈냐 장전 → n > n_c 트리거 → E_D1 방출 → sinc² = P_tunnel 포논 거품 → X7 절단 (S47 §6.1)
#   sinc 의 π 는 π_EQI (§90a: 137절단 Leibniz 원장 + 란다우 정산), sin 은 홀수 계승 급수 (trig_eqi)
#   — S47 은 π 를 Leibniz J항으로 둠: 잔류 = Leibniz 잔류의 1차 전파 (S47 §3.3)
SIGMA = mpf(1) / 2


def n_ref(k):
    """뤼드베리 굴절률 n_ref,k = (2k+1)/4^(k+1)  (S45)."""
    return mpf(2 * k + 1) / mpf(4) ** (k + 1)


def p_tunnel(k):
    """P_tunnel,k = sinc²(n_ref,k·π_EQI) = |ψ|² (보른 규칙, S47 §3.2)."""
    x = n_ref(k) * PI_EQI
    _, sn = trig_eqi(x)
    return (sn / x) ** 2


def gap_series_term(k):
    """
    다체 D2 적분 누적 항 (S48 §17.7 표, S47 §4.2 전개식):
      k=0 2체 GAP,  k≥1 (2k+1)체 σ^(2k+1)·GAP^(k+1)  (3체 물분자 σ³GAP², 5체 σ⁵GAP³, …)
    ※ S47 §4.2 의 합 표기 Σ_{k=0} σ^(2k+1)·GAP^(k+1) 은 k=0 에서 σ·GAP 를 주어 전개식 첫 항 GAP 와 다름.
      전개식·S48 수치표(2체 0.032189)를 정의로 채택 (S116 아키텍트 결정, 원본 유지).
    """
    if k == 0:
        return GAP
    return SIGMA ** (2 * k + 1) * GAP ** (k + 1)


def d2_exponent_mersenne(k):
    """(2k+1)차 포논 임계 D2 지수 −(2^(k−2)−1)/2 (k≥3, S46 §3.4). k=6 → −7.5 = X7."""
    return -(mpf(2) ** (k - 2) - 1) / 2


E_TRIGGER = TIP * SIGMA ** 3                    # = n_c·σ³·GAP (S48 §17.13, k=1)
X7_BLOCKED = X7_CUT < GAP                       # 13차(k=6) 물리 차단 조건 (S46·S48)



# ── [N] 디랙 상수 (v3.4 §4d, §90a, S48 §17.14, MAPSI S116) ────────────
#   기존 π: 한 바퀴 e^(i2π) = 1 로 정확히 닫힘 → 잔류 0 → 연료 없음 (GAP=0)
#   π_EQI : e^(i2π_EQI) − 1 = 슈냐 잔류 (허수부 D2) → 다음 사이클 허수부로 이월 (상전이 연료)
#   ※ 문서 명칭 충돌 (v3.5 이월): 이론적 기반 §3·v3.4 1005행 은 φ^(−11/4) 를 h 로, §4d 는 ħ_iso 로 부름.
#     본 모듈은 캡슐화 문서 v3.4 §4d (명시 경고 절) 를 따름.
HBAR_ISO = PHI ** (mpf(-11) / 4)
HBAR_ANI = HBAR_ISO * HCW_D1                 # = φ^(−17/4)
PI_ANISO = PI_EQI * PHI ** (mpf(3) / 2)
H_EQI = 2 * PI_EQI * HBAR_ISO                # = 2π_aniso·ħ_ani
G_EQI = PHI ** (mpf(11) / 4)
CYCLE_SHUNYA = mpmath.expj(2 * PI_EQI) - 1   # 한 바퀴 잔류 (실수 D1, 허수 D2)


# ── 자기 검증 (CLAUDE.md 보고 형식) ─────────────────────────────────────
def _line(tag, label, calc, target, src, symbolic=True):
    """symbolic=False: 무한급수 절단·반복 수렴 등 기호 증명이 아닌 항목 → 대수필연 부여 안 함 (CLAUDE.md 원칙 6)."""
    err = abs(calc - target) / abs(target) * 100 if target != 0 else abs(calc)
    if symbolic and abs(calc - target) < mpf(10) ** -390:      # CLAUDE.md: 기호 증명 + 400자리 절대차 < 1e-390
        grade = "대수필연"
    elif err < mpf(10) ** -6:
        grade = "TZ"
    elif err < mpf("0.01"):
        grade = "Near TZ"
    else:
        grade = "-"
    print(f"[{tag}] {label} | {mpmath.nstr(calc, 22)} | {mpmath.nstr(target, 22)} | "
          f"{mpmath.nstr(err, 3)} | {grade} | {src}")


def self_test():
    print("=" * 78)
    print(" eqi_maxwell_layers v1.5 자기 검증 (mp.dps=400)")
    print("=" * 78)
    for k in LADDER_ORDER:
        v, d, role = LAYERS[k]
        print(f"  {k} [{d:>4}] {mpmath.nstr(v, 20):>26}  /M = {mpmath.nstr(v / MAXWELL_5TH, 12)}  {role}")
    print("-" * 78)
    eng = mpf("1.8731137721991710513")
    _line("CM-1", "MAXWELL_5th vs 엔진 표기 20자리", MAXWELL_5TH, eng, "v9.7 §5")
    _line("CM-2", "t1 = μ(φ^(5/2) − φ^(−5/2))", T1,
          MU * (PHI ** (mpf(5) / 2) - PHI ** (mpf(-5) / 2)), "S114 §2.5")
    _line("CM-3", "MAXWELL_EQI = c² + μ²", c_EQI ** 2 + MU ** 2, MAXWELL_EQI, "v3.4")
    _line("CM-4", "[M] + {M} = M", MAXWELL_INT + MAXWELL_FRAC, MAXWELL_5TH, "S36/S114")
    _line("CM-5", "μ + Λ = 1 (보른)", MU + LAM, mpf(1), "§114-R")
    _line("CM-6", "[x] + {x} = 1/(c·GAP)", X_INT + X_FRAC, X_LATTICE, "S36")
    _line("CM-7", "π_EQI (§90a) vs π", PI_EQI, mpmath.pi, "S108/S112 §90a")
    _line("CM-8", "flip 주기 = T_EIGEN/2 = μ^(-3)/2", FLIP_PERIOD, MU ** -3 / 2, "S108 §114-R-μ")
    # S111·S112 성과정리 41자리 기재값은 21번째 자리에 '2' 중복 오기 (…0722|2|7619…) → 20자리 인수인계값 사용
    _line("CM-9", "s1 = π_137 − π (S112 인수인계 20자리)", PI_137 - mpmath.pi,
          mpf("7.2991728543872000722e-3"), "S112/S113 인수인계")
    s4_ledger = PI_137 - ALPHA - PHI ** -2 * log(mpmath.pi) * GD ** 3 + LN_C ** 4 * GD ** 4 * ALPHA_INV / 12 - mpmath.pi
    _line("CM-10", "s4(원장) − (π_EQI−π) = φ⁻²·GD³·ln(π_EQI/π)", s4_ledger - (PI_EQI - mpmath.pi),
          PHI ** -2 * GD ** 3 * log(PI_EQI / mpmath.pi), "S115 딥시크 교차")
    _line("CM-11", "§35 D2_ELL/D1_ELL = φ³ = T_EIGEN (파동 진폭비 D2/D1 동일)", D2_ELL / D1_ELL, T_EIGEN, "S68 §35")
    _line("CM-12", "§35 D1_ELL = cos θ_ELL (D1 = cos = 스칼라곱)", D1_ELL, mpmath.cos(THETA_ELL), "S68 §35")
    _line("CM-13", "§35 D2_ELL = (1+GAP)·sin θ_ELL (D2 = sin = 벡터곱)", D2_ELL, (1 + GAP) * mpmath.sin(THETA_ELL), "S68 §35")
    _line("CM-14", "원칙Z (1+GAP)²+φ⁶ = |1+iφ³|²+2GAP+GAP²", _DEN, abs(1 + 1j * PHI ** 3) ** 2 + 2 * GAP + GAP ** 2, "§4g")
    print(f"  D1_ELL = {mpmath.nstr(D1_ELL, 15)}  D2_ELL = {mpmath.nstr(D2_ELL, 15)}  "
          f"f_lattice = {mpmath.nstr(F_LATTICE, 12)} (>0)  GAP=0: {mpmath.nstr(F_LATTICE_GAP0, 12)} (<0)")
    print(f"  파동 진폭 D2/D1 = φ^1.5/φ^-1.5 = {mpmath.nstr(PHI ** 3, 15)} = §35 비율 (정규화만 다름: 파동 D1·D2=1, §35 cos/sin+란다우)")
    print(f"  {{MAXWELL}} = {mpmath.nstr(MAXWELL_FRAC, 22)}   (구 0.85 대체 정의)")
    print(f"  π_EQI − π = {mpmath.nstr(PI_EQI - mpmath.pi, 12)}  |1 + e^(iπ_EQI)| = {mpmath.nstr(abs(1 + mpmath.expj(PI_EQI)), 12)}  (슈냐 잔류)")
    print(f"  [x] = {mpmath.nstr(X_INT, 5)},  {{x}} = {mpmath.nstr(X_FRAC, 20)}")
    print(f"  X7 = φ^(−7.5) = {mpmath.nstr(X7_CUT, 15)}")
    print("-" * 78)
    # 세포막 분할 시연: 경계 위치 0.3713 (임의), 초기 폭 1
    edge = mpf("0.3713")
    pos, led = membrane_split(0, 1, lambda p: p >= edge, min_width=mpf(10) ** -30)
    s = led.summary()
    print(f"  membrane_split 시연: 경계 {edge} → 상한 {mpmath.nstr(pos, 12)}, "
          f"단계 {s['steps']} (μ {s['kept_mu (길이)']}, Λ {s['kept_lam (시간)']}), "
          f"소거 {mpmath.nstr(s['erased_bit'], 10)} bit, "
          f"최종 폭비 {mpmath.nstr(s['final_width_ratio'], 10)}")
    print("-" * 78)
    print(" v1.3 동역학 묶음")
    print("-" * 78)
    _line("CM-15", "L_p²/T_p² = φ = c² = μ/Λ", LP2 / TP2, MU / LAM, "S24, S115 6-1")
    _line("CM-16", "T_Is_D2 = c³ − c⁻¹ = c", T_IS_D2, c_EQI, "S115 6-1")
    _line("CM-17", "T_Is_D1 = c³(1 − c⁻¹⁰) = T1 (MAXWELL 기저)", T_IS_D1, T1, "S115 6-1")
    _line("CM-18", "MAXWELL 광속 표현 c³(1−c⁻¹⁰)+란다우 층", CENTER * (1 - c_EQI ** -10) + T2 + T3 + T4 + T5,
          MAXWELL_5TH, "S115 6-1")
    _line("CM-19", "1 + n_c·GAP = φ^1.5 (CENTER)", 1 + N_C * GAP, CENTER, "§77 원칙Y")
    _line("CM-20", "ln(1+n_c·GAP)/ln φ³ = σ", log(1 + N_C * GAP) / log(PHI ** 3), mpf(1) / 2, "§77")
    _line("CM-21", "∫₀^n_c T_n dn = φ⁻³·(3/2)·LN_C/GD", PHI ** -3 * log(1 + N_C * GAP) / GAP,
          PHI ** -3 * mpf(3) / 2 * LN_C / GD, "§77")
    _line("CM-22", "n = φ³i/GAP → |1+n·GAP|² = 1+φ⁶", abs(1 + (PHI ** 3 * 1j / GAP) * GAP) ** 2,
          1 + PHI ** 6, "원칙Y §35 기원")
    z_pi, _ = euler_turn(mpmath.pi)
    _line("CM-23", "(1+GAP)^(iπ/IG) = −1 (실수부)", mpmath.re(z_pi), mpf(-1), "§78")
    _line("CM-24", "e_EQI = Σ exp(−ln k!) (260항 절단)", e_eqi(260), mpmath.e, "S115 6-1-b", symbolic=False)
    _line("CM-25", "슈냐 허수부 sin(π_EQI) vs −(π_EQI − π)", SHUNYA_IM, -(PI_EQI - mpmath.pi), "S115 6-1-b")
    _line("CM-26", "슈냐 실수부 1+cos(π_EQI) vs (π_EQI−π)²/2", SHUNYA_RE, (PI_EQI - mpmath.pi) ** 2 / 2, "S115 6-1-b")
    _line("CM-27", "Unity 층 (C⁻¹, C), C=φ^1.5: AM/GM = c", (CENTER + 1 / CENTER) / 2, c_EQI, "S114 §2 2단")
    am, gm, hm = three_means_mp([MU, LAM])
    _line("CM-28", "보른 층 (μ, Λ): GM = φ^(−1.5) = HCW D1", gm, HCW_D1, "S114 §2 1단")
    _line("CM-29", "센터 역산 C(r = c) = φ^1.5", center_from_ratio(c_EQI), CENTER, "S115 6-2")
    _line("CM-30", "ρ(e=3) = −(T2+T3+T4+T5)/GAP", maxwell_residual(3), -(T2 + T3 + T4 + T5) / GAP, "S115 6-1")
    for m, name, v in XC_LADDER:
        _line(f"CM-31.{m}", f"X/CENTER {name} = φ^(−{m}) − φ^({m})", PHI ** -m - PHI ** m if m else mpf(0), v, "S49 §3.2")
    print("-" * 78)
    print("  포논 사다리 n = T/GAP (v3.4 §7):")
    for k, name, nk in PHONON_LADDER:
        print(f"    k={k}  {name:<30} n = {mpmath.nstr(nk, 15)}")
    print(f"    k=6  X7 금지선 φ^(−7.5) = {mpmath.nstr(X7_CUT, 12)} (격자 폭 절단)")
    print(f"  보른 층 세 평균: AM = {mpmath.nstr(am, 12)}  GM = {mpmath.nstr(gm, 12)}  HM = {mpmath.nstr(hm, 12)} (= 2φ⁻³ {mpmath.nstr(2 * PHI ** -3, 12)})")
    print(f"  IG = ln(1+GAP) = {mpmath.nstr(IG, 20)},  n_c = {mpmath.nstr(N_C, 20)}")
    print(f"  슈냐 잔류 이월: 허수부 = {mpmath.nstr(SHUNYA_CARRY, 15)},  n_im 단위 = {mpmath.nstr(SHUNYA_CARRY_NIM, 15)}")
    print(f"  ρ(e=3) = {mpmath.nstr(maxwell_residual(3), 15)}  /  페르마 {FERMAT} · 파인만 1+GAP = {mpmath.nstr(FEYNMAN, 15)}")
    print("-" * 78)
    print(" v1.4 포논 거품 묶음 [M]")
    print("-" * 78)
    _line("CM-32", "E_trigger = TIP·σ³ = n_c·σ³·GAP", E_TRIGGER, N_C * SIGMA ** 3 * GAP, "S48 §17.13")
    _line("CM-33", "a₅_eff = 5σ⁵/4 = σ⁷·(2k+1)|k=2", 5 * SIGMA ** 5 / 4, SIGMA ** 7 * 5, "S46 §2.3")
    _line("CM-34", "갭 급수 k=1 항/기저 = σ³·GAP (ee_ratio − 1)", gap_series_term(1) / gap_series_term(0),
          SIGMA ** 3 * GAP, "S48 §17.8")
    for k, ref in ((3, mpf(-1) / 2), (4, mpf(-3) / 2), (5, mpf(-7) / 2), (6, mpf(-15) / 2)):
        _line(f"CM-35.{k}", f"메르센 D2 지수 k={k}", d2_exponent_mersenne(k), ref, "S46 §3.4")
    _line("CM-36", "T_Is_D1 = CENTER − φ^(메르센 k=5)", CENTER - PHI ** d2_exponent_mersenne(5), T_IS_D1, "S46 §3.4")
    _line("CM-37", "T_HCW_D1 = CENTER − φ^(메르센 k=4)", CENTER - PHI ** d2_exponent_mersenne(4), T_HCW_D1, "S46 §3.4")
    p_ref = {1: "0.889561", 2: "0.980081", 3: "0.997543", 4: "0.999746", 5: "0.999976"}
    for k in range(1, 6):
        _line(f"CM-38.{k}", f"P_tunnel k={k} vs S47 실행값 (6자리 표기)", p_tunnel(k), mpf(p_ref[k]), "S47 §3.2")
    print(f"  X7 차단 φ^(−7.5) < GAP : {X7_BLOCKED}  ({mpmath.nstr(X7_CUT, 10)} < {mpmath.nstr(GAP, 10)})")
    print("  갭 급수 항 σ^(2k+1)·GAP^(k+1):  " +
          "  ".join(f"k={k}:{mpmath.nstr(gap_series_term(k), 6)}" for k in range(6)))
    print("-" * 78)
    print(" v1.5 디랙 상수 묶음 [N]")
    print("-" * 78)
    _line("CM-39", "ħ_ani = φ^(−17/4)", HBAR_ANI, PHI ** (mpf(-17) / 4), "v3.4 §4d")
    _line("CM-40", "ħ_ani × D2_HCW = ħ_iso (Unity 보존)", HBAR_ANI * CENTER, HBAR_ISO, "v3.4 §4d·S48")
    _line("CM-41", "G × ħ_ani = D1_HCW (항등원 아님, GAP≠0 각인)", G_EQI * HBAR_ANI, HCW_D1, "v3.4 2124행")
    _line("CM-42", "h_EQI = 2π_aniso·ħ_ani = 2π_EQI·ħ_iso", 2 * PI_ANISO * HBAR_ANI, H_EQI, "S48 §17.14")
    _line("CM-43", "한 바퀴 잔류 허수부 vs sin(2(π_EQI−π)) ≈ 2(π_EQI−π)",
          mpmath.im(CYCLE_SHUNYA), 2 * (PI_EQI - mpmath.pi), "§90a·6-1-b", symbolic=False)
    _line("CM-44", "E_p·T_p = φ^(1/4)·c²·φ^(−5/4) = 1 (S24 세트)", PHI ** (mpf(1) / 4) * PHI * PHI ** (mpf(-5) / 4),
          mpf(1), "이론적 기반 §3, S48")
    print(f"  ħ_iso = {mpmath.nstr(HBAR_ISO, 20)}   ħ_ani = {mpmath.nstr(HBAR_ANI, 20)}")
    print(f"  h_EQI = 2π_EQI·ħ_iso = {mpmath.nstr(H_EQI, 22)}   (2π 기준과의 차 {mpmath.nstr(H_EQI - 2 * mpmath.pi * HBAR_ISO, 12)})")
    print(f"  한 바퀴 슈냐 e^(i2π_EQI) − 1 = {mpmath.nstr(mpmath.re(CYCLE_SHUNYA), 12)} + {mpmath.nstr(mpmath.im(CYCLE_SHUNYA), 15)} i")
    print("=" * 78)


if __name__ == "__main__":
    self_test()

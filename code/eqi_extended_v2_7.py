"""
EQI S118 — 확장 번역 코드 v2.7 (법칙 일반형 + 엔진 다리 + §4d + 컴프턴 + 볼츠만 + 란다우어)
  v2.6 → v2.7 (perplexity 잔여 표기 3+2건, 계산 정의·기준값·허용오차 불변, v2.6 원본·로그 보존)
    S-5 설명 "한 바퀴 직후" (π_EQI + π = 2π + ε, ε > 0) / S-4 → S-4a~S-4e 고유 태그 /
    ledger 설명 "법칙별 X 단위 환산은 X0 를 곱함" / 실행 파일명 갱신 / §4d 미적용 줄 = v2 당시 상태 명시
  v2.5 → v2.6 (perplexity v2.4 검토 반영, 이전 판 원본·로그 보존, 수치·O 관찰 불변)
    ① 란다우어 두 장부 분리
       [고정 K0 장부] 열에너지 K0 = kT° 고정, 비용 K0·C = K0·(I + D) (C·I·D 는 K0 단위 수)
       [확장 K(n) 장부] 열에너지 K(n) = K0·Z(n), 1비트 하한 K(n)·ln 2
    ② LD-7 η(n→0) → 1 = 대수적 극한 (근거). Jun 2014 긴 프로토콜 시간 극한과의 동일시는 별도 해석 (I) 줄
  v2.4 → v2.5 (perplexity v2.3 검토 반영, v2.3·v2.4 원본·로그 보존)
    ① 볼츠만: "Σp = Z" → 가중치 합 Σw = Z(n), 확률 합 Σp = 1 로 구분
    ② 볼츠만 계 A·B·Z_aniso 연결 = 같은 계가 아니라 "분배함수 값의 접점" 으로 명시
    ③ 컴프턴 이동: 환산 기준 (ƛ, ħ_iso) 과 한 바퀴 기준 (λ = 2π_EQI·ƛ, H_EQI) 을 따로 출력·검산
  v2.3 → v2.4: 란다우어 소거 확장 번역, 검산 묶음 LD
    원문 (perplexity S118 1차 조사, Type IV): Sagawa–Ueda 2009 PRL 102, 250602 Eq. (2)–(4)
      W_meas ≥ −kT(H − I) + ΔF^M,  W_eras ≥ kT·H − ΔF^M,  W_meas + W_eras ≥ kT·I
    비율 1 (대칭 기억, ΔF^M = 0, 1비트 H = ln 2): W_eras / (kT·ln 2) = 1 → Z(n)
    EQI 대응: kT(n) = kT°·Z(n) (열 단위가 사이클로 열림), 세 장부 C / I / D 는 2009 Eq. (4) 꼴
      (C = 측정·소거 총비용, I = 정보, D = 여유) — 원문 정의가 아니라 EQI 대응 (I 표시)
    실험 앵커 (kT 단위): Jun 2014 PRL 113, 190601 완전 소거 0.71 ± 0.03 kT (외삽), Bérut 2012 Nature 483, 187
  v2.2 → v2.3: 볼츠만 분포 확장 번역 (S116 번역 v4 §13 기등록 위에), 검산 묶음 BZ
    Z(n) = 1 + n·GAP 를 분배함수 (가중치 합 Σw) 로 읽음, 확률 합 Σp = 1: 바닥 준위 (무게 1, 페르마 단일 경로) + n 겹 들뜬 준위
    (볼츠만 인자 e^(−Δ/kT) = GAP, Δ/kT = |ln GAP|). 사이클 1 회 = 무게 GAP 인 경로 하나 추가 (원칙U 파인만 경로 합)
  v2.1 → v2.2
    ① 상수명 분리: A_S24 = φ^(−11/4) (S24 "h", 엔진 h_EQI) = HBAR_ISO (공통 모듈, 수치 동일·이름만 다름)
       HBAR_ANI = φ^(−17/4) (진성 ħ), H_EQI = 2π_EQI·HBAR_ISO (한 바퀴 작용)
       근거: S116 성과정리 §3 MAPSI 확정 (ħ = 진성 ħ_ani, π 는 한 바퀴 작용 h 로 갈 때만)
             S116 번역 v4 머리말 (ħ_iso 결과 = 등방성 투영 병기)
    ② S24 71줄은 그대로 (회귀 기준). §4d 는 별도 LAWS (…_ani) 와 검산 묶음 D4 로 추가
    ③ 컴프턴: S116 번역 v4 §18-B (기등록) 위에 확장 번역 (λ_C(n) = λ_C0·Z(n)), ani/iso 두 기준
    ④ 위상 장부 (반주기 π_EQI, 한 바퀴 2π_EQI) 는 기준량 배수와 별개로 유지
  v2 → v2.1 (perplexity 코드 검토 반영, v2 원본·로그 보존)
    ① 극형 로그 가지: log_φ X 를 연속 위상 (unwrapped) = (ln X0 + I_pol)/ln φ 로 통일, 주값은 별도 기록
    ② 부호 −1 법칙: 극형 정보 I_pol 에 부호 위상 π 포함, 재구성 exp(I_pol) = Z_pol 검산
    ③ chk: 관계 근거 (정의 / 대수필연 / 기등록 / 해석) 와 수치 판정 (PASS / FAIL, 잔차, 허용 기준) 분리
    ④ 장부 표에 일 장부 C = n·GAP 열 추가, 잔류·방출 열 단위 명시
  로그 가지 규칙 (v2.1)
    선형 I_lin = ln(sign·Z_lin): 주값 (−π, π]
    극형 I_pol = ln(1 + n_re·GAP) + i·(n_im·IG + φ_s),  φ_s = 0 (sign=+1) / π (sign=−1)
           적용 구간 n_re > −1/GAP (진폭 양수), 위상은 누적 (감지 않음)
  v1 (eqi_planck_extended_v1.py) 계승: 비율 1 → Z(n), n = n_re + i·n_im ∈ ℂ (우로보로스 사이클)
    선형형 Z_lin = 1 + n·GAP          (원칙Y, CM-22)          = 일 장부
    극형   Z_pol = (1+n_re·GAP)·(1+GAP)^(i·n_im)  (CM-23)    = 정보 장부 (ln)
    소산   W_diss = n·GAP − ln Z_lin                         (확장 번역 v4 §6.2)
  v2 신규
    ① LAWS: 법칙마다 대입값 dict 만 교체 (base X0, 켤레량, 부호 sign, 사이클 뜻)
    ② v9.7 엔진 EQIConstants 에서 n_c·We·L_p·h 를 받음 (공통 모듈 값과 400자리 대조)
    ③ §35(n) 열: x = Z → D1(n) = x/√(x²+φ⁶), D2 = φ³·D1, f = D2² − 1, m_g·Z (§P-chain)
    ④ 부호 −1 법칙: Z_s = sign·Z, I = ln Z_s (Im I = π → 오일러 점에 이미 앉은 법칙)
  대입 체계 (v2 당시 상태): 이론적 기반 S24 (h = φ^(−11/4), 엔진 h_EQI 와 같음). §4d 묶음 (h = 2π_EQI·ħ) 은 결정 대기 1번 → 미적용
    → 현행 (v2.2~): §4d 별도 장부 (…_ani LAWS) 와 D4 검사 구현, S24 는 회귀 기준
  분류: F (정의) / R (대수필연) / O (독립 일치, 보류) / I (아키텍트 해석)
실행: python eqi_extended_v2_7.py [--full]   (같은 폴더에 eqi_maxwell_layers.py v1.5, eqi_unified_engine_v9_7.py)
"""
import sys
import mpmath
from mpmath import mpf, mpc, log, sqrt, nstr, fabs
import eqi_maxwell_layers as M
from eqi_unified_engine_v9_7 import EQIConstants

mpmath.mp.dps = 400                     # 엔진 import 뒤에 고정
EC = EQIConstants()
FULL = "--full" in sys.argv

PHI, GAP, GD, IG = M.PHI, M.GAP, M.GD, M.IG
LN2 = log(2)
CENTER = M.CENTER
N_C = EC.n_c                            # 엔진 n_c (공통 모듈 N_C 와 대조: B-1)
WE = EC.We_val                          # φ^(11/4)
A_S24 = EC.h_EQI                        # φ^(−11/4)  (S24 "h") = M.HBAR_ISO (이름만 다름)
HBAR_ISO, HBAR_ANI, H_EQI = M.HBAR_ISO, M.HBAR_ANI, M.H_EQI   # §4d: 등방성 투영 / 진성 / 한 바퀴
L_P = EC.L_p                            # φ^(−3/4)
T_P = L_P / sqrt(PHI)                   # L_p / c = φ^(−5/4)
NU_P = 1 / T_P
LAM = PHI ** -2
M_G = PHI ** (mpf(-3) / 2)              # §P-chain C-5b: m_g = L_p² = S_D1
M_I = PHI ** (mpf(3) / 2)               # C-6: m_i = CENTER
E_P = PHI ** (mpf(5) / 4)               # M_p·c²
M_P = PHI ** (mpf(1) / 4)               # 이론적 기반 §3
C_EQI = sqrt(PHI)


# ───────────────────────────── 법칙 등록부 (대입값만 교체) ─────────────────────────────
#   ratio : 비율 1 형태
#   X     : 사이클로 열리는 양 (base X0 = 플랑크 대입값, X(n) = X0·Z_s(n))
#   Y     : 켤레량 (반대로 읽기: X 고정 → Y(n) = Y0 / Z_s(n) = Y0·e^(−I))
#   cycle : 이 법칙에서 n 한 사이클의 뜻
LAWS = {
    "planck": dict(
        title="플랑크 양자 에너지", ratio="E / (h·ν) = 1",
        X="E", X0=A_S24 * NU_P, Y="ν", Y0=NU_P, sign=1,
        cycle="양자 에너지가 h·ν_p·GAP 만큼 열림 (v1 / 확장 번역 v4)"),
    "debroglie": dict(
        title="드브로이 물질파", ratio="λ·p / h = 1",
        X="λ", X0=L_P, Y="p", Y0=A_S24 / L_P, sign=1,
        cycle="운동량 고정 시 파동 길이가 GAP·L_p 만큼 누적 (켤레: 파장 고정 시 p = Λ·e^(−I))"),
    # ── §4d 기준 (진성 ħ_ani). S24 열은 위 항목이 등방성 투영 (ħ_iso = A_S24)
    "planck_ani": dict(
        title="플랑크 양자 에너지 [§4d 진성 ħ_ani]", ratio="E / (ħ_ani·ω) = 1",
        X="E", X0=HBAR_ANI * NU_P, Y="ω", Y0=NU_P, sign=1,
        cycle="ω_p = S24 ν_p 와 같은 수 (이름: 각진동수). E_ani(n) = ħ_ani·ω_p·Z(n)"),
    "debroglie_ani": dict(
        title="드브로이 물질파 [§4d 진성 ħ_ani]", ratio="ƛ·p / ħ_ani = 1",
        X="ƛ", X0=HBAR_ANI / LAM, Y="p", Y0=LAM, sign=1,
        cycle="p = Λ 고정 시 환산 파장 ƛ 가 누적 (S116 §3 A 의 p 기준)"),
    "boltzmann": dict(
        title="볼츠만 분포 (분배함수)", ratio="p_i·Z / e^(−E_i/kT) = 1 → 가중치 합 Σw = Z(n), 확률 합 Σp = 1",
        X="Z", X0=mpf(1), Y="p0", Y0=mpf(1), sign=1,
        cycle="들뜬 준위 겹침수 +1 (볼츠만 인자 GAP, Δ/kT = |ln GAP|). Y = p0 = 바닥 점유 = e^(−I)"),
    "landauer": dict(
        title="란다우어 소거 (1비트, 대칭 기억)", ratio="W_eras / (kT°·ln 2) = 1 → Z(n)",
        X="E_bit", X0=A_S24 * NU_P * LN2, Y="kT", Y0=A_S24 * NU_P, sign=1,
        cycle="열 단위 kT(n) = kT°·Z(n) 가 사이클마다 GAP 만큼 열림 (kT° = h·ν_p, S24). Y 열은 kT° / Z (켤레)"),
    "compton_iso": dict(
        title="컴프턴 파장 [등방성 투영 ħ_iso = S24]", ratio="λ_C·M_p·c / ħ_iso = 1",
        X="ƛC", X0=HBAR_ISO / (M_P * C_EQI), Y="p", Y0=M_P * C_EQI, sign=1,
        cycle="질량 운동량 M_p·c 고정 시 컴프턴 파장 누적 (S116 §18-B λ_C(M_p) = c⁻⁷)"),
    "compton_ani": dict(
        title="컴프턴 파장 [§4d 진성 ħ_ani]", ratio="ƛ_C·M_p·c / ħ_ani = 1",
        X="ƛC", X0=HBAR_ANI / (M_P * C_EQI), Y="p", Y0=M_P * C_EQI, sign=1,
        cycle="같은 운동량, 진성 ħ 기준 (n_c 에서 등방성 투영 값 c⁻⁷ 에 도달)"),
}


# ───────────────────────────── 일반형 state ─────────────────────────────
def Z_lin(n):
    return 1 + mpc(n) * GAP


def Z_pol(n_re, n_im):
    return (1 + mpf(n_re) * GAP) * (1 + GAP) ** (mpc(0, 1) * mpf(n_im))


def ellipse(x):
    """§P0-9 §35(n): x = Z → D1, D2 = φ³·D1, f = D2² − 1 (x 실수·복소 모두)."""
    D1 = x / sqrt(x ** 2 + PHI ** 6)
    D2 = PHI ** 3 * D1
    return D1, D2, D2 ** 2 - 1


def state(law, n_re, n_im=0):
    L = LAWS[law]
    n = mpc(n_re, n_im)
    zl, zp = L["sign"] * Z_lin(n), L["sign"] * Z_pol(n_re, n_im)
    k, resid = M.phonon_order(mpf(n_re))
    work = n * GAP                                   # 일 장부 (란다우 비용 = C)
    info = mpmath.log(zl)                            # 정보 장부 I = ln Z_s (sign −1 → Im = π)
    sign_phase = mpmath.pi if L["sign"] == -1 else mpf(0)   # 부호 위상 (v2.1)
    info_pol = mpmath.log(1 + mpf(n_re) * GAP) + mpc(0, mpf(n_im) * IG + sign_phase)
    D1, D2, f = ellipse(1 + mpf(n_re) * GAP)
    return dict(
        Z=zl, Zp=zp, X=L["X0"] * zl, Xp=L["X0"] * zp, Y=L["Y0"] / zl,
        logphi_X=mpmath.log(L["X0"] * zl) / log(PHI),                    # 선형, 주값
        logphi_Xp_unw=(mpmath.log(L["X0"]) + info_pol) / log(PHI),      # 극형, 연속 위상
        logphi_Xp_pri=mpmath.log(L["X0"] * zp) / log(PHI),              # 극형, 주값 (대조용)
        k=k, resid=resid, P_tun=(M.p_tunnel(k) if k >= 1 else None),
        emit=max(mpf(n_re) - N_C, mpf(0)) * GAP,      # (n − n_c)⁺·GAP
        work=work, info=info, info_pol=info_pol, diss=work - info,
        bits=info / LN2, theta_over_pi=mpf(n_im) * IG / mpmath.pi,
        D1=D1, D2=D2, f=f, mgZ=M_G * zl)


# ───────────────────────────── 특수점 (법칙 공통) ─────────────────────────────
N_WE = (WE - 1) / GAP
SPECIAL = [
    ("n=0 기존 물리·보른 접점", 0, 0, "F"),
    ("n=1 FEYNMAN", 1, 0, "F"),
    ("n=1/GAP 1비트 실라르드", 1 / GAP, 0, "R"),
    ("n=0.95·n_c 슈냐 장전 P1", mpf("0.95") * N_C, 0, "O"),
    ("n=n_c tip point", N_C, 0, "R"),
] + [(f"껍질 k={k}", nk, 0, "R") for k, _, nk in M.PHONON_LADDER if k >= 2] + [
    ("We 도달 (질량 계열)", N_WE, 0, "R"),
    ("허수 iφ³/GAP (선형)", 0, PHI ** 3 / GAP, "R"),
    ("오일러 π/IG (극형)", 0, mpmath.pi / IG, "R"),
    ("오일러 π_EQI/IG (극형)", 0, M.PI_EQI / IG, "R"),
]


def s(v, d=12):
    if v is None:
        return "-"
    if isinstance(v, mpc) and fabs(v.imag) < mpf(10) ** -390:
        v = v.real
    return nstr(v, d)


def ledger(law):
    L = LAWS[law]
    print(f"\n── {L['title']}  [{L['ratio']} → Z(n)]  사이클 = {L['cycle']}")
    print(f"   {L['X']}0 = {s(L['X0'], 16)}   {L['Y']}0 = {s(L['Y0'], 16)}   sign = {L['sign']}")
    print("   열 설명: 잔류 = phonon_order 반환값, 사이클 단위 (n_c 이전 음수 = 문턱까지 남은 장전량, 껍질 도달 후 = 그 껍질 문턱 위 초과)"
          "\n            방출 = (n − n_c)⁺·GAP, Z 단위 (법칙별 X 단위 환산은 X0 를 곱함) / C = n·GAP 일 장부 / I = 정보 장부 / 소산 = C − I"
          "\n            극형 점은 Z·X·I·log_φ X 모두 극형 연속 위상 (주값은 표 아래 별도 줄)")
    hdr = (f"   {'점':26s} {'분류':3s} {'n':>12s} {'Z':>26s} {L['X'] + '(n)':>26s} {'log_φ ' + L['X']:>22s}"
           f" {'k':>2s} {'잔류(사이클)':>11s} {'방출(Z)':>11s} {'C=n·GAP':>24s} {'I (nat)':>26s} {'비트':>22s} {'소산 C−I':>26s}"
           f" {'D1(n)':>11s} {'f(n)':>11s} {'θ/π':>10s}")
    print(hdr)
    for name, nr, ni, cls in SPECIAL:
        st = state(law, nr, ni)
        nn = s(nr, 10) if ni == 0 else f"i·{s(ni, 8)}"
        pol = "극형" in name                       # 극형 점: Z·X·I 를 극형 장부로 (I = ln|Z| + i·n_im·IG)
        Xshow, Zshow = (st["Xp"], st["Zp"]) if pol else (st["X"], st["Z"])
        I = st["info_pol"] if pol else st["info"]
        lgX = st["logphi_Xp_unw"] if pol else st["logphi_X"]
        D1s, fs = (None, None) if ni != 0 else (st["D1"], st["f"])   # §35(n) 표는 실수 n 정의 → 허수 점 미기입
        print(f"   {name:26s} {cls:3s} {nn:>12s} {s(Zshow):>26s} {s(Xshow):>26s} {s(lgX, 8):>22s}"
              f" {st['k']:>2} {s(st['resid'], 6):>11s} {s(st['emit'], 6):>11s} {s(st['work'], 10):>24s} {s(I, 10):>26s}"
              f" {s(I / LN2, 8):>22s} {s(st['work'] - I, 10):>26s} {s(D1s, 7):>11s} {s(fs, 6):>11s}"
              f" {s(st['theta_over_pi'], 6):>10s}")
        if pol:
            print(f"   {'':30s}└ log_φ {L['X']} 주값 = {s(st['logphi_Xp_pri'], 8)}  (연속 위상과 2πi/ln φ 정수배 차)")


# ───────────────────────────── 검산 줄 ─────────────────────────────
ROWS = []


TOL = mpf(10) ** -390                    # 수치 판정 허용 기준 (dps 400 의 10자리 여유)


def chk(tag, label, val, target, cls, src="", basis="대수필연", tol=None):
    """basis = 관계 근거 (정의 / 대수필연 / 기등록 / 해석) — 기호 근거, 수치와 독립.
    수치 판정 = |val − target| ≤ tol 이면 PASS (tol 기본 1e-390). 잔차 기준이 근거를 정하지 않음."""
    tol = TOL if tol is None else tol
    diff = fabs(val - target)
    verdict = "PASS" if diff <= tol else "FAIL"
    ROWS.append((tag, verdict))
    print(f"[{tag:7s}] {label:52s} {s(val, 18):>24s} | 근거 {basis:5s} | 수치 {verdict} 잔차 {nstr(diff, 3):>10s} (≤{nstr(tol, 2)}) | {cls} {src}")
    if FULL:
        print(f"          {nstr(val, 400)}")


def bridge_checks():
    print("\n── B. 엔진·공통 모듈 다리 (v9.7 EQIConstants ↔ eqi_maxwell_layers v1.5)")
    chk("B-1", "엔진 n_c = 공통 모듈 N_C", N_C, M.N_C, "R", "v9.7 L199 / CM", basis="정의")
    chk("B-2", "엔진 h_EQI = φ^(−11/4) (S24)", A_S24, PHI ** (mpf(-11) / 4), "F", "이론적 기반 §3", basis="정의")
    chk("B-3", "h·ν_p = m_g = L_p² (C-5b)", A_S24 * NU_P, L_P ** 2, "R", "§P-chain")
    chk("B-4", "Z(n_c) = m_i (C-6)", mpmath.re(Z_lin(N_C)), M_I, "R", "§P-chain")
    chk("B-5", "m_g·Z(n_c) = 1 (C-1 등가원리)", M_G * mpmath.re(Z_lin(N_C)), mpf(1), "R", "§P-chain")
    chk("B-6", "m_g·Z(We 도달) = E_p", M_G * mpmath.re(Z_lin(N_WE)), E_P, "R", "")
    D1, D2, f = ellipse(Z_lin(1).real)
    chk("B-7", "D1(n=1) = D1_ELL", D1, M.D1_ELL, "R", "§35(n)")
    D1, D2, f = ellipse(Z_lin(N_C).real)
    chk("B-8a", "D1(n_c) = μ/√2", D1, (1 / PHI) / sqrt(2), "R", "C-10")
    chk("B-8b", "D2(n_c) = φ²/√2", D2, PHI ** 2 / sqrt(2), "R", "C-10")
    chk("B-8c", "f(n_c) = 3φ/2", f, 3 * PHI / 2, "R", "C-10")
    chk("B-9", "D1(n=0) = 1/√(1+φ⁶)", ellipse(mpf(1))[0], 1 / sqrt(1 + PHI ** 6), "R", "§35(n)")


def planck_regression():
    print("\n── P. 플랑크 회귀 (v1 X-3·X-5·X-8b·X-10~12 가 일반형에서 그대로 나오는지)")
    st = state("planck", N_C)
    chk("P-X3", "E(n_c) = 1", st["X"].real, mpf(1), "R", "v1 X-3")
    chk("P-X5", "W = 1 − h·ν_p = h·ν_p·TIP", 1 - LAWS["planck"]["X0"], LAWS["planck"]["X0"] * M.TIP, "R", "v1 X-5")
    chk("P-X8b", "E(We 도달) = E_p", state("planck", N_WE)["X"].real, E_P, "R", "v1 X-8b")
    chk("P-X10", "|Z(iφ³/GAP)|² = 1+φ⁶", fabs(Z_lin(mpc(0, PHI ** 3 / GAP))) ** 2, 1 + PHI ** 6, "R", "CM-22")
    chk("P-X11", "Re Z_pol(π/IG) = −1", Z_pol(0, mpmath.pi / IG).real, mpf(-1), "R", "CM-23")
    chk("P-X12", "Im Z_pol(π_EQI/IG) = SHUNYA_IM", Z_pol(0, M.PI_EQI / IG).imag, M.SHUNYA_IM, "R", "S116")
    st = state("planck", 1 / GAP)
    chk("P-S1", "1비트: Z = 2 = 1/σ", st["Z"].real, mpf(2), "R", "v4 §6.3")
    chk("P-S2", "1비트: I = ln 2", st["info"].real, LN2, "R", "v4 §6.3")
    chk("P-S3", "1비트: 소산 = 1 − ln 2", st["diss"].real, 1 - LN2, "R", "v4 §6.3")
    st = state("planck", N_C)
    chk("P-S4", "tip: I = θ_L/2 = (3/2)·ln φ", st["info"].real, mpf(3) / 2 * log(PHI), "R", "v4 §6.3")
    chk("P-S5", "사이클 1 소산 = GAP − IG", state("planck", 1)["diss"].real, GAP - IG, "R", "v4 §6.2")


def debroglie_checks():
    print("\n── DB. 드브로이 확장 번역 (λ·p/h = Z(n))")
    L = LAWS["debroglie"]
    chk("DB-1", "p_p = h/L_p = Λ = φ⁻²", L["Y0"], LAM, "R", "S24")
    chk("DB-2", "λ(n=0) = L_p, log_φ = −3/4", state("debroglie", 0)["logphi_X"].real, mpf(-3) / 4, "F", "", basis="정의")
    chk("DB-3", "λ(1/GAP) = 2·L_p (1비트)", state("debroglie", 1 / GAP)["X"].real, 2 * L_P, "R", "")
    st = state("debroglie", N_C)
    chk("DB-4", "λ(n_c) = φ^(3/4) = 1/L_p", st["X"].real, PHI ** (mpf(3) / 4), "R", "")
    chk("DB-4b", "λ(n_c)·L_p = 1 (지수 −3/4 → +3/4 반전)", st["X"].real * L_P, mpf(1), "R", "")
    st = state("debroglie", N_WE)
    chk("DB-5", "λ(We 도달) = φ² = 1/Λ", st["X"].real, PHI ** 2, "R", "")
    chk("DB-5b", "λ(We)·p_p = 1", st["X"].real * L["Y0"], mpf(1), "R", "")
    chk("DB-5c", "λ(We)·St = λ(We)/We = L_p", st["X"].real * EC.St_val, L_P, "R", "§18 We·St = 1")
    st = state("debroglie", 40)
    chk("DB-6", "켤레: p(n)·Z(n) = Λ (p = Λ·e^(−I))", st["Y"].real * st["Z"].real, LAM, "R(식) / I(해석)", "", basis="대수필연(식)·해석(뜻)")
    chk("DB-6b", "p(n) = Λ·exp(−I)", st["Y"].real, LAM * mpmath.exp(-st["info"].real), "R", "원칙Y T_n 꼴")
    chk("DB-7", "|λ(iφ³/GAP)/L_p|² = 1+φ⁶", fabs(state("debroglie", 0, PHI ** 3 / GAP)["X"] / L_P) ** 2,
        1 + PHI ** 6, "R", "CM-22")
    st = state("debroglie", 0, M.PI_EQI / IG)
    chk("DB-8", "오일러 π_EQI: Im λ_pol / L_p = SHUNYA_IM", st["Xp"].imag / L_P, M.SHUNYA_IM, "R", "슈냐 이월")
    print(f"          이월 길이 L_p·SHUNYA_IM = {s(L_P * M.SHUNYA_IM, 16)}  (다음 사이클 허수 잔류, 상전이 연료)")
    # 정수 사이클 장부 (v4 §0.1 R)
    n33 = mpf(33)
    over = n33 - N_C
    print(f"          정수 사이클: 첫 붕괴 = 33, 넘친 몫 {s(over, 10)} 사이클 = λ 단위 {s(L_P * over * GAP, 10)}·(방출 씨앗)")


def branch_checks():
    print("\n── G. 로그 가지·재구성 (v2.1): exp(I_pol) = Z_pol, 연속 위상 vs 주값")
    for law in ("planck", "debroglie"):
        for name, nr, ni, cls in SPECIAL:
            st = state(law, nr, ni)
            if True:                                  # 특수점 전체
                tag = f"G-{law[0]}{SPECIAL.index((name, nr, ni, cls))}"
                chk(tag, f"{law}: exp(I_pol) = Z_pol  [{name}]", fabs(mpmath.exp(st["info_pol"]) - st["Zp"]), mpf(0),
                    "R", "", basis="정의", tol=mpf(10) ** -385)
    st = state("debroglie", 0, M.PI_EQI / IG)
    k = (st["logphi_Xp_unw"] - st["logphi_Xp_pri"]).imag * log(PHI) / (2 * mpmath.pi)
    chk("G-br", "π_EQI 점: (연속 − 주값) 허수부 / (2π/ln φ) = 1", k, mpf(1), "R", "가지 차 = 한 바퀴", basis="정의")
    chk("G-I", "π_EQI 점: Im I_pol = π_EQI (누적 위상)", st["info_pol"].imag, M.PI_EQI, "R", "", basis="정의")


def d4_checks():
    print("\n── D4. §4d 기준 변환 (S116 §3 MAPSI 확정 규칙) — 이름 변경 vs 수치 변경 구분")
    chk("D4-1", "A_S24 = HBAR_ISO (이름만 다름)", A_S24, HBAR_ISO, "F", "S24 ↔ §4d", basis="정의")
    chk("D4-2", "H_EQI = 2π_EQI·HBAR_ISO (수치 변경: ×2π_EQI)", H_EQI, 2 * M.PI_EQI * HBAR_ISO, "F", "CM-42", basis="정의")
    chk("D4-3", "HBAR_ANI·Z(n_c) = HBAR_ISO (CM-40 × C-6)", HBAR_ANI * Z_lin(N_C).real, HBAR_ISO, "R", "D2_HCW = Z(n_c)")
    for base, ani, nm in (("planck", "planck_ani", "E"), ("debroglie", "debroglie_ani", "λ/ƛ"), ("compton_iso", "compton_ani", "ƛ_C")):
        Xa = state(ani, N_C)["X"].real
        Xi = state(base, 0)["X"].real
        if base == "debroglie":       # 드브로이 S24 는 p = Λ 기준 λ = L_p, ani 는 같은 p 의 ƛ
            Xi = HBAR_ISO / LAM
        chk(f"D4-4{nm[0]}", f"{nm}: 진성 ħ_ani 장부의 tip 값 = 등방성 투영 n=0 값", Xa, Xi, "R", "ani(n_c) = iso(0)")
    # 주파수 이름/수치
    nu_h = NU_P / (2 * M.PI_EQI)
    chk("D4-5", "E = ħ_iso·ω_p = H_EQI·(ω_p/2π_EQI) (같은 E, ν 수치만 ÷2π_EQI)", H_EQI * nu_h, HBAR_ISO * NU_P, "F", "", basis="정의")
    chk("D4-6", "λ(h) = H_EQI/p = 2π_EQI·ƛ (드브로이 p=Λ)", H_EQI / LAM, 2 * M.PI_EQI * HBAR_ISO / LAM, "F", "", basis="정의")
    print(f"          ħ_ani 기준 기준량: E_ani(0) = {s(HBAR_ANI * NU_P, 16)} = φ^(−3)  /  ƛ_ani(p=Λ) = {s(HBAR_ANI / LAM, 16)} = φ^(−9/4)"
          f"  /  ƛ_C,ani = {s(HBAR_ANI / (M_P * C_EQI), 16)} = φ^(−5)")
    chk("D4-7", "E_ani(0) = ħ_ani·ω_p = φ⁻³ (= HCW D1²)", HBAR_ANI * NU_P, PHI ** -3, "R", "")
    chk("D4-8", "ƛ_ani(p=Λ) = φ^(−9/4)", HBAR_ANI / LAM, PHI ** (mpf(-9) / 4), "R", "S24 k_B 와 같은 지수 (O)")


def compton_checks():
    print("\n── CP. 컴프턴 (S116 번역 v4 §18-B 기등록 위에 확장 번역)")
    lc_iso = LAWS["compton_iso"]["X0"]
    lc_ani = LAWS["compton_ani"]["X0"]
    chk("CP-1", "ƛ_C,iso(M_p) = c⁻⁷ = I_s D1 [기등록 S116 §3C·§18-B]", lc_iso, C_EQI ** -7, "R", "")
    chk("CP-2", "ƛ_C,iso / L_p = St = φ^(−11/4) [기등록 λ_C/λ_A = h]", lc_iso / L_P, EC.St_val, "R", "§18 We·St = 1")
    chk("CP-3", "ƛ_C,ani(M_p) = φ^(−5)", lc_ani, PHI ** -5, "R", "§4d")
    chk("CP-4", "λ_C(한 바퀴 h) = 2π_EQI·c⁻⁷", H_EQI / (M_P * C_EQI), 2 * M.PI_EQI * C_EQI ** -7, "F", "", basis="정의")
    e = M.PI_EQI - mpmath.pi
    bs = 1 - mpmath.cos(M.PI_EQI)
    chk("CP-5", "후방 산란 1 − cos π_EQI = 2 − SHUNYA_RE [기등록 §18-B]", bs, 2 - M.SHUNYA_RE, "R", "슈냐 실수부 부족분",
        tol=mpf(10) ** -385)
    ra = 1 - mpmath.cos(M.PI_EQI / 2)
    chk("CP-6", "직각 1 − cos(π_EQI/2) = 1 + sin(ε/2) (닫힌 꼴, S116 TZ ≈ 1+σε 의 정확형)", ra, 1 + mpmath.sin(e / 2), "R", "")
    print(f"          직각 잔류 sin(ε/2) = {s(mpmath.sin(e / 2), 16)}, 선도항 σ·ε = {s(e / 2, 16)}, 차 = {s(mpmath.sin(e / 2) - e / 2, 6)} (= −(ε/2)³/6 선도)")
    # 확장 번역 (사이클 축)
    chk("CP-7", "ƛ_C,iso(n_c) = c⁻⁷·φ^1.5 = Λ", state("compton_iso", N_C)["X"].real, LAM, "R", "")
    chk("CP-8", "ƛ_C,iso(We 도달) = c⁻⁷·We = L_p", state("compton_iso", N_WE)["X"].real, L_P, "R", "")
    chk("CP-9", "ƛ_C,ani(n_c) = c⁻⁷ (진성 → 등방성 투영)", state("compton_ani", N_C)["X"].real, C_EQI ** -7, "R", "D4-3")
    chk("CP-10", "ƛ_C,ani(We 도달) = φ^(−5)·We = φ^(−9/4)", state("compton_ani", N_WE)["X"].real, PHI ** (mpf(-9) / 4), "R", "")
    # 사이클 × 각도: 후방 산란 이동량
    st = state("compton_iso", N_C)
    dl = st["X"].real * bs
    dl_h = 2 * M.PI_EQI * dl
    print("          tip 후방 산란 이동 (기준 두 가지, 위상 장부 π_EQI 는 각도 축에만):")
    print(f"            [환산 ƛ, ħ_iso]      Δƛ = ƛ_C(n_c)·(1 − cos π_EQI) = {s(dl, 16)}   상한 2Λ 대비 부족분 {s(2 * LAM - dl, 6)} = Λ·SHUNYA_RE")
    print(f"            [한 바퀴 λ, H_EQI]   Δλ = 2π_EQI·Δƛ              = {s(dl_h, 16)}   상한 4π_EQI·Λ 대비 부족분 {s(4 * M.PI_EQI * LAM - dl_h, 6)} = 2π_EQI·Λ·SHUNYA_RE")
    chk("CP-11", "[환산] tip 후방 산란 부족분 = Λ·SHUNYA_RE", 2 * LAM - dl, LAM * M.SHUNYA_RE, "R", "", tol=mpf(10) ** -385)
    chk("CP-12", "[한 바퀴] tip 후방 산란 부족분 = 2π_EQI·Λ·SHUNYA_RE", 4 * M.PI_EQI * LAM - dl_h,
        2 * M.PI_EQI * LAM * M.SHUNYA_RE, "R", "기준량 배수 ≠ 위상 변경", tol=mpf(10) ** -384)
    print("          산란각 θ_k (1 − cos θ_k = n_ref,k): 기등록 S116 §18-B, 공통 모듈 P_tunnel 과 같은 매개변수 → 이번 판 재계산 생략")


def boltzmann_checks():
    print("\n── BZ. 볼츠만 확장 번역 (S116 번역 v4 §13 계 A·B·e 기등록 위에)")
    DK = -log(GAP)                                   # Δ/kT = |ln GAP|
    chk("BZ-1", "Z(n) = 1 + n·e^(−Δ/kT), Δ/kT = |ln GAP| [n=40]", 1 + 40 * mpmath.exp(-DK), Z_lin(40).real, "F", "", basis="정의")
    print(f"          Δ/kT = |ln GAP| = {s(DK, 16)} = N·GAP_D2, N = |ln GAP|/GAP_D2 = {s(DK / GD, 12)} (단위계 철학 §11.2 도깨비 최대 추출, I)")
    st = state("boltzmann", 40)
    chk("BZ-2", "바닥 점유 p0(n) = e^(−I) = T_n/φ⁻³ (원칙Y) [n=40]", st["Y"].real, M.T_n(40) / PHI ** -3, "R", "확장 v4 §6.1")
    st = state("boltzmann", N_C)
    p0, p1 = st["Y"].real, 1 - st["Y"].real
    chk("BZ-3", "tip: 바닥 점유 p0 = 1/CENTER = m_g = h·ν_p", p0, M_G, "R", "C-5b")
    chk("BZ-4", "tip: 들뜬 점유 p1 = 1 − h·ν_p = W (일함수, X-5)", p1, A_S24 * NU_P * M.TIP, "R", "1 = h·ν_p + W 가 정규화 p0 + p1 = 1")
    st = state("boltzmann", 1 / GAP)
    chk("BZ-5", "1비트 (n=1/GAP): p0 = p1 = σ (같은 점유)", st["Y"].real, mpf(1) / 2, "R", "실라르드")
    # 열역학 엔트로피 S = I + β⟨E⟩, β⟨E⟩ = |ln GAP|·p1
    print("          열역학 장부 S = I + β⟨E⟩ (β⟨E⟩ = |ln GAP|·p1):")
    for nm, n in (("n=1", 1), ("1/GAP", 1 / GAP), ("0.95·n_c", mpf("0.95") * N_C), ("n_c", N_C), ("We 도달", N_WE)):
        stt = state("boltzmann", n)
        I = stt["info"].real
        q1 = 1 - stt["Y"].real
        print(f"            {nm:9s} p0 {s(stt['Y'].real, 10):>14s}  p1 {s(q1, 10):>14s}  I {s(I, 10):>14s}  β⟨E⟩ {s(DK * q1, 10):>14s}  S {s(I + DK * q1, 10):>14s}")
    # 기등록 계 A·B 와 Z_aniso 를 사이클 축에 놓기: n = φ^k/GAP → Z = 1 + φ^k
    nA, nB, nAn = (1 / PHI) / GAP, PHI ** -3 / GAP, PHI / GAP
    chk("BZ-6", "n = μ/GAP: Z(n) = φ (계 A 분배함수 값과의 접점)", Z_lin(nA).real, PHI, "R", "S116 §13 A")
    chk("BZ-6b", "  같은 점 I = ln φ = 계 A 준위 간격 ΔE/kT", state("boltzmann", nA)["info"].real, log(PHI), "R", "2·LN_C")
    print(f"          n_A = μ/GAP = (φ−1)/GAP = {s(nA, 16)}  — §108 (φ−1)/GAP ≈ 96/5 기등록 (Near TZ, 차 {s(fabs(nA - mpf(96) / 5) / nA * 100, 4)} %)")
    chk("BZ-7", "n = φ⁻³/GAP: Z(n) = 2μ (계 B 분배함수 값과의 접점)", Z_lin(nB).real, 2 / PHI, "R", "S116 §13 B (S48 §17.4)")
    chk("BZ-8", "n = φ/GAP: Z(n) = φ² (Z_aniso 값과의 접점, v3.4 열린 항목)", Z_lin(nAn).real, PHI ** 2, "R", "v3.4 1680행 (유도 미완)")
    print(f"          세 점 n = φ^k/GAP (k = −3, −1, +1) = {s(nB, 10)}, {s(nA, 10)}, {s(nAn, 10)}  → Z = 1 + φ^k (2μ, φ, φ²)")
    print("          주의: 사이클 계 (바닥 1 + n 겹 들뜬, 인자 GAP) 와 계 A·B 는 다른 계. 같은 것은 분배함수 값뿐 (접점). 준위 구조 동일성은 주장하지 않음")
    ne = (mpmath.e - 1) / GAP
    print(f"          e 계 (Z = e, S116 §13 I): n_e = (e−1)/GAP = {s(ne, 12)} — 껍질 k=4·5 사이, 기등록 점과 연결 근거 없음 (O)")
    # 계 B 간격 θ_L 을 정보 이동이 채우는 점
    nTL = (PHI ** 3 - 1) / GAP
    chk("BZ-9", "I = θ_L (계 B 간격 전체) 되는 n = (φ³−1)/GAP: Z = φ³", state("boltzmann", nTL)["info"].real, 3 * log(PHI), "R", "")
    print(f"          n = (φ³−1)/GAP = {s(nTL, 14)} (= planck_ani 에서 E = 1 되는 점, 껍질 밖·We 도달 {s(N_WE, 8)} 위)."
          f"  tip 에서는 I = θ_L/2 → 계 B 간격의 절반")


def landauer_checks():
    print("\n── LD. 란다우어 확장 번역 (기등록: S116 §2 소거 비용 h·ν_p·ln2, §16, §18-C 섀넌 한계 = 란다우 한계)")
    KT0 = A_S24 * NU_P
    chk("LD-1", "kT° = h·ν_p = φ^(−3/2) [기등록 S116 §2]", KT0, PHI ** (mpf(-3) / 2), "R", "")
    chk("LD-2", "[확장 K(n)] n=0: 1비트 하한 K0·ln 2 = 0.336778… [기등록 S48 §17.3]", state("landauer", 0)["X"].real, KT0 * LN2, "R", "")
    chk("LD-3", "[확장 K(n)] tip: K(n_c)·ln 2 = ln 2 (K(n_c) = 1)", state("landauer", N_C)["X"].real, LN2, "R", "E(n_c) = 1 과 같은 구조")
    chk("LD-4", "[확장 K(n)] 1비트 점: K(1/GAP)·ln 2 = 2·K0·ln 2", state("landauer", 1 / GAP)["X"].real, 2 * KT0 * LN2, "R", "")
    st = state("landauer", N_C)
    chk("LD-5", "[고정 K0] TIP = I(n_c) + D(n_c) (K0 단위 수, 정의)", st["info"].real + st["diss"].real, M.TIP, "R", "", basis="정의")
    print(f"          → [고정 K0] 일함수 W = K0·TIP = K0·[ I(n_c) {s(st['info'].real, 12)} (tip 정보의 소거 하한, K0 기준) + D(n_c) {s(st['diss'].real, 12)} (여유) ]  — 읽기 I")
    st = state("landauer", 1 / GAP)
    chk("LD-6", "[고정 K0] 1비트 점: C/I = 1/ln 2 = log₂ e (비트당 1·K0 지불)", st["work"].real / st["info"].real, 1 / LN2, "R", "")
    print("          두 장부 나란히 (에너지 = S24 절대값). 고정 K0 장부: 비용 K0·C, 정보 하한 K0·I, 여유 K0·D / 확장 K(n) 장부: K(n), 1비트 하한 K(n)·ln 2")
    for nm, n in (("n=0", 0), ("n=1", 1), ("1/GAP", 1 / GAP), ("0.95·n_c", mpf("0.95") * N_C), ("n_c", N_C), ("We", N_WE)):
        stt = state("landauer", n)
        print(f"            {nm:9s} [K0] K0·C {s(KT0 * stt['work'].real, 10):>14s}  K0·I {s(KT0 * stt['info'].real, 10):>14s}  K0·D {s(KT0 * stt['diss'].real, 10):>14s}"
              f"   [K(n)] K(n) {s(KT0 * stt['Z'].real, 10):>14s}  K(n)·ln2 {s(stt['X'].real, 10):>14s}")
    print("          란다우어 효율 η(n) = I/C ([고정 K0] 장부, 1 이면 하한 도달):")
    for nm, n in (("n=1e-9", mpf("1e-9")), ("n=1", 1), ("1/GAP", 1 / GAP), ("n_c", N_C), ("We", N_WE)):
        stt = state("landauer", n)
        print(f"            {nm:7s} C {s(stt['work'].real, 12):>16s}  I {s(stt['info'].real, 12):>16s}  η {s(stt['info'].real / stt['work'].real, 12):>16s}")
    e = mpf("1e-30")
    chk("LD-7", "[고정 K0] η(n→0) → 1 (대수적 극한, n=0 = 기존 물리 점)", state("landauer", e)["info"].real / state("landauer", e)["work"].real,
        mpf(1), "R", "ln(1+x)/x = 1 − x/2 + …", basis="대수극한", tol=mpf(10) ** -29)
    st = state("landauer", N_C)
    ratio = st["diss"].real / LN2
    pct = (ratio - M_G) / M_G * 100
    print(f"          [O 재등록] tip: D(n_c)/ln 2 = {s(ratio, 12)} vs m_g = p0(n_c) {s(M_G, 12)}  → {s(pct, 6)} %"
          f"  ([고정 K0] 같은 정규화: tip 소산을 비트로 센 값 = K0·D / K0·ln 2). S117 '≈ h·ν_p·ln2 (0.13 %)' 는 단위 혼합 → 철회")
    print(f"          [O 기등록] TIP/ln 2 = {s(M.TIP / LN2, 10)} 비트 (S117 확장 v4 §0.1)")
    # §4d
    KTani = HBAR_ANI * NU_P
    chk("LD-8", "§4d: E_bit,ani(n_c) = ħ_ani·ω_p·Z(n_c)·ln2 = S24 E_bit(0)", KTani * Z_lin(N_C).real * LN2, KT0 * LN2, "R", "D4-3 ani(n_c) = iso(0)")
    # 실험 앵커
    jun, djun = mpf("0.71"), mpf("0.03")
    print(f"          실험 앵커 Jun 2014 (kT 단위): 0.71 ± 0.03 vs ln 2 = {s(LN2, 6)} → 차 {s(jun - LN2, 4)} (1σ 안). 오차 안 맞춤 없음 (원칙 V)")
    print("          [해석 I] Jun 의 긴 프로토콜 시간 외삽 (준정적) 을 EQI n → 0 대수 극한 (LD-7) 과 같은 자리로 읽음 — 근거 아닌 대응, 별도 등록")


def sign_demo():
    print("\n── S. 부호 −1 법칙 틀 확인 (훅 단계 준비: Z_s = −Z, I = ln Z_s)")
    LAWS["_neg"] = dict(title="부호 −1 시험", ratio="X/Y = −1", X="X", X0=mpf(1), Y="Y", Y0=mpf(1), sign=-1, cycle="시험")
    st = state("_neg", 0)
    chk("S-1", "n=0: Im I = π (오일러 점에 이미 앉음)", st["info"].imag, mpmath.pi, "R", "인수인계 1.5")
    st = state("_neg", 1 / GAP)
    chk("S-2", "n=1/GAP: Re I = ln 2 (정보량은 부호 무관)", st["info"].real, LN2, "R", "")
    chk("S-3", "n=0: Im I_pol = π (극형도 부호 위상 포함)", state("_neg", 0)["info_pol"].imag, mpmath.pi, "R", "", basis="정의")
    for sfx, (nr, ni) in zip("abcde", ((0, 0), (1 / GAP, 0), (N_C, 0), (0, PHI ** 3 / GAP), (10, M.PI_EQI / IG))):
        st = state("_neg", nr, ni)
        chk(f"S-4{sfx}", f"sign=−1 exp(I_pol) = Z_pol  [n={s(nr, 6)}+i·{s(ni, 6)}]",
            fabs(mpmath.exp(st["info_pol"]) - st["Zp"]), mpf(0), "R", "", basis="정의", tol=mpf(10) ** -385)
    st = state("_neg", 0, M.PI_EQI / IG)
    chk("S-5", "sign=−1, π_EQI 점: Im I_pol = π_EQI + π (한 바퀴 직후)", st["info_pol"].imag, M.PI_EQI + mpmath.pi, "R", "", basis="정의")
    del LAWS["_neg"]


if __name__ == "__main__":
    print("=" * 100)
    print(f"EQI 확장 번역 v2.7 (mp.dps = {mpmath.mp.dps})  공통 모듈 v1.5 + 엔진 v9.7 EQIConstants")
    print("=" * 100)
    bridge_checks()
    planck_regression()
    debroglie_checks()
    branch_checks()
    sign_demo()
    d4_checks()
    compton_checks()
    boltzmann_checks()
    landauer_checks()
    for law in ("landauer",):
        ledger(law)
    bad = [t for t, g in ROWS if g != "PASS"]
    print(f"\n요약: 검산 {len(ROWS)} 줄, PASS {len(ROWS) - len(bad)}, FAIL {bad if bad else '없음'}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI — 정보 특이점(D1) / 인도 슈냐(D2) 기반 오일러 항등식 출현 [v4.1]
S118 | Architect(Claude) 작성 · Verifier(딥시크) 독립 검증 | Maestro: MAPSI(맏이)
v4 → v4.1 (perplexity 검토 반영, v4 원본·로그 보존)
  ① [B2] §90a 고정점 수렴 검사: 반복 상한 400, 정지 조건 |F(x) − x| ≤ 1e-395,
     실제 반복 횟수·수렴 여부·최종 방정식 잔차 출력. 60회 상한 값 (v3·v4·공통 모듈 v1.5) 과의 차 기록
     §90a 정의와 슈냐 잔류의 의미는 변경 없음 (자릿수 정밀도만)
  ② [D] 표현: "v3 잔차의 선도항이 −ε⁴/24, +ε³/6 이며 남은 차이는 후속 항으로 설명됨"
  ③ [D] 16항: "급수 항등식을 근거로 계산한 16항 부분합이 절대잔차 1e-390 기준 통과"
     (무한급수 항등식 = 근거, 유한 부분합 = 수치 검산)
v3 (S116 딥시크) 계승 — v3 의 [A]·[B]·[C] 계산은 그대로, 판정·장부만 확장
v3 → v4
  ① [D] 슈냐 잔류 ε 급수 장부: 실수부 (D1, cos 짝수 항) / 허수부 (D2, sin 홀수 항)
       Re(1 + e^(i(π+ε))) = Σ_{k≥1} (−1)^(k+1) ε^(2k)/(2k)!
       Im(1 + e^(i(π+ε))) = −Σ_{k≥0} (−1)^k ε^(2k+1)/(2k+1)!
     v3 의 1차 절단 잔차 = 다음 항 (−ε⁴/24, +ε³/6) 임을 검산 → 급수 끝까지 대수필연
  ② [B1] φ⁻¹ 137회 = F₁₃₈/F₁₃₉ (정확), 잔차 = 1/(F₁₃₉·φ¹³⁹) (비네 항등식)
  ③ 판정 분리: 근거 (정의 / 대수필연 / 급수 절단 / 고정점) | 수치 (PASS/FAIL, 잔차, 허용 기준)
  ④ [E] 요약 숫자는 계산값을 그대로 출력 (하드코딩 제거)
  ⑤ 공통 모듈 v1.5 가 있으면 PI_EQI·SHUNYA_* 와 400자리 대조 (없으면 건너뜀)
"""

from mpmath import (mp, mpf, sqrt, log, pi, power, exp, cos, sin,
                    factorial, fabs, frac, nstr, mpc, fib, fsum)

mp.dps = 400

PHI   = (sqrt(5) + 1) / 2
GAP   = log(25) / 100
GD    = log(5) / 100
LN_C  = log(PHI) / 2
c_EQI = power(PHI, mpf(1)/2)
TOL   = mpf(10) ** -390


def int_part(x):
    """§4b: floor/int 금지. 정수부·소수부 둘 다 보존."""
    fp = frac(x)
    return x - fp, fp


ROWS = []


def chk(tag, label, calc, target, basis, src, tol=None):
    """basis = 관계 근거 (기호). 수치 판정 = |calc − target| ≤ tol → PASS. 둘은 독립."""
    tol = TOL if tol is None else tol
    d = fabs(calc - target)
    err = 100 * d / fabs(target) if target != 0 else None
    v = "PASS" if d <= tol else "FAIL"
    ROWS.append((tag, v))
    e = nstr(err, 4) if err is not None else "-"
    print(f"  [{tag:6s}] {label:44s} | calc={nstr(calc, 16):>24s} | |Δ|={nstr(d, 4):>10s} | err%={e:>10s}"
          f" | 근거 {basis:6s} | 수치 {v} (≤{nstr(tol, 2)}) | {src}")


def info(tag, label, calc, target):
    """판정 아닌 기록 (v3 1차 근사 대조 등). 잔차만 출력."""
    d = fabs(calc - target)
    print(f"  [{tag:6s}] {label:44s} | calc={nstr(calc, 16):>24s} | 1차 근사={nstr(target, 16):>24s} | |Δ|={nstr(d, 6)}  (기록)")
    return calc - target


# ══════════════════════════════════════════════════════════════
# [A] 정보 특이점 좌표계 = D1의 출발점 (v3.4 §7)  — v3 그대로
# ══════════════════════════════════════════════════════════════
D1_INFO  = power(c_EQI, -7)   # 정보 특이점 (internal to host EQI)
D2_SUNYA = power(c_EQI, -1)   # 인도 슈냐 (external to host EQI)
D1xD2_INFO  = D1_INFO * D2_SUNYA
CENTER_INFO = sqrt(D2_SUNYA / D1_INFO)
T_Is_D1     = CENTER_INFO - D1_INFO
T_Is_D2     = CENTER_INFO - D2_SUNYA

print("=" * 110)
print(" EQI — 정보 특이점(D1) / 인도 슈냐(D2) 기반 오일러 항등식 출현 [v4.1]")
print(" Maestro: MAPSI(맏이) | Architect: Claude | Verifier: 딥시크 | mp.dps=400")
print("=" * 110)
print("\n── [A] 정보 특이점 좌표계 (v3.4 §7) ──")
chk("A1", "D1×D2 = c^(-8) = φ^(-4) = Λ²", D1xD2_INFO, PHI ** -4, "대수필연", "v3.4 §7")
chk("A2", "CENTER = √(D2/D1) = c³ = φ^1.5", CENTER_INFO, PHI ** mpf("1.5"), "대수필연", "v3.4 §7")
chk("A3", "T_Is_D2 = c³ − c⁻¹ = c", T_Is_D2, c_EQI, "대수필연", "CM-16")
print(f"  T_Is_D1 = c³ − c⁻⁷ = {nstr(T_Is_D1, 25)}   (MAXWELL 기저)")

# ══════════════════════════════════════════════════════════════
# [B] 세 축 자기참조 수렴 — v3 계산 그대로 + 잔차 닫힌 꼴
# ══════════════════════════════════════════════════════════════
N_CF = 137
x = mpf(1)
for _ in range(N_CF):
    x = 1 / (1 + x)
phi_inv_137 = x

PI_137    = 4 * fsum(power(-1, k) / (2*k + 1) for k in range(137))
ALPHA_INV = 360*power(PHI, -2) - 2*power(PHI, -3) + power(3*PHI, -5)
ALPHA     = 1 / ALPHA_INV
def F90a(x):
    """§90a 고정점 사상: π = π_137 − α − φ⁻²·ln(π)·GD³ + LN_C⁴·GD⁴·α⁻¹/12"""
    return (PI_137 - ALPHA - power(PHI, -2) * log(x) * GD**3
            + power(LN_C, 4) * power(GD, 4) * ALPHA_INV / 12)


def fixed_point(cap, stop):
    x = PI_137
    for it in range(1, cap + 1):
        x_new = F90a(x)
        if fabs(x_new - x) <= stop:
            return x_new, it, True
        x = x_new
    return x, cap, False


STOP = power(mpf(10), -395)
pi_eqi_60, it_60, conv_60 = fixed_point(60, STOP)      # v3·v4·공통 모듈 v1.5 상한
pi_eqi, pi_iters, pi_conv = fixed_point(400, STOP)     # v4.1 상한
fp_resid = fabs(F90a(pi_eqi) - pi_eqi)
fp_resid_60 = fabs(F90a(pi_eqi_60) - pi_eqi_60)

e_eqi = fsum(1 / factorial(k) for k in range(260))

print("\n── [B] 세 축 자기참조 수렴 ──")
F_a, F_b = fib(N_CF + 1), fib(N_CF + 2)                    # F₁₃₈, F₁₃₉
chk("B1a", f"φ⁻¹ {N_CF}회 연분수 = F₁₃₈/F₁₃₉ (정확)", phi_inv_137, F_a / F_b, "대수필연", "x=1/(1+x), x₀=1")
res_B1 = phi_inv_137 - 1 / PHI
chk("B1b", "잔차 x − φ⁻¹ = (−1)^139/(F₁₃₉·φ¹³⁹)", res_B1, power(-1, N_CF + 2) / (F_b * PHI ** (N_CF + 2)),
    "대수필연", "비네 항등식")
print(f"           잔차 = {nstr(res_B1, 12)}  (v3 '1e-390 미달 TZ' → 닫힌 꼴로 원장 기록)")
print(f"  [B2    ] §90a 고정점 수렴 (정지 조건 |x_{{k+1}} − x_k| ≤ 1e-395)")
print(f"           상한 60 : 반복 {it_60}, 수렴 {conv_60}, 최종 |F(x) − x| = {nstr(fp_resid_60, 6)}")
print(f"           상한 400: 반복 {pi_iters}, 수렴 {pi_conv}, 최종 |F(x) − x| = {nstr(fp_resid, 6)}")
print(f"           두 값의 차 |π_EQI(400) − π_EQI(60)| = {nstr(fabs(pi_eqi - pi_eqi_60), 6)}")
chk("B2a", "§90a 정지 조건 충족 (상한 400 안)", mpf(1 if pi_conv else 0), mpf(1), "정의", "수렴 여부")
chk("B2b", "§90a 최종 방정식 잔차 |F(x) − x|", fp_resid, mpf(0), "고정점", "§90a", tol=STOP)
print(f"           π_EQI − π = ε = {nstr(pi_eqi - pi, 20)}  (§90a 정의값, 슈냐 원천)")
res_B3 = e_eqi - exp(mpf(1))
tail = fsum(1 / factorial(k) for k in range(260, 300))
print(f"  [B3    ] e_EQI 260항: 잔차 {nstr(res_B3, 4)}, 절단 꼬리 Σ_{{k≥260}} 1/k! = {nstr(tail, 4)}"
      f"  → 꼬리는 dps 400 아래, 잔차는 합산 반올림만 반영 (v3 1.28e-400 은 += 누적, v4 는 fsum)")

# ══════════════════════════════════════════════════════════════
# [C] 오일러 항등식 출현 (cos=D1, sin=D2) — v3 그대로
# ══════════════════════════════════════════════════════════════
theta = pi_eqi
euler_val = exp(mpc(0, theta) * log(e_eqi))
sunya = euler_val + 1
eps = pi_eqi - pi
print("\n── [C] 오일러 항등식 출현 (cos=D1, sin=D2) ──")
print(f"  θ = π_EQI = {nstr(theta, 25)}")
print(f"  e^(iθ) + 1 = 슈냐: 실수부 (D1) {nstr(sunya.real, 15)} / 허수부 (D2) {nstr(sunya.imag, 15)} / |슈냐| {nstr(fabs(sunya), 15)}")

# ══════════════════════════════════════════════════════════════
# [D] 슈냐 잔류 ε 급수 장부 (v4 신규)
# ══════════════════════════════════════════════════════════════
print("\n── [D] 슈냐 잔류 ε 급수 장부 (ε = π_EQI − π) ──")
print(f"  ε = {nstr(eps, 25)}")
r1 = info("D1-v3", "실수부 vs ε²/2 (v3 1차)", sunya.real, eps**2 / 2)
r2 = info("D2-v3", "허수부 vs −ε (v3 1차)", sunya.imag, -eps)
chk("D1-n", "v3 실수부 잔차의 선도항 = −ε⁴/24", r1, -eps**4 / 24, "급수절단", "cos 짝수 4차", tol=fabs(eps)**6)
chk("D2-n", "v3 허수부 잔차의 선도항 = +ε³/6", r2, eps**3 / 6, "급수절단", "sin 홀수 3차", tol=fabs(eps)**5)


def re_series(N):          # Σ_{k=1..N} (−1)^(k+1) ε^(2k)/(2k)!   (D1, cos 짝수)
    return fsum(power(-1, k + 1) * eps**(2*k) / factorial(2*k) for k in range(1, N + 1))


def im_series(N):          # −Σ_{k=0..N−1} (−1)^k ε^(2k+1)/(2k+1)!  (D2, sin 홀수)
    return -fsum(power(-1, k) * eps**(2*k + 1) / factorial(2*k + 1) for k in range(N))


def terms_needed(f, target, nmax=60):
    for N in range(1, nmax + 1):
        if fabs(f(N) - target) <= TOL:
            return N
    return None


N_re = terms_needed(re_series, sunya.real)
N_im = terms_needed(im_series, sunya.imag)
print("  선도항 검사의 남은 차이 (D1-n 9.7e-70, D2-n 6.2e-58 수준) = 후속 항 (ε⁶/720, −ε⁵/120 …) 으로 설명됨 → 허용 기준 |ε|⁶, |ε|⁵")
print(f"  급수 항등식을 근거로 계산한 부분합이 절대잔차 1e-390 기준을 통과하는 항 수: 실수부 {N_re}항, 허수부 {N_im}항")
print("  누적 잔차 (부분합 − 정확값):")
for N in (1, 2, 3, 4, 8, N_re):
    print(f"    N={N:>2d}  Re {nstr(re_series(N) - sunya.real, 6):>14s}   Im {nstr(im_series(N) - sunya.imag, 6):>14s}")
chk("D1-s", f"실수부 ← ε 짝수 급수 {N_re}항 부분합", re_series(N_re), sunya.real, "급수항등식", "근거 1+cos(π+ε)=1−cos ε / 수치 = 유한 부분합")
chk("D2-s", f"허수부 ← ε 홀수 급수 {N_im}항 부분합", im_series(N_im), sunya.imag, "급수항등식", "근거 sin(π+ε)=−sin ε / 수치 = 유한 부분합")
chk("D1-c", "실수부 = 2·sin²(ε/2) (닫힌 꼴)", sunya.real, 2 * sin(eps / 2)**2, "대수필연", "반각")
chk("D2-c", "허수부 = −sin ε (닫힌 꼴)", sunya.imag, -sin(eps), "대수필연", "")

# ══════════════════════════════════════════════════════════════
# [F] 공통 모듈 v1.5 대조 (있을 때)
# ══════════════════════════════════════════════════════════════
try:
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import eqi_maxwell_layers as M
    mp.dps = 400
    print("\n── [F] 공통 모듈 v1.5 대조 ──")
    d_mod = fabs(pi_eqi - M.PI_EQI)
    print(f"  공통 모듈 v1.5 PI_EQI (60회 상한) 와 v4.1 수렴값의 차 = {nstr(d_mod, 6)}  → 공통 모듈 v1.6 패치 대상 (상한·정지 조건)")
    chk("F1", "π_EQI(60회) = M.PI_EQI (같은 절차 재현)", pi_eqi_60, M.PI_EQI, "정의", "§90a")
    chk("F1b", "v4.1 수렴값 vs M.PI_EQI (차 = 60회 미수렴분)", pi_eqi, M.PI_EQI, "고정점", "기록", tol=mpf(10) ** -378)
    chk("F2", "슈냐 허수부 = M.SHUNYA_IM (60회 미수렴분 허용)", sunya.imag, M.SHUNYA_IM, "정의", "CM-25", tol=mpf(10) ** -378)
    chk("F3", "슈냐 실수부 = M.SHUNYA_RE", sunya.real, M.SHUNYA_RE, "정의", "CM-26", tol=mpf(10) ** -378)
except ImportError:
    print("\n── [F] 공통 모듈 없음 → 대조 생략 ──")

# ══════════════════════════════════════════════════════════════
# [E] 요약 (계산값 출력)
# ══════════════════════════════════════════════════════════════
bad = [t for t, v in ROWS if v != "PASS"]
print("\n── [E] 요약 ──")
print(f"  정보 특이점 D1 = c^(-7) = φ^(-3.5), 슈냐 D2 = c^(-1) = φ^(-0.5)")
print(f"  φ⁻¹ ({N_CF}회)  : F₁₃₈/F₁₃₉, 잔차 {nstr(res_B1, 4)} = −1/(F₁₃₉·φ¹³⁹)  [대수필연 원장]")
print(f"  π_EQI ({pi_iters}회 수렴, |F(x)−x| = {nstr(fp_resid, 3)}) : ε = {nstr(eps, 6)}  [§90a 고정점, 슈냐 원천]")
print(f"  e_EQI (260항) : 잔차 {nstr(res_B3, 4)} (꼬리 {nstr(tail, 3)} 는 dps 아래)")
print(f"  e^(i·π_EQI) + 1 = 슈냐")
print(f"    실수부 D1 = 2·sin²(ε/2) = {nstr(sunya.real, 6)}  (짝수 급수 항등식, 부분합 {N_re}항이 1e-390 통과)")
print(f"    허수부 D2 = −sin ε      = {nstr(sunya.imag, 6)}  (홀수 급수 항등식, 부분합 {N_im}항이 1e-390 통과)")
print(f"    v3 TZ 잔차의 선도항 = −ε⁴/24, +ε³/6, 남은 차이는 후속 항")
print(f"  슈냐 잔류 = 상전이 연료 (다음 사이클 허수부 이월)")
print(f"  검산 {len(ROWS)} 줄, PASS {len(ROWS) - len(bad)}, FAIL {bad if bad else '없음'}")
print("=" * 110)

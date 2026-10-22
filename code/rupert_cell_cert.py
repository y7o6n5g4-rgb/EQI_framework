#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
회전 칸 하나의 독립 인증 — 고정 검사 방향 + 회전 볼 U/L + 균형 증명서 (구간 산술)
(원본 엔진·legacy Rupert 코드 무수정. 전체 R8 탐색 아님. EQI 상수 미사용.)

정리 (칸 단위 엄격 포함 배제):
  칸 Q 의 모든 배치 q = (R_out, R_in) 과 임의 평행이동 τ 에 대해
  안쪽 그림자 I(q) + τ ⊂ int O(q) 이면, 임의의 (단위일 필요 없는) 방향 n 에 대해
      h_I(q,n) + n·τ < h_O(q,n)          (지지함수는 양의 동차, n 이 실제 변 법선일 필요 없음)
  고정 방향 n_j, 가중치 w_j ≥ 0, s = Σw_j > 0, r = Σ w_j n_j 에 대해 가중합하면
      r·τ < Σ w_j [h_O(q,n_j) − h_I(q,n_j)] ≤ Σ w_j ḡ_j,   ḡ_j = upper_h_O_j − lower_h_I_j
  포함을 실현하는 τ 는 ‖τ‖ ≤ T_max (아래) 이므로 r·τ ≥ −T_max‖r‖ →
      Σ w ḡ + T_max‖r‖ > 0  이 포함의 필요조건.
  ∴ B = [Σ w ḡ + T_max‖r‖]/s ≤ 0 이면 칸 Q 전체에서 엄격 포함 불가 (인증).  r = 0 정확이면 T 항 소거.
  B 의 부호를 구간 산술로 확정하지 못하면 UNKNOWN.

T_max: 회전 중심 = 원점 (P 꼭짓점 좌표 기준). 투영은 직교 투영 (회전 후 (x,y)), 원점 → 평면 원점.
  R_O = max_v ‖v‖ (바깥 물체, 투영 전 3D, 회전 불변), R_I = λ·max_v ‖v‖ (안쪽).
  직교 투영은 거리를 늘리지 않으므로 그림자 꼭짓점은 반지름 R_O, R_I 원판 안 (칸 전체 공통).
  I+τ ⊂ O 이면 I 의 한 점 y (‖y‖ ≤ R_I) 에 대해 y+τ ∈ O → ‖τ‖ ≤ R_O + R_I = T_max.
  (0 ∈ I 이면 ‖τ‖ ≤ R_O 로 더 좁힐 수 있으나 여기서는 보수적으로 R_O + R_I 사용)

U/L (단위 아닌 n 포함 일반형, |n|R 로 스케일):  q = n·w0, p = ‖n × w0‖, w0 = R_0 v (칸 중심 배치)
  U = |n|R            if q ≥ |n|R cosρ   else  q cosρ + p sinρ
  L = −|n|R           if q ≤ −|n|R cosρ  else  q cosρ − p sinρ
  구간 산술에서 분기 조건이 미확정이면 U = |n|R, L = −|n|R (항상 유효한 자명 경계) 를 사용.
  ρ ≥ π 이면 자명 경계.  상대 회전각 ρ 는 Euler 인수 각 변화 합 (축 회전 곱, 양측 불변 거리 삼각부등식).
  R_out = R_out0·Δ 가 아니라 R_out = (R_out R_out0ᵀ) R_out0 → R_out v = Δ' w0, ∠Δ' = d(R_out0, R_out) ≤ ρ.

실행: python3 code/rupert_cell_cert.py
"""

import itertools
import sys
import time

import numpy as np
import mpmath
from mpmath import iv, mp, mpf
import scipy
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

iv.dps = 40
mp.dps = 40
SEP = "=" * 104


# ── 구간 기본 연산 ────────────────────────────────────────────────
def F(x, k=16):
    """구간 출력: 점 구간이면 한 값, 아니면 [a, b]"""
    if hasattr(x, "_mpi_"):
        lo, hi = mp.make_mpf(x._mpi_[0]), mp.make_mpf(x._mpi_[1])
        if lo == hi:
            return mpmath.nstr(lo, k)
        return f"[{mpmath.nstr(lo, k)}, {mpmath.nstr(hi, k)}]"
    return mpmath.nstr(x, k)


def I(x):
    """정확 십진/분수 문자열 또는 float → 점 구간 (float 은 이진값 그대로 정확)"""
    return iv.mpf(x)


def irz(a):
    c, s = iv.cos(a), iv.sin(a)
    return [[c, -s, I(0)], [s, c, I(0)], [I(0), I(0), I(1)]]


def iry(a):
    c, s = iv.cos(a), iv.sin(a)
    return [[c, I(0), s], [I(0), I(1), I(0)], [-s, I(0), c]]


def imul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), I(0)) for j in range(3)] for i in range(3)]


def imv(A, v):
    return [sum((A[i][k] * v[k] for k in range(3)), I(0)) for i in range(3)]


def inorm(v):
    return iv.sqrt(sum((x * x for x in v), I(0)))


def icross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def idot(a, b):
    return sum((x * y for x, y in zip(a, b)), I(0))


def U_iv(n, w0, rho):
    """max n·(Δw0) over ∠Δ ≤ ρ — 상계 구간 (분기 미확정 시 자명 경계)"""
    nn, R = inorm(n), inorm(w0)
    if rho.a >= mpmath.pi:
        return nn * R
    q, p = idot(n, w0), inorm(icross(n, w0))
    c, s = iv.cos(rho), iv.sin(rho)
    d = q - nn * R * c
    if d.a >= 0:
        return nn * R
    if d.b < 0:
        return q * c + p * s
    return nn * R


def L_iv(n, w0, rho):
    """min n·(Δw0) over ∠Δ ≤ ρ — 하계 구간 (분기 미확정 시 자명 경계)"""
    nn, R = inorm(n), inorm(w0)
    if rho.a >= mpmath.pi:
        return -nn * R
    q, p = idot(n, w0), inorm(icross(n, w0))
    c, s = iv.cos(rho), iv.sin(rho)
    d = q + nn * R * c
    if d.b <= 0:
        return -nn * R
    if d.a > 0:
        return q * c - p * s
    return -nn * R


# ── 칸 인증 ──────────────────────────────────────────────────────
def certify(name, V_out, V_in, R_out0, R_in0, rho_out, rho_in, dirs, weights, lam=1, note=""):
    """
    V_out, V_in : 꼭짓점 (정확 문자열/정수 성분), 회전 중심 = 원점
    R_out0, R_in0 : 칸 중심 회전 (구간 행렬), rho_* : 상대 회전각 상계 (구간)
    dirs : 고정 검사 방향 (float 2-벡터, 정확 이진값으로 사용; 단위일 필요 없음)
    weights : w_j ≥ 0 (float, 정확 이진값으로 사용)
    """
    t0 = time.time()
    lamI = I(lam)
    Vo = [[I(c) for c in v] for v in V_out]
    Vi = [[lamI * I(c) for c in v] for v in V_in]
    Wo = [imv(R_out0, v) for v in Vo]
    Wi = [imv(R_in0, v) for v in Vi]
    rows = []
    for (nx, ny), w in zip(dirs, weights):
        n3 = [I(float(nx)), I(float(ny)), I(0)]
        uo = None
        for w0 in Wo:
            u = U_iv(n3, w0, rho_out)
            uo = u.b if uo is None or u.b > uo else uo
        li = None
        for w0 in Wi:
            l = L_iv(n3, w0, rho_in)
            li = l.a if li is None or l.a > li else li
        gbar = I(uo) - I(li)                     # 상한 끝점끼리 차 → 구간 (상한 = uo − li)
        rows.append((float(nx), float(ny), float(w), uo, li, gbar))
    wI = [I(float(r[2])) for r in rows]
    s = sum(wI, I(0))
    r_vec = [sum((wI[j] * I(rows[j][k]) for j in range(len(rows))), I(0)) for k in (0, 1)]
    r_norm = iv.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2)
    R_O = max((inorm(v).b for v in Vo))
    R_I = max((inorm(v).b for v in Vi))
    T_max = I(R_O) + I(R_I)
    num = sum((wI[j] * rows[j][5] for j in range(len(rows))), I(0))
    exact_balance = (r_vec[0].a == 0 and r_vec[0].b == 0 and r_vec[1].a == 0 and r_vec[1].b == 0)
    B = (num if exact_balance else num + T_max * r_norm) / s
    status = "인증 (엄격 포함 배제)" if B.b <= 0 else ("UNKNOWN (B > 0 확정: 배제 불가)" if B.a > 0 else "UNKNOWN (부호 미확정)")
    el = time.time() - t0
    print(f"  [{name}] {note}")
    print(f"     ρ_out ≤ {F(rho_out.b, 12)},  ρ_in ≤ {F(rho_in.b, 12)},  λ = {F(lamI, 20)}")
    print(f"     T_max = R_O + R_I = {F(R_O, 15)} + {F(R_I, 15)}  (3D 반지름, 회전 중심 = 원점, 투영 전)")
    print(f"     {'방향 n_j (x, y)':<36} | {'w_j':<22} | {'upper h_O':<22} | {'lower h_I':<22} | ḡ_j 상한")
    for nx, ny, w, uo, li, gb in rows:
        print(f"     ({nx: .15f}, {ny: .15f}) | {w:<22.17g} | {F(uo, 16):<22} | {F(li, 16):<22} | "
              f"{F(gb.b, 12)}")
    print(f"     Σw = {F(s, 18)},  잔류 균형 r = ({F(r_vec[0], 6)}, {F(r_vec[1], 6)}),"
          f"  ‖r‖ ≤ {F(r_norm.b, 6)}  ({'정확 균형 → T 항 소거' if exact_balance else '근사 균형 → T_max‖r‖ 보정'})")
    print(f"     B 구간 = [{F(B.a, 22)}, {F(B.b, 22)}]  (폭 {F(B.delta, 3)})  →  {status}   ({el:.3f}s)")
    return B, status, el


# ── 중심 배치에서 방향·가중치 선택 (휴리스틱; 정당성은 위 정리가 보장) ──
def center_choice(Vo_f, Vi_f, Ro, Ri, lam=1.0):
    A = (Ro @ Vo_f.T).T[:, :2]
    Bv = lam * (Ri @ Vi_f.T).T[:, :2]
    eq = ConvexHull(A).equations
    n, b = eq[:, :2], -eq[:, 2]
    g = b - (Bv @ n.T).max(axis=0)
    Aeq = np.vstack([np.ones(len(n)), n.T])
    res = linprog(g, A_eq=Aeq, b_eq=[1, 0, 0], bounds=[(0, None)] * len(n), method="highs-ds")
    w = np.where(res.x > 1e-14, res.x, 0.0)
    keep = w > 0
    Mres = linprog([-1, 0, 0], A_ub=np.column_stack([np.ones(len(n)), n]), b_ub=g,
                   bounds=[(None, None)] * 3, method="highs")
    return n[keep], w[keep], -Mres.fun


def float_rot_R8(x):
    al, ph2, th2, th1, ph1 = x
    rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return ry(-th1) @ rz(-ph1), rz(al) @ ry(-th2) @ rz(-ph2)


def iv_rot_R8(xs):
    al, ph2, th2, th1, ph1 = [I(v) for v in xs]
    return imul(iry(-th1), irz(-ph1)), imul(imul(irz(al), iry(-th2)), irz(-ph2))


def sample_check(Vo_f, Vi_f, center, half, dirs, w, n_samp=2000, seed=1):
    """비인증 sanity: 칸 내 무작위 배치의 실제 LP 최적 M 과 Σw g(q) 최대 (B 상한 이하여야 함)"""
    rng = np.random.default_rng(seed)
    Mmax, Smax = -np.inf, -np.inf
    for _ in range(n_samp):
        x = center + (rng.uniform(-1, 1, 5) * half)
        Ro, Ri = float_rot_R8(x)
        A = (Ro @ Vo_f.T).T[:, :2]
        Bv = (Ri @ Vi_f.T).T[:, :2]
        eq = ConvexHull(A).equations
        n, b = eq[:, :2], -eq[:, 2]
        g = b - (Bv @ n.T).max(axis=0)
        r = linprog([-1, 0, 0], A_ub=np.column_stack([np.ones(len(n)), n]), b_ub=g,
                    bounds=[(None, None)] * 3, method="highs")
        Mmax = max(Mmax, -r.fun)
        hO = np.array([(A @ d).max() for d in dirs])
        hI = np.array([(Bv @ d).max() for d in dirs])
        Smax = max(Smax, float(w @ (hO - hI)) / w.sum())
    return Mmax, Smax


if __name__ == "__main__":
    T_ALL = time.time()
    print(f"python {sys.version.split()[0]} | mpmath {mpmath.__version__} (iv.dps={iv.dps}) | numpy {np.__version__} | "
          f"scipy {scipy.__version__}")
    print(SEP)
    print("0. 회귀시험 A/B/C (정육면체, 이전 검증과 같은 투영 쌍; ρ_out = 0)")
    print(SEP)
    cube = [list(v) for v in itertools.product((-1, 1), repeat=3)]
    s2, s6 = iv.sqrt(2), iv.sqrt(6)
    e1 = [I(1) / s2, -I(1) / s2, I(0)]
    e2 = [I(1) / s6, I(1) / s6, -I(2) / s6]
    e3 = icross(e1, e2)
    R_hex = [e1, e2, e3]                         # 행 = 투영 기저 → 회전 후 (x,y) = (e1·v, e2·v)
    Id = [[I(1), I(0), I(0)], [I(0), I(1), I(0)], [I(0), I(0), I(1)]]
    lam0 = 2 * s2 / (1 + iv.sqrt(3))
    d60 = [(0.5, float(mpmath.sqrt(3) / 2)), (-0.5, -float(mpmath.sqrt(3) / 2))]   # 60°, 240° (이진 근사; |n| 오차 처리됨)
    zero = I(0)
    # A: λ_safe — 배제 불가여야 함 (실제 통과 배치)
    lam_safe = (1 + lam0) / 2
    certify("A", cube, cube, R_hex, Id, zero, zero, d60, [0.5, 0.5], lam=lam_safe,
            note="λ_safe = (1+λ0)/2: 엄격 포함 존재 → 배제 불가 (UNKNOWN 이 정답)")
    # B: λ = λ0 (1 + 1e-12) — 배제
    lamB = lam0 * (1 + I("1e-12"))
    certify("B", cube, cube, R_hex, Id, zero, zero, d60, [0.5, 0.5], lam=lamB,
            note="λ = λ0·(1+1e-12) > λ0, 반대 방향 60°/240° 가중치 ½")
    # C: 평면 정사각형 그림자 (z=0) 를 |θ| ≤ π/360 회전 — 이전 C 와 동일 대상
    sq = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]
    certify("C", cube, sq, R_hex, Id, zero, iv.pi / 360, d60, [0.5, 0.5], lam=I(26) / 25,
            note="λ = 26/25, 안쪽 = 평면 정사각형 그림자 (z=0), ρ_in = π/360 (이전 C 재현)")
    certify("C3D", cube, cube, R_hex, Id, zero, iv.pi / 360, d60, [0.5, 0.5], lam=I(26) / 25,
            note="참고: 안쪽을 3D 정육면체로 두고 회전 볼 (탈평면 회전 포함) ρ_in = π/360")

    print(SEP)
    print("1. Zeng 족 P_t (t = 11/20) 회전 칸 — 바깥·안쪽 모두 회전")
    print("   인코딩 (R8 과 동일): R_out = Ry(−θ₁)Rz(−φ₁),  R_in = Rz(α)Ry(−θ₂)Rz(−φ₂),  x = (α, φ₂, θ₂, θ₁, φ₁)")
    print("   상대 회전각 상계: ρ_out = h_θ₁ + h_φ₁,  ρ_in = h_α + h_θ₂ + h_φ₂ (반폭 합; 축 회전 곱의 삼각부등식)")
    print("   ※ 반폭은 단위시험 입력값이며 EQI 격자 세분 법칙이 아님")
    print(SEP)
    P0 = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    t_iv = I(11) / I(20)                         # t = 11/20 (정확 유리수의 구간 포함)
    Pt_iv = [[I(c) for c in p] for p in P0] + [[-t_iv * I(c) for c in p] for p in P0]
    Pt_f = np.array([[float(c) for c in p] for p in P0] + [[-0.55 * c for c in p] for p in P0])

    def certify_Pt(name, center_str, half_str, note):
        center = np.array([float(c) for c in center_str])
        half = np.array([float(h) for h in half_str])
        Ro_f, Ri_f = float_rot_R8(center)
        dirs, w, Mc = center_choice(Pt_f, Pt_f, Ro_f, Ri_f)
        Ro_iv, Ri_iv = iv_rot_R8(center_str)
        h = [I(x) for x in half_str]
        rho_out = h[3] + h[4]
        rho_in = h[0] + h[1] + h[2]
        print(f"  칸 중심 x = ({', '.join(center_str)}),  반폭 = ({', '.join(half_str)})")
        print(f"  중심 배치 LP 최적 여유 M_center = {Mc:.6e} [float]  → 중심 쌍대 해에서 방향 {len(dirs)}개·가중치 선택")
        B, st, el = certify(name, Pt_iv, Pt_iv, Ro_iv, Ri_iv, rho_out, rho_in, dirs, w, lam=1, note=note)
        Mmax, Smax = sample_check(Pt_f, Pt_f, center, half, dirs, w)
        print(f"     [비인증 sanity, 무작위 2000 배치] 실제 M 최대 = {Mmax:.6e},  Σw(h_O−h_I)/Σw 최대 = {Smax:.6e}"
              f"  (≤ B 상한 {F(B.b, 10)} : {Smax <= float(B.b) + 1e-12})")
        return st

    st1 = certify_Pt("P1", ["0.7", "2.1", "1.3", "0.4", "0.9"], ["0.001"] * 5,
                     "비대칭 중심 배치, 바깥·안쪽 회전 모두 변화")
    print()
    st2 = certify_Pt("P2", ["0", "0.9", "0.4", "0.4", "0.9"], ["0.001"] * 5,
                     "동일 방향 배치 포함 칸 (중심: R_in0 = R_out0, α=0, θ₂=θ₁, φ₂=φ₁) — UNKNOWN 정상")
    print(SEP)
    print(f"총 소요 {time.time() - T_ALL:.2f}s")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
루퍼트 경계 증명서 — Perplexity 제안식 독립 검증 (기존 엔진·역사 코드·TASKS 무수정)

검증 대상 (제안식, 확정 성과로 취급하지 않음):
  [1] 평행이동 LP  M = max_{τ,m} m  s.t.  n_j·τ + m ≤ g_j,  g_j = b_j − h_I(n_j)
      증명서 w_j ≥ 0, Σw_j = 1, Σw_j n_j = 0  ⇒  M ≤ Σ w_j g_j
  [2] 회전 볼 상·하계  L ≤ n·(Rv) ≤ U  (상대 회전각 ≤ ρ)
  [3] 정육면체 기준시험 A/B/C

출력 구분:  [기호] = sympy 정확 계산 / [수치] = mpmath 50자리 (인증 아님) /
           [구간] = mpmath.iv 구간 산술 (반올림 포함 보증) / [float] = 부동소수점 (인증 아님)
EQI 상수 (GAP, μ, Λ 등) 는 사용하지 않음. 다각형 포함 판정만 다룸.
실행: python3 code/rupert_cert_verify.py
"""

import itertools
import sys

import numpy as np
import mpmath
from mpmath import mp, mpf, iv
import scipy
from scipy.optimize import linprog
import sympy as sp

mp.dps = 50
iv.dps = 50
SEP = "=" * 100


def hdr(s):
    print(SEP)
    print(s)
    print(SEP)


print(f"python {sys.version.split()[0]} | mpmath {mpmath.__version__} | sympy {sp.__version__} | "
      f"numpy {np.__version__} | scipy {scipy.__version__}")

# ════════════════════════════════════════════════════════════════════════════
# 1. 평행이동 LP 와 쌍대 증명서
# ════════════════════════════════════════════════════════════════════════════
hdr("1. 평행이동 LP · 쌍대 증명서")
print("""  [기호 증명] (τ, m) 이 실현가능하면 모든 j 에 대해 m ≤ g_j − n_j·τ.
    w_j ≥ 0 를 곱해 더하면  m·Σw ≤ Σw g − τ·(Σ w n).  Σw = 1, Σwn = 0 (정확) 이면  m ≤ Σ w g.
    실현가능한 모든 (τ, m) 에 성립 → M ≤ Σ w g.  (O 의 유계성·볼록성 불필요, w 조건만 필요)
  [강쌍대] 쌍대 LP: min Σw g  s.t.  w ≥ 0, Σw = 1, Σw n = 0.
    쌍대 실현가능 ⇔ 0 ∈ conv{n_j} ⇔ O 유계 (법선이 양의 일차결합으로 평면을 덮음).
    원 LP 는 항상 실현가능 (m → −∞) → 쌍대 실현가능하면 M = min Σ w g (강쌍대, 최적값 도달).
  [≤3 양의 가중치] 쌍대 실현가능 영역은 등식 3개 (Σw=1, Σw n 의 2성분) 를 가진 유계 다면체 →
    최적값은 꼭짓점(기저해)에서 도달, 기저해의 양의 성분 ≤ 3 (카라테오도리). 조건: 0 ∈ conv{n_j}.
    서로 반대인 법선 쌍이 최적이면 양의 가중치 2개 (예: 기준시험 B).
  [의미] M > 0: 어떤 τ 에서 I+τ ⊂ int O (모든 여유 > 0, 엄격 포함).
         M = 0: 닫힌 포함은 가능하나 엄격 포함 불가 (경계 접촉).
         M < 0: 어떤 평행이동으로도 I+τ ⊄ O.
    ※ 고정 투영 쌍 하나의 판정. 루퍼트 통과 여부는 모든 투영 쌍에 대한 sup 이 필요.
  [제약 집합] 불통과 증명 (M 상계): O ⊆ {x : n_j·x ≤ b_j} 이면 충분 (b_j ≥ h_O(n_j), 변 일부만 써도 됨 —
    완화 문제의 최적값 ≥ 진짜 M 이므로 Σw g 는 여전히 진짜 M 의 상계).
    통과 증명 (M > 0): 반대 방향 필요 — O 의 모든 변을 쓰고 b_j ≤ h_O(n_j) (보수적 내측) 이어야
    {n_j·x ≤ b_j} ⊆ O 가 보장됨. 안쪽 지지값 h_I 는 불통과 증명에 하계, 통과 증명에 상계를 써야 함.""")


def lp_primal(n, g):
    """max m s.t. m + n_j·τ ≤ g_j  (변수 m, τx, τy 모두 자유)"""
    A = np.column_stack([np.ones(len(n)), n])
    r = linprog([-1, 0, 0], A_ub=A, b_ub=g, bounds=[(None, None)] * 3, method="highs")
    return -r.fun, r.x


def lp_dual(n, g):
    """min Σ w g s.t. w ≥ 0, Σw = 1, Σ w n = 0"""
    A_eq = np.vstack([np.ones(len(n)), n.T])
    r = linprog(g, A_eq=A_eq, b_eq=[1, 0, 0], bounds=[(0, None)] * len(n), method="highs-ds")
    return r.fun, r.x


rng = np.random.default_rng(20261022)
worst_gap, max_pos = 0.0, 0
for trial in range(200):
    k = rng.integers(3, 12)
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    n = np.c_[np.cos(ang), np.sin(ang)]
    g = rng.normal(size=k)
    try:
        Mp, _ = lp_primal(n, g)
        Md, w = lp_dual(n, g)
    except Exception:
        continue
    if not np.isfinite(Mp):
        continue
    worst_gap = max(worst_gap, abs(Mp - Md))
    max_pos = max(max_pos, int(np.sum(w > 1e-12)))
print(f"  [float] 무작위 다각형 200개: |M_primal − M_dual| 최대 = {worst_gap:.2e},"
      f" 쌍대 최적해(단체법 꼭짓점) 양의 가중치 최대 개수 = {max_pos}")

# 반례: 거의 균형인 가중치를 정확 증명서처럼 쓰면 틀린 상계가 나옴
print("  [반례] 균형 오차가 있는 가중치를 정확 증명서로 쓰는 경우")
n = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
g = np.array([0.0, 1.0, 1.0, 1.0])            # O = [−1,0]×[−1,1], I = {원점}
M_true, tau = lp_primal(n, g)
eps = 1e-6
w_bad = np.array([0.5 + eps, 0.5 - eps, 0.0, 0.0])     # Σw = 1, Σw n = (2ε, 0) ≠ 0
naive = float(w_bad @ g)
r_res = w_bad @ n
print(f"      O = [−1,0]×[−1,1], I = {{0}}, g = (0,1,1,1): 진짜 M = {M_true:.9f} (τ* = ({tau[1]:.3f}, {tau[2]:.3f}))")
print(f"      w = (½+ε, ½−ε, 0, 0), ε = {eps:g}:  Σw g = {naive:.9f},  Σw n = ({r_res[0]:.1e}, {r_res[1]:.1e})")
print(f"      → 'M ≤ Σw g' 주장 거짓: {naive < M_true - 1e-12}  (Σwn ≠ 0 이면 m ≤ Σwg − τ·Σwn 이라 τ 항이 남음)")
print("  [수정식] w ≥ 0, s = Σw > 0, r = Σ w n 일 때,  ‖τ‖ ≤ T 인 모든 τ 에 대해  m ≤ (Σ w g + T·‖r‖)/s.")
print("      엄격 통과 (m > 0) 는 I+τ ⊂ O 를 요구 → ‖τ‖ ≤ T := R_O + R_I (원점 기준 외접 반지름 합) 로 제한 가능.")
print("      → (Σ w g + T‖r‖)/s ≤ 0 이면 이 투영 쌍의 엄격 통과 없음 (수치 가중치로도 유효). 단 s, r 계산 자체도")
print("        구간 산술 또는 정확 산술로 해야 인증이 됨.")
T = 2**0.5 + 0.0
print(f"      반례에 적용 (T = √2 + 0): (Σwg + T‖r‖)/s = {(naive + T*np.linalg.norm(r_res))/w_bad.sum():.9f}"
      f"  ≥ M = {M_true:.9f} → 모순 없음")

# ════════════════════════════════════════════════════════════════════════════
# 2. 회전 볼 상·하계
# ════════════════════════════════════════════════════════════════════════════
hdr("2. 회전 볼 상·하계 U, L")
print("""  [기호 증명] ‖Rv‖ = R 이고, 상대 회전각 ≤ ρ 인 회전들로 v 가 갈 수 있는 점 집합은
    v 를 중심으로 한 구면 모자 {w : ‖w‖ = R, ∠(v, w) ≤ ρ} 와 정확히 같다
    (∠(v,Rv) ≤ 회전각, 그리고 모자 안 임의 w 는 v×w 축 회전으로 각 ∠(v,w) ≤ ρ 에 도달).
    β = ∠(n, v) (cosβ = q/R, R sinβ = p) 이면 n·w = R cos∠(n,w), ∠(n,w) ∈ [max(0, β−ρ), min(π, β+ρ)].
    max = R cos(max(0, β−ρ)):  β ≤ ρ ⇔ q ≥ R cosρ (ρ∈[0,π], cos 단조감소) → R,  아니면 q cosρ + p sinρ.
    min = R cos(min(π, β+ρ)):  β+ρ ≥ π ⇔ q ≤ −R cosρ → −R,  아니면 q cosρ − p sinρ.
    → 제안식 U, L 은 정확하며 모자 위에서 도달되는 최적(타이트) 경계.
  [특수 경우] R = 0: q = p = 0, U = L = 0 (분기 무관하게 일치).  ρ = 0: U = L = q.
    ρ = π: U = R, L = −R (모든 방향).  q = ±R: p = 0, 경계는 R cos(β∓ρ) 형태로 연속.
  [조건] v 는 회전 중심 기준 좌표여야 함 (평행이동 τ 와 분리). n 은 단위 벡터.
    이 경계는 '상대 회전각 ≤ ρ 인 모든 회전' (SO(3) 의 측지 볼) 에 대한 것이며,
    Euler 각도 칸의 상(像)은 이 볼의 부분집합일 뿐 같은 집합이 아니다 (보수적 상계).""")


def UL(n, v, rho, stable=True):
    n = [mpf(x) for x in n]
    v = [mpf(x) for x in v]
    R = mpmath.sqrt(sum(x * x for x in v))
    q = sum(a * b for a, b in zip(n, v))
    if stable:
        cx = [n[1] * v[2] - n[2] * v[1], n[2] * v[0] - n[0] * v[2], n[0] * v[1] - n[1] * v[0]]
        p = mpmath.sqrt(sum(x * x for x in cx))          # p = ‖n × v‖  (R²−q² 음수 반올림 회피)
    else:
        p = mpmath.sqrt(R * R - q * q)
    c, s = mpmath.cos(rho), mpmath.sin(rho)
    U = R if q >= R * c else q * c + p * s
    L = -R if q <= -R * c else q * c - p * s
    return U, L


def rot(axis, ang):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rot_mp(axis, ang):
    ax = [mpf(float(x)) for x in axis]
    nn = mpmath.sqrt(sum(x * x for x in ax))
    k = [x / nn for x in ax]
    K = mpmath.matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return mpmath.eye(3) + mpmath.sin(ang) * K + (1 - mpmath.cos(ang)) * (K * K)


viol, worst_tight = 0, mpf(0)
for trial in range(300):
    v = rng.normal(size=3) * rng.uniform(0.1, 3)
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    rho = mpf(float(rng.uniform(0, np.pi)))
    U, L = UL(n, v, rho)
    vm = mpmath.matrix([mpf(float(x)) for x in v])
    nm = [mpf(float(x)) for x in n]
    for _ in range(40):
        ax = rng.normal(size=3)
        a = rho * mpf(float(rng.uniform(0, 1)))
        w = rot_mp(ax, a) * vm
        d = sum(nm[i] * w[i] for i in range(3))
        if d > U + mpf(10)**-40 or d < L - mpf(10)**-40:
            viol += 1
    # 타이트성: v 를 n 쪽으로 (평면 n,v 내에서) 각 min(ρ, β) 회전 → U 도달
    nv = np.cross(v, n)
    if np.linalg.norm(nv) > 1e-9:
        beta = mpmath.acos(sum(nm[i] * mpf(float(v[i])) for i in range(3)) / mpmath.sqrt(sum(mpf(float(x))**2 for x in v)))
        w = rot_mp(nv, min(rho, beta)) * vm
        d = sum(nm[i] * w[i] for i in range(3))
        worst_tight = max(worst_tight, abs(d - U))
print(f"  [수치 50자리] 무작위 300 경우 × 회전 40개: L ≤ n·Rv ≤ U 위반 = {viol}건,"
      f"  U 도달 회전에서 |n·Rv − U| 최대 = {mpmath.nstr(worst_tight, 3)}")
for lab, n, v, rho in (("R=0", (1, 0, 0), (0, 0, 0), mpf(1)), ("ρ=0", (1, 0, 0), (0.3, 0.4, 0), mpf(0)),
                       ("ρ=π", (1, 0, 0), (0.3, 0.4, 0), mpf(mp.pi)), ("q=+R", (1, 0, 0), (2, 0, 0), mpf("0.3")),
                       ("q=−R", (1, 0, 0), (-2, 0, 0), mpf("0.3"))):
    U, L = UL(n, v, rho)
    print(f"  [특수] {lab:<5}: U = {mpmath.nstr(U, 12)},  L = {mpmath.nstr(L, 12)}")
v = np.array([0.1, 0.2, 0.3])
nf = v / np.linalg.norm(v)
Rf, qf = np.linalg.norm(v), nf @ v
print(f"  [float 반례] v=(0.1,0.2,0.3), n=v/‖v‖: R²−q² = {Rf * Rf - qf * qf:.3e} "
      f"→ sqrt = {np.sqrt(Rf * Rf - qf * qf) if Rf * Rf - qf * qf >= 0 else 'NaN (음수)'} (참값 0, √ε 크기 오차);"
      f"  ‖n×v‖ = {np.linalg.norm(np.cross(nf, v)):.3e}")
for vv in ([0.1, 0.2, 0.3], [0.7, 0.1, 0.2], [1/3, 1/7, 1/9], [0.3, 0.6, 0.9]):
    vv = np.array(vv)
    nn = vv / np.linalg.norm(vv)
    d = np.linalg.norm(vv)**2 - (nn @ vv)**2
    print(f"      v={vv.round(4)}: float R²−q² = {d:.3e}  ({'음수 반올림' if d < 0 else '≥ 0'})")
print("      수정: p = max(0, R²−q²)^(1/2) 또는 p = ‖n × v‖ (본 스크립트 사용).")

print("""  [Euler 각 합 = 상대 회전각 상계 조건]
    d(A,B) = ∠(AᵀB) 는 SO(3) 의 양측 불변 거리 (회전각, 0 ≤ d ≤ π).
    R(a) = R_1(a_1)···R_k(a_k), 각 R_i 가 고정 축 둘레 단위 속도 회전이면
      d(R(a), R(b)) ≤ Σ_i d(R_i(a_i), R_i(b_i)) = Σ_i min(|a_i−b_i| mod 2π, 2π − …)  (삼각부등식 + 양측 불변성).
    조건: (i) 각 인수가 '각도 = 회전각' 인 축 회전 (Rz, Ry, Rx 등, 스케일 없음),
          (ii) 축은 고정 (공간 고정 또는 연쇄된 물체 고정 모두 위 곱 형태면 성립),
          (iii) 합이 π 를 넘으면 ρ = π 로 절단 (U = R, L = −R, 정보 없음).
    방향 구면 좌표 (θ, φ) 를 쓰는 경우 Rz(φ)Ry(θ) 처럼 곱으로 쓰면 성립. φ 의 실제 변위는 sinθ 배로 더 작음 (상계는 보수적).""")
viol_e, worst_ratio = 0, 0.0
for _ in range(2000):
    a = rng.uniform(-np.pi, np.pi, 3)
    da = rng.uniform(-0.5, 0.5, 3)
    def E(x):
        return rot([0, 0, 1], x[0]) @ rot([0, 1, 0], x[1]) @ rot([0, 0, 1], x[2])
    M_ = E(a).T @ E(a + da)
    ang = np.arccos(np.clip((np.trace(M_) - 1) / 2, -1, 1))
    bound = min(np.pi, np.sum(np.abs(da)))
    if ang > bound + 1e-12:
        viol_e += 1
    worst_ratio = max(worst_ratio, ang / bound)
print(f"  [float] Rz·Ry·Rz 무작위 2000쌍: 상대 회전각 > Σ|Δ| 위반 = {viol_e}건, 최대 비 (각/Σ|Δ|) = {worst_ratio:.6f}")

# ════════════════════════════════════════════════════════════════════════════
# 3. 정육면체 기준시험
# ════════════════════════════════════════════════════════════════════════════
hdr("3. 정육면체 K = [−1,1]³ 기준시험 (이 투영 쌍 한정, 정육면체 전체 판정 아님)")
s2, s3, s6 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)
e1 = sp.Matrix([1, -1, 0]) / s2
e2 = sp.Matrix([1, 1, -2]) / s6
verts = [sp.Matrix(v) for v in itertools.product((-1, 1), repeat=3)]
sh = sorted({(sp.nsimplify(sp.simplify(e1.dot(v))), sp.nsimplify(sp.simplify(e2.dot(v)))) for v in verts},
            key=lambda p: float(sp.atan2(p[1], p[0])))
expect = {(s2, sp.sqrt(sp.Rational(2, 3))), (s2, -sp.sqrt(sp.Rational(2, 3))), (-s2, sp.sqrt(sp.Rational(2, 3))),
          (-s2, -sp.sqrt(sp.Rational(2, 3))), (0, sp.sqrt(sp.Rational(8, 3))), (0, -sp.sqrt(sp.Rational(8, 3)))}
hull_pts = [p for p in sh if p != (0, 0)]
ok_shadow = {tuple(sp.simplify(c) for c in p) for p in hull_pts} == {tuple(sp.simplify(c) for c in p) for p in expect}
print(f"  [기호] 바깥 그림자 꼭짓점 (원점 투영 2점 제외) = {hull_pts}")
print(f"         제안 좌표 (±√2, ±√(2/3)), (0, ±√(8/3)) 와 일치: {ok_shadow}")
# 정육각형 O 의 변: 법선 각 0, ±60°, ±120°, 180°, b = √2 (내접 반지름)
psi = [0, sp.pi/3, 2*sp.pi/3, sp.pi, 4*sp.pi/3, 5*sp.pi/3]
normals = [(sp.cos(a), sp.sin(a)) for a in psi]
b_sym = []
for nx, ny in normals:
    b_sym.append(sp.simplify(max((nx*p[0] + ny*p[1] for p in hull_pts), key=lambda z: float(z))))
print(f"  [기호] 바깥 변 법선 각 0,60,…,300° 의 b_j = {b_sym}  (모두 √2 = 내접 반지름: {all(sp.simplify(b - s2) == 0 for b in b_sym)})")
sq = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
lam0 = 2*s2/(1 + s3)


def g_sym(lam, theta=0):
    out = []
    for (nx, ny), b in zip(normals, b_sym):
        # λ > 0 이면 최대 꼭짓점은 λ 와 무관 → λ=1 에서 선택 후 λ 배
        h1 = max(((nx*(x*sp.cos(theta) - y*sp.sin(theta)) + ny*(x*sp.sin(theta) + y*sp.cos(theta)))
                  for x, y in sq), key=lambda z: float(z))
        h = lam*h1
        out.append(sp.simplify(b - h))
    return out


# A. 엄격 포함
lam_safe = (1 + lam0)/2
gA = g_sym(lam_safe)
minA = min(gA, key=lambda z: float(z))
print(f"  [A 기호] λ0 = 2√2/(1+√3) = {sp.N(lam0, 30)},  λ_safe = (1+λ0)/2 = {sp.N(lam_safe, 30)}")
print(f"           τ = 0 에서 g_j = {[sp.N(x, 15) for x in gA]}")
print(f"           최소 여유 = {sp.simplify(minA)} = {sp.N(minA, 30)}  > 0: {bool(sp.N(minA, 60) > 0)}  → 엄격 포함 (M ≥ 최소 여유 > 0)")
print(f"           λ_safe > 1: {bool(sp.N(lam_safe, 60) > 1)}  (이 투영 쌍에서 통과 배치 실재)")
# B. 고정 투영 쌍 최적성: 법선 60°, 240° 가중치 ½
lam = sp.symbols('lam', positive=True)
gB = g_sym(lam)
cert = sp.simplify(sp.Rational(1, 2)*gB[1] + sp.Rational(1, 2)*gB[4])
bal = sp.simplify(sp.Rational(1, 2)*sp.Matrix(normals[1]) + sp.Rational(1, 2)*sp.Matrix(normals[4]))
root = sp.solve(sp.Eq(cert, 0), lam)
print(f"  [B 기호] 법선 60°, 240° 가중치 ½: Σw n = {list(bal)} (정확히 0),  Σw g(λ) = {cert}")
print(f"           Σw g = 0 의 해 λ = {root} = λ0: {[sp.simplify(r - lam0) == 0 for r in root]}")
print(f"           λ > λ0 ⇒ Σw g < 0 ⇒ M < 0 (모든 평행이동에서 포함 불가). λ = λ0 ⇒ M ≤ 0 (엄격 포함 불가).")
lp_n = np.array([[float(sp.cos(a)), float(sp.sin(a))] for a in psi])
for lv in (float(lam0) - 1e-9, float(lam0), float(lam0) + 1e-9):
    gg = np.array([float(b_sym[j]) - max(lv*(lp_n[j] @ np.array(p)) for p in sq) for j in range(6)])
    print(f"           [float LP 대조] λ = λ0{lv - float(lam0):+.0e}: M = {lp_primal(lp_n, gg)[0]: .3e}")
# C. 1차원 회전 구간
lamC = mpf(26)/25
rhoC = iv.pi/360
print("  [C] λ = 26/25, 안쪽 그림자 평면 회전 |θ| ≤ π/360, 법선 60° 와 240° (가중치 ½)")
print("      [기호] 정사각형 회전 θ 의 n(ψ) 지지값: h = λ(|cos(ψ−θ)| + |sin(ψ−θ)|);  중심 대칭 → h(−n) = h(n)")
print("             Σw g(θ) = √2 − λ(cos(π/3−θ) + sin(π/3−θ)) = √2·(1 − λ·sin(7π/12 − θ))  (|θ| ≤ π/360 에서 두 항 양수)")
print("             sin 은 [7π/12 − π/360, 7π/12 + π/360] ⊂ (π/2, π) 에서 단조감소 → 최댓값은 θ = −π/360 (= 105.5°) 끝점")
th = mp.pi/360
end = mpmath.sqrt(2)*(1 - lamC*mpmath.sin(7*mp.pi/12 + th))
print(f"      [수치 50자리] θ = −π/360 끝점: Σw g = {mpmath.nstr(end, 25)}  (< 0: {end < 0})")
NSUB = 64
worst_hi = None
for i in range(NSUB):
    lo = -rhoC + 2*rhoC*iv.mpf(i)/NSUB
    hi = -rhoC + 2*rhoC*iv.mpf(i + 1)/NSUB
    T_ = iv.mpf([lo.a, hi.b])
    val = iv.sqrt(2) - iv.mpf(lamC)*(iv.cos(iv.pi/3 - T_) + iv.sin(iv.pi/3 - T_))
    worst_hi = val.b if worst_hi is None or val.b > worst_hi else worst_hi
print(f"      [구간 산술] θ 구간 {NSUB}등분, 각 부분구간에서 Σw g 구간값의 상한 최대 = {mpmath.nstr(worst_hi, 20)}"
      f"  (< 0: {worst_hi < 0}) → 구간 전체 포함 불가 보증")
# 회전 볼 경계 (2D, 하계 L) 로 같은 결론: h ≥ max_v L_v
Lb = []
for psi_ in (mp.pi/3, 4*mp.pi/3):
    nn = (mpmath.cos(psi_), mpmath.sin(psi_), 0)
    Lb.append(max(UL(nn, (lamC*x, lamC*y, 0), mp.pi/360)[1] for x, y in sq))   # 평면 회전 = z축 회전 (3D 경계의 특수형)
certL = mpmath.sqrt(2) - (Lb[0] + Lb[1])/2
print(f"      [회전 볼 하계] h(n) ≥ max_v L_v (ρ = π/360): Σw (b − max L) = {mpmath.nstr(certL, 25)}  (< 0: {certL < 0})")
print("      ※ C 는 '이 투영 쌍 + 평면 회전 1차원 구간' 에서 λ = 26/25 불가능성만 보증. 정육면체 전체의 비루퍼트 판정 아님")
print("        (정육면체는 루퍼트, ν = 3√2/4). λ0 는 이 투영 쌍의 값이지 정육면체 전체 최적값 아님. 각도 간격은 단위시험용.")
print(SEP)
print("검증 종료")

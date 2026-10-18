#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task R6 — Zeng 족 P_t 통과 경계 수치 탐색 (TASKS.md v9.7, 엔진 수정 없음)
근거: Zeng arXiv:2604.26531 §1.3 정의, P_(11/20) 비루퍼트(dwrensha Lean), P_(3/5) 루퍼트(Fredriksson).
  P_t = conv{p_i, −t·p_i},  p0=(1,1,1), p1=(1,−1,−1), p2=(−1,1,−1), p3=(−1,−1,1)

방법
  ν_found(t) = max λ : 회전 R_in·P_t 를 λ배 하고 평면 평행이동 s 한 투영이
               회전 R_out·P_t 투영(볼록 껍질) 안에 들어가는 최대 배율.
  고정 회전에서 λ 최대화는 3변수 LP:  λ·h_Q(n_k) + n_k·s ≤ b_k  (바깥 다각형 변 k,
  h_Q = 안쪽 투영의 지지함수) → 변 3개 조합의 꼭짓점 전수로 정확히 풂.
  바깥 최적화: 두 회전(회전 벡터 3+3) Nelder–Mead, 단계 축소 재시작, 다중 시작점(4코어).
  t 를 0.600 → 0.550 으로 내리며 직전 t 최적 배치를 시작점에 추가 (연속법).
  검증: 최적 배치를 scipy HiGHS LP 로 독립 재계산, t=1(정육면체) 에서 3√2/4 재현.
판정 없음: '통과 발견 하한'만 표기. 탐색 실패 ≠ 비루퍼트. EQI 상수 대조 없음.
수치 바닥: 같은 방향 배치는 λ=1 이 정확값이며 부동소수 잡음 ~1e-12 → ν−1 ≤ 1e-9 는 '통과 미발견'.
Date: 2026-10-18
"""

import itertools
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot

P0 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
TRIPLES = np.array(list(itertools.combinations(range(16), 3)))
FLOOR = 1e-9                                   # 통과 발견 판정 문턱 (수치 바닥 위)
N_RANDOM = 160
N_CONT = 40


def Pt(t):
    return np.vstack([P0, -t * P0])


def outer_inner(x, V):
    A = Rot.from_rotvec(x[:3]).apply(V)[:, :2]
    B = Rot.from_rotvec(x[3:6]).apply(V)[:, :2]
    eq = ConvexHull(A).equations
    return eq[:, :2], -eq[:, 2], B


def lam(x, V):
    """고정 회전에서 최대 λ (LP 꼭짓점 전수)"""
    n, b, B = outer_inner(x, V)
    h = (B @ n.T).max(axis=0)
    M = np.column_stack([h, n])
    tr = TRIPLES[(TRIPLES < len(n)).all(axis=1)]
    Ms, bs = M[tr], b[tr]
    ok = np.abs(np.linalg.det(Ms)) > 1e-14
    sol = np.linalg.solve(Ms[ok], bs[ok][..., None])[..., 0]
    feas = (sol @ M.T <= b + 1e-12).all(axis=1)
    return sol[feas, 0].max() if feas.any() else 0.0


def lam_highs(x, V):
    """독립 검증: 원래 포함 조건 (안쪽 꼭짓점 8개 × 바깥 변) 을 HiGHS LP 로"""
    n, b, B = outer_inner(x, V)
    nq = B @ n.T
    Aub = np.column_stack([nq.T.reshape(-1), np.repeat(n[:, 0], len(B)), np.repeat(n[:, 1], len(B))])
    r = linprog([-1, 0, 0], A_ub=Aub, b_ub=np.repeat(b, len(B)),
                bounds=[(0, None), (None, None), (None, None)], method="highs")
    return r.x[0]


def local_opt(x, V, rounds=4, scale=0.3):
    f = lambda y: -lam(y, V)
    for k in range(rounds):
        simplex = None if k == 0 else x + np.vstack([np.zeros(6), np.eye(6) * scale / 3**k])
        r = minimize(f, x, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-13, fatol=1e-15, initial_simplex=simplex))
        x = r.x
    return -r.fun, x


def job(args):
    t, x0 = args
    return local_opt(np.asarray(x0), Pt(t))


def scan_t(t, pool, cont=None, n_random=N_RANDOM, n_cont=N_CONT, seed=0):
    rng = np.random.default_rng(seed + int(round(t * 1e6)))
    starts = [rng.normal(size=6) * 2 for _ in range(n_random)]
    if cont is not None:
        for i in range(n_cont):
            starts.append(cont + rng.normal(size=6) * (0.002 if i % 2 else 0.02))
    res = pool.map(job, [(t, s) for s in starts])
    res.sort(key=lambda r: -r[0])
    # 상위 5개 정밀화 (단계 축소 재시작 6회, 작은 시작 심플렉스)
    pol = pool.map(job_polish, [(t, r[1]) for r in res[:5]])
    pol.sort(key=lambda r: -r[0])
    best_v, best_x = pol[0]
    n_pass = sum(1 for r in res if r[0] - 1 > FLOOR)
    return dict(t=t, v=best_v, x=best_x, v2=pol[1][0], n_pass=n_pass, n=len(res),
                highs=lam_highs(best_x, Pt(t)))


def job_polish(args):
    t, x0 = args
    return local_opt(np.asarray(x0), Pt(t), rounds=6, scale=0.02)


def fmt(r):
    nu1 = r["v"] - 1
    tag = "통과 발견" if nu1 > FLOOR else "통과 미발견"
    return (f"  t={r['t']:.6f} | ν_found−1 = {nu1: .6e} | {tag:<6} | 2위(정밀화) {r['v2'] - 1: .3e} | "
            f"HiGHS 재계산 {r['highs'] - 1: .6e} | ν−1>1e-9 시작점 {r['n_pass']}/{r['n']}")


def params(r):
    x = r["x"]
    q1, q2 = Rot.from_rotvec(x[:3]).as_quat(), Rot.from_rotvec(x[3:6]).as_quat()
    return (f"      R_out 회전벡터 = [{x[0]: .10f}, {x[1]: .10f}, {x[2]: .10f}]  쿼터니언(x,y,z,w) = "
            f"[{', '.join(f'{c: .10f}' for c in q1)}]\n"
            f"      R_in  회전벡터 = [{x[3]: .10f}, {x[4]: .10f}, {x[5]: .10f}]  쿼터니언(x,y,z,w) = "
            f"[{', '.join(f'{c: .10f}' for c in q2)}]")


if __name__ == "__main__":
    t0 = time.time()
    out = []
    p = lambda s="": (print(s, flush=True), out.append(s))
    p("=" * 110)
    p("R6. Zeng 족 P_t 통과 경계 수치 탐색 — '통과 발견 하한' (판정 없음, 탐색 실패 ≠ 비루퍼트)")
    p("=" * 110)
    p(f"  시작점: 무작위 {N_RANDOM} + 연속법 {N_CONT} (직전 t 최적 배치 섭동), 상위 5개 정밀화. 문턱 ν−1 > {FLOOR:g}")
    with Pool(4) as pool:
        # 검증 1: t=1 (정육면체) → 3√2/4
        rc = scan_t(1.0, pool, n_random=40)
        p(f"  [검증] t=1 정육면체: ν_found = {rc['v']:.13f}  (3√2/4 = {3 * 2**0.5 / 4:.13f}, 차 {abs(rc['v'] - 3 * 2**0.5 / 4):.1e})")
        # 격자 0.600 → 0.550
        grid = [round(0.600 - 0.005 * i, 3) for i in range(11)]
        res = {}
        cont = None
        for t in grid:
            r = scan_t(t, pool, cont)
            res[t] = r
            if r["v"] - 1 > FLOOR:
                cont = r["x"]
            p(fmt(r))
            p(params(r))
        p("")
        p(f"  [검증] t=0.600 Fredriksson 값(≈1+4e-6) 대비: ν_found−1 = {res[0.6]['v'] - 1:.6e}")
        # 이분 탐색: 통과 발견 최소 격자점과 그 아래 미발견 격자점 사이
        passing = [t for t in grid if res[t]["v"] - 1 > FLOOR]
        failing = [t for t in grid if res[t]["v"] - 1 <= FLOOR]
        p("")
        if passing and failing and min(passing) > max([f for f in failing if f < min(passing)], default=-1):
            hi = min(passing)
            lo = max(f for f in failing if f < hi)
            cont = res[hi]["x"]
            p(f"  이분 탐색: 구간 [{lo:.3f} 미발견, {hi:.3f} 발견]")
            for _ in range(6):
                mid = (lo + hi) / 2
                r = scan_t(mid, pool, cont, n_random=80, n_cont=80)
                p(fmt(r))
                if r["v"] - 1 > FLOOR:
                    hi, cont = mid, r["x"]
                    p(params(r))
                else:
                    lo = mid
            p("")
            p(f"  t* 수치 추정 구간 (통과 발견 하한 기준): ({lo:.6f}, {hi:.6f}]")
            p(f"    → t = {hi:.6f} 에서 통과 발견, t = {lo:.6f} 이하에서는 본 탐색으로 통과 미발견")
        elif failing == []:
            p(f"  격자 전 구간 통과 발견 — t* < {min(grid):.3f} (본 탐색 기준)")
        else:
            p(f"  통과 발견 격자점: {passing}  /  미발견: {failing} — 단조 구간 아님, 이분 탐색 생략")
        p("  ※ 판정 없음. EQI 상수 대조 없음. '미발견' 은 비루퍼트 증명이 아님.")
        p(f"  총 소요 {time.time() - t0:.0f} s")
    open("taskR6_Pt_scan_output.txt", "w").write("\n".join(out) + "\n")

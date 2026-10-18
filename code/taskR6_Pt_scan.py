#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task R6 — Zeng 족 P_t 통과 경계 수치 탐색 (TASKS.md v9.7, 엔진 수정 없음) — 연속 추적판
근거: Zeng arXiv:2604.26531 §1.3 정의, P_(11/20) 비루퍼트(dwrensha Lean), P_(3/5) 루퍼트(Fredriksson).
  P_t = conv{p_i, −t·p_i},  p0=(1,1,1), p1=(1,−1,−1), p2=(−1,1,−1), p3=(−1,−1,1)

MAPSI 지시 (재실행 방식)
  1. 연속 추적: t=0.590 최적 배치에서 시작, t 를 0.0025 씩 낮추며 직전 t 최적 배치에서 국소 최적화만.
  2. 대상 t: 0.5875, 0.585, 0.5825, 0.580, 0.5775, 0.575 (0.55~0.57 계산 안 함:
     0.55 = Lean 형식증명 비루퍼트, 0.5~0.57 = Zeng 탐색 실패 보고).
  3. t 마다 시간 상한 5분. 상한 도달 시 그 시점 최선값 보고.
  4. ν−1 < 1e-6 배치는 mpmath 50자리로 포함 여유 재검증.
  5. 통과 여유가 처음 0 이하가 되는 t 에서 멈추고 직전 t 와의 구간을 경계 추정으로 보고.

방법
  ν(t; R_out, R_in) = max λ : λ·proj(R_in P_t) + s ⊂ proj(R_out P_t)  (평면 평행이동 s 포함)
  고정 회전에서 3변수 LP  λ·h_Q(n_k) + n_k·s ≤ b_k  를 변 3개 조합 꼭짓점 전수로 정확히 풂.
  국소 최적화: Nelder–Mead 단계 축소 재시작. 4코어에서 같은 시작 배치에 서로 다른 미소 섭동
  (0, 1e-4, 1e-3, 1e-2 rad) 을 준 4개 국소 탐색을 병렬 수행, 최선값 채택 (무작위 전역 시작 없음).
  mpmath 50자리 검증: float 회전벡터를 정확한 입력으로 보고 Rodrigues 회전·투영·볼록 다각형 변·
  LP 꼭짓점을 50자리로 재계산, 안쪽 8꼭짓점의 최소 포함 여유(slack)까지 출력.
판정 없음: '통과 발견 하한'만 표기. 탐색 실패 ≠ 비루퍼트. EQI 상수 대조 없음.
Date: 2026-10-18
"""

import itertools
import time
from multiprocessing import Pool

import numpy as np
import mpmath
from mpmath import mp, mpf
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot

P0 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
TRIPLES = np.array(list(itertools.combinations(range(16), 3)))
T_LIST = [0.5875, 0.585, 0.5825, 0.580, 0.5775, 0.575]
T_START = 0.590
TIME_CAP = 300.0                               # s, t 마다
PERTURB = [0.0, 1e-4, 1e-3, 1e-2]
# 이전 실행(무작위 다중 시작, 중단됨) t=0.590 최적 배치 — 회전벡터 (R_out, R_in)
X_0590 = np.array([0.2637138622, 0.7536138881, -0.6121645081,
                   2.9930300741, -1.0086169854, -1.2069833282])


def Pt(t):
    return np.vstack([P0, -t * P0])


def lam(x, V):
    A = Rot.from_rotvec(x[:3]).apply(V)[:, :2]
    B = Rot.from_rotvec(x[3:6]).apply(V)[:, :2]
    eq = ConvexHull(A).equations
    n, b = eq[:, :2], -eq[:, 2]
    h = (B @ n.T).max(axis=0)
    M = np.column_stack([h, n])
    tr = TRIPLES[(TRIPLES < len(n)).all(axis=1)]
    Ms, bs = M[tr], b[tr]
    ok = np.abs(np.linalg.det(Ms)) > 1e-14
    sol = np.linalg.solve(Ms[ok], bs[ok][..., None])[..., 0]
    feas = (sol @ M.T <= b + 1e-12).all(axis=1)
    return sol[feas, 0].max() if feas.any() else 0.0


def local_track(args):
    """시작 배치 x0 에서 국소 최적화만 (시간 상한 내 단계 축소 재시작 반복)"""
    t, x0, eps, seed, cap = args
    V = Pt(t)
    rng = np.random.default_rng(seed)
    x = np.asarray(x0) + rng.normal(size=6) * eps
    f = lambda y: -lam(y, V)
    best_v, best_x = lam(x, V), x.copy()
    t0 = time.time()
    scale, stall, rounds = 0.02, 0, 0
    while time.time() - t0 < cap and stall < 6:
        simplex = best_x + np.vstack([np.zeros(6), np.eye(6) * scale])
        r = minimize(f, best_x, method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-14, fatol=1e-16, initial_simplex=simplex))
        rounds += 1
        if -r.fun > best_v + 1e-16:
            best_v, best_x, stall = -r.fun, r.x, 0
        else:
            stall += 1
        scale = scale / 3 if scale > 1e-7 else 0.02          # 축소 후 재확장 순환
    return best_v, best_x, rounds, time.time() - t0, time.time() - t0 >= cap


# ── mpmath 50자리 검증 ─────────────────────────────────────────────
def rodrigues_mp(rv):
    rv = [mpf(float(c)) for c in rv]
    th = mpmath.sqrt(sum(c * c for c in rv))
    if th == 0:
        return mpmath.eye(3)
    k = [c / th for c in rv]
    K = mpmath.matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return mpmath.eye(3) + mpmath.sin(th) * K + (1 - mpmath.cos(th)) * (K * K)


def verify_mp(x, t):
    mp.dps = 50
    tt = mpf(repr(float(t)))
    V = [[mpf(int(c)) for c in p] for p in P0] + [[-tt * int(c) for c in p] for p in P0]
    Ro, Ri = rodrigues_mp(x[:3]), rodrigues_mp(x[3:6])
    A = [(Ro * mpmath.matrix(v)) for v in V]
    B = [(Ri * mpmath.matrix(v)) for v in V]
    A2 = [(a[0], a[1]) for a in A]
    B2 = [(b_[0], b_[1]) for b_ in B]
    hv = ConvexHull(np.array([[float(a), float(b_)] for a, b_ in A2])).vertices   # 반시계 순서
    edges = []
    for i in range(len(hv)):
        p, q = A2[hv[i]], A2[hv[(i + 1) % len(hv)]]
        nx, ny = q[1] - p[1], -(q[0] - p[0])               # 반시계 → 바깥 법선
        nn = mpmath.sqrt(nx * nx + ny * ny)
        nx, ny = nx / nn, ny / nn
        edges.append((nx, ny, nx * p[0] + ny * p[1]))
    hq = [max(nx * qx + ny * qy for qx, qy in B2) for nx, ny, _ in edges]
    best = None
    for i, j, k in itertools.combinations(range(len(edges)), 3):
        M = mpmath.matrix([[hq[m], edges[m][0], edges[m][1]] for m in (i, j, k)])
        if abs(mpmath.det(M)) < mpf(10)**-40:
            continue
        sol = mpmath.lu_solve(M, mpmath.matrix([edges[m][2] for m in (i, j, k)]))
        lam_, sx, sy = sol[0], sol[1], sol[2]
        if all(lam_ * hq[m] + edges[m][0] * sx + edges[m][1] * sy <= edges[m][2] + mpf(10)**-45
               for m in range(len(edges))):
            if best is None or lam_ > best[0]:
                best = (lam_, sx, sy)
    lam_, sx, sy = best
    # 안쪽 8꼭짓점 직접 포함 여유 (λ 를 1+ (λ−1)/2 로 둔 엄격 포함 확인 포함)
    slack = min(e[2] - (e[0] * (lam_ * qx + sx) + e[1] * (lam_ * qy + sy)) for e in edges for qx, qy in B2)
    lam_half = 1 + (lam_ - 1) / 2
    slack_half = min(e[2] - (e[0] * (lam_half * qx + sx) + e[1] * (lam_half * qy + sy))
                     for e in edges for qx, qy in B2)
    return lam_, slack, slack_half, (sx, sy)


def params(x):
    q1, q2 = Rot.from_rotvec(x[:3]).as_quat(), Rot.from_rotvec(x[3:6]).as_quat()
    return (f"      R_out 회전벡터 = [{x[0]: .14f}, {x[1]: .14f}, {x[2]: .14f}]\n"
            f"      R_in  회전벡터 = [{x[3]: .14f}, {x[4]: .14f}, {x[5]: .14f}]\n"
            f"      R_out 쿼터니언(x,y,z,w) = [{', '.join(f'{c: .12f}' for c in q1)}]\n"
            f"      R_in  쿼터니언(x,y,z,w) = [{', '.join(f'{c: .12f}' for c in q2)}]")


def track(t, x_prev, pool):
    jobs = [(t, x_prev, eps, 100 + i, TIME_CAP) for i, eps in enumerate(PERTURB)]
    res = pool.map(local_track, jobs)
    res.sort(key=lambda r: -r[0])
    return res


if __name__ == "__main__":
    out = []
    p = lambda s="": (print(s, flush=True), out.append(s))
    T0 = time.time()
    p("=" * 110)
    p("R6. Zeng 족 P_t 통과 경계 — 연속 추적 (warm start, 국소 최적화만, t 당 5분 상한)")
    p("    '통과 발견 하한' 표기. 판정 없음, 탐색 실패 ≠ 비루퍼트, EQI 상수 대조 없음")
    p("=" * 110)
    p(f"  국소 탐색 4개 병렬: 시작 배치 미소 섭동 {PERTURB} rad. 결과 열: 최선 / 4개 값 / 라운드·소요")
    with Pool(4) as pool:
        # 시작: t=0.590 배치 재정밀화 (이전 실행 출력은 회전벡터 10자리 → 국소 재수렴)
        res = track(T_START, X_0590, pool)
        x_prev = res[0][1]
        v = res[0][0]
        p(f"  [시작] t={T_START:.4f} | ν−1 = {v - 1:.9e}  (이전 실행 기록 7.151136e-07) | "
          f"4개: {', '.join(f'{r[0] - 1:.3e}' for r in res)}")
        p(params(x_prev))
        prev_t, prev_v = T_START, v
        boundary = None
        for t in T_LIST:
            res = track(t, x_prev, pool)
            v, x, rounds, el, capped = res[0]
            p("")
            p(f"  t={t:.4f} | ν_found−1 = {v - 1: .9e} | 4개: {', '.join(f'{r[0] - 1: .3e}' for r in res)} | "
              f"라운드 {rounds}, {el:.0f}s{' (5분 상한 도달 — 그 시점 최선값)' if capped else ''}")
            p(params(x))
            pos = v - 1 > 0
            if v - 1 < 1e-6:
                lam_mp, slack, slack_half, s = verify_mp(x, t)
                pos = lam_mp - 1 > mpf(10)**-40
                p(f"      [mpmath 50자리] λ−1 = {mpmath.nstr(lam_mp - 1, 20)}")
                p(f"                      평행이동 s = ({mpmath.nstr(s[0], 20)}, {mpmath.nstr(s[1], 20)})")
                p(f"                      λ 에서 최소 포함 여유 = {mpmath.nstr(slack, 6)} (LP 꼭짓점, 0 이 정상)")
                p(f"                      λ'=1+(λ−1)/2 에서 최소 포함 여유 = {mpmath.nstr(slack_half, 20)}"
                  f"  ({'> 0: 엄격 포함 (통과 배치 확인)' if slack_half > 0 else '≤ 0: 엄격 포함 아님'})")
            if not pos:
                boundary = (t, prev_t)
                p(f"      → 통과 여유 0 이하: 여기서 정지")
                break
            x_prev, prev_t, prev_v = x, t, v
        p("")
        if boundary:
            p(f"  경계 추정 구간 (통과 발견 하한 기준): t* ∈ ({boundary[0]:.4f}, {boundary[1]:.4f}]")
            p(f"    t = {boundary[1]:.4f} 통과 발견 (ν−1 = {prev_v - 1:.6e}), "
              f"t = {boundary[0]:.4f} 연속 추적으로 통과 미발견")
        else:
            p(f"  대상 t 전부 통과 발견 — t* ≤ {T_LIST[-1]:.4f} (본 추적 기준)")
        p("  ※ 판정 없음. '미발견' 은 비루퍼트 증명이 아님 (국소 추적 한계).")
        p(f"  총 소요 {time.time() - T0:.0f} s")
    open("taskR6_Pt_scan_output.txt", "w").write("\n".join(out) + "\n")

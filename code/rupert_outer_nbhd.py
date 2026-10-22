#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
바깥 투영 방향의 작은 2차원 이웃 Q 로 모든-축 국소정리 확장 시험 (P_{11/20})
원본 엔진·legacy·보존 파일 (rupert_all_axes.py 포함) 무수정. EQI 상수 미사용. t_shape = 11/20.

설정: R_out(q) = Ry(−q1)Rz(−q2),  q0 = (2/5, 9/10),
      Q = [2/5 − 1/1024, 2/5 + 1/1024] × [9/10 − 1/1024, 9/10 + 1/1024]  (반폭 = 국소정리 시험 입력; EQI 격자 법칙 아님)
      R_in(q, θ, u) = R_out(q)·exp(θ[u]×)   (공통 회전 구조; 바깥·안쪽 독립 볼 처리 안 함)
공통항 기호 소거: 변 j 의 접촉 꼭짓점 v_j ∈ {a_j, b_j} (변 양 끝) 이면 b_j(q) = ν_j(q)·R_out(q) v_j 가 정확 항등식
      (n_j ⟂ 투영 변 벡터). 따라서
      b_j − ν_j·R_out e^{θ[u]×} v_j = −sinθ·u·(v_j × ν′_j) − (1−cosθ)·[(u·v_j)(u·ν′_j) − v_j·ν′_j],  ν′_j = R_out(q)ᵀν_j(q)
      θ = 0 의 h_O − h_I = 0 항은 구간으로 빼지 않고 처음부터 나타나지 않음 → sinθ, 1−cosθ 계수만 구간으로 감쌈.
증명서 k = (정확 균형 쌍대 꼭짓점 J, 접촉 선택): B(q,θ,u)/s′ ≤ −sinθ·d̂_k(q)·u + (1−cosθ)·b̂_k(q,u)
q 상자 처리: q 를 구간으로 넣어 R_out(q), 투영 꼭짓점, 법선, w*(q), s′(q), d̂_k(q) 를 구간으로 감쌈
      (구간 산술 포함성: 상자의 모든 q 의 정확값을 포함). 필요시 Q 를 균일 2^L × 2^L 하위상자로 나눔 (L ≤ LMAX, 사전 고정).
실행: python3 code/rupert_outer_nbhd.py
"""

import itertools
import json
import math
import sys
import time
from fractions import Fraction as Fr

import numpy as np
import mpmath
from mpmath import iv, mp

iv.dps = 30
mp.dps = 30
SEP = "=" * 108
out = []
T0 = time.time()

ORDER = [2, 3, 1, 0, 7]          # q0 의 바깥 오각형 꼭짓점 반시계 순서 (rupert_all_axes 와 동일)
HALF = Fr(1, 1024)
Q0 = (Fr(2, 5), Fr(9, 10))
SPHERE_INIT = 4                  # 면당 초기 4×4 (기존과 동일)
MAXD = 7                         # 구면 칸 최대 분할 깊이 (기존과 동일)
LMAX = 3                         # Q 하위상자 최대 수준 (사전 고정; 반폭 조정 아님)


def log(s=""):
    print(s, flush=True)
    out.append(s)


def I(x):
    if isinstance(x, Fr):
        return iv.mpf(x.numerator) / iv.mpf(x.denominator)
    return iv.mpf(x)


def lo(x):
    return mp.make_mpf(x._mpi_[0])


def hi(x):
    return mp.make_mpf(x._mpi_[1])


def to_fr(x):
    m, e = x.man_exp
    return Fr(m) * (Fr(2) ** e if e >= 0 else Fr(1, 2 ** (-e)))


def fl(x, k=12):
    """보증 하한 표시: 소수 k 자리에서 내림 (표시값 ≤ 실제 하한)."""
    f = to_fr(mp.mpf(x))
    n = math.floor(f * 10 ** k)
    s = "-" if n < 0 else ""
    n = abs(n)
    return f"{s}{n // 10 ** k}.{n % 10 ** k:0{k}d}"


def cl(x, k=12):
    """보증 상한 표시: 소수 k 자리에서 올림."""
    f = to_fr(mp.mpf(x))
    n = math.ceil(f * 10 ** k)
    s = "-" if n < 0 else ""
    n = abs(n)
    return f"{s}{n // 10 ** k}.{n % 10 ** k:0{k}d}"


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


def itr(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def icross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def idot(a, b):
    return sum((x * y for x, y in zip(a, b)), I(0))


def inorm(v):
    return iv.sqrt(sum((x ** 2 for x in v), I(0)))


P0 = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
T = Fr(11, 20)
V_EX = [tuple(Fr(c) for c in p) for p in P0] + [tuple(-T * c for c in p) for p in P0]
Viv = [[I(c) for c in v] for v in V_EX]
VNORM = [inorm(v) for v in Viv]


# ── q 상자 하나의 기하·균형·증명서 ───────────────────────────────
def setup(q1, q2):
    """q1, q2: 구간 (점이면 폭 0). 반환: 구조 인증 결과와 증명서."""
    Ro = imul(iry(-q1), irz(-q2))
    RoT = itr(Ro)
    W = [imv(Ro, v) for v in Viv]
    P = [(w[0], w[1]) for w in W]
    edges, strict_all, strict_min = [], True, None
    for i in range(len(ORDER)):
        a_, b_ = ORDER[i], ORDER[(i + 1) % len(ORDER)]
        ex, ey = P[b_][0] - P[a_][0], P[b_][1] - P[a_][1]
        n = (ey, -ex)                                    # 반시계 순서의 바깥 법선 (비정규화, 각 q 에서 정확)
        for k in range(8):
            if k in (a_, b_):
                continue
            s = n[0] * (P[k][0] - P[a_][0]) + n[1] * (P[k][1] - P[a_][1])
            strict_all &= hi(s) < 0
            strict_min = hi(s) if strict_min is None else max(strict_min, hi(s))
        nu = [n[0], n[1], I(0)]
        nrm = iv.sqrt(n[0] ** 2 + n[1] ** 2)
        edges.append(dict(a=a_, b=b_, n=n, nu=nu, nup=imv(RoT, nu), nrm=nrm))
    DV, infeasible, undetermined = [], [], []
    for (i, j, k) in itertools.combinations(range(len(edges)), 3):
        N = [edges[x]["n"] for x in (i, j, k)]
        A = iv.matrix([[I(1), I(1), I(1)], [N[0][0], N[1][0], N[2][0]], [N[0][1], N[1][1], N[2][1]]])
        det = iv.det(A)
        if lo(det) <= 0 <= hi(det):
            undetermined.append(((i, j, k), "det 0 포함"))
            continue
        wv = iv.lu_solve(A, iv.matrix([I(1), I(0), I(0)]))
        w = [wv[x] for x in range(3)]
        if all(lo(x) > 0 for x in w):
            s_unit = sum((w[t] * edges[x]["nrm"] for t, x in enumerate((i, j, k))), I(0))
            DV.append(dict(idx=(i, j, k), w=w, det=det, s=s_unit))
        elif any(hi(x) < 0 for x in w):
            infeasible.append((i, j, k))
        else:
            undetermined.append(((i, j, k), "w 부호 미확정"))
    certs = []
    for dv in DV:
        for choice in itertools.product(*[(edges[x]["a"], edges[x]["b"]) for x in dv["idx"]]):
            d = [I(0), I(0), I(0)]
            terms = []
            for wj, x, v in zip(dv["w"], dv["idx"], choice):
                c = icross(Viv[v], edges[x]["nup"])
                d = [d[t] + wj * c[t] for t in range(3)]
                terms.append((wj, x, v))
            certs.append(dict(dv=dv, choice=choice, dhat=[x / dv["s"] for x in d], terms=terms))
    return dict(edges=edges, strict=strict_all, strict_max=strict_min, DV=DV, infeasible=infeasible,
                undetermined=undetermined, certs=certs, W=W)


def face_u(face, s, t):
    if face == 0:
        v = [I(1), s, t]
    elif face == 1:
        v = [I(-1), s, t]
    elif face == 2:
        v = [s, I(1), t]
    elif face == 3:
        v = [s, I(-1), t]
    elif face == 4:
        v = [s, t, I(1)]
    else:
        v = [s, t, I(-1)]
    nrm = inorm(v)
    return [x / nrm for x in v], nrm


def b_hat(st, c, U):
    tot = I(0)
    for wj, x, v in c["terms"]:
        vp = st["edges"][x]["nup"]
        tot += wj * (idot(Viv[v], vp) - idot(U, Viv[v]) * idot(U, vp))
    return tot / c["dv"]["s"]


def cover(st):
    """구면 S² 칸 덮기. 칸마다 증명서 하나 선택, 그 칸(× q 상자) 전체의 하한 > 0 인증."""
    certs = st["certs"]
    cells_ok, cells_fail = [], []
    nrm_min = None
    g = SPHERE_INIT
    stack = [(f, Fr(-1) + Fr(2 * i, g), Fr(-1) + Fr(2 * (i + 1), g), Fr(-1) + Fr(2 * j, g), Fr(-1) + Fr(2 * (j + 1), g), 0)
             for f in range(6) for i in range(g) for j in range(g)]
    while stack:
        f, s0, s1, t0, t1, dep = stack.pop()
        U, nrm = face_u(f, iv.mpf([I(s0).a, I(s1).b]), iv.mpf([I(t0).a, I(t1).b]))
        nrm_min = lo(nrm) if nrm_min is None else min(nrm_min, lo(nrm))
        lows = [lo(idot(c["dhat"], U)) for c in certs]
        kbest = max(range(len(certs)), key=lambda k: lows[k])
        if lows[kbest] > 0:
            bh = hi(b_hat(st, certs[kbest], U))
            a_iv = iv.mpf(lows[kbest])
            dl = I(1) if bh <= 0 else iv.mpf(lows[kbest]) / iv.mpf(bh)
            dlo = min(mp.mpf(1), lo(dl))
            cells_ok.append(dict(f=f, s=(s0, s1), t=(t0, t1), dep=dep, k=kbest, a=lows[kbest], b=bh, delta=dlo))
            del a_iv
        elif dep < MAXD:
            sm, tm = (s0 + s1) / 2, (t0 + t1) / 2
            stack += [(f, s0, sm, t0, tm, dep + 1), (f, sm, s1, t0, tm, dep + 1),
                      (f, s0, sm, tm, t1, dep + 1), (f, sm, s1, tm, t1, dep + 1)]
        else:
            cells_fail.append(dict(f=f, s=(s0, s1), t=(t0, t1), dep=dep, k=kbest, a=lows[kbest]))
    return cells_ok, cells_fail, nrm_min


def bmax_interval(st):
    """증명서별 Σ w_j ‖ν′_j‖ ‖v_j‖ / s′ 의 구간 상한 최대 (‖ν′_j‖ = |n_j|)."""
    return max(hi(sum((wj * st["edges"][x]["nrm"] * VNORM[v] for wj, x, v in c["terms"]), I(0)) / c["dv"]["s"])
               for c in st["certs"])


BMAX_SYM = iv.sqrt(3)   # 기호 상계: Σ w_j|n_j|‖v_j‖/Σ w_j|n_j| ≤ max_j ‖v_j‖ ≤ √3  (w_j|n_j| > 0)


def deltas(cells_ok, a_min):
    d_uni = min(mp.mpf(1), lo(iv.mpf(a_min) / BMAX_SYM))
    d_cell = min(c["delta"] for c in cells_ok)
    return d_uni, d_cell


def summarize_dv(st, tag):
    log(f"  [{tag}] 오각형 순서 {ORDER} 유지·나머지 꼭짓점 엄격 내부 (구간): {st['strict']}"
        f"  (n_j·(p_k − p_a) 상한 최대 {mpmath.nstr(st['strict_max'], 6)} < 0)")
    for d in st["DV"]:
        log(f"    J{d['idx']}: det ∈ [{fl(lo(d['det']), 8)}, {cl(hi(d['det']), 8)}] (0 배제) → w*(q) 존재·유일;"
            f" w 하한 ({', '.join(fl(lo(x), 8) for x in d['w'])}) > 0;  s′ ≥ {fl(lo(d['s']), 8)} > 0")
    log(f"    쌍대 실현불가 인증 (det≠0, 어떤 w 상한 < 0): {st['infeasible']}")
    log(f"    미확정: {st['undetermined'] if st['undetermined'] else '없음'}")


# ═════════════════════════════════════════════════════════════════
log(SEP)
log("A. 기존 모든-축 인증 재현·감사 (q = q0 점)")
log(SEP)
log("  피복 논증: u ∈ S² 의 절댓값 최대 성분 i (|u_i| = m ≥ 1/√3 > 0) 를 잡으면 u/m 은 면 x_i = sign(u_i) 위에 있고")
log("    나머지 두 성분 ∈ [−1, 1]. 따라서 6면 × [−1,1]² 의 정규화 상이 S² 전체. 각 면 [−1,1]² 를 닫힌 칸 4×4 로 분할")
log("    (끝점 공유, 누락 경계 없음). 칸 구간 U = v/‖v‖ 는 칸의 모든 정규화 점을 포함 (구간 포함성).")
st0 = setup(I(Q0[0]), I(Q0[1]))
summarize_dv(st0, "q0")
log(f"  증명서 {len(st0['certs'])}개")
tA = time.time()
ok0, fail0, nmin0 = cover(st0)
tA = time.time() - tA
a0 = min(c["a"] for c in ok0) if not fail0 else None
log(f"  정규화 분모 ‖v‖ 하한 최소 = {fl(nmin0, 6)} ≥ 1 (한 성분이 정확히 ±1) → 0 배제: {nmin0 >= 1}")
log(f"  칸: 인증 {len(ok0)}, 실패 {len(fail0)}  ({tA:.1f}s).  선택 증명서 하한은 lo(d̂_k·U) — U 전체에 대한 구간 하한이므로 칸 전체 유효")
log("  분기: d̂·U, b̂ 계산에 max/abs/조건 분기 없음. 구조 판정(엄격 내부·쌍대 부호)은 미확정이면 제외/실패로 처리")
log(f"  a0_cert (보증 하한, 내림 표시) = {fl(a0, 15)}   [기존 보고 0.094877909200231]")
bmax0_iv = bmax_interval(st0)
d_uni0, d_cell0 = deltas(ok0, a0)
log(f"  균일 δ 도출: bmax = √3 (기호; 구간 계산 상한 {cl(bmax0_iv, 12)}),  δ_uni = lo(min(1, a0/√3)) = {fl(d_uni0, 13)}")
log(f"  칸별 δ 도출: 칸마다 lo(min(1, a_칸/b_칸)) (iv 나눗셈 하한), 최소 = {fl(d_cell0, 13)}")
log("  기존 표시값 대조: 균일 0.0547777864169, 칸별 0.0656753763868 (기존은 mpf 나눗셈 + nstr 반올림)")
for name, prev, new in (("균일", "0.0547777864169", d_uni0), ("칸별", "0.0656753763868", d_cell0)):
    ok = to_fr(mp.mpf(new)) >= Fr(prev)
    log(f"    {name}: 기존 표시 {prev} ≤ 보증 하한? {ok}" + ("" if ok else "  → 기존 표시값이 보증 하한보다 큼 (반올림 올림); 정정 필요"))

# ═════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("B–E. 바깥 방향 이웃 Q = q0 ± 1/1024 (각 성분)")
log(SEP)
log("  공통 회전: R_in = R_out(q)·exp(θ[u]×). 접촉 꼭짓점이 변 양 끝이므로 b_j(q) − ν_j·R_out(q)v_j = 0 이 기호 항등식 →")
log("  θ=0 공통항 소거 후 sinθ·(u·(v_j×ν′_j)), (1−cosθ)·(v_j·ν′_j − (u·v_j)(u·ν′_j)) 계수만 q 상자 구간으로 감쌈.")
result = None
for L in range(LMAX + 1):
    nb = 2 ** L
    log(f"  ── Q 하위상자 수준 L = {L} ({nb}×{nb})")
    tL = time.time()
    boxes = []
    all_ok = True
    for i in range(nb):
        for j in range(nb):
            q1a = Q0[0] - HALF + 2 * HALF * Fr(i, nb)
            q1b = Q0[0] - HALF + 2 * HALF * Fr(i + 1, nb)
            q2a = Q0[1] - HALF + 2 * HALF * Fr(j, nb)
            q2b = Q0[1] - HALF + 2 * HALF * Fr(j + 1, nb)
            q1 = iv.mpf([I(q1a).a, I(q1b).b])
            q2 = iv.mpf([I(q2a).a, I(q2b).b])
            st = setup(q1, q2)
            struct_ok = st["strict"] and len(st["DV"]) == 3 and not st["undetermined"]
            rec = dict(i=i, j=j, q1=(q1a, q1b), q2=(q2a, q2b), st=st, struct_ok=struct_ok)
            if not st["strict"]:
                rec.update(ok=False, why="오각형 구조/엄격 내부 미인증")
                all_ok = False
                boxes.append(rec)
                continue
            if not st["certs"]:
                rec.update(ok=False, why="양수 정확 균형 쌍대 꼭짓점 없음")
                all_ok = False
                boxes.append(rec)
                continue
            cok, cfail, nmin = cover(st)
            rec.update(cells_ok=cok, cells_fail=cfail, nrm_min=nmin)
            if cfail:
                rec.update(ok=False, why=f"구면 칸 실패 {len(cfail)}")
                all_ok = False
            else:
                rec.update(ok=True, a=min(c["a"] for c in cok), bmax_iv=bmax_interval(st))
            boxes.append(rec)
    tL = time.time() - tL
    nok = sum(1 for b in boxes if b["ok"])
    log(f"     하위상자 인증 {nok}/{len(boxes)}  ({tL:.1f}s)")
    if all_ok:
        result = dict(L=L, boxes=boxes, t=tL)
        break
    worst = [b for b in boxes if not b["ok"]][:4]
    for b in worst:
        extra = ""
        if "cells_fail" in b and b["cells_fail"]:
            cf = max(b["cells_fail"], key=lambda c: c["a"])
            extra = f"; 최선 실패 칸 면 {cf['f']} 하한 {mpmath.nstr(cf['a'], 6)}"
        log(f"     실패 상자 ({b['i']},{b['j']}): {b['why']}{extra}")
    last_boxes = boxes

if result is None:
    log("  ⇒ 사전 고정한 방법 (L ≤ 3, 구면 깊이 ≤ 7) 으로 Q 인증 미완. 반폭은 조정하지 않음. 병목은 위 실패 상자.")
    boxes = last_boxes
else:
    boxes = result["boxes"]
    L = result["L"]
    st_first = boxes[0]["st"]
    log("  [D. 기하·균형 — 하위상자별]")
    for b in boxes:
        summarize_dv(b["st"], f"상자 ({b['i']},{b['j']}) q1∈[{b['q1'][0]},{b['q1'][1]}] q2∈[{b['q2'][0]},{b['q2'][1]}]")
    same_dv = all([d["idx"] for d in b["st"]["DV"]] == [d["idx"] for d in st0["DV"]] for b in boxes)
    log(f"  쌍대 꼭짓점 집합이 q0 와 동일 ({[d['idx'] for d in st0['DV']]}): {same_dv}")
    aQ = min(b["a"] for b in boxes)
    bQ_iv = max(b["bmax_iv"] for b in boxes)
    all_cells = [c for b in boxes for c in b["cells_ok"]]
    dQ_uni = min(mp.mpf(1), lo(iv.mpf(aQ) / BMAX_SYM))
    dQ_cell = min(c["delta"] for c in all_cells)
    bQ_cells = max(c["b"] for c in all_cells)
    nmin = min(b["nrm_min"] for b in boxes)
    used = sorted({c["k"] for c in all_cells})
    depth = {}
    for c in all_cells:
        depth[c["dep"]] = depth.get(c["dep"], 0) + 1
    log("  [E. 결합 인증]")
    log(f"  하위상자 {len(boxes)}개 × 구면 칸: 인증 칸 총 {len(all_cells)}, 실패 0, 깊이 분포 {dict(sorted(depth.items()))},"
        f" 정규화 분모 하한 최소 {fl(nmin, 6)}")
    log(f"  사용 증명서 인덱스 {len(used)}개: {used}")
    log(f"  ⇒ ∀q ∈ Q, ∀‖u‖=1: max_k d̂_k(q)·u ≥ a_Q = {fl(aQ, 15)} > 0   (보증 하한, 내림 표시)")
    log(f"  이차항: b_Q (균일, 기호) = √3;  구간 계산 상한 max_k Σw‖ν′‖‖v‖/s′ ≤ {cl(bQ_iv, 12)};  칸별 b̂ 상한 최대 {cl(bQ_cells, 12)}")
    log(f"  δ_Q 균일 = lo(min(1, a_Q/√3)) = {fl(dQ_uni, 13)}")
    log(f"  δ_Q 칸별 = min_칸 lo(min(1, a_칸/b_칸)) = {fl(dQ_cell, 13)}")
    dQ = max(dQ_uni, dQ_cell)
    log(f"  ⇒ δ_Q = {fl(dQ, 13)} (두 값 모두 유효, 큰 쪽).  이 작은 바깥 방향 이웃 Q 에서 모든 상대 회전축과 0 < θ ≤ δ_Q 의 엄격 포함 배제")
    log("     (θ ∈ (0,1]: sinθ ≥ 5θ/6, 1−cosθ ≤ θ²/2 → θ < (5/3)a/b 이면 B < 0; δ = min(1, a/b) 는 그 안쪽)")
    result.update(aQ=aQ, bQ_iv=bQ_iv, bQ_cells=bQ_cells, dQ_uni=dQ_uni, dQ_cell=dQ_cell, dQ=dQ, used=used,
                  n_cells=len(all_cells), depth=depth, same_dv=same_dv)
log("  θ = 0 은 동일 투영의 경계 사례 (B = 0) 로 별도. 전체 배치 공간·다른 대칭 배치는 범위 밖.")
t_cert = time.time() - T0
log(f"  인증 계산 소요 {t_cert:.1f}s")

# ═════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("[비인증] 무작위 float 점검 — 인증과 분리")
log(SEP)
rand_info = None
try:
    from scipy.optimize import linprog
    from scipy.spatial.transform import Rotation as Rot

    Vf = np.array([[float(c) for c in v] for v in V_EX])

    def Ryf(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rzf(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def Mf(q1, q2, u, th):
        Ro = Ryf(-q1) @ Rzf(-q2)
        O = (Vf @ Ro.T)[:, :2]
        N, bb = [], []
        for i in range(5):
            a, b = O[ORDER[i]], O[ORDER[(i + 1) % 5]]
            e = b - a
            n = np.array([e[1], -e[0]])
            n /= np.linalg.norm(n)
            N.append(n)
            bb.append(n @ a)
        N, bb = np.array(N), np.array(bb)
        R = Ro @ Rot.from_rotvec(th * u).as_matrix()
        In = (Vf @ R.T)[:, :2]
        g = bb - np.max(In @ N.T, axis=0)
        r = linprog([0, 0, -1], A_ub=np.hstack([N, np.ones((5, 1))]), b_ub=g, bounds=[(None, None)] * 3)
        return -r.fun

    rng = np.random.default_rng(7)
    dref = float(result["dQ"]) if result else 0.05
    worst = -9.0
    h = float(HALF)
    for _ in range(2000):
        q1 = 0.4 + rng.uniform(-h, h)
        q2 = 0.9 + rng.uniform(-h, h)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        for th in (dref, dref / 3, dref / 30, 1e-4):
            worst = max(worst, Mf(q1, q2, u, th) / th)
    rand_info = dict(samples=8000, max_M_over_theta=worst)
    log(f"  무작위 q ∈ Q 2000 × 축 × θ ∈ {{δ_Q, δ_Q/3, δ_Q/30, 1e−4}}: max M/θ = {worst:.6f} (단위 법선, 실제 LP)")
except Exception as e:  # 점검 실패는 인증과 무관
    log(f"  점검 생략: {e!r}")

log(f"총 소요 {time.time() - T0:.1f}s")


def frs(x):
    return f"{x.numerator}/{x.denominator}"


data = {"environment": {"python": sys.version.split()[0], "mpmath": mpmath.__version__, "iv_dps": iv.dps},
        "params": {"q0": [frs(Q0[0]), frs(Q0[1])], "half_width": frs(HALF), "order": ORDER,
                   "sphere_init": SPHERE_INIT, "sphere_maxdepth": MAXD, "Q_level_max": LMAX},
        "audit_q0": {"cells_ok": len(ok0), "cells_fail": len(fail0), "a0_lower": fl(a0, 25),
                     "bmax_symbolic": "sqrt(3)", "bmax_interval_upper": cl(bmax0_iv, 20),
                     "delta_uniform_lower": fl(d_uni0, 20), "delta_cellwise_lower": fl(d_cell0, 20),
                     "normalizer_min_lower": fl(nmin0, 10)},
        "Q": None, "random_check_noncertified": rand_info}
if result:
    data["Q"] = {"level": result["L"], "n_boxes": len(boxes), "n_cells": result["n_cells"],
                 "depth_hist": {str(k): v for k, v in sorted(result["depth"].items())},
                 "dual_vertices_same_as_q0": result["same_dv"],
                 "a_Q_lower": fl(result["aQ"], 25), "b_Q_symbolic": "sqrt(3)",
                 "b_Q_interval_upper": cl(result["bQ_iv"], 20), "b_cells_upper": cl(result["bQ_cells"], 20),
                 "delta_Q_uniform_lower": fl(result["dQ_uni"], 20), "delta_Q_cellwise_lower": fl(result["dQ_cell"], 20),
                 "delta_Q": fl(result["dQ"], 20), "used_certificates": result["used"],
                 "boxes": [{"q1": [frs(b["q1"][0]), frs(b["q1"][1])], "q2": [frs(b["q2"][0]), frs(b["q2"][1])],
                            "a_lower": fl(b["a"], 20),
                            "dual_vertices": [{"edges": list(d["idx"]),
                                               "w_lower": [fl(lo(x), 15) for x in d["w"]],
                                               "w_upper": [cl(hi(x), 15) for x in d["w"]],
                                               "det": [fl(lo(d["det"]), 12), cl(hi(d["det"]), 12)],
                                               "s_lower": fl(lo(d["s"]), 15)} for d in b["st"]["DV"]],
                            "strict_max_upper": cl(b["st"]["strict_max"], 12)} for b in boxes]}
else:
    data["Q"] = {"certified": False,
                 "failed_boxes": [{"i": b["i"], "j": b["j"], "why": b["why"]} for b in boxes if not b["ok"]]}
with open("code/rupert_outer_nbhd_data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
with open("code/rupert_outer_nbhd_output.txt", "w") as f:
    f.write("\n".join(out) + "\n")

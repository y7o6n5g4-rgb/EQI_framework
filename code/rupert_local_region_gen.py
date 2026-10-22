#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국소 인증 영역의 기계 판독 증명서 생성기 (탐색기; mpmath iv, LU 경로)
영역: P_{11/20}, Q = q0 ± 1/1024 (q0 = (2/5, 9/10)), 모든 단위 상대 회전축 u, 0 < θ ≤ δ.
기하·증명서 함수는 rupert_outer_nbhd.py (ed8c701) 와 동일한 정의를 복사 (보존본 무수정).
출력: code/rupert_local_region_cert.json (증명서), code/rupert_local_region_gen_output.txt
독립 재검증: code/rupert_local_region_verify.py (저장 자료만 사용, Fraction 구간 + Cramer)
실행: python3 code/rupert_local_region_gen.py
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
DELTA_SAFE = Fr(1, 20)
q1 = iv.mpf([I(Q0[0] - HALF).a, I(Q0[0] + HALF).b])
q2 = iv.mpf([I(Q0[1] - HALF).a, I(Q0[1] + HALF).b])
st = setup(q1, q2)
log(SEP)
log("국소 인증 영역 증명서 생성 (탐색기, mpmath iv LU 경로)")
log(SEP)
summarize_dv(st, "Q")
cok, cfail, nmin = cover(st)
ok = st["strict"] and not cfail and len(st["certs"]) == 24
log(f"  구면 칸 인증 {len(cok)}, 실패 {len(cfail)}, 정규화 분모 하한 최소 {fl(nmin, 6)}")
aQ = min(c["a"] for c in cok)
dQ_cell = min(c["delta"] for c in cok)
dQ_uni = min(mp.mpf(1), lo(iv.mpf(aQ) / BMAX_SYM))
safe_ok = all(to_fr(mp.mpf(c["delta"])) >= DELTA_SAFE for c in cok)
log(f"  a_Q ≥ {fl(aQ, 15)},  δ_Q(칸별) ≥ {fl(dQ_cell, 13)},  δ_Q(균일) ≥ {fl(dQ_uni, 13)}")
log(f"  δ_safe = 1/20 ≤ 모든 칸 δ_칸 (보증 하한): {safe_ok}   (균일 δ 로는 1/20 미달 → δ_safe 는 칸별 도출에 의존)")
if ok and safe_ok:
    log("  [정리 후보 — 독립 검증기로 재확인 필요] ∀q∈Q, ∀‖u‖=1, 0 < θ ≤ 1/20: B < 0  (θ = 0 은 별도 보조정리)")
cells = []
for c in sorted(cok, key=lambda c: (c["f"], c["s"][0], c["t"][0])):
    cells.append({"face": c["f"], "s": [str(c["s"][0]), str(c["s"][1])], "t": [str(c["t"][0]), str(c["t"][1])],
                  "depth": c["dep"], "cert": c["k"], "a_lower": fl(c["a"], 20), "b_upper": cl(c["b"], 20),
                  "delta_lower": fl(c["delta"], 20)})
cert = {
    "format": "rupert_local_region_cert v1",
    "environment": {"python": sys.version.split()[0], "mpmath": mpmath.__version__, "iv_dps": iv.dps},
    "display_rule": "하한은 정확 유리수 변환 후 소수 k 자리 내림(floor), 상한은 올림(ceil). 표시값은 보증값보다 유리하지 않음.",
    "object": {"t_shape": "11/20", "definition": "P_t = conv{p_i, -t p_i}, p_i in P0",
               "P0": [list(p) for p in P0], "vertices": [[str(x) for x in v] for v in V_EX]},
    "rotation": {"factor_order": "R_out(q) = Ry(-q1) @ Rz(-q2); R_in(q,theta,u) = R_out(q) @ expm(theta*[u]x); shadow = first two coords of R v",
                 "Ry(a)": "[[cos a,0,sin a],[0,1,0],[-sin a,0,cos a]]", "Rz(a)": "[[cos a,-sin a,0],[sin a,cos a,0],[0,0,1]]"},
    "Q": {"q1": [str(Q0[0] - HALF), str(Q0[0] + HALF)], "q2": [str(Q0[1] - HALF), str(Q0[1] + HALF)],
          "center": [str(Q0[0]), str(Q0[1])], "half_width": str(HALF)},
    "pentagon": {"order_ccw": ORDER,
                 "edges": [{"j": j, "a": e["a"], "b": e["b"], "normal": "n_j = (ey, -ex), e = p_b - p_a (unnormalized, outward)",
                            "contact_set_theta0": [e["a"], e["b"]]} for j, e in enumerate(st["edges"])]},
    "balance": {"system": "[[1,1,1],[n_x(j1),n_x(j2),n_x(j3)],[n_y(j1),n_y(j2),n_y(j3)]] w = (1,0,0)",
                "s_prime": "sum_j w_j |n_j|",
                "dual_vertices": [{"edges": list(d["idx"]), "det": [fl(lo(d["det"]), 15), cl(hi(d["det"]), 15)],
                                   "w": [[fl(lo(x), 15), cl(hi(x), 15)] for x in d["w"]],
                                   "s_prime_lower": fl(lo(d["s"]), 15)} for d in st["DV"]]},
    "certificates": {"definition": "k = (J, (v_j)_{j in J}), v_j in contact set of edge j; "
                                   "dhat_k = sum w_j (v_j x R^T nu_j) / s'; bhat_k(u) = sum w_j [v_j.nu'_j - (u.v_j)(u.nu'_j)] / s', nu'_j = R^T (n_j,0)",
                     "list": [{"k": k, "J": list(c["dv"]["idx"]), "contact": list(c["choice"])} for k, c in enumerate(st["certs"])]},
    "sphere": {"map": "face 0:(1,s,t) 1:(-1,s,t) 2:(s,1,t) 3:(s,-1,t) 4:(s,t,1) 5:(s,t,-1); u = v/|v|; s,t in [-1,1]",
               "cells": cells},
    "bound": {"inequality": "B(q,theta,u)/s' <= -sin(theta) dhat_k.u + (1-cos(theta)) bhat_k(u)  (exact)",
              "trig": "theta in (0,1]: sin theta >= 5 theta/6, 1-cos theta <= theta^2/2",
              "cell_delta": "delta_cell = min(1, a_cell/b_cell) (b_cell>0), else 1; valid since theta < (5/3) a/b suffices"},
    "results": {"a_Q_lower": fl(aQ, 25), "delta_Q_cellwise_lower": fl(dQ_cell, 20),
                "delta_Q_uniform_lower": fl(dQ_uni, 20), "b_symbolic": "sqrt(3)",
                "delta_safe": "1/20", "delta_safe_le_all_cells": safe_ok},
    "theta0_lemma": "theta=0: inner shadow = outer shadow O (compact, area>0). O+tau in int(O) impossible for all tau: "
                    "tau=0 -> boundary points of O not in int(O); tau!=0 -> x maximizing tau.x on O gives x+tau not in O.",
}
with open("code/rupert_local_region_cert.json", "w") as f:
    json.dump(cert, f, ensure_ascii=False, indent=1)
log(f"  증명서 기록: code/rupert_local_region_cert.json (칸 {len(cells)}, 증명서 {len(st['certs'])})")
log(f"총 소요 {time.time() - T0:.1f}s")
with open("code/rupert_local_region_gen_output.txt", "w") as f:
    f.write("\n".join(out) + "\n")

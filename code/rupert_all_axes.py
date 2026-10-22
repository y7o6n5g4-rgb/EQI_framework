#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고정 바깥 배치에서 모든 상대 회전축의 국소 배제 인증 (P_{11/20}, 오각형 바깥 투영 고정)
원본 엔진·legacy·보존 파일 무수정. EQI 상수 미사용. t_shape = 11/20 (t_EQI·T_model 과 무관).

설정: R_out = Ry(−2/5)Rz(−9/10) (정확 유리수 각),  R_in(θ) = R_out·exp(θ[u]×), u ∈ S² (단위축), θ > 0.
     (θ < 0 은 축 −u 와 같으므로 S² 전체 + θ > 0 으로 양 부호를 덮음)
증명서 k = (정확 균형 쌍대 꼭짓점 J = {j1,j2,j3}, 각 변 j ∈ J 의 접촉 꼭짓점 선택 v_j ∈ S_j):
  B(θ) ≤ Σ_{j∈J} w_j [b_j − ν_j·R_out e^{θ[u]×} v_j] = −sinθ·a_k(u) + (1−cosθ)·b_k(u)       (정확, Rodrigues)
  a_k(u) = Σ w_j ν_j·R_out(u×v_j) = d_k·u,   d_k = Σ w_j (v_j × R_outᵀν_j)        (u 에 일차)
  b_k(u) = −Σ w_j ν_j·R_out(u×(u×v_j)) = Σ w_j [v_j·ν'_j − (u·v_j)(u·ν'_j)],  ν' = R_outᵀν
  단위 여유 환산: n_j = |n_j| n̂_j → 단위 법선 가중치 w_j|n_j|, 합 s′_k = Σ w_j|n_j|. d̂_k = d_k/s′_k, b̂_k = b_k/s′_k
  (B/s′ 는 단위 법선 기준 여유; 부호·δ 불변)
인증 목표: ∀u ∈ S²: max_k d̂_k·u ≥ a0 > 0  (⇔ 원점 중심 반지름 a0 공 ⊆ D = conv{d̂_k})
  방법: 정육면체-구면 사상 u = (±1, s, t)/‖·‖ 등 6면, (s,t) 칸 적응 분할, 구간 산술로 칸마다
        max_k lower(d̂_k·U) ≥ (칸 하계) 인증. 칸별로 그 k 의 b̂_k(U) 상한 → 칸별 δ.
  float convex hull 은 a0 추정·구조 확인에만 사용.
δ: θ ∈ (0,1] 에서 sinθ ≥ 5θ/6, 1−cosθ ≤ θ²/2  →  (5/6)aθ > (1/2)bθ² ⇔ θ < (5/3)a/b.
   δ = min(1, a/b) (b > 0) 는 이 조건을 엄격히 만족 (θ = a/b 에서 5/6 > 1/2), b ≤ 0 이면 δ = 1.
실행: python3 code/rupert_all_axes.py
"""

import itertools
import json
import sys
import time
from fractions import Fraction as Fr

import numpy as np
import mpmath
from mpmath import iv, mp
from scipy.spatial import ConvexHull

iv.dps = 30
mp.dps = 30
SEP = "=" * 108
out = []
T0 = time.time()


def log(s=""):
    print(s, flush=True)
    out.append(s)


def I(x):
    return iv.mpf(x)


def lo(x):
    return mp.make_mpf(x._mpi_[0])


def hi(x):
    return mp.make_mpf(x._mpi_[1])


def F(x, k=12):
    a, b = lo(x), hi(x)
    return mpmath.nstr(a, k) if a == b else f"[{mpmath.nstr(a, k)}, {mpmath.nstr(b, k)}]"


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


# ── 고정 바깥 배치 ───────────────────────────────────────────────
P0 = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
T = Fr(11, 20)
V_EX = [tuple(Fr(c) for c in p) for p in P0] + [tuple(-T * c for c in p) for p in P0]
Viv = [[I(c.numerator) / I(c.denominator) for c in v] for v in V_EX]
Ro = imul(iry(-I(2) / 5), irz(-I(9) / 10))
RoT = itr(Ro)
W = [imv(Ro, v) for v in Viv]
S2 = [(w[0], w[1]) for w in W]
Sf = np.array([[float(lo(x)) for x in s] for s in S2])
hull2 = ConvexHull(Sf)
order = [int(x) for x in hull2.vertices]
edges = []
for i in range(len(order)):
    a_, b_ = order[i], order[(i + 1) % len(order)]
    ex, ey = S2[b_][0] - S2[a_][0], S2[b_][1] - S2[a_][1]
    n = (ey, -ex)
    b0 = n[0] * S2[a_][0] + n[1] * S2[a_][1]
    strict = all(hi(n[0] * S2[k][0] + n[1] * S2[k][1] - b0) < 0 for k in range(8) if k not in (a_, b_))
    nu = [n[0], n[1], I(0)]
    edges.append(dict(a=a_, b=b_, n=n, nu=nu, nup=imv(RoT, nu), nrm=iv.sqrt(n[0] ** 2 + n[1] ** 2), strict=strict))

log(SEP)
log("고정 바깥 배치: P_{11/20}, R_out = Ry(−2/5)Rz(−9/10), 바깥 그림자 오각형 꼭짓점 (반시계) " + str(order))
log(f"  모든 변에서 나머지 꼭짓점 엄격 내부 (구간 인증): {all(e['strict'] for e in edges)} → 접촉 집합 S_j = 변 양 끝 (θ=0 동률 정확)")
log(SEP)

# ── A. 정확 균형 쌍대 꼭짓점 ─────────────────────────────────────
log("A. 정확 균형의 보장 방식")
DV = []
for (i, j, k) in itertools.combinations(range(len(edges)), 3):
    N = [edges[x]["n"] for x in (i, j, k)]
    A = iv.matrix([[I(1), I(1), I(1)], [N[0][0], N[1][0], N[2][0]], [N[0][1], N[1][1], N[2][1]]])
    det = iv.det(A)
    if lo(det) <= 0 <= hi(det):
        log(f"  변 ({i},{j},{k}): det 구간이 0 포함 → 유일성 미인증, 제외")
        continue
    wv = iv.lu_solve(A, iv.matrix([I(1), I(0), I(0)]))
    w = [wv[x] for x in range(3)]
    r = [sum((w[t] * N[t][c] for t in range(3)), I(0)) for c in (0, 1)]
    if all(lo(x) > 0 for x in w):
        s_unit = sum((w[t] * edges[x]["nrm"] for t, x in enumerate((i, j, k))), I(0))
        DV.append(dict(idx=(i, j, k), w=w, det=det, r=r, s=s_unit))
        log(f"  변 ({i},{j},{k}): det = {F(det, 10)} (0 배제) → 정확 연립해 w* 존재·유일; 구간 해 w ∈ "
            f"({', '.join(F(x, 10) for x in w)}), 하한 > 0 → w* > 0")
        log(f"      재계산 잔차 Σw n 구간 = ({F(r[0], 3)}, {F(r[1], 3)}) — 0 을 포함하지만 이것이 r=0 의 근거가 아님;"
            f" 근거는 w* 가 연립식 행 2·3 (Σw n = 0) 의 정확해라는 사실")
        log(f"      단위 여유 환산 s′ = Σ w|n_j| = {F(s_unit, 12)}")
    else:
        log(f"  변 ({i},{j},{k}): 해의 성분 부호 비양수 (또는 미확정) → 쌍대 실현불가, 제외")
log(f"  → 정확 균형 쌍대 꼭짓점 {len(DV)}개")

# ── B. 유한 증명서 집합 ─────────────────────────────────────────
log("")
log(SEP)
log("B. 유한 증명서 집합: (쌍대 꼭짓점) × (각 변 접촉 꼭짓점 2 선택)³")
log(SEP)
CERTS = []
for dv in DV:
    for choice in itertools.product(*[(edges[x]["a"], edges[x]["b"]) for x in dv["idx"]]):
        d = [I(0), I(0), I(0)]
        terms = []
        for wj, x, v in zip(dv["w"], dv["idx"], choice):
            c = icross(Viv[v], edges[x]["nup"])           # c_j(v) = v × R_outᵀν_j,  α_j(v;u) = u·c_j(v)
            d = [d[t] + wj * c[t] for t in range(3)]
            terms.append((wj, x, v))
        dhat = [x / dv["s"] for x in d]
        CERTS.append(dict(dv=dv, choice=choice, d=d, dhat=dhat, terms=terms))
log(f"  증명서 {len(CERTS)}개 = {len(DV)} 쌍대 꼭짓점 × 8 접촉 선택")
log("  d_k = Σ w_j (v_j × R_outᵀν_j)  (항등식 ν·R(u×v) = (Rᵀν)·(u×v) = u·(v × Rᵀν)),  d̂_k = d_k/s′")
# 회귀: 기존 세 축 ± 에서 max_k d̂_k·u 와 이전 a/s′
AXES = [("(1,0,0)", [I(1), I(0), I(0)]), ("(0,1,0)", [I(0), I(1), I(0)]),
        ("(1,2,3)/√14", [I(1) / iv.sqrt(14), I(2) / iv.sqrt(14), I(3) / iv.sqrt(14)])]
PREV = {("(1,0,0)", "+"): "0.565196624404", ("(1,0,0)", "−"): "0.787141150123",
        ("(0,1,0)", "+"): "1.03203091699", ("(0,1,0)", "−"): "0.64931170572",
        ("(1,2,3)/√14", "+"): "1.17569240514", ("(1,2,3)/√14", "−"): "1.2449378797"}
log("  [회귀] 기존 여섯 경로: max_k d̂_k·u  vs  rupert_local_ineq (9a77ac0) 의 a/s′")
reg = []
for name, u in AXES:
    for sg, lab in ((1, "+"), (-1, "−")):
        uu = [sg * x for x in u]
        vals = [idot(c["dhat"], uu) for c in CERTS]
        best = max(vals, key=lambda v: lo(v))
        prev = mpmath.mpf(PREV[(name, lab)])
        ae = abs(lo(best) - prev)
        log(f"    {name:<12} {lab}: {F(best, 14):<22} 이전 {PREV[(name, lab)]:<16} 절대차 {mpmath.nstr(ae, 3)},"
            f" 상대차 {mpmath.nstr(ae / prev, 3)}")
        reg.append(dict(axis=name, sign=lab, now=mpmath.nstr(lo(best), 20), prev=PREV[(name, lab)],
                        abs_err=mpmath.nstr(ae, 4), rel_err=mpmath.nstr(ae / prev, 4)))

# ── C. 모든 축: float 구조 + 구간 덮기 인증 ─────────────────────
log("")
log(SEP)
log("C. 모든 단위축에서 max_k d̂_k·u > 0 — D = conv{d̂_k}")
log(SEP)
Df = np.array([[float((lo(x) + hi(x)) / 2) for x in c["dhat"]] for c in CERTS])
h3 = ConvexHull(Df)
dist = -h3.equations[:, 3]                                # 원점에서 각 면 평면까지 (단위 법선)
a0_float = dist.min()
f_arg = int(np.argmin(dist))
log(f"  [float, 비인증] D 꼭짓점 {len(h3.vertices)}/{len(CERTS)}, 면 {len(h3.simplices)}, 원점 내부: {bool(np.all(dist > 0))}")
log(f"  [float, 비인증] min_u max_k d̂_k·u ≈ 원점-면 최소거리 = {a0_float:.12f},"
    f" 최약 방향 u* ≈ ({', '.join(f'{x:.6f}' for x in h3.equations[f_arg, :3])})")


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
    nrm = iv.sqrt(sum((x ** 2 for x in v), I(0)))
    return [x / nrm for x in v]


def b_hat(c, U):
    tot = I(0)
    for wj, x, v in c["terms"]:
        vp = edges[x]["nup"]
        tot += wj * (idot(Viv[v], vp) - idot(U, Viv[v]) * idot(U, vp))
    return tot / c["dv"]["s"]


MAXD = 7
cells_ok, cells_fail = [], []
stack = [(f, mpmath.mpf(-1) + 2 * mpmath.mpf(i) / 4, mpmath.mpf(-1) + 2 * mpmath.mpf(i + 1) / 4,
          mpmath.mpf(-1) + 2 * mpmath.mpf(j) / 4, mpmath.mpf(-1) + 2 * mpmath.mpf(j + 1) / 4, 0)
         for f in range(6) for i in range(4) for j in range(4)]
tC = time.time()
while stack:
    f, s0, s1, t0, t1, dep = stack.pop()
    U = face_u(f, iv.mpf([s0, s1]), iv.mpf([t0, t1]))
    lows = [lo(idot(c["dhat"], U)) for c in CERTS]
    kbest = int(np.argmax([float(x) for x in lows]))
    if lows[kbest] > 0:
        bh = hi(b_hat(CERTS[kbest], U))
        cells_ok.append((f, s0, s1, t0, t1, dep, kbest, lows[kbest], bh))
    elif dep < MAXD:
        sm, tm = (s0 + s1) / 2, (t0 + t1) / 2
        stack += [(f, s0, sm, t0, tm, dep + 1), (f, sm, s1, t0, tm, dep + 1),
                  (f, s0, sm, tm, t1, dep + 1), (f, sm, s1, tm, t1, dep + 1)]
    else:
        cells_fail.append((f, s0, s1, t0, t1, dep, kbest, lows[kbest]))
tC = time.time() - tC
covered = len(cells_fail) == 0
a0_cert = min(c[7] for c in cells_ok) if cells_ok else None
log(f"  [구간 인증] 정육면체-구면 6면 × 초기 4×4, 적응 분할 최대 깊이 {MAXD}: 인증 칸 {len(cells_ok)}, 실패 칸 {len(cells_fail)}"
    f"  ({tC:.1f}s)")
depth_hist = {}
for c in cells_ok:
    depth_hist[c[5]] = depth_hist.get(c[5], 0) + 1
log(f"     인증 칸 깊이 분포: {dict(sorted(depth_hist.items()))}")
if covered:
    log(f"  ⇒ 모든 단위축 u 에 대해 max_k d̂_k·u ≥ a0_cert = {mpmath.nstr(a0_cert, 14)} > 0  (구간 인증)")
    log(f"     float 추정 a0 ≈ {a0_float:.12f} (칸 하계는 칸 크기만큼 보수적; 인증값 ≤ 실제값)")
else:
    log("  ⇒ 이 유한 증명서 집합·분할 깊이로 인증 미완 (국소 통과 존재나 EQI 실패를 뜻하지 않음)")
    for c in cells_fail[:10]:
        log(f"     실패 칸: 면 {c[0]}, s ∈ [{mpmath.nstr(c[1], 6)}, {mpmath.nstr(c[2], 6)}], t ∈ [{mpmath.nstr(c[3], 6)},"
            f" {mpmath.nstr(c[4], 6)}], 최선 하계 {mpmath.nstr(c[7], 6)}")
used = sorted({c[6] for c in cells_ok})
log(f"  사용된 증명서 {len(used)}개: " + ", ".join(f"k{k}=J{CERTS[k]['dv']['idx']}·v{CERTS[k]['choice']}" for k in used))

# ── D. 이차항 상계와 δ ───────────────────────────────────────────
log("")
log(SEP)
log("D. 이차항 균일 상계와 δ")
log(SEP)
bmax_uniform = max(hi(sum((wj * iv.sqrt(sum((y ** 2 for y in edges[x]["nup"]), I(0))) *
                            iv.sqrt(sum((y ** 2 for y in Viv[v]), I(0))) for wj, x, v in c["terms"]), I(0)) / c["dv"]["s"])
                   for c in CERTS)
log(f"  균일 상계 (조잡): |ν′·(u×(u×v))| ≤ ‖ν′‖‖v‖ → b̂_k(u) ≤ Σ w_j‖ν′_j‖‖v_j‖/s′ ≤ bmax_uniform = {mpmath.nstr(bmax_uniform, 12)}")
if covered:
    bmax_cells = max(max(c[8], mpmath.mpf(0)) for c in cells_ok)
    delta_uniform = min(mpmath.mpf(1), a0_cert / bmax_uniform) if bmax_uniform > 0 else mpmath.mpf(1)
    delta_cells = min((min(mpmath.mpf(1), c[7] / c[8]) if c[8] > 0 else mpmath.mpf(1)) for c in cells_ok)
    log(f"  칸별 상계: 각 칸에서 사용한 증명서의 b̂_k(U) 구간 상한, 최대 bmax_cells = {mpmath.nstr(bmax_cells, 12)}")
    log(f"  δ (균일: min(1, a0_cert/bmax_uniform))         = {mpmath.nstr(delta_uniform, 12)}")
    log(f"  δ (칸별: min_칸 min(1, a_칸/b_칸))              = {mpmath.nstr(delta_cells, 12)}")
    delta = max(delta_uniform, delta_cells)
    log(f"  ⇒ 증명 범위: 모든 단위축 u, 0 < θ ≤ δ = {mpmath.nstr(delta, 12)} 에서 B(θ)/s′ ≤ −(5/6)θ·a + (1/2)θ²·b < 0"
        f"  (고정 바깥 배치 하나에서 모든 상대 회전축의 국소 엄격 포함 배제)")
    log("     δ 근거: θ ∈ (0,1] 에서 sinθ ≥ θ − θ³/6 ≥ 5θ/6, 1−cosθ ≤ θ²/2 (독립 확인), θ < (5/3)a/b 이면 충분,"
        " δ = min(1, a/b) 는 그 안쪽")
else:
    delta = None
log("  θ = 0 은 별도 경계 상태 (B(0) = 0). 바깥 방향이 변하는 이웃·다른 대칭 배치·전체 배치 공간은 범위 밖.")
log(f"총 소요 {time.time() - T0:.1f}s")

with open("code/rupert_all_axes_data.json", "w") as f:
    json.dump({"environment": {"python": sys.version.split()[0], "mpmath": mpmath.__version__, "iv_dps": iv.dps},
               "outer": {"R_out": "Ry(-2/5) @ Rz(-9/10)", "hull_order": order,
                         "edges": [[e["a"], e["b"]] for e in edges]},
               "dual_vertices": [{"edges": list(d["idx"]), "w": [[mpmath.nstr(lo(x), 25), mpmath.nstr(hi(x), 25)] for x in d["w"]],
                                  "det": [mpmath.nstr(lo(d["det"]), 15), mpmath.nstr(hi(d["det"]), 15)],
                                  "s_unit": [mpmath.nstr(lo(d["s"]), 25), mpmath.nstr(hi(d["s"]), 25)]} for d in DV],
               "certificates": [{"k": k, "edges": list(c["dv"]["idx"]), "contact_vertices": list(c["choice"]),
                                 "dhat": [[mpmath.nstr(lo(x), 25), mpmath.nstr(hi(x), 25)] for x in c["dhat"]]}
                                for k, c in enumerate(CERTS)],
               "regression_six_paths": reg,
               "cover": {"max_depth": MAXD, "n_ok": len(cells_ok), "n_fail": len(cells_fail),
                         "a0_cert": mpmath.nstr(a0_cert, 25) if a0_cert is not None else None,
                         "a0_float_estimate": a0_float, "bmax_uniform": mpmath.nstr(bmax_uniform, 25),
                         "delta": mpmath.nstr(delta, 25) if delta is not None else None,
                         "used_certificates": used}}, f, ensure_ascii=False, indent=1)
with open("code/rupert_all_axes_output.txt", "w") as f:
    f.write("\n".join(out) + "\n")

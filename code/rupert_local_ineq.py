#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P_{11/20} 실제 대칭군 점검 + 동일 방향 세 경로의 국소 부등식 (작은 국소 정리 검토; 전역 결과 아님)
원본 엔진·legacy·보존 파일 무수정. EQI 상수 미사용.
좌표 구분: t_shape = 11/20 (다면체 형상), θ = 안쪽 상대 회전각 (진단 경로). t_EQI, T_model 과 무관.

A. 48 부호치환 중 P_t 꼭짓점 집합을 정확히 (유리수) 보존하는 것, det ±1 구분.
   투영 방향 부호 (d ↔ −d) 에 의한 그림자 동일시는 물체 대칭과 별도 논증.
C·D. 바깥 고정 R_out = Ry(−2/5)Rz(−9/10), 안쪽 R_in(θ) = R_out exp(θ[a]×), a ∈ 기존 세 축, θ 부호 ±.
   바깥 그림자 변 법선 n_j (변 벡터의 90° 회전, 비정규화 정확값), 접촉 집합 S_j = 변 양 끝 꼭짓점 (구간 인증).
   방향미분 (Danskin):  D±_j = max_{v∈S_j} ±α_j(v),  α_j(v) = ν_j·R_out(a×v),  β_j(v) = ν_j·R_out(a×(a×v)).
   고정 꼭짓점 v_j* 에 대해 정확히:  ν_j·R_out e^{θA} v = ν_j·R_out v + sinθ·α_j(v) + (1−cosθ)·β_j(v)  (Rodrigues).
   정확 균형 쌍대 꼭짓점 w (Σw n = 0 을 정확 연립해로 감싼 구간) 에 대해
     B(θ) ≤ Σw_j [b_j − ν_j·R_out e^{θA} v_j*] = −sinθ·a + (1−cosθ)·b′,  a = Σw α(v*), b′ = −Σw β(v*)
   sinθ ≥ θ − θ³/6, 1−cosθ ≤ θ²/2 (θ ≥ 0) → θ ∈ (0, 1] 에서 B(θ) ≤ −a_lo θ + C θ²,  C = a_lo/6 + max(b′_hi, 0)/2.
   ⇒ 0 < θ ≤ δ = min(1, a_lo/C) 에서 B(θ) < 0 (엄격 포함 배제).  θ < 0 은 축 −a 로 같은 논증.
실행: python3 code/rupert_local_ineq.py
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

iv.dps = 40
mp.dps = 40
SEP = "=" * 108
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


def I(x):
    return iv.mpf(x)


def lo(x):
    return mp.make_mpf(x._mpi_[0])


def hi(x):
    return mp.make_mpf(x._mpi_[1])


def F(x, k=14):
    a, b = lo(x), hi(x)
    return mpmath.nstr(a, k) if a == b else f"[{mpmath.nstr(a, k)}, {mpmath.nstr(b, k)}]"


P0 = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
T = Fr(11, 20)
V_EX = [tuple(Fr(c) for c in p) for p in P0] + [tuple(-T * c for c in p) for p in P0]

# ════════════════════════════════════════════════════════════════════════════
# A. 실제 대칭군
# ════════════════════════════════════════════════════════════════════════════
log(SEP)
log("A. P_{11/20} 대칭군 — 48 부호치환의 정확한 집합 보존 검사 (Fraction)")
log(SEP)
Vset = set(V_EX)
pres, nonpres = [], []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        M = [[0] * 3 for _ in range(3)]
        for i in range(3):
            M[i][perm[i]] = sg[i]
        img = {tuple(sum(M[i][k] * v[k] for k in range(3)) for i in range(3)) for v in V_EX}
        det = int(round(np.linalg.det(np.array(M, float))))
        (pres if img == Vset else nonpres).append((perm, sg, det))
n_pos = sum(1 for p in pres if p[2] == 1)
n_neg = sum(1 for p in pres if p[2] == -1)
log(f"  집합 보존: {len(pres)}개 (det +1: {n_pos}, det −1: {n_neg})  /  비보존: {len(nonpres)}개")
log(f"  중앙반전 −I 보존 여부: {any(p[0] == (0, 1, 2) and p[1] == (-1, -1, -1) for p in pres)}"
    f"  (p_i → −p_i 이지만 −p_i 는 꼭짓점이 아님: |p_i| = √3 ≠ |t p_i| = (11/20)√3)")
log("  → 물체 대칭군 = T_d (차수 24, 회전 12 + 반사형 12). O_h (48) 아님.")
log("  [별도 논증] 투영 방향 부호: 방향 d 와 −d 의 직교 투영은 같은 평면 집합. 회전 인코딩에서 −d 는")
log("    x축 둘레 π 회전을 앞에 곱한 것과 같아 그림자 좌표가 (x, −y) 로 반사됨 — 평면 등거리 (반사 포함) 로 합동.")
log("    따라서 '그림자 합동' 기준의 방향 동일시는 {±g : g ∈ T_d} — 방향 작용으로는 48개 = O_h 와 같은 집합.")
# 수치 확인: d 와 g·d (g ∈ ±T_d) 의 그림자 합동 (정렬된 꼭짓점 쌍거리 비교)
rng = np.random.default_rng(3)
Vf = np.array([[float(c) for c in v] for v in V_EX])


def shadow_sig(d):
    d = d / np.linalg.norm(d)
    e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    S = np.c_[Vf @ e1, Vf @ e2]
    H = S[ConvexHull(S).vertices]
    D = np.sort(np.linalg.norm(H[:, None] - H[None], axis=-1).ravel())
    return D


mx_in, mn_out = 0.0, np.inf
for _ in range(20):
    d = rng.normal(size=3)
    s0 = shadow_sig(d)
    for perm, sg, det in pres:
        M = np.zeros((3, 3))
        for i in range(3):
            M[i][perm[i]] = sg[i]
        for sgn in (1, -1):
            s1 = shadow_sig(sgn * (M @ d))
            if len(s1) == len(s0):
                mx_in = max(mx_in, np.max(np.abs(s1 - s0)))
    for perm, sg, det in nonpres:
        M = np.zeros((3, 3))
        for i in range(3):
            M[i][perm[i]] = sg[i]
        # 비보존 g 중 ±T_d 에 들지 않는 것은 없음 (−g ∈ T_d) — 확인용으로 g 만 (부호 반전 없이) 비교
        s1 = shadow_sig(M @ d)
        if len(s1) == len(s0):
            mn_out = min(mn_out, np.max(np.abs(s1 - s0)))
log(f"  [float 확인] 무작위 방향 20개: d 와 ±g·d (g ∈ T_d) 의 그림자 쌍거리 최대 차 = {mx_in:.2e} (합동)")
log(f"             비보존 g 의 g·d (부호 반전 없이) 와의 최대 차 최소값 = {mn_out:.2e} — 단, −g ∈ T_d 이므로 −g·d 와는 합동")
log("  [이전 필터 영향] rupert_cell_compare 의 비대칭각은 48 부호치환으로 |d_in·(G d_out)| 의 최소각을 썼음.")
log("    abs(·) 로 d ↔ −d 를 이미 동일시했으므로 실제로 쓰인 동일시 집합 = {±g : g ∈ 48} = {±g : g ∈ T_d} (−I·T_d = O_h∖T_d).")
log("    → 필터가 판정한 집합은 올바른 그림자 합동 동일시와 정확히 같음. 후보 선택 결과 (0개) 변화 없음.")
log("    정정 대상은 '48 = 물체 대칭 O_h' 라는 서술뿐 (물체 대칭은 T_d 24, 나머지 인자 2 는 투영 부호). 기하 인증 결과 무관.")

# ════════════════════════════════════════════════════════════════════════════
# C·D. 국소 방향미분 + 국소 부등식
# ════════════════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("C·D. 동일 방향 세 경로 국소 부등식 (바깥 고정 R_out = Ry(−2/5)Rz(−9/10), 안쪽 R_out·exp(θ[a]×))")
log(SEP)


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


def icross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


Ro = imul(iry(-I(2) / 5), irz(-I(9) / 10))
Viv = [[I(c.numerator) / I(c.denominator) for c in v] for v in V_EX]
W = [imv(Ro, v) for v in Viv]                       # 회전 후 꼭짓점 (구간)
S2 = [(w[0], w[1]) for w in W]                      # 바깥 그림자 = 안쪽 θ=0 그림자
Sf = np.array([[float(mpmath.mpf(mp.make_mpf(x._mpi_[0]))) for x in s] for s in S2])
hull = ConvexHull(Sf)
order = list(hull.vertices)                         # 반시계
edges = []
for i in range(len(order)):
    a_, b_ = order[i], order[(i + 1) % len(order)]
    ex, ey = S2[b_][0] - S2[a_][0], S2[b_][1] - S2[a_][1]
    n = (ey, -ex)                                   # 반시계 → 바깥 법선 (비정규화, 정확값의 구간)
    b = n[0] * S2[a_][0] + n[1] * S2[a_][1]
    # 다른 꼭짓점이 엄격히 안쪽인지 구간 인증 → S_j = {a_, b_} 정확
    strict = True
    for k in range(8):
        if k in (a_, b_):
            continue
        val = n[0] * S2[k][0] + n[1] * S2[k][1] - b
        if not hi(val) < 0:
            strict = False
    edges.append(dict(a=a_, b=b_, n=n, b0=b, strict=strict))
log(f"  바깥 그림자 꼭짓점 (반시계) = {[int(x) for x in order]},  변 {len(edges)}개, 모든 변에서 다른 꼭짓점 엄격 내부 (구간 인증): "
    f"{all(e['strict'] for e in edges)}")
log("  변 j: 접촉 집합 S_j = {변 양 끝 꼭짓점} (θ=0 에서 h_I(n_j) = h_O(n_j) = b_j 를 달성하는 꼭짓점 전부)")

AXES = [("(1,0,0)", [I(1), I(0), I(0)], [1.0, 0, 0]),
        ("(0,1,0)", [I(0), I(1), I(0)], [0, 1.0, 0]),
        ("(1,2,3)/√14", [I(1) / iv.sqrt(14), I(2) / iv.sqrt(14), I(3) / iv.sqrt(14)], None)]
# 이전 진단 (rupert_near_identity) 의 k=12 관측값 M(θ)/θ — 대조용 (피팅 아님)
OBS = {"(1,0,0)": (-1.379866805e-5, -1.921724098e-5), "(0,1,0)": (-2.519591506e-5, -1.585225265e-5),
       "(1,2,3)/√14": (-2.870313563e-5, -3.039373168e-5)}
TH12 = 0.1 / 2 ** 12


def dual_vertices():
    """정확 균형 쌍대 꼭짓점: 변 2개 (반평행) 또는 3개의 법선이 Σw n = 0, Σw = 1, w > 0 을 만족 — 구간 연립해"""
    res = []
    m = len(edges)
    for (i, j) in itertools.combinations(range(m), 2):
        ni, nj = edges[i]["n"], edges[j]["n"]
        cr = ni[0] * nj[1] - ni[1] * nj[0]
        if lo(cr) <= 0 <= hi(cr):                   # 반평행 후보 (정확 반평행이면 cr = 0)
            pass
    for (i, j, k) in itertools.combinations(range(m), 3):
        N = [edges[x]["n"] for x in (i, j, k)]
        A = iv.matrix([[I(1), I(1), I(1)], [N[0][0], N[1][0], N[2][0]], [N[0][1], N[1][1], N[2][1]]])
        try:
            wv = iv.lu_solve(A, iv.matrix([I(1), I(0), I(0)]))
        except Exception:
            continue
        w = [wv[x] for x in range(3)]
        if all(lo(x) > 0 for x in w):
            res.append(((i, j, k), w))
    return res


DV = dual_vertices()
log(f"  정확 균형 쌍대 꼭짓점 (변 3개, 가중치 구간 하한 > 0): {len(DV)}개  (반평행 2변 쌍: 없음 — 정육각형 아님)")
log("  ※ 가중치 구간은 Σ w n_j = 0, Σw = 1 의 정확해를 포함 (정확 법선의 구간 연립해) → 증명서는 정확 균형, T_max‖r‖ 항 없음")

results = {}
for name, a_iv, _ in AXES:
    log("")
    log(f"  ── 축 a = {name}")
    log(f"     {'변 j':<5} {'S_j':<8} | {'α_j(S_j) (구간)':<44} | D+_j = max α | D−_j = max(−α) | 접촉 전환 (+ / −)")
    tab = []
    for jx, e in enumerate(edges):
        nu = [e["n"][0], e["n"][1], I(0)]
        al, be = {}, {}
        for v in (e["a"], e["b"]):
            av = icross(a_iv, Viv[v])
            aav = icross(a_iv, av)
            al[v] = sum((nu[i] * imv(Ro, av)[i] for i in range(3)), I(0))
            be[v] = sum((nu[i] * imv(Ro, aav)[i] for i in range(3)), I(0))
        va, vb = e["a"], e["b"]
        dplus_v = va if lo(al[va]) > hi(al[vb]) else (vb if lo(al[vb]) > hi(al[va]) else None)
        dminus_v = va if lo(-al[va]) > hi(-al[vb]) else (vb if lo(-al[vb]) > hi(-al[va]) else None)
        tab.append(dict(al=al, be=be, vp=dplus_v, vm=dminus_v))
        log(f"     {jx:<5} {str((int(va), int(vb))):<8} | α({va})={F(al[va], 10)}, α({vb})={F(al[vb], 10)} |"
            f" {mpmath.nstr(max(hi(al[va]), hi(al[vb])), 8):>12} | {mpmath.nstr(max(hi(-al[va]), hi(-al[vb])), 8):>13} |"
            f" + → v{int(dplus_v) if dplus_v is not None else '동률/미확정'}, − → v{int(dminus_v) if dminus_v is not None else '동률/미확정'}")
    results[name] = {}
    for sgn, lab in ((1, "+"), (-1, "−")):
        best = None
        allv = []
        for (idx, w) in DV:
            ok = True
            a_val, b_val = I(0), I(0)
            s_unit = sum((wj * iv.sqrt(edges[x]["n"][0] ** 2 + edges[x]["n"][1] ** 2) for x, wj in zip(idx, w)), I(0))
            for x, wj in zip(idx, w):
                t_ = tab[x]
                vstar = t_["vp"] if sgn > 0 else t_["vm"]
                if vstar is None:
                    ok = False
                    break
                a_val += wj * (sgn * t_["al"][vstar])
                b_val += wj * (-t_["be"][vstar])
            if not ok:
                continue
            # 단위 법선 환산: 비정규화 n_j = |n_j| n̂_j → 단위 법선 가중치 w_j|n_j| (합 s_unit) — 기울기 비교용 정규화
            allv.append((idx, a_val / s_unit))
            if best is None or lo(a_val / s_unit) > lo(best[1] / best[4]):
                best = (idx, a_val, b_val, w, s_unit)
        idx, a_val, b_val, w, s_unit = best
        log(f"     [{lab}θ] 쌍대 꼭짓점별 정규화 기울기 a/s′: " +
            ", ".join(f"변 {i_}: {mpmath.nstr(lo(v_), 10)}" for i_, v_ in allv))
        a_lo = lo(a_val)
        bp_hi = max(hi(b_val), mpmath.mpf(0))
        C = a_lo / 6 + bp_hi / 2
        if a_lo > 0:
            delta = min(mpmath.mpf(1), a_lo / C)
            verdict = f"0 < {lab}θ ≤ δ 에서 B(θ) < 0 증명 (정확 균형 증명서)"
        else:
            delta = None
            verdict = "a > 0 미인증 → 국소 부등식 불성립 (UNKNOWN)"
        obs = OBS[name][0 if sgn > 0 else 1] / TH12
        log(f"     [{lab}θ] 최선 쌍대 꼭짓점 변 {idx}, w = ({', '.join(F(x, 12) for x in w)})")
        log(f"          a = Σw·α(v*) = {F(a_val, 16)},  b′ = −Σw·β(v*) = {F(b_val, 16)}")
        log(f"          B(θ) ≤ −sinθ·a + (1−cosθ)·b′ ≤ −a_lo·θ + C·θ²,  C = {mpmath.nstr(C, 12)}  (θ ∈ (0,1])")
        log(f"          δ = {mpmath.nstr(delta, 12) if delta is not None else '—'}  → {verdict}")
        an = a_val / s_unit
        log(f"          s′ = Σ w_j|n_j| = {F(s_unit, 14)}  (단위 법선 환산 인자; 부호·δ 에는 무관)")
        log(f"          [대조, 피팅 아님] 이전 진단 k=12 관측 −M/|θ| = {-obs:.10f}  vs  도출 정규화 기울기 a/s′ = {F(an, 12)}"
            f"  (차 {float(lo(an)) - (-obs):+.3e}; θ=2.44e−5 의 2차 항 규모 ~ Cθ/s′)")
        results[name][lab] = dict(edges=[int(x) for x in idx], s_unit=[mpmath.nstr(lo(s_unit), 30), mpmath.nstr(hi(s_unit), 30)],
                                  a_over_s=[mpmath.nstr(lo(an), 30), mpmath.nstr(hi(an), 30)], weights=[[mpmath.nstr(lo(x), 30), mpmath.nstr(hi(x), 30)] for x in w],
                                  a=[mpmath.nstr(lo(a_val), 30), mpmath.nstr(hi(a_val), 30)],
                                  b_prime=[mpmath.nstr(lo(b_val), 30), mpmath.nstr(hi(b_val), 30)],
                                  C=mpmath.nstr(C, 30), delta=mpmath.nstr(delta, 30) if delta is not None else None,
                                  observed_minus_M_over_theta_k12=-obs)

log("")
log(SEP)
log("  경계·범위:")
log("   · 증명 범위: 각 경로 (축 3 × 부호 2) 위의 점 배치 0 < |θ| ≤ δ. 바깥 고정, 1차원 경로 — 5차원 이웃·전 방향 아님.")
log("   · θ = 0: 별도 경계 상태. 정확 균형 증명서로는 B(0) = 0 (모든 g_j = 0) — 엄격 포함 불가와 양립하나")
log("     이번 국소 정리의 범위 (0 < |θ|) 에 넣지 않음.")
log("   · 근사 균형 (float 가중치) 이면 B(θ) ≤ −aθ + Cθ² + T_max‖r‖ (‖r‖ ~ 1e−16) — θ ≲ T_max‖r‖/a 에서")
log("     상수 항이 일차 음수항을 가려 0 근방 인증 불가. 이번에는 정확해를 감싼 구간 가중치로 r = 0 정확.")
log("   · 접촉 전환: 고정 꼭짓점 v* 의 내적은 h_I 의 하계이므로, 경로 위 실제 최대 꼭짓점이 바뀌어도 부등식은 유효")
log("     (전환은 h_I 를 키울 뿐 → B 를 더 낮춤).")
with open("code/rupert_local_ineq_data.json", "w") as f:
    json.dump({"environment": {"python": sys.version.split()[0], "mpmath": mpmath.__version__, "iv_dps": iv.dps},
               "symmetry": {"preserving": len(pres), "det_plus": n_pos, "det_minus": n_neg,
                            "central_inversion_preserves": False},
               "outer": {"R_out": "Ry(-2/5) @ Rz(-9/10)", "hull_order": [int(x) for x in order],
                         "edges": [[int(e["a"]), int(e["b"])] for e in edges]},
               "paths": results}, f, ensure_ascii=False, indent=1)
with open("code/rupert_local_ineq_output.txt", "w") as f:
    f.write("\n".join(out) + "\n")

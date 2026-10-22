#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구간 노름 견고성 점검 + 동일 방향 근방 추적 (진단 시험; 전체 공간 증명·EQI 격자 규칙 도출 아님)
원본 엔진·legacy Rupert 코드·보존 파일 무수정. EQI 상수 미사용.
좌표 구분: t_shape = P_t 형상 매개변수 (여기서 11/20). θ = 회전각 (진단용). t_EQI·T_model 과 무관.
          동일 방향 배치의 M = 0 을 t_EQI = 0 이나 계산복잡도 명제와 동일시하지 않음.

A. 구간 노름: 보존본 rupert_cell_cert.inorm (x*x) vs 견고판 rupert_iv_robust.inorm (x**2)
B. 동일 방향 근방 진단 프로토콜 (사전 고정):
   물체 P_{11/20} (정확), 바깥 회전 R_out = Ry(−0.4)·Rz(−0.9) (기존 P2 중심의 바깥 회전),
   안쪽 회전 R_in(θ) = R_out · exp(θ·[a]×),  축 a ∈ {(1,0,0), (0,1,0), (1,2,3)/√14},
   θ = ±(1/10)·2^(−k), k = 0..12, θ = 0 은 기준점 (별도). 총 3·(26+1) = 81 점 배치.
C. 점 배치 진단: float LP M (비인증), 고정 방향 점 증명서 B (구간), 활성 방향·가중치·접촉 꼭짓점,
   잔류 균형 보정량, 실제 상대 회전각.
D. 작은 이웃 (사전 순서: 축 순서대로 축마다 첫 인증 사례 1개, k 오름차순, + 먼저 → 최대 3개):
   매개 (θ₁, φ₁, θ) 반폭 h = |θ₀|/8 (진단 입력),  R_out = Ry(−θ₁)Rz(−φ₁),  R_in = R_out·exp(θ[a]×)
   볼: ρ_out = h_θ₁ + h_φ₁,  ρ_in = h_θ₁ + h_φ₁ + h_θ  /  인수: v → exp(θ[a]×) → Rz(−φ₁) → Ry(−θ₁)  /  결합.
실행: PYTHONPATH=code python3 code/rupert_near_identity.py
"""

import contextlib
import io
import itertools
import json
import sys
import time

import numpy as np
import mpmath
from mpmath import iv, mp
import scipy
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

import rupert_iv_robust as R
import rupert_cell_cert as OLD                      # 보존본 (비교용, 수정 없음)

iv.dps = 40
mp.dps = 40
I = R.I
SEP = "=" * 110
out_lines = []


def log(s=""):
    print(s, flush=True)
    out_lines.append(s)


def F(x, k=14):
    if hasattr(x, "_mpi_"):
        lo, hi = mp.make_mpf(x._mpi_[0]), mp.make_mpf(x._mpi_[1])
        return mpmath.nstr(lo, k) if lo == hi else f"[{mpmath.nstr(lo, k)}, {mpmath.nstr(hi, k)}]"
    return mpmath.nstr(x, k)


def hi(x):
    return mp.make_mpf(x._mpi_[1])


def lo(x):
    return mp.make_mpf(x._mpi_[0])


# ════════════════════════════════════════════════════════════════════════════
# A. 구간 노름 점검
# ════════════════════════════════════════════════════════════════════════════
def try_eval(f, *a):
    try:
        return f(*a), None
    except Exception as e:                       # 예외는 기록만 (0·성공으로 바꾸지 않음)
        return None, f"{type(e).__name__}: {e}"


def part_A():
    log(SEP)
    log("A. 구간 노름 점검 — 보존본 inorm (x*x) vs 견고판 inorm (x**2)")
    log(SEP)
    z = [I(0), I(0), I(0)]
    small = [iv.mpf(["-1e-3", "2e-3"]), iv.mpf(["-3e-4", "1e-4"]), iv.mpf(["-1e-5", "1e-5"])]
    for lab, v in (("영벡터", z), ("각 성분이 0 포함하는 작은 구간 벡터", small)):
        o, eo = try_eval(OLD.inorm, v)
        n, en = try_eval(R.inorm, v)
        log(f"  [{lab}] 보존본: {F(o) if o is not None else '예외 ' + eo}   견고판: {F(n) if n is not None else '예외 ' + en}")
        sq_old = sum((x * x for x in v), I(0))
        sq_new = sum((x ** 2 for x in v), I(0))
        log(f"      Σx*x = {F(sq_old)}   Σx**2 = {F(sq_new)}")
    n3 = [I(1), I(0), I(0)]
    cases = [("정확 평행 n ∥ v", n3, [I(2), I(0), I(0)]),
             ("거의 평행 (1e-20 기울기)", n3, [I(2), I("1e-20"), I(0)]),
             ("반평행 n ∥ −v", n3, [I(-2), I(0), I(0)]),
             ("0 포함 구간 v", n3, small)]
    for lab, n, v in cases:
        for rho_lab, rho in (("ρ=0", I(0)), ("ρ=π/7", iv.pi / 7), ("ρ=π", iv.pi)):
            uo, eu_o = try_eval(OLD.U_iv, n, v, rho)
            lo_, el_o = try_eval(OLD.L_iv, n, v, rho)
            un, eu = try_eval(R.U_iv, n, v, rho)
            ln, el = try_eval(R.L_iv, n, v, rho)
            log(f"  [{lab}, {rho_lab}] 보존본 U={F(uo, 10) if uo is not None else '예외(' + eu_o.split(':')[0] + ')'}"
                f" L={F(lo_, 10) if lo_ is not None else '예외(' + el_o.split(':')[0] + ')'}"
                f"  | 견고판 U={F(un, 10) if un is not None else '예외 ' + eu} L={F(ln, 10) if ln is not None else '예외 ' + el}")


def regress_A():
    """A/B/C/C3D/P1/P2: 보존본과 견고판으로 같은 입력의 B 계산·비교"""
    log("")
    log("  회귀시험 A/B/C/C3D/P1/P2 — 같은 입력, 보존본 vs 견고판 B 상한")
    cube = [[I(c) for c in v] for v in itertools.product((-1, 1), repeat=3)]
    s2, s6 = iv.sqrt(2), iv.sqrt(6)
    e1 = [I(1) / s2, -I(1) / s2, I(0)]
    e2 = [I(1) / s6, I(1) / s6, -I(2) / s6]
    Rhex = [e1, e2, R.icross(e1, e2)]
    Id = [[I(1), I(0), I(0)], [I(0), I(1), I(0)], [I(0), I(0), I(1)]]
    lam0 = 2 * s2 / (1 + iv.sqrt(3))
    d60 = [(0.5, float(mpmath.sqrt(3) / 2)), (-0.5, -float(mpmath.sqrt(3) / 2))]
    sq = [[I(1), I(1), I(0)], [I(1), I(-1), I(0)], [I(-1), I(1), I(0)], [I(-1), I(-1), I(0)]]
    P0 = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    Pt = [[I(c) for c in p] for p in P0] + [[-(I(11) / I(20)) * I(c) for c in p] for p in P0]
    Pt_f = np.array([[float(c) for c in p] for p in P0] + [[-0.55 * c for c in p] for p in P0])

    def rot_R8(xs):
        al, ph2, th2, th1, ph1 = [I(v) for v in xs]
        return R.imul(R.iry(-th1), R.irz(-ph1)), R.imul(R.imul(R.irz(al), R.iry(-th2)), R.irz(-ph2))

    tests = [("A", cube, cube, Rhex, Id, I(0), I(0), d60, [0.5, 0.5], (1 + lam0) / 2),
             ("B", cube, cube, Rhex, Id, I(0), I(0), d60, [0.5, 0.5], lam0 * (1 + I("1e-12"))),
             ("C", cube, sq, Rhex, Id, I(0), iv.pi / 360, d60, [0.5, 0.5], I(26) / I(25)),
             ("C3D", cube, cube, Rhex, Id, I(0), iv.pi / 360, d60, [0.5, 0.5], I(26) / I(25))]
    for lab, cen in (("P1", ["0.7", "2.1", "1.3", "0.4", "0.9"]), ("P2", ["0", "0.9", "0.4", "0.4", "0.9"])):
        Ro_f, Ri_f = OLD.float_rot_R8(np.array([float(c) for c in cen]))
        dn, wn, _ = OLD.center_choice(Pt_f, Pt_f, Ro_f, Ri_f)
        Ro, Ri = rot_R8(cen)
        tests.append((lab, Pt, Pt, Ro, Ri, I("0.002"), I("0.003"), [tuple(map(float, d)) for d in dn],
                      [float(x) for x in wn], I(1)))
    for lab, Vo, Vi, Ro, Ri, rho_o, rho_i, dirs, w, lam in tests:
        Vi_s = [[lam * c for c in v] for v in Vi]
        res = {}
        for name, mod in (("보존본", OLD), ("견고판", R)):
            try:
                Wo = [R.imv(Ro, v) for v in Vo]
                Wi = [R.imv(Ri, v) for v in Vi_s]
                up, lw = [], []
                for nx, ny in dirs:
                    n3 = [I(nx), I(ny), I(0)]
                    up.append(max(hi(mod.U_iv(n3, w0, rho_o)) for w0 in Wo))
                    lw.append(max(lo(mod.L_iv(n3, w0, rho_i)) for w0 in Wi))
                T = I(max(hi(R.inorm(v)) for v in Vo)) + I(max(hi(R.inorm(v)) for v in Vi_s))
                B = R.B_from(up, lw, dirs, w, T)[0]
                res[name] = B
            except Exception as e:
                res[name] = f"예외 {type(e).__name__}"
        bo, bn = res["보존본"], res["견고판"]
        same = (not isinstance(bo, str) and not isinstance(bn, str) and lo(bo) == lo(bn) and hi(bo) == hi(bn))
        log(f"    [{lab:<3}] 보존본 B = {F(bo) if not isinstance(bo, str) else bo:<36} 견고판 B = "
            f"{F(bn) if not isinstance(bn, str) else bn:<36} {'동일 (끝점 일치)' if same else '차이'}"
            f"  → {'인증' if not isinstance(bn, str) and hi(bn) <= 0 else 'UNKNOWN'}")


# ════════════════════════════════════════════════════════════════════════════
# B·C. 동일 방향 근방 점 배치 진단
# ════════════════════════════════════════════════════════════════════════════
P0 = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
PT = [[I(c) for c in p] for p in P0] + [[-(I(11) / I(20)) * I(c) for c in p] for p in P0]
PT_F = np.array([[float(c) for c in p] for p in P0] + [[-0.55 * c for c in p] for p in P0])
TH1, PH1 = "0.4", "0.9"                             # 기존 P2 중심의 바깥 각 (정확 십진)
AXES = [("(1,0,0)", [I(1), I(0), I(0)], np.array([1.0, 0, 0])),
        ("(0,1,0)", [I(0), I(1), I(0)], np.array([0, 1.0, 0])),
        ("(1,2,3)/√14", [I(1) / iv.sqrt(14), I(2) / iv.sqrt(14), I(3) / iv.sqrt(14)],
         np.array([1.0, 2.0, 3.0]) / np.sqrt(14))]
T_MAX = I(2) * iv.sqrt(3)                           # R_O + R_I = √3 + √3 (원점 중심 3D, 투영 전)


def rz_f(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def ry_f(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def exp_f(theta, k):
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def point_diag(theta_str, ax):
    th = float(mpmath.mpf(theta_str))
    Ro_f = ry_f(-0.4) @ rz_f(-0.9)
    Ri_f = Ro_f @ exp_f(th, ax[2])
    A = (Ro_f @ PT_F.T).T[:, :2]
    Bv = (Ri_f @ PT_F.T).T[:, :2]
    hull = ConvexHull(A)
    eq = hull.equations
    n, b = eq[:, :2], -eq[:, 2]
    hI = Bv @ n.T
    g = b - hI.max(axis=0)
    rP = linprog([-1, 0, 0], A_ub=np.column_stack([np.ones(len(n)), n]), b_ub=g,
                 bounds=[(None, None)] * 3, method="highs")
    M = -rP.fun
    rD = linprog(g, A_eq=np.vstack([np.ones(len(n)), n.T]), b_eq=[1, 0, 0],
                 bounds=[(0, None)] * len(n), method="highs-ds")
    act = np.where(rD.x > 1e-14)[0]
    dirs = [tuple(map(float, n[j])) for j in act]
    w = [float(rD.x[j]) for j in act]
    contacts = []
    for j in act:
        ov = tuple(sorted(int(i) for i in hull.simplices[j]))
        mx = hI[:, j].max()
        iv_ = tuple(int(i) for i in np.where(hI[:, j] >= mx - 1e-12)[0])
        contacts.append((ov, iv_))
    # 구간 점 증명서
    Ro = R.imul(R.iry(-I(TH1)), R.irz(-I(PH1)))
    Ri = R.imul(Ro, R.iexp_axis(I(theta_str) if theta_str != "0" else I(0), ax[1]))
    Wo = [R.imv(Ro, v) for v in PT]
    Wi = [R.imv(Ri, v) for v in PT]
    up, lw = [], []
    for nx, ny in dirs:
        n3 = [I(nx), I(ny), I(0)]
        up.append(max(hi(R.idot(n3, x)) for x in Wo))
        lw.append(max(lo(R.idot(n3, x)) for x in Wi))
    B, s, r, rn, corr, exact = R.B_from(up, lw, dirs, w, T_MAX)
    ang = float(np.arccos(np.clip((np.trace(Ro_f.T @ Ri_f) - 1) / 2, -1, 1)))
    return dict(theta=theta_str, th=th, M=M, B=B, dirs=dirs, w=w, contacts=contacts, corr=corr, rn=rn,
                exact=exact, ang=ang, up=up, lw=lw)


def part_BC():
    log("")
    log(SEP)
    log("B·C. 동일 방향 근방 점 배치 진단 (P_{11/20}, R_out = Ry(−0.4)Rz(−0.9), R_in = R_out·exp(θ[a]×))")
    log("     ※ '동일 방향 근방 진단' 별도 시험 — 기존 '비대칭각 > 0.05' 시험을 대체·성공 처리하지 않음")
    log(SEP)
    allres = {}
    for ax in AXES:
        log(f"  축 a = {ax[0]}")
        log(f"   {'θ':>14} | {'float M (비인증)':>16} | {'점 B 상한 (구간)':>18} | {'B 폭':>9} | {'T‖r‖ 보정':>9} |"
            f" {'실제 회전각':>11} | 활성 방향 수 | 판정 | 접촉 (바깥 변 꼭짓점 / 안쪽 꼭짓점)")
        rows = []
        thetas = [("0", 0)] + [(f"{sg}{mpmath.nstr(mpmath.mpf(1) / 10 / 2**k, 30)}", k)
                                for k in range(13) for sg in ("", "-")]
        for ts, k in thetas:
            d = point_diag(ts, ax)
            d["k"] = k
            st = "인증" if hi(d["B"]) <= 0 else "UNKNOWN"
            d["status"] = st
            rows.append(d)
            log(f"   {d['th']: .6e} | {d['M']: .6e} | {mpmath.nstr(hi(d['B']), 10):>18} | {F(d['B'].delta, 2):>9} |"
                f" {mpmath.nstr(hi(d['corr']), 3):>9} | {d['ang']:.4e} | {len(d['dirs']):>2} | {st:<7} |"
                f" {'; '.join(f'{c[0]}/{c[1]}' for c in d['contacts'])}")
        # 접촉 변화 위치
        changes = []
        for sg in (1, -1):
            seq = [d for d in rows if d["k"] != 0 or d["theta"] != "0"]
            seq = [d for d in rows if d["theta"] != "0" and (d["th"] > 0) == (sg > 0)]
            seq.sort(key=lambda d: d["k"])
            for a_, b_ in zip(seq, seq[1:]):
                if sorted(a_["contacts"]) != sorted(b_["contacts"]):
                    changes.append(f"{'+' if sg > 0 else '−'}: k={a_['k']}→{b_['k']}")
        log(f"   활성 접촉 변화 위치: {', '.join(changes) if changes else '없음'}")
        # 감소율 관측
        for sg in (1, -1):
            seq = sorted([d for d in rows if d["theta"] != "0" and (d["th"] > 0) == (sg > 0)], key=lambda d: d["k"])
            ratios = [seq[i]["M"] / seq[i + 1]["M"] for i in range(len(seq) - 1) if seq[i + 1]["M"] != 0]
            log(f"   M(k)/M(k+1) 관측 ({'+' if sg > 0 else '−'}θ): {', '.join(f'{x:.3f}' for x in ratios)}"
                f"  (2 ≈ 선형, 4 ≈ 이차 — 관측일 뿐 정리 아님)")
        allres[ax[0]] = rows
        log("")
    return allres


# ════════════════════════════════════════════════════════════════════════════
# D. 작은 이웃 시험
# ════════════════════════════════════════════════════════════════════════════
def part_D(allres):
    log(SEP)
    log("D. 작은 이웃 시험 (사전 순서: 축마다 첫 인증 사례, k 오름차순, + 먼저; 최대 3개)")
    log("   매개 (θ₁, φ₁, θ), 반폭 h = |θ₀|/8 (진단 입력),  R_out = Ry(−θ₁)Rz(−φ₁),  R_in = R_out·exp(θ[a]×)")
    log("   볼: ρ_out = 2h, ρ_in = 3h  (양측 불변 거리 삼각부등식)  /  인수: v → exp(θ[a]×)v → Rz(−φ₁) → Ry(−θ₁)")
    log(SEP)
    chosen = []
    for ax in AXES:
        rows = allres[ax[0]]
        cand = sorted([d for d in rows if d["theta"] != "0" and d["status"] == "인증"],
                      key=lambda d: (d["k"], 0 if d["th"] > 0 else 1))
        if cand:
            chosen.append((ax, cand[0]))
    nb_records = []
    for ax, d in chosen:
        th0 = I(d["theta"])
        h = abs(th0) / 8
        contains0 = bool(lo(abs(th0) - h) <= 0)
        th_int = iv.mpf([lo(th0 - h), hi(th0 + h)])
        t1_int = iv.mpf([lo(I(TH1) - h), hi(I(TH1) + h)])
        p1_int = iv.mpf([lo(I(PH1) - h), hi(I(PH1) + h)])
        Ro0 = R.imul(R.iry(-I(TH1)), R.irz(-I(PH1)))
        Ri0 = R.imul(Ro0, R.iexp_axis(th0, ax[1]))
        rho_o, rho_i = 2 * h, 3 * h
        dirs, w = d["dirs"], d["w"]
        t0 = time.time()
        Wo0 = [R.imv(Ro0, v) for v in PT]
        Wi0 = [R.imv(Ri0, v) for v in PT]
        ub, lb = [], []
        for nx, ny in dirs:
            n3 = [I(nx), I(ny), I(0)]
            ub.append(max(hi(R.U_iv(n3, w0, rho_o)) for w0 in Wo0))
            lb.append(max(lo(R.L_iv(n3, w0, rho_i)) for w0 in Wi0))
        tb = time.time() - t0
        t0 = time.time()
        Wo = [R.apply_ry(-t1_int, R.apply_rz(-p1_int, v)) for v in PT]
        Wi = [R.apply_ry(-t1_int, R.apply_rz(-p1_int, R.apply_exp_axis(th_int, ax[1], v))) for v in PT]
        uf, lf = [], []
        for nx, ny in dirs:
            n3 = [I(nx), I(ny), I(0)]
            uf.append(max(hi(R.idot(n3, x)) for x in Wo))
            lf.append(max(lo(R.idot(n3, x)) for x in Wi))
        tf = time.time() - t0
        uc = [min(a, b) for a, b in zip(ub, uf)]
        lc = [max(a, b) for a, b in zip(lb, lf)]
        # 비인증 sanity: 이웃 내 무작위 배치의 실제 M 최대
        rng = np.random.default_rng(11)
        Mmax = -np.inf
        hf = float(hi(h))
        for _ in range(500):
            t1 = 0.4 + rng.uniform(-1, 1) * hf
            p1 = 0.9 + rng.uniform(-1, 1) * hf
            tt = d["th"] + rng.uniform(-1, 1) * hf
            Ro_f = ry_f(-t1) @ rz_f(-p1)
            Ri_f = Ro_f @ exp_f(tt, ax[2])
            A = (Ro_f @ PT_F.T).T[:, :2]
            Bv = (Ri_f @ PT_F.T).T[:, :2]
            eq = ConvexHull(A).equations
            n, b = eq[:, :2], -eq[:, 2]
            g = b - (Bv @ n.T).max(axis=0)
            rr = linprog([-1, 0, 0], A_ub=np.column_stack([np.ones(len(n)), n]), b_ub=g,
                         bounds=[(None, None)] * 3, method="highs")
            Mmax = max(Mmax, -rr.fun)
        log(f"  축 {ax[0]}, θ₀ = {d['th']:.6e} (k={d['k']}), h = {mpmath.nstr(hi(h), 6)}, θ=0 포함: {contains0},"
            f" 점 B = {mpmath.nstr(hi(d['B']), 8)}, 점 M = {d['M']:.4e}")
        rec = {"axis": ax[0], "theta0": d["theta"], "k": d["k"], "h": mpmath.nstr(hi(h), 20), "contains_theta0": contains0,
               "point_B_hi": mpmath.nstr(hi(d["B"]), 25), "point_M_float": d["M"], "sample_M_max_float": Mmax}
        for name, (u, l, tt) in (("볼", (ub, lb, tb)), ("인수", (uf, lf, tf)), ("결합", (uc, lc, tb + tf))):
            B, s, r, rn, corr, exact = R.B_from(u, l, dirs, w, T_MAX)
            st = "인증" if hi(B) <= 0 else "UNKNOWN"
            log(f"     {name:<4} B 상한 = {mpmath.nstr(hi(B), 12):<22} 폭 = {F(B.delta, 3):<10} → {st:<8} ({tt * 1e3:.1f} ms)")
            rec[name] = {"B_hi": mpmath.nstr(hi(B), 25), "B_lo": mpmath.nstr(lo(B), 25), "status": st, "time_ms": tt * 1e3}
        worse_up = sum(1 for a, b in zip(ub, uf) if b > a)
        worse_lo = sum(1 for a, b in zip(lb, lf) if b < a)
        best = min(hi(R.B_from(u, l, dirs, w, T_MAX)[0]) for u, l in ((ub, lb), (uf, lf), (uc, lc)))
        cause = ("여유 크기 (표본 M 최대가 0 에 가까움)" if Mmax > -abs(float(best)) * 0.1 and best > 0 else
                 ("상계 보수성 (표본 M 최대 ≪ 0 인데 B > 0)" if best > 0 else "해당 없음 (인증)"))
        log(f"     방향별: 인수 outer upper > 볼 {worse_up}/{len(dirs)}, 인수 inner lower < 볼 {worse_lo}/{len(dirs)}")
        log(f"     [비인증 sanity 500 배치] 이웃 내 실제 M 최대 = {Mmax:.4e};  한계 분류: {cause}")
        rec["limit_cause"] = cause
        nb_records.append(rec)
    if not chosen:
        log("  점 상계가 음수로 인증된 사례 없음 → 이웃 시험 0개")
    return nb_records


if __name__ == "__main__":
    T0 = time.time()
    log(f"환경: python {sys.version.split()[0]} | mpmath {mpmath.__version__} (iv.dps={iv.dps}) | numpy {np.__version__}"
        f" | scipy {scipy.__version__}")
    part_A()
    regress_A()
    allres = part_BC()
    nb = part_D(allres)
    log(SEP)
    log("C3D 대상 확인: 이전 비교 (rupert_cell_compare) 의 C3D 는 M1 = 정육면체·λ=26/25·안쪽 SO(3) 볼 (중심 항등, ρ=π/360),")
    log("  M2 = 같은 물체·λ 에 안쪽 Rz(θ), |θ| ≤ π/360 (1차원) — 같은 물체·같은 λ 이지만 회전 영역이 다름 (볼 ⊇ 평면 회전).")
    log("  같은 Euler 칸 비교가 아니었음. (이전 C 는 평면 정사각형 그림자 = 다른 물체 정의.)")
    log(f"총 소요 {time.time() - T0:.1f}s")
    dump = {"environment": {"python": sys.version.split()[0], "mpmath": mpmath.__version__, "iv_dps": iv.dps,
                            "numpy": np.__version__, "scipy": scipy.__version__},
            "protocol": {"object": "P_t, t_shape = 11/20 (exact)", "R_out": "Ry(-0.4) @ Rz(-0.9)",
                         "R_in": "R_out @ exp(theta [a]x)", "axes": [a[0] for a in AXES],
                         "theta": "±(1/10)·2^(-k), k=0..12, plus 0", "T_max": "2√3 (R_O + R_I, 3D, 원점 중심)"},
            "points": {ax: [{"theta": d["theta"], "k": d["k"], "M_float": d["M"],
                             "B_hi": mpmath.nstr(hi(d["B"]), 25), "B_lo": mpmath.nstr(lo(d["B"]), 25),
                             "status": d["status"], "rot_angle": d["ang"],
                             "dirs": [[float(x).hex() for x in dd] for dd in d["dirs"]],
                             "weights": [float(x).hex() for x in d["w"]],
                             "contacts": [[list(c[0]), list(c[1])] for c in d["contacts"]],
                             "residual_corr_hi": mpmath.nstr(hi(d["corr"]), 6)} for d in rows]
                       for ax, rows in allres.items()},
            "neighborhoods": nb}
    with open("code/rupert_near_identity_data.json", "w") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    with open("code/rupert_near_identity_output.txt", "w") as f:
        f.write("\n".join(out_lines) + "\n")

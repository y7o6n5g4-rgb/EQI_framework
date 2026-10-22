#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경계 가까운 비대칭 칸의 상계 비교 — 회전 볼 U/L vs 회전 인수별 구간 vs 결합
(전체 탐색 아님. 원본 엔진·legacy Rupert 코드 무수정. EQI 상수 미사용.)

방법 (모두 같은 칸·같은 고정 검사 방향·같은 가중치에 적용):
  M1 (볼)  : SO(3) 측지 볼 U/L — 칸 중심 회전 R0, 상대 회전각 ρ = 반폭 합 (rupert_cell_cert 과 동일 구현)
  M2 (인수): 실제 인수 순서 보존 구간 계산 — 각도 구간 [c−h, c+h] 의 cos/sin 구간으로
             v → Rz(·)v → Ry(·)(…) → … 를 꼭짓점마다 순서대로 감싸고, n·(Rv) 구간을 직접 얻음
  M3 (결합): 방향별 outer upper = min(M1, M2), inner lower = max(M1, M2)
판정: B = [Σ w ḡ + T_max‖r‖]/Σw 의 상한 ≤ 0 → 엄격 포함 배제 인증, 그 외 UNKNOWN (양의 B 는 통과 인증 아님).

실행: python3 code/rupert_cell_compare.py
"""

import json
import sys
import time

import numpy as np
import mpmath
from mpmath import iv, mp
import scipy

import rupert_cell_cert as C                        # U_iv, L_iv, 구간 연산, 중심 선택, 인코딩 (수정 없음)

iv.dps = 40
mp.dps = 40
I = C.I
SEED = 20261022
N_CAND_MAX, T_CAND_MAX = 2000, 60.0
M_LO, M_HI = -1e-3, -1e-6                           # 후보 선택 구간 (시험용, EQI 임계치 아님)
ASYM_MIN = 0.05                                     # 비대칭 기준: 몸체 투영 방향 간 최소 각 (rad, 시험용)
LO = np.array([0, 0, 0, 0, 0], float)
HI = np.array([2 * np.pi, 2 * np.pi, np.pi, np.pi / 2, np.pi / 2])
WIDTH_FACTORS = (50, 10)                            # 반폭 h = |M_center|/factor (시험 입력, 격자 법칙 아님)
P0 = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
T_NUM, T_DEN = 11, 20
PT_F = np.array([[float(c) for c in p] for p in P0] + [[-T_NUM / T_DEN * c for c in p] for p in P0])
PT_IV = [[I(c) for c in p] for p in P0] + [[-(I(T_NUM) / I(T_DEN)) * I(c) for c in p] for p in P0]
SIGNED_PERMS = [np.array(P) * np.array(s)[:, None]
                for P in [np.eye(3)[list(p)] for p in __import__("itertools").permutations(range(3))]
                for s in __import__("itertools").product((1, -1), repeat=3)]


# ── 정확 저장용 직렬화 ────────────────────────────────────────────
def mpf_exact(x):
    s, man, e, bc = x._mpf_
    return {"man": str((-1) ** s * man), "exp2": e, "dec": mpmath.nstr(x, 30)}


def iv_exact(x):
    lo, hi = mp.make_mpf(x._mpi_[0]), mp.make_mpf(x._mpi_[1])
    return {"lo": mpf_exact(lo), "hi": mpf_exact(hi)}


def fhex(v):
    return {"hex": float(v).hex(), "repr": repr(float(v))}


# ── 회전 인수 ────────────────────────────────────────────────────
def apply_rz(a, v):
    c, s = iv.cos(a), iv.sin(a)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def apply_ry(a, v):
    c, s = iv.cos(a), iv.sin(a)
    return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]


def angle_iv(c, h):
    return iv.mpf([(I(c) - I(h)).a, (I(c) + I(h)).b])


def rot_out_factor(x_int, v):
    """R_out = Ry(−θ₁)·Rz(−φ₁):  v → Rz(−φ₁)v → Ry(−θ₁)(·)"""
    al, ph2, th2, th1, ph1 = x_int
    return apply_ry(-th1, apply_rz(-ph1, v))


def rot_in_factor(x_int, v):
    """R_in = Rz(α)·Ry(−θ₂)·Rz(−φ₂):  v → Rz(−φ₂) → Ry(−θ₂) → Rz(α)"""
    al, ph2, th2, th1, ph1 = x_int
    return apply_rz(al, apply_ry(-th2, apply_rz(-ph2, v)))


# ── 칸별 지지함수 경계 ───────────────────────────────────────────
def bounds_ball(cell, dirs):
    up, lo = [], []
    for nx, ny in dirs:
        n3 = [I(nx), I(ny), I(0)]
        up.append(max(C.U_iv(n3, w0, cell["rho_out"]).b for w0 in cell["Wo0"]))
        lo.append(max(C.L_iv(n3, w0, cell["rho_in"]).a for w0 in cell["Wi0"]))
    return up, lo


def bounds_factor(cell, dirs):
    up, lo = [], []
    Wo = [cell["fo"](v) for v in cell["Vo"]]
    Wi = [cell["fi"](v) for v in cell["Vi"]]
    for nx, ny in dirs:
        n3 = [I(nx), I(ny), I(0)]
        up.append(max(C.idot(n3, w).b for w in Wo))
        lo.append(max(C.idot(n3, w).a for w in Wi))
    return up, lo


def B_of(up, lo, dirs, w, T_max):
    wI = [I(x) for x in w]
    s = sum(wI, I(0))
    r = [sum((wI[j] * I(dirs[j][k]) for j in range(len(w))), I(0)) for k in (0, 1)]
    rn = iv.sqrt(r[0] ** 2 + r[1] ** 2)
    num = sum((wI[j] * (I(up[j]) - I(lo[j])) for j in range(len(w))), I(0))
    exact = all(c.a == 0 and c.b == 0 for c in r)
    B = (num if exact else num + T_max * rn) / s
    return B, s, r, exact


def run_methods(cell, dirs, w):
    out = {}
    t0 = time.time()
    ub, lb = bounds_ball(cell, dirs)
    tb = time.time() - t0
    t0 = time.time()
    uf, lf = bounds_factor(cell, dirs)
    tf = time.time() - t0
    uc = [min(a, b) for a, b in zip(ub, uf)]
    lc = [max(a, b) for a, b in zip(lb, lf)]
    for name, (u, l, tt) in (("M1_ball", (ub, lb, tb)), ("M2_factor", (uf, lf, tf)), ("M3_combined", (uc, lc, tb + tf))):
        t0 = time.time()
        B, s, r, exact = B_of(u, l, dirs, w, cell["T_max"])
        tt += time.time() - t0
        out[name] = dict(up=u, lo=l, B=B, s=s, r=r, exact=exact, time=tt,
                         status="인증" if B.b <= 0 else "UNKNOWN")
    return out


# ── 칸 구성 ──────────────────────────────────────────────────────
def make_Pt_cell(label, center, half):
    """center, half: float 5-벡터 (이진값 그대로 정확 입력)"""
    x0 = [I(c) for c in center]
    Ro0 = C.imul(C.iry(-x0[3]), C.irz(-x0[4]))
    Ri0 = C.imul(C.imul(C.irz(x0[0]), C.iry(-x0[1 + 1])), C.irz(-x0[1]))
    x_int = [angle_iv(c, h) for c, h in zip(center, half)]
    hI = [I(h) for h in half]
    rho_out = hI[3] + hI[4]
    rho_in = hI[0] + hI[1] + hI[2]
    R = max(C.inorm(v).b for v in PT_IV)
    return dict(label=label, kind="P_t", center=list(center), half=list(half),
                Vo=PT_IV, Vi=PT_IV, Wo0=[C.imv(Ro0, v) for v in PT_IV], Wi0=[C.imv(Ri0, v) for v in PT_IV],
                rho_out=rho_out, rho_in=rho_in, fo=lambda v: rot_out_factor(x_int, v),
                fi=lambda v: rot_in_factor(x_int, v), T_max=I(R) + I(R), R_O=R, R_I=R)


def make_C3D_cell():
    cube = [[I(c) for c in v] for v in __import__("itertools").product((-1, 1), repeat=3)]
    s2, s6 = iv.sqrt(2), iv.sqrt(6)
    e1 = [I(1) / s2, -I(1) / s2, I(0)]
    e2 = [I(1) / s6, I(1) / s6, -I(2) / s6]
    e3 = C.icross(e1, e2)
    R_hex = [e1, e2, e3]
    lam = I(26) / I(25)
    Vi = [[lam * c for c in v] for v in cube]
    th = iv.mpf([(-iv.pi / 360).a, (iv.pi / 360).b])
    R_O = max(C.inorm(v).b for v in cube)
    R_I = max(C.inorm(v).b for v in Vi)
    return dict(label="C3D", kind="cube", center=[0.0], half=None,
                Vo=cube, Vi=Vi, Wo0=[C.imv(R_hex, v) for v in cube], Wi0=Vi,
                rho_out=I(0), rho_in=iv.pi / 360,
                fo=lambda v: C.imv(R_hex, v), fi=lambda v: apply_rz(th, v),
                T_max=I(R_O) + I(R_I), R_O=R_O, R_I=R_I)


def body_dir(R):
    return R.T @ np.array([0.0, 0.0, 1.0])


def asym_angle(x):
    Ro, Ri = C.float_rot_R8(x)
    do, di = body_dir(Ro), body_dir(Ri)
    return min(np.arccos(np.clip(abs(di @ (G @ do)), -1, 1)) for G in SIGNED_PERMS)


def sample_cell(center, half, dirs, w, n=1000, seed=7):
    """비인증 sanity: 칸 내 무작위 배치의 실제 M 최대, Σw(h_O−h_I)/Σw 최대"""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    Mmax = Smax = -np.inf
    for _ in range(n):
        x = np.asarray(center) + rng.uniform(-1, 1, 5) * np.asarray(half)
        Ro, Ri = C.float_rot_R8(x)
        A = (Ro @ PT_F.T).T[:, :2]
        Bv = (Ri @ PT_F.T).T[:, :2]
        eq = ConvexHull(A).equations
        nn, b = eq[:, :2], -eq[:, 2]
        g = b - (Bv @ nn.T).max(axis=0)
        res = linprog([-1, 0, 0], A_ub=np.column_stack([np.ones(len(nn)), nn]), b_ub=g,
                      bounds=[(None, None)] * 3, method="highs")
        Mmax = max(Mmax, -res.fun)
        Smax = max(Smax, float(np.dot(w, [(A @ d).max() - (Bv @ d).max() for d in dirs])) / float(np.sum(w)))
    return Mmax, Smax


def record(cell, dirs, w, res, extra):
    d = {"label": cell["label"], "object": None, "cell": None, "rotation": None,
         "directions": [{"x": fhex(a), "y": fhex(b)} for a, b in dirs],
         "weights": [fhex(x) for x in w],
         "T_max": {"definition": "R_O + R_I, 3D 꼭짓점 반지름 (회전 중심 = 원점, 투영 전); 직교 투영 비확대",
                   "R_O_upper": mpf_exact(mp.make_mpf(cell["R_O"]._mpi_[1])),
                   "R_I_upper": mpf_exact(mp.make_mpf(cell["R_I"]._mpi_[1])),
                   "interval": iv_exact(cell["T_max"])},
         "methods": {}}
    d.update(extra)
    for m, r in res.items():
        d["methods"][m] = {
            "outer_upper": [mpf_exact(mp.make_mpf(u._mpi_[1])) for u in r["up"]],
            "inner_lower": [mpf_exact(mp.make_mpf(l._mpi_[0])) for l in r["lo"]],
            "sum_w": iv_exact(r["s"]), "residual_r": [iv_exact(c) for c in r["r"]],
            "exact_balance": r["exact"], "B": iv_exact(r["B"]), "status": r["status"], "time_s": r["time"]}
    return d


def P_obj():
    return {"type": "Zeng P_t = conv{p_i, -t p_i}", "t": f"{T_NUM}/{T_DEN}",
            "p": P0, "vertices_exact": [[str(c) for c in p] for p in P0] +
            [[f"{-T_NUM * c}/{T_DEN}" for c in p] for p in P0]}


ROT_P = {"outer": "R_out = Ry(-theta1) @ Rz(-phi1)", "inner": "R_in = Rz(alpha) @ Ry(-theta2) @ Rz(-phi2)",
         "angles_order": "x = (alpha, phi2, theta2, theta1, phi1)",
         "Rz(a)": "[[cos a,-sin a,0],[sin a,cos a,0],[0,0,1]]", "Ry(a)": "[[cos a,0,sin a],[0,1,0],[-sin a,0,cos a]]",
         "projection": "회전 후 (x, y) 성분 (z축 방향 직교 투영), 회전 중심 = 원점",
         "rho_bound": "rho_out = h_theta1 + h_phi1, rho_in = h_alpha + h_theta2 + h_phi2"}


if __name__ == "__main__":
    T_ALL = time.time()
    out_lines = []

    def log(s=""):
        print(s, flush=True)
        out_lines.append(s)

    env = {"python": sys.version.split()[0], "mpmath": mpmath.__version__, "numpy": np.__version__,
           "scipy": scipy.__version__, "iv_dps": iv.dps}
    log(f"환경 {env}")
    certs = []
    summary = []

    # ── 회귀: P1, P2, C3D ──
    log("=" * 110)
    log("회귀 칸 (P1, P2, C3D)")
    log("=" * 110)
    regress = [("P1", [0.7, 2.1, 1.3, 0.4, 0.9]), ("P2", [0.0, 0.9, 0.4, 0.4, 0.9])]
    cells = []
    for lab, cen in regress:
        half = [0.001] * 5
        Ro_f, Ri_f = C.float_rot_R8(np.array(cen))
        dirs_np, w_np, Mc = C.center_choice(PT_F, PT_F, Ro_f, Ri_f)
        cells.append((make_Pt_cell(lab, cen, half), [tuple(map(float, d)) for d in dirs_np],
                      [float(x) for x in w_np], Mc, asym_angle(np.array(cen)), lab))
    c3 = make_C3D_cell()
    d60 = [(0.5, float(mpmath.sqrt(3) / 2)), (-0.5, -float(mpmath.sqrt(3) / 2))]
    cells.append((c3, d60, [0.5, 0.5], None, None, "C3D"))

    # ── 후보 선택 ──
    log("")
    log("=" * 110)
    log(f"후보 조사: P_{{{T_NUM}/{T_DEN}}}, 시드 {SEED}, 균등 표본 (Λ = [0,2π]²×[0,π]×[0,π/2]²), 상한 {N_CAND_MAX}개 또는 {T_CAND_MAX:.0f}s")
    log(f"  선택: float LP M_center ∈ [{M_LO:g}, {M_HI:g}] 이고 비대칭각 > {ASYM_MIN} rad (몸체 투영 방향의 48 부호치환 최소각)")
    log("  (선택 범위·비대칭 기준·반폭 규칙은 시험용 입력이며 EQI 임계치·격자 법칙이 아님)")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    picked, n_tried, n_inM = [], 0, 0
    allM = []                                        # 분포 기록용 (선택 기준과 무관)
    while n_tried < N_CAND_MAX and time.time() - t0 < T_CAND_MAX and len(picked) < 10:
        x = LO + rng.random(5) * (HI - LO)
        n_tried += 1
        Ro_f, Ri_f = C.float_rot_R8(x)
        try:
            dirs_np, w_np, Mc = C.center_choice(PT_F, PT_F, Ro_f, Ri_f)
        except Exception:
            continue
        allM.append((Mc, x))
        if M_LO <= Mc <= M_HI:
            n_inM += 1
            a = asym_angle(x)
            if a > ASYM_MIN:
                picked.append((x, dirs_np, w_np, Mc, a))
    log(f"  조사 {n_tried}개 ({time.time() - t0:.1f}s), M 범위 내 {n_inM}개, 비대칭 통과·선택 {len(picked)}개")
    Ms = np.array([m for m, _ in allM])
    top = sorted(allM, key=lambda z: -z[0])[:3]
    log(f"  [분포 참고, 선택에 미사용] M_center: 최대 {Ms.max():.4e}, 중앙 {np.median(Ms):.4e}, 최소 {Ms.min():.4e};"
        f" M > −1e-2: {int(np.sum(Ms > -1e-2))}개, M > −5e-2: {int(np.sum(Ms > -5e-2))}개")
    for m, x in top:
        log(f"    상위 M {m:.4e} (비대칭각 {asym_angle(x):.4f})")
    for k, (x, dn, wn, Mc, a) in enumerate(picked):
        log(f"    S{k}: x = ({', '.join(f'{v:.6f}' for v in x)}), M_center = {Mc:.4e}, 비대칭각 = {a:.4f}")
        for fct in WIDTH_FACTORS:
            half = [abs(Mc) / fct] * 5
            cells.append((make_Pt_cell(f"S{k}/h=|M|/{fct}", list(map(float, x)), half),
                          [tuple(map(float, d)) for d in dn], [float(v) for v in wn], Mc, a, f"S{k}"))

    # ── 모든 방법 적용 ──
    log("")
    log("=" * 110)
    log("칸별 결과 (B 상한 / 폭 / 판정 / 시간)")
    log("=" * 110)
    for cell, dirs, w, Mc, a, base in cells:
        res = run_methods(cell, dirs, w)
        hdr = f"  [{cell['label']}]"
        if Mc is not None:
            hdr += f" M_center = {Mc:.4e}, 비대칭각 = {a:.4f}, ρ_out ≤ {C.F(cell['rho_out'].b, 6)}, ρ_in ≤ {C.F(cell['rho_in'].b, 6)}"
        else:
            hdr += f" (정육면체 육각 투영 쌍, λ = 26/25; M1 = ρ_in=π/360 볼, M2 = Rz(θ), |θ| ≤ π/360 인수 구간)"
        log(hdr)
        for m in ("M1_ball", "M2_factor", "M3_combined"):
            r = res[m]
            log(f"     {m:<12} B 상한 = {C.F(r['B'].b, 14):<24} 폭 = {C.F(r['B'].delta, 3):<10} → {r['status']:<8}"
                f" ({r['time'] * 1e3:.1f} ms)")
        # 방향별 경계 비교 (방법 2 가 더 나쁜 방향 수)
        worse_up = sum(1 for u1, u2 in zip(res["M1_ball"]["up"], res["M2_factor"]["up"]) if u2 > u1)
        worse_lo = sum(1 for l1, l2 in zip(res["M1_ball"]["lo"], res["M2_factor"]["lo"]) if l2 < l1)
        log(f"     방향별: M2 outer upper 가 M1 보다 큼 {worse_up}/{len(dirs)}, M2 inner lower 가 M1 보다 작음 {worse_lo}/{len(dirs)}")
        if cell["kind"] == "P_t":
            Mmax, Smax = sample_cell(cell["center"], cell["half"], dirs, w)
            log(f"     [비인증 sanity 1000 배치] 실제 M 최대 = {Mmax:.4e}, Σw(h_O−h_I)/Σw 최대 = {Smax:.4e}"
                f" (≤ 최선 B 상한: {Smax <= min(float(mp.make_mpf(res[m]['B']._mpi_[1])) for m in res) + 1e-12})")
            extra = {"object": P_obj(), "rotation": ROT_P,
                     "cell": {"center": [fhex(v) for v in cell["center"]], "half_width": [fhex(v) for v in cell["half"]],
                              "intervals": [[fhex(c - h), fhex(c + h)] for c, h in zip(cell["center"], cell["half"])],
                              "interval_note": "구간 = [c−h, c+h] 를 mpmath iv 로 외향 반올림 (위 float 값은 참고용 근사)",
                              "M_center_float": Mc, "asym_angle": a}}
        else:
            extra = {"object": {"type": "cube [-1,1]^3", "inner_scale": "26/25"},
                     "rotation": {"outer": "rows (e1, e2, e1×e2), e1=(1,-1,0)/√2, e2=(1,1,-2)/√6, 고정",
                                  "inner_M1": "SO(3) 볼, 중심 = 항등, rho = π/360 (탈평면 회전 포함)",
                                  "inner_M2": "Rz(theta), theta ∈ [−π/360, π/360] (평면 회전 1차원 집합)",
                                  "note": "M1 집합 ⊇ M2 집합. 같은 칸이 아님 — 비교는 참고"},
                     "cell": {"theta_interval": "[-pi/360, pi/360]"}}
        certs.append(record(cell, dirs, w, res, extra))
        summary.append((cell["label"], base, {m: res[m] for m in res}))

    # ── 요약 ──
    log("")
    log("=" * 110)
    log("방법별 요약 (선택 칸 S*, 회귀 칸 별도)")
    log("=" * 110)
    for grp, filt in (("회귀 (P1, P2, C3D)", lambda b: not b.startswith("S")), ("선택 칸 S*", lambda b: b.startswith("S"))):
        rows = [s for s in summary if filt(s[1])]
        log(f"  {grp}: 칸 {len(rows)}개")
        for m in ("M1_ball", "M2_factor", "M3_combined"):
            n_cert = sum(1 for s in rows if s[2][m]["status"] == "인증")
            tt = sum(s[2][m]["time"] for s in rows)
            log(f"     {m:<12} 인증 {n_cert}/{len(rows)},  총 시간 {tt * 1e3:.1f} ms")
    log(f"총 소요 {time.time() - T_ALL:.1f}s")
    with open("code/rupert_cell_compare_certificates.json", "w") as f:
        json.dump({"environment": env, "seed": SEED, "selection": {
            "n_tried": n_tried, "n_in_M_range": n_inM, "n_picked": len(picked), "M_range": [M_LO, M_HI],
            "M_center_max_observed": float(Ms.max()),
            "asym_min_rad": ASYM_MIN, "width_rule": "h = |M_center|/factor, factor ∈ (50, 10) — 시험 입력"},
            "theorem": "B = [Σw(outer_upper−inner_lower) + T_max‖r‖]/Σw ≤ 0 ⇒ 칸 전체 엄격 포함 배제 (r=0 정확이면 T 항 없음)",
            "number_format": "mpf_exact: 값 = man × 2^exp2 (정확), dec 는 30자리 표시; float 은 hex 로 정확 저장",
            "certificates": certs}, f, ensure_ascii=False, indent=1)
    with open("code/rupert_cell_compare_output.txt", "w") as f:
        f.write("\n".join(out_lines) + "\n")

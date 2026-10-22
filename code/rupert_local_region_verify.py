#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국소 인증 영역 독립 재검증기 — 저장 증명서 (code/rupert_local_region_cert.json) 만 사용
- 새 탐색·float LP·무작위 표본 없음. mpmath/numpy/scipy 미사용 (표준 라이브러리 Fraction 만).
- 산술: 유리수 구간 (Fraction 끝점, 연산마다 2^-PREC 격자로 바깥 반올림). cos/sin 은 유리점 교대급수
  + 잘림 오차 상계, 구간 끝점에서의 단조성 (구간 ⊂ (0, 3/2) ⊂ (0, π/2)) 으로 감쌈. sqrt 는 정수 isqrt.
- 균형 해: Cramer 공식 (3×3 행렬식 명시식) — 생성기의 mpmath LU 경로와 구현 분리.
- 증명서에서 읽는 것: 정의 입력 (t_shape, P0, Q 끝점, 회전 인수 순서, 오각형 순서, 쌍대 꼭짓점 변 3개,
  증명서 (J, 접촉 꼭짓점), 구면 칸 끝점과 칸별 선택 증명서). 생성기의 수치 결과 (a, b, δ, w 구간) 는
  판정에 쓰지 않고, 마지막에 대조 정보로만 표시.
- 어느 단계라도 미확정이면 FAIL/UNKNOWN 으로 종료 (exit code 1).
실행: python3 code/rupert_local_region_verify.py [증명서 경로]
"""

import json
import math
import sys
from fractions import Fraction as Fr

PREC = 200
SCALE = 1 << PREC
CERT = sys.argv[1] if len(sys.argv) > 1 else "code/rupert_local_region_cert.json"
OUT = None if len(sys.argv) > 1 else "code/rupert_local_region_verify_output.txt"   # 변조 시험 등 다른 입력이면 파일 미기록
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


def stop(status, msg):
    log(f"*** {status}: {msg}")
    if OUT:
        with open(OUT, "w") as f:
            f.write("\n".join(out) + "\n")
    sys.exit(1)


def rdn(x):
    return Fr(math.floor(x * SCALE), SCALE)


def rup(x):
    return Fr(math.ceil(x * SCALE), SCALE)


class R:
    """유리수 구간 [lo, hi], 바깥 반올림."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None, exact=False):
        hi = lo if hi is None else hi
        lo, hi = Fr(lo), Fr(hi)
        if lo > hi:
            raise ValueError("빈 구간")
        self.lo, self.hi = (lo, hi) if exact else (rdn(lo), rup(hi))

    def __add__(s, o):
        o = o if isinstance(o, R) else R(o, exact=True)
        return R(s.lo + o.lo, s.hi + o.hi)
    __radd__ = __add__

    def __neg__(s):
        return R(-s.hi, -s.lo, exact=True)

    def __sub__(s, o):
        o = o if isinstance(o, R) else R(o, exact=True)
        return R(s.lo - o.hi, s.hi - o.lo)

    def __rsub__(s, o):
        return R(o, exact=True) - s

    def __mul__(s, o):
        o = o if isinstance(o, R) else R(o, exact=True)
        p = (s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi)
        return R(min(p), max(p))
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = o if isinstance(o, R) else R(o, exact=True)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("분모 구간이 0 포함")
        p = (s.lo / o.lo, s.lo / o.hi, s.hi / o.lo, s.hi / o.hi)
        return R(min(p), max(p))

    def sq(s):
        if s.lo >= 0:
            return R(s.lo ** 2, s.hi ** 2)
        if s.hi <= 0:
            return R(s.hi ** 2, s.lo ** 2)
        return R(0, max(s.lo ** 2, s.hi ** 2))

    def sqrt(s):
        if s.lo < 0:
            raise ValueError("sqrt 음수 구간")
        lo_ = Fr(math.isqrt(math.floor(s.lo * SCALE * SCALE)), SCALE)
        hi_ = Fr(math.isqrt(math.ceil(s.hi * SCALE * SCALE)) + 1, SCALE)
        return R(lo_, hi_, exact=True)

    def __repr__(s):
        return f"[{float(s.lo):.12g}, {float(s.hi):.12g}]"


ZERO = R(0, exact=True)


def series_cos_sin(x):
    """유리수 x (|x| < 3/2) 에서 cos x, sin x 를 감싸는 구간 (교대급수, 항 단조감소 구간에서 잘림 오차 ≤ 다음 항)."""
    assert abs(x) < Fr(3, 2)
    res = []
    for start in (0, 1):  # 0: cos, 1: sin
        term = Fr(1) if start == 0 else Fr(x)
        k = start
        total = Fr(0)
        sign = 1
        while True:
            total += sign * term
            nxt = term * x * x / ((k + 1) * (k + 2))
            k += 2
            sign = -sign
            if abs(nxt) < Fr(1, SCALE * SCALE):
                err = abs(nxt)
                res.append(R(total - err, total + err))
                break
            term = nxt
    return res[0], res[1]


def cos_sin_interval(a, b):
    """[a, b] ⊂ (0, 3/2) ⊂ (0, π/2) 에서 cos 감소, sin 증가 → 끝점으로 감쌈."""
    if not (0 < a <= b < Fr(3, 2)):
        stop("UNKNOWN", f"각 구간 [{a}, {b}] 이 단조 구간 (0, 3/2) 밖")
    ca, sa = series_cos_sin(a)
    cb, sb = series_cos_sin(b)
    return R(cb.lo, ca.hi, exact=True), R(sa.lo, sb.hi, exact=True)


def dot(a, b):
    t = ZERO
    for x, y in zip(a, b):
        t = t + x * y
    return t


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def F(x, k=12, mode="lo"):
    v = x.lo if mode == "lo" else x.hi
    n = math.floor(v * 10 ** k) if mode == "lo" else math.ceil(v * 10 ** k)
    s = "-" if n < 0 else ""
    n = abs(n)
    return f"{s}{n // 10 ** k}.{n % 10 ** k:0{k}d}"


def Ffr(v, k=12):
    n = math.floor(v * 10 ** k)
    s = "-" if n < 0 else ""
    n = abs(n)
    return f"{s}{n // 10 ** k}.{n % 10 ** k:0{k}d}"


def Cfr(v, k=12):
    n = math.ceil(v * 10 ** k)
    s = "-" if n < 0 else ""
    n = abs(n)
    return f"{s}{n // 10 ** k}.{n % 10 ** k:0{k}d}"


SEP = "=" * 100
log(SEP)
log("국소 인증 영역 독립 재검증 (Fraction 구간, Cramer, 저장 증명서만 사용)")
log(SEP)
C = json.load(open(CERT))
log(f"  증명서 형식: {C['format']}; 산술 격자 2^-{PREC}; Python {sys.version.split()[0]} (외부 수치 패키지 미사용)")

# ── 1. 정확 입력 복원 ───────────────────────────────────────────
t = Fr(C["object"]["t_shape"])
P0 = [tuple(Fr(c) for c in p) for p in C["object"]["P0"]]
V = [tuple(Fr(c) for c in p) for p in P0] + [tuple(-t * c for c in p) for p in P0]
Vfile = [tuple(Fr(x) for x in v) for v in C["object"]["vertices"]]
if t != Fr(11, 20) or V != Vfile:
    stop("FAIL", "t_shape 또는 꼭짓점이 정의 conv{p_i, −t p_i} 와 불일치")
log(f"[1] t_shape = {t}, 꼭짓점 8개 정의에서 재생성·일치")
q1a, q1b = (Fr(x) for x in C["Q"]["q1"])
q2a, q2b = (Fr(x) for x in C["Q"]["q2"])
if not (q1a == Fr(2, 5) - Fr(1, 1024) and q1b == Fr(2, 5) + Fr(1, 1024)
        and q2a == Fr(9, 10) - Fr(1, 1024) and q2b == Fr(9, 10) + Fr(1, 1024)):
    stop("FAIL", "Q 끝점이 q0 ± 1/1024 와 불일치")
log(f"    Q = [{q1a}, {q1b}] × [{q2a}, {q2b}] (= (2/5, 9/10) ± 1/1024)")
if not C["rotation"]["factor_order"].startswith("R_out(q) = Ry(-q1) @ Rz(-q2)"):
    stop("FAIL", "회전 인수 순서 불일치")
log("    회전: R_out(q) = Ry(−q1)·Rz(−q2), R_in = R_out(q)·exp(θ[u]×). 닫힌 꼴 (검증기 독립 전개):")
log("      R_out = [[c1c2, c1s2, −s1], [−s2, c2, 0], [s1c2, s1s2, c1]]  (c1 = cos q1 등)")
c1, s1 = cos_sin_interval(q1a, q1b)
c2, s2 = cos_sin_interval(q2a, q2b)
Rm = [[c1 * c2, c1 * s2, -s1], [-s2, c2, ZERO], [s1 * c2, s1 * s2, c1]]
# 직교성 검사 (구간 내 정확 직교 행렬 존재와 정합)
for i in range(3):
    for j in range(3):
        g = dot(Rm[i], Rm[j])
        target = 1 if i == j else 0
        if not (g.lo <= target <= g.hi):
            stop("FAIL", "회전 구간 행렬의 직교성 정합 실패")
Vi = [[R(c, exact=True) for c in v] for v in V]
Pq = [(dot(Rm[0], v), dot(Rm[1], v)) for v in Vi]       # 투영 = 회전 후 처음 두 좌표

# ── 2. Q 전체의 오각형·접촉 구조 ─────────────────────────────────
order = C["pentagon"]["order_ccw"]
if sorted(set(order)) != sorted(order) or len(order) != 5 or not all(0 <= x < 8 for x in order):
    stop("FAIL", "오각형 순서 형식 오류")
edges = []
worst = None
for j in range(5):
    a, b = order[j], order[(j + 1) % 5]
    ej = C["pentagon"]["edges"][j]
    if (ej["a"], ej["b"]) != (a, b) or sorted(ej["contact_set_theta0"]) != sorted([a, b]):
        stop("FAIL", f"변 {j} 정의/접촉 집합 불일치")
    ex, ey = Pq[b][0] - Pq[a][0], Pq[b][1] - Pq[a][1]
    for k in range(8):
        if k in (a, b):
            continue
        cr = ex * (Pq[k][1] - Pq[a][1]) - ey * (Pq[k][0] - Pq[a][0])   # 왼쪽 (반시계 내부) 이면 > 0
        if cr.lo <= 0:
            stop("UNKNOWN" if cr.hi > 0 else "FAIL", f"변 ({a},{b}) 에 대해 꼭짓점 {k} 의 엄격 내부 미인증")
        worst = cr.lo if worst is None else min(worst, cr.lo)
    n = (ey, -ex)
    nu = [n[0], n[1], ZERO]
    nup = [Rm[0][c] * n[0] + Rm[1][c] * n[1] for c in range(3)]      # R_outᵀ ν
    edges.append(dict(a=a, b=b, n=n, nup=nup, nrm=(n[0].sq() + n[1].sq()).sqrt()))
log(f"[2] 모든 q ∈ Q: 각 변 (반시계) 에 대해 나머지 6 꼭짓점이 엄격히 왼쪽 (외적 하한 최소 {Ffr(worst, 8)} > 0)")
log(f"    → 그림자 = 오각형 {order}, θ=0 접촉 집합 = 변 양 끝 (Q 전체에서 구조 불변)")

# ── 3. θ = 0 보조정리 ───────────────────────────────────────────
area2 = ZERO
for j in range(5):
    p, r = Pq[order[j]], Pq[order[(j + 1) % 5]]
    area2 = area2 + (p[0] * r[1] - p[1] * r[0])
area = area2 / 2
if area.lo <= 0:
    stop("UNKNOWN", "Q 전체에서 오각형 면적 양수 미인증")
log(f"[3] θ = 0 보조정리: 안쪽 그림자 = 바깥 그림자 O (같은 다각형). O 면적 ∈ [{F(area, 8)}, {F(area, 8, 'hi')}] > 0 (Q 전체)")
log("    증명: O 는 컴팩트이고 내부가 비어 있지 않은 볼록 다각형.")
log("      τ = 0: O 의 경계점 x (변 위의 점) 는 x ∉ int(O) → O ⊄ int(O).")
log("      τ ≠ 0: x* = argmax_{x∈O} τ·x 이면 τ·(x*+τ) = max_O τ·x + |τ|² > max_O τ·x → x*+τ ∉ O → O+τ ⊄ int(O).")
log("    ⇒ θ = 0 에서 모든 τ 에 대해 O+τ ⊂ int(O) 불가능. (B(0) = 0 은 그대로 두며 음수로 바꾸지 않음.)")
log("    LP 관점: θ=0 이면 g_j = 0 (기호), 양수 w*(q) 로 m ≤ 0, τ = 0 에서 m = 0 → M(0) = 0 (엄격 여유 없음).")

# ── 4. 정확 균형 (Cramer) ───────────────────────────────────────
DV = {}
for d in C["balance"]["dual_vertices"]:
    J = tuple(d["edges"])
    if len(set(J)) != 3 or not all(0 <= x < 5 for x in J):
        stop("FAIL", f"쌍대 꼭짓점 {J} 형식 오류")
    N = [edges[x]["n"] for x in J]
    one = R(1, exact=True)
    M = [[one, one, one], [N[0][0], N[1][0], N[2][0]], [N[0][1], N[1][1], N[2][1]]]
    D = det3(M)
    if D.lo <= 0 <= D.hi:
        stop("UNKNOWN", f"J{J}: det 구간이 0 포함 (유일성 미인증)")
    rhs = [one, ZERO, ZERO]
    w = []
    for c in range(3):
        Mc = [[rhs[r] if cc == c else M[r][cc] for cc in range(3)] for r in range(3)]
        w.append(det3(Mc) / D)
    if not all(x.lo > 0 for x in w):
        stop("UNKNOWN", f"J{J}: Cramer 해의 양수성 미인증")
    s = ZERO
    for wj, x in zip(w, J):
        s = s + wj * edges[x]["nrm"]
    if s.lo <= 0:
        stop("UNKNOWN", f"J{J}: s′ 양수 미인증")
    DV[J] = dict(w=w, s=s, D=D)
    log(f"[4] J{J}: det (Cramer 명시식) ∈ [{F(D, 6)}, {F(D, 6, 'hi')}] ∌ 0 → 모든 q 에서 w*(q) 존재·유일;"
        f" w ∈ ({', '.join(f'[{F(x, 6)}, {F(x, 6, 'hi')}]' for x in w)}) > 0; s′ ≥ {F(s, 6)}")
log("    정확 균형 근거: w*(q) 가 Σw n(q) = 0, Σw = 1 의 정확해 (구간 잔차 0 포함 여부는 사용하지 않음)")

# ── 5. 증명서 재구성 ────────────────────────────────────────────
CERTS = []
for c in C["certificates"]["list"]:
    J = tuple(c["J"])
    if J not in DV:
        stop("FAIL", f"증명서 {c['k']}: 쌍대 꼭짓점 {J} 미검증")
    ch = c["contact"]
    if len(ch) != 3 or any(ch[i] not in (edges[J[i]]["a"], edges[J[i]]["b"]) for i in range(3)):
        stop("FAIL", f"증명서 {c['k']}: 접촉 꼭짓점이 변 양 끝이 아님")
    dv = DV[J]
    d = [ZERO, ZERO, ZERO]
    for wj, x, v in zip(dv["w"], J, ch):
        cc = cross(Vi[v], edges[x]["nup"])
        d = [d[i] + wj * cc[i] for i in range(3)]
    CERTS.append(dict(J=J, ch=ch, dhat=[x / dv["s"] for x in d], dv=dv))
if len(CERTS) != 24 or len({(c["J"], tuple(c["ch"])) for c in CERTS}) != 24:
    stop("FAIL", "증명서 24개 (서로 다름) 아님")
log(f"[5] 증명서 24개 정의 재구성 (J ∈ {sorted(DV)}, 접촉 = 변 양 끝, d̂_k = Σw_j(v_j × R_outᵀν_j)/s′)")

# ── 6. 구면 피복 완전성 ─────────────────────────────────────────
cells = C["sphere"]["cells"]
byface = {f: [] for f in range(6)}
for c in cells:
    s0, s1 = (Fr(x) for x in c["s"])
    t0, t1 = (Fr(x) for x in c["t"])
    if not (-1 <= s0 < s1 <= 1 and -1 <= t0 < t1 <= 1) or c["face"] not in byface:
        stop("FAIL", "칸 형식/범위 오류")
    byface[c["face"]].append((s0, s1, t0, t1))
for f, rs in byface.items():
    area_sum = sum((r[1] - r[0]) * (r[3] - r[2]) for r in rs)
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            a_, b_ = rs[i], rs[j]
            if min(a_[1], b_[1]) > max(a_[0], b_[0]) and min(a_[3], b_[3]) > max(a_[2], b_[2]):
                stop("FAIL", f"면 {f}: 칸 내부 겹침")
    if area_sum != 4:
        stop("FAIL", f"면 {f}: 칸 면적 합 {area_sum} ≠ 4 (피복 불완전)")
log(f"[6] 면마다 칸이 [−1,1]² 안에 있고 내부 서로소, 면적 합 = 4 (정확) → 닫힌 칸 합집합 = [−1,1]²")
log("    (합집합은 닫힘; 여집합이 비어 있지 않으면 열린 집합이라 양의 면적 → 면적 합 4 와 모순)")
log("    6면 정규화 상 = S² (최대 절댓값 성분으로 나누면 해당 면 위). 칸 총 " + str(len(cells)))


# ── 7. 칸별 a > 0, b, δ 재계산 ──────────────────────────────────
def face_v(f, S, T):
    one = R(1, exact=True)
    return {0: [one, S, T], 1: [-one, S, T], 2: [S, one, T], 3: [S, -one, T], 4: [S, T, one], 5: [S, T, -one]}[f]


DSAFE = Fr(C["results"]["delta_safe"])
if DSAFE != Fr(1, 20):
    stop("FAIL", "delta_safe 가 1/20 아님")
amin = bmax = dmin = None
safe_all = True
cmp_a = 0
for c in cells:
    k = c["cert"]
    if not (0 <= k < 24):
        stop("FAIL", "칸 증명서 인덱스 오류")
    S = R(Fr(c["s"][0]), Fr(c["s"][1]), exact=True)
    T = R(Fr(c["t"][0]), Fr(c["t"][1]), exact=True)
    v = face_v(c["face"], S, T)
    nrm = (v[0].sq() + v[1].sq() + v[2].sq()).sqrt()
    if nrm.lo < 1:
        stop("UNKNOWN", "정규화 분모 하한 < 1")
    U = [x / nrm for x in v]
    ce = CERTS[k]
    a = dot(ce["dhat"], U)
    if a.lo <= 0:
        stop("UNKNOWN", f"칸 (면 {c['face']}, s {c['s']}, t {c['t']}): 선택 증명서 {k} 의 일차 하한 ≤ 0")
    b = ZERO
    for wj, x, vv in zip(ce["dv"]["w"], ce["J"], ce["ch"]):
        vp = edges[x]["nup"]
        b = b + wj * (dot(Vi[vv], vp) - dot(U, Vi[vv]) * dot(U, vp))
    b = b / ce["dv"]["s"]
    dl = Fr(1) if b.hi <= 0 else min(Fr(1), (R(a.lo, exact=True) / R(b.hi, exact=True)).lo)
    # δ_safe 직접 조건: θ ∈ (0, 1/20] 에서 (5/6)aθ > (1/2)θ² b  ⇐  a_lo ≥ (1/20) b_hi  (⇔ 1/20 ≤ a/b)
    if not (b.hi <= 0 or a.lo >= DSAFE * b.hi):
        safe_all = False
    amin = a.lo if amin is None else min(amin, a.lo)
    bmax = b.hi if bmax is None else max(bmax, b.hi)
    dmin = dl if dmin is None else min(dmin, dl)
    cmp_a = max(cmp_a, abs(float(a.lo) - float(Fr(c["a_lower"]))))
log(f"[7] 96칸 재계산 (칸별 선택 증명서만 사용):")
log(f"    min a (일차 하한) ≥ {Ffr(amin, 15)} > 0,  max b (이차 상한) ≤ {Cfr(bmax, 12)}")
log(f"    검증기 자체 δ 하한 = min_칸 min(1, a_lo/b_hi) ≥ {Ffr(dmin, 13)}")
log(f"    δ_safe = 1/20 ≤ 모든 칸 a/b: {safe_all}")
if not safe_all:
    stop("UNKNOWN", "δ_safe = 1/20 를 일부 칸에서 재확인하지 못함")

# ── 8. 생성기 수치 대조 (정보용, 판정 무관) ─────────────────────
gen_d = Fr(C["results"]["delta_Q_cellwise_lower"])
log("[8] (정보) 생성기 수치 대조 — 판정에 사용하지 않음")
log(f"    칸별 a 하한: 검증기 vs 생성기 최대 차 {cmp_a:.3e}")
log(f"    생성기 δ_Q(칸별) 표시 {C['results']['delta_Q_cellwise_lower'][:15]}…  ≤ 검증기 δ 하한? {gen_d <= dmin}"
    + ("" if gen_d <= dmin else "  (검증기 경로로는 이 값까지 재확인 못함; δ_safe 판정과 무관)"))

log("")
log(SEP)
log("PASS — 독립 재검증 완료")
log("  정리: P_{11/20} 의 지정한 바깥 방향 이웃 Q = (2/5, 9/10) ± 1/1024 (R_out(q) = Ry(−q1)Rz(−q2)) 에서,")
log("        모든 상대 회전축 u 와 0 ≤ θ ≤ 1/20 에 대해 엄격한 투영 포함이 불가능하다.")
log("        (θ > 0: B < 0, 즉 포함 자체 불가 / θ = 0: 보조정리 [3])")
log(f"  검증기 자체 δ 하한 {Ffr(dmin, 13)} (≥ 1/20). 일반 칸 배제와 결합 가능한 국소 인증 영역 하나이며 전역 판정이 아님.")
log(SEP)
if OUT:
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")

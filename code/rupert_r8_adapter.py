#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국소정리의 R8 접속 어댑터 (보조 모듈; legacy R8 판정기 대체 아님, legacy·보존 파일 무수정)

검증한 국소정리 (rupert_local_region_scope.json, 증명서 해시 고정):
  P_{11/20}, q ∈ Q = [2043/5120, 2053/5120] × [4603/5120, 4613/5120], R_out(q) = Ry(−q1)Rz(−q2),
  R_in = R_out(q)·Δ, angle(Δ) ≤ 1/20 (모든 축, θ = 0 포함) → 엄격한 투영 포함 불가능.

R8 인코딩 (code/taskR8_Pt_lattice.py 실제 코드에서 확인):
  x = (α, φ2, θ2, θ1, φ1);  R_out = Ry(−θ1)·Rz(−φ1);  R_in = Rz(α)·Ry(−θ2)·Rz(−φ2);  투영 = 회전 후 (x, y);
  물체 Pt(t) = vstack(P0, −t·P0) (t 는 float), λ = 1.
  → q = (θ1, φ1) 이 국소정리의 q 와 같은 convention. Δ = R_outᵀ R_in.

칸 전체 적용 조건 (모두 구간, 바깥 반올림, 칸 전체):
  1) 물체 정의가 정확히 P_{11/20} (P0 정확 일치, t 정확 유리수 11/20, λ = 1, encoding R8_v1). 아니면 NOT_APPLICABLE.
  2) θ1, φ1 구간 ⊆ Q.  확실히 밖 → UNKNOWN(OUTSIDE_Q), 걸침/미확정 → UNKNOWN(STRADDLES_Q).
  3) angle(Δ) ≤ 1/20 를 칸 전체에서 인증. Δ ∈ SO(3) 이고 trace Δ = 1 + 2cos θ, θ ∈ [0, π] 에서 cos 감소이므로
     θ ≤ 1/20 ⇔ trace Δ ≥ 1 + 2cos(1/20).  (acos 미사용)
     M1: 칸 전체 구간으로 trace Δ 직접 계산 → lo(trace) ≥ hi(1 + 2cos(1/20)) 이면 인증.
     M2: 쌍불변 거리 d(R,S) = angle(RᵀS) 의 삼각부등식. Δ(x) = Rz(φ1)Ry(θ1)Rz(α)Ry(−θ2)Rz(−φ2) 의 각 인수는
         단위속도 1-매개 부분군이므로 d(Δ(x), Δ(c)) ≤ Σ_i |x_i − c_i| ≤ H (H = 반폭 합).
         따라서 angle(Δ(x)) ≤ angle(Δ(c)) + H. 중심 c 의 trace 구간으로 cos angle(Δ(c)) ≥ κ 를 얻고,
         κ ≥ hi(cos(1/20 − H)) (1/20 − H > 0) 이면 칸 전체 인증. (중심값만 쓰는 것이 아니라 H 로 칸 전체를 덮음)
     밖 인증: M1 hi(trace) < lo(1 + 2cos(1/20)) 또는 M2 angle(Δ(c)) − H > 1/20 → UNKNOWN(OUTSIDE_THETA).
     그 외 → UNKNOWN(STRADDLES_THETA) (경계 걸침 또는 과대평가로 미확정).
  θ = 0 (Δ = I) 은 검증 정리 범위 [0, 1/20] 에 포함 (동일 투영 보조정리).
반환: LOCAL_NO_STRICT_CONTAINMENT / NOT_APPLICABLE / UNKNOWN (+ detail). UNKNOWN 을 통과/불통과로 바꾸지 않음.
산술: 표준 라이브러리 Fraction 구간 (2^-200 격자 바깥 반올림), π 는 Machin 공식 구간, cos/sin 은 테일러 + 라그랑주 잔차.
"""

import hashlib
import json
import math
from fractions import Fraction as Fr

PREC = 200
SCALE = 1 << PREC
SCOPE = "code/rupert_local_region_scope.json"
CERT = "code/rupert_local_region_cert.json"
P0_EXACT = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]


def rdn(x):
    return Fr(math.floor(x * SCALE), SCALE)


def rup(x):
    return Fr(math.ceil(x * SCALE), SCALE)


class R:
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

    def hull(s, o):
        return R(min(s.lo, o.lo), max(s.hi, o.hi), exact=True)

    def __repr__(s):
        return f"[{float(s.lo):.15g}, {float(s.hi):.15g}]"


def _atan_inv(n):
    """atan(1/n) 구간 (교대급수, 항 단조감소 → 잘림 오차 ≤ 다음 항)."""
    x = Fr(1, n)
    total, k, term, sign = Fr(0), 0, x, 1
    while True:
        total += sign * term / (2 * k + 1)
        k += 1
        term = term * x * x
        sign = -sign
        nxt = term / (2 * k + 1)
        if nxt < Fr(1, SCALE * SCALE):
            return R(total - nxt, total + nxt)


PI = 16 * _atan_inv(5) - 4 * _atan_inv(239)          # Machin


def _taylor(x, start):
    """유리수 x 에서 cos (start=0) / sin (start=1) 구간: 부분합 + 라그랑주 잔차 |x|^N/N!."""
    x = Fr(x)
    total, n, term = Fr(0), start, (Fr(1) if start == 0 else x)
    sign = 1
    while True:
        total += sign * term
        sign = -sign
        term = term * x * x / ((n + 1) * (n + 2))
        n += 2
        bound = abs(term)                               # |x|^n / n! (다음 차수) ≥ 라그랑주 잔차
        if bound < Fr(1, SCALE * SCALE) and n > abs(x) + 2:
            return R(max(total - bound, Fr(-1)), min(total + bound, Fr(1)))


def _crit_range(X, offset):
    """X 안에 있을 수 있는 (m + offset)π 의 정수 m 범위 (보수적 상위집합)."""
    if X.lo >= 0:
        m_lo = math.floor(X.lo / PI.hi - offset) - 1
    else:
        m_lo = math.floor(X.lo / PI.lo - offset) - 1
    if X.hi >= 0:
        m_hi = math.ceil(X.hi / PI.lo - offset) + 1
    else:
        m_hi = math.ceil(X.hi / PI.hi - offset) + 1
    res = []
    for m in range(m_lo, m_hi + 1):
        c = R(m + offset, exact=True) * PI
        if c.hi >= X.lo and c.lo <= X.hi:
            res.append(m)
    return res


def icos(X):
    r = _taylor(X.lo, 0).hull(_taylor(X.hi, 0))
    for m in _crit_range(X, Fr(0)):
        v = Fr(1) if m % 2 == 0 else Fr(-1)
        r = r.hull(R(v, exact=True))
    return r


def isin(X):
    r = _taylor(X.lo, 1).hull(_taylor(X.hi, 1))
    for m in _crit_range(X, Fr(1, 2)):
        v = Fr(1) if m % 2 == 0 else Fr(-1)
        r = r.hull(R(v, exact=True))
    return r


ZERO = R(0, exact=True)
ONE = R(1, exact=True)


def rz(a):
    c, s = icos(a), isin(a)
    return [[c, -s, ZERO], [s, c, ZERO], [ZERO, ZERO, ONE]]


def ry(a):
    c, s = icos(a), isin(a)
    return [[c, ZERO, s], [ZERO, ONE, ZERO], [-s, ZERO, c]]


def mm(A, B):
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3)] for i in range(3)]


def trace_delta(al, ph2, th2, th1, ph1):
    """trace(R_outᵀ R_in) = Σ_ij R_out[i][j]·R_in[i][j] (R8 인코딩 그대로)."""
    Ro = mm(ry(-th1), rz(-ph1))
    Ri = mm(mm(rz(al), ry(-th2)), rz(-ph2))
    t = ZERO
    for i in range(3):
        for j in range(3):
            t = t + Ro[i][j] * Ri[i][j]
    return t


def _parse_box(units, box):
    xs = []
    for lo_s, hi_s in box:
        lo_, hi_ = Fr(lo_s), Fr(hi_s)
        if lo_ > hi_:
            raise ValueError("칸 끝점 순서 오류")
        if units == "rad":
            xs.append(R(lo_, hi_, exact=True))
        elif units == "pi":
            a, b = R(lo_, exact=True) * PI, R(hi_, exact=True) * PI
            xs.append(R(a.lo, b.hi, exact=True))
        else:
            raise ValueError("units 는 rad 또는 pi")
    return xs, [(Fr(l), Fr(h)) for l, h in box]


def _halfwidth_upper(units, ends):
    H = sum((h - l) / 2 for l, h in ends)
    return H if units == "rad" else (R(H, exact=True) * PI).hi


def _center(units, ends):
    cs = []
    for l, h in ends:
        c = (l + h) / 2
        if units == "rad":
            cs.append(R(c, exact=True))
        else:
            cs.append(R(c, exact=True) * PI)
    return cs


class LocalTheorem:
    def __init__(self, scope_path=SCOPE, cert_path=CERT):
        self.scope = json.load(open(scope_path))
        vt = self.scope["verified_theorem"]
        with open(cert_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.loaded_ok = vt["verifier_status"] == "PASS" and vt["certificate_sha256"] == h
        self.cert_hash = h
        self.q1 = tuple(Fr(x) for x in vt["Q"]["q1"])
        self.q2 = tuple(Fr(x) for x in vt["Q"]["q2"])
        self.theta_max = Fr(vt["theta_range"][1])
        self.t_shape = Fr(vt["object"]["t_shape"])
        self.thr = ONE + 2 * icos(R(self.theta_max, exact=True))     # 1 + 2cos(1/20)

    def classify(self, obj, units, box):
        rec = {"theorem_cert_sha256": self.cert_hash, "theta_range": ["0", str(self.theta_max)]}
        if not self.loaded_ok:
            return dict(rec, status="UNKNOWN", detail="SCOPE_NOT_VERIFIED",
                        reason="검증 범위 기록이 PASS 가 아니거나 증명서 해시 불일치")
        # 1) 물체·정의
        t = obj.get("t")
        if isinstance(t, float):
            return dict(rec, status="NOT_APPLICABLE", detail="OBJECT",
                        reason=f"t 가 float ({t!r} = {Fr(t)}) — 정확 유리수 정의 아님, 11/20 과 다름")
        try:
            t = Fr(str(t))
        except (ValueError, ZeroDivisionError):
            return dict(rec, status="NOT_APPLICABLE", detail="OBJECT", reason="t 해석 불가")
        if obj.get("family") != "Zeng_Pt" or obj.get("P0") != P0_EXACT or t != self.t_shape:
            return dict(rec, status="NOT_APPLICABLE", detail="OBJECT", reason=f"물체 정의 불일치 (t = {t})")
        if Fr(str(obj.get("lambda", "1"))) != 1 or obj.get("encoding") != "R8_v1":
            return dict(rec, status="NOT_APPLICABLE", detail="DEFINITION", reason="λ ≠ 1 또는 인코딩 불일치")
        xs, ends = _parse_box(units, box)
        al, ph2, th2, th1, ph1 = xs
        # 2) q ⊆ Q
        qin = th1.lo >= self.q1[0] and th1.hi <= self.q1[1] and ph1.lo >= self.q2[0] and ph1.hi <= self.q2[1]
        qout = th1.hi < self.q1[0] or th1.lo > self.q1[1] or ph1.hi < self.q2[0] or ph1.lo > self.q2[1]
        rec["q_box"] = {"theta1": repr(th1), "phi1": repr(ph1)}
        if not qin:
            return dict(rec, status="UNKNOWN", detail="OUTSIDE_Q" if qout else "STRADDLES_Q",
                        reason="q 구간이 Q 밖" if qout else "q 구간이 Q 경계를 걸치거나 포함 미확정")
        # 3) angle(Δ) ≤ 1/20
        tr = trace_delta(al, ph2, th2, th1, ph1)
        m1_in = tr.lo >= self.thr.hi
        m1_out = tr.hi < self.thr.lo
        H = _halfwidth_upper(units, ends)
        c = _center(units, ends)
        trc = trace_delta(*c)
        kappa = (trc.lo - 1) / 2                         # cos angle(Δ(c)) ≥ kappa
        m2_in = False
        if H < self.theta_max:
            m2_in = kappa >= icos(R(self.theta_max - H, exact=True)).hi
        # 밖 인증 (M2): angle(Δ(c)) > 1/20 + H ⇐ cos angle(Δ(c)) ≤ (trc.hi − 1)/2 < lo(cos(1/20 + H)), 1/20 + H < π
        m2_out = False
        if self.theta_max + H < PI.lo:
            m2_out = (trc.hi - 1) / 2 < icos(R(self.theta_max + H, exact=True)).lo
        rec.update(trace_box=repr(tr), threshold=repr(self.thr), trace_center=repr(trc), H_upper=float(H),
                   M1_in=m1_in, M2_in=m2_in, M1_out=m1_out, M2_out=m2_out)
        if m1_in or m2_in:
            return dict(rec, status="LOCAL_NO_STRICT_CONTAINMENT", detail="M1" if m1_in else "M2",
                        reason="칸 전체: 물체 P_{11/20}, q ⊆ Q, angle(Δ) ≤ 1/20 인증")
        if m1_out or m2_out:
            return dict(rec, status="UNKNOWN", detail="OUTSIDE_THETA", reason="상대 회전각 > 1/20 (국소정리 적용 밖)")
        return dict(rec, status="UNKNOWN", detail="STRADDLES_THETA",
                    reason="상대 회전각이 1/20 경계를 걸치거나 구간 과대평가로 미확정")

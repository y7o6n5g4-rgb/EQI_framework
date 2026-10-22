#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구간 노름 견고판 — rupert_cell_cert.py (보존본, 수정 없음) 의 구간 보조함수 수정 산출물.

변경점 (유일한 수학적 변경):
  inorm(v) = sqrt(Σ x*x)  →  sqrt(Σ x**2)
  mpmath.iv 에서 x*x 는 두 인수를 독립 구간으로 곱하므로 0 을 포함하는 x 에 대해 음수 하한을 낸다
  (예: [−1e−3, 2e−3]*[−1e−3, 2e−3] = [−2e−6, 4e−6]). 이 경우 iv.sqrt 는 ComplexResult 예외를 낸다.
  x**2 는 같은 구간의 제곱으로 [0, max(|a|,|b|)²] 을 준다 (포함 관계 유지, 더 좁음).
예외 처리: 구간연산 예외는 잡아서 0 이나 인증 성공으로 바꾸지 않는다 (호출자에게 그대로 전파).
U/L 은 |n| 일반형 (단위 아닌 n 허용), p = ‖n × w0‖, 분기 미확정·ρ ≥ π 이면 자명 경계 ±|n|R.
"""

from mpmath import iv
import mpmath


def I(x):
    return iv.mpf(x)


def isq(x):
    return x ** 2


def inorm(v):
    return iv.sqrt(sum((isq(x) for x in v), I(0)))


def idot(a, b):
    return sum((x * y for x, y in zip(a, b)), I(0))


def icross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


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


def iexp_axis(theta, k):
    """exp(θ·[k]×) (Rodrigues), k = 단위 축 (구간). R = I + sinθ K + (1−cosθ) K²"""
    s, c = iv.sin(theta), iv.cos(theta)
    K = [[I(0), -k[2], k[1]], [k[2], I(0), -k[0]], [-k[1], k[0], I(0)]]
    K2 = imul(K, K)
    Id = [[I(1), I(0), I(0)], [I(0), I(1), I(0)], [I(0), I(0), I(1)]]
    return [[Id[i][j] + s * K[i][j] + (1 - c) * K2[i][j] for j in range(3)] for i in range(3)]


def apply_exp_axis(theta, k, v):
    """v → v + sinθ (k×v) + (1−cosθ) k×(k×v)  (인수 구간 계산용, 꼭짓점별)"""
    kv = icross(k, v)
    kkv = icross(k, kv)
    s, c = iv.sin(theta), iv.cos(theta)
    return [v[i] + s * kv[i] + (1 - c) * kkv[i] for i in range(3)]


def apply_rz(a, v):
    c, s = iv.cos(a), iv.sin(a)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def apply_ry(a, v):
    c, s = iv.cos(a), iv.sin(a)
    return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]


def U_iv(n, w0, rho):
    nn, R = inorm(n), inorm(w0)
    if rho.a >= mpmath.pi:
        return nn * R
    q, p = idot(n, w0), inorm(icross(n, w0))
    c, s = iv.cos(rho), iv.sin(rho)
    d = q - nn * R * c
    if d.a >= 0:
        return nn * R
    if d.b < 0:
        return q * c + p * s
    return nn * R


def L_iv(n, w0, rho):
    nn, R = inorm(n), inorm(w0)
    if rho.a >= mpmath.pi:
        return -nn * R
    q, p = idot(n, w0), inorm(icross(n, w0))
    c, s = iv.cos(rho), iv.sin(rho)
    d = q + nn * R * c
    if d.b <= 0:
        return -nn * R
    if d.a > 0:
        return q * c - p * s
    return -nn * R


def B_from(up, lo, dirs, w, T_max):
    """B = [Σw(up−lo) + T_max‖r‖]/Σw  (r=0 정확이면 T 항 없음). up/lo 는 끝점(구간 점) 목록"""
    wI = [I(x) for x in w]
    s = sum(wI, I(0))
    r = [sum((wI[j] * I(dirs[j][k]) for j in range(len(w))), I(0)) for k in (0, 1)]
    rn = iv.sqrt(isq(r[0]) + isq(r[1]))
    num = sum((wI[j] * (I(up[j]) - I(lo[j])) for j in range(len(w))), I(0))
    exact = all(c.a == 0 and c.b == 0 for c in r)
    corr = I(0) if exact else T_max * rn
    return (num + corr) / s, s, r, rn, corr, exact

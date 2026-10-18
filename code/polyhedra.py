#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정다면체 5 + 아르키메데스 13 좌표 (mp.dps=400) 및 면 추출 — Task R / R5 공용
(Task R 스크립트에서 분리. 좌표·검증 내용 변경 없음)
"""

import itertools

import mpmath
from mpmath import mp, mpf, sqrt, pi
import numpy as np
from scipy.spatial import ConvexHull

mp.dps = 400
PHI = (sqrt(5) + 1) / 2

t_trib = mpmath.findroot(lambda t: t**3 - t**2 - t - 1, mpf("1.839"))
s2 = sqrt(2)


def even_perms(v):
    a, b, cc = v
    return [(a, b, cc), (b, cc, a), (cc, a, b)]


def odd_perms(v):
    a, b, cc = v
    return [(a, cc, b), (cc, b, a), (b, a, cc)]


def all_perms(v):
    return even_perms(v) + odd_perms(v)


def signs(v, parity=None):
    """부호 전개. parity: None=전부, 'even+'/'odd+' = 양수 부호 개수(0 아닌 성분 기준) 짝/홀"""
    out = []
    for sg in itertools.product((1, -1), repeat=3):
        if parity is not None:
            nplus = sum(1 for s_, comp in zip(sg, v) if s_ > 0 and comp != 0)
            if parity == "even+" and nplus % 2:
                continue
            if parity == "odd+" and nplus % 2 == 0:
                continue
        out.append(tuple(s_*comp for s_, comp in zip(sg, v)))
    return out


def build(gens):
    pts = []
    for perm_fn, base, parity in gens:
        for pv in perm_fn(base):
            pts.extend(signs(pv, parity))
    return dedupe(pts)


def dedupe(pts):
    uniq = []
    seen = set()
    for p in pts:
        key = tuple(round(float(t), 9) for t in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def rot(axis, ang):
    """Rodrigues 회전행렬 (mpmath, 400자리)"""
    n = sqrt(sum(a_**2 for a_ in axis))
    kx, ky, kz = (a_/n for a_ in axis)
    cs, sn = mpmath.cos(ang), mpmath.sin(ang)
    v = 1 - cs
    return mpmath.matrix([[cs + kx*kx*v, kx*ky*v - kz*sn, kx*kz*v + ky*sn],
                          [ky*kx*v + kz*sn, cs + ky*ky*v, ky*kz*v - kx*sn],
                          [kz*kx*v - ky*sn, kz*ky*v + kx*sn, cs + kz*kz*v]])


def snub_dodecahedron():
    """
    다듬은 정십이면체: 정이십면체 회전군 I(60원)에 의한 한 점 p 의 궤도.
    p 는 5회축(a5)·3회축(a3)·2회축(a2) 회전 상의 거리가 모두 같다는 조건으로 400자리 해.
    (좌표 공식 대신 군 궤도로 구성 — 검증: V=60, 80·3각+12·5각, E=150)
    """
    a5 = (mpf(0), mpf(1), PHI)                 # 정이십면체 꼭짓점 방향
    a3 = (mpf(1), mpf(1), mpf(1))              # 면 (0,1,φ),(1,φ,0),(φ,0,1) 중심 방향
    a2 = (PHI, mpf(1), PHI + 1)                # 모서리 (0,1,φ)–(φ,0,1) 중점 방향
    R5, R3, R2 = rot(a5, 2*pi/5), rot(a3, 2*pi/3), rot(a2, pi)

    def vec(th, ph):
        return mpmath.matrix([mpmath.sin(th)*mpmath.cos(ph), mpmath.sin(th)*mpmath.sin(ph), mpmath.cos(th)])

    def d2(u, v):
        w = u - v
        return w[0]**2 + w[1]**2 + w[2]**2

    def eqs(th, ph):
        q = vec(th, ph)
        return [d2(q, R5*q) - d2(q, R3*q), d2(q, R3*q) - d2(q, R2*q)]

    g = [a5[i]/sqrt(sum(t**2 for t in a5)) + a3[i]/sqrt(3) + a2[i]/sqrt(sum(t**2 for t in a2)) for i in range(3)]
    th0 = mpmath.acos(g[2]/sqrt(sum(t**2 for t in g)))
    ph0 = mpmath.atan2(g[1], g[0])
    th, ph = mpmath.findroot(eqs, (th0, ph0))
    q = vec(th, ph)
    # 군 I 생성 (R5, R3 폐포)
    group = [mpmath.eye(3)]
    frontier = [mpmath.eye(3)]
    key = lambda m: tuple(round(float(m[i, j]), 8) for i in range(3) for j in range(3))
    seen = {key(group[0])}
    while frontier:
        nxt = []
        for m in frontier:
            for gen in (R5, R3):
                mm = gen*m
                k = key(mm)
                if k not in seen:
                    seen.add(k)
                    group.append(mm)
                    nxt.append(mm)
        frontier = nxt
    assert len(group) == 60
    pts = [tuple(m*q) for m in group]
    return dedupe(pts)


def tet_parity(v):
    """정사면체·깎은 정사면체: 음수 부호 개수 짝수"""
    out = []
    for sg in itertools.product((1, -1), repeat=3):
        if sum(1 for s_ in sg if s_ < 0) % 2 == 0:
            out.append(tuple(s_*comp for s_, comp in zip(sg, v)))
    return out


F = PHI
SOLIDS = [
    # 이름, 기대 V, 대칭군, 쌍대, 기대 면 {변수: 개수}, 꼭짓점
    ("정사면체", 4, "T_d", "정사면체(자기쌍대)", {3: 4},
     [p for pv in [(mpf(1), mpf(1), mpf(1))] for p in tet_parity(pv)]),
    ("정육면체", 8, "O_h", "정팔면체", {4: 6}, build([(lambda v: [v], (mpf(1),)*3, None)])),
    ("정팔면체", 6, "O_h", "정육면체", {3: 8}, build([(all_perms, (mpf(1), mpf(0), mpf(0)), None)])),
    ("정십이면체", 20, "I_h", "정이십면체", {5: 12},
     build([(lambda v: [v], (mpf(1),)*3, None), (even_perms, (mpf(0), 1/F, F), None)])),
    ("정이십면체", 12, "I_h", "정십이면체", {3: 20}, build([(even_perms, (mpf(0), mpf(1), F), None)])),
    ("깎은 정사면체", 12, "T_d", "삼중사면체(카탈랑)", {3: 4, 6: 4},
     dedupe([p for pv in all_perms((mpf(3), mpf(1), mpf(1))) for p in tet_parity(pv)])),
    ("육팔면체", 12, "O_h", "마름모십이면체(카탈랑)", {3: 8, 4: 6},
     build([(all_perms, (mpf(1), mpf(1), mpf(0)), None)])),
    ("깎은 정육면체", 24, "O_h", "삼중팔면체(카탈랑)", {3: 8, 8: 6},
     build([(all_perms, (s2 - 1, mpf(1), mpf(1)), None)])),
    ("깎은 정팔면체", 24, "O_h", "사중육면체(카탈랑)", {4: 6, 6: 8},
     build([(all_perms, (mpf(0), mpf(1), mpf(2)), None)])),
    ("마름모육팔면체", 24, "O_h", "삼각이십사면체(카탈랑)", {3: 8, 4: 18},
     build([(all_perms, (mpf(1), mpf(1), 1 + s2), None)])),
    ("깎은 육팔면체", 48, "O_h", "육각팔면체(카탈랑)", {4: 12, 6: 8, 8: 6},
     build([(all_perms, (mpf(1), 1 + s2, 1 + 2*s2), None)])),
    ("다듬은 정육면체", 24, "O (카이랄)", "오각이십사면체(카탈랑)", {3: 32, 4: 6},
     build([(even_perms, (mpf(1), 1/t_trib, t_trib), "even+"),
            (odd_perms, (mpf(1), 1/t_trib, t_trib), "odd+")])),
    ("십이이십면체", 30, "I_h", "마름모삼십면체(카탈랑)", {3: 20, 5: 12},
     build([(even_perms, (mpf(0), mpf(0), F), None),
            (even_perms, (mpf(1)/2, F/2, F**2/2), None)])),
    ("깎은 정십이면체", 60, "I_h", "삼중이십면체(카탈랑)", {3: 20, 10: 12},
     build([(even_perms, (mpf(0), 1/F, 2 + F), None), (even_perms, (1/F, F, 2*F), None),
            (even_perms, (F, mpf(2), F**2), None)])),
    ("깎은 정이십면체", 60, "I_h", "오각십이면체(카탈랑)", {5: 12, 6: 20},
     build([(even_perms, (mpf(0), mpf(1), 3*F), None), (even_perms, (mpf(1), 2 + F, 2*F), None),
            (even_perms, (F, mpf(2), F**3), None)])),
    ("마름모십이이십면체", 60, "I_h", "삼각육십면체(카탈랑)", {3: 20, 4: 30, 5: 12},
     build([(even_perms, (mpf(1), mpf(1), F**3), None), (even_perms, (F**2, F, 2*F), None),
            (even_perms, (2 + F, mpf(0), F**2), None)])),
    ("깎은 십이이십면체", 120, "I_h", "육각이십면체(카탈랑)", {4: 30, 6: 20, 10: 12},
     build([(even_perms, (1/F, 1/F, 3 + F), None), (even_perms, (2/F, F, 1 + 2*F), None),
            (even_perms, (1/F, F**2, -1 + 3*F), None), (even_perms, (2*F - 1, mpf(2), 2 + F), None),
            (even_perms, (F, mpf(3), 2*F), None)])),
    ("다듬은 정십이면체", 60, "I (카이랄)", "오각육십면체(카탈랑)", {3: 80, 5: 12}, snub_dodecahedron()),
]


def face_data(pts):
    arr = np.array([[float(t) for t in p] for p in pts])
    hull = ConvexHull(arr)
    groups = {}
    for simp, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 7))
        groups.setdefault(key, set()).update(simp.tolist())
    faces = list(groups.values())
    return faces

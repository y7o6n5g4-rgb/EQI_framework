#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task R — 루퍼트 문제 EQI 전개 (TASKS.md v9.7, 독립 스크립트, 엔진 수정 없음)
mp.dps = 400  |  상수 정의 = CLAUDE.md / v9.6 엔진 동일
원칙: ν(Nieuwland)는 순수 무차원 대수적 수 → 정확값 대상은 400자리 대수필연만 인정.
      GAP·GD(ln5 포함 초월수) 항으로 ν 를 맞추는 피팅 없음. 하한값은 등급 대상 아님.
  R0  동치 사슬 앵커
  R1  I_h 추측 ρ 의 대수 구조
  R2  정확값 대수 대조 (GAP 상쇄 표현만, 유한 탐색)
  R3  기지 사례 데이터표 (등급 없음)
  R4  사전 예측 초안 (등록 아님)
문헌값 출처: TASKS.md (perplexity S114 조사, Type IV). 그 밖의 문헌값은 사용하지 않음.
Date: 2026-10-18
"""

import io
import os
import sys
import contextlib
import itertools

import mpmath
from mpmath import mp, mpf, sqrt, log, pi, fabs, nstr, power, exp
import sympy as sp

mp.dps = 400
TOL = mpf(10)**-390                       # 대수필연 기준

# ════════════════════════════════════════════════════════════════
# 핵심 상수 (CLAUDE.md)
# ════════════════════════════════════════════════════════════════
PHI    = (sqrt(5) + 1) / 2
GAP    = log(25) / 100
GD     = log(5) / 100
LN_C   = log(PHI) / 2
mu     = 1 / PHI
Lam    = PHI**-2
c      = sqrt(PHI)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import eqi_unified_engine_v9_6 as eng
    C = eng.EQIConstants()

n_c = (power(PHI, mpf(3)/2) - 1) / GAP
assert fabs(n_c - C.n_c) < TOL


def alg(tag, label, calc, target, src):
    """[§tag] label | calc | target | 400자리 절대 차 | 등급 | 출처"""
    e = fabs(calc - target)
    g = "대수필연" if e < TOL else "대수필연 미달"
    print(f"[{tag}] {label} | {nstr(calc, 20)} | {nstr(target, 20)} | {nstr(e, 6)} | {g} | {src}")
    return e < TOL


def psi(n):
    """§35 보른: ψ_re = (1+nGAP)/√((1+nGAP)²+φ⁶),  ψ_im = φ³/√(…)"""
    x = 1 + n*GAP
    d = sqrt(x**2 + PHI**6)
    return x/d, PHI**3/d


# ════════════════════════════════════════════════════════════════
# R0. 동치 사슬 앵커
# ════════════════════════════════════════════════════════════════
print("=" * 78)
print("R0. 동치 사슬 앵커")
print("=" * 78)
M = C.MAXWELL_5th                          # 엔진 §5 MAXWELL (5차 닫힘)
S_D1 = log(PHI) * M / (1 + M)              # = ln(1/ε₀) 몫
S_D2 = log(PHI) / (1 + M)                  # = |ln μ₀| 몫
eps0 = exp(-S_D1)
mu0  = exp(-S_D2)
print(f"  M (엔진 §5 MAXWELL_5th) = {nstr(M, 30)}")
print(f"  S_D1 = lnφ·M/(1+M) = {nstr(S_D1, 30)}")
print(f"  S_D2 = lnφ/(1+M)   = {nstr(S_D2, 30)}")
print(f"  ε₀ = e^(−S_D1) = {nstr(eps0, 30)},  μ₀ = e^(−S_D2) = {nstr(mu0, 30)}")
alg("R0-1a", "ε₀μ₀ = 1/c²", eps0*mu0, 1/c**2, "S63 §23")
alg("R0-1b", "1/c² = φ^(-1)", 1/c**2, 1/PHI, "S63 §23")
alg("R0-2a", "ln(1/ε₀)+|ln μ₀| = 2·ln c", log(1/eps0) + fabs(log(mu0)), 2*log(c), "S63 §23")
alg("R0-2b", "2·ln c = ln φ", 2*log(c), log(PHI), "S63 §23")
alg("R0-3a", "S_D1 + S_D2 = lnφ (합)", S_D1 + S_D2, log(PHI), "구성 항등")
alg("R0-3b", "S_D1/S_D2 = M (비)", S_D1/S_D2, M, "구성 항등")
print("  ※ R0-1~3은 S_D1·S_D2 정의(M 분할)에서 대수적으로 따라 나오는 구성 항등식.")

for nm, n in (("n_c", n_c), ("34", mpf(34)), ("37", mpf(37)), ("137", mpf(137))):
    re_, im_ = psi(n)
    alg(f"R0-4.{nm}", f"ψ_re²+ψ_im² = 1 (n={nm})", re_**2 + im_**2, mpf(1), "S68 §35")
re_c, im_c = psi(n_c)
alg("R0-5a", "n_c: ψ_re = μ/√2", re_c, mu/sqrt(2), "Task 2 §P0-9")
alg("R0-5b", "n_c: ψ_im = c/√2", im_c, c/sqrt(2), "Task 2 §P0-9")
alg("R0-5c", "n_c: |ψ_re|² = Λ/2", re_c**2, Lam/2, "Task 2 §P0-9")
alg("R0-5d", "n_c: |ψ_im|² = φ/2", im_c**2, PHI/2, "Task 2 §P0-9")
alg("R0-5e", "φ/2 = cos(π/5)", PHI/2, mpmath.cos(pi/5), "대수 항등")

# ════════════════════════════════════════════════════════════════
# R1. I_h 추측 ρ 의 대수 구조
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("R1. I_h 추측 ρ: 2025ρ⁸−11970ρ⁶+17009ρ⁴−9000ρ²+2000 의 최소 양근")
print("=" * 78)
x = sp.symbols('x')
s5 = sp.sqrt(5)
P = 2025*x**8 - 11970*x**6 + 17009*x**4 - 9000*x**2 + 2000
fl = sp.factor_list(P)
print(f"  Q 위 factor_list: {fl}")
print(f"  Q 위 기약: {len(fl[1]) == 1 and fl[1][0][1] == 1}")
fq5 = sp.factor(P, extension=s5)
print(f"  Q(√5) 위 인수분해: {fq5}")
y = sp.symbols('y')
Py = sp.expand(P.subs(x**2, y))
print(f"  ρ² = y 대입 4차: {Py}")
print(f"  Q(√5) 위 (y): {sp.factor(Py, extension=s5)}")

rho2 = (133 + 44*sqrt(5) - sqrt(9369 + 4504*sqrt(5))) / 90
rho = sqrt(rho2)
Pn = lambda t: 2025*t**8 - 11970*t**6 + 17009*t**4 - 9000*t**2 + 2000
print(f"  ρ  = {nstr(rho, 60)}")
alg("R1-1", "P(ρ) = 0 (대입 잔여)", Pn(rho), mpf(0), "arXiv:2509.08190 Table 1")
roots = mpmath.polyroots([2025, 0, -11970, 0, 17009, 0, -9000, 0, 2000], maxsteps=400, extraprec=800)
pos = sorted([mpmath.re(r) for r in roots if fabs(mpmath.im(r)) < mpf(10)**-300 and mpmath.re(r) > 0])
print(f"  실수 양근: {[nstr(r, 15) for r in pos]}")
alg("R1-2", "ρ = 최소 양근", rho, pos[0], "arXiv:2509.08190 Table 1")
phi_s = (1 + s5) / 2
print(f"  기호 확인: 133+44√5 − (89+88φ) = {sp.simplify(133 + 44*s5 - (89 + 88*phi_s))}")
print(f"  기호 확인: 20+8√5 − 4√5·φ³ = {sp.simplify(20 + 8*s5 - 4*s5*phi_s**3)}")
alg("R1-3a", "133+44√5 = 89+88φ", 133 + 44*sqrt(5), 89 + 88*PHI, "계수 φ 표현")
alg("R1-3b", "20+8√5 = 4√5·φ³", 20 + 8*sqrt(5), 4*sqrt(5)*PHI**3, "계수 φ 표현")
for tv in ("1.01082306075208", "1.01082306075264"):
    t = mpf(tv)
    print(f"  원문 Table {tv}: ρ − 값 = {nstr(rho - t, 12)}  ({'값 < ρ' if t < rho else '값 > ρ'})")

# ════════════════════════════════════════════════════════════════
# R2. 정확값 대수 대조 (GAP 상쇄 표현만)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("R2. 정확값 대수 대조 — GAP 상쇄 성분만 (CENTER·HCW·n_c 행·§35(n_c))")
print("=" * 78)
nu_cube = 3*sqrt(2)/4
D1nc = 1/(PHI*sqrt(2))
D2nc = PHI**2/sqrt(2)
fnc = 3*PHI/2
alg("R2-0", "3√2/4 = f(n_c)·D1(n_c)", fnc*D1nc, nu_cube, "Task 2 §P0-9 / C-10")
x_nc = 1 + n_c*GAP
D1nc_g = x_nc/sqrt(x_nc**2 + PHI**6)
f_nc_g = (PHI**3*D1nc_g)**2 - 1               # f(n_c) = D2(n_c)² − 1, §35 정의 경로
alg("R2-0b", "3√2/4 = f(n_c)·D1(n_c) (n_c·GAP 직접 경로)", f_nc_g*D1nc_g, nu_cube, "GAP 상쇄 확인")

comps = [
    ("CENTER=φ^(3/2)",  power(PHI, mpf(3)/2)),
    ("HCW_D1=φ^(-3/2)", power(PHI, mpf(-3)/2)),
    ("HCW_D2=φ^(3/2)",  power(PHI, mpf(3)/2)),
    ("D1(n_c)",         D1nc),
    ("D2(n_c)",         D2nc),
    ("f(n_c)",          fnc),
    ("ψ_re(n_c)",       re_c),
    ("ψ_im(n_c)",       im_c),
    ("|ψ_re(n_c)|²",    re_c**2),
    ("|ψ_im(n_c)|²",    im_c**2),
]
coefs = sorted({sp.Rational(p, q) for p in range(1, 7) for q in range(1, 7)})
combos = [("1", mpf(1))]
for a, va in comps:
    combos.append((a, va))
for (a, va), (b, vb) in itertools.combinations_with_replacement(comps, 2):
    combos.append((f"{a}·{b}", va*vb))
for (a, va), (b, vb) in itertools.permutations(comps, 2):
    combos.append((f"{a}/{b}", va/vb))
print(f"  성분 {len(comps)}개, 조합 {len(combos)}개 (단일·곱·비, 상수 1 포함), 유리계수 {len(coefs)}개 (p,q ≤ 6)")


def search(name, target):
    hits = []
    for lab, v in combos:
        r = target / v
        for k in coefs:
            if fabs(r - mpf(k.p)/k.q) < TOL * fabs(r):
                hits.append((str(k), lab))
    print(f"  [{name}] 대수필연 조합: {len(hits)}개")
    for k, lab in hits[:20]:
        print(f"      {name} = ({k})·{lab}")
    if len(hits) > 20:
        print(f"      … 외 {len(hits) - 20}개 (동치 표기)")
    return hits


search("3√2/4 (대조군)", nu_cube)
h_tet = search("√6/(1+√2)", sqrt(6)/(1 + sqrt(2)))
h_rho = search("ρ", rho)
print(f"  결과: √6/(1+√2) → {'없음' if not h_tet else '목록 위'},  ρ → {'없음' if not h_rho else '목록 위'}")
print("  ※ 근사(비대수필연) 조합은 보고하지 않음.")

# ════════════════════════════════════════════════════════════════
# R3. 기지 사례 데이터표 (등급 없음)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("R3. 정다면체 5 + 아르키메데스 13 데이터표 (등급 없음)")
print("=" * 78)

from polyhedra import SOLIDS, face_data  # 좌표 생성 (Task R5 와 공용 모듈)

# ν 상태: TASKS.md 제공 문헌값만 사용 (그 외 = 'TASKS 미제공')
NU = {
    "정육면체": ("정확", "3√2/4"), "정팔면체": ("정확", "3√2/4"),
    "정사면체": ("하한", "≥ √6/(1+√2)"),
    "정십이면체": ("추측", "ρ"), "정이십면체": ("추측", "ρ"),
    "마름모십이이십면체": ("비루퍼트(미감사)", "ν=1"),
    "다듬은 정육면체": ("미해결", "-"), "다듬은 정십이면체": ("미해결", "-"),
}


def plane_dist(pts, face):
    i, j, k = list(face)[:3]
    a, b, cc = (mpmath.matrix(pts[i]), mpmath.matrix(pts[j]), mpmath.matrix(pts[k]))
    u, v = b - a, cc - a
    n = mpmath.matrix([u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]])
    nn = sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    return fabs(n[0]*a[0] + n[1]*a[1] + n[2]*a[2]) / nn


def exact_form(val):
    """(r/R)² 의 최소다항식 (mpmath.findpoly, 400자리 검증) + 2차 이하면 근호 표현"""
    for deg in range(1, 13):
        cf = mpmath.findpoly(val, deg, maxcoeff=10**20, maxsteps=100000)
        if cf:
            if cf[0] < 0:
                cf = [-cc_ for cc_ in cf]
            # 가짜 관계 배제: 계수 자릿수×(차수+1) 가 400자리보다 충분히 작아야 함
            if (deg + 1) * len(str(max(abs(int(cc_)) for cc_ in cf))) > 300:
                continue
            poly = sum(sp.Integer(int(cc_)) * x**(deg - i) for i, cc_ in enumerate(cf))
            resid = mpmath.polyval([mpf(int(cc_)) for cc_ in cf], val)
            if fabs(resid) > TOL:
                continue
            fl_ = sp.factor_list(poly)
            irr = len(fl_[1]) == 1 and fl_[1][0][1] == 1
            rad = None
            if deg <= 2:
                for r_ in sp.solve(poly, x):
                    if abs(float(r_) - float(val)) < 1e-12:
                        rad = sp.nsimplify(sp.sqrt(r_))
            return poly, irr, rad, resid
    return None, None, None, None


rows = []
print(f"  {'다면체':<12} | V  | 대칭군     | 면 구성(검증)          | r/R (r=중심→최근접 면 평면)        | ν 상태")
for name, V, grp, dual, fexp, pts in SOLIDS:
    R2_ = [p[0]**2 + p[1]**2 + p[2]**2 for p in pts]
    R2v = R2_[0]
    on_sphere = all(fabs(r_ - R2v) < mpf(10)**-380 for r_ in R2_)
    faces = face_data(pts)
    fcnt = {}
    for f in faces:
        fcnt[len(f)] = fcnt.get(len(f), 0) + 1
    # 등변 검증: 최소 거리 쌍 개수 = 모서리 수 E = V + F − 2
    d2 = sorted((sum((pts[i][k] - pts[j][k])**2 for k in range(3)), i, j)
                for i in range(len(pts)) for j in range(i + 1, len(pts)))
    emin = d2[0][0]
    E = sum(1 for dd, _, _ in d2 if fabs(dd - emin) < mpf(10)**-300)
    ok = (len(pts) == V and on_sphere and fcnt == fexp and E == V + len(faces) - 2)
    r = min(plane_dist(pts, f) for f in faces)
    rR = r / sqrt(R2v)
    poly, irr, rad, resid = exact_form(rR**2)
    st, val = NU.get(name, ("TASKS 미제공", "-"))
    rows.append((name, V, grp, dual, rR, poly, irr, rad, st, val))
    fstr = " + ".join(f"{fcnt[k]}·{k}각" for k in sorted(fcnt))
    print(f"  {name:<12} | {V:<3}| {grp:<10} | {fstr:<20} {'✓' if ok else '✗검증실패'} | {nstr(rR, 20):<24} | {st} {val}")

print()
print("  (r/R)² 정확값 = 최소다항식 (mpmath.findpoly 탐지 → 400자리 대입 잔여 < 1e-390 확인 → sympy 기약성)")
for name, V, grp, dual, rR, poly, irr, rad, st, val in rows:
    rs = f"  r/R = {rad}" if rad is not None else ""
    print(f"    {name:<12}: {sp.expand(poly) if poly is not None else '미탐지(차수≤12, 계수≤1e20)'} = 0"
          f"  [기약 {irr}]{rs}")

print()
print("  쌍대 관계 / ν 일치 (목록 내 쌍대 쌍은 정다면체뿐; 아르키메데스 쌍대 = 카탈랑, 목록 밖)")
print("    정육면체 ↔ 정팔면체: ν 3√2/4 = 3√2/4 → 일치 (정확값)")
print("    정십이면체 ↔ 정이십면체: ν 추측 ρ = ρ → 일치 (추측값)")
print("    정사면체 (자기쌍대): 해당 없음")
print("  Noperthedron 90V: 좌표 미확보 (arxiv.org 접속이 네트워크 정책으로 차단, HTTP 403) → 표에서 제외")
print("  마름모십이이십면체 비루퍼트 판정 = zenodo 10.5281/zenodo.23159625, 미감사 (TASKS 표기 그대로)")

# ════════════════════════════════════════════════════════════════
# R4. 사전 예측 초안 (등록 아님)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("R4. 사전 예측 초안 (등록 아님)")
print("=" * 78)
known_R = [r_ for r_ in rows if r_[8] in ("정확", "하한", "추측")]
known_N = [r_ for r_ in rows if r_[8].startswith("비루퍼트")]
print("  기지 사례 (TASKS 제공분, 좌표 확보분):")
print(f"    루퍼트 {len(known_R)}: " + ", ".join(f"{r_[0]}(r/R={nstr(r_[4], 6)})" for r_ in known_R))
print(f"    비루퍼트 {len(known_N)}: " + ", ".join(f"{r_[0]}(r/R={nstr(r_[4], 6)})" for r_ in known_N)
      + "  (+ Noperthedron: 좌표 없음)")
print("  R2 결과: GAP 상쇄 성분의 대수필연 조합은 정육면체·정팔면체(3√2/4)만 존재 → 판정 기준 불가.")
maxR = max(r_[4] for r_ in known_R)
minN = min(r_[4] for r_ in known_N)
print(f"  R3 단일 변수 분리 가능성: max r/R(루퍼트) = {nstr(maxR, 10)} < min r/R(비루퍼트) = {nstr(minN, 10)}"
      f" → {'분리됨' if maxR < minN else '분리 안 됨'}")
if maxR < minN:
    print("  기준 후보 (초안): 'r/R < θ 이면 루퍼트', θ ∈ (max 루퍼트, min 비루퍼트)")
    print("    ⚠ 비루퍼트 표본 1개(미감사), Noperthedron 데이터 없음, 아르키메데스 11종 상태 미제공 →")
    print("      표본 내 분리일 뿐, 피팅 수준의 근거. 원칙 V/AA 상 예측력 주장 불가.")
    for nm in ("다듬은 정육면체", "다듬은 정십이면체"):
        r_ = next(rr for rr in rows if rr[0] == nm)
        side = "θ 하한 이하 → '루퍼트' 쪽" if r_[4] <= maxR else (
            "θ 상한 이상 → '비루퍼트' 쪽" if r_[4] >= minN else "구간 (max 루퍼트, min 비루퍼트) 내부 → 판정 불가")
        print(f"    초안 적용 {nm}: r/R = {nstr(r_[4], 10)} → {side}")

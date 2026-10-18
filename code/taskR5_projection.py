#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task R5 — 투영 효율(MAXWELL 정의) + 산술·기하·조화평균 부등식 (TASKS.md v9.7, 엔진 수정 없음)
근거: MAXWELL = "T² 복잡성이 L² 경계에 투영되는 효율" (v3.4 §4-H).
      λP 그림자 면적 = λ²·A, 통과에는 λ²·A(u') < A(u) 필요 → ν ≤ √(A_max/A_min).
      AM ≥ GM ≥ HM, 등호 ⇔ 모든 방향 그림자 면적 동일 (구, GAP=0 극한) → 통과 여유 0.
모든 값 = 탐색 데이터 (등급 없음). 경계값을 EQI 상수에 맞추지 않음.

계산 방법
  A(u)  : 볼록 다면체 그림자 면적 = ½ Σ_f |u·a_f|  (a_f = 면 넓이 벡터, 코시 투영 공식).
          투영 꼭짓점 2D 볼록 껍질 넓이와 무작위 방향에서 교차 검증.
  A(u) = 존소토프 Z = ½Σ[−a_f, a_f] 의 지지함수 h_Z(u).
  A_min : min_{|u|=1} h_Z = Z 의 내접 거리 = Z 면 법선(a_i × a_j) 전수에서의 최소 → 유한 전수 (정확).
  A_max : max_{|u|=1} h_Z = Z 꼭짓점 최대 노름. 고정점 반복 u ← v(u)/|v(u)|,
          v(u) = ½Σ sign(u·a_f) a_f (h 단조 증가) 를 다중 시작점에서 수행, 두 독립 시드 묶음 일치 확인.
  AM, GM, HM : 피보나치 구면 격자 평균 (N, N/4 두 해상도 차 = 적분 오차 추정).
               AM 은 코시 공식 AM = 겉넓이/4 와 대조.
Date: 2026-10-18
"""

import io
import contextlib

import numpy as np
from scipy.spatial import ConvexHull

with contextlib.redirect_stdout(io.StringIO()):
    from polyhedra import SOLIDS

CATALAN = {
    "깎은 정사면체": "삼중사면체 (triakis tetrahedron)",
    "육팔면체": "마름모십이면체 (rhombic dodecahedron)",
    "깎은 정육면체": "삼중팔면체 (triakis octahedron)",
    "깎은 정팔면체": "사중육면체 (tetrakis hexahedron)",
    "마름모육팔면체": "삼각이십사면체 (deltoidal icositetrahedron)",
    "깎은 육팔면체": "육각팔면체 (disdyakis dodecahedron)",
    "다듬은 정육면체": "오각이십사면체 (pentagonal icositetrahedron)",
    "십이이십면체": "마름모삼십면체 (rhombic triacontahedron)",
    "깎은 정십이면체": "삼중이십면체 (triakis icosahedron)",
    "깎은 정이십면체": "오각십이면체 (pentakis dodecahedron)",
    "마름모십이이십면체": "삼각육십면체 (deltoidal hexecontahedron)",
    "깎은 십이이십면체": "육각이십면체 (disdyakis triacontahedron)",
    "다듬은 정십이면체": "오각육십면체 (pentagonal hexecontahedron)",
}

# 상태 라벨 (TASKS.md, S114 perplexity — 원문/저장소 구분 유지)
NONRUPERT = {
    "마름모십이이십면체": "비루퍼트 (저장소/zenodo)",
    "다듬은 정육면체": "비루퍼트 (tom7/noperts 보고)",
    "삼각육십면체 (deltoidal hexecontahedron)": "비루퍼트 (tom7/noperts 보고)",
    "오각육십면체 (pentagonal hexecontahedron)": "비루퍼트 (tom7/noperts 보고)",
}
HOLD = {"다듬은 정십이면체"}


def label(name):
    if name in HOLD:
        return "보류"
    if name in NONRUPERT:
        return "N"
    return "R"


def label_text(name):
    if name in HOLD:
        return "보류 (사전 예측용)"
    if name in NONRUPERT:
        return NONRUPERT[name]
    return "루퍼트"


def area_vectors(pts):
    """면 넓이 벡터 (바깥 방향). 동일 평면 삼각형은 합쳐 면 하나로."""
    hull = ConvexHull(pts)
    groups = {}
    for simp, eq in zip(hull.simplices, hull.equations):
        a, b, cc = pts[simp]
        av = 0.5 * np.cross(b - a, cc - a)
        if np.dot(av, eq[:3]) < 0:
            av = -av
        key = tuple(np.round(eq, 7))
        groups.setdefault(key, [np.zeros(3), set(), eq])
        groups[key][0] = groups[key][0] + av
        groups[key][1].update(simp.tolist())
    return hull, [(g[0], g[1], g[2]) for g in groups.values()]


def merged_generators(avs):
    """평행(반대 방향) 면 넓이 벡터를 합침: |u·a| + |u·(−a)| = 2|u·a|"""
    gens = []
    for a in avs:
        n = a / np.linalg.norm(a)
        for i, g in enumerate(gens):
            gn = g / np.linalg.norm(g)
            if abs(abs(np.dot(n, gn)) - 1) < 1e-10:
                gens[i] = g + (a if np.dot(a, g) > 0 else -a)
                break
        else:
            gens.append(a.copy())
    return np.array(gens)


def A_of(U, G):
    """A(u) = ½ Σ |u·a_f|  (G 는 병합 전 전체 면 넓이 벡터 또는 병합 생성자 — 동일 값)"""
    return 0.5 * np.abs(U @ G.T).sum(axis=-1)


def hull_area_2d(pts, u):
    """검증용: u 에 수직인 평면으로 투영한 꼭짓점의 2D 볼록 껍질 넓이"""
    u = u / np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    P2 = np.c_[pts @ e1, pts @ e2]
    return ConvexHull(P2).volume


def fib_sphere(N):
    i = np.arange(N) + 0.5
    z = 1 - 2 * i / N
    r = np.sqrt(1 - z * z)
    th = np.pi * (1 + 5**0.5) * i
    return np.c_[r * np.cos(th), r * np.sin(th), z]


def sphere_means(G, N, chunk=50000):
    U = fib_sphere(N)
    sA = slnA = sinv = 0.0
    for k in range(0, N, chunk):
        A = A_of(U[k:k + chunk], G)
        sA += A.sum()
        slnA += np.log(A).sum()
        sinv += (1 / A).sum()
    return sA / N, np.exp(slnA / N), N / sinv


def A_min_exact(G):
    best = np.inf
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            n = np.cross(G[i], G[j])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            best = min(best, A_of(n / nn, G))
    return best


def A_max_fixed_point(G, seeds):
    vals = []
    for u in seeds:
        u = u / np.linalg.norm(u)
        for _ in range(200):
            s = np.sign(u @ G.T)
            s[s == 0] = 1
            v = 0.5 * (s @ G)
            un = v / np.linalg.norm(v)
            if np.allclose(un, u, atol=1e-15):
                break
            u = un
        vals.append(A_of(u, G))
    vals = np.array(vals)
    return vals.max(), int(np.sum(vals > vals.max() * (1 - 1e-12))), len(vals)


def analyze(name, kind, pts):
    pts = np.asarray(pts, dtype=float)
    hull, faces = area_vectors(pts)
    avs = [f[0] for f in faces]
    S = sum(np.linalg.norm(a) for a in avs)
    G = merged_generators(avs)
    rng = np.random.default_rng(20261018)
    # 교차 검증: 코시 공식 vs 2D 볼록 껍질
    cross = max(abs(A_of(u / np.linalg.norm(u), G) - hull_area_2d(pts, u)) / hull_area_2d(pts, u)
                for u in rng.normal(size=(20, 3)))
    a = A_min_exact(G)
    seedsA = np.vstack([fib_sphere(1500), rng.normal(size=(500, 3))])
    seedsB = np.random.default_rng(7).normal(size=(2000, 3))
    bA, hitA, nA = A_max_fixed_point(G, seedsA)
    bB, hitB, nB = A_max_fixed_point(G, seedsB)
    b = max(bA, bB)
    AM, GM, HM = sphere_means(G, 400000)
    AM4, GM4, HM4 = sphere_means(G, 100000)
    return dict(name=name, kind=kind, V=len(pts), F=len(faces), S=S, a=a, b=b,
                bA=bA, bB=bB, hitA=hitA, nA=nA, hitB=hitB, nB=nB, cross=cross,
                AM=AM, GM=GM, HM=HM, dAM=abs(AM - S / 4) / (S / 4),
                dGM=abs(GM - GM4) / GM, dHM=abs(HM - HM4) / HM,
                faces=faces)


rows = []
for name, V, grp, dual, fexp, pts in SOLIDS:
    kind = "정다면체" if name.startswith("정") else "아르키메데스"
    r = analyze(name, kind, pts)
    rows.append(r)
    if kind == "아르키메데스":
        # 카탈랑 = 극쌍대 (원점 중심 구에 대한 역수 변환; 반지름은 모양에 무관)
        dpts = [eq[:3] / (-eq[3]) for _, _, eq in r["faces"]]
        cname = CATALAN[name]
        rc = analyze(cname, "카탈랑", dpts)
        assert rc["V"] == r["F"] and rc["F"] == V, (cname, rc["V"], rc["F"])
        rows.append(rc)

print("=" * 100)
print("R5. 그림자 면적 극값·세 평균 (탐색 데이터, 등급 없음)")
print("=" * 100)
print("  수치 오차 열: cross = 코시 공식 vs 2D 볼록 껍질 최대 상대 차 (20방향)")
print("               A_max 수렴 = 시드 묶음 A/B 최댓값 상대 차, (도달 시작점 수/전체)")
print("               dAM = |AM − 겉넓이/4|/(겉넓이/4),  dGM·dHM = N=4e5 vs 1e5 상대 차")
hdr = (f"  {'다면체':<44}|{'라벨':<4}| V  | F  | {'A_min':>11} | {'A_max':>11} | {'√(Amax/Amin)':>12} | "
       f"{'AM/GM':>12} | {'GM/HM':>12} | {'c=AM₂/GM₂':>11}")
print(hdr)
for r in rows:
    a, b = r["a"], r["b"]
    r["k_root"] = np.sqrt(b / a)
    r["k_amgm"] = r["AM"] / r["GM"]
    r["k_gmhm"] = r["GM"] / r["HM"]
    r["AM2"], r["GM2"], r["HM2"] = (a + b) / 2, np.sqrt(a * b), 2 * a * b / (a + b)
    r["k_c"] = r["AM2"] / r["GM2"]
    print(f"  {r['name']:<44}|{label(r['name']):<4}| {r['V']:<3}| {r['F']:<3}| {a:11.8f} | {b:11.8f} | "
          f"{r['k_root']:12.9f} | {r['k_amgm']:12.9f} | {r['k_gmhm']:12.9f} | {r['k_c']:11.9f}")

print()
print("  상세 (A 는 외접구 반지름 R 기준 원좌표 단위; 지표는 축척 무관)")
for r in rows:
    print(f"  [{r['kind']}] {r['name']} — {label_text(r['name'])}")
    print(f"     A_min = {r['a']:.12f}   A_max = {r['b']:.12f}  (A: {r['bA']:.12f}, {r['hitA']}/{r['nA']}; "
          f"B: {r['bB']:.12f}, {r['hitB']}/{r['nB']}; 상대 차 {abs(r['bA'] - r['bB']) / r['b']:.1e})")
    print(f"     AM = {r['AM']:.10f} (겉넓이/4 = {r['S'] / 4:.10f}, dAM {r['dAM']:.1e})   "
          f"GM = {r['GM']:.10f} (dGM {r['dGM']:.1e})   HM = {r['HM']:.10f} (dHM {r['dHM']:.1e})")
    print(f"     AM ≥ GM ≥ HM: {r['AM'] >= r['GM'] >= r['HM']}   두 점: AM₂ = {r['AM2']:.10f}, "
          f"GM₂ = {r['GM2']:.10f}, HM₂ = {r['HM2']:.10f}, AM₂·HM₂/GM₂² = {r['AM2'] * r['HM2'] / r['GM2']**2:.12f}"
          f"   cross {r['cross']:.1e}")

print()
print("=" * 100)
print("판정: 지표별 루퍼트(R) / 비루퍼트(N) 완전 분리 여부 (보류 표본 제외)")
print("=" * 100)
R_ = [r for r in rows if label(r["name"]) == "R"]
N_ = [r for r in rows if label(r["name"]) == "N"]
H_ = [r for r in rows if label(r["name"]) == "보류"]
print(f"  표본: R {len(R_)}, N {len(N_)}, 보류 {len(H_)} (Noperthedron: 좌표 미확보로 제외)")
sep_any = False
for key, nm in (("k_root", "√(A_max/A_min)"), ("k_amgm", "AM/GM"), ("k_gmhm", "GM/HM"), ("k_c", "c 지표 AM₂/GM₂")):
    rv = [r[key] for r in R_]
    nv = [r[key] for r in N_]
    if max(rv) < min(nv):
        sep_any = True
        res = f"분리됨 (R < N), 간격 = min N − max R = {min(nv) - max(rv):.3e}  [{max(rv):.9f} | {min(nv):.9f}]"
    elif max(nv) < min(rv):
        sep_any = True
        res = f"분리됨 (N < R), 간격 = min R − max N = {min(rv) - max(nv):.3e}  [{max(nv):.9f} | {min(rv):.9f}]"
    else:
        ov = [r["name"] for r in R_ if min(nv) <= r[key] <= max(nv)]
        res = (f"분리 없음  (R 범위 {min(rv):.6f}~{max(rv):.6f}, N 범위 {min(nv):.6f}~{max(nv):.6f}; "
               f"N 범위 안의 R {len(ov)}개)")
    print(f"  {nm:<16}: {res}")
    for h in H_:
        print(f"      보류 {h['name']}: {h[key]:.9f}")
print(f"  결과: {'분리 지표 있음 (위 목록)' if sep_any else '없음'}")
print("  ※ 경계를 EQI 상수에 맞추지 않음. 탐색 데이터, 등급 없음.")

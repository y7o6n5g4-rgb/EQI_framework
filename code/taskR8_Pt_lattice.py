#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task R8 — 적응형 플랑크 길이-시간 격자 정보 포논 동역학으로 본 루퍼트 (Zeng 족 P_t)
(TASKS.md v9.7, 엔진 수정 없음. 첫 실행 = 장부 데이터 확보 전용, 판정·EQI 상수 대조 없음)

배치 공간 Λ (5차원 각도, 시간축):  [0,2π]² × [0,π] × [0,π/2]²
  본 구현의 성분 배정 (Zeng 원문 순서 미확보 → 아래 배정을 명시 가정으로 사용):
    d0 = α  ∈ [0,2π]   안쪽 투영의 평면 내 회전
    d1 = φ₂ ∈ [0,2π]   안쪽 투영 방향 방위각
    d2 = θ₂ ∈ [0,π]    안쪽 투영 방향 극각
    d3 = θ₁ ∈ [0,π/2]  바깥 투영 방향 극각  ┐ P_t 대칭(T_d, 짝수 부호 반전·좌표 치환)과
    d4 = φ₁ ∈ [0,π/2]  바깥 투영 방향 방위각 ┘ 대척(±n) 으로 바깥 방향을 한 팔분공간으로 축소
  회전: R_out = Ry(−θ₁)·Rz(−φ₁),  R_in = Rz(α)·Ry(−θ₂)·Rz(−φ₂),  투영 = 회전 후 (x, y)
  각 인수는 좌표축 회전 → 각도당 꼭짓점 이동 ≤ R_max·|Δ각|  (R_max = √3 = 최대 꼭짓점 노름)
길이축: 평면 평행이동 τ — 선형계획 (최소 포함 여유 최대화)
칸 판정 (격자 단계 L, 각 변 폭 = 원래 폭 × 2^(−L)):
  m_c = max_τ min_{바깥 변 k} [ b_k − max_j n_k·(q_j + τ) ]   (λ = 1, 칸 중심)
  δ   = R_max · Σ(반폭)
  m_c + 2δ < 0 → D1 (닫힘, [x]),   m_c > 0 → D2* (통과 실재),   그 외 → D2 (미결 {x}) → 2^5 분할
세분 상한: 2^(−L) < φ^(−7.5) 가 되는 L = 6 에서 정지 (X7 금지선)
장부: D1·D2·D2* 부피 비율 (합 = 1), 소거 비트 b_L = −log₂(D2_L/D2_(L−1)), 누적 비트,
      b_L·ln2 (D1 쪽 엔트로피 생성, 복식부기), tanδ_L = D2_L/D1_L, ln D1_L, ln D2_L
대상 t: 0.55 (보정: D2→0 기대), 0.590 (보정: D2* 기대), 0.575 (미지). t 당 15분 상한.
Date: 2026-10-19
"""

import itertools
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import ConvexHull

P0 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
LO = np.array([0, 0, 0, 0, 0], float)
HI = np.array([2*np.pi, 2*np.pi, np.pi, np.pi/2, np.pi/2])
W0 = HI - LO
R_MAX = math.sqrt(3.0)
PHI = (1 + 5**0.5) / 2
L_MAX = 6
TIME_CAP = 900.0
TRIPLES = np.array(list(itertools.combinations(range(16), 3)))
CHUNK = 2000
OFFS = np.array(list(itertools.product((-1, 1), repeat=5)), float)    # 32 자식 방향


def Pt(t):
    return np.vstack([P0, -t * P0])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def margin(x, V):
    """칸 중심 x 에서 λ=1 최소 포함 여유 m_c (τ 최적화 포함)"""
    al, ph2, th2, th1, ph1 = x
    A = (ry(-th1) @ rz(-ph1) @ V.T).T[:, :2]
    B = (rz(al) @ ry(-th2) @ rz(-ph2) @ V.T).T[:, :2]
    eq = ConvexHull(A).equations                  # n·x + d ≤ 0, |n| = 1
    n, b = eq[:, :2], -eq[:, 2]
    c = b - (B @ n.T).max(axis=0)                 # m + n_k·τ ≤ c_k
    M = np.column_stack([np.ones(len(n)), n])
    tr = TRIPLES[(TRIPLES < len(n)).all(axis=1)]
    Ms, cs = M[tr], c[tr]
    ok = np.abs(np.linalg.det(Ms)) > 1e-14
    sol = np.linalg.solve(Ms[ok], cs[ok][..., None])[..., 0]
    feas = (sol @ M.T <= c + 1e-12).all(axis=1)
    return sol[feas, 0].max()


def eval_chunk(args):
    t, X = args
    V = Pt(t)
    return np.array([margin(x, V) for x in X])


def run_t(t, pool, log):
    T0 = time.time()
    total_vol = 1.0
    centers = np.array([(LO + HI) / 2])
    D1 = D2s = 0.0
    D2_prev = 1.0
    cum_bits = 0.0
    ledger = []
    reached = -1
    for L in range(L_MAX + 1):
        w = W0 * 2.0**(-L)
        delta = R_MAX * float(np.sum(w / 2))
        cell_vol = 2.0**(-5 * L)                  # 정규화 부피 (전체 = 1)
        n = len(centers)
        # 시간 예산 확인 (칸당 평균 비용 추정 후 부분 처리 금지 → 단계 단위로 완결)
        chunks = [(t, centers[i:i + CHUNK]) for i in range(0, n, CHUNK)]
        ms = []
        done = 0
        aborted = False
        for res in pool.imap(eval_chunk, chunks):
            ms.append(res)
            done += len(res)
            if time.time() - T0 > TIME_CAP:
                aborted = True
                break
        if aborted:
            log(f"    L={L}: 칸 {n:,} 중 {done:,} 평가 후 15분 상한 도달 → 이 단계 미완, 도달 단계 L={reached}")
            break
        m = np.concatenate(ms)
        d1 = m + 2 * delta < 0
        d2s = m > 0
        d2 = ~(d1 | d2s)
        D1 += d1.sum() * cell_vol
        D2s += d2s.sum() * cell_vol
        D2 = d2.sum() * cell_vol
        tot = D1 + D2 + D2s
        b_L = -math.log2(D2 / D2_prev) if D2 > 0 and D2_prev > 0 else float("inf")
        cum_bits += b_L if math.isfinite(b_L) else 0.0
        tan_d = D2 / D1 if D1 > 0 else float("inf")
        reached = L
        ledger.append(dict(L=L, n=n, delta=delta, D1=D1, D2=D2, D2s=D2s, tot=tot, b=b_L, cum=cum_bits,
                           tan=tan_d, mmin=m.min(), mmax=m.max(), m_d2=m[d2] if d2.any() else np.array([]),
                           el=time.time() - T0))
        r = ledger[-1]
        log(f"    L={L} | 칸 {n:>9,} | δ={delta:.4e} | D1={D1:.6e} D2={D2:.6e} D2*={D2s:.6e} | 합={tot:.12f} | "
            f"b_L={b_L:.4f} bit, 누적 {cum_bits:.4f} bit, ×ln2 = {cum_bits * math.log(2):.4f} | "
            f"tanδ={tan_d:.4e} | ln D1={math.log(D1) if D1 > 0 else float('-inf'):.4f}, "
            f"ln D2={math.log(D2) if D2 > 0 else float('-inf'):.4f} | m_c∈[{m.min():.4e}, {m.max():.4e}] | {r['el']:.0f}s")
        D2_prev = D2
        if D2 == 0 or L == L_MAX:
            break
        kids = centers[d2][:, None, :] + OFFS[None, :, :] * (w / 4)[None, None, :]
        centers = kids.reshape(-1, 5)
    return ledger, reached


if __name__ == "__main__":
    out = []

    def log(s=""):
        print(s, flush=True)
        out.append(s)

    log("=" * 120)
    log("R8. 적응형 플랑크 길이-시간 격자 — Zeng 족 P_t 배치 공간 D1/D2 분할 장부 (첫 실행: 장부 데이터 전용)")
    log("=" * 120)
    log(f"  세분 상한 L_MAX = {L_MAX}: 2^(−6) = {2**-6:.6f} < φ^(−7.5) = {PHI**-7.5:.6f} (X7 금지선);"
        f" 2^(−5) = {2**-5:.6f} ≥ φ^(−7.5)")
    log(f"  R_max = √3, δ_L = R_max·Σ(반폭) = {R_MAX * float(np.sum(W0 / 2)):.4f}·2^(−L);  판정 D1: m_c + 2δ < 0")
    summary = {}
    for t in (0.55, 0.590, 0.575):
        # t 마다 새 작업자 풀: 상한 도달 시 미처리 작업이 다음 t 의 시간 예산을 잠식하지 않도록 종료
        pool = Pool(4)
        try:
            log("")
            log(f"  ── t = {t} " + {0.55: "(보정점: Lean 비루퍼트, D2→0 기대)",
                                    0.590: "(보정점: 루퍼트, D2* 존재 기대)",
                                    0.575: "(경계 구간 내부, 미지)"}[t])
            ledger, reached = run_t(t, pool, log)
        finally:
            pool.terminate()
            pool.join()
        summary[t] = (ledger, reached)
        last = ledger[-1]
        md2 = last["m_d2"]
        if len(md2):
            log(f"    [진단] 최종 단계 D2 칸 m_c 분포: 최대 {md2.max():.4e}, 중앙 {np.median(md2):.4e}, "
                f"|m_c| < 1e-6 인 칸 {int(np.sum(np.abs(md2) < 1e-6)):,} / {len(md2):,}")
    log("")
    log("=" * 120)
    log("  보정점 확인")
    l55, l59 = summary[0.55][0][-1], summary[0.590][0][-1]
    ok55 = l55["D2"] == 0
    ok59 = l59["D2s"] > 0
    log(f"    t=0.55 : D2 → 0 기대  → 최종 D2 = {l55['D2']:.6e} (도달 L={summary[0.55][1]})  {'기대 일치' if ok55 else '기대와 다름'}")
    log(f"    t=0.590: D2* 존재 기대 → 최종 D2* = {l59['D2s']:.6e} (도달 L={summary[0.590][1]})  {'기대 일치' if ok59 else '기대와 다름'}")
    if not (ok55 and ok59):
        log("    → 보정점이 기대와 다름: 방법 결함으로 보고, t=0.575 결과는 보류 (TASKS 지시)")
        log("      결함 후보 (구조적, 계산으로 확인 가능):")
        log("       ① 안쪽 = 바깥 (같은 방향, 또는 P_t 대칭으로 연결된 방향) 배치에서 m = 0 정확 →")
        log("          이런 점을 포함한 칸은 m_c → 0 이라 m_c + 2δ < 0 을 영원히 만족 못함 (모든 t 공통, D2 ≠ 0 하한)")
        log("       ② 통과 영역 크기: t=0.590 에서 ν−1 ≈ 7e-7 → 통과 배치 영역이 각도 폭 ≪ 칸 폭(L=6: 2π/64)")
        log("          → 칸 중심 표본으로는 D2* 를 거의 만날 수 없음")
    log("  ※ 판정·EQI 상수 대조 없음 (첫 실행 = 장부 데이터 확보 전용).")
    open("taskR8_Pt_lattice_output.txt", "w").write("\n".join(out) + "\n")

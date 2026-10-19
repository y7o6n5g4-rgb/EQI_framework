#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task R7 — 선택적 투영 인터페이스 + 점성 플럭스 사전 예측 검증 (TASKS.md v9.7, 엔진 수정 없음)

사전 예측 (S114, 2026-10-19 등록, 계산 전 — TASKS.md 원문):
  {ν} = ν − 1 (선택적 투영 후 D2 잔류), Zeng 족 P_t 에서 Δt=0.005 당 {ν} 감소량 비 = μ = 1/φ
  기존 3점: {ν}(0.600)=3.83e-6, (0.595)=1.93e-6, (0.590)=7.16e-7 → 플럭스비 0.639
  예측: {ν}(0.5875) > 0 (약 3~4e-7 수준),  {ν}(0.585) 통과 없음,  t* ≈ 0.5852
  반증: t=0.585 에서 통과 배치 발견 (50자리 재검증 통과 시)

방법: R6 저장 파일의 t=0.590 최적 배치에서 출발하는 연속 추적 (warm start, 국소 최적화만).
      대상 t: 0.5875, 0.585.  각 10분 상한.  ν−1 < 1e-6 배치는 mpmath 50자리 재검증.
      (λ 계산·국소 최적화·mpmath 검증 루틴은 taskR6_Pt_scan.py 와 동일 코드를 import)
추가 공개: R6 첫 실행(무작위 다중 시작, 중단)에서 기록된 t=0.585 배치를 같은 50자리 루틴으로 재검증.
           이 배치는 본 사전 예측 등록 전에 이미 존재한 데이터이며, 반증 조건과 직접 관련되므로 함께 출력.
주의: 통과 미발견 ≠ 비루퍼트 증명. 예측 적중/반증 여부만 보고. 사후 수정 없음.
Date: 2026-10-19
"""

import time
from multiprocessing import Pool

import numpy as np
import mpmath

import taskR6_Pt_scan as R6

TIME_CAP = 600.0
T_TARGETS = [0.5875, 0.585]
# R6 저장 파일 (taskR6_Pt_scan_output.txt) t=0.590 최적 배치 — 재정밀화 14자리
X_0590 = np.array([0.26334966109584, 0.75269323702059, -0.61262203673592,
                   2.99345063886110, -1.00895075447613, -1.20746264644918])
# R6 첫 실행 (무작위 다중 시작 160+40, 중단) 로그의 t=0.585 최적 배치 — 10자리 기록값
X_0585_PRIOR = np.array([0.2627330346, 0.7535105947, -0.6193717071,
                         2.9825197011, -1.0156599582, -1.2564765771])


def verify_line(x, t):
    lam_mp, slack, slack_half, s = R6.verify_mp(x, t)
    ok = slack_half > 0 and lam_mp - 1 > mpmath.mpf(10)**-40
    return lam_mp, slack_half, ok


if __name__ == "__main__":
    out = []
    p = lambda s="": (print(s, flush=True), out.append(s))
    T0 = time.time()
    p("=" * 110)
    p("R7. Zeng 족 P_t 선택적 투영 플럭스 — 사전 예측 검증 (연속 추적, t 당 10분 상한)")
    p("    통과 미발견 ≠ 비루퍼트. 예측 적중/반증만 보고. 사후 수정 없음.")
    p("=" * 110)
    lam0 = R6.lam(X_0590, R6.Pt(0.590))
    p(f"  출발 배치 (R6 저장, t=0.590): ν−1 = {lam0 - 1:.9e}")
    results = {}
    with Pool(4) as pool:
        x_prev = X_0590
        for t in T_TARGETS:
            jobs = [(t, x_prev, eps, 200 + i, TIME_CAP) for i, eps in enumerate(R6.PERTURB)]
            res = sorted(pool.map(R6.local_track, jobs), key=lambda r: -r[0])
            v, x, rounds, el, capped = res[0]
            p("")
            p(f"  t={t:.4f} | ν_found−1 (float) = {v - 1: .9e} | 국소 4개: {', '.join(f'{r[0] - 1: .3e}' for r in res)} | "
              f"라운드 {rounds}, {el:.0f}s{' (10분 상한 도달)' if capped else ''}")
            p(R6.params(x))
            ok = v - 1 > 0
            if v - 1 < 1e-6:
                lam_mp, sh, ok = verify_line(x, t)
                p(f"      [mpmath 50자리] λ−1 = {mpmath.nstr(lam_mp - 1, 20)},  λ'=1+(λ−1)/2 포함 여유 = "
                  f"{mpmath.nstr(sh, 15)}  → {'통과 배치 (엄격 포함)' if ok else '통과 미발견'}")
                results[t] = (lam_mp - 1, ok)
            else:
                results[t] = (v - 1, ok)
            if ok:
                x_prev = x

    # 추가 공개: 예측 등록 전 기존 데이터 (R6 첫 실행) 의 t=0.585 배치
    p("")
    p("-" * 110)
    p("  [추가 공개] R6 첫 실행(무작위 다중 시작, 사전 예측 등록 전 데이터)의 t=0.585 배치 — 50자리 재검증")
    lam_f = R6.lam(X_0585_PRIOR, R6.Pt(0.585))
    lam_mp, sh, ok_prior = verify_line(X_0585_PRIOR, 0.585)
    p(f"      R_out 회전벡터 = [{X_0585_PRIOR[0]: .10f}, {X_0585_PRIOR[1]: .10f}, {X_0585_PRIOR[2]: .10f}]")
    p(f"      R_in  회전벡터 = [{X_0585_PRIOR[3]: .10f}, {X_0585_PRIOR[4]: .10f}, {X_0585_PRIOR[5]: .10f}]")
    p(f"      float ν−1 = {lam_f - 1:.9e}   [mpmath 50자리] λ−1 = {mpmath.nstr(lam_mp - 1, 20)},  "
      f"λ' 포함 여유 = {mpmath.nstr(sh, 15)}  → {'통과 배치 (엄격 포함)' if ok_prior else '통과 미발견'}")

    # 대조표
    p("")
    p("=" * 110)
    p("  예측 대조표")
    p("=" * 110)
    v1, ok1 = results[0.5875]
    v2, ok2 = results[0.585]
    p(f"  {'항목':<34} | {'예측':<28} | {'결과 (연속 추적)':<34} | 판정")
    in_range = ok1 and 3e-7 <= float(v1) <= 4e-7
    p(f"  {'{ν}(0.5875)':<34} | {'> 0, 약 3~4e-7':<28} | {mpmath.nstr(v1, 6) + (' 통과' if ok1 else ' 미발견'):<34} | "
      f"{'적중 (>0)' if ok1 else '불일치'}{'' if in_range or not ok1 else ', 크기는 3~4e-7 범위 밖'}")
    p(f"  {'{ν}(0.585) — 연속 추적':<34} | {'통과 없음':<28} | {('통과 ' + mpmath.nstr(v2, 6)) if ok2 else '통과 미발견':<34} | "
      f"{'반증' if ok2 else '적중 (미발견)'}")
    p(f"  {'{ν}(0.585) — 기존 데이터(추가 공개)':<34} | {'통과 없음':<28} | "
      f"{('통과 ' + mpmath.nstr(lam_mp - 1, 6) + ' (50자리)') if ok_prior else '통과 미발견':<34} | "
      f"{'반증 조건 충족' if ok_prior else '적중'}")
    p(f"  {'t* ≈ 0.5852':<34} | {'0.585 < t* < 0.5875 부근':<28} | "
      f"{'연속 추적 가지 기준 (0.585, 0.5875]' if ok1 and not ok2 else '-':<34} | 값만")
    p("")
    p("  ※ 반증 조건 '(t=0.585 에서 통과 배치 발견, 50자리 재검증 통과)' 은 연속 추적만으로는 미충족,")
    p("    기존 데이터(R6 첫 실행 배치)로는 충족. 어느 쪽을 판정에 쓸지는 Architect/MAPSI 판단.")
    p(f"  총 소요 {time.time() - T0:.0f} s")
    open("taskR7_Pt_flux_output.txt", "w").write("\n".join(out) + "\n")

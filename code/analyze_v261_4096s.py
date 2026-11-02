#!/usr/bin/env python3
"""
v26.1 조셉슨 터널링 4096초 JSON 분석기
GW170814 4096초 vs 32초 비교 분석
"""
import json
import numpy as np
import sys
import os

PHI       = (np.sqrt(5) + 1) / 2
GAP       = np.log(25) / 100
W_H       = PHI**(-2)
W_L       = PHI**(-1)
TIP_POINT = PHI**1.5 - 1
n_c       = TIP_POINT / GAP   # 32.874

# 32초 기준값 (H-L)
REF_32S = {
    "tunnel_gap_min_t": 1.0,
    "tunnel_gap_min_val": 0.1373,
    "K_dev_min_t": 12.5,
    "phi_dev_min_t": 30.5,
    "n_max_t": 1.0,
    "n_max_val": 2.87,
}

def zone(n):
    if n < 0:       return "ZONE-X"
    elif n < n_c:   return "ZONE-0/1"
    elif n < 2*n_c: return "ZONE-2+"
    else:           return "ZONE-3+"

def analyze(filepath):
    print(f"\n{'='*70}")
    print("  v26.1 조셉슨 터널링 4096초 분석기")
    print(f"{'='*70}")
    print(f"  파일: {os.path.basename(filepath)}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    chunks = data.get("chunk_results", [])
    n_chunks = len(chunks)
    fs = data.get("fs", 4096)
    chunk_size = data.get("chunk_size", 2048)
    dt = chunk_size / fs  # 청크당 시간(초)

    print(f"  총 청크: {n_chunks:,}개")
    print(f"  청크 크기: {chunk_size} samples ({dt:.3f}s/chunk)")
    print(f"  총 시간: {n_chunks * dt:.1f}초")
    print(f"  fs: {fs} Hz")

    # 배열 추출
    tgap = np.array([c["tunnel_gap"] for c in chunks])
    kdev = np.array([c["K_dev"]      for c in chunks])
    pdev = np.array([c["phi_dev"]    for c in chunks])
    ln2r = np.array([c["ln2_ratio"]  for c in chunks])
    n4   = np.array([c["eqi4_n"]     for c in chunks])
    sig  = np.array([c["eqi4_sigma"] for c in chunks])
    t_ax = np.arange(n_chunks) * dt  # 시간축 (초)

    # 합병 시각 (4096초 데이터에서 합병 위치)
    # GW170814 GPS: 1185389807, 데이터 시작 기준
    t_merger = data.get("merger_offset_s", None)

    print(f"\n{'─'*70}")
    print("  [1] tunnel_gap 통계")
    print(f"{'─'*70}")
    tg_min_idx = np.argmin(tgap)
    tg_max_idx = np.argmax(tgap)
    print(f"  최솟값: {tgap[tg_min_idx]:.6f}  t={t_ax[tg_min_idx]:.2f}s (chunk {tg_min_idx})")
    print(f"  최댓값: {tgap[tg_max_idx]:.6f}  t={t_ax[tg_max_idx]:.2f}s")
    print(f"  평균:   {tgap.mean():.6f} ± {tgap.std():.6f}")
    print(f"  [32초 기준] 최솟값 t={REF_32S['tunnel_gap_min_t']}s  val={REF_32S['tunnel_gap_min_val']}")

    print(f"\n{'─'*70}")
    print("  [2] K_dev 통계")
    print(f"{'─'*70}")
    kd_min_idx = np.argmin(kdev)
    print(f"  최솟값: {kdev[kd_min_idx]:.6f}  t={t_ax[kd_min_idx]:.2f}s (chunk {kd_min_idx})")
    print(f"  평균:   {kdev.mean():.6f} ± {kdev.std():.6f}")
    print(f"  [32초 기준] K_dev 최솟값 t={REF_32S['K_dev_min_t']}s")

    print(f"\n{'─'*70}")
    print("  [3] n_ouroboros (eqi4_n) 분포")
    print(f"{'─'*70}")
    n_max_idx = np.argmax(n4)
    n_min_idx = np.argmin(n4)
    print(f"  최댓값: {n4[n_max_idx]:.4f}  t={t_ax[n_max_idx]:.2f}s (chunk {n_max_idx})")
    print(f"  최솟값: {n4[n_min_idx]:.4f}  t={t_ax[n_min_idx]:.2f}s")
    print(f"  평균:   {n4.mean():.4f} ± {n4.std():.4f}")
    print(f"  n_c:    {n_c:.3f}")
    print(f"  [32초 기준] n 최대 t={REF_32S['n_max_t']}s  val={REF_32S['n_max_val']}")

    # ZONE 분포
    zones = [zone(n) for n in n4]
    for z in ["ZONE-X", "ZONE-0/1", "ZONE-2+", "ZONE-3+"]:
        cnt = zones.count(z)
        print(f"  {z:12s}: {cnt:5d}청크 ({cnt/n_chunks*100:.1f}%)")

    print(f"\n{'─'*70}")
    print("  [4] phi_dev 통계")
    print(f"{'─'*70}")
    pd_min_idx = np.argmin(pdev)
    print(f"  최솟값: {pdev[pd_min_idx]:.6f}  t={t_ax[pd_min_idx]:.2f}s (chunk {pd_min_idx})")
    print(f"  평균:   {pdev.mean():.6f} ± {pdev.std():.6f}")
    print(f"  [32초 기준] phi_dev 최솟값 t={REF_32S['phi_dev_min_t']}s")

    print(f"\n{'─'*70}")
    print("  [5] 합병 전후 구간 분석")
    print(f"{'─'*70}")

    # 합병 전/후 구간 분할 (데이터 중간 기준)
    mid = n_chunks // 2
    # 32초 데이터에서 합병은 t≈15.5s → 4096초에서는 마지막 구간
    # 합병 전: 처음 ~4080초, 합병 후: 마지막 ~16초
    pre_end = max(0, n_chunks - 32)  # 합병 전 (약 마지막 16초 제외)
    post_start = pre_end

    tg_pre  = tgap[:pre_end]
    tg_post = tgap[post_start:]
    n_pre   = n4[:pre_end]
    n_post  = n4[post_start:]

    print(f"  인스파이럴 구간 (0~{t_ax[pre_end-1]:.0f}s, {pre_end}청크):")
    print(f"    tunnel_gap 평균: {tg_pre.mean():.6f}")
    print(f"    n_ouro 평균:     {n_pre.mean():.4f}")
    print(f"    tunnel_gap 최소: {tg_pre.min():.6f}  t={t_ax[np.argmin(tg_pre)]:.2f}s")

    print(f"  합병 구간 ({t_ax[post_start]:.0f}s~끝, {len(tg_post)}청크):")
    print(f"    tunnel_gap 평균: {tg_post.mean():.6f}")
    print(f"    n_ouro 평균:     {n_post.mean():.4f}")
    print(f"    tunnel_gap 최소: {tg_post.min():.6f}")

    print(f"\n{'─'*70}")
    print("  [6] 시간대별 tunnel_gap 구간 평균")
    print(f"{'─'*70}")
    # 전체를 10구간으로 나눠 평균
    seg = n_chunks // 10
    for i in range(10):
        s = i * seg
        e = (i+1) * seg if i < 9 else n_chunks
        tg_seg = tgap[s:e]
        n_seg  = n4[s:e]
        t_s = t_ax[s]
        t_e = t_ax[e-1]
        tg_min_s = tg_seg.min()
        tg_min_t = t_ax[s + np.argmin(tg_seg)]
        print(f"  {t_s:7.0f}~{t_e:7.0f}s | tgap 평균={tg_seg.mean():.5f} "
              f"최소={tg_min_s:.5f}(t={tg_min_t:.0f}s) | n평균={n_seg.mean():.3f}")

    print(f"\n{'─'*70}")
    print("  [7] 32초 vs 4096초 비교 요약")
    print(f"{'─'*70}")
    print(f"  {'항목':25s} {'32초 H-L':15s} {'4096초':15s}")
    print(f"  {'─'*55}")
    print(f"  {'tunnel_gap 최솟값':25s} {REF_32S['tunnel_gap_min_val']:<15.4f} {tgap.min():<15.6f}")
    print(f"  {'tunnel_gap 최솟값 시각':25s} t={REF_32S['tunnel_gap_min_t']:<13.1f} t={t_ax[tg_min_idx]:<13.1f}")
    print(f"  {'n_ouro 최댓값':25s} {REF_32S['n_max_val']:<15.3f} {n4.max():<15.3f}")
    print(f"  {'n_ouro 최댓값 시각':25s} t={REF_32S['n_max_t']:<13.1f} t={t_ax[n_max_idx]:<13.1f}")
    print(f"  {'n_ouro 평균':25s} {'—':<15s} {n4.mean():<15.4f}")
    print(f"  {'n_c 기준 비율':25s} {'—':<15s} {n4.mean()/n_c*100:<15.1f}%")

    print(f"\n{'─'*70}")
    print("  [8] EQI 상수 연결")
    print(f"{'─'*70}")
    print(f"  GAP      = {GAP:.6f}")
    print(f"  n_c      = {n_c:.4f}")
    print(f"  W_H      = φ^(-2) = {W_H:.6f}")
    print(f"  W_L      = φ^(-1) = {W_L:.6f}")
    print(f"  K_dev target = W_H = {W_H:.6f}")
    print(f"  tunnel_gap 이론 최솟값: 0 (완전 쿠퍼쌍 응집)")

    print(f"\n{'='*70}")
    print("  분석 완료")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        import glob
        files = glob.glob("eqi_monster_v26_1_boltzmann_*.json")
        if files:
            # 가장 큰 파일 (4096초 JSON)
            bg = max(files, key=os.path.getsize)
            print(f"ℹ️  가장 큰 JSON 선택: {bg}")
            analyze(bg)
        else:
            print("사용법: python analyze_v261_4096s.py [json파일명]")

#!/usr/bin/env python3
"""
EQI Monster v89_faraday JSON 분석기
배경 데이터 + GW 이벤트 JSON 비교 분석

사용법:
  # 단일 파일 분석
  python analyze_faraday_json.py eqi_monster_v89_faraday_XXXX.json

  # 다중 파일 비교 (배경 + 이벤트)
  python analyze_faraday_json.py *.json

  # 레이블 지정
  python analyze_faraday_json.py file1.json:SGWB file2.json:GW150914 file3.json:GW170817
"""

import json
import numpy as np
import sys
import os
import glob

PHI    = (1 + np.sqrt(5)) / 2
GAP    = np.log(25) / 100
W_L    = PHI**(-1)
W_H    = PHI**(-2)
MAXWELL = 1.8731137721991709

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_arrays(data):
    """chunk_results에서 지표별 배열 추출"""
    chunks = data.get("chunk_results", [])
    if not chunks:
        return None

    keys = [
        "time_s", "G_ratio", "grid_balance", "SH_ratio",
        "faraday_exchange", "faraday_abs",
        "psi_re", "psi_im", "born_rule", "phase_angle_deg",
        "psi_center_dist", "viscosity_global", "boltzmann_local",
        "E_topo_norm", "c_mem_ratio", "ln2_ratio", "eqi4_n", "eqi4_sigma"
    ]
    arrays = {}
    for k in keys:
        try:
            arrays[k] = np.array([c.get(k, np.nan) for c in chunks])
        except:
            arrays[k] = np.full(len(chunks), np.nan)
    return arrays

def stat_line(arr, name, ideal=None, unit=""):
    """통계 한 줄 출력"""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print(f"  {name:<22} : 데이터 없음")
        return
    mn, mx, avg, sd = valid.min(), valid.max(), valid.mean(), valid.std()
    line = f"  {name:<22} : avg={avg:.6f}  std={sd:.6f}  min={mn:.6f}  max={mx:.6f}"
    if ideal is not None:
        dev = abs(avg - ideal)
        line += f"  [이상={ideal:.4f} 편차={dev:.6f}]"
    if unit:
        line += f" {unit}"
    print(line)

def find_convergence(arr, times, target, mode="min_abs_diff"):
    """수렴 시각 탐색"""
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.nan, np.nan
    if mode == "min_abs_diff":
        idx = np.argmin(np.abs(arr[valid] - target))
        idxs = np.where(valid)[0][idx]
    elif mode == "max":
        idxs = np.nanargmax(arr)
    elif mode == "min":
        idxs = np.nanargmin(arr)
    return times[idxs], arr[idxs]

def analyze_single(data, label=""):
    arr = extract_arrays(data)
    if arr is None:
        print(f"  [{label}] chunk_results 없음")
        return {}

    t    = arr["time_s"]
    n    = len(t)
    T    = t[-1] if n > 0 else 0
    fs_c = t[1] - t[0] if n > 1 else 0.5  # chunk 간격(초)

    print(f"\n{'='*72}")
    print(f"  📊 [{label}]  총 {n} chunks  |  시간범위: {t[0]:.1f}s ~ {T:.1f}s")
    print(f"{'='*72}")

    # ── §1 우주 고유 상수 ───────────────────────────────────────
    print(f"\n[§1] 우주 고유 상수")
    stat_line(arr["G_ratio"],    "G_ratio",    ideal=W_L)
    gr_err = np.abs(arr["G_ratio"] - W_L) / W_L * 100
    print(f"  {'G_ratio 오차%':<22} : avg={gr_err.mean():.8f}%  max={gr_err.max():.8f}%")
    stat_line(arr["born_rule"],  "born_rule",  ideal=1.0)
    br_err = np.abs(arr["born_rule"] - 1.0)
    print(f"  {'born_rule 편차':<22} : avg={br_err.mean():.2e}  max={br_err.max():.2e}")

    # ── §2 격자 균형 (상대론) ──────────────────────────────────
    print(f"\n[§2] 상대론 격자 균형 (L²↔T²)")
    stat_line(arr["grid_balance"],  "grid_balance", ideal=1.0)
    stat_line(arr["SH_ratio"],      "SH_ratio",     ideal=PHI)
    stat_line(arr["phase_angle_deg"],"phase_angle°", ideal=45.0, unit="°")

    t_gb, v_gb = find_convergence(arr["grid_balance"], t, 1.0, "max")
    print(f"  {'grid_balance→1.0':<22} : t={t_gb:.2f}s  값={v_gb:.6f}")

    t_ph, v_ph = find_convergence(arr["phase_angle_deg"], t, 45.0, "min_abs_diff")
    print(f"  {'phase→45°':<22} : t={t_ph:.2f}s  값={v_ph:.3f}°  편차={abs(v_ph-45):.3f}°")

    # ── §3 패러데이 교환 ───────────────────────────────────────
    print(f"\n[§3] 패러데이 교환 효율 (MAXWELL={MAXWELL:.6f})")
    stat_line(arr["faraday_abs"],    "faraday_abs",  ideal=0.0)
    t_fe, v_fe = find_convergence(arr["faraday_abs"], t, 0.0, "min")
    print(f"  {'faraday_abs 최소':<22} : t={t_fe:.2f}s  값={v_fe:.6f}")
    stat_line(arr["faraday_exchange"],"faraday_exch")

    # ── §4 복소평면 상전이 ─────────────────────────────────────
    print(f"\n[§4] 복소평면 0→√φ 상전이")
    stat_line(arr["psi_re"],  "psi_re",  ideal=PHI**(-1.5))
    stat_line(arr["psi_im"],  "psi_im",  ideal=PHI**(+1.5))

    # ── §5 점성-복원력 켤레 ────────────────────────────────────
    print(f"\n[§5] 점성-복원력 켤레")
    stat_line(arr["viscosity_global"], "viscosity_global", ideal=W_H)
    stat_line(arr["boltzmann_local"],  "boltzmann_local",  ideal=1.0)
    t_bl, v_bl = find_convergence(arr["boltzmann_local"], t, 1.0, "max")
    print(f"  {'boltzmann→1.0':<22} : t={t_bl:.2f}s  값={v_bl:.8f}")

    # ── §6 수렴 시각 통합 ─────────────────────────────────────
    print(f"\n[§6] 수렴 시각 통합")
    t_3종 = t_gb  # grid/faraday/phase 동시
    delta_t = t_3종 - t_bl
    print(f"  boltzmann 수렴     : t={t_bl:.2f}s")
    print(f"  신규 3종 수렴      : t={t_3종:.2f}s")
    print(f"  Δt (3종-bolt)     : {delta_t:.2f}s")
    if T > 0:
        print(f"  볼츠만 위치       : {t_bl/T*100:.1f}% (전체 {T:.0f}s)")
        print(f"  3종 수렴 위치     : {t_3종/T*100:.1f}%")
        # φ² 분할 확인
        phi2_t = T / PHI**2
        phi1_t = T / PHI
        print(f"  T/φ²={phi2_t:.2f}s  T/φ={phi1_t:.2f}s  (볼츠만 오차: {abs(t_bl-phi2_t):.2f}s)")

    # ── §7 t=2.5s HCW flip ────────────────────────────────────
    print(f"\n[§7] t=2.5s HCW flip 구간")
    mask_25 = np.abs(t - 2.5) < fs_c
    if mask_25.any():
        i25 = np.where(mask_25)[0][0]
        print(f"  chunk {i25}  t={t[i25]:.2f}s")
        for k in ["grid_balance","faraday_abs","phase_angle_deg","born_rule","boltzmann_local"]:
            v = arr[k][i25]
            print(f"    {k:<22} = {v:.6f}")
    else:
        print(f"  t=2.5s chunk 없음 (chunk 간격={fs_c:.1f}s)")

    # ── §8 φ 구조 분석 (배경 데이터 특화) ─────────────────────
    if T > 100:  # 배경 데이터일 때만
        print(f"\n[§8] φ 시간 분할 구조 (배경 전용)")
        for n_phi, label_phi in [(1,"T/φ"),(2,"T/φ²"),(3,"T/φ³"),(4,"T/φ⁴")]:
            pt = T / PHI**n_phi
            # 해당 시각 근처 chunk
            idx = np.argmin(np.abs(t - pt))
            gb_v = arr["grid_balance"][idx]
            bl_v = arr["boltzmann_local"][idx]
            fa_v = arr["faraday_abs"][idx]
            print(f"  {label_phi:<8} = {pt:8.2f}s  "
                  f"grid_bal={gb_v:.4f}  bolt={bl_v:.6f}  farad={fa_v:.4f}")

    # 반환값
    return {
        "label": label,
        "n_chunks": n,
        "T_total": T,
        "G_ratio_avg": float(np.nanmean(arr["G_ratio"])),
        "G_ratio_err_pct": float(gr_err.mean()),
        "born_rule_avg": float(np.nanmean(arr["born_rule"])),
        "grid_balance_avg": float(np.nanmean(arr["grid_balance"])),
        "grid_balance_max": float(np.nanmax(arr["grid_balance"])),
        "phase_avg": float(np.nanmean(arr["phase_angle_deg"])),
        "faraday_abs_avg": float(np.nanmean(arr["faraday_abs"])),
        "faraday_abs_min": float(np.nanmin(arr["faraday_abs"])),
        "boltzmann_avg": float(np.nanmean(arr["boltzmann_local"])),
        "t_boltzmann": float(t_bl),
        "t_3종수렴": float(t_3종),
        "delta_t": float(delta_t),
        "t25_grid_balance": float(arr["grid_balance"][np.argmin(np.abs(t-2.5))]) if (np.abs(t-2.5)<fs_c).any() else np.nan,
        "t25_phase": float(arr["phase_angle_deg"][np.argmin(np.abs(t-2.5))]) if (np.abs(t-2.5)<fs_c).any() else np.nan,
    }

def compare_all(results):
    """다중 파일 비교표"""
    if len(results) < 2:
        return
    print(f"\n{'='*90}")
    print("  📋 전체 비교표")
    print(f"{'='*90}")

    # 헤더
    print(f"  {'레이블':<18} {'chunks':>6} {'bolt_t':>8} {'3종_t':>8} "
          f"{'Δt':>8} {'gb_avg':>8} {'phase_avg':>9} {'t2.5_gb':>8} {'t2.5_°':>8}")
    print(f"  {'-'*88}")

    for r in results:
        if not r:
            continue
        print(f"  {r['label']:<18} "
              f"{r['n_chunks']:>6} "
              f"{r['t_boltzmann']:>8.1f} "
              f"{r['t_3종수렴']:>8.1f} "
              f"{r['delta_t']:>8.1f} "
              f"{r['grid_balance_avg']:>8.4f} "
              f"{r['phase_avg']:>9.3f}° "
              f"{r.get('t25_grid_balance', float('nan')):>8.4f} "
              f"{r.get('t25_phase', float('nan')):>8.3f}°")

    print(f"\n  [공통 확인]")
    gr_errs = [r['G_ratio_err_pct'] for r in results if r]
    br_avgs = [r['born_rule_avg'] for r in results if r]
    print(f"  G_ratio 오차%   : {[f'{e:.8f}%' for e in gr_errs]}")
    print(f"  born_rule 평균  : {[f'{b:.8f}' for b in br_avgs]}")

    print(f"\n  [Δt 분류]")
    for r in results:
        if not r:
            continue
        dt = r['delta_t']
        if dt < 10:
            cat = "BBH 단순"
        elif dt < 20:
            cat = "BNS/3Det"
        elif dt < 100:
            cat = "복합"
        else:
            cat = "배경 SGWB"
        print(f"  {r['label']:<18} Δt={dt:>8.1f}s  → [{cat}]")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # 현재 디렉토리에서 자동 탐색
        files = sorted(glob.glob("eqi_monster_v89_faraday_*.json"))
        if not files:
            print("❌ JSON 파일 없음. 사용법:")
            print("   python analyze_faraday_json.py [file.json] ...")
            sys.exit(1)
        print(f"📂 자동 탐색: {len(files)}개 파일 발견")
        labeled = [(f, os.path.basename(f)[:40]) for f in files]
    else:
        labeled = []
        for arg in args:
            if ":" in arg:
                path, lbl = arg.split(":", 1)
            else:
                path = arg
                # 파일명에서 레이블 추출 시도
                bn = os.path.basename(path)
                for kw in ["GW150914","GW170817","GW170814","SGWB","sgwb","background"]:
                    if kw.lower() in bn.lower():
                        lbl = kw; break
                else:
                    lbl = bn[:20]
            labeled.append((path, lbl))

    print("="*72)
    print("  EQI Monster v89_faraday — JSON 분석기")
    print(f"  PHI={PHI:.6f}  GAP={GAP:.6f}  MAXWELL={MAXWELL:.6f}")
    print("="*72)

    all_results = []
    for path, lbl in labeled:
        if not os.path.exists(path):
            print(f"⚠️ 파일 없음: {path}")
            all_results.append(None)
            continue
        size_mb = os.path.getsize(path) / 1e6
        print(f"\n📁 로드: {path}  ({size_mb:.1f} MB)")
        try:
            data = load_json(path)
            r = analyze_single(data, label=lbl)
            all_results.append(r)
        except Exception as ex:
            print(f"❌ 오류: {ex}")
            all_results.append(None)

    # 다중 파일 비교
    valid = [r for r in all_results if r]
    if len(valid) >= 2:
        compare_all(valid)

    print(f"\n✅ 분석 완료")

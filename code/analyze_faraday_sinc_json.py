#!/usr/bin/env python3
"""
EQI Monster v89_faraday_sinc JSON 분석기 v2
sinc 지표 추가 + 6채널 통합 비교

사용법:
  python analyze_faraday_sinc_json.py file.json:레이블 ...
  python analyze_faraday_sinc_json.py *.json
"""

import json
import numpy as np
import sys
import os
import glob

PHI    = (1 + np.sqrt(5)) / 2
GAP    = np.log(25) / 100
GAP_D2 = np.log(5) / 100
W_L    = PHI**(-1)
W_H    = PHI**(-2)
MAXWELL = 1.8731137721991709
C_LN_PI = np.log(np.pi)
SINC_F  = 3/np.pi           # sinc(π/6)×π = 3 → sinc(π/6) = 3/π
QEC_OPT = 1 - 3/np.pi       # f오비탈 최적 QEC 열림 = 0.04507

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_arrays(data):
    chunks = data.get("chunk_results", [])
    if not chunks:
        return None
    keys = [
        "time_s", "G_ratio", "grid_balance", "SH_ratio",
        "faraday_exchange", "faraday_abs",
        "psi_re", "psi_im", "born_rule", "phase_angle_deg",
        "psi_center_dist", "viscosity_global", "boltzmann_local",
        "E_topo_norm", "sinc_H", "sinc_H_abs", "qec_openness"
    ]
    arrays = {}
    for k in keys:
        try:
            arrays[k] = np.array([c.get(k, np.nan) for c in chunks])
        except:
            arrays[k] = np.full(len(chunks), np.nan)
    return arrays

def find_convergence(arr, times, target, mode="min_abs_diff"):
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.nan, np.nan
    if mode == "min_abs_diff":
        idx  = np.argmin(np.abs(arr[valid] - target))
        idxs = np.where(valid)[0][idx]
    elif mode == "max":
        idxs = np.nanargmax(arr)
    elif mode == "min":
        idxs = np.nanargmin(arr)
    return times[idxs], arr[idxs]

def stat_line(arr, name, ideal=None):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print(f"  {name:<25} : 데이터 없음")
        return
    mn, mx, avg, sd = valid.min(), valid.max(), valid.mean(), valid.std()
    line = f"  {name:<25} : avg={avg:.6f}  std={sd:.6f}  min={mn:.6f}  max={mx:.6f}"
    if ideal is not None:
        line += f"  [이상={ideal:.4f} 편차={abs(avg-ideal):.6f}]"
    print(line)

def analyze_single(data, label=""):
    arr = extract_arrays(data)
    if arr is None:
        print(f"  [{label}] chunk_results 없음")
        return {}

    t   = arr["time_s"]
    n   = len(t)
    T   = t[-1] if n > 0 else 0
    fs_c = t[1]-t[0] if n > 1 else 0.5

    print(f"\n{'='*72}")
    print(f"  📊 [{label}]  총 {n} chunks  |  시간범위: {t[0]:.1f}s ~ {T:.1f}s")
    print(f"{'='*72}")

    # §1 우주 고유 상수
    print(f"\n[§1] 우주 고유 상수")
    stat_line(arr["G_ratio"],   "G_ratio",   ideal=W_L)
    gr_err = np.abs(arr["G_ratio"]-W_L)/W_L*100
    print(f"  {'G_ratio 오차%':<25} : avg={gr_err.mean():.8f}%  max={gr_err.max():.8f}%")
    stat_line(arr["born_rule"], "born_rule",  ideal=1.0)

    # §2 격자 균형
    print(f"\n[§2] 상대론 격자 균형")
    stat_line(arr["grid_balance"],   "grid_balance",   ideal=1.0)
    stat_line(arr["phase_angle_deg"],"phase_angle°",   ideal=45.0)
    t_gb, v_gb = find_convergence(arr["grid_balance"], t, 1.0, "max")
    t_ph, v_ph = find_convergence(arr["phase_angle_deg"], t, 45.0, "min_abs_diff")
    print(f"  {'grid_balance→1.0':<25} : t={t_gb:.2f}s  값={v_gb:.6f}")
    print(f"  {'phase→45°':<25} : t={t_ph:.2f}s  값={v_ph:.3f}°")

    # §3 패러데이
    print(f"\n[§3] 패러데이 교환")
    stat_line(arr["faraday_abs"], "faraday_abs", ideal=0.0)
    t_fe, v_fe = find_convergence(arr["faraday_abs"], t, 0.0, "min")
    print(f"  {'faraday_abs 최소':<25} : t={t_fe:.2f}s  값={v_fe:.6f}")

    # §4 볼츠만
    print(f"\n[§4] 점성-복원력 켤레")
    stat_line(arr["boltzmann_local"], "boltzmann_local", ideal=1.0)
    t_bl, v_bl = find_convergence(arr["boltzmann_local"], t, 1.0, "max")
    print(f"  {'boltzmann→1.0':<25} : t={t_bl:.2f}s  값={v_bl:.8f}")

    # §5 sinc 지표 (신규)
    print(f"\n[§5] sinc 지표 — D2→D1 수렴 / QEC 열림")
    print(f"  [이론] sinc(π/6)×π=3 = |ln(GAP)|/ln(π)  (f오비탈, 현재 우주)")
    print(f"  [이론] QEC 최적 열림 = 1-3/π = {QEC_OPT:.5f}  (f오비탈 3D)")

    if not np.all(np.isnan(arr["sinc_H_abs"])):
        stat_line(arr["sinc_H_abs"],   "sinc_H_abs",   ideal=1.0)
        stat_line(arr["qec_openness"], "qec_openness", ideal=QEC_OPT)
        t_sinc, v_sinc = find_convergence(arr["sinc_H_abs"], t, 1.0, "max")
        t_qec,  v_qec  = find_convergence(arr["qec_openness"], t, 0.0, "min")
        print(f"  {'sinc_H→1.0 최근접':<25} : t={t_sinc:.2f}s  값={v_sinc:.6f}")
        print(f"  {'QEC열림→0 최소':<25} : t={t_qec:.2f}s  값={v_qec:.6f}")
        delta_sinc_3jong = t_sinc - t_gb
        print(f"  {'Δ(sinc-3종)':<25} : {delta_sinc_3jong:+.2f}s")
        # Δ 분류
        if delta_sinc_3jong > 2:
            cat = "BBH형 (QEC 늦게 닫힘, 링다운 2단계)"
        elif delta_sinc_3jong < -100:
            cat = "배경형 (QEC 압도적으로 먼저)"
        elif delta_sinc_3jong < -1:
            cat = "D1분기형 (QEC 선행)"
        else:
            cat = "동시형 (BNS/3Det, 완전 동기)"
        print(f"  {'유형':<25} : [{cat}]")
    else:
        t_sinc = np.nan
        print(f"  sinc 데이터 없음 (v89_faraday 버전으로 재실행 필요)")

    # §6 수렴 시각 통합
    print(f"\n[§6] 수렴 시각 통합")
    print(f"  boltzmann 수렴  : t={t_bl:.2f}s")
    print(f"  신규 3종 수렴   : t={t_gb:.2f}s")
    print(f"  sinc_H 수렴     : t={t_sinc:.2f}s" if not np.isnan(t_sinc) else f"  sinc_H 수렴     : N/A")
    delta_3_bolt = t_gb - t_bl
    print(f"  Δ(3종-bolt)    : {delta_3_bolt:.2f}s")
    if T > 0:
        print(f"  볼츠만 위치     : {t_bl/T*100:.1f}%  (T/φ²={T/PHI**2:.1f}s 오차={abs(t_bl-T/PHI**2):.1f}s)")
        print(f"  3종 위치        : {t_gb/T*100:.1f}%")
        if not np.isnan(t_sinc):
            print(f"  sinc 위치       : {t_sinc/T*100:.1f}%  (T/φ^2.05={T/PHI**2.05:.1f}s 오차={abs(t_sinc-T/PHI**2.05):.1f}s)")

    # §7 t=2.5s HCW flip
    print(f"\n[§7] t=2.5s HCW flip")
    mask = np.abs(t - 2.5) < fs_c
    if mask.any():
        i25 = np.where(mask)[0][0]
        for k, label_k in [
            ("grid_balance","grid_balance"),
            ("faraday_abs","faraday_abs"),
            ("phase_angle_deg","phase_angle_deg"),
            ("born_rule","born_rule"),
            ("boltzmann_local","boltzmann_local"),
            ("sinc_H_abs","sinc_H_abs"),
            ("qec_openness","qec_openness"),
        ]:
            v = arr[k][i25]
            print(f"    {label_k:<22} = {v:.6f}")

    # §8 φ 시간 분할 (배경 전용)
    if T > 100:
        print(f"\n[§8] φ 시간 분할 (배경)")
        for n_p, lbl_p in [(1,"T/φ"),(2,"T/φ²"),(3,"T/φ³"),(4,"T/φ⁴")]:
            pt = T / PHI**n_p
            idx = np.argmin(np.abs(t - pt))
            gb_v = arr["grid_balance"][idx]
            bl_v = arr["boltzmann_local"][idx]
            sc_v = arr["sinc_H_abs"][idx] if not np.all(np.isnan(arr["sinc_H_abs"])) else np.nan
            print(f"  {lbl_p:<8} = {pt:8.2f}s  grid_bal={gb_v:.4f}  bolt={bl_v:.6f}  sinc={sc_v:.4f}")

    return {
        "label": label, "n_chunks": n, "T_total": T,
        "G_ratio_err_pct": float(gr_err.mean()),
        "born_rule_avg": float(np.nanmean(arr["born_rule"])),
        "grid_balance_avg": float(np.nanmean(arr["grid_balance"])),
        "phase_avg": float(np.nanmean(arr["phase_angle_deg"])),
        "faraday_abs_avg": float(np.nanmean(arr["faraday_abs"])),
        "boltzmann_avg": float(np.nanmean(arr["boltzmann_local"])),
        "t_boltzmann": float(t_bl),
        "t_3종수렴": float(t_gb),
        "t_sinc": float(t_sinc) if not np.isnan(t_sinc) else None,
        "delta_3_bolt": float(delta_3_bolt),
        "delta_sinc_3jong": float(t_sinc - t_gb) if not np.isnan(t_sinc) else None,
        "t25_grid_balance": float(arr["grid_balance"][np.argmin(np.abs(t-2.5))]) if (np.abs(t-2.5)<fs_c).any() else np.nan,
        "t25_sinc_H_abs": float(arr["sinc_H_abs"][np.argmin(np.abs(t-2.5))]) if (np.abs(t-2.5)<fs_c).any() and not np.all(np.isnan(arr["sinc_H_abs"])) else np.nan,
        "t25_qec_openness": float(arr["qec_openness"][np.argmin(np.abs(t-2.5))]) if (np.abs(t-2.5)<fs_c).any() and not np.all(np.isnan(arr["qec_openness"])) else np.nan,
    }

def compare_all(results):
    if len(results) < 2:
        return
    print(f"\n{'='*95}")
    print("  📋 전체 비교표 — sinc 포함")
    print(f"{'='*95}")
    print(f"  {'레이블':<18} {'bolt_t':>7} {'3종_t':>7} {'sinc_t':>8} "
          f"{'Δ(sinc-3종)':>12} {'t2.5_gb':>8} {'t2.5_sinc':>10} {'t2.5_qec':>9}")
    print(f"  {'-'*93}")
    for r in results:
        if not r: continue
        t_sinc = r.get('t_sinc') or float('nan')
        d_s3 = r.get('delta_sinc_3jong')
        d_s3_str = f"{d_s3:+.1f}s" if d_s3 is not None else "N/A"
        t_sinc_str = f"{t_sinc:.1f}s" if not (t_sinc != t_sinc) else "N/A"
        print(f"  {r['label']:<18} "
              f"{r['t_boltzmann']:>7.1f}s "
              f"{r['t_3종수렴']:>7.1f}s "
              f"{t_sinc_str:>8} "
              f"{d_s3_str:>12} "
              f"{r.get('t25_grid_balance',float('nan')):>8.4f} "
              f"{r.get('t25_sinc_H_abs',float('nan')):>10.5f} "
              f"{r.get('t25_qec_openness',float('nan')):>9.5f}")

    print(f"\n  [공통 확인]")
    gr_errs = [r['G_ratio_err_pct'] for r in results if r]
    br_avgs = [r['born_rule_avg'] for r in results if r]
    print(f"  G_ratio 오차%  : {[f'{e:.6f}%' for e in gr_errs]}")
    print(f"  born_rule 평균 : {[f'{b:.8f}' for b in br_avgs]}")

    print(f"\n  [Δ(sinc-3종) 분류]")
    for r in results:
        if not r: continue
        d = r.get('delta_sinc_3jong')
        if d is None: cat = "N/A (sinc 없음)"
        elif d > 2:    cat = "BBH형 (QEC 늦음)"
        elif d < -100: cat = "배경형 (QEC 압도적 선행)"
        elif d < -1:   cat = "D1분기형 (QEC 선행)"
        else:          cat = "동시형 (BNS/3Det)"
        d_str = f"{d:+.1f}s" if d is not None else "N/A"
        print(f"  {r['label']:<18} Δ={d_str:>9}  → [{cat}]")

    print(f"\n  [t=2.5s sinc_H 단조 순서]")
    valid = [(r['label'], r.get('t25_sinc_H_abs',float('nan')), r.get('t25_qec_openness',float('nan')))
             for r in results if r]
    valid_sorted = sorted([v for v in valid if v[1]==v[1]], key=lambda x: x[1])
    for lbl, sinc, qec in valid_sorted:
        print(f"  {lbl:<18} sinc={sinc:.5f}  qec={qec:.5f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        files = sorted(glob.glob("eqi_monster_v89_faraday_*.json"))
        if not files:
            print("❌ JSON 파일 없음"); sys.exit(1)
        print(f"📂 자동 탐색: {len(files)}개")
        labeled = [(f, os.path.basename(f)[:35]) for f in files]
    else:
        labeled = []
        for arg in args:
            if ":" in arg:
                path, lbl = arg.split(":", 1)
            else:
                path = arg
                bn = os.path.basename(path)
                for kw in ["GW150914","GW170817","GW170814","SGWB","sgwb","background"]:
                    if kw.lower() in bn.lower():
                        lbl = kw; break
                else:
                    lbl = bn[:20]
            labeled.append((path, lbl))

    print("="*72)
    print("  EQI Monster v89_faraday_sinc — JSON 분석기 v2")
    print(f"  PHI={PHI:.6f}  MAXWELL={MAXWELL:.6f}")
    print(f"  sinc(π/6)×π=3  QEC_opt=1-3/π={QEC_OPT:.5f}")
    print("="*72)

    all_results = []
    for path, lbl in labeled:
        if not os.path.exists(path):
            print(f"⚠️ 파일 없음: {path}"); all_results.append(None); continue
        size_mb = os.path.getsize(path)/1e6
        print(f"\n📁 [{lbl}]  {path}  ({size_mb:.1f} MB)")
        try:
            data = load_json(path)
            r = analyze_single(data, label=lbl)
            all_results.append(r)
        except Exception as ex:
            import traceback; traceback.print_exc()
            all_results.append(None)

    valid = [r for r in all_results if r]
    if len(valid) >= 2:
        compare_all(valid)

    print(f"\n✅ 분석 완료")

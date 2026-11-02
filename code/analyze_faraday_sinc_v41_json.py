#!/usr/bin/env python3
"""
EQI Monster v89_faraday_sinc_v41 JSON 분석기 v3
3계층 통합 (RLC + 패러데이/sinc + v41 볼츠만) 비교 분석

사용법:
  python analyze_faraday_sinc_v41_json.py file.json:레이블 ...
  python analyze_faraday_sinc_v41_json.py *.json
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
QEC_OPT = 1 - 3/np.pi
e_const = np.e

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_arrays(data):
    chunks = data.get("chunk_results", [])
    if not chunks:
        return None
    keys = [
        "time_s", "G_ratio", "grid_balance", "faraday_abs",
        "born_rule", "phase_angle_deg", "boltzmann_local",
        "sinc_H_abs", "qec_openness",
        "nexus_signal", "casm_ratio", "curv_diff",
        "e_H", "e_H_deviation",
    ]
    arrays = {}
    for k in keys:
        try:
            arrays[k] = np.array([c.get(k, np.nan) for c in chunks])
        except:
            arrays[k] = np.full(len(chunks), np.nan)
    return arrays

def find_conv(arr, times, target, mode="min_abs_diff"):
    valid = ~np.isnan(arr)
    if not valid.any(): return np.nan, np.nan
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
        print(f"  {name:<26} : 데이터 없음"); return
    mn, mx, avg, sd = valid.min(), valid.max(), valid.mean(), valid.std()
    line = f"  {name:<26} : avg={avg:.6f}  std={sd:.6f}  min={mn:.6f}  max={mx:.6f}"
    if ideal is not None:
        line += f"  [이상={ideal:.4f} 편차={abs(avg-ideal):.6f}]"
    print(line)

def analyze_single(data, label=""):
    arr = extract_arrays(data)
    if arr is None:
        print(f"  [{label}] chunk_results 없음"); return {}

    t    = arr["time_s"]
    n    = len(t)
    T    = t[-1] if n > 0 else 0
    fs_c = t[1]-t[0] if n > 1 else 0.5

    print(f"\n{'='*75}")
    print(f"  📊 [{label}]  {n} chunks  |  {t[0]:.1f}s ~ {T:.1f}s")
    print(f"{'='*75}")

    # §1 우주 고유 상수
    print(f"\n[§1] 우주 고유 상수")
    gr_err = np.abs(arr["G_ratio"]-W_L)/W_L*100
    print(f"  G_ratio 오차%            : avg={gr_err.mean():.8f}%  max={gr_err.max():.8f}%")
    br_dev = np.abs(arr["born_rule"]-1.0)
    print(f"  born_rule 편차           : avg={br_dev.mean():.2e}  max={br_dev.max():.2e}")

    # §2 Layer A
    print(f"\n[§2] Layer A — RLC 중력 재구성")
    t_gr, v_gr = find_conv(arr["G_ratio"], t, W_L, "min")
    print(f"  G_ratio 최솟값           : t={t_gr:.2f}s")

    # §3 Layer B
    print(f"\n[§3] Layer B — 패러데이+sinc")
    t_gb, v_gb = find_conv(arr["grid_balance"], t, 1.0, "max")
    t_fe, v_fe = find_conv(arr["faraday_abs"], t, 0.0, "min")
    t_bl, v_bl = find_conv(arr["boltzmann_local"], t, 1.0, "max")
    print(f"  grid_balance→1.0        : t={t_gb:.2f}s")
    print(f"  faraday_abs 최소         : t={t_fe:.2f}s")
    print(f"  boltzmann→1.0           : t={t_bl:.2f}s")

    has_sinc = not np.all(np.isnan(arr["sinc_H_abs"]))
    if has_sinc:
        t_sinc, _ = find_conv(arr["sinc_H_abs"], t, 1.0, "max")
        t_qec,  _ = find_conv(arr["qec_openness"], t, 0.0, "min")
        print(f"  sinc_H→1.0              : t={t_sinc:.2f}s")
        print(f"  QEC열림→0               : t={t_qec:.2f}s")
        delta_sinc = t_sinc - t_gb
        print(f"  Δ(sinc-3종)             : {delta_sinc:+.2f}s")
    else:
        t_sinc = np.nan; delta_sinc = np.nan

    # §4 Layer C
    print(f"\n[§4] Layer C — v41 볼츠만 e창발")
    has_v41 = not np.all(np.isnan(arr["nexus_signal"]))
    if has_v41:
        stat_line(arr["nexus_signal"],  "nexus_signal")
        stat_line(arr["casm_ratio"],    "casm_ratio",   ideal=1.0)
        stat_line(arr["e_H_deviation"], "e_H_deviation",ideal=0.0)
        t_nex, v_nex = find_conv(arr["nexus_signal"],  t, 0.0, "max")
        t_cas, v_cas = find_conv(arr["casm_ratio"],    t, 1.0, "min")
        t_cur, v_cur = find_conv(arr["curv_diff"],     t, 0.0, "max")
        t_dev, v_dev = find_conv(arr["e_H_deviation"], t, 0.0, "max")
        print(f"  nexus 최대              : t={t_nex:.2f}s  값={v_nex:.4e}")
        print(f"  casm 최소               : t={t_cas:.2f}s  값={v_cas:.8f}")
        print(f"  curv_diff 최대          : t={t_cur:.2f}s  값={v_cur:.4e}")
        print(f"  e_H_deviation 최대      : t={t_dev:.2f}s  값={v_dev:.8f}")
        print(f"  curv = G_ratio?         : "
              f"{'✓ 동기' if abs(t_cur-t_gr)<0.6 else f'차이 {t_cur-t_gr:.1f}s'}")
        print(f"  nexus = boltzmann?      : "
              f"{'✓ 동기' if abs(t_nex-t_bl)<0.6 else f'차이 {t_nex-t_bl:.1f}s'}")
    else:
        t_nex=t_cas=t_cur=t_dev=np.nan

    # §5 수렴 시각 통합
    print(f"\n[§5] 3계층 수렴 시각 통합")
    print(f"  Layer C (nexus/casm) : t={t_nex:.2f}s")
    print(f"  Layer C (curv_diff)  : t={t_cur:.2f}s")
    print(f"  Layer A (G_ratio)    : t={t_gr:.2f}s")
    print(f"  Layer B (boltzmann)  : t={t_bl:.2f}s")
    print(f"  Layer B (3종)        : t={t_gb:.2f}s")
    print(f"  Layer B (sinc)       : t={t_sinc:.2f}s")
    print(f"  Layer C (e_dev)      : t={t_dev:.2f}s")
    if T > 100:
        print(f"  [배경] T/φ²={T/PHI**2:.1f}s  T/φ^2.05={T/PHI**2.05:.1f}s")
        print(f"  nexus vs T/φ²   : 오차 {abs(t_nex-T/PHI**2):.1f}s")
        print(f"  sinc  vs T/φ^2.05: 오차 {abs(t_sinc-T/PHI**2.05):.1f}s")

    # §6 t=2.5s HCW flip
    print(f"\n[§6] t=2.5s HCW flip")
    mask = np.abs(t - 2.5) < fs_c
    if mask.any():
        i25 = np.where(mask)[0][0]
        for k in ["G_ratio","grid_balance","faraday_abs",
                  "sinc_H_abs","qec_openness",
                  "nexus_signal","casm_ratio","e_H_deviation"]:
            v = arr[k][i25]
            if not np.isnan(v):
                print(f"    {k:<24} = {v:.8f}")

    # §7 φ 분할 (배경 전용)
    if T > 100:
        print(f"\n[§7] φ 시간 분할 (배경)")
        for n_p, lbl_p in [(1,"T/φ"),(2,"T/φ²"),(3,"T/φ³"),(4,"T/φ⁴")]:
            pt  = T / PHI**n_p
            idx = np.argmin(np.abs(t - pt))
            gb_v  = arr["grid_balance"][idx]
            bl_v  = arr["boltzmann_local"][idx]
            sc_v  = arr["sinc_H_abs"][idx] if has_sinc else np.nan
            nex_v = arr["nexus_signal"][idx] if has_v41 else np.nan
            dev_v = arr["e_H_deviation"][idx] if has_v41 else np.nan
            print(f"  {lbl_p:<8}={pt:8.1f}s  "
                  f"gb={gb_v:.4f}  bolt={bl_v:.6f}  "
                  f"sinc={sc_v:.4f}  nexus={nex_v:.4e}  e_dev={dev_v:.6f}")

    return {
        "label": label, "n_chunks": n, "T_total": T,
        "G_ratio_err_pct": float(gr_err.mean()),
        "born_rule_dev": float(br_dev.mean()),
        "t_gr": float(t_gr),
        "t_boltzmann": float(t_bl),
        "t_3jong": float(t_gb),
        "t_sinc": float(t_sinc) if not np.isnan(t_sinc) else None,
        "t_nexus": float(t_nex) if not np.isnan(t_nex) else None,
        "t_curv": float(t_cur) if not np.isnan(t_cur) else None,
        "t_edev": float(t_dev) if not np.isnan(t_dev) else None,
        "delta_sinc": float(delta_sinc) if not np.isnan(delta_sinc) else None,
        "t25_sinc": float(arr["sinc_H_abs"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_sinc else np.nan,
        "t25_nexus": float(arr["nexus_signal"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_v41 else np.nan,
        "t25_edev": float(arr["e_H_deviation"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_v41 else np.nan,
    }

def compare_all(results):
    if len(results) < 2: return
    print(f"\n{'='*100}")
    print("  📋 3계층 통합 비교표")
    print(f"{'='*100}")
    print(f"  {'레이블':<18} {'C(nex)':>8} {'A(G_r)':>8} {'B(bolt)':>8} "
          f"{'B(3종)':>8} {'B(sinc)':>8} {'C(e_dev)':>9} {'Δsinc':>7}")
    print(f"  {'-'*98}")
    for r in results:
        if not r: continue
        def fs(v): return f"{v:.1f}s" if v is not None else "N/A"
        ds = f"{r['delta_sinc']:+.1f}s" if r.get('delta_sinc') is not None else "N/A"
        print(f"  {r['label']:<18} "
              f"{fs(r.get('t_nexus')):>8} "
              f"{fs(r.get('t_gr')):>8} "
              f"{fs(r.get('t_boltzmann')):>8} "
              f"{fs(r.get('t_3jong')):>8} "
              f"{fs(r.get('t_sinc')):>8} "
              f"{fs(r.get('t_edev')):>9} "
              f"{ds:>7}")

    print(f"\n  [우주 고유 상수 공통 확인]")
    for r in results:
        if not r: continue
        print(f"  {r['label']:<18} "
              f"G_ratio_err={r['G_ratio_err_pct']:.6f}%  "
              f"born_rule_dev={r['born_rule_dev']:.2e}")

    print(f"\n  [Layer C 검증]")
    print(f"  {'레이블':<18} {'nexus=bolt?':>12} {'curv=G_r?':>10} "
          f"{'t2.5_sinc':>10} {'t2.5_nexus':>11} {'t2.5_edev':>10}")
    print(f"  {'-'*76}")
    for r in results:
        if not r: continue
        t_nex = r.get('t_nexus'); t_bl = r.get('t_boltzmann')
        t_cur = r.get('t_curv');  t_gr = r.get('t_gr')
        nb = "✓" if (t_nex is not None and t_bl is not None and abs(t_nex-t_bl)<0.6) else \
             f"{(t_nex or 0)-(t_bl or 0):+.1f}s"
        cg = "✓" if (t_cur is not None and t_gr is not None and abs(t_cur-t_gr)<0.6) else \
             f"{(t_cur or 0)-(t_gr or 0):+.1f}s"
        print(f"  {r['label']:<18} {nb:>12} {cg:>10} "
              f"{r.get('t25_sinc',float('nan')):>10.5f} "
              f"{r.get('t25_nexus',float('nan')):>11.5e} "
              f"{r.get('t25_edev',float('nan')):>10.6f}")

    print(f"\n  [Δ(sinc-3종) 분류]")
    for r in results:
        if not r: continue
        d = r.get('delta_sinc')
        if d is None:       cat = "N/A"
        elif d > 2:         cat = "BBH형 (QEC 늦음)"
        elif d < -100:      cat = "배경형 (QEC 압도적 선행)"
        elif d < -1:        cat = "D1분기형 (QEC 선행)"
        else:               cat = "동시형 (BNS/3Det)"
        ds = f"{d:+.1f}s" if d is not None else "N/A"
        print(f"  {r['label']:<18} Δ={ds:>9}  → [{cat}]")

    print(f"\n  [e_dev 채널 분기]")
    for r in results:
        if not r: continue
        t_dev = r.get('t_edev')
        t_3   = r.get('t_3jong')
        if t_dev is not None and t_3 is not None:
            diff = t_dev - t_3
            print(f"  {r['label']:<18} e_dev={t_dev:.1f}s  3종={t_3:.1f}s  "
                  f"차이={diff:+.1f}s")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        files = sorted(glob.glob("eqi_monster_v89_faraday_sinc_v41_*.json"))
        if not files:
            # fallback: sinc만 있는 JSON도 시도
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
                for kw in ["GW150914","GW170817","GW170814","SGWB","sgwb"]:
                    if kw.lower() in bn.lower():
                        lbl = kw; break
                else:
                    lbl = bn[:20]
            labeled.append((path, lbl))

    print("="*75)
    print("  EQI Monster v89_faraday_sinc_v41 — JSON 분석기 v3")
    print(f"  3계층: RLC + 패러데이/sinc + v41 볼츠만")
    print(f"  PHI={PHI:.6f}  MAXWELL={MAXWELL:.6f}  e={e_const:.6f}")
    print(f"  QEC_opt=1-3/π={QEC_OPT:.5f}")
    print("="*75)

    all_results = []
    for path, lbl in labeled:
        if not os.path.exists(path):
            print(f"⚠️ 파일 없음: {path}"); all_results.append(None); continue
        size_mb = os.path.getsize(path)/1e6
        print(f"\n📁 [{lbl}]  ({size_mb:.1f} MB)")
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

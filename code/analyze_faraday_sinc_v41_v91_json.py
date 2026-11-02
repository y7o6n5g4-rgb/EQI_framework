#!/usr/bin/env python3
"""
EQI 4계층+Layer E JSON 분석기 v4.1
Layer A/B/C/D/E 전체 비교 분석

사용법:
  python analyze_faraday_sinc_v41_v91_json.py file.json:레이블 ...
  python analyze_faraday_sinc_v41_v91_json.py *.json
"""

import json
import numpy as np
import sys, os, glob

PHI          = (1+np.sqrt(5))/2
GAP          = np.log(25)/100
GAP_D2       = np.log(5)/100
W_L          = PHI**(-1)
W_H          = PHI**(-2)
D1           = PHI**(-1.5)
D2           = PHI**(+1.5)
c_EQI        = PHI**0.5
Z_BASE       = D1 + D2          # = 2√φ = 2c_EQI
MAXWELL      = 1.8731137721991709
QEC_P        = 1 - 2/np.pi      # p오비탈 QEC = sinc_maxwell 임계
D1_ELL_TGT   = 0.236740008222
TIP_POINT    = PHI**1.5 - 1.0
FF_ZEROS     = [14.1347, 21.022, 25.011, 30.425, 32.935, 37.586]

def load_json(fp):
    with open(fp,'r',encoding='utf-8') as f: return json.load(f)

def extract(data):
    chunks = data.get("chunk_results",[])
    if not chunks: return None
    keys = [
        "time_s","G_ratio","grid_balance","faraday_abs",
        "born_rule","boltzmann_local",
        "sinc_H_abs","qec_openness","e_maxwell",
        "nexus_signal","casm_ratio","e_H_deviation",
        "unity_product","unity_rd","n_ouroboros",
        "carnot_eta","tau_ratio","bern_true",
        "delta_H","eqi4_sigma_ell_H",
        "riemann_ratio","mach_signal","z0_14_1347",
        "qec_maxwell_theory",
    ]
    arr = {}
    for k in keys:
        try:    arr[k] = np.array([c.get(k,np.nan) for c in chunks])
        except: arr[k] = np.full(len(chunks),np.nan)
    return arr

def conv(arr, t, target, mode="min_abs_diff"):
    v = ~np.isnan(arr)
    if not v.any(): return np.nan, np.nan
    if   mode=="min_abs_diff":
        idx=np.argmin(np.abs(arr[v]-target)); idxs=np.where(v)[0][idx]
    elif mode=="max": idxs=np.nanargmax(arr)
    elif mode=="min": idxs=np.nanargmin(arr)
    return t[idxs], arr[idxs]

def stat(arr, name, ideal=None):
    v = arr[~np.isnan(arr)]
    if len(v)==0: print(f"  {name:<28}: 없음"); return
    s = f"  {name:<28}: avg={v.mean():.6f}  std={v.std():.6f}  min={v.min():.6f}  max={v.max():.6f}"
    if ideal is not None:
        s += f"  [이상={ideal:.4f} 편차={abs(v.mean()-ideal):.6f}]"
    print(s)

def get_layer_e(data):
    """JSON에서 Layer E 추출"""
    le = data.get("layer_E_fermat_feynman", {})
    if not le: return None
    return {
        "n_avg":        le.get("n_avg_ouroboros", np.nan),
        "Z_D1":         le.get("Z_D1_fermat", np.nan),
        "Z_D2":         le.get("Z_D2_feynman", np.nan),
        "Z_EQI":        le.get("Z_EQI", np.nan),
        "Z_norm":       le.get("Z_EQI_norm", np.nan),
        "Z_M":          le.get("Z_EQI_maxwell", np.nan),
        "E_D1_em":      le.get("E_D1_emission_mean", np.nan),
        "gap_traces":   le.get("gap_traces", {}),
        "gap_traces_M": le.get("gap_traces_maxwell", {}),
        "TIP_POINT":    le.get("TIP_POINT", TIP_POINT),
    }

def analyze(data, label=""):
    arr = extract(data)
    if arr is None: print(f"  [{label}] 없음"); return {}

    t=arr["time_s"]; n=len(t); T=t[-1] if n>0 else 0
    fs_c = t[1]-t[0] if n>1 else 0.5

    print(f"\n{'='*78}")
    print(f"  📊 [{label}]  {n} chunks  |  {t[0]:.1f}s ~ {T:.1f}s")
    print(f"{'='*78}")

    # §1 우주 고유 상수
    print(f"\n[§1] 우주 고유 상수")
    gr_err = np.abs(arr["G_ratio"]-W_L)/W_L*100
    print(f"  G_ratio 오차%           : avg={gr_err.mean():.8f}%")
    br_dev = np.abs(arr["born_rule"]-1.0)
    print(f"  born_rule 편차          : avg={br_dev.mean():.2e}")

    # §2 Layer A/B
    print(f"\n[§2] Layer A+B — RLC + 패러데이 + sinc_maxwell")
    t_gr,_  = conv(arr["G_ratio"],         t, W_L, "min")
    t_gb,_  = conv(arr["grid_balance"],    t, 1.0, "max")
    t_bl,_  = conv(arr["boltzmann_local"], t, 1.0, "max")
    t_si,v_si = conv(arr["sinc_H_abs"],    t, 1.0, "max")
    t_up,v_up = conv(arr["unity_product"], t, MAXWELL, "min_abs_diff")
    print(f"  G_ratio 최솟값          : t={t_gr:.2f}s")
    print(f"  boltzmann→1.0           : t={t_bl:.2f}s")
    print(f"  grid_balance→1.0 (3종)  : t={t_gb:.2f}s")
    print(f"  sinc_maxwell→최대       : t={t_si:.2f}s  값={v_si:.5f}")
    print(f"  unity→MAXWELL           : t={t_up:.2f}s  값={v_up:.6f}")
    ds = t_si - t_gb
    print(f"  Δ(sinc-3종)             : {ds:+.2f}s")
    if   ds > 2:    cat="BBH형"
    elif ds < -100: cat="배경형"
    elif ds < -1:   cat="D1분기형"
    else:           cat="동시형(BNS/3Det)"
    print(f"  유형                    : [{cat}]")
    stat(arr["e_maxwell"], "e_maxwell", ideal=1.0)

    # §3 Layer C
    print(f"\n[§3] Layer C — v41 볼츠만 e창발")
    has_c = not np.all(np.isnan(arr["nexus_signal"]))
    if has_c:
        t_nex,_ = conv(arr["nexus_signal"],  t, 0.0, "max")
        t_dev,_ = conv(arr["e_H_deviation"], t, 0.0, "max")
        print(f"  nexus 최대              : t={t_nex:.2f}s")
        print(f"  e_H_dev 최대            : t={t_dev:.2f}s")
        print(f"  nexus=bolt?             : "
              f"{'✓' if abs(t_nex-t_bl)<0.6 else f'Δ={t_nex-t_bl:.1f}s'}")
    else:
        t_nex=t_dev=np.nan

    # §4 Layer D
    print(f"\n[§4] Layer D — v9.1 비등방성 타원 + 라그랑지")
    has_d = not np.all(np.isnan(arr["unity_product"]))
    if has_d:
        t_ell,v_ell = conv(arr["eqi4_sigma_ell_H"], t, 0.0, "min")
        t_rim,v_rim = conv(arr["riemann_ratio"],     t, 0.0, "max")
        stat(arr["n_ouroboros"],      "n_ouroboros")
        stat(arr["carnot_eta"],       "carnot_eta")
        stat(arr["eqi4_sigma_ell_H"], "ell_sigma(→0)", ideal=0.0)
        print(f"  ell_sigma 최솟값        : t={t_ell:.2f}s  값={v_ell:.8f}")
        print(f"  riemann_ratio 최대      : t={t_rim:.2f}s  값={v_rim:.4f}")
        print(f"  curv=G_ratio?           : "
              f"{'✓' if abs(t_rim-t_gr)<0.6 else '별도 확인 필요'}")
    else:
        t_ell=t_rim=np.nan; v_ell=np.nan

    # §5 Layer E
    print(f"\n[§5] Layer E — v18_10 페르마-파인만 경량")
    le = get_layer_e(data)
    if le:
        n_avg = le["n_avg"]
        Z_norm= le["Z_norm"]
        Z_M   = le["Z_M"]
        E_em  = le["E_D1_em"]
        print(f"  n_ouroboros 평균        : {n_avg:.6f}")
        print(f"  Z_EQI = {le['Z_EQI']:.6f}")
        print(f"  Z_EQI_norm (÷2√φ)       : {Z_norm:.8f}  "
              f"[{'D2팽창>광속' if Z_norm>1 else 'D1수축<광속'}]")
        print(f"  Z_EQI/MAXWELL           : {Z_M:.8f}  "
              f"[{'포논>교환효율' if Z_M>1 else '포논<교환효율'}]")
        print(f"  E_D1_emission 평균      : {E_em:.6f}")
        print(f"  팽창비=D2×n×GAP/2√φ    : {le['Z_EQI']/Z_BASE-1:.6f}")
        print(f"\n  gap_traces/MAXWELL:")
        gm = le["gap_traces_M"]
        for z in FF_ZEROS:
            key = f"Z_{z}_norm"
            val = gm.get(key, np.nan)
            mark = " ★" if not np.isnan(val) and abs(abs(val)-1.0)<0.05 else ""
            print(f"    Z_{z:<8.4f}: {val:+.6f}{mark}")
    else:
        n_avg=Z_norm=Z_M=E_em=np.nan
        print(f"  Layer E 데이터 없음 (layer_E_fermat_feynman 키 없음)")

    # §6 수렴 시각 통합
    print(f"\n[§6] 5계층 수렴 시각 통합")
    rows = [
        ("[C] nexus",      t_nex),
        ("[B] boltzmann",  t_bl),
        ("[A] G_ratio",    t_gr),
        ("[B] 3종",        t_gb),
        ("[B] sinc_M",     t_si),
        ("[D] unity",      t_up),
        ("[D] ell_sigma",  t_ell),
        ("[D] riemann",    t_rim),
    ]
    for name, tv in rows:
        ts = f"{tv:.2f}s" if not np.isnan(tv) else "N/A"
        print(f"  {name:<20}: t={ts}")
    if T > 100:
        print(f"  [배경] T/φ²={T/PHI**2:.1f}s  T/φ^2.05={T/PHI**2.05:.1f}s")

    # §7 t=2.5s
    print(f"\n[§7] t=2.5s HCW flip")
    mask = np.abs(t-2.5)<fs_c
    t25_vals = {}
    if mask.any():
        i25 = np.where(mask)[0][0]
        for k in ["G_ratio","grid_balance","sinc_H_abs","qec_openness",
                  "nexus_signal","unity_product","n_ouroboros",
                  "carnot_eta","delta_H","eqi4_sigma_ell_H",
                  "e_maxwell","riemann_ratio","z0_14_1347"]:
            v = arr[k][i25]
            if not np.isnan(v):
                print(f"    {k:<28} = {v:.8f}")
                t25_vals[k] = float(v)

    # §8 φ 분할 (배경 전용)
    if T > 100:
        print(f"\n[§8] φ 시간 분할 (배경)")
        for n_p,lp in [(1,"T/φ"),(2,"T/φ²"),(3,"T/φ³"),(4,"T/φ⁴")]:
            pt  = T/PHI**n_p
            idx = np.argmin(np.abs(t-pt))
            gb_v = arr["grid_balance"][idx]
            bl_v = arr["boltzmann_local"][idx]
            sc_v = arr["sinc_H_abs"][idx]
            up_v = arr["unity_product"][idx]
            el_v = arr["eqi4_sigma_ell_H"][idx]
            print(f"  {lp:<8}={pt:8.1f}s  "
                  f"gb={gb_v:.4f}  bolt={bl_v:.6f}  "
                  f"sinc={sc_v:.4f}  unity={up_v:.4f}  ell={el_v:.6f}")

    return {
        "label":label, "n_chunks":n, "T_total":T,
        "G_ratio_err_pct": float(gr_err.mean()),
        "t_gr":float(t_gr), "t_bl":float(t_bl),
        "t_gb":float(t_gb), "t_sinc":float(t_si),
        "t_nexus": float(t_nex) if not np.isnan(t_nex) else None,
        "t_unity": float(t_up)  if not np.isnan(t_up)  else None,
        "t_ell":   float(t_ell) if not np.isnan(t_ell) else None,
        "t_rim":   float(t_rim) if not np.isnan(t_rim) else None,
        "ell_min": float(v_ell) if not np.isnan(v_ell) else None,
        "delta_sinc": float(ds),
        "cat": cat,
        # Layer E
        "n_avg":  float(n_avg)  if not np.isnan(n_avg)  else None,
        "Z_norm": float(Z_norm) if not np.isnan(Z_norm) else None,
        "Z_M":    float(Z_M)    if not np.isnan(Z_M)    else None,
        "E_D1_em":float(E_em)   if not np.isnan(E_em)   else None,
        # t=2.5s
        "t25": t25_vals,
        # gap_traces
        "gap_traces_M": le["gap_traces_M"] if le else {},
    }

def compare(results):
    valid = [r for r in results if r]
    if len(valid)<2: return

    print(f"\n{'='*100}")
    print("  📋 5계층 통합 비교표")
    print(f"{'='*100}")

    # 수렴 시각표
    print(f"\n  수렴 시각:")
    print(f"  {'채널':<18} {'C(nex)':>7} {'B(bolt)':>7} {'B(3종)':>7} "
          f"{'B(sinc)':>8} {'D(unity)':>9} {'D(ell)':>8} {'Δsinc':>7}")
    print(f"  {'-'*78}")
    for r in valid:
        def fs(v): return f"{v:.1f}s" if v is not None else "N/A"
        print(f"  {r['label']:<18} "
              f"{fs(r.get('t_nexus')):>7} "
              f"{fs(r.get('t_bl')):>7} "
              f"{fs(r.get('t_gb')):>7} "
              f"{fs(r.get('t_sinc')):>8} "
              f"{fs(r.get('t_unity')):>9} "
              f"{fs(r.get('t_ell')):>8} "
              f"{r['delta_sinc']:>+7.1f}s  [{r['cat']}]")

    # Layer E 비교표
    print(f"\n  Layer E — 페르마-파인만:")
    print(f"  {'채널':<18} {'n_avg':>8} {'Z_norm':>8} {'Z/M':>8} "
          f"{'E_D1_em':>9} {'분류':>12}")
    print(f"  {'-'*72}")
    for r in valid:
        n  = r.get('n_avg')
        zn = r.get('Z_norm')
        zm = r.get('Z_M')
        em = r.get('E_D1_em')
        if n is not None:
            cls = "D2팽창>광속" if (zn or 0)>1 else "D1수축<광속"
            print(f"  {r['label']:<18} "
                  f"{n:>8.3f} {zn:>8.4f} {zm:>8.4f} "
                  f"{em:>9.4f} {cls:>12}")

    # gap_traces 부호 패턴
    print(f"\n  gap_traces/MAXWELL 부호 패턴:")
    print(f"  {'채널':<18}", end="")
    for z in FF_ZEROS:
        print(f"  Z_{z:.3f}", end="")
    print()
    print(f"  {'-'*90}")
    for r in valid:
        gm = r.get("gap_traces_M", {})
        print(f"  {r['label']:<18}", end="")
        for z in FF_ZEROS:
            val = gm.get(f"Z_{z}_norm", np.nan)
            if not np.isnan(val):
                sign = "+" if val>=0 else "-"
                mark = "★" if abs(abs(val)-1.0)<0.05 else " "
                print(f"  {mark}{sign}{abs(val):.3f}", end="")
            else:
                print(f"   N/A ", end="")
        print()

    # Z_21 부호 분류
    print(f"\n  Z_21.022 부호 분류:")
    for r in valid:
        gm = r.get("gap_traces_M", {})
        v21 = gm.get("Z_21.022_norm", np.nan)
        if not np.isnan(v21):
            cls = "BBH형(양수)" if v21>0 else "BNS/배경형(음수)"
            print(f"  {r['label']:<18} Z_21={v21:+.4f}  [{cls}]")

    # ell_sigma 정렬
    print(f"\n  ell_sigma 최솟값 (라그랑지 안정점):")
    es = sorted([r for r in valid if r.get('ell_min') is not None],
                key=lambda x: x['ell_min'])
    for r in es:
        print(f"  {r['label']:<18} ell_min={r['ell_min']:.8f}")

    # t=2.5s 비교
    print(f"\n  t=2.5s HCW flip:")
    print(f"  {'채널':<18} {'sinc_M':>8} {'e_M':>8} {'unity':>8} "
          f"{'delta_H':>10} {'ell':>10} {'n_ouro':>8} {'carnot':>8}")
    print(f"  {'-'*85}")
    for r in valid:
        t25 = r.get('t25', {})
        print(f"  {r['label']:<18} "
              f"{t25.get('sinc_H_abs',float('nan')):>8.4f} "
              f"{t25.get('e_maxwell',float('nan')):>8.4f} "
              f"{t25.get('unity_product',float('nan')):>8.4f} "
              f"{t25.get('delta_H',float('nan')):>10.4f} "
              f"{t25.get('eqi4_sigma_ell_H',float('nan')):>10.6f} "
              f"{t25.get('n_ouroboros',float('nan')):>8.2f} "
              f"{t25.get('carnot_eta',float('nan')):>8.4f}")

    # G_ratio 오차%
    print(f"\n  G_ratio 오차% (이벤트 강도):")
    gs = sorted(valid, key=lambda x: x['G_ratio_err_pct'])
    for r in gs:
        print(f"  {r['label']:<18} {r['G_ratio_err_pct']:.8f}%")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        files = sorted(glob.glob("eqi_4layer_*.json"))
        if not files:
            files = sorted(glob.glob("eqi_monster_v89_faraday_sinc_v41_*.json"))
        if not files:
            print("❌ JSON 없음"); sys.exit(1)
        print(f"📂 자동: {len(files)}개")
        labeled = [(f, os.path.basename(f)[:35]) for f in files]
    else:
        labeled = []
        for arg in args:
            if ":" in arg:
                path, lbl = arg.split(":",1)
            else:
                path=arg; bn=os.path.basename(path)
                for kw in ["GW150914","GW170817","GW170814","SGWB","sgwb"]:
                    if kw.lower() in bn.lower(): lbl=kw; break
                else: lbl=bn[:20]
            labeled.append((path,lbl))

    print("="*78)
    print("  EQI 5계층 JSON 분석기 v4.1")
    print(f"  Layer A/B/C/D/E + Layer E 페르마-파인만")
    print(f"  PHI={PHI:.6f}  MAXWELL={MAXWELL:.6f}")
    print(f"  Z_BASE=2√φ={Z_BASE:.6f}  TIP={TIP_POINT:.6f}")
    print(f"  FF_ZEROS={FF_ZEROS}")
    print("="*78)

    all_results = []
    for path, lbl in labeled:
        if not os.path.exists(path):
            print(f"⚠️ 없음: {path}"); all_results.append(None); continue
        mb = os.path.getsize(path)/1e6
        print(f"\n📁 [{lbl}]  ({mb:.1f} MB)")
        try:
            r = analyze(load_json(path), label=lbl)
            all_results.append(r)
        except Exception as ex:
            import traceback; traceback.print_exc()
            all_results.append(None)

    valid = [r for r in all_results if r]
    if len(valid)>=2:
        compare(valid)

    print(f"\n✅ 분석 완료")

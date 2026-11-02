#!/usr/bin/env python3
"""
EQI 4계층 통합 JSON 분석기 v5 (S100 대차대조표 패치)
Layer A (RLC) + Layer B (패러데이/sinc_maxwell) +
Layer C (v41 볼츠만) + Layer D (v9.1 비등방성 타원)
+ [S100] 외부효과 대차대조표 (장부 ①§35 차변 / 장부 ②엔트로피 교환)
  장부 ① 차변 잔액  = delta_H − D1_ELL   (§35 capillary 계정, 부호 보존)
  장부 ② 외부효과   = unity − MAXWELL     → n_ledger = 외부효과/GAP
         unity = MAXWELL + n×GAP  (부정적분 1+n×GAP 형태)
  산입률            = MAXWELL / unity

사용법:
  python analyze_faraday_sinc_v41_v91_json.py file.json:레이블 ...
  python analyze_faraday_sinc_v41_v91_json.py *.json
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
MAXWELL      = 1.8731137721991709
QEC_OPT      = 1 - 3/np.pi       # f오비탈 QEC 최적
QEC_P_ORBITAL= 1 - 2/np.pi       # p오비탈 QEC = sinc_maxwell 임계
D1_ELL_TARGET= 0.236740008222
SINC_M_CRIT  = 2/np.pi           # unity=MAXWELL 임계 sinc값
e_const      = np.e

def load_json(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract(data):
    chunks = data.get("chunk_results", [])
    if not chunks: return None
    keys = [
        "time_s","G_ratio","grid_balance","faraday_abs",
        "born_rule","phase_angle_deg","boltzmann_local",
        "sinc_H_abs","qec_openness",
        "nexus_signal","casm_ratio","e_H_deviation",
        "unity_product","unity_rd","n_ouroboros",
        "carnot_eta","tau_ratio","bern_true",
        "delta_H","eqi4_sigma_ell_H",
        "riemann_ratio","mach_signal","z0_14_1347",
        "e_maxwell","qec_maxwell_theory",
    ]
    arr = {}
    for k in keys:
        try:
            arr[k] = np.array([c.get(k, np.nan) for c in chunks])
        except:
            arr[k] = np.full(len(chunks), np.nan)
    return arr

def stat(arr, name, ideal=None):
    v = arr[~np.isnan(arr)]
    if len(v)==0: print(f"  {name:<28}: 없음"); return
    mn,mx,avg,sd = v.min(),v.max(),v.mean(),v.std()
    s = f"  {name:<28}: avg={avg:.6f}  std={sd:.6f}  min={mn:.6f}  max={mx:.6f}"
    if ideal is not None:
        s += f"  [이상={ideal:.4f} 편차={abs(avg-ideal):.6f}]"
    print(s)

def conv(arr, t, target, mode="min_abs_diff"):
    v = ~np.isnan(arr)
    if not v.any(): return np.nan, np.nan
    if   mode=="min_abs_diff":
        idx=np.argmin(np.abs(arr[v]-target)); idxs=np.where(v)[0][idx]
    elif mode=="max": idxs=np.nanargmax(arr)
    elif mode=="min": idxs=np.nanargmin(arr)
    return t[idxs], arr[idxs]

def analyze(data, label=""):
    arr = extract(data)
    if arr is None: print(f"  [{label}] 없음"); return {}

    t = arr["time_s"]; n=len(t); T=t[-1] if n>0 else 0
    fs_c = t[1]-t[0] if n>1 else 0.5

    print(f"\n{'='*78}")
    print(f"  📊 [{label}]  {n} chunks  |  {t[0]:.1f}s ~ {T:.1f}s")
    print(f"{'='*78}")

    # §1 우주 고유 상수
    print(f"\n[§1] 우주 고유 상수")
    gr_err = np.abs(arr["G_ratio"]-W_L)/W_L*100
    print(f"  G_ratio 오차%           : avg={gr_err.mean():.8f}%")
    br_dev = np.abs(arr["born_rule"]-1.0)
    print(f"  born_rule 편차          : avg={br_dev.mean():.2e}  max={br_dev.max():.2e}")

    # §2 Layer A
    print(f"\n[§2] Layer A — RLC")
    t_gr, _ = conv(arr["G_ratio"], t, W_L, "min")
    print(f"  G_ratio 최솟값          : t={t_gr:.2f}s")

    # §3 Layer B — 패러데이+sinc_maxwell
    print(f"\n[§3] Layer B — 패러데이 + sinc_maxwell (MAXWELL 엔트로피장 기반)")
    t_gb, _ = conv(arr["grid_balance"], t, 1.0, "max")
    t_bl, _ = conv(arr["boltzmann_local"], t, 1.0, "max")
    t_sinc, v_sinc = conv(arr["sinc_H_abs"], t, 1.0, "max")
    print(f"  grid_balance→1.0        : t={t_gb:.2f}s")
    print(f"  boltzmann→1.0           : t={t_bl:.2f}s")
    print(f"  sinc_maxwell→최대       : t={t_sinc:.2f}s  값={v_sinc:.5f}")
    stat(arr["sinc_H_abs"],   "sinc_H_abs(maxwell)")
    stat(arr["e_maxwell"],    "e_maxwell",    ideal=1.0)
    delta_sinc_3jong = t_sinc - t_gb
    print(f"  Δ(sinc-3종)             : {delta_sinc_3jong:+.2f}s")
    # 분류
    if   delta_sinc_3jong > 2:    cat = "BBH형"
    elif delta_sinc_3jong < -100: cat = "배경형"
    elif delta_sinc_3jong < -1:   cat = "D1분기형"
    else:                          cat = "동시형(BNS/3Det)"
    print(f"  유형                    : [{cat}]")
    print(f"  [이론] sinc_maxwell 임계: 2/π={SINC_M_CRIT:.5f} (unity=MAXWELL)")
    print(f"  [이론] QEC_p열림        : 1-2/π={QEC_P_ORBITAL:.5f}")

    # §4 Layer C
    print(f"\n[§4] Layer C — v41 볼츠만 e창발")
    has_c = not np.all(np.isnan(arr["nexus_signal"]))
    if has_c:
        t_nex, v_nex = conv(arr["nexus_signal"],  t, 0.0, "max")
        t_cas, _     = conv(arr["casm_ratio"],     t, 1.0, "min")
        t_dev, v_dev = conv(arr["e_H_deviation"],  t, 0.0, "max")
        print(f"  nexus 최대              : t={t_nex:.2f}s  값={v_nex:.4e}")
        print(f"  casm 최소               : t={t_cas:.2f}s")
        print(f"  e_H_dev 최대            : t={t_dev:.2f}s  값={v_dev:.6f}")
        print(f"  nexus=bolt?             : "
              f"{'✓' if abs(t_nex-t_bl)<0.6 else f'Δ={t_nex-t_bl:.1f}s'}")
    else:
        t_nex=t_cas=t_dev=np.nan

    # §5 Layer D — v9.1 비등방성 타원
    print(f"\n[§5] Layer D — v9.1 비등방성 타원 + 라그랑지 최솟값")
    has_d = not np.all(np.isnan(arr["unity_product"]))
    if has_d:
        t_up, v_up   = conv(arr["unity_product"],    t, MAXWELL, "min_abs_diff")
        t_ell, v_ell = conv(arr["eqi4_sigma_ell_H"], t, 0.0, "min")
        t_rim, v_rim = conv(arr["riemann_ratio"],    t, 0.0, "max")
        stat(arr["unity_product"],    "unity_product",   ideal=MAXWELL)
        stat(arr["n_ouroboros"],      "n_ouroboros")
        stat(arr["carnot_eta"],       "carnot_eta")
        stat(arr["delta_H"],          "delta_H",         ideal=D1_ELL_TARGET)
        stat(arr["eqi4_sigma_ell_H"], "ell_sigma(→0)",   ideal=0.0)
        stat(arr["riemann_ratio"],    "riemann_ratio")
        stat(arr["e_maxwell"],        "e_maxwell",        ideal=1.0)
        print(f"  unity→MAXWELL           : t={t_up:.2f}s  값={v_up:.6f}")
        print(f"  unity=sinc 동기?        : "
              f"{'✓' if abs(t_up-t_sinc)<0.6 else f'Δ={t_up-t_sinc:.1f}s'}")
        print(f"  ell_sigma→0 (라그랑지 안정점): t={t_ell:.2f}s  값={v_ell:.8f}")
        print(f"  riemann_ratio 최대      : t={t_rim:.2f}s  값={v_rim:.4f}")
        print(f"  [이론] D1_ell_target    = {D1_ELL_TARGET:.9f}")
        print(f"  [이론] ell_sigma=0 = 라그랑지 최솟값 = 최소 작용 원리 달성")
        # 배경 전용
        if T > 100:
            print(f"  [배경] T/φ²={T/PHI**2:.1f}s  "
                  f"ell_t={t_ell:.1f}s  unity_t={t_sinc:.1f}s")
    else:
        t_up=t_ell=t_rim=np.nan; v_ell=np.nan

    # §6 수렴 시각 통합
    print(f"\n[§6] 4계층 수렴 시각 통합")
    print(f"  [C] nexus               : t={t_nex:.2f}s")
    print(f"  [A] G_ratio             : t={t_gr:.2f}s")
    print(f"  [B] boltzmann           : t={t_bl:.2f}s")
    print(f"  [B] grid_balance(3종)   : t={t_gb:.2f}s")
    print(f"  [B] sinc_maxwell        : t={t_sinc:.2f}s")
    print(f"  [D] unity→MAXWELL       : t={t_up:.2f}s")
    print(f"  [D] ell_sigma→0         : t={t_ell:.2f}s")
    print(f"  [D] riemann 최대        : t={t_rim:.2f}s")

    # §7 t=2.5s
    print(f"\n[§7] t=2.5s HCW flip")
    mask = np.abs(t-2.5)<fs_c
    if mask.any():
        i25 = np.where(mask)[0][0]
        for k in ["G_ratio","grid_balance","sinc_H_abs","qec_openness",
                  "nexus_signal","unity_product","n_ouroboros",
                  "carnot_eta","delta_H","eqi4_sigma_ell_H",
                  "e_maxwell","riemann_ratio","z0_14_1347"]:
            v = arr[k][i25]
            if not np.isnan(v):
                print(f"    {k:<28} = {v:.8f}")

    # §8 φ 분할 (배경 전용)
    if T > 100:
        print(f"\n[§8] φ 시간 분할 (배경)")
        for n_p,lp in [(1,"T/φ"),(2,"T/φ²"),(3,"T/φ³"),(4,"T/φ⁴")]:
            pt  = T/PHI**n_p
            idx = np.argmin(np.abs(t-pt))
            gb_v  = arr["grid_balance"][idx]
            bl_v  = arr["boltzmann_local"][idx]
            sc_v  = arr["sinc_H_abs"][idx]
            up_v  = arr["unity_product"][idx] if has_d else np.nan
            el_v  = arr["eqi4_sigma_ell_H"][idx] if has_d else np.nan
            em_v  = arr["e_maxwell"][idx] if has_d else np.nan
            print(f"  {lp:<8}={pt:8.1f}s  "
                  f"gb={gb_v:.4f}  bolt={bl_v:.6f}  sinc={sc_v:.4f}  "
                  f"unity={up_v:.4f}  ell={el_v:.6f}  e_M={em_v:.4f}")

    led = ledger(arr, label) if has_d else None

    return {
        "ledger": led,
        "label":label, "n_chunks":n, "T_total":T,
        "G_ratio_err_pct": float(gr_err.mean()),
        "t_gr":float(t_gr), "t_bl":float(t_bl),
        "t_gb":float(t_gb), "t_sinc":float(t_sinc),
        "t_nexus": float(t_nex) if not np.isnan(t_nex) else None,
        "t_unity": float(t_up)  if not np.isnan(t_up)  else None,
        "t_ell":   float(t_ell) if not np.isnan(t_ell) else None,
        "t_rim":   float(t_rim) if not np.isnan(t_rim) else None,
        "ell_min": float(v_ell) if not np.isnan(v_ell) else None,
        "delta_sinc": float(delta_sinc_3jong),
        "t25_sinc":   float(arr["sinc_H_abs"][np.argmin(np.abs(t-2.5))]) if mask.any() else np.nan,
        "t25_e_M":    float(arr["e_maxwell"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_d else np.nan,
        "t25_unity":  float(arr["unity_product"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_d else np.nan,
        "t25_delta_H":float(arr["delta_H"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_d else np.nan,
        "t25_ell":    float(arr["eqi4_sigma_ell_H"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_d else np.nan,
        "t25_n_ouro": float(arr["n_ouroboros"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_d else np.nan,
        "t25_carnot": float(arr["carnot_eta"][np.argmin(np.abs(t-2.5))]) if mask.any() and has_d else np.nan,
    }

# ════════════════════════════════════════════════════════════════
# [S100] 외부효과 대차대조표
# ════════════════════════════════════════════════════════════════
def ledger(arr, label=""):
    t    = arr["time_s"]
    up   = arr["unity_product"]
    dH   = arr["delta_H"]
    nou  = arr["n_ouroboros"]
    if np.all(np.isnan(up)) or np.all(np.isnan(dH)):
        print("  [대차대조표] Layer D 없음 — 생략")
        return None

    ext      = up - MAXWELL                 # 장부② 외부효과
    n_led    = ext / GAP                    # GAP 단위 계정
    rate     = MAXWELL / up                 # 산입률
    d1_res   = dH - D1_ELL_TARGET           # 장부① 차변 잔액 (부호)
    n_check  = n_led - nou                  # 엔진 n_ouro와 정합 확인

    ok = ~np.isnan(d1_res)
    i_bal = int(np.nanargmin(np.abs(d1_res)))   # 차변 균형 시점
    n_at  = n_led[i_bal]
    frac_1gap = float(np.mean(np.abs(n_led[~np.isnan(n_led)]) < 1.0))
    frac_pos  = float(np.mean(n_led[~np.isnan(n_led)] > 0))

    print(f"\n  ── [S100] 외부효과 대차대조표 ──")
    stat(d1_res, "장부① 차변잔액(dH−D1_ELL)", ideal=0.0)
    stat(ext,    "장부② 외부효과(unity−MAX)", ideal=0.0)
    stat(n_led,  "장부② n = 외부효과/GAP",   ideal=0.0)
    stat(rate,   "산입률 MAXWELL/unity",      ideal=1.0)
    print(f"  n_ledger − n_ouro (정합): max|Δ|={np.nanmax(np.abs(n_check)):.3e}")
    print(f"  차변 균형 시점: t={t[i_bal]:.2f}s  잔액={d1_res[i_bal]:+.8f}  "
          f"그때 n={n_at:+.3f}  산입률={rate[i_bal]:.5f}")
    print(f"  |n|<1 (1 GAP 이내) 청크 비율 = {frac_1gap:.4f}")
    print(f"  n>0 (D2 대변 우세) 청크 비율  = {frac_pos:.4f}")

    # 청크별 장부 CSV
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)[:40] or "ledger"
    fn = f"ledger_S100_{safe}.csv"
    with open(fn, "w", encoding="utf-8") as f:
        f.write("time_s,delta_H,d1_residual,unity,external,n_ledger,n_ouro,rate\n")
        for k in range(len(t)):
            f.write(f"{t[k]:.4f},{dH[k]:.10e},{d1_res[k]:.10e},{up[k]:.10e},"
                    f"{ext[k]:.10e},{n_led[k]:.10e},{nou[k]:.10e},{rate[k]:.10e}\n")
    print(f"  💾 청크별 장부 저장: {fn}")

    return {
        "d1_res_min_abs": float(np.abs(d1_res[i_bal])),
        "d1_res_signed":  float(d1_res[i_bal]),
        "t_balance":      float(t[i_bal]),
        "n_at_balance":   float(n_at),
        "rate_at_balance":float(rate[i_bal]),
        "n_avg":          float(np.nanmean(n_led)),
        "rate_avg":       float(np.nanmean(rate)),
        "frac_1gap":      frac_1gap,
        "frac_pos":       frac_pos,
    }

def compare(results):
    valid = [r for r in results if r]
    if len(valid)<2: return
    print(f"\n{'='*105}")
    print("  📋 4계층 통합 비교표")
    print(f"{'='*105}")

    # 수렴 시각 비교
    print(f"\n  수렴 시각표:")
    print(f"  {'채널':<18} {'C(nex)':>7} {'B(bolt)':>7} {'B(3종)':>7} "
          f"{'B(sinc)':>8} {'D(unity)':>9} {'D(ell)':>8} {'Δsinc':>7}")
    print(f"  {'-'*80}")
    for r in valid:
        def fs(v): return f"{v:.1f}s" if v is not None else "N/A"
        ds = f"{r['delta_sinc']:+.1f}s"
        print(f"  {r['label']:<18} "
              f"{fs(r.get('t_nexus')):>7} "
              f"{fs(r.get('t_bl')):>7} "
              f"{fs(r.get('t_gb')):>7} "
              f"{fs(r.get('t_sinc')):>8} "
              f"{fs(r.get('t_unity')):>9} "
              f"{fs(r.get('t_ell')):>8} "
              f"{ds:>7}")

    # t=2.5s 비교
    print(f"\n  t=2.5s HCW flip 비교표:")
    print(f"  {'채널':<18} {'sinc_M':>8} {'e_M':>8} {'unity':>8} "
          f"{'delta_H':>10} {'ell_sig':>10} {'n_ouro':>8} {'carnot':>8}")
    print(f"  {'-'*88}")
    for r in valid:
        print(f"  {r['label']:<18} "
              f"{r.get('t25_sinc',float('nan')):>8.4f} "
              f"{r.get('t25_e_M',float('nan')):>8.4f} "
              f"{r.get('t25_unity',float('nan')):>8.4f} "
              f"{r.get('t25_delta_H',float('nan')):>10.4f} "
              f"{r.get('t25_ell',float('nan')):>10.6f} "
              f"{r.get('t25_n_ouro',float('nan')):>8.2f} "
              f"{r.get('t25_carnot',float('nan')):>8.4f}")

    # ell_sigma 정렬
    print(f"\n  ell_sigma 최솟값 정렬 (→0 = 라그랑지 안정점):")
    ell_sorted = sorted([r for r in valid if r.get('ell_min') is not None],
                        key=lambda x: x['ell_min'])
    for r in ell_sorted:
        t_ell = r.get('t_ell')
        ts = f"{t_ell:.1f}s" if t_ell is not None else "N/A"
        print(f"  {r['label']:<18} ell_min={r['ell_min']:.8f}  t={ts}")

    # sinc=unity 동기 확인
    print(f"\n  sinc_maxwell = unity_product 동기 확인:")
    for r in valid:
        ts = r.get('t_sinc'); tu = r.get('t_unity')
        if ts is not None and tu is not None:
            sync = "✓" if abs(ts-tu)<0.6 else f"Δ={ts-tu:.1f}s"
            print(f"  {r['label']:<18} sinc={ts:.1f}s  unity={tu:.1f}s  [{sync}]")

    # Δsinc 분류
    print(f"\n  Δ(sinc-3종) 분류:")
    for r in valid:
        d = r.get('delta_sinc',0)
        if   d > 2:    cat="BBH형"
        elif d < -100: cat="배경형"
        elif d < -1:   cat="D1분기형"
        else:          cat="동시형"
        print(f"  {r['label']:<18} Δ={d:+.1f}s  [{cat}]")

    # [S100] 대차대조표 비교
    led_rows = [r for r in valid if r.get("ledger")]
    if led_rows:
        print(f"\n  [S100] 외부효과 대차대조표 비교 (차변 균형 시점 기준):")
        print(f"  {'채널':<18} {'t_bal':>8} {'차변잔액':>13} {'n@bal':>9} "
              f"{'산입률@bal':>10} {'n_avg':>9} {'|n|<1':>7} {'n>0':>7}")
        print(f"  {'-'*90}")
        for r in sorted(led_rows, key=lambda x: x["ledger"]["d1_res_min_abs"]):
            L = r["ledger"]
            print(f"  {r['label']:<18} {L['t_balance']:>7.1f}s "
                  f"{L['d1_res_signed']:>+13.8f} {L['n_at_balance']:>+9.3f} "
                  f"{L['rate_at_balance']:>10.5f} {L['n_avg']:>+9.3f} "
                  f"{L['frac_1gap']:>7.3f} {L['frac_pos']:>7.3f}")

    # G_ratio 오차
    print(f"\n  G_ratio 오차% (이벤트 강도):")
    gr_sorted = sorted(valid, key=lambda x: x['G_ratio_err_pct'])
    for r in gr_sorted:
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
                path = arg; bn = os.path.basename(path)
                for kw in ["GW150914","GW170817","GW170814","SGWB","sgwb","background"]:
                    if kw.lower() in bn.lower(): lbl=kw; break
                else: lbl = bn[:20]
            labeled.append((path,lbl))

    print("="*78)
    print("  EQI 4계층 JSON 분석기 v5 (S100 대차대조표)")
    print(f"  Layer A/B/C/D + 라그랑지 최솟값 + sinc_maxwell")
    print(f"  PHI={PHI:.6f}  MAXWELL={MAXWELL:.6f}")
    print(f"  D1_ell_target={D1_ELL_TARGET:.9f}")
    print(f"  sinc_M 임계=2/π={SINC_M_CRIT:.5f}  QEC_p=1-2/π={QEC_P_ORBITAL:.5f}")
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

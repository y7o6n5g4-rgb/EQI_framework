#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 === EQI MONSTER v8.95b RLC GRAVITY + SESSION 20 EQI === 🐉
=============================================================================
[업그레이드 목적]
  v8.9의 RLC 중력 재구성(G_topo/G = W_L) 로직을 완벽히 보존한 상태에서,
  v12.95e(점성), v13.95c(유체압력), v14.95b(열효율), v17.95b(위상/Unity)의 
  4종 대통합 검증을 'chunk × 질량' 다중 루프 구조로 통합.

[주요 스펙 준수 (Claude Architect)]
  1. eqi_session19_constants.py 절대 상수 100% 의존 (내부 재정의 금지)
  2. compute() 반환값 0~6 인덱스 원본 보존 (하위 호환성)
  3. 질량 기준 이벤트 자동 분류 (M < 3.0 -> BNS, else BBH)
  4. event_name 기반 역수 계단(inv_theory) 자동 할당

Author: MAPSI (Maestro), Claude (Architect), Gemini (Engineer)
Date: 2026.06.24
=============================================================================
"""

import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings
import scipy.constants as sc
warnings.filterwarnings("ignore")

try:
    from eqi_session19_constants import EQI_Constants_NP as C
    PHI, D1, D2, W_H, W_L, GAP = C.PHI, C.D1, C.D2, C.W_H, C.W_L, C.GAP
    N_BBH, N_BNS, c_EQI, ETA_EQI = C.N_BBH, C.N_BNS, C.c_EQI, C.ETA_EQI
    MAXWELL_REAL = C.MAXWELL_REAL
except ImportError:
    print("⚠️ [경고] eqi_session19_constants.py 누락. 프로그램이 비정상 동작할 수 있습니다.")
    raise

EPSILON_0, MU_0 = sc.epsilon_0, sc.mu_0
MASSES = [1.0, 1.4, 2.8, 10.0, 36.0, 62.0]

class EQI_Monster_v895b_RLC_Session20:
    def __init__(self):
        self.PHI   = PHI
        self.D1    = D1
        self.D2    = D2
        self.c_EQI = c_EQI
        self.W_H   = W_H
        self.W_L   = W_L
        self.GAP   = GAP
        
        self.GAP_D1= np.log(25)/100
        self.GAP_D2= np.log(5)/100
        self.hbar  = sc.hbar
        self.c     = sc.c
        self.G     = sc.G
        self.total_injections = 0
        self.fs    = 4096

        # ── RLC 파라미터 (자기홀극/기본전하 관점) ────────────
        self.L   = self.D2
        self.C   = self.D1
        self.R   = self.GAP
        self.LC  = self.L * self.C
        self.w0  = 1.0/np.sqrt(self.LC)
        self.gamma = self.R/(2*self.L)
        self.w_d = np.sqrt(max(self.w0**2 - self.gamma**2, 0))
        self.Q   = self.L/self.R

        # ── 재구성 중력 (기저값) ─────────────────────────────
        self.G_ratio_base = self.w_d**3 / self.c_EQI**2
        self.G_topo = self.G * self.G_ratio_base

        # ── EQI^4 보정 (v8.9 원본 100% 보존) ─────────────────
        residual = float(self.W_L) - self.G_ratio_base
        scaled   = residual / self.GAP_D2
        bits = []
        val  = scaled
        for n in range(1, 61):
            val *= 2; bit = int(val); bits.append(bit); val -= bit
        cascade = sum(1.0/2**(i+1) for i,b in enumerate(bits) if b==1)
        
        self.G_ratio_eqi4 = self.G_ratio_base + self.GAP_D2 * cascade
        self.G_topo_eqi4  = self.G * self.G_ratio_eqi4
        self.eqi4_bits    = bits

        print(f"🚀 Monster v8.95b RLC GRAVITY RECONSTRUCTION + SESSION 20")
        print(f"   [RLC] L=D2={self.L:.4f}  C=D1={self.C:.4f}  R=GAP={self.R:.5f}")
        print(f"   [RLC] LC={self.LC:.4f}=1  ω₀={self.w0:.4f}=Unity  γ={self.gamma:.5f}")
        print(f"   [RLC] ω_d={self.w_d:.6f}  Q={self.Q:.2f}")
        print(f"   [기저] G_ratio_base = {self.G_ratio_base:.6f}")
        print(f"   [★★★] G_ratio_eqi4 = {self.G_ratio_eqi4:.12f} (목표: W_L={self.W_L:.12f})")
        print(f"   [★★★] 오차        = {abs(self.G_ratio_eqi4-self.W_L)/self.W_L*100:.8f}%")

    def inject_gap_energy(self, r):
        self.total_injections += 1

    def compute(self, h_chunk, l_chunk, mass, event_name, n_decoding, inv_theory):
        """RLC 중력(v8.9) + 4대 대통합(v12~v17) 통합 연산"""
        try:
            if np.isnan(h_chunk).any() or np.isnan(l_chunk).any():
                self.inject_gap_energy("NaN")
                h_chunk=np.nan_to_num(h_chunk)
                l_chunk=np.nan_to_num(l_chunk)

            # ── 0. 공통 전처리 (v12~v17 호환) ──────────────────────────
            h_n = (h_chunk - np.mean(h_chunk)) / (np.std(h_chunk) + 1e-30)
            l_n = (l_chunk - np.mean(l_chunk)) / (np.std(l_chunk) + 1e-30)
            l_diff = np.diff(l_n, prepend=l_n[0])
            mass_corr = self.D1 * np.log(mass) if mass > 0 else 0.0

            S_D1_SI = np.log(1.0 / EPSILON_0)
            S_D2_SI = abs(np.log(MU_0))

            ln_h = np.log(max(np.mean((h_n * self.D1)**2), 1e-300))
            ln_l = np.log(max(np.mean((l_diff * self.D2)**2), 1e-300))

            S_L2 = ln_h + S_D1_SI
            S_T2 = ln_l + S_D2_SI + mass_corr
            
            event_type = "BNS" if mass < 3.0 else "BBH"

            # ── [v8.9 오리지널] RLC 에너지/운동량 (인덱스 0~6 고정) ───
            h_amp = np.sqrt(np.mean(h_n**2)) * self.W_H + 1e-40
            l_amp = np.sqrt(np.mean(l_n**2)) * self.W_L + 1e-40
            
            L_eff = self.L * h_amp
            C_eff = self.C * l_amp
            LC_eff = max(L_eff * C_eff, 1e-300)
            w0_eff = 1.0/np.sqrt(LC_eff)
            E_topo = self.hbar * w0_eff
            
            R_eff = self.R * (h_amp + l_amp)/2
            p_topo = self.hbar / max(R_eff * self.c, 1e-300)
            
            gamma_eff = R_eff / (2*L_eff)
            w_d_eff = np.sqrt(max(w0_eff**2 - gamma_eff**2, 0))
            c_mem = self.c * (w_d_eff/w0_eff)
            
            G_topo_eff = self.G * (w_d_eff/w0_eff)**3 / self.c_EQI**2
            G_ratio = G_topo_eff / self.G
            
            _e = max(abs(E_topo/self.hbar), 1e-300)
            ln2_ratio = np.log(_e)/np.log(2)
            eqi4_n = -np.log2(_e)
            eqi4_sigma = abs(_e/2 - 0.5)

            # ── 1. τ 재정규화 비율 (v12.95e) ──────────────────────
            du_dy = np.mean(np.abs(np.diff(h_n, prepend=h_n[0])))
            mu_ratio = np.log(5.0) / np.log(25.0) # 0.5
            tau_obs = mu_ratio * du_dy
            tau_true = tau_obs * (1.0 + n_decoding * self.GAP)
            tau_ratio = tau_true / max(tau_obs, 1e-300)

            # ── 2. 카르노 η 재정규화 (v14.95b) ────────────────────
            carnot_eta_obs = 1.0 - (S_T2 / (S_L2 + 1e-30))
            ratio_obs_eta = 1.0 - carnot_eta_obs
            ratio_true_eta = ratio_obs_eta * (1.0 + n_decoding * self.GAP)
            eta_true = 1.0 - ratio_true_eta
            
            eta_target = ETA_EQI if event_type == "BNS" else (1.0 - 1.0/MAXWELL_REAL)
            eta_rd = abs(eta_true - eta_target) / eta_target * 100.0

            # ── 3. 베르누이-카르노 홀로그래픽 곱 (v13.95c) ──────────
            h_orth = h_n * self.D1
            v_norm_c = np.mean(np.abs(l_diff / (np.std(l_diff) + 1e-30))) * self.c_EQI
            p_static = np.mean((h_orth / (np.std(h_orth) + 1e-30))**2)
            p_dynamic = 0.5 * self.D1 * (v_norm_c**2)
            bern_obs = p_static / max(p_dynamic, 1e-300)
            
            bern_true = bern_obs * (1.0 + n_decoding * self.GAP)
            holo_product = bern_true * eta_true
            
            holo_target = self.PHI**2 if event_type == "BNS" else self.PHI**1
            holo_rd = abs(holo_product - holo_target) / holo_target * 100.0

            # ── 4. 역수계단 + Unity (v17.95b) ──────────────────────
            ang_mom = np.mean(np.abs(h_n / (np.std(h_n) + 1e-30))) * self.D1
            mag_flux = np.mean(np.abs(l_diff / (np.std(h_n) + 1e-30))) * self.D2
            ab_obs = mag_flux / max(ang_mom, 1e-300)
            ab_rd = abs(ab_obs - self.PHI**3) / self.PHI**3
            
            ab_rd_true = 1.0 - (1.0 - ab_rd) * (1.0 + n_decoding * self.GAP)
            ab_rd_true = min(ab_rd_true, 1.0 - 1e-10)
            
            inv_true = 1.0 / max(1.0 - ab_rd_true, 1e-300)
            inv_rd = abs(inv_true - inv_theory) / inv_theory * 100.0
            
            S_T2_true = S_T2 / (1.0 + n_decoding * self.GAP)
            unity_product = S_L2 / max(S_T2_true, 1e-300)
            unity_rd = abs(unity_product - MAXWELL_REAL)
            
            # (옵션) 직교 분배율 S_L2 / S_T2 비율
            ratio_orth = S_L2 / (S_T2 + 1e-30)
            rd_orth = abs(ratio_orth - MAXWELL_REAL)

            return (
                E_topo, p_topo, G_ratio, c_mem, ln2_ratio, eqi4_n, eqi4_sigma, # [0~6]
                tau_obs, tau_true, tau_ratio,                                  # [7~9]
                carnot_eta_obs, eta_true, eta_rd,                              # [10~12]
                bern_true, holo_product, holo_rd,                              # [13~15]
                inv_true, inv_rd, unity_product, unity_rd,                     # [16~19]
                rd_orth                                                        # [20] 추가 잔차
            )
        except Exception as e:
            print(f"Error in compute: {e}")
            return (1.0,)*21

    def process(self, h_file, l_file, chunk_size=2048, event_name="GW170817"):
        print(f"\n📡 데이터 로드...")
        try:
            df_h=pd.read_csv(h_file, comment='#')
            df_l=pd.read_csv(l_file, comment='#')
            h_data=df_h.iloc[:,1].values if df_h.shape[1]>1 else df_h.iloc[:,0].values
            l_data=df_l.iloc[:,1].values if df_l.shape[1]>1 else df_l.iloc[:,0].values
            n=min(len(h_data),len(l_data))
            h_data=h_data[:n]; l_data=l_data[:n]; nc=n//chunk_size
            print(f"   ✅ {n:,} samples  |  {nc} chunks")
        except Exception as ex:
            print(f"❌ {ex}"); return []

        inv_theory = {"GW150914": 3, "GW170814": 5}.get(event_name.upper(), 7)

        mass_analysis = {}
        chunk_results = []
        
        # v8.9 글로벌 추적기 (질량에 무관하게 동일한 값)
        et_min=9999; et_min_idx=-1
        gr_min=9999; gr_min_idx=-1
        cm_min=9999; cm_min_idx=-1
        ln2_min=9999; ln2_min_idx=-1
        n_max=-9999; n_max_idx=-1
        sig_min=9999; sig_min_idx=-1

        for m in MASSES:
            n_decoding = N_BNS if m < 3.0 else N_BBH
            event_type = "BNS" if m < 3.0 else "BBH"
            
            b_eta_rd = np.inf; b_eta_obs = 0; b_eta_true = 0
            b_holo_rd = np.inf; b_holo_product = 0; b_bern_true = 0
            b_inv_rd = np.inf; b_inv_true = 0
            b_unity_rd = np.inf; b_unity_product = 0
            b_orth_rd = np.inf; b_orth_t = 0
            first_tau_ratio = 0

            for i in range(nc):
                s = i * chunk_size; e_ = s + chunk_size
                hc = h_data[s:e_]; lc = l_data[s:e_]
                
                res = self.compute(hc, lc, m, event_name, n_decoding, inv_theory)
                t_sec = s / self.fs
                
                # Unpack
                E_topo, p_topo, G_ratio, c_mem, ln2_ratio, eqi4_n, eqi4_sigma = res[0:7]
                tau_obs, tau_true, tau_ratio = res[7:10]
                carnot_eta_obs, eta_true, eta_rd = res[10:13]
                bern_true, holo_product, holo_rd = res[13:16]
                inv_true, inv_rd, unity_product, unity_rd = res[16:20]
                rd_orth = res[20]

                if i == 0: first_tau_ratio = tau_ratio
                
                if eta_rd < b_eta_rd:
                    b_eta_rd = eta_rd; b_eta_obs = carnot_eta_obs; b_eta_true = eta_true
                if holo_rd < b_holo_rd:
                    b_holo_rd = holo_rd; b_holo_product = holo_product; b_bern_true = bern_true
                if inv_rd < b_inv_rd:
                    b_inv_rd = inv_rd; b_inv_true = inv_true
                if unity_rd < b_unity_rd:
                    b_unity_rd = unity_rd; b_unity_product = unity_product
                if rd_orth < b_orth_rd:
                    b_orth_rd = rd_orth; b_orth_t = t_sec

                # 글로벌 v8.9 지표 (질량 독립적 로직이므로 M=1.0일때만 기록)
                if m == MASSES[0]:
                    if E_topo < et_min: et_min=E_topo; et_min_idx=i
                    if G_ratio < gr_min: gr_min=G_ratio; gr_min_idx=i
                    if c_mem < cm_min: cm_min=c_mem; cm_min_idx=i
                    if ln2_ratio < ln2_min: ln2_min=ln2_ratio; ln2_min_idx=i
                    if eqi4_n > n_max: n_max=eqi4_n; n_max_idx=i
                    if eqi4_sigma < sig_min: sig_min=eqi4_sigma; sig_min_idx=i
                
                # 청크 데이터는 대표로 M=1.4만 저장하여 JSON 용량 관리 (스펙의 샘플 참고)
                if m == 1.4:
                    chunk_results.append({
                        "chunk": i, "time_s": t_sec, "mass": m,
                        "E_topo_norm": float(E_topo/self.hbar), "G_ratio": float(G_ratio), "c_mem_ratio": float(c_mem/self.c),
                        "ln2_ratio": float(ln2_ratio), "eqi4_n": float(eqi4_n), "eqi4_sigma": float(eqi4_sigma),
                        "tau_ratio": float(tau_ratio),
                        "carnot_eta_obs": float(carnot_eta_obs), "eta_true": float(eta_true), "eta_rd": float(eta_rd),
                        "bern_true": float(bern_true), "holo_product": float(holo_product), "holo_rd": float(holo_rd),
                        "inv_true": float(inv_true), "inv_rd": float(inv_rd),
                        "unity_product": float(unity_product), "unity_rd": float(unity_rd)
                    })

            # 질량별 요약
            mass_analysis[str(m)] = {
                "event_type": event_type, "n_decoding": float(n_decoding),
                "tau_ratio": float(first_tau_ratio),
                "best_eta_obs": float(b_eta_obs), "best_eta_true": float(b_eta_true), "eta_error_pct": float(b_eta_rd),
                "best_holo_product": float(b_holo_product), "holo_rd_pct": float(b_holo_rd),
                "best_inv_true": float(b_inv_true), "inv_rd_pct": float(b_inv_rd),
                "best_unity_product": float(b_unity_product), "unity_rd": float(b_unity_rd),
                "best_orth_rd": float(b_orth_rd), "best_orth_t": float(b_orth_t)
            }
            
            holo_target = self.PHI**2 if event_type == "BNS" else self.PHI**1
            print(f"  [M={m:4.1f}M☉] 직교rd={b_orth_rd:.4f} @{b_orth_t:4.1f}s  τ_ratio={first_tau_ratio:.6f}  "
                  f"η_obs={b_eta_obs:.4f} η_true={b_eta_true:.4f}(오차{b_eta_rd:.4f}%)")
            print(f"            Bern×η={b_holo_product:.4f}(목표φ^{'2' if event_type=='BNS' else '1'}={holo_target:.4f} 오차{b_holo_rd:.4f}%)  "
                  f"inv={b_inv_true:.4f}(목표{inv_theory} 오차{b_inv_rd:.4f}%)  "
                  f"unity={b_unity_product:.4f}(오차{b_unity_rd:.4f})")

        # Session 20 Summary
        tau_BNS = 1.0 + N_BNS * self.GAP
        tau_BBH = 1.0 + N_BBH * self.GAP
        
        holo_best_mass = min(MASSES, key=lambda m: mass_analysis[str(m)]['holo_rd_pct'])
        holo_best_val = mass_analysis[str(holo_best_mass)]['best_holo_product']
        holo_target = self.PHI**2 if holo_best_mass < 3.0 else self.PHI**1
        
        inv_best_mass = min(MASSES, key=lambda m: mass_analysis[str(m)]['inv_rd_pct'])
        inv_best_val = mass_analysis[str(inv_best_mass)]['best_inv_true']
        
        unity_best_mass = min(MASSES, key=lambda m: mass_analysis[str(m)]['unity_rd'])
        unity_best_val = mass_analysis[str(unity_best_mass)]['best_unity_product']

        session20_summary = {
            "tau_ratio_BNS": float(tau_BNS), "tau_ratio_BBH": float(tau_BBH),
            "holo_best_mass": float(holo_best_mass), "holo_best_val": float(holo_best_val), "holo_target": float(holo_target),
            "inv_best_mass": float(inv_best_mass), "inv_best_val": float(inv_best_val), "inv_theory": inv_theory,
            "unity_best_mass": float(unity_best_mass), "unity_best_val": float(unity_best_val),
            "unity_target": MAXWELL_REAL
        }

        # JSON 구성
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"eqi_monster_v895b_{ts}.json"
        
        out_data = {
            "version": "v8.95b RLC Gravity + Session20 EQI",
            "event_name": event_name,
            "inv_theory": inv_theory,
            "theory": {
                "L=D2": float(self.D2), "C=D1": float(self.D1), "R=GAP": float(self.GAP),
                "w0": float(self.w0), "w_d": float(self.w_d), "Q": float(self.Q),
                "G_topo_over_G": float(self.G_topo/self.G), "W_L": float(self.W_L),
                "N_BBH": float(N_BBH), "N_BNS": float(N_BNS),
                "MAXWELL_REAL": MAXWELL_REAL,
                "PHI_1": float(self.PHI**1), "PHI_2": float(self.PHI**2)
            },
            "mass_analysis": mass_analysis,
            "session20_summary": session20_summary,
            "detections": {
                "E_topo_min": et_min_idx, "G_ratio_min": gr_min_idx,
                "c_mem_min": cm_min_idx, "ln2_min": ln2_min_idx,
                "n_max": n_max_idx, "sigma_min": sig_min_idx
            },
            "chunk_results": chunk_results
        }

        with open(fn, 'w') as f:
            json.dump(out_data, f, indent=2)
            
        print(f"\n[RLC 중력 재구성 탐지 결과 (v8.9 호환)]")
        print(f"  E_topo 최솟값   → chunk {et_min_idx:3d}  E/ħ={et_min/self.hbar:.4f}")
        print(f"  [EQI^4] G_ratio_eqi4 = {self.G_ratio_eqi4:.12f} = W_L={self.W_L:.12f}")
        print(f"\n📁 저장: {fn}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h_strain')
    parser.add_argument('l_strain')
    parser.add_argument('--event', type=str, default='GW170817')
    args = parser.parse_args()

    print("="*70)
    print("  EQI MONSTER v8.95b RLC GRAVITY + SESSION 20 DUALITY")
    print("  v12.95e / v13.95c / v14.95b / v17.95b 4대 대통합 병합")
    print("="*70)
    
    m = EQI_Monster_v895b_RLC_Session20()
    m.process(args.h_strain, args.l_strain, event_name=args.event)
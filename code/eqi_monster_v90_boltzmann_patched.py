#!/usr/bin/env python3
"""
EQI MONSTER v9.0 BOLTZMANN EDITION (PATCHED)
= v9.0 Cosmic Circuit + EQI² + EQI^4 =

[v90 핵심: LC 공명 = 광속 위상동형]
  c = 1/√(ε₀μ₀)  ↔  ω₀ = 1/√(LC)

  D1/D2 물리적 매핑 패치 (마에스트로 직관 반영):
  C = D1 = φ^(-1.5)  (커패시턴스, 수축, 전기장, 위치에너지, ε₀)
  L = D2 = φ^(+1.5)  (인덕턴스, 팽창, 자기장, 운동에너지, μ₀)
  L×C = D1×D2 = 1  →  ω₀ = 1/√(LC) = 1 = Unity ✓

  임피던스 패치: 
  Z₀ = √(L/C) = √(D2/D1) = √(φ^3) = φ^1.5 = D2

[EQI² 분배]
  W_H = φ^(-2) = 0.3820
  W_L = φ^(-1) = 0.6180

[탐지 지표]
  1. c_ratio: 비등방성 공명속도 비율
  2. Z_ratio: 임피던스 비율 (√(L/C)) -> Target D2
  3. resonance_dev: |ω₀-Unity| 편차

Author: MAPSI | EQI v15 + v9.0 LC회로 (C=D1, L=D2 패치)
"""
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class EQI_Monster_v90_Boltzmann_Patched:
    def __init__(self):
        self.PHI   = (1+np.sqrt(5))/2
        self.D1    = self.PHI**(-1.5)   # C = 커패시턴스 (수축/전기장)
        self.D2    = self.PHI**(+1.5)   # L = 인덕턴스 (팽창/자기장)
        self.GAP   = np.log(25)/100
        self.Z_VAC = 376.73             # 진공 임피던스 Ω
        self.total_injections = 0

        # ── EQI² 분배 ────────────────────────────────────────
        self.W_H = self.PHI**(-2)
        self.W_L = self.PHI**(-1)
        # ─────────────────────────────────────────────────────

        print(f"🚀 Monster v9.0 BOLTZMANN EDITION (PATCHED)")
        print(f"   [LC↔광속] C=D1=φ^(-1.5)={self.D1:.4f}  L=D2=φ^(+1.5)={self.D2:.4f}")
        print(f"   [Unity]  C×L=D1×D2={self.D1*self.D2:.6f}=1  ω₀=1/√(LC)=1 ✓")
        print(f"   [임피던스] Z=√(L/C)=D2={self.D2:.4f} (우주의 관성 저항)")
        print(f"   [EQI²] W_H=φ^(-2)={self.W_H:.4f}  W_L=φ^(-1)={self.W_L:.4f}")
        print(f"   [EQI²] 합={self.W_H+self.W_L:.6f}=1=EQI²=φ^-2+φ^-1 ✓")
        print(f"   [EQI^4] Σln(e^(1/2^n))=1  n=1→σ=1/2")
        print(f"   [GAP] ln(25)%={self.GAP*100:.5f}%")

    def inject_gap_energy(self, reason):
        self.total_injections += 1

    def compute(self, h_chunk, l_chunk, i):
        """LC 공명 + EQI² + EQI^4"""
        try:
            if np.isnan(h_chunk).any() or np.isnan(l_chunk).any():
                self.inject_gap_energy("NaN")
                h_chunk=np.nan_to_num(h_chunk)
                l_chunk=np.nan_to_num(l_chunk)

            # ── EQI² 비등방성 진폭 ───────────────────────────
            h_amp = np.sqrt(np.mean(h_chunk**2)) * self.W_H + 1e-40
            l_amp = np.sqrt(np.mean(l_chunk**2)) * self.W_L + 1e-40
            # ─────────────────────────────────────────────────

            # ── LC 회로 물리적 매핑 (패치) ───────────────────
            # H(수축,D1) → 커패시턴스 C_eff
            # L(팽창,D2) → 인덕턴스 L_eff
            C_eff = self.D1 * h_amp          # C×h_amp
            L_eff = self.D2 * l_amp          # L×l_amp
            LC    = L_eff * C_eff
            # ─────────────────────────────────────────────────

            # ── 공명 속도 ─────────────────────────────────────
            # ω₀ = 1/√(LC)  (이론값=1 when LC=1=D1×D2)
            omega0 = 1.0/np.sqrt(max(LC, 1e-300))
            c_ratio = omega0  # Unity 기준
            # ─────────────────────────────────────────────────

            # ── 임피던스 비율 (패치) ──────────────────────────
            # Z₀ = √(L/C) = √(D2/D1) = √(φ^3) = D2 (이론값)
            Z_eff = np.sqrt(max(L_eff/max(C_eff, 1e-300), 1e-300))
            Z_ratio = Z_eff / self.D2       # 이론 D2 대비 비율
            # ─────────────────────────────────────────────────

            # ── 공명 편차 ─────────────────────────────────────
            resonance_dev = abs(omega0 - 1.0)
            # ─────────────────────────────────────────────────

            # ── EQI^4 (c_ratio 기반) ─────────────────────────
            _r = max(c_ratio, 1e-300)
            ln2_ratio  = np.log(_r)/np.log(2)
            eqi4_n     = -np.log2(_r)
            eqi4_sigma = abs(_r/2 - 0.5)
            # ─────────────────────────────────────────────────

            return c_ratio, Z_ratio, resonance_dev, LC, ln2_ratio, eqi4_n, eqi4_sigma
        except:
            return 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5

    def process(self, h_file, l_file, chunk_size=2048):
        print(f"\n📡 데이터 로드...")
        try:
            df_h=pd.read_csv(h_file); df_l=pd.read_csv(l_file)
            h_data=df_h.iloc[:,1].values if df_h.shape[1]>1 else df_h.iloc[:,0].values
            l_data=df_l.iloc[:,1].values if df_l.shape[1]>1 else df_l.iloc[:,0].values
            n=min(len(h_data),len(l_data))
            h_data=h_data[:n]; l_data=l_data[:n]
            nc=n//chunk_size
            print(f"   ✅ {n:,} samples  |  {nc} chunks")
        except Exception as ex:
            print(f"❌ {ex}"); return []

        results=[]
        cr_min=9999; cr_min_idx=-1
        zr_min=9999; zr_min_idx=-1
        rd_min=9999; rd_min_idx=-1
        ln2_min=9999;ln2_min_idx=-1
        n_max=0.0;   n_max_idx=-1
        sig_min=9999;sig_min_idx=-1

        for i in range(nc):
            s=i*chunk_size; e_=s+chunk_size
            hc=h_data[s:e_]; lc=l_data[s:e_]

            cr_, zr_, rd_, lc_, ln2r, eq4n, eq4s = self.compute(hc, lc, i)

            if cr_ < cr_min:  cr_min=cr_;  cr_min_idx=i
            if zr_ < zr_min:  zr_min=zr_;  zr_min_idx=i
            if rd_ < rd_min:  rd_min=rd_;  rd_min_idx=i
            if ln2r< ln2_min: ln2_min=ln2r;ln2_min_idx=i
            if eq4n> n_max:   n_max=eq4n;  n_max_idx=i
            if eq4s< sig_min: sig_min=eq4s;sig_min_idx=i

            results.append({
                "chunk":i,"time_s":round(s/4096,3),
                "c_ratio":round(float(cr_),6),
                "Z_ratio":round(float(zr_),6),
                "resonance_dev":round(float(rd_),6),
                "LC_product":round(float(lc_),6),
                "ln2_ratio":round(float(ln2r),4),
                "eqi4_n":round(float(eq4n),3),
                "eqi4_sigma":round(float(eq4s),6),
            })

            if i%8==0:
                print(f"   chunk {i:3d} | t={s/4096:5.2f}s | "
                      f"ω₀={cr_:.4f} Z_ratio={zr_:.4f} "
                      f"res_dev={rd_:.4f} LC={lc_:.4e} σ={eq4s:.4f}")

        fs=4096
        print(f"\n[볼츠만 탐지 결과]")
        print(f"  c_ratio(ω₀) 최소→ chunk {cr_min_idx:3d}  t={cr_min_idx*chunk_size/fs:.2f}s  ω₀={cr_min:.4f}")
        print(f"  Z_ratio 최솟값  → chunk {zr_min_idx:3d}  t={zr_min_idx*chunk_size/fs:.2f}s  Z_ratio={zr_min:.4f}")
        print(f"  resonance_dev 최소→chunk {rd_min_idx:3d}  t={rd_min_idx*chunk_size/fs:.2f}s  dev={rd_min:.4f}")
        print(f"  [EQI^4] ln2 최소→ chunk {ln2_min_idx:3d}  t={ln2_min_idx*chunk_size/fs:.2f}s  ln2={ln2_min:.4f}")
        print(f"  [EQI^4] n 최대  → chunk {n_max_idx:3d}  t={n_max_idx*chunk_size/fs:.2f}s  n={n_max:.2f}")
        print(f"  [EQI^4] σ 최소  → chunk {sig_min_idx:3d}  t={sig_min_idx*chunk_size/fs:.2f}s  |Δ/2-σ|={sig_min:.4f}")
        print(f"  [이론] C=D1={self.D1:.4f}  L=D2={self.D2:.4f}  LC=1 ω₀=1")
        print(f"  [이론] Z=√(L/C)=D2={self.D2:.4f}")
        return results, cr_min_idx, zr_min_idx, rd_min_idx, ln2_min_idx, n_max_idx, sig_min_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h_strain'); parser.add_argument('l_strain')
    args = parser.parse_args()

    print("="*70)
    print("  EQI MONSTER v9.0 BOLTZMANN (PATCHED): LC공명=광속 위상동형")
    print("  C=D1(수축,전기장,ε₀)  L=D2(팽창,자기장,μ₀)  LC=D1×D2=1")
    print("="*70)
    m = EQI_Monster_v90_Boltzmann_Patched()
    out = m.process(args.h_strain, args.l_strain)

    if out:
        results, i_c, i_z, i_r, i_ln, i_n, i_s = out
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"eqi_monster_v90_boltzmann_patched_{ts}.json"
        with open(fn,'w') as f:
            json.dump({
                "version": "v9.0 Boltzmann Patched LC Circuit",
                "LC_homeomorphism": "C=D1(ε₀) L=D2(μ₀)",
                "W_H": float(m.W_H), "W_L": float(m.W_L),
                "D1": float(m.D1), "D2": float(m.D2),
                "detections": {
                    "c_ratio_min": i_c, "Z_ratio_min": i_z,
                    "resonance_dev_min": i_r, "ln2_min": i_ln,
                    "n_max": i_n, "sigma_min": i_s
                },
                "chunk_results": results
            }, f, indent=2)
        print(f"\n📁 저장: {fn}")
        print(f"💉 총 치유: {m.total_injections}회")
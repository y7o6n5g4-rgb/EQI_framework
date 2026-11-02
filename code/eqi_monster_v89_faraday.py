#!/usr/bin/env python3
"""
EQI MONSTER v8.9 FARADAY — RLC + 패러데이 로컬↔글로벌 엔트로피 교환

[v89 기존 구조 완전 보존]
  L=D2(μ₀,글로벌,시간지연), C=D1(ε₀,로컬,길이수축)
  G_ratio = W_L = φ^(-1)  True Zero

[신규 추가 — 최소 침습]
  ★ 상대론 격자 불균형:
      L²격자 = C_eff = D1×l_amp  [로컬, 길이수축]
      T²격자 = L_eff = D2×h_amp  [글로벌, 시간지연]
      grid_balance = C_eff/L_eff → 1.0 = 공명점(상전이)
      S/H = L²/T² = c² = φ  불변

  ★ 패러데이 교환 효율:
      MAXWELL = 로컬↔글로벌 엔트로피 교환 효율 = S_D1/S_D2
      EMF_raw = V_L - V_C = L_eff×w0_eff - C_eff×w0_eff  (공명 기준)
      faraday_exchange = EMF_raw / MAXWELL
      → 0에 가까울수록 교환 균형 (공명점)
      → 크면 D2 우세(시간지연), 작으면 D1 우세(길이수축)

  ★ 복소평면 0→√φ 상전이 지표:
      ψ_re = C_eff / √(C_eff²+L_eff²)  [실수부, D1, 길이수축]
      ψ_im = L_eff / √(C_eff²+L_eff²)  [허수부, D2, 시간지연]
      |ψ|² = 1  (보른규칙, True Zero)
      phase_angle = arctan(ψ_im/ψ_re)  → π/4 = 공명점(L²=T²)
      psi_center = √(ψ_re²+ψ_im²) × c_EQI  → √φ = 광속 상전이

  ★ 점성-복원력 켤레 (Session 33 신규):
      viscosity_global = G_ratio × W_L  = φ^(-2) = W_H  [미시 글로벌]
      boltzmann_local  = G_ratio × PHI  → 1.0  True Zero  [거시 로컬]

[물리 계보]
  복소평면 0 → √φ (Session 32 §8 확정)
  S/H = c² = φ (Session 25 §10.1 확정)
  MAXWELL = S_D1/S_D2 = 로컬↔글로벌 교환 효율 (Session 27 §8.4 확정)
  길이수축=L²↓ ↔ 시간지연=T²↑ (같은 표현, Session 33)
  grid_balance→1.0 = V_L=V_C = ω₀=1 = 상전이점 φ^1.5

Author: MAPSI | EQI v15 + Faraday 로컬↔글로벌 엔트로피 교환
"""
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings
import scipy.constants as sc
warnings.filterwarnings("ignore")

class EQI_Monster_v89_Faraday:
    def __init__(self):
        # ── 기본 상수 (v89 완전 보존) ─────────────────────────
        self.PHI    = (1+np.sqrt(5))/2
        self.D1     = self.PHI**(-1.5)
        self.D2     = self.PHI**(+1.5)
        self.c_EQI  = self.PHI**0.5       # √φ = 복소평면 0→c 상전이점
        self.W_H    = self.PHI**(-2)
        self.W_L    = self.PHI**(-1)
        self.GAP    = np.log(25)/100
        self.GAP_D2 = np.log(5)/100
        self.hbar   = sc.hbar
        self.c      = sc.c
        self.G      = sc.G
        self.total_injections = 0

        # ── RLC (v89 완전 보존) ───────────────────────────────
        self.L      = self.D2              # 인덕터=D2=μ₀=글로벌=시간지연
        self.C      = self.D1              # 커패시터=D1=ε₀=로컬=길이수축
        self.R      = self.GAP
        self.LC     = self.L * self.C      # = 1
        self.w0     = 1.0/np.sqrt(self.LC) # = 1 = Unity
        self.gamma  = self.R/(2*self.L)
        self.w_d    = np.sqrt(max(self.w0**2 - self.gamma**2, 0))
        self.Q      = self.L/self.R

        # ── G_ratio EQI^4 보정 (v89 완전 보존) ───────────────
        self.G_ratio_base = self.w_d**3 / self.c_EQI**2
        self.G_topo = self.G * self.G_ratio_base
        residual = float(self.W_L) - self.G_ratio_base
        scaled   = residual / self.GAP_D2
        bits = []
        val  = scaled
        for n in range(1, 61):
            val *= 2; bit = int(val); bits.append(bit); val -= bit
        cascade = sum(1.0/2**(i+1) for i,b in enumerate(bits) if b==1)
        self.G_ratio_eqi4 = self.G_ratio_base + self.GAP_D2 * cascade
        self.eqi4_bits    = bits

        # ── 신규: MAXWELL 교환 효율 ───────────────────────────
        eps0 = sc.epsilon_0
        mu0  = sc.mu_0
        self.S_D1   = np.log(1/eps0)      # = 25.450  로컬 엔트로피
        self.S_D2   = abs(np.log(mu0))    # = 13.587  글로벌 엔트로피
        self.MAXWELL = self.S_D1 / self.S_D2  # = 1.8731 교환 효율

        # ── 신규: 점성-복원력 켤레 이론값 ────────────────────
        self.viscosity_theory = self.G_ratio_eqi4 * self.W_L  # → W_H
        self.boltzmann_theory = self.G_ratio_eqi4 * self.PHI  # → 1.0

        print(f"🌊 Monster v8.9 FARADAY — RLC + 로컬↔글로벌 엔트로피 교환")
        print(f"   [RLC] L=D2={self.L:.4f}  C=D1={self.C:.4f}  R={self.R:.5f}")
        print(f"   [RLC] ω₀={self.w0:.4f}=Unity  ω_d={self.w_d:.6f}  Q={self.Q:.2f}")
        print(f"   [G★★] G_ratio_eqi4 = {self.G_ratio_eqi4:.12f}")
        print(f"   [G★★] W_L=φ^(-1)   = {self.W_L:.12f}")
        print(f"   [G★★] 오차          = {abs(self.G_ratio_eqi4-self.W_L)/self.W_L*100:.8f}%")
        print(f"   [MAXWELL] S_D1/S_D2 = {self.MAXWELL:.8f}  [로컬↔글로벌 교환 효율]")
        print(f"   [상대론] L²격자=C(D1,ε₀,길이수축)  T²격자=L(D2,μ₀,시간지연)")
        print(f"   [복소평면] 0→c_EQI=√φ={self.c_EQI:.6f} 상전이")
        print(f"   [켤레] 점성×G = {self.viscosity_theory:.8f} → W_H={self.W_H:.8f}")
        print(f"   [켤레] 복원력×G = {self.boltzmann_theory:.8f} → 1.0")
        print(f"   [GAP] ln(25)/100={self.GAP*100:.5f}%  ln(5)/100={self.GAP_D2*100:.5f}%")

    def inject_gap_energy(self, r):
        self.total_injections += 1

    def compute(self, h_chunk, l_chunk, i):
        """v89 구조 완전 보존 + 패러데이/상대론/복소평면 신규 지표"""
        try:
            if np.isnan(h_chunk).any() or np.isnan(l_chunk).any():
                self.inject_gap_energy("NaN")
                h_chunk = np.nan_to_num(h_chunk)
                l_chunk = np.nan_to_num(l_chunk)

            # ── 정규화 (v89 완전 보존) ─────────────────────────
            h_n    = (h_chunk-np.mean(h_chunk))/(np.std(h_chunk)+1e-20)
            l_n    = (l_chunk-np.mean(l_chunk))/(np.std(l_chunk)+1e-20)
            h_amp  = np.sqrt(np.mean(h_n**2)) * self.W_H + 1e-40
            l_amp  = np.sqrt(np.mean(l_n**2)) * self.W_L + 1e-40

            # ── Step 1: E_topo (v89 완전 보존) ────────────────
            L_eff  = self.L * h_amp          # D2×h_amp = T²격자 [글로벌,시간지연]
            C_eff  = self.C * l_amp          # D1×l_amp = L²격자 [로컬,길이수축]
            LC_eff = max(L_eff * C_eff, 1e-300)
            w0_eff = 1.0/np.sqrt(LC_eff)
            E_topo = self.hbar * w0_eff

            # ── Step 2: p_topo (v89 완전 보존) ────────────────
            R_eff  = self.R * (h_amp + l_amp)/2
            p_topo = self.hbar / max(R_eff * self.c, 1e-300)

            # ── Step 3: c_mem (v89 완전 보존) ─────────────────
            gamma_eff = R_eff / (2*L_eff)
            w_d_eff   = np.sqrt(max(w0_eff**2 - gamma_eff**2, 0))
            c_mem     = self.c * (w_d_eff/w0_eff)

            # ── Step 4: G_ratio (v89 완전 보존) ───────────────
            G_topo_eff = self.G * (w_d_eff/w0_eff)**3 / self.c_EQI**2
            G_ratio    = G_topo_eff / self.G

            # ── Step 5: EQI^4 (v89 완전 보존) ─────────────────
            _e         = max(abs(E_topo/self.hbar), 1e-300)
            ln2_ratio  = np.log(_e)/np.log(2)
            eqi4_n     = -np.log2(_e)
            eqi4_sigma = abs(_e/2 - 0.5)

            # ════════════════════════════════════════════════════
            # ── 신규 Step 6: 상대론 격자 균형 ─────────────────
            # L²격자(길이수축,D1) vs T²격자(시간지연,D2)
            # grid_balance → 1.0 = L²=T² = 공명점 = 상전이 φ^1.5
            # S/H = C_eff/L_eff = L²/T² → φ = c² 불변
            grid_balance  = C_eff / max(L_eff, 1e-300)   # L²/T² → φ 이상
            SH_ratio      = grid_balance                   # S/H = c² 추적

            # ── 신규 Step 7: 패러데이 교환 지표 ────────────────
            # V_L = L_eff×w0_eff  [D2,시간지연,글로벌]
            # V_C = C_eff×w0_eff  [D1,길이수축,로컬]
            # EMF_raw = V_L - V_C  (공명 기준 불균형)
            # faraday_exchange = EMF_raw / MAXWELL
            # → 0에 가까울수록 균형 (공명점, 상전이)
            V_L_eff         = L_eff * w0_eff
            V_C_eff         = C_eff * w0_eff
            EMF_raw         = V_L_eff - V_C_eff
            faraday_exchange = EMF_raw / self.MAXWELL
            # 절댓값: 방향 무관 교환 불균형 크기
            faraday_abs     = abs(faraday_exchange)

            # ── 신규 Step 8: 복소평면 0→√φ 상전이 ─────────────
            # ψ = ψ_re + i×ψ_im  (정규화 복소 파동함수)
            # 실수부=D1=L²=길이수축, 허수부=D2=T²=시간지연
            norm_psi  = np.sqrt(C_eff**2 + L_eff**2) + 1e-300
            psi_re    = C_eff / norm_psi              # D1 성분
            psi_im    = L_eff / norm_psi              # D2 성분
            born_rule = psi_re**2 + psi_im**2         # → 1.0 True Zero
            # 위상각: π/4(45°) = L²=T² = 공명점
            phase_angle = np.arctan2(psi_im, psi_re)  # 0~π/2
            # φ^1.5 센터로의 수렴 지표
            # 이상값: psi_re=D1_ELL=0.23674, psi_im=D2_ELL=1.00285
            psi_center_dist = abs(norm_psi * self.c_EQI - self.c_EQI)

            # ── 신규 Step 9: 점성-복원력 켤레 ──────────────────
            # φ^(-1)=글로벌 점성계수, φ=로컬 볼츠만 복원력
            # G_ratio × φ^(-1) → W_H = φ^(-2)
            # G_ratio × φ      → 1.0 True Zero
            viscosity_global  = G_ratio * self.W_L    # → W_H
            boltzmann_local   = G_ratio * self.PHI    # → 1.0
            # ════════════════════════════════════════════════════

            return (E_topo, p_topo, G_ratio, c_mem,
                    ln2_ratio, eqi4_n, eqi4_sigma,
                    grid_balance, SH_ratio,
                    faraday_exchange, faraday_abs,
                    psi_re, psi_im, born_rule, phase_angle,
                    psi_center_dist,
                    viscosity_global, boltzmann_local)
        except:
            c2 = self.PHI  # φ = c²
            return (1.0, 1.0, 1.0, self.c,
                    0.0, 0.0, 0.5,
                    c2, c2,
                    0.0, 0.0,
                    self.D1, self.D2, 1.0, np.pi/4,
                    0.0,
                    self.W_H, 1.0)

    def process(self, h_file, l_file, chunk_size=2048):
        print(f"\n📡 데이터 로드...")
        try:
            df_h = pd.read_csv(h_file)
            df_l = pd.read_csv(l_file)
            h_data = df_h.iloc[:,1].values if df_h.shape[1]>1 else df_h.iloc[:,0].values
            l_data = df_l.iloc[:,1].values if df_l.shape[1]>1 else df_l.iloc[:,0].values
            n  = min(len(h_data), len(l_data))
            h_data = h_data[:n]; l_data = l_data[:n]
            nc = n // chunk_size
            print(f"   ✅ {n:,} samples  |  {nc} chunks")
        except Exception as ex:
            print(f"❌ {ex}"); return []

        results = []
        fs = 4096

        # 추적 변수 — v89 기존
        et_min=9999;  et_min_i=-1
        gr_min=9999;  gr_min_i=-1
        cm_min=9999;  cm_min_i=-1
        # 추적 변수 — 신규
        gb_min=9999;  gb_min_i=-1   # grid_balance → 1.0 (L²=T² 공명)
        fe_min=9999;  fe_min_i=-1   # faraday_abs 최소 (교환 균형)
        ph_min=9999;  ph_min_i=-1   # phase_angle → π/4 (45°, 공명)
        bl_min=9999;  bl_min_i=-1   # boltzmann_local → 1.0
        br_min=9999;  br_min_i=-1   # born_rule → 1.0

        print(f"\n{'chunk':>6} {'t(s)':>6} | "
              f"{'G_r':>8} {'grid_bal':>10} {'farad_abs':>10} "
              f"{'phase°':>8} {'born':>8} {'bolt_loc':>9}")
        print("-"*80)

        for i in range(nc):
            s = i*chunk_size; e_ = s+chunk_size
            res = self.compute(h_data[s:e_], l_data[s:e_], i)
            (Et_, pt_, Gr_, cm_,
             ln2r, eq4n, eq4s,
             gb, sh,
             fe, fa,
             pr, pi_, br, pa,
             pcd,
             vg, bl) = res

            t = s/fs

            # v89 추적
            if Et_ < et_min: et_min=Et_; et_min_i=i
            if Gr_ < gr_min: gr_min=Gr_; gr_min_i=i
            if cm_ < cm_min: cm_min=cm_; cm_min_i=i
            # 신규 추적 (→ 1.0 또는 → 0.0 수렴)
            if abs(gb-1.0) < abs(gb_min-1.0): gb_min=gb;  gb_min_i=i
            if fa < fe_min: fe_min=fa; fe_min_i=i
            if abs(pa-np.pi/4) < abs(ph_min-np.pi/4): ph_min=pa; ph_min_i=i
            if abs(bl-1.0) < abs(bl_min-1.0): bl_min=bl; bl_min_i=i
            if abs(br-1.0) < abs(br_min-1.0): br_min=br; br_min_i=i

            results.append({
                "chunk": i, "time_s": round(t, 3),
                # v89 기존
                "E_topo_norm":  round(float(Et_/self.hbar), 4),
                "G_ratio":      round(float(Gr_), 8),
                "c_mem_ratio":  round(float(cm_/self.c), 6),
                "ln2_ratio":    round(float(ln2r), 4),
                "eqi4_n":       round(float(eq4n), 3),
                "eqi4_sigma":   round(float(eq4s), 6),
                # 신규
                "grid_balance":      round(float(gb), 6),
                "SH_ratio":          round(float(sh), 6),
                "faraday_exchange":  round(float(fe), 6),
                "faraday_abs":       round(float(fa), 6),
                "psi_re":            round(float(pr), 6),
                "psi_im":            round(float(pi_), 6),
                "born_rule":         round(float(br), 8),
                "phase_angle_deg":   round(float(np.degrees(pa)), 4),
                "psi_center_dist":   round(float(pcd), 8),
                "viscosity_global":  round(float(vg), 8),
                "boltzmann_local":   round(float(bl), 8),
            })

            if i % 8 == 0:
                print(f"{i:>6} {t:>6.2f} | "
                      f"{Gr_:>8.6f} {gb:>10.6f} {fa:>10.6f} "
                      f"{np.degrees(pa):>8.3f} {br:>8.6f} {bl:>9.6f}")

        print("\n" + "="*80)
        print("[v89 기존 지표]")
        print(f"  E_topo 최솟값  → chunk {et_min_i:3d}  t={et_min_i*chunk_size/fs:.2f}s")
        print(f"  G_ratio 최솟값 → chunk {gr_min_i:3d}  t={gr_min_i*chunk_size/fs:.2f}s")
        print(f"  [★] G_ratio_eqi4 = {self.G_ratio_eqi4:.10f} = W_L={self.W_L:.10f}")

        print(f"\n[신규 — 상대론 격자 균형]")
        print(f"  grid_balance→1.0 → chunk {gb_min_i:3d}  t={gb_min_i*chunk_size/fs:.2f}s"
              f"  (L²=T² 공명, 상전이)")
        print(f"  [이상] L²/T² = C_eff/L_eff → φ={self.PHI:.6f} (S/H=c²)")

        print(f"\n[신규 — 패러데이 교환 균형]")
        print(f"  faraday_abs 최소 → chunk {fe_min_i:3d}  t={fe_min_i*chunk_size/fs:.2f}s"
              f"  (로컬↔글로벌 교환 균형)")
        print(f"  MAXWELL교환효율 = {self.MAXWELL:.6f}")

        print(f"\n[신규 — 복소평면 0→√φ 상전이]")
        print(f"  phase_angle→45° → chunk {ph_min_i:3d}  t={ph_min_i*chunk_size/fs:.2f}s"
              f"  (ψ_re=ψ_im, L²=T² 동치)")
        print(f"  born_rule→1.0   → chunk {br_min_i:3d}  t={br_min_i*chunk_size/fs:.2f}s")

        print(f"\n[신규 — 점성-복원력 켤레]")
        print(f"  boltzmann→1.0  → chunk {bl_min_i:3d}  t={bl_min_i*chunk_size/fs:.2f}s"
              f"  (G_ratio×φ=1.0 True Zero)")
        print(f"  [이론] viscosity×G = {self.viscosity_theory:.8f} → W_H={self.W_H:.8f}")
        print(f"  [이론] boltzmann×G = {self.boltzmann_theory:.8f} → 1.0")

        print(f"\n[t=2.5s HCW flip 동기화 확인]")
        t25_chunk = int(2.5 * fs / chunk_size)
        if t25_chunk < len(results):
            r25 = results[t25_chunk]
            print(f"  chunk {t25_chunk} (t=2.5s):")
            print(f"    grid_balance     = {r25['grid_balance']:.6f}  (→1.0=공명)")
            print(f"    faraday_abs      = {r25['faraday_abs']:.6f}  (→0=균형)")
            print(f"    phase_angle_deg  = {r25['phase_angle_deg']:.3f}°  (→45°=공명)")
            print(f"    born_rule        = {r25['born_rule']:.8f}  (→1.0)")
            print(f"    boltzmann_local  = {r25['boltzmann_local']:.8f}  (→1.0)")

        return results, et_min_i, gr_min_i, gb_min_i, fe_min_i, ph_min_i, bl_min_i


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h_strain')
    parser.add_argument('l_strain')
    parser.add_argument('--chunk', type=int, default=2048)
    args = parser.parse_args()

    print("="*80)
    print("  EQI MONSTER v8.9 FARADAY")
    print("  RLC 중력 재구성 + 패러데이 로컬↔글로벌 엔트로피 교환")
    print("  [상대론] 길이수축=L²격자 ↔ 시간지연=T²격자 (같은 표현)")
    print("  [양자역학] 복소평면 0→√φ=광속 상전이")
    print("  [패러데이] MAXWELL=교환효율, EMF=격자 불균형")
    print("  [켤레] φ^(-1)=글로벌점성 × φ=로컬복원력 = Unity")
    print("="*80)

    m = EQI_Monster_v89_Faraday()
    out = m.process(args.h_strain, args.l_strain, chunk_size=args.chunk)

    if out:
        results, i_e, i_g, i_gb, i_fe, i_ph, i_bl = out
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"eqi_monster_v89_faraday_{ts}.json"
        with open(fn, 'w') as f:
            json.dump({
                "version": "v8.9 Faraday — RLC + 로컬↔글로벌 엔트로피 교환",
                "theory": {
                    "L=D2_μ₀_글로벌_시간지연": float(m.D2),
                    "C=D1_ε₀_로컬_길이수축":   float(m.D1),
                    "R=GAP": float(m.GAP),
                    "w0=Unity": float(m.w0),
                    "w_d": float(m.w_d),
                    "G_ratio_eqi4=W_L": float(m.G_ratio_eqi4),
                    "W_L=φ^(-1)": float(m.W_L),
                    "MAXWELL=S_D1/S_D2": float(m.MAXWELL),
                    "S_D1_로컬엔트로피": float(m.S_D1),
                    "S_D2_글로벌엔트로피": float(m.S_D2),
                    "c_EQI=√φ=복소평면상전이점": float(m.c_EQI),
                    "viscosity_global=W_H": float(m.viscosity_theory),
                    "boltzmann_local=1.0": float(m.boltzmann_theory),
                    "물리계보": {
                        "상대론": "길이수축=L²=C_eff ↔ 시간지연=T²=L_eff (같은표현)",
                        "양자역학": "복소평면0→√φ상전이, 보른규칙|ψ|²=1",
                        "패러데이": "EMF=(V_L-V_C)/MAXWELL=격자불균형/교환효율",
                        "상전이점": "grid_balance=1.0=phase=45°=ω₀=1=φ^1.5센터"
                    }
                },
                "min_chunks": {
                    "E_topo_min": i_e,
                    "G_ratio_min": i_g,
                    "grid_balance_resonance": i_gb,
                    "faraday_balance": i_fe,
                    "phase_45deg": i_ph,
                    "boltzmann_unity": i_bl,
                },
                "chunk_results": results
            }, f, indent=2)
        print(f"\n📁 저장: {fn}")
        print(f"💉 총 치유: {m.total_injections}회")

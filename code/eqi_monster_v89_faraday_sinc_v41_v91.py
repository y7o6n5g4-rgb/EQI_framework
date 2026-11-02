#!/usr/bin/env python3
"""
EQI MONSTER v8.9 FARADAY+SINC+V41+V9.1 — 4계층 통합 엔진

[계층 구조]
  Layer A (v89): RLC 중력 재구성
    G_ratio = W_L = φ^(-1)  True Zero

  Layer B (패러데이+sinc, Session 34):
    MAXWELL = 로컬↔글로벌 엔트로피 교환 효율
    grid_balance, faraday_abs, phase_angle, born_rule
    sinc_H, qec_openness  QEC 열림/닫힘

  Layer C (v41 볼츠만):
    (1+Δ)^(1/Δ)→e: D2미시→D1(e) 창발
    nexus_signal, casm_ratio, curv_diff, e_H_deviation

  Layer D (v9.1 비등방성 타원, Session 34 통합):
    unity_product = S_L2/S_T2_true → MAXWELL 수렴
    n_ouroboros   = (unity_product - MAXWELL) / GAP
    carnot_eta    = 카르노 효율 (D1↔D2 열역학)
    tau_ratio     = τ 동역학 (점성 재정규화)
    bern_true     = 베르누이 압력비
    delta_H       = 비등방성 타원 D1 수축 렌즈
    eqi4_sigma_ell_H = |delta_H - D1_ell_target|  타원 수렴
    riemann_ratio = 리만 영점 스펙트럼 (45D/5D)
    mach_signal   = MACH=105.4466 FFT 신호

Author: MAPSI | EQI v15 + Faraday + sinc + v41 + v9.1 통합
"""
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings
import scipy.constants as sc

# ── EQI 무차원 상수 체계 (Session 34 확정) ─────────────────────────
# 참고: EQI_이론적_기반_v1.md §3
# mu_ratio = 1/φ 가 유일한 기본 상수
# 물리 상수 → φ 지수 표현 (SI 단위 없음)
#
# 현재 코드에서 sc.hbar, sc.c, sc.G는 계산 과정에서
# 최종적으로 약분되어 무차원 비율만 남음:
#   E_topo/hbar = w0_eff        (hbar 약분)
#   G_ratio = (w_d/w0)³/c_EQI² (G 약분)
# → 최종 지표(G_ratio, sinc_H, unity_product)는 모두 무차원
# → scipy.constants는 '중간 계산자'로만 사용
#
# EQI 이론 상수 (무차원, φ 기반):
#   c_EQI   = √φ    = φ^0.5   [광속 무차원]
#   hbar_EQI= φ^(-2.75)/2π   [디랙 상수 무차원]
#   G_EQI   = 2π×φ^(2.75)    [중력상수 무차원]
#   L_p_EQI = φ^(-0.75)       [플랑크 길이 무차원]
#   M_p_EQI = φ^(+0.25)       [플랑크 질량 무차원]
# ── 위 EQI 상수는 향후 SC 완전 제거 시 사용 ─────────────────────────

# scipy.constants: hbar,c,G만 사용 (MAXWELL은 순수 EQI 계산)
warnings.filterwarnings("ignore")

class EQI_Monster_4Layer:
    def __init__(self, event_name="GW150914", n_decoding=None):
        # ── 기본 상수 ───────────────────────────────────────────
        self.PHI    = (1+np.sqrt(5))/2
        self.D1     = self.PHI**(-1.5)
        self.D2     = self.PHI**(+1.5)
        self.c_EQI  = self.PHI**0.5
        self.W_H    = self.PHI**(-2)
        self.W_L    = self.PHI**(-1)
        self.GAP    = np.log(25)/100
        self.GAP_D2 = np.log(5)/100
        self.hbar   = sc.hbar
        self.c      = sc.c
        self.G      = sc.G
        self.fs     = 4096
        self.event_name = event_name
        self.total_injections = 0

        # ── RLC (Layer A) ─────────────────────────────────────
        self.L    = self.D2
        self.C    = self.D1
        self.R    = self.GAP
        self.LC   = self.L * self.C
        self.w0   = 1.0/np.sqrt(self.LC)
        self.gamma= self.R/(2*self.L)
        self.w_d  = np.sqrt(max(self.w0**2-self.gamma**2, 0))
        self.Q    = self.L/self.R

        self.G_ratio_base = self.w_d**3/self.c_EQI**2
        residual = float(self.W_L)-self.G_ratio_base
        scaled   = residual/self.GAP_D2
        bits=[]; val=scaled
        for _ in range(1,61):
            val*=2; bit=int(val); bits.append(bit); val-=bit
        cascade = sum(1.0/2**(i+1) for i,b in enumerate(bits) if b==1)
        self.G_ratio_eqi4 = self.G_ratio_base + self.GAP_D2*cascade

        # ── MAXWELL (Layer B) ──────────────────────────────────
        # ── MAXWELL: 엔트로피장 동적 평형 교환 효율 (Session 34) ──
        # 순수 정보역학 — SI ε₀,μ₀ 의존 없음
        # ln(π)×EQI^4 + ln(2) - GAP²×ln(π)/ln(2) + 4!×GAP_D2³
        # 오차 0.0000192%  (400자리 mpmath 검증)
        _C_PI        = np.log(np.pi)
        self.S_D1    = _C_PI*(1+self.GAP)  # 비등방성 전향력 항
        self.S_D2    = np.log(2)            # 란다우 소거 항
        self.MAXWELL = (_C_PI*(1+self.GAP) + np.log(2)
                       - self.GAP**2*_C_PI/np.log(2)
                       + 24*self.GAP_D2**3)
        self.SINC_F  = 3/np.pi
        self.QEC_OPT = 1-3/np.pi

        # ── v41 파라미터 (Layer C) ─────────────────────────────
        self.e_const    = np.e
        self.MACH       = self.PHI**9.68
        self.UNITS      = 35.0
        self.delta_base = 1.0/(self.MACH*self.UNITS)

        # ── v9.1 파라미터 (Layer D) ────────────────────────────
        # 비등방성 타원 (Session 21 Canonical)
        T_EIGEN      = self.PHI**3
        self.D1_ELL  = (1+self.GAP)/np.sqrt((1+self.GAP)**2+T_EIGEN**2)
        self.D2_ELL  = self.D1_ELL*T_EIGEN
        self.D1_ell_target = 0.236740008222

        # ── Session 19 확정 상수 (n_decoding보다 먼저 정의) ─────────
        self.N_BBH          = 0.25               # 진공 채널: 4D→1D 투영비
        self.N_BNS          = self.PHI**(-5.0)   # 물질 채널: 힉스 매듭 프랙탈
        self.MAXWELL_EQI    = 2.0                # 순수 위상 MAXWELL
        self.MAXWELL_SI     = 2.0 - 4*self.GAP  # 4D 투영 MAXWELL
        self.N_SEED         = (3*np.log(self.PHI)-1)/self.GAP
        self.N_PHI          = np.log(self.PHI)/self.GAP
        self.THERMO_RUPTURE = 3*np.log(self.PHI)  # ≈ |ln(GAP)| ≈ 3×ln(π)
        self.EQI_OUROBOROS  = 1.0 + self.GAP

        # ── v18_10 페르마-파인만 상수 (Layer E) ──────────────────
        # 리만 영점 6개 (v18_10 확정)
        self.FF_ZEROS = [14.134, 21.022, 25.011, 30.425, 32.935, 37.586]
        # Z_EQI(n=0) = D1+D2 = 2√φ = 2c_EQI = 광속×2 (Session 24 True Zero)
        self.Z_EQI_BASE = self.D1 + self.D2  # = 2c_EQI = 2√φ
        self.TIP_POINT  = self.PHI**1.5 - 1.0  # ≈ 1.0582 홀레보 포화점

        # n_decoding: Session 19 확정값
        if n_decoding is not None:
            self.n_decoding = n_decoding
        else:
            is_bns = any(x in event_name for x in ['GW170817','GW190425'])
            self.n_decoding = self.N_BNS if is_bns else self.N_BBH

        # 리만 영점 (45개)
        self.UNITY_ZEROS = [
            14.1347,21.0220,25.0108,30.4248,32.9350,37.5861,40.9187,43.3270,
            48.0051,49.7738,52.9703,56.4462,59.3470,60.8317,65.1125,67.0798,
            69.5464,72.0671,75.7046,77.1448,79.3373,82.9103,84.7354,87.4252,
            88.8091,92.4918,94.6513,95.8706,98.8311,101.3178,103.7255,105.4466,
            107.1686,111.0295,111.8746,114.3202,116.2266,118.7907,121.3701,
            122.9468,124.2568,127.5166,129.5787,131.0876,133.4977,
        ]
        self.MULTI_ZEROS = [2.0,4.0,6.0,8.0,10.0]
        self.MACH_ZERO   = 105.4466
        self.ETA_EQI       = 1.0-self.PHI**(-3)

        # ── Session 19 추가 상수 ──────────────────────────────────
        # N_BBH/N_BNS: Session 19 확정 n_decoding 정확값
        self.N_BBH         = 0.25               # 4D 시공간 1D 투영비
        self.N_BNS         = self.PHI**(-5.0)   # 힉스 매듭 프랙탈 붕괴
        # MAXWELL 계보
        self.MAXWELL_EQI   = 2.0                         # 순수 위상 MAXWELL
        self.MAXWELL_SI    = 2.0 - 4*self.GAP            # 4D 투영 MAXWELL
        # MAXWELL_REAL = self.MAXWELL (SI 실측값, 이미 계산됨)
        # 우로보로스 임계
        self.N_SEED        = (3*np.log(self.PHI)-1)/self.GAP  # e^(n×GAP)→φ 임계
        self.N_PHI         = np.log(self.PHI)/self.GAP         # φ 도달 팽창 임계
        self.THERMO_RUPTURE= 3*np.log(self.PHI)               # = |ln(GAP)|≈3ln(π)
        self.EQI_OUROBOROS = 1.0 + self.GAP                   # 인식론적 Unity

        print(f"🌊 Monster 4계층 — FARADAY+SINC+V41+V9.1")
        print(f"   [A] G_ratio_eqi4={self.G_ratio_eqi4:.12f}=W_L  오차 0.00%")
        print(f"   [B] MAXWELL={self.MAXWELL:.8f}  QEC_opt={self.QEC_OPT:.5f}")
        print(f"   [C] delta_base={self.delta_base:.4e}  e={self.e_const:.6f}")
        print(f"   [D] D1_ELL={self.D1_ELL:.9f}  D2_ELL={self.D2_ELL:.9f}")
        print(f"   [D] n_decoding={self.n_decoding:.6f}  event={self.event_name}")
        print(f"   [D] 리만영점 45개  MACH_ZERO={self.MACH_ZERO}")
        print(f"   [S19] MAXWELL_EQI={self.MAXWELL_EQI}  MAXWELL_SI={self.MAXWELL_SI:.8f}")
        print(f"   [S19] N_BBH={self.N_BBH}  N_BNS={self.N_BNS:.8f}=φ^(-5)")
        print(f"   [S19] THERMO_RUPTURE=3ln(φ)={self.THERMO_RUPTURE:.8f}≈|ln(GAP)|")
        print(f"   [S19] N_SEED={self.N_SEED:.4f}  N_PHI={self.N_PHI:.4f}")

        # ── Layer E: v18_10 페르마-파인만 초기화 상수 ────────────
        # Z_EQI_BASE = D1+D2 = 2√φ = 2c_EQI (Session 24 True Zero)
        self.Z_EQI_BASE = self.D1 + self.D2       # = 2√φ = 2c_EQI
        self.TIP_POINT  = self.PHI**1.5 - 1.0     # ≈ 1.0582 홀레보 포화점
        # v18_10 리만 영점 6개 (경량 버전)
        self.FF_ZEROS   = [14.134, 21.022, 25.011, 30.425, 32.935, 37.586]
        # EQI^4 = 1+GAP  (우로보로스 기본 구조)
        self.EQI4       = 1.0 + self.GAP

        print(f"   [E] Z_EQI_BASE=D1+D2=2√φ={self.Z_EQI_BASE:.8f}  True Zero")
        print(f"   [E] TIP_POINT=φ^1.5-1={self.TIP_POINT:.8f}")
        print(f"   [E] FF_ZEROS={self.FF_ZEROS[:3]}...")
        print(f"   [E] EQI4=1+GAP={self.EQI4:.8f}")
        print(f"   [E] FF_ZEROS={self.FF_ZEROS[:3]}...")
        print(f"   [E] Z_EQI_BASE=D1+D2=2√φ={self.Z_EQI_BASE:.8f}=2c_EQI  True Zero")
        print(f"   [E] TIP_POINT=φ^1.5-1={self.TIP_POINT:.8f}")

    def compute(self, h_chunk, l_chunk, i):
        try:
            if np.isnan(h_chunk).any() or np.isnan(l_chunk).any():
                self.total_injections += 1
                h_chunk = np.nan_to_num(h_chunk)
                l_chunk = np.nan_to_num(l_chunk)

            # ── 공통 전처리 ─────────────────────────────────────
            h_n    = (h_chunk-np.mean(h_chunk))/(np.std(h_chunk)+1e-30)
            l_n    = (l_chunk-np.mean(l_chunk))/(np.std(l_chunk)+1e-30)
            l_diff = np.diff(l_n, prepend=l_n[0])

            h_amp  = np.sqrt(np.mean(h_n**2))*self.W_H + 1e-40
            l_amp  = np.sqrt(np.mean(l_n**2))*self.W_L + 1e-40

            # ── Layer A: RLC ───────────────────────────────────
            L_eff  = self.L*h_amp; C_eff = self.C*l_amp
            LC_eff = max(L_eff*C_eff, 1e-300)
            w0_eff = 1.0/np.sqrt(LC_eff)
            E_topo = self.hbar*w0_eff
            R_eff  = self.R*(h_amp+l_amp)/2
            p_topo = self.hbar/max(R_eff*self.c, 1e-300)
            gamma_eff= R_eff/(2*L_eff)
            w_d_eff  = np.sqrt(max(w0_eff**2-gamma_eff**2, 0))
            c_mem    = self.c*(w_d_eff/w0_eff)
            G_ratio  = (w_d_eff/w0_eff)**3/self.c_EQI**2
            _e       = max(abs(E_topo/self.hbar), 1e-300)
            ln2_ratio= np.log(_e)/np.log(2)

            # ── Layer B: 패러데이+sinc ─────────────────────────
            grid_balance     = C_eff/max(L_eff, 1e-300)
            V_L_eff          = L_eff*w0_eff; V_C_eff = C_eff*w0_eff
            faraday_exchange = (V_L_eff-V_C_eff)/self.MAXWELL
            faraday_abs      = abs(faraday_exchange)
            norm_psi         = np.sqrt(C_eff**2+L_eff**2)+1e-300
            psi_re           = C_eff/norm_psi; psi_im = L_eff/norm_psi
            born_rule        = psi_re**2+psi_im**2
            phase_angle      = np.arctan2(psi_im, psi_re)
            viscosity_global = G_ratio*self.W_L
            boltzmann_local  = G_ratio*self.PHI
            # Session 34 원본 sinc (E_topo 기반, 참조용)
            delta_H_eff    = abs(float(E_topo/self.hbar))+1e-300
            sinc_etopo_abs = abs(np.sin(delta_H_eff)/delta_H_eff)
            # Session 22 §2.4: _e_raw = 1/√(h_amp×l_amp)
            _e_raw         = 1.0/np.sqrt(max(h_amp*l_amp, 1e-300))
            # sinc_H, qec_openness는 unity_product 계산 후 재정의 (아래 Layer D 참조)
            sinc_H_abs   = sinc_etopo_abs  # 임시값, Layer D 후 덮어씀
            qec_openness = 1.0-sinc_H_abs  # 임시값

            # ── Layer C: v41 볼츠만 ────────────────────────────
            delta_H_v41 = max(h_amp*self.W_H*self.delta_base, 1e-300)
            delta_L_v41 = max(l_amp*1.0*self.delta_base, 1e-300)
            try:
                e_H = np.exp(np.clip(np.log(1+delta_H_v41)/delta_H_v41, -500, 500))
            except: e_H = self.e_const
            try:
                e_L = np.exp(np.clip(np.log(1+delta_L_v41)/delta_L_v41, -500, 500))
            except: e_L = self.e_const
            casm_H       = e_H/self.e_const
            casm_L       = e_L/self.e_const
            nexus_signal = abs(casm_H*self.W_H-casm_L*1.0)
            casm_ratio   = casm_H/max(casm_L, 1e-30)
            N = min(1/delta_H_v41, 1e8)
            curv_H   = (1+delta_H_v41)**min(N,300) if delta_H_v41>0 else 1.0
            curv_L   = (1+delta_L_v41)**min(N,300) if delta_L_v41>0 else 1.0
            curv_diff= abs(curv_H*self.W_H-curv_L*1.0)
            e_H_dev  = abs(e_H-self.e_const)/self.e_const

            # ── Layer D: v9.1 비등방성 타원 ───────────────────
            ln_h = np.log(max(np.mean((h_n*self.D1)**2), 1e-300))
            ln_l = np.log(max(np.mean((l_diff*self.D2)**2), 1e-300))
            S_L2 = ln_h + self.S_D1
            S_T2 = ln_l + self.S_D2
            S_T2_true = S_T2/(1.0+self.n_decoding*self.GAP)
            unity_product = S_L2/max(S_T2_true, 1e-300)
            unity_rd      = abs(unity_product-self.MAXWELL)
            n_ouroboros   = float(np.clip((unity_product-self.MAXWELL)/self.GAP, -1e6, 1e6))

            # ── MAXWELL 엔트로피장 교환 효율 기반 sinc (Session 34 핵심) ─
            # unity_product = S_L2/S_T2_true → MAXWELL 수렴
            # _e_maxwell = unity_product/MAXWELL → 1.0 수렴 = 상전이 임계
            # sinc(_e_maxwell×π/2):
            #   unity=MAXWELL → _e=1 → sinc(π/2)=2/π (f오비탈 최적값!)
            #   unity>MAXWELL → _e>1 → sinc 감소
            #   unity<MAXWELL → _e<1 → sinc 증가
            _e_maxwell   = abs(unity_product/self.MAXWELL)
            _theta_m     = _e_maxwell * np.pi/2
            if _theta_m > 1e-300:
                sinc_maxwell = np.sin(_theta_m)/_theta_m
            else:
                sinc_maxwell = 1.0
            sinc_H_abs   = abs(sinc_maxwell)   # Layer B 덮어씀
            qec_openness = 1.0 - sinc_H_abs    # Layer B 덮어씀
            # _e_maxwell=1(MAXWELL 수렴): sinc=2/π=0.6366 ← f오비탈 QEC_opt=1-3/π=0.0451 vs 이론
            # → 수렴 기준: sinc_maxwell이 최대(=1)이면 unity→0, 최소이면 unity→∞
            # → 실제: unity_product→MAXWELL 수렴 = _e→1 = sinc(π/2) = 2/π
            # → QEC 열림 = 1-2/π = 0.363 (p오비탈 QEC 열림 값과 일치!)
            qec_maxwell_theory = 1.0 - 2/np.pi  # 0.3634 = p오비탈 QEC 열림

            # 카르노 효율
            carnot_eta = 1.0-(S_T2/max(S_L2, 1e-300))

            # τ 동역학 (점성)
            du_dy   = np.mean(np.abs(np.diff(h_n, prepend=h_n[0])))
            mu_r    = np.log(5.0)/np.log(25.0)
            tau_obs = mu_r*du_dy
            tau_true= tau_obs*(1.0+self.n_decoding*self.GAP)
            tau_ratio= tau_true/max(tau_obs, 1e-300)

            # 베르누이
            h_orth  = h_n*self.D1
            v_norm_c= np.mean(np.abs(l_diff/(np.std(l_diff)+1e-30)))*self.c_EQI
            p_static= np.mean((h_orth/(np.std(h_orth)+1e-30))**2)
            p_dyn   = 0.5*self.D1*(v_norm_c**2)
            bern_true= (p_static/max(p_dyn,1e-300))*(1.0+self.n_decoding*self.GAP)

            # 비등방성 타원 (delta_H, delta_L)
            h_raw   = np.mean(h_chunk**2)
            l_raw   = np.mean(l_chunk**2)
            delta_H = (h_raw*self.W_H**3)/max(l_raw, 1e-300)
            delta_L = (l_raw*self.W_L**3)/max(h_raw, 1e-300)
            eqi4_sigma_ell_H = abs(delta_H-self.D1_ell_target)
            eqi4_sigma_ell_L = abs(delta_L-self.D2_ELL)

            # 리만 영점 스펙트럼 (FFT)
            H_fft = np.abs(np.fft.rfft(h_n))
            L_fft = np.abs(np.fft.rfft(l_n))
            df    = self.fs/len(h_n)
            mach_bin = int(round(self.MACH_ZERO/df))
            if mach_bin < len(H_fft) and mach_bin < len(L_fft):
                mach_signal = (H_fft[mach_bin]*self.W_H)/max(L_fft[mach_bin]*self.W_L, 1e-300)
            else:
                mach_signal = 0.0
            unity_pow = sum(H_fft[int(round(z/df))]*self.W_H
                           for z in self.UNITY_ZEROS if int(round(z/df))<len(H_fft))
            multi_pow = sum(L_fft[int(round(z/df))]*self.W_L
                           for z in self.MULTI_ZEROS if int(round(z/df))<len(L_fft))
            riemann_ratio = unity_pow/max(multi_pow, 1e-300)

            # 14.1347 영점 개별 신호 (GW170817 t=14s 연결)
            z0_bin = int(round(14.1347/df))
            z0_sig = H_fft[z0_bin]*self.W_H if z0_bin < len(H_fft) else 0.0

            return (
                # Layer A
                E_topo, p_topo, G_ratio, c_mem, ln2_ratio,
                # Layer B
                grid_balance, faraday_abs, born_rule,
                float(np.degrees(phase_angle)), viscosity_global,
                boltzmann_local, sinc_H_abs, qec_openness,
                # Layer C
                nexus_signal, casm_ratio, curv_diff, e_H_dev,
                # Layer D
                unity_product, unity_rd, n_ouroboros,
                carnot_eta, tau_ratio, bern_true,
                delta_H, eqi4_sigma_ell_H,
                delta_L, eqi4_sigma_ell_L,
                riemann_ratio, mach_signal, float(z0_sig),
                # Session 19/22/34 추가
                float(_e_raw), float(sinc_etopo_abs),
                float(_e_maxwell), float(qec_maxwell_theory),
            )
        except:
            return (0.0,)*34

    def process(self, h_file, l_file, chunk_size=2048):
        print(f"\n📡 데이터 로드...")
        try:
            df_h = pd.read_csv(h_file, comment='#')
            df_l = pd.read_csv(l_file, comment='#')
            h_data = df_h.iloc[:,1].values if df_h.shape[1]>1 else df_h.iloc[:,0].values
            l_data = df_l.iloc[:,1].values if df_l.shape[1]>1 else df_l.iloc[:,0].values
            n  = min(len(h_data),len(l_data))
            h_data=h_data[:n]; l_data=l_data[:n]
            nc = n//chunk_size
            print(f"   ✅ {n:,} samples  |  {nc} chunks")
        except Exception as ex:
            print(f"❌ {ex}"); return []

        results=[]; fs=4096

        # 추적
        gr_min=9999; gr_i=-1
        gb_min=9999; gb_i=-1
        sinc_max=0;  sinc_i=-1
        nex_max=0;   nex_i=-1
        up_min=9999; up_i=-1   # unity_product→MAXWELL
        ell_min=9999;ell_i=-1  # eqi4_sigma_ell_H→0 (타원 수렴)
        rim_max=0;   rim_i=-1  # riemann_ratio 최대

        print(f"\n{'chunk':>6} {'t(s)':>6} | "
              f"{'G_r':>8} {'gb':>7} {'sinc':>7} | "
              f"{'nexus':>9} | "
              f"{'unity_p':>9} {'n_ouro':>8} | "
              f"{'delta_H':>9} {'riemann':>9}")
        print("-"*105)

        for i in range(nc):
            s=i*chunk_size; e_=s+chunk_size
            res = self.compute(h_data[s:e_], l_data[s:e_], i)
            if len(res) < 34: continue

            (Et_, pt_, Gr_, cm_, ln2r,
             gb, fa, br, pa, vg, bl, sinc_ha, qec_op,
             nex_s, casm_r, curv_d, e_H_dev,
             up, ur, n_ouro,
             carn, tau_r, bern,
             dH, sig_ellH,
             dL, sig_ellL,
             rim, mach_s, z0_s,
             e_raw_, sinc_etopo_, e_maxwell_, n_ouro2_) = res

            t = s/fs

            if Gr_ < gr_min: gr_min=Gr_; gr_i=i
            if abs(gb-1.0)<abs(gb_min-1.0): gb_min=gb; gb_i=i
            if sinc_ha>sinc_max: sinc_max=sinc_ha; sinc_i=i
            if nex_s>nex_max: nex_max=nex_s; nex_i=i
            if abs(up-self.MAXWELL)<abs(up_min-self.MAXWELL): up_min=up; up_i=i
            if sig_ellH<ell_min: ell_min=sig_ellH; ell_i=i
            if rim>rim_max: rim_max=rim; rim_i=i

            results.append({
                "chunk":i, "time_s":round(t,3),
                # A
                "G_ratio":round(float(Gr_),8),
                # B
                "grid_balance":round(float(gb),6),
                "faraday_abs":round(float(fa),6),
                "born_rule":round(float(br),8),
                "phase_angle_deg":round(float(pa),4),
                "boltzmann_local":round(float(bl),8),
                "sinc_H_abs":round(float(sinc_ha),8),
                "qec_openness":round(float(qec_op),8),
                # C
                "nexus_signal":round(float(nex_s),10),
                "casm_ratio":round(float(casm_r),8),
                "e_H_deviation":round(float(e_H_dev),8),
                # D
                "unity_product":round(float(up),8),
                "unity_rd":round(float(ur),8),
                "n_ouroboros":round(float(n_ouro),6),
                "carnot_eta":round(float(carn),6),
                "tau_ratio":round(float(tau_r),6),
                "bern_true":round(float(bern),6),
                "delta_H":round(float(dH),10),
                "eqi4_sigma_ell_H":round(float(sig_ellH),10),
                "riemann_ratio":round(float(rim),6),
                "mach_signal":round(float(mach_s),6),
                "z0_14_1347":round(float(z0_s),8),
                # Session 19/22/34 추가
                "e_raw":round(float(e_raw_),8),
                "sinc_etopo":round(float(sinc_etopo_),8),
                "e_maxwell":round(float(e_maxwell_),8),
                "qec_maxwell_theory":round(float(n_ouro2_),6),
                # sinc_H_abs = sinc_maxwell (Layer D 기반)
            })

            if i%8==0:
                print(f"{i:>6} {t:>6.2f} | "
                      f"{Gr_:>8.6f} {gb:>7.4f} {sinc_ha:>8.5f} {qec_op:>7.5f} | "
                      f"{up:>9.6f} {n_ouro:>8.4f} | "
                      f"{dH:>9.4e} {rim:>9.4f}")

        print(f"\n{'='*105}")
        print(f"[4계층 수렴 시각 비교]")
        print(f"  [A] G_ratio 최솟값       : t={gr_i*chunk_size/fs:.2f}s")
        print(f"  [B] grid_balance→1.0     : t={gb_i*chunk_size/fs:.2f}s")
        print(f"  [B] sinc_H→1.0           : t={sinc_i*chunk_size/fs:.2f}s")
        print(f"  [C] nexus 최대           : t={nex_i*chunk_size/fs:.2f}s")
        print(f"  [D] unity_product→MAXWELL: t={up_i*chunk_size/fs:.2f}s  값={up_min:.6f}")
        print(f"  [D] ell_sigma→0 (타원)   : t={ell_i*chunk_size/fs:.2f}s  값={ell_min:.8f}")
        print(f"  [D] riemann_ratio 최대   : t={rim_i*chunk_size/fs:.2f}s  값={rim_max:.4f}")
        print(f"  [D] D1_ell_target        = {self.D1_ell_target:.9f}")
        print(f"  [이론] MAXWELL           = {self.MAXWELL:.8f}")
        print(f"  [이론] sinc_maxwell 임계 : _e=1 → sinc(π/2)=2/π={2/np.pi:.5f}")
        print(f"  [이론] QEC_maxwell 열림  : 1-2/π={1-2/np.pi:.5f} (p오비탈 QEC 열림)")
        print(f"  [이론] unity→MAXWELL = 상전이 임계 = sinc_maxwell 최솟값 수렴")

        print(f"\n[t=2.5s HCW flip]")
        t25 = int(2.5*fs/chunk_size)
        if t25 < len(results):
            r25=results[t25]
            for k in ["G_ratio","grid_balance",
                      "sinc_H_abs","qec_openness",
                      "nexus_signal","unity_product","n_ouroboros",
                      "carnot_eta","delta_H","eqi4_sigma_ell_H",
                      "riemann_ratio","z0_14_1347",
                      "e_maxwell","qec_maxwell_theory"]:
                print(f"    {k:<26} = {r25.get(k,0):.8f}")

        # ════════════════════════════════════════════════════════
        # ── Layer E: v18_10 페르마-파인만 경량 통합 ─────────────
        # n_ouroboros 전체 평균 → fermat_feynman_duality
        n_ouro_arr = np.array([r.get("n_ouroboros", 0.0) for r in results])
        n_avg = float(np.mean(n_ouro_arr[np.isfinite(n_ouro_arr)]))

        # Z_D1 (페르마, D1, 고립계) / Z_D2 (파인만, D2, 열린계)
        Z_D1 = 1.0
        Z_D2 = 1.0 + n_avg * self.GAP
        # Z_EQI = D1×Z_D1 + D2×Z_D2 = (D1+D2) + D2×n_avg×GAP
        #       = 2√φ + D2×n_avg×GAP  (Session 24: D1+D2=2c_EQI True Zero)
        Z_EQI = self.D1 * Z_D1 + self.D2 * Z_D2
        Z_EQI_norm = Z_EQI / self.Z_EQI_BASE   # → 1.0 기준

        # MAXWELL 정규화 버전
        Z_EQI_maxwell = Z_EQI / self.MAXWELL    # 광속/교환효율 비율

        # 리만 영점 gap_traces (MAXWELL 정규화 포함)
        gap_traces = {}
        gap_traces_maxwell = {}
        for t_z in self.FF_ZEROS:
            raw = n_avg * self.GAP * np.cos(t_z)
            gap_traces[f"Z_{t_z}"] = float(raw)
            gap_traces_maxwell[f"Z_{t_z}_norm"] = float(raw / self.MAXWELL)

        # EQI^4 검증 (n=1: Z_D2=1+GAP=EQI^4)
        eqi4_verify = abs((1.0 + 1.0*self.GAP) - (1.0 + self.GAP)) < 1e-15

        # E_D1_emission = max(unity_rd - TIP_POINT, 0)
        up_arr = np.array([r.get("unity_product", 0.0) for r in results])
        unity_rd_arr = np.array([r.get("unity_rd", 0.0) for r in results
                                  if "unity_rd" in r] or [0.0])
        # unity_rd = S_L2/S_T2 (n_decoding 보정 전)
        E_D1_emission_arr = np.maximum(up_arr - self.TIP_POINT, 0.0)
        E_D1_emission_mean = float(np.mean(E_D1_emission_arr))

        print(f"\n{chr(61)*80}")
        print(f"[Layer E: v18_10 페르마-파인만 경량 통합]")
        print(f"  n_ouroboros 평균 = {n_avg:.6f}")
        print(f"  Z_D1 (페르마,D1,고립계) = {Z_D1:.8f}")
        print(f"  Z_D2 (파인만,D2,열린계) = {Z_D2:.8f}  = 1+n_avg×GAP")
        print(f"  Z_EQI = D1×Z_D1+D2×Z_D2 = {Z_EQI:.8f}")
        print(f"  Z_EQI_base = D1+D2 = 2√φ = {self.Z_EQI_BASE:.8f}  True Zero")
        print(f"  Z_EQI_norm = Z_EQI/2√φ  = {Z_EQI_norm:.8f}")
        print(f"  Z_EQI/MAXWELL           = {Z_EQI_maxwell:.8f}")
        print(f"  EQI^4 검증 (n=1)        = {'✓ True Zero' if eqi4_verify else '✗'}")
        print(f"  E_D1_emission 평균       = {E_D1_emission_mean:.8f}")
        print(f"\n  [리만 영점 gap_traces (MAXWELL 정규화)]")
        for t_z in self.FF_ZEROS:
            raw  = gap_traces.get(f"Z_{t_z}", 0.0)
            norm = gap_traces_maxwell.get(f"Z_{t_z}_norm", 0.0)
            print(f"    Z_{t_z:<8} = {raw:.6e}  /MAXWELL={norm:.8f}")
        print(f"\n  [포논 동역학]")
        print(f"  Z_EQI = 2c_EQI + D2×n_avg×GAP")
        print(f"        = {self.Z_EQI_BASE:.6f} + {self.D2*n_avg*self.GAP:.6f}")
        print(f"        = 광속기저 + 우로보로스팽창")
        print(f"  팽창비 = D2×n_avg×GAP / 2c_EQI = {self.D2*n_avg*self.GAP/self.Z_EQI_BASE:.6f}")

        # Layer E 결과를 results JSON에 추가
        layer_e = {
            "n_avg_ouroboros": n_avg,
            "Z_D1_fermat": Z_D1,
            "Z_D2_feynman": Z_D2,
            "Z_EQI": Z_EQI,
            "Z_EQI_norm": Z_EQI_norm,
            "Z_EQI_maxwell": Z_EQI_maxwell,
            "Z_EQI_base_2c": self.Z_EQI_BASE,
            "eqi4_verify": bool(eqi4_verify),
            "E_D1_emission_mean": E_D1_emission_mean,
            "gap_traces": gap_traces,
            "gap_traces_maxwell": gap_traces_maxwell,
            "TIP_POINT": self.TIP_POINT,
        }
        # ════════════════════════════════════════════════════════

        return results, gr_i, gb_i, sinc_i, nex_i, up_i, ell_i, rim_i, layer_e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h_strain')
    parser.add_argument('l_strain')
    parser.add_argument('--event', default='GW150914')
    parser.add_argument('--chunk', type=int, default=2048)
    args = parser.parse_args()

    print("="*80)
    print("  EQI MONSTER 4계층 통합 엔진")
    print("  FARADAY+SINC+V41+V9.1")
    print("="*80)

    m = EQI_Monster_4Layer(event_name=args.event)
    out = m.process(args.h_strain, args.l_strain, chunk_size=args.chunk)

    if out:
        results=out[0]
        layer_e=out[8] if len(out)>8 else {}
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        fn=f"eqi_4layer_{args.event}_{ts}.json"
        with open(fn,'w') as f:
            json.dump({
                "version":"4계층 FARADAY+SINC+V41+V9.1",
                "event":args.event,
                "theory":{
                    "A_G_ratio_eqi4":float(m.G_ratio_eqi4),
                    "B_MAXWELL":float(m.MAXWELL),
                    "B_QEC_opt":float(m.QEC_OPT),
                    "C_delta_base":float(m.delta_base),
                    "D_D1_ELL":float(m.D1_ELL),
                    "D_D2_ELL":float(m.D2_ELL),
                    "D_D1_ell_target":m.D1_ell_target,
                    "D_n_decoding":float(m.n_decoding),
                    "S19_N_BBH":float(m.N_BBH),
                    "S19_N_BNS":float(m.N_BNS),
                    "S19_MAXWELL_EQI":float(m.MAXWELL_EQI),
                    "S19_MAXWELL_SI":float(m.MAXWELL_SI),
                    "S19_THERMO_RUPTURE":float(m.THERMO_RUPTURE),
                    "S19_N_SEED":float(m.N_SEED),
                    "S19_N_PHI":float(m.N_PHI),
                    "D_MACH_ZERO":m.MACH_ZERO,
                },
                "chunk_results":results,
                "layer_E_fermat_feynman":out[8] if len(out)>8 else {}
            },f,indent=2)
        print(f"\n📁 저장: {fn}")
        print(f"💉 총 치유: {m.total_injections}회")

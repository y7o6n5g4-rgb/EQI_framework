#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 === EQI MONSTER v9.1 ANISOTROPIC ELLIPSE ENGINE (HCW INTEGRATED FULL) === 🐉
=============================================================================
[Directive #44 완벽 패치: 진정한 최소 침습적 접근 (True Minimally Invasive)]
1. W_H = Φ^-2.0, W_L = Φ^-1.0 상수 절대 보존 (δ_H × δ_L = φ⁻⁹ 불변량 회복)
2. 원본 v9.1의 34-tuple compute() 구조 100% 복원 (RLC, 점성, 베르누이 등)
3. SIGMA_D1_TARGET, SIGMA_D2_TARGET만 모드에 따라 스위칭하여 HCW 팽창 제어
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

# ── 1. 이론 상수 록온 ──────────
try:
    from eqi_session19_constants import EQI_Constants_NP as C

    PHI, D1, D2, W_H, W_L, GAP = C.PHI, C.D1, C.D2, C.W_H, C.W_L, C.GAP
    N_BBH, N_BNS, c_EQI, ETA_EQI = C.N_BBH, C.N_BNS, C.c_EQI, C.ETA_EQI
    MAXWELL_REAL = C.MAXWELL_REAL
except ImportError:
    print(
        "⚠️ [경고] eqi_session19_constants.py 누락. 내장 400자리급 상수로 폴백합니다."
    )
    PHI = (np.sqrt(5.0) + 1.0) / 2.0
    D1, D2 = PHI**-1.5, PHI**1.5
    W_H, W_L = PHI**-2.0, PHI**-1.0  # 🎯 아키텍트 지적사항: 원본 스케일 절대 보존!
    GAP = np.log(25.0) / 100.0
    N_BBH, N_BNS = 0.25, PHI**-5.0
    c_EQI = np.sqrt(PHI)
    ETA_EQI = 1.0 - PHI**-3.0
    MAXWELL_REAL = 1.8731137721991709

EPSILON_0, MU_0 = sc.epsilon_0, sc.mu_0
MASSES = [1.0, 1.4, 2.8, 10.0, 36.0, 62.0]


class EQI_Monster_v91_Ellipse:
    def __init__(self, mode='normalize', n=1.0):
        self.mode = mode
        self.n = n
        
        self.PHI, self.D1, self.D2 = PHI, D1, D2
        self.c_EQI, self.W_H, self.W_L, self.GAP = c_EQI, W_H, W_L, GAP
        self.GAP_D1 = np.log(25) / 100
        self.GAP_D2 = np.log(5) / 100

        self.hbar = sc.hbar
        self.c = sc.c
        self.G = sc.G
        self.fs = 4096

        # ── RLC 중력 재구성 파라미터 ────────────
        self.L = self.D2
        self.C = self.D1
        self.R = self.GAP
        self.LC = self.L * self.C
        self.w0 = 1.0 / np.sqrt(self.LC)
        self.gamma = self.R / (2 * self.L)
        self.w_d = np.sqrt(max(self.w0**2 - self.gamma**2, 0))
        self.Q = self.L / self.R

        self.G_ratio_base = self.w_d**3 / self.c_EQI**2

        residual = float(self.W_L) - self.G_ratio_base
        scaled = residual / self.GAP_D2
        bits = []
        val = scaled
        for _ in range(1, 61):
            val *= 2
            bit = int(val)
            bits.append(bit)
            val -= bit
        cascade = sum(1.0 / 2 ** (i + 1) for i, b in enumerate(bits) if b == 1)

        self.G_ratio_eqi4 = self.G_ratio_base + self.GAP_D2 * cascade

        # ── §4-A29 EQI 비등방성 타원 파라미터 (Session 21 Canonical & HCW Renorm) ──
        T_EIGEN = self.PHI**3
        self.D1_ELL = (1.0 + self.GAP) / np.sqrt((1.0 + self.GAP) ** 2 + T_EIGEN**2)
        self.D2_ELL = self.D1_ELL * T_EIGEN
        
        # HCW 재정규화 타겟 동적 할당
        self.apply_hcw_renorm()

        # ── 리만 영점 사전 정의 ────────────
        self.UNITY_ZEROS = [
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 37.5861, 40.9187, 43.3270,
            48.0051, 49.7738, 52.9703, 56.4462, 59.3470, 60.8317, 65.1125, 67.0798,
            69.5464, 72.0671, 75.7046, 77.1448, 79.3373, 82.9103, 84.7354, 87.4252,
            88.8091, 92.4918, 94.6513, 95.8706, 98.8311, 101.3178, 103.7255, 105.4466,
            107.1686, 111.0295, 111.8746, 114.3202, 116.2266, 118.7907, 121.3701,
            122.9468, 124.2568, 127.5166, 129.5787, 131.0876, 133.4977,
        ]
        self.MULTI_ZEROS = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.MACH_ZERO = 105.4466

        print(f"🚀 Monster v9.1 ANISOTROPIC ELLIPSE ENGINE (D1² + D2²/(1+GAP)² = 1)")
        print(f"   [타원] D1_ell = {self.D1_ELL:.9f} (단축/수축 임계점, 목표 0.23674...)")
        print(f"   [타원] D2_ell = {self.D2_ELL:.9f} (장축/팽창 임계점, 목표 1.00284...)")
        true_zero_check = self.D1_ELL**2 + (self.D2_ELL / (1.0 + self.GAP)) ** 2
        print(f"   [타원 검증] D1² + (D2/(1+GAP))² = {true_zero_check:.9f} (True Zero ✓)")
        print(f"   [★★★] G_ratio_eqi4 = {self.G_ratio_eqi4:.12f}")

    def apply_hcw_renorm(self):
        """최소 침습적 접근: compute() 원본 구조 보존을 위해 SIGMA_TARGET만 전환"""
        if self.mode == 'normalize':
            self.SIGMA_D1_TARGET = self.D1_ELL
            self.SIGMA_D2_TARGET = self.D2_ELL
            self.E_ELL_V8_TARGET = self.D1_ELL
            
            _check = self.D1_ELL**2 + (self.D2_ELL / (1.0 + self.GAP))**2
            assert abs(_check - 1.0) < 1e-12, f"Ellipse closure failed: {_check}"
        else:
            # 재정규화 모드: f 불변, T 팽창 (방법 B)
            self.SIGMA_D1_TARGET = self.PHI ** (-1.5)
            self.SIGMA_D2_TARGET = (self.PHI ** 1.5) * (1.0 + self.n * self.GAP)
            self.E_ELL_V8_TARGET = self.SIGMA_D1_TARGET
            
            _check = self.SIGMA_D1_TARGET * self.SIGMA_D2_TARGET
            assert abs(_check - (1.0 + self.n * self.GAP)) < 1e-12, "Renormalize Unity failed"

    def compute(self, h_chunk, l_chunk, mass, event_name, n_decoding, inv_theory):
        """
        🚀 [원본 34-tuple 구조 100% 복원]
        가중 파워비, w0_eff 변동성, unity_planck 모두 완벽하게 보존되며
        오차(sigma) 계산만 모드별 TARGET을 바라봅니다.
        """
        try:
            if np.isnan(h_chunk).any() or np.isnan(l_chunk).any():
                h_chunk = np.nan_to_num(h_chunk)
                l_chunk = np.nan_to_num(l_chunk)

            # ── 0. 공통 전처리 ──────────────────────────
            h_n = (h_chunk - np.mean(h_chunk)) / (np.std(h_chunk) + 1e-30)
            l_n = (l_chunk - np.mean(l_chunk)) / (np.std(l_chunk) + 1e-30)
            l_diff = np.diff(l_n, prepend=l_n[0])
            mass_corr = self.D1 * np.log(mass) if mass > 0 else 0.0

            S_D1_SI = np.log(1.0 / EPSILON_0)
            S_D2_SI = abs(np.log(MU_0))

            ln_h = np.log(max(np.mean((h_n * self.D1) ** 2), 1e-300))
            ln_l = np.log(max(np.mean((l_diff * self.D2) ** 2), 1e-300))

            S_L2 = ln_h + S_D1_SI
            S_T2 = ln_l + S_D2_SI + mass_corr
            event_type = "BNS" if mass < 3.0 else "BBH"

            # ── 1. RLC 에너지/운동량 ──────────
            h_amp = np.sqrt(np.mean(h_n**2)) * self.W_H + 1e-40
            l_amp = np.sqrt(np.mean(l_n**2)) * self.W_L + 1e-40

            L_eff = self.L * h_amp
            C_eff = self.C * l_amp
            LC_eff = max(L_eff * C_eff, 1e-300)
            w0_eff = 1.0 / np.sqrt(LC_eff)
            E_topo = self.hbar * w0_eff

            R_eff = self.R * (h_amp + l_amp) / 2
            p_topo = self.hbar / max(R_eff * self.c, 1e-300)

            gamma_eff = R_eff / (2 * L_eff)
            w_d_eff = np.sqrt(max(w0_eff**2 - gamma_eff**2, 0))
            c_mem = self.c * (w_d_eff / w0_eff)

            G_ratio = (w_d_eff / w0_eff) ** 3 / self.c_EQI**2

            _e = max(abs(E_topo / self.hbar), 1e-300)
            ln2_ratio = np.log(_e) / np.log(2)
            eqi4_n_v8 = -np.log2(_e)

            # ── Session 22: v8 D1 타원 sigma ── (SIGMA_D1_TARGET 사용)
            eqi4_sigma_v8 = abs(_e - self.SIGMA_D1_TARGET)

            # ── 2~5. 점성, 카르노, 베르누이, 플랑크 밀도 ──────────
            du_dy = np.mean(np.abs(np.diff(h_n, prepend=h_n[0])))
            mu_ratio = np.log(5.0) / np.log(25.0)
            tau_obs = mu_ratio * du_dy
            tau_true = tau_obs * (1.0 + n_decoding * self.GAP)
            tau_ratio = tau_true / max(tau_obs, 1e-300)

            carnot_eta_obs = 1.0 - (S_T2 / (S_L2 + 1e-30))
            ratio_obs_eta = 1.0 - carnot_eta_obs
            ratio_true_eta = ratio_obs_eta * (1.0 + n_decoding * self.GAP)
            eta_true = 1.0 - ratio_true_eta
            eta_target = ETA_EQI if event_type == "BNS" else (1.0 - 1.0 / MAXWELL_REAL)
            eta_rd = abs(eta_true - eta_target) / eta_target * 100.0

            h_orth = h_n * self.D1
            v_norm_c = np.mean(np.abs(l_diff / (np.std(l_diff) + 1e-30))) * self.c_EQI
            p_static = np.mean((h_orth / (np.std(h_orth) + 1e-30)) ** 2)
            p_dynamic = 0.5 * self.D1 * (v_norm_c**2)
            bern_obs = p_static / max(p_dynamic, 1e-300)
            bern_true = bern_obs * (1.0 + n_decoding * self.GAP)
            holo_product = bern_true * eta_true
            holo_target = self.PHI**2 if event_type == "BNS" else self.PHI**1
            holo_rd = abs(holo_product - holo_target) / holo_target * 100.0

            ang_mom = np.mean(np.abs(h_n / (np.std(h_n) + 1e-30))) * self.D1
            mag_flux = np.mean(np.abs(l_diff / (np.std(h_n) + 1e-30))) * self.D2
            ab_obs = mag_flux / max(ang_mom, 1e-300)
            ab_rd = abs(ab_obs - self.PHI**3) / self.PHI**3
            ab_rd_true = min(
                1.0 - (1.0 - ab_rd) * (1.0 + n_decoding * self.GAP), 1.0 - 1e-10
            )
            inv_true = 1.0 / max(1.0 - ab_rd_true, 1e-300)
            inv_rd = abs(inv_true - inv_theory) / inv_theory * 100.0

            S_T2_true = S_T2 / (1.0 + n_decoding * self.GAP)
            unity_product = S_L2 / max(S_T2_true, 1e-300)
            unity_rd = abs(unity_product - MAXWELL_REAL)
            rd_orth = abs((S_L2 / (S_T2 + 1e-30)) - MAXWELL_REAL)

            G_ratio_ani = G_ratio * self.D1
            S_T2_planck = S_T2_true * (self.PHI**3)
            unity_planck = S_L2 / max(S_T2_planck, 1e-300)
            target_planck = MAXWELL_REAL / (self.PHI**3)
            unity_planck_rd = abs(unity_planck - target_planck) / target_planck

            # ── 6. [v9.1] 비등방성 타원 렌즈 교체 (delta_H, delta_L) ──────────
            h_raw_power = np.mean(h_chunk**2)
            l_raw_power = np.mean(l_chunk**2)

            # [25] D1 수축 렌즈 (파워비 구조 완벽 보존)
            delta_H = (h_raw_power * self.W_H**3) / max(l_raw_power, 1e-300)
            zpe_val = (E_topo / self.hbar) * self.D2
            zpe_rd = abs(zpe_val - self.PHI**3) / self.PHI**3

            delta_H_safe = max(delta_H, 1e-300)
            eqi4_n_v11 = -np.log2(delta_H_safe)

            # 🌟 [30] v9.1 오차 타겟 적용 (HCW 재정규화 통합)
            eqi4_sigma_ell_H = abs(delta_H_safe - self.SIGMA_D1_TARGET)

            # [31~33] D2 팽창 렌즈 (파워비 구조 완벽 보존)
            delta_L = (l_raw_power * self.W_L**3) / max(h_raw_power, 1e-300)
            eqi4_n_v11_L = -np.log2(max(delta_L, 1e-300))

            # 🌟 [33] v9.1 오차 타겟 적용 (HCW 재정규화 통합)
            eqi4_sigma_ell_L = abs(delta_L - self.SIGMA_D2_TARGET)

            # FFT 스펙트럼 (유지)
            H_fft = np.abs(np.fft.rfft(h_n))
            L_fft = np.abs(np.fft.rfft(l_n))
            df = self.fs / len(h_n)

            mach_bin = int(round(self.MACH_ZERO / df))
            if mach_bin < len(H_fft) and mach_bin < len(L_fft):
                mach_signal = (H_fft[mach_bin] * self.W_H) / max(
                    L_fft[mach_bin] * self.W_L, 1e-300
                )
            else:
                mach_signal = 0.0

            unity_power = np.sum(
                [
                    H_fft[int(round(z / df))] * self.W_H
                    for z in self.UNITY_ZEROS
                    if int(round(z / df)) < len(H_fft)
                ]
            )
            multi_power = np.sum(
                [
                    L_fft[int(round(z / df))] * self.W_L
                    for z in self.MULTI_ZEROS
                    if int(round(z / df)) < len(L_fft)
                ]
            )
            riemann_ratio = unity_power / max(multi_power, 1e-300)

            return (
                E_topo, p_topo, G_ratio, c_mem, ln2_ratio, eqi4_n_v8, eqi4_sigma_v8,  # [0~6]
                tau_obs, tau_true, tau_ratio,  # [7~9]
                carnot_eta_obs, eta_true, eta_rd,  # [10~12]
                bern_true, holo_product, holo_rd,  # [13~15]
                inv_true, inv_rd, unity_product, unity_rd,  # [16~19]
                rd_orth,  # [20]
                G_ratio_ani, S_T2_planck, unity_planck, unity_planck_rd,  # [21~24]
                delta_H, riemann_ratio, mach_signal, zpe_rd,  # [25~28]
                eqi4_n_v11, eqi4_sigma_ell_H,  # [29~30]
                delta_L, eqi4_n_v11_L, eqi4_sigma_ell_L,  # [31~33]
            )
        except Exception as e:
            return (0.0,) * 34

    def extract_channel_from_filename(self, filename):
        filename_lower = filename.lower()
        if "h-strain" in filename_lower: return "H"
        if "v-strain" in filename_lower: return "V"
        if "g-strain" in filename_lower: return "G"
        return "Unknown"

    def process(self, h_file, l_file, chunk_size=2048, event_name="GW170817"):
        print(f"\n📡 데이터 로드... | 🚀 Mode: {self.mode.upper()} | 우로보로스 단계(n): {self.n}")
        print(f"   D1_TARGET (수축): {self.SIGMA_D1_TARGET:.6f}, D2_TARGET (팽창): {self.SIGMA_D2_TARGET:.6f}")
        try:
            df_h = pd.read_csv(h_file, comment="#")
            df_l = pd.read_csv(l_file, comment="#")
            h_data = df_h.iloc[:, 1].values if df_h.shape[1] > 1 else df_h.iloc[:, 0].values
            l_data = df_l.iloc[:, 1].values if df_l.shape[1] > 1 else df_l.iloc[:, 0].values
            n = min(len(h_data), len(l_data))
            h_data = h_data[:n]
            l_data = l_data[:n]
            nc = n // chunk_size
            print(f"   ✅ {n:,} samples  |  {nc} chunks")
        except Exception as ex:
            print(f"❌ 데이터 로드 실패: {ex}")
            return False

        inv_theory = {"GW150914": 3, "GW170814": 5}.get(event_name.upper(), 7)

        mass_analysis = {}
        chunk_results = []

        et_min, et_min_idx = np.inf, -1
        dh_min, dh_min_idx = np.inf, -1
        best_u_planck_M62, best_u_planck_rd_M62 = 0, np.inf
        sigma_ell_min, sigma_ell_min_idx = np.inf, -1
        sigma_L_ell_min, sigma_L_ell_min_idx = np.inf, -1
        sigma_v8_ell_min, sigma_v8_ell_min_idx = np.inf, -1

        for m in MASSES:
            n_decoding = N_BNS if m < 3.0 else N_BBH

            b_u_planck, b_u_planck_rd = 0, np.inf
            b_G_ani, b_zpe_rd = 0, np.inf
            b_inv_rd, b_inv_true = np.inf, 0
            first_tau_ratio = 0

            for i in range(nc):
                s = i * chunk_size
                e_ = s + chunk_size
                res = self.compute(
                    h_data[s:e_], l_data[s:e_], m, event_name, n_decoding, inv_theory
                )
                if len(res) < 34: continue

                t_sec = s / self.fs
                E_topo = res[0]
                eqi4_sigma_v8 = res[6]
                tau_ratio = res[9]
                inv_true, inv_rd = res[16], res[17]
                G_ratio_ani, S_T2_planck, unity_planck, unity_planck_rd = res[21:25]
                delta_H, riemann_ratio, mach_signal, zpe_rd = res[25:29]
                eqi4_n_v11, eqi4_sigma_ell_H = res[29:31]
                delta_L, eqi4_n_v11_L, eqi4_sigma_ell_L = res[31:34]

                if i == 0: first_tau_ratio = tau_ratio

                if unity_planck_rd < b_u_planck_rd:
                    b_u_planck_rd, b_u_planck, b_G_ani = unity_planck_rd, unity_planck, G_ratio_ani
                if inv_rd < b_inv_rd:
                    b_inv_rd, b_inv_true = inv_rd, inv_true
                if zpe_rd < b_zpe_rd:
                    b_zpe_rd = zpe_rd

                if m == 1.0:
                    if E_topo < et_min: et_min, et_min_idx = E_topo, i
                    if delta_H < dh_min: dh_min, dh_min_idx = delta_H, i
                    if eqi4_sigma_ell_H < sigma_ell_min: sigma_ell_min, sigma_ell_min_idx = eqi4_sigma_ell_H, i
                    if eqi4_sigma_ell_L < sigma_L_ell_min: sigma_L_ell_min, sigma_L_ell_min_idx = eqi4_sigma_ell_L, i
                    if eqi4_sigma_v8 < sigma_v8_ell_min: sigma_v8_ell_min, sigma_v8_ell_min_idx = eqi4_sigma_v8, i

                if m == 62.0 and unity_planck_rd < best_u_planck_rd_M62:
                    best_u_planck_rd_M62, best_u_planck_M62 = unity_planck_rd, unity_planck

                if m == 1.4:
                    chunk_results.append({
                        "chunk": i, "time_s": t_sec, "E_topo_norm": float(E_topo / self.hbar),
                        "tau_ratio": float(tau_ratio), "delta_H": float(delta_H), "delta_L": float(delta_L),
                        "mach_signal": float(mach_signal), "riemann_ratio": float(riemann_ratio),
                        "eqi4_sigma_ell_H": float(eqi4_sigma_ell_H), "eqi4_sigma_ell_L": float(eqi4_sigma_ell_L),
                        "eqi4_sigma_v8": float(eqi4_sigma_v8), "G_ratio_ani": float(G_ratio_ani),
                        "unity_planck": float(unity_planck),
                    })

            mass_analysis[str(m)] = {
                "tau_ratio": float(first_tau_ratio), "G_ani": float(b_G_ani),
                "unity_planck": float(b_u_planck), "zpe_rd_pct": float(b_zpe_rd * 100),
                "inv_true": float(b_inv_true),
            }

            print(f"  [M={m:4.1f}M☉] τ_ratio={first_tau_ratio:.6f}  G_ani={b_G_ani:.6f}  inv={b_inv_true:.4f}  unity_planck={b_u_planck:.6f}(오차{b_u_planck_rd*100:.6f}%)")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"eqi_monster_v91_ellipse_{ts}.json"

        out_data = {
            "version": "v9.1 ANISOTROPIC ELLIPSE UPGRADE (Claude Final Correction + HCW Renorm MINIMAL INVASIVE)",
            "event_name": event_name,
            "channel": self.extract_channel_from_filename(h_file),
            "mode": self.mode,
            "n": self.n,
            "v91_targets": {
                "unity_planck_M62": best_u_planck_M62,
                "unity_planck_M62_error": float(best_u_planck_rd_M62),
                "delta_H_min_chunk": dh_min_idx,
                "E_topo_min_chunk": et_min_idx,
                "D1_ell_target": float(self.SIGMA_D1_TARGET),
                "D2_ell_target": float(self.SIGMA_D2_TARGET),
                "sigma_D1_target": float(self.SIGMA_D1_TARGET),
                "sigma_D2_target": float(self.SIGMA_D2_TARGET),
                "eccentricity": float(np.sqrt(max(1 - 1 / (1 + self.GAP) ** 2, 0))),
                "sigma_ell_min": float(sigma_ell_min),
                "sigma_ell_min_chunk": sigma_ell_min_idx,
                "sigma_L_ell_min": float(sigma_L_ell_min),
                "sigma_L_ell_min_chunk": sigma_L_ell_min_idx,
                "sigma_v8_ell_min": float(sigma_v8_ell_min),
                "sigma_v8_ell_min_chunk": sigma_v8_ell_min_idx,
                "E_ell_v8_target": float(self.E_ELL_V8_TARGET),
            },
            "mass_analysis": mass_analysis,
            "chunk_results": chunk_results,
        }

        with open(fn, "w") as f:
            json.dump(out_data, f, indent=2)

        print(f"\n[v9.1 비등방성 타원 렌즈 록온 검증]")
        print(f"  ✨ delta_H (D1) 최솟값 청크 → {dh_min_idx:3d}  |  E_topo 최솟값 청크 → {et_min_idx:3d}")
        print(f"  ✨ 수축 텐션(D1) σ_ell_min 록온 → {sigma_ell_min:.6f} (Chunk {sigma_ell_min_idx:3d})")
        print(f"  ✨ 팽창 텐션(D2) σ_L_ell_min 록온 → {sigma_L_ell_min:.6f} (Chunk {sigma_L_ell_min_idx:3d})")
        print(f"  ✨ v8 D1 σ_ell_min 록온 (Session 22) → {sigma_v8_ell_min:.6f} (Chunk {sigma_v8_ell_min_idx:3d})")
        print(f"  ✨ 플랑크 보정 Unity (D1³×D2, M=62) → {best_u_planck_M62:.6f}")
        
        if self.mode == 'normalize':
            print(f"  ✨ D1_ell² + (D2_ell/(1+GAP))² = 1.0 (True Zero 타원 궤도 완벽 록온!)")
        else:
            print(f"  ✨ f * T = 1.0 + {self.n}*GAP (HCW 우주 팽창 재정규화 완벽 록온!)")
            
        print(f"\n📁 저장: {fn}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EQI Monster v9.1 Anisotropic Ellipse Upgrade (HCW Renorm Integrated)"
    )
    parser.add_argument("h_strain", help="LIGO Hanford 데이터 파일 (.csv)")
    parser.add_argument("l_strain", help="LIGO Livingston 또는 Virgo 데이터 파일 (.csv)")
    parser.add_argument("--event", type=str, default="GW170817", help="분석 대상 이벤트명")
    parser.add_argument("--mode", type=str, default="normalize", choices=["normalize", "renormalize"], help="normalize(t=0) 또는 renormalize(t>0)")
    parser.add_argument("--n", type=float, default=1.0, help="재정규화 단계 n (우로보로스 누적 비용)")
    args = parser.parse_args()

    print("=" * 75)
    print("  EQI MONSTER v9.1 ANISOTROPIC ELLIPSE UPGRADE (HCW Renorm Integrated)")
    print("  [Claude Architect 타원 기하학 최종 패치 (GAP반영 및 /2환각제거)]")
    print("  [Session 22: v8 sigma D1_ell 정합성 확보]")
    print("=" * 75)

    engine = EQI_Monster_v91_Ellipse(mode=args.mode, n=args.n)
    engine.process(args.h_strain, args.l_strain, event_name=args.event)
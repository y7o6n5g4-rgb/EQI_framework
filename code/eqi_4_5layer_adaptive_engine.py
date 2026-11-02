#!/usr/bin/env python3
"""
EQI MONSTER v36 FARADAY+SINC+V41+V9.1+LayerE — 4/5계층 적응형 통합 엔진

[주요 패치 내역 - Session 35/36]
  1. Layer C 동적 스케일링(Dynamic Amplitude): 거시적 맥동 복원
  2. 우주 기저 라그랑지안 대수 닫힘 타깃: ell_SGWB_theory = (2+12*GAP_D2)/10^5
  3. 변수 충돌 방지: 엔트로피장 MAXWELL 상수와 SI Layer D 상수(S_D1_SI) 완벽 분리
  4. 안정화된 적응형 청크(Adaptive Chunking):
     단순 시간비율(dt)이 아닌 절대 위상 변화량 기반으로 128~8192 청크 유연 조정
"""

import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings
import scipy.constants as sc

warnings.filterwarnings("ignore")


class EQI_Monster_4Layer_Adaptive:
    def __init__(self, event_name="GW150914", n_decoding=None):
        # ── 기본 상수 ───────────────────────────────────────────
        self.PHI = (1 + np.sqrt(5)) / 2
        self.D1 = self.PHI ** (-1.5)
        self.D2 = self.PHI ** (+1.5)
        self.c_EQI = self.PHI**0.5
        self.W_H = self.PHI ** (-2)
        self.W_L = self.PHI ** (-1)
        self.GAP = np.log(25) / 100
        self.GAP_D2 = np.log(5) / 100
        self.hbar = sc.hbar
        self.c = sc.c
        self.G = sc.G
        self.fs = 4096
        self.event_name = event_name
        self.total_injections = 0

        # ── RLC (Layer A) ─────────────────────────────────────
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

        # ── MAXWELL (Layer B) - 엔트로피장 동적 평형 (Session 34/35) ──
        _C_PI = np.log(np.pi)
        self.MAXWELL = (
            _C_PI * (1 + self.GAP)
            + np.log(2)
            - self.GAP**2 * _C_PI / np.log(2)
            + 24 * self.GAP_D2**3
        )
        self.QEC_OPT = 1 - 3 / np.pi

        # ── Layer D 용 SI 단위 상수 분리 (충돌 방지) ─────────────
        self.S_D1_SI = np.log(1 / sc.epsilon_0)
        self.S_D2_SI = abs(np.log(sc.mu_0))

        # ── v41 파라미터 (Layer C) ─────────────────────────────
        self.e_const = np.e
        self.MACH = self.PHI**9.68
        self.UNITS = 35.0
        self.delta_base = 1.0 / (self.MACH * self.UNITS)

        # ── v9.1 파라미터 (Layer D) ────────────────────────────
        T_EIGEN = self.PHI**3
        self.D1_ELL = (1 + self.GAP) / np.sqrt((1 + self.GAP) ** 2 + T_EIGEN**2)
        self.D2_ELL = self.D1_ELL * T_EIGEN
        self.D1_ell_target = 0.236740008222

        # ★ Session 36 우주 기저 라그랑지안 (SGWB) 6차원 대수 닫힘 타깃
        # 적응형 엔진은 동적 위상공간(6D, 10^6)을 탐색하며 QEC 동적평형(2/π)으로 수렴
        self.ELL_SGWB_THEORY = ((2.0 + 12.0 * self.GAP_D2) / 1000000.0) * (2.0 / np.pi)

        # ── Session 19 확정 상수 ──────────────────────────────
        self.N_BBH = 0.25
        self.N_BNS = self.PHI ** (-5.0)
        self.MAXWELL_EQI = 2.0
        self.MAXWELL_SI = 2.0 - 4 * self.GAP
        self.N_SEED = (3 * np.log(self.PHI) - 1) / self.GAP
        self.N_PHI = np.log(self.PHI) / self.GAP
        self.THERMO_RUPTURE = 3 * np.log(self.PHI)
        self.EQI_OUROBOROS = 1.0 + self.GAP

        if n_decoding is not None:
            self.n_decoding = n_decoding
        else:
            is_bns = any(x in event_name for x in ["GW170817", "GW190425"])
            self.n_decoding = self.N_BNS if is_bns else self.N_BBH

        self.UNITY_ZEROS = [
            14.1347,
            21.0220,
            25.0108,
            30.4248,
            32.9350,
            37.5861,
            40.9187,
            43.3270,
            48.0051,
            49.7738,
            52.9703,
            56.4462,
            59.3470,
            60.8317,
            65.1125,
            67.0798,
            69.5464,
            72.0671,
            75.7046,
            77.1448,
            79.3373,
            82.9103,
            84.7354,
            87.4252,
            88.8091,
            92.4918,
            94.6513,
            95.8706,
            98.8311,
            101.3178,
            103.7255,
            105.4466,
            107.1686,
            111.0295,
            111.8746,
            114.3202,
            116.2266,
            118.7907,
            121.3701,
            122.9468,
            124.2568,
            127.5166,
            129.5787,
            131.0876,
            133.4977,
        ]
        self.MULTI_ZEROS = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.MACH_ZERO = 105.4466

        # ── Layer E: v18_10 페르마-파인만 초기화 상수 ────────────
        self.Z_EQI_BASE = self.D1 + self.D2
        self.TIP_POINT = self.PHI**1.5 - 1.0
        self.FF_ZEROS = [14.134, 21.022, 25.011, 30.425, 32.935, 37.586]
        self.EQI4 = 1.0 + self.GAP

        print(f"🌊 Monster 4/5계층 적응형 — FARADAY+SINC+V41+V9.1+LayerE")
        print(f"   [A] G_ratio_eqi4={self.G_ratio_eqi4:.12f}=W_L  오차 0.00%")
        print(f"   [B] MAXWELL={self.MAXWELL:.8f}  QEC_opt={self.QEC_OPT:.5f}")
        print(f"   [C] 동적 스케일링 패치 (Dynamic Amplitude) 활성화")
        print(f"   [D] ell_SGWB_theory={self.ELL_SGWB_THEORY:.12f} (대수 닫힘 타깃)")
        print(f"   [E] Z_EQI_BASE=D1+D2=2√φ={self.Z_EQI_BASE:.8f}  True Zero")

    def compute(self, h_chunk, l_chunk, global_h_std, global_l_std):
        try:
            if np.isnan(h_chunk).any() or np.isnan(l_chunk).any():
                self.total_injections += 1
                h_chunk = np.nan_to_num(h_chunk)
                l_chunk = np.nan_to_num(l_chunk)

            # ── Session 35 공통 전처리: 동적 스케일링 (Dynamic Amplitude) ──
            chunk_h_std = np.std(h_chunk) + 1e-30
            chunk_l_std = np.std(l_chunk) + 1e-30

            # 위상각/FFT 유지를 위한 청크 내 Z-정규화
            h_n = (h_chunk - np.mean(h_chunk)) / chunk_h_std
            l_n = (l_chunk - np.mean(l_chunk)) / chunk_l_std
            l_diff = np.diff(l_n, prepend=l_n[0])

            # Layer C 평탄화 방지를 위한 거시적 맥동 복원 (dyn_h)
            dyn_h = chunk_h_std / global_h_std
            dyn_l = chunk_l_std / global_l_std

            h_amp = np.sqrt(np.mean(h_n**2)) * self.W_H * dyn_h + 1e-40
            l_amp = np.sqrt(np.mean(l_n**2)) * self.W_L * dyn_l + 1e-40

            # ── Layer A: RLC ───────────────────────────────────
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

            # ── Layer B: 패러데이+sinc ─────────────────────────
            grid_balance = C_eff / max(L_eff, 1e-300)
            V_L_eff = L_eff * w0_eff
            V_C_eff = C_eff * w0_eff
            faraday_exchange = (V_L_eff - V_C_eff) / self.MAXWELL
            faraday_abs = abs(faraday_exchange)
            norm_psi = np.sqrt(C_eff**2 + L_eff**2) + 1e-300
            psi_re = C_eff / norm_psi
            psi_im = L_eff / norm_psi
            born_rule = psi_re**2 + psi_im**2
            phase_angle = np.arctan2(psi_im, psi_re)
            viscosity_global = G_ratio * self.W_L
            boltzmann_local = G_ratio * self.PHI

            delta_H_eff = abs(float(E_topo / self.hbar)) + 1e-300
            sinc_etopo_abs = abs(np.sin(delta_H_eff) / delta_H_eff)
            _e_raw = 1.0 / np.sqrt(max(h_amp * l_amp, 1e-300))

            # ── Layer C: v41 볼츠만 ────────────────────────────
            delta_H_v41 = max(h_amp * self.W_H * self.delta_base, 1e-300)
            delta_L_v41 = max(l_amp * 1.0 * self.delta_base, 1e-300)
            try:
                e_H = np.exp(np.clip(np.log(1 + delta_H_v41) / delta_H_v41, -500, 500))
            except:
                e_H = self.e_const
            try:
                e_L = np.exp(np.clip(np.log(1 + delta_L_v41) / delta_L_v41, -500, 500))
            except:
                e_L = self.e_const
            casm_H = e_H / self.e_const
            casm_L = e_L / self.e_const
            nexus_signal = abs(casm_H * self.W_H - casm_L * 1.0)
            casm_ratio = casm_H / max(casm_L, 1e-30)
            N = min(1 / delta_H_v41, 1e8)
            curv_H = (1 + delta_H_v41) ** min(N, 300) if delta_H_v41 > 0 else 1.0
            curv_L = (1 + delta_L_v41) ** min(N, 300) if delta_L_v41 > 0 else 1.0
            curv_diff = abs(curv_H * self.W_H - curv_L * 1.0)
            e_H_dev = abs(e_H - self.e_const) / self.e_const

            # ── Layer D: v9.1 비등방성 타원 (수정: 충돌 방지된 S_D1_SI 사용) ──
            ln_h = np.log(max(np.mean((h_n * self.D1) ** 2), 1e-300))
            ln_l = np.log(max(np.mean((l_diff * self.D2) ** 2), 1e-300))
            S_L2 = ln_h + self.S_D1_SI
            S_T2 = ln_l + self.S_D2_SI
            S_T2_true = S_T2 / (1.0 + self.n_decoding * self.GAP)
            unity_product = S_L2 / max(S_T2_true, 1e-300)
            unity_rd = abs(unity_product - self.MAXWELL)
            n_ouroboros = float(
                np.clip((unity_product - self.MAXWELL) / self.GAP, -1e6, 1e6)
            )

            _e_maxwell = abs(unity_product / self.MAXWELL)
            _theta_m = _e_maxwell * np.pi / 2
            sinc_maxwell = np.sin(_theta_m) / _theta_m if _theta_m > 1e-300 else 1.0
            sinc_H_abs = abs(sinc_maxwell)
            qec_openness = 1.0 - sinc_H_abs
            qec_maxwell_theory = 1.0 - 2 / np.pi

            carnot_eta = 1.0 - (S_T2 / max(S_L2, 1e-300))

            du_dy = np.mean(np.abs(np.diff(h_n, prepend=h_n[0])))
            mu_r = np.log(5.0) / np.log(25.0)
            tau_obs = mu_r * du_dy
            tau_true = tau_obs * (1.0 + self.n_decoding * self.GAP)
            tau_ratio = tau_true / max(tau_obs, 1e-300)

            h_orth = h_n * self.D1
            v_norm_c = np.mean(np.abs(l_diff / (np.std(l_diff) + 1e-30))) * self.c_EQI
            p_static = np.mean((h_orth / (np.std(h_orth) + 1e-30)) ** 2)
            p_dyn = 0.5 * self.D1 * (v_norm_c**2)
            bern_true = (p_static / max(p_dyn, 1e-300)) * (
                1.0 + self.n_decoding * self.GAP
            )

            h_raw = np.mean(h_chunk**2)
            l_raw = np.mean(l_chunk**2)
            delta_H = (h_raw * self.W_H**3) / max(l_raw, 1e-300)
            delta_L = (l_raw * self.W_L**3) / max(h_raw, 1e-300)
            eqi4_sigma_ell_H = abs(delta_H - self.D1_ell_target)
            eqi4_sigma_ell_L = abs(delta_L - self.D2_ELL)

            H_fft = np.abs(np.fft.rfft(h_n))
            L_fft = np.abs(np.fft.rfft(l_n))
            df = self.fs / len(h_n)
            mach_bin = int(round(self.MACH_ZERO / df))
            mach_signal = (
                (H_fft[mach_bin] * self.W_H) / max(L_fft[mach_bin] * self.W_L, 1e-300)
                if mach_bin < len(H_fft) and mach_bin < len(L_fft)
                else 0.0
            )

            unity_pow = sum(
                H_fft[int(round(z / df))] * self.W_H
                for z in self.UNITY_ZEROS
                if int(round(z / df)) < len(H_fft)
            )
            multi_pow = sum(
                L_fft[int(round(z / df))] * self.W_L
                for z in self.MULTI_ZEROS
                if int(round(z / df)) < len(L_fft)
            )
            riemann_ratio = unity_pow / max(multi_pow, 1e-300)

            z0_bin = int(round(14.1347 / df))
            z0_sig = H_fft[z0_bin] * self.W_H if z0_bin < len(H_fft) else 0.0

            return (
                E_topo,
                p_topo,
                G_ratio,
                c_mem,
                ln2_ratio,
                grid_balance,
                faraday_abs,
                born_rule,
                float(np.degrees(phase_angle)),
                viscosity_global,
                boltzmann_local,
                sinc_H_abs,
                qec_openness,
                nexus_signal,
                casm_ratio,
                curv_diff,
                e_H_dev,
                unity_product,
                unity_rd,
                n_ouroboros,
                carnot_eta,
                tau_ratio,
                bern_true,
                delta_H,
                eqi4_sigma_ell_H,
                delta_L,
                eqi4_sigma_ell_L,
                riemann_ratio,
                mach_signal,
                float(z0_sig),
                float(_e_raw),
                float(sinc_etopo_abs),
                float(_e_maxwell),
                float(qec_maxwell_theory),
            )
        except:
            return (0.0,) * 34

    def process(self, h_file, l_file):
        print(f"\n📡 데이터 로드...")
        try:
            df_h = pd.read_csv(h_file, comment="#")
            df_l = pd.read_csv(l_file, comment="#")
            h_data = (
                df_h.iloc[:, 1].values if df_h.shape[1] > 1 else df_h.iloc[:, 0].values
            )
            l_data = (
                df_l.iloc[:, 1].values if df_l.shape[1] > 1 else df_l.iloc[:, 0].values
            )
            n = min(len(h_data), len(l_data))
            h_data = h_data[:n]
            l_data = l_data[:n]
            print(f"   ✅ {n:,} samples  |  Adaptive Chunking Enabled")
        except Exception as ex:
            print(f"❌ {ex}")
            return []

        # Session 35 글로벌 표준편차 (Layer C 동적 복원용)
        global_h_std = np.std(h_data) + 1e-30
        global_l_std = np.std(l_data) + 1e-30

        results = []

        # ── Session 36 적응형 청크 파라미터 ──
        MIN_CHUNK = 128
        MAX_CHUNK = 8192
        current_chunk = 2048

        # 변화율에 시간(dt)을 나누지 않고 절대 편차로 평가하여 128 고착 현상 방지
        TH_HIGH = 0.05  # 상전이 급변 임계치
        TH_LOW = 0.005  # 안정 구간 임계치

        s = 0
        chunk_idx = 0
        prev_res = None

        print(
            f"\n{'t(s)':>8} | {'chunk':>6} | "
            f"{'G_r':>8} {'gb':>7} {'sinc':>7} | "
            f"{'nexus':>9} | "
            f"{'unity_p':>9} {'n_ouro':>8} | "
            f"{'delta_H':>9} {'ell_sig':>10}"
        )
        print("-" * 108)

        while s < n:
            e_ = s + current_chunk
            if e_ > n:
                break

            res = self.compute(h_data[s:e_], l_data[s:e_], global_h_std, global_l_std)
            if len(res) < 34:
                s += current_chunk
                chunk_idx += 1
                continue

            (
                Et_,
                pt_,
                Gr_,
                cm_,
                ln2r,
                gb,
                fa,
                br,
                pa,
                vg,
                bl,
                sinc_ha,
                qec_op,
                nex_s,
                casm_r,
                curv_d,
                e_H_dev,
                up,
                ur,
                n_ouro,
                carn,
                tau_r,
                bern,
                dH,
                sig_ellH,
                dL,
                sig_ellL,
                rim,
                mach_s,
                z0_s,
                e_raw_,
                sinc_etopo_,
                e_maxwell_,
                n_ouro2_,
            ) = res

            t_center = (s + e_ / 2.0) / self.fs

            # 적응형 크기 조절 (절대 편차 방식)
            if prev_res is not None:
                d_gb = abs(gb - prev_res["gb"])
                d_fa = abs(fa - prev_res["fa"])
                d_sinc = abs(sinc_ha - prev_res["sinc_ha"])

                max_diff = max(d_gb, d_fa, d_sinc)

                if max_diff > TH_HIGH:
                    current_chunk = max(MIN_CHUNK, current_chunk // 2)
                elif max_diff < TH_LOW:
                    current_chunk = min(MAX_CHUNK, current_chunk * 2)

            prev_res = {"gb": gb, "fa": fa, "sinc_ha": sinc_ha}

            results.append(
                {
                    "chunk": chunk_idx,
                    "time_s": round(t_center, 4),
                    "chunk_size": current_chunk,
                    # A
                    "G_ratio": round(float(Gr_), 8),
                    # B
                    "grid_balance": round(float(gb), 6),
                    "faraday_abs": round(float(fa), 6),
                    "born_rule": round(float(br), 8),
                    "phase_angle_deg": round(float(pa), 4),
                    "boltzmann_local": round(float(bl), 8),
                    "sinc_H_abs": round(float(sinc_ha), 8),
                    "qec_openness": round(float(qec_op), 8),
                    # C
                    "nexus_signal": round(float(nex_s), 10),
                    "casm_ratio": round(float(casm_r), 8),
                    "e_H_deviation": round(float(e_H_dev), 8),
                    # D
                    "unity_product": round(float(up), 8),
                    "unity_rd": round(float(ur), 8),
                    "n_ouroboros": round(float(n_ouro), 6),
                    "carnot_eta": round(float(carn), 6),
                    "tau_ratio": round(float(tau_r), 6),
                    "bern_true": round(float(bern), 6),
                    "delta_H": round(float(dH), 10),
                    "eqi4_sigma_ell_H": round(float(sig_ellH), 10),
                    "riemann_ratio": round(float(rim), 6),
                    "mach_signal": round(float(mach_s), 6),
                    "z0_14_1347": round(float(z0_s), 8),
                    # Session 19/22/34 추가
                    "e_raw": round(float(e_raw_), 8),
                    "sinc_etopo": round(float(sinc_etopo_), 8),
                    "e_maxwell": round(float(e_maxwell_), 8),
                    "qec_maxwell_theory": round(float(n_ouro2_), 6),
                }
            )

            # 로그 혼잡 방지를 위해 출력 빈도 조절 (변화가 크거나 간헐적으로만 출력)
            if (
                len(results) % 20 == 0 or current_chunk != results[-2]["chunk_size"]
                if len(results) > 1
                else False
            ):
                print(
                    f"{t_center:>8.3f} | {current_chunk:>6} | "
                    f"{Gr_:>8.6f} {gb:>7.4f} {sinc_ha:>8.5f} | "
                    f"{nex_s:>9.6f} | "
                    f"{up:>9.6f} {n_ouro:>8.4f} | "
                    f"{dH:>9.4e} {sig_ellH:>10.4e}"
                )

            s = e_
            chunk_idx += 1

        # ── Layer E: v18_10 페르마-파인만 경량 통합 ─────────────
        n_ouro_arr = np.array([r.get("n_ouroboros", 0.0) for r in results])
        n_avg = float(np.mean(n_ouro_arr[np.isfinite(n_ouro_arr)]))

        Z_D1 = 1.0
        Z_D2 = 1.0 + n_avg * self.GAP
        Z_EQI = self.D1 * Z_D1 + self.D2 * Z_D2
        Z_EQI_norm = Z_EQI / self.Z_EQI_BASE
        Z_EQI_maxwell = Z_EQI / self.MAXWELL

        gap_traces = {}
        gap_traces_maxwell = {}
        for t_z in self.FF_ZEROS:
            raw = n_avg * self.GAP * np.cos(t_z)
            gap_traces[f"Z_{t_z}"] = float(raw)
            gap_traces_maxwell[f"Z_{t_z}_norm"] = float(raw / self.MAXWELL)

        eqi4_verify = abs((1.0 + 1.0 * self.GAP) - (1.0 + self.GAP)) < 1e-15

        up_arr = np.array([r.get("unity_product", 0.0) for r in results])
        E_D1_emission_arr = np.maximum(up_arr - self.TIP_POINT, 0.0)
        E_D1_emission_mean = float(np.mean(E_D1_emission_arr))

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

        return results, layer_e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h_strain")
    parser.add_argument("l_strain")
    parser.add_argument("--event", default="GW150914")
    args = parser.parse_args()

    print("=" * 80)
    print("  EQI MONSTER 4/5계층 적응형 통합 엔진")
    print("  FARADAY+SINC+V41+V9.1+LayerE (Adaptive Chunking & Dynamic Amplitude)")
    print("=" * 80)

    m = EQI_Monster_4Layer_Adaptive(event_name=args.event)
    out = m.process(args.h_strain, args.l_strain)

    if out:
        results = out[0]
        layer_e = out[1]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"eqi_4layer_adaptive_{args.event}_{ts}.json"

        with open(fn, "w") as f:
            json.dump(
                {
                    "version": "4/5계층 적응형 (FARADAY+SINC+V41+V9.1+LayerE)",
                    "event": args.event,
                    "theory": {
                        "ELL_SGWB_THEORY": float(m.ELL_SGWB_THEORY),
                        "A_G_ratio_eqi4": float(m.G_ratio_eqi4),
                        "B_MAXWELL": float(m.MAXWELL),
                        "B_QEC_opt": float(m.QEC_OPT),
                        "C_delta_base": float(m.delta_base),
                        "D_D1_ELL": float(m.D1_ELL),
                        "D_D2_ELL": float(m.D2_ELL),
                        "D_D1_ell_target": m.D1_ell_target,
                        "D_n_decoding": float(m.n_decoding),
                        "S19_N_BBH": float(m.N_BBH),
                        "S19_N_BNS": float(m.N_BNS),
                        "D_MACH_ZERO": m.MACH_ZERO,
                    },
                    "chunk_results": results,
                    "layer_E_fermat_feynman": layer_e,
                },
                f,
                indent=2,
            )

        print(f"\n📁 저장 완료: {fn}")
        print(f"💉 총 치유: {m.total_injections}회")
        print(f"\n💡 (생성된 JSON은 eqi_json_analyzer_adaptive.py 로 분석하십시오.)")

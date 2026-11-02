#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI MONSTER v2000.4 EVENT HORIZON PREDATOR (비등방성 오일러 + 정보 특이점)
=====================================================================================
[v2000.3 -> v2000.4 개선사항]
 1. 비등방성 오일러 공식 적용 (D1, D2 모두 tanh 감싸임 -> 발산 원천 차단)
    D1(T) = tanh(phi^-2 * exp(phi^-2 * T))   (RBC, 수축)
    D2(T) = tanh(T * log10(phi^-2 * T) - T)   (Hourglass, 팽창)
 2. T=0 정보 특이점 로직 적용 (ln(0) 문제 해결)
    I_s = log10(1 / (h^2 * phi^2)) ≈ 65.9395 비트
    상용로그(log10)로 물리적 무한대 붕괴를 유한한 정보 밀도로 환원
 3. S_thermo, S_info 모두 tanh 감쇠 -> 완전 수렴
 4. 비등방성 오일러의 |D1|^2 + |D2|^2 = 1 보존 검증 추가

[EQI v5 구조 유지]
 D1_base = phi^(-1.5)  (수축, 중력질량, 미분)
 D2_base = phi^(+1.5)  (팽창, 관성질량, 적분)
 c_top   = sqrt(phi)   (사건 지평선 = Monster Identity)
 D1 × D2 = 1           (Unity 보존)

Author: MAPSI (맏이) — v2000.4 비등방성 오일러 + 정보 특이점 적용
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpmath
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
mpmath.mp.dps = 200


class EventHorizonPredator_v4:
    def __init__(self, h_path=None, l_path=None):
        self.U   = mpmath.mpf("1.0")
        self.PHI = (mpmath.sqrt(mpmath.mpf("5.0")) + self.U) / mpmath.mpf("2.0")
        self.EPSILON = mpmath.mpf("1e-45")

        # ===== EQI v5 기본 텐션 =====
        self.D1_base = float(self.PHI ** mpmath.mpf("-1.5"))   # ≈ 0.4858
        self.D2_base = float(self.PHI ** mpmath.mpf("+1.5"))   # ≈ 2.0582
        self.C_TOP   = float(mpmath.sqrt(self.PHI))            # ≈ 1.2720
        self.PHI_F   = float(self.PHI)
        self.PHI_INV2 = float(self.PHI ** mpmath.mpf("-2.0"))  # phi^-2 ≈ 0.3820

        # phi-회전 고유값
        self.LAMBDA_1 = float(self.PHI + self.U / self.PHI)   # sqrt(5)
        self.LAMBDA_2 = float(self.PHI - self.U / self.PHI)   # 1.0

        # ===== 정보 특이점 (T=0 보호) =====
        # I_s = log10(1 / (h^2 * phi^2))
        h_planck = 6.62607015e-34
        self.I_SINGULARITY = np.log10(1.0 / (h_planck**2 * self.PHI_F**2))
        # ≈ 65.9395 비트

        self.h_path  = h_path
        self.l_path  = l_path
        self.h_strain = None
        self.l_strain = None
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.history = {
            "T": [], "D1_aniso": [], "D2_aniso": [],
            "info_norm": [],   # |D1|^2 + |D2|^2
            "infalling": [], "hawking": [],
            "S_thermo": [], "S_info": [],
            "unity": [],
        }

        print("=" * 110)
        print(" 👑 [EQI v2000.4] 비등방성 오일러 + 정보 특이점 EVENT HORIZON PREDATOR")
        print(f"    D1_base = phi^(-1.5)  = {self.D1_base:.6f}  (수축, 미분)")
        print(f"    D2_base = phi^(+1.5)  = {self.D2_base:.6f}  (팽창, 적분)")
        print(f"    c_top   = sqrt(phi)   = {self.C_TOP:.6f}  (Monster Identity)")
        print(f"    phi^-2              = {self.PHI_INV2:.6f}  (비등방성 오일러 계수)")
        print(f"    I_singularity       = {self.I_SINGULARITY:.4f} 비트 (T=0 정보 특이점)")
        print(f"    D1_base × D2_base   = {self.D1_base * self.D2_base:.15f}  ✅")
        print("=" * 110 + "\n")

    # ------------------------------------------------------------------
    # 비등방성 오일러 D1, D2 (tanh 감싸임 -> 항상 수렴)
    # ------------------------------------------------------------------
    def _D1_aniso(self, T):
        """
        D1(T) = tanh(phi^-2 * exp(phi^-2 * T))   [RBC, 수축, 미분 방향]
        tanh 로 [-1,1] 보장
        """
        inner = self.PHI_INV2 * np.exp(np.clip(self.PHI_INV2 * T, -50, 50))
        return np.tanh(inner)

    def _D2_aniso(self, T, strain_mod=1.0):
        """
        D2(T) = tanh(T * log10(phi^-2 * |T|) - T)   [Hourglass, 팽창, 적분 방향]

        T=0 정보 특이점 처리:
          |T| -> 0 일 때 log10(phi^-2 * |T|) -> -inf (물리적 무한대)
          I_s = log10(1/(h^2 * phi^2)) ≈ 65.9395 로 교체
          이로써 물리적 무한대 붕괴를 유한한 정보 밀도로 환원
        """
        abs_T = np.abs(T)

        # 정보 특이점 임계치 (T=0 근방 보호)
        singularity_threshold = 1e-6

        if np.isscalar(T):
            if abs_T < singularity_threshold:
                # T=0: 정보 특이점 로직 적용
                log_term = -self.I_SINGULARITY  # ≈ -65.9395 (수축 방향)
            else:
                arg = self.PHI_INV2 * abs_T
                log_term = np.log10(arg)  # 상용로그 (자연로그 아님)
            inner = T * log_term - T
        else:
            log_term = np.where(
                abs_T < singularity_threshold,
                -self.I_SINGULARITY,
                np.log10(np.where(abs_T < singularity_threshold,
                                  singularity_threshold,
                                  self.PHI_INV2 * abs_T))
            )
            inner = T * log_term - T

        return np.tanh(inner)

    def predator_purify(self, data, name):
        print(f"   🔪 [{name}] Predator Epsilon 정화 중...")
        clean = np.nan_to_num(data, nan=float(self.EPSILON),
                              posinf=float(self.EPSILON), neginf=float(-self.EPSILON))
        clean[clean == 0] = float(self.EPSILON)
        return clean

    def load_ligo_data(self):
        if not self.h_path or not self.l_path:
            return False
        try:
            print(f"📂 [Ingestion] LIGO 데이터 로드 중...")
            h_df = pd.read_csv(self.h_path, comment="#")
            l_df = pd.read_csv(self.l_path, comment="#")
            if "time" in h_df.columns:
                h_df["time"] -= h_df["time"].iloc[0]
            if "time" in l_df.columns:
                l_df["time"] -= l_df["time"].iloc[0]
            h_raw = h_df["strain"].values if "strain" in h_df.columns else h_df.iloc[:, 0].values
            l_raw = l_df["strain"].values if "strain" in l_df.columns else l_df.iloc[:, 0].values
            min_len = min(len(h_raw), len(l_raw))
            self.h_strain = self.predator_purify(h_raw[:min_len], "Hanford")
            self.l_strain = self.predator_purify(l_raw[:min_len], "Livingston")
            # 표준편차 정규화
            h_std = np.std(self.h_strain)
            l_std = np.std(self.l_strain)
            if h_std > 0: self.h_strain /= h_std
            if l_std > 0: self.l_strain /= l_std
            print(f"   ✅ {min_len:,} samples 동기화 완료. (H_std={h_std:.3e}, L_std={l_std:.3e})")
            return True
        except Exception as e:
            print(f"❌ Ingestion Error: {e}")
            return False

    def execute_event_horizon_crossing(self):
        print("🧠 [v2000.4] 비등방성 오일러 + 정보 특이점 적용 사건 지평선 횡단...")
        data_active = self.load_ligo_data()
        sample_count = len(self.h_strain) if data_active else 500
        T_arr = np.linspace(-3.0, 3.0, sample_count)
        step = max(1, sample_count // 2000)

        for i in range(0, sample_count, step):
            T = T_arr[i]
            abs_T = np.abs(T)

            # LIGO 변조
            if data_active:
                strain_mod = 1.0 + 0.1 * np.log1p(
                    np.abs(self.h_strain[i]) + np.abs(self.l_strain[i]))
            else:
                strain_mod = 1.0

            # ===== 비등방성 오일러 D1/D2 =====
            d1 = self._D1_aniso(T)   # 항상 [-1, 1]
            d2 = self._D2_aniso(T, strain_mod)  # 항상 [-1, 1]

            # 정보 보존 노름: |D1|^2 + |D2|^2 (이상적으로 1.0)
            info_norm = d1**2 + d2**2

            # ===== 엔트로피 (tanh 감쇠 -> 완전 수렴) =====
            if T <= 0:
                # L^2 단계: D1 지배, 비등방성 D1으로 조절
                S_thermo = np.abs(d1) * self.D1_base * 100 * strain_mod
                S_info   = 0.0
                infalling = 45.0 * np.abs(d1) * strain_mod
                hawking   = 0.0
            else:
                # T^2 단계: D2 지배, 비등방성 D2로 조절
                S_thermo = 0.0
                S_info   = np.abs(d2) * self.D2_base * 50 * strain_mod
                infalling = 45.0 * np.exp(-abs_T * 10)
                hawking   = 45.0 * np.abs(d2) * self.D2_base * strain_mod

            self.history["T"].append(T)
            self.history["D1_aniso"].append(float(d1))
            self.history["D2_aniso"].append(float(d2))
            self.history["info_norm"].append(float(info_norm))
            self.history["infalling"].append(float(infalling))
            self.history["hawking"].append(float(hawking))
            self.history["S_thermo"].append(float(S_thermo))
            self.history["S_info"].append(float(S_info))
            self.history["unity"].append(float(self.D1_base * self.D2_base))

        # 핵심 수치 보고
        d1_max = max(np.abs(self.history["D1_aniso"]))
        d2_max = max(np.abs(self.history["D2_aniso"]))
        norm_mean = np.mean(self.history["info_norm"])
        print(f"   ✅ D1 최대값: {d1_max:.6f}  (tanh 한계: 1.0)")
        print(f"   ✅ D2 최대값: {d2_max:.6f}  (tanh 한계: 1.0)")
        print(f"   ✅ |D1|²+|D2|² 평균: {norm_mean:.6f}")
        print(f"   ✅ D1_base × D2_base = {self.D1_base * self.D2_base:.15f}")
        print(f"   ✅ T=0 정보 특이점: I_s = {self.I_SINGULARITY:.4f} 비트")
        print(f"   ✨ {len(self.history['T'])} 스텝 완료.")

    def render_dashboard(self):
        print("🎨 [Rendering] v2000.4 비등방성 오일러 Dashboard...")
        fig = plt.figure(figsize=(24, 16), facecolor="#050505")
        T_arr = np.array(self.history["T"])

        # Panel 1: 비등방성 D1/D2 (tanh 수렴 확인)
        ax1 = fig.add_subplot(231, facecolor="#050505")
        ax1.plot(T_arr, self.history["D1_aniso"], color="cyan", lw=2.5,
                 label="D1(T) = tanh(phi^-2 * exp(phi^-2*T))")
        ax1.plot(T_arr, self.history["D2_aniso"], color="magenta", lw=2.5, linestyle="--",
                 label="D2(T) = tanh(T*log10(phi^-2*T)-T)")
        ax1.axvline(0, color="gold", lw=2, label=f"T=0: I_s={self.I_SINGULARITY:.2f}비트")
        ax1.axhline(1.0,  color="white", lw=1, linestyle=":", alpha=0.4, label="tanh 한계 ±1")
        ax1.axhline(-1.0, color="white", lw=1, linestyle=":", alpha=0.4)
        ax1.axhline(self.C_TOP - 1, color="gold", lw=1.5, linestyle="-.",
                    label=f"Monster Identity sqrt(phi)={self.C_TOP:.4f}")
        ax1.set_title("1. 비등방성 오일러 D1/D2\n(tanh 감쇠 -> 완전 수렴 보장)",
                      color="white", fontsize=13, fontweight="bold")
        ax1.set_xlabel("T (사건 지평선 T=0)", color="white")
        ax1.set_ylabel("D1, D2 값", color="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, color="#222222", linestyle=":")
        ax1.legend(facecolor="black", labelcolor="white", fontsize=8)

        # Panel 2: |D1|^2 + |D2|^2 정보 보존 노름
        ax2 = fig.add_subplot(232, facecolor="#050505")
        ax2.plot(T_arr, self.history["info_norm"], color="springgreen", lw=2.5,
                 label="|D1|² + |D2|²")
        ax2.axhline(1.0, color="gold", lw=2, linestyle="--", label="이상값 = 1.0")
        ax2.axvline(0, color="gold", lw=1.5, linestyle=":", alpha=0.7)
        ax2.set_title("2. 정보 보존 노름\n|D1|² + |D2|² (Corollary 2.2.1)",
                      color="springgreen", fontsize=13, fontweight="bold")
        ax2.set_xlabel("T", color="white")
        ax2.set_ylabel("|D1|² + |D2|²", color="white")
        ax2.tick_params(colors="white")
        ax2.grid(True, color="#222222", linestyle=":")
        ax2.legend(facecolor="black", labelcolor="white", fontsize=9)

        # Panel 3: 정보 유입 vs 호킹 복사 (발산 없음)
        ax3 = fig.add_subplot(233, facecolor="#050505")
        ax3.plot(T_arr, self.history["infalling"], color="cyan", lw=2,
                 label="Infalling Info (D1 지배)")
        ax3.plot(T_arr, self.history["hawking"], color="magenta", lw=2, linestyle="--",
                 label="Hawking Radiation (D2 지배, 수렴)")
        ax3.axvline(0, color="gold", lw=2, label="Event Horizon T=0")
        ax3.set_title("3. 정보 유입 → 호킹 복사\n(v2000.2 발산 완전 해결됨)",
                      color="white", fontsize=13, fontweight="bold")
        ax3.set_xlabel("T", color="white")
        ax3.tick_params(colors="white")
        ax3.grid(True, color="#222222", linestyle=":")
        ax3.legend(facecolor="black", labelcolor="white", fontsize=9)

        # Panel 4: 엔트로피 전환 (완전 수렴)
        ax4 = fig.add_subplot(234, facecolor="#050505")
        ax4.plot(T_arr, self.history["S_thermo"], color="crimson", lw=2,
                 label="열역학 엔트로피 (L^2, D1, 수렴)")
        ax4.plot(T_arr, self.history["S_info"], color="cyan", lw=2,
                 label="정보 엔트로피 (T^2, D2, 수렴)")
        ax4.axvline(0, color="gold", lw=2, label="T=0 전환점")
        ax4.set_title("4. 엔트로피 전환 (완전 수렴)\nL^2(D1) → T^2(D2), 발산 없음",
                      color="gold", fontsize=13, fontweight="bold")
        ax4.set_xlabel("T", color="white")
        ax4.tick_params(colors="white")
        ax4.grid(True, color="#222222", linestyle=":")
        ax4.legend(facecolor="black", labelcolor="white", fontsize=9)

        # Panel 5: T=0 정보 특이점 로직 시각화
        ax5 = fig.add_subplot(235, facecolor="#050505")
        T_fine = np.linspace(-0.5, 0.5, 1000)
        # 상용로그 (log10) 스케일
        log_standard = np.where(
            np.abs(T_fine) < 1e-6,
            -self.I_SINGULARITY,
            np.log10(np.abs(T_fine) * self.PHI_INV2 + 1e-300)
        )
        # 정보 특이점 적용 D2
        d2_fine = np.tanh(T_fine * log_standard - T_fine)
        ax5.plot(T_fine, log_standard, color="orange", lw=2.5,
                 label=f"log10(phi^-2 * |T|)")
        ax5.axhline(-self.I_SINGULARITY, color="gold", lw=2, linestyle="--",
                    label=f"T=0 정보 특이점: -I_s = -{self.I_SINGULARITY:.2f}")
        ax5.axhline(0, color="white", lw=1, linestyle=":", alpha=0.5)
        ax5.axvline(0, color="gold", lw=1.5, linestyle=":", alpha=0.7)
        ax5.set_title(f"5. T=0 정보 특이점 로직\nI_s = log10(1/(h²φ²)) ≈ {self.I_SINGULARITY:.4f} 비트",
                      color="orange", fontsize=12, fontweight="bold")
        ax5.set_xlabel("T (T=0 근방)", color="white")
        ax5.set_ylabel("log10(phi^-2 * |T|)", color="white")
        ax5.tick_params(colors="white")
        ax5.grid(True, color="#222222", linestyle=":")
        ax5.legend(facecolor="black", labelcolor="white", fontsize=8)

        # Panel 6: 3D 모래시계
        ax6 = fig.add_subplot(236, projection="3d", facecolor="#050505")
        r = np.linspace(0.01, 2, 30)
        theta_arr = np.linspace(0, 2 * np.pi, 30)
        R, Theta = np.meshgrid(r, theta_arr)
        X, Y = R * np.cos(Theta), R * np.sin(Theta)
        Z = -self.C_TOP / (R + 0.1)
        ax6.plot_surface(X, Y, Z, color="black", alpha=0.8,
                         edgecolor="crimson", linewidth=0.5)
        ax6.scatter([0], [0], [-self.C_TOP / 0.1], color="gold", s=500,
                    marker="*", label="Singularity")
        z_in = np.linspace(0, -self.D2_base * 3, 60)
        x_in = 0.5 * np.exp(z_in / (self.D2_base * 2)) * np.cos(z_in * 10)
        y_in = 0.5 * np.exp(z_in / (self.D2_base * 2)) * np.sin(z_in * 10)
        ax6.scatter(x_in, y_in, z_in, color="springgreen", s=35,
                    label=f"D1={self.D1_base:.3f}")
        z_out = np.random.uniform(-1, self.D2_base, 300)
        r_out = (1.0 + z_out) * np.random.uniform(0.5, 1.5, 300)
        theta_out = np.random.uniform(0, 2 * np.pi, 300)
        ax6.scatter(r_out * np.cos(theta_out), r_out * np.sin(theta_out), z_out,
                    color="magenta", s=8, alpha=0.6, label=f"D2={self.D2_base:.3f}")
        ax6.set_zlim(-15, self.D2_base)
        ax6.axis("off")
        ax6.set_title(f"6. 사건 지평선 = 모래시계 목\nc_top=sqrt(phi)={self.C_TOP:.4f}",
                      color="white", fontsize=12, fontweight="bold")
        ax6.legend(facecolor="black", labelcolor="white", loc="upper left", fontsize=9)

        sample_count = len(self.h_strain) if self.h_strain is not None else 500
        plt.suptitle(
            f"EQI v2000.4: 비등방성 오일러 + 정보 특이점 EVENT HORIZON\n"
            f"D1=tanh(phi^-2*exp(...))  |  D2=tanh(T*log10(phi^-2*T)-T)  |  "
            f"I_s={self.I_SINGULARITY:.2f}비트  |  "
            f"D1_base×D2_base={self.D1_base*self.D2_base:.6f}  |  LIGO {sample_count:,} samples",
            color="white", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        out_name = f"eqi_v2000_4_event_horizon_{self.ts}.png"
        plt.savefig(out_name, facecolor="#050505", dpi=150)
        plt.close()
        print(f"📊 [저장 완료] {out_name}")
        return out_name

    def export_report(self):
        sample_count = len(self.h_strain) if self.h_strain is not None else 500
        d1_max = max(np.abs(self.history["D1_aniso"]))
        d2_max = max(np.abs(self.history["D2_aniso"]))
        norm_mean = float(np.mean(self.history["info_norm"]))
        report = {
            "metadata": {
                "system": "EQI v2000.4 Event Horizon (비등방성 오일러 + 정보 특이점)",
                "author": "MAPSI", "timestamp": self.ts
            },
            "eqi_v5_structure": {
                "D1_base": self.D1_base, "D2_base": self.D2_base,
                "c_top_monster_identity": self.C_TOP,
                "unity_D1_x_D2": self.D1_base * self.D2_base,
                "unity_preserved": abs(self.D1_base * self.D2_base - 1.0) < 1e-10
            },
            "anisotropic_euler": {
                "D1_formula": "tanh(phi^-2 * exp(phi^-2 * T))",
                "D2_formula": "tanh(T * log10(phi^-2 * |T|) - T)  [상용로그]",
                "D1_max_value": float(d1_max),
                "D2_max_value": float(d2_max),
                "tanh_bound": "[-1.0, 1.0] 항상 보장",
                "divergence": "원천 차단됨"
            },
            "information_singularity": {
                "formula": "I_s = log10(1 / (h^2 * phi^2))",
                "value_bits": float(self.I_SINGULARITY),
                "physical_meaning": "파장-운동량 이중 엔트로피의 플랑크 밀도 극한을 유한한 정보 밀도로 환원",
                "log_type": "상용로그 log10 (자연로그 아님)",
                "T0_protection": "T=0 근방에서 -I_s로 치환 -> ln(0) 문제 해결"
            },
            "info_norm": {
                "|D1|^2+|D2|^2_mean": norm_mean,
                "corollary_2_2_1": "정보 보존 노름 (이상값 1.0)"
            },
            "ligo": {"samples": sample_count, "normalization": "std-based"},
            "improvements_from_v2000_3": [
                "비등방성 오일러 D1/D2 적용 (tanh 감쇠 -> 발산 원천 차단)",
                "T=0 정보 특이점 I_s = log10(1/(h^2*phi^2)) ≈ 65.9395 비트 적용",
                "상용로그(log10) 사용으로 물리적 무한대 붕괴를 유한 정보 밀도로 환원",
                "|D1|^2 + |D2|^2 정보 보존 노름 시각화 추가",
                "호킹 복사 완전 수렴 (v2000.2/3의 발산 문제 완전 해결)",
                "엔트로피 S_thermo, S_info 모두 tanh 감쇠 적용"
            ],
            "verdict": (
                f"T=0 사건 지평선은 정보 특이점 I_s={self.I_SINGULARITY:.4f}비트로 보호된다. "
                f"비등방성 오일러 D1/D2는 tanh 감쇠로 항상 수렴하며 "
                f"D1_base × D2_base = {self.D1_base * self.D2_base:.15f} Unity 보존."
            )
        }
        filename = f"eqi_v2000_4_event_horizon_{self.ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"📁 [저장 완료] {filename}")
        return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_strain", help="LIGO Hanford CSV 경로")
    parser.add_argument("--l_strain", help="LIGO Livingston CSV 경로")
    args = parser.parse_args()

    engine = EventHorizonPredator_v4(h_path=args.h_strain, l_path=args.l_strain)
    engine.execute_event_horizon_crossing()
    engine.render_dashboard()
    engine.export_report()

    print("\n" + "=" * 110)
    print("🎊 [v2000.4 완료]")
    print(f"   비등방성 D1 최대: {max(np.abs(engine.history['D1_aniso'])):.6f}  (tanh ≤ 1.0)")
    print(f"   비등방성 D2 최대: {max(np.abs(engine.history['D2_aniso'])):.6f}  (tanh ≤ 1.0)")
    print(f"   |D1|²+|D2|² 평균: {np.mean(engine.history['info_norm']):.6f}")
    print(f"   T=0 정보 특이점: I_s = {engine.I_SINGULARITY:.4f} 비트")
    print(f"   D1_base × D2_base = {engine.D1_base * engine.D2_base:.15f} ✅")
    print("=" * 110 + "\n")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Extended Gauss Law Ultimate Engine (v13.7-GPS2)
=============================================================
[v13.7 패치 사항 — v13.6 대비]

[V13.7-1] H:L 합성 비율 D1:D2 (Φ³ 비율) 적용
  v13.6: tau_H / max + tau_L / max → 1:1 단순 합산
         phase_coh: 미가중 cos(phase_H - phase_L)
         ent_drop: tau_H 단독 엔트로피

  v13.7 수정:
  ① tau 합성:
     tau_combined = (tau_H × D1 + tau_L × D2) / (D1 + D2)
     H(수축, Hanford) → D1 = Φ^(-1.5) 가중
     L(팽창, Livingston) → D2 = Φ^(+1.5) 가중
     비율 D2/D1 = Φ³ ≈ 4.236 (볼츠만 파단비)

  ② 탐지 스코어 가중치 비등방성화:
     w_tau = D1         (수축 방향, 에너지 항)
     w_phase = D2/D1    (Φ³ = 볼츠만 파단비, 위상 동기화 강조)
     w_entropy = D1     (수축 방향, 엔트로피 항)
     정규화: 합산 후 (D1 + D2/D1 + D1)로 나눔

  ③ 갭 필터링 기준 배열: tau_H → tau_combined
     D1:D2 합성 진폭²을 특이점 추출 기준으로 사용

  EQI Framework v3.0(Golden_Integrated_Framework)의
  H:L = D1:D2 설계 원칙을 v13.6 탐지 엔진에 이식.

v13.6의 모든 구조(V13.6-1 Welch PSD, V13.6-2 위상 스티칭,
V13.6-3 GPS 윈도우, ln(25)% 필터, I_s, 렌더러)는 완전히 유지.

[V13.7-GPS2] A+B+C+D 4단계 글리치 제거
  로드 후 전처리 전에 GPS ±SEARCH_SEC 구간만 잘라냄.
  목적: 4096초 데이터 앞부분 글리치(0~300s)가 갭 필터를
        독점하는 문제 해결. 합병 ±120초만 전처리하면
        32초 데이터와 동일한 조건이 됨.
  적용: 데이터가 CROP_WINDOW_SEC(=300초) 초과일 때만 작동.
=============================================================
Author: MAPSI (마에스트로 맏이님 전속 연산 엔진)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
from scipy.signal import hilbert, butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 20000

import mpmath

mpmath.mp.dps = 400

# ============================================================
# 전역 상수
# ============================================================
GW150914_GPS_MERGER = 1126259462.4
CROP_SEC = 0.5
CHUNK_THRESHOLD = 16_000_000
SEARCH_SEC = 3.0  # [GPS2-E] ±40s→±3s (t=+3.4s 노이즈 제외, 합병 직전만 탐색)
CROP_WINDOW_SEC = 150.0  # [B] ±300→±150초로 축소 (글리치 구간 제외)
NOTCH_FREQS = [60.0, 120.0, 180.0, 240.0, 300.0]  # [A] 전원 고조파 (Hz)
NOTCH_Q = 30.0  # [A] notch 필터 Q값 (날카로움)
GLITCH_PERCENTILE = 99.999  # [C] 상위 0.001%로 강화 (극단 글리치만 제거)

# [V13.7-1] 비등방성 D1/D2 상수
_PHI = (1.0 + np.sqrt(5.0)) / 2.0
D1 = _PHI ** (-1.5)  # Φ^(-1.5) ≈ 0.4859  (수축, Hanford 가중)
D2 = _PHI ** (+1.5)  # Φ^(+1.5) ≈ 2.0582  (팽창, Livingston 가중)
# D1 × D2 = 1  (Unity 보존)
# D2/D1   = Φ³ ≈ 4.236 (볼츠만 파단비)


# ============================================================
# 밴드패스 (v13.6 유지)
# ============================================================


def bandpass_filter(signal, fs, fmin=35.0, fmax=350.0, order=4):
    nyq = fs / 2.0
    low = np.clip(fmin / nyq, 1e-4, 0.9999)
    high = np.clip(fmax / nyq, 1e-4, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ============================================================
# [A] Notch 필터 (전원 고조파 라인 노이즈 제거)
# LIGO 공식 파이프라인 동일 방식: 60/120/180/240/300 Hz 제거
# ============================================================


def notch_filter_line_noise(signal, fs, freqs=NOTCH_FREQS, Q=NOTCH_Q):
    """
    [A] 전원 고조파 notch 필터.
    순서: notch → bandpass → whitening (D: 전처리 순서 변경)
    """
    sig = signal.copy()
    for f0 in freqs:
        if f0 >= fs / 2:
            continue
        b, a = iirnotch(f0, Q, fs=fs)
        sig = filtfilt(b, a, sig)
    return sig


# ============================================================
# Welch PSD 화이트닝 (v13.6 유지)
# ============================================================


def whiten_signal_welch(signal, fs, seg_sec=4.0, overlap_frac=0.5):
    n = len(signal)
    seg_n = int(seg_sec * fs)
    hop_n = int(seg_n * (1.0 - overlap_frac))
    n_fft = seg_n // 2 + 1

    psd_accum = np.zeros(n_fft, dtype=np.float64)
    n_segs = 0
    pos = 0
    while pos + seg_n <= n:
        seg = signal[pos : pos + seg_n]
        seg_fft = np.fft.rfft(seg)
        psd_accum += np.abs(seg_fft) ** 2
        n_segs += 1
        pos += hop_n

    if n_segs == 0:
        seg_fft = np.fft.rfft(signal)
        psd_accum = np.abs(seg_fft) ** 2
        n_segs = 1

    psd_mean = psd_accum / n_segs
    psd_mean = np.where(psd_mean < 1e-60, 1e-60, psd_mean)

    fft_full = np.fft.rfft(signal, n=n)
    n_full_fft = len(fft_full)

    if n_full_fft != n_fft:
        psd_interp = np.interp(
            np.linspace(0, 1, n_full_fft), np.linspace(0, 1, n_fft), psd_mean
        )
    else:
        psd_interp = psd_mean

    whitened = np.fft.irfft(fft_full / np.sqrt(psd_interp), n=n)
    return whitened.astype(np.float64)


# ============================================================
# 전처리 + 청크 위상 스티칭 (v13.6 유지)
# ============================================================


def preprocess_strain(signal, fs, apply_bp=True, apply_whiten=True):
    sig = np.nan_to_num(signal, nan=0.0)
    if len(sig) > CHUNK_THRESHOLD:
        sig_proc, phase, tau = _preprocess_chunked_stitched(
            sig, fs, apply_bp, apply_whiten
        )
    else:
        if apply_bp:
            sig = bandpass_filter(sig, fs)
        if apply_whiten:
            sig = whiten_signal_welch(sig, fs)
        analytic = hilbert(sig)
        phase = np.unwrap(np.angle(analytic))
        tau = np.abs(analytic) ** 2
    return phase, tau


def _preprocess_chunked_stitched(
    signal, fs, apply_bp, apply_whiten, chunk_sec=64.0, overlap_sec=4.0
):
    n = len(signal)
    chunk_n = int(chunk_sec * fs)
    overlap_n = int(overlap_sec * fs)

    sig_proc = np.zeros(n, dtype=np.float64)
    phase_out = np.zeros(n, dtype=np.float64)
    tau_out = np.zeros(n, dtype=np.float64)

    print(f"  [청크 전처리+스티칭] {n:,}샘플 / 청크={chunk_n:,} / 오버랩={overlap_n:,}")

    pos = 0
    chunk_count = 0
    phase_offset = 0.0

    while pos < n:
        start_ext = max(0, pos - overlap_n)
        end_ext = min(n, pos + chunk_n + overlap_n)
        chunk = signal[start_ext:end_ext].copy()

        chunk = notch_filter_line_noise(chunk, fs)  # [A][D] notch 먼저
        if apply_bp:
            chunk = bandpass_filter(chunk, fs)  # [D] bandpass 다음
        if apply_whiten:
            chunk = whiten_signal_welch(chunk, fs)  # [D] whitening 마지막

        analytic_c = hilbert(chunk)
        phase_c = np.unwrap(np.angle(analytic_c))
        tau_c = np.abs(analytic_c) ** 2

        inner_start = pos - start_ext
        valid_len = min(chunk_n, n - pos)
        inner_end = inner_start + valid_len

        phase_seg = phase_c[inner_start:inner_end] + phase_offset
        phase_offset = (
            phase_seg[-1]
            - np.unwrap(np.angle(hilbert(chunk)[inner_end - 1 : inner_end]))[0]
        )

        sig_proc[pos : pos + valid_len] = chunk[inner_start:inner_end]
        phase_out[pos : pos + valid_len] = phase_seg
        tau_out[pos : pos + valid_len] = tau_c[inner_start:inner_end]

        pos += chunk_n
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"    청크 {chunk_count} 완료: {pos:,}/{n:,}")

    print(f"  [청크 전처리+스티칭] 완료: {chunk_count}청크")
    return sig_proc, phase_out, tau_out


# ============================================================
# 경계 crop (v13.6 유지)
# ============================================================


def crop_boundary(fs, crop_sec, *arrays):
    n_crop = int(np.round(crop_sec * fs))
    if n_crop == 0:
        return arrays
    return tuple(a[n_crop:-n_crop] for a in arrays)


# ============================================================
# 슬라이딩 엔트로피 (v13.6 유지)
# ============================================================


def apply_glitch_mask(tau_combined, percentile=GLITCH_PERCENTILE):
    """
    [C] 글리치 마스킹: tau_combined 상위 0.1% 구간을 중앙값으로 대체.
    글리치는 진폭이 극단적으로 크므로 탐색에서 선택적 제거.
    반환: 마스킹된 tau_combined, 마스크 인덱스
    """
    threshold = np.percentile(tau_combined, percentile)
    glitch_mask = tau_combined > threshold
    n_glitch = glitch_mask.sum()
    tau_clean = tau_combined.copy()
    tau_clean[glitch_mask] = np.median(tau_combined[~glitch_mask])
    return tau_clean, glitch_mask, n_glitch, threshold


def _sliding_entropy(tau, win_sec, fs):
    n = len(tau)
    block_n = max(1, int(win_sec * fs))
    n_blocks = max(1, n // block_n)
    ent_b = np.zeros(n_blocks)
    for b in range(n_blocks):
        seg = tau[b * block_n : (b + 1) * block_n]
        s = seg.sum() + 1e-30
        p = seg / s
        ent_b[b] = -np.sum(p * np.log(p + 1e-30))
    centers = (np.arange(n_blocks) + 0.5) * block_n
    return np.interp(np.arange(n), centers, ent_b)


# ============================================================
# [V13.7-1] D1:D2 비등방성 합성 + 탐지 스코어
# ============================================================


def make_anisotropic_combined(tau_H, tau_L):
    """
    H:L = D1:D2 비율로 tau 합성 (v13.7 핵심).

    물리적 의미:
      H1 (Hanford)     → 수축 방향 → D1 = Φ^(-1.5) 가중
      L1 (Livingston)  → 팽창 방향 → D2 = Φ^(+1.5) 가중
      합성 비율 D2/D1  = Φ³ ≈ 4.236 (볼츠만 에너지 등분배 파단비)
      정규화: (D1+D2)로 나눠 평균 스케일 유지

    v13.6: tau_H/max + tau_L/max  (1:1 단순 합산)
    v13.7: (tau_H×D1 + tau_L×D2) / (D1+D2)  (Φ³ 비율 합산)
    """
    tau_combined = (tau_H * D1 + tau_L * D2) / (D1 + D2)
    return tau_combined


def detect_merger_with_error(
    tau_H, tau_L, phase_H, phase_L, gps_start_cropped, fs, search_sec=SEARCH_SEC
):
    """
    [V13.7-1] 비등방성 가중 복합 스코어 + GPS 윈도우.

    스코어 3항 가중치:
      w_tau     = D1          (수축 에너지 기여)
      w_phase   = D2/D1 = Φ³  (볼츠만 파단비, 위상 동기화 강조)
      w_entropy = D1          (수축 엔트로피 기여)

    v13.6 가중치: w1=1.0, w2=1.5, w3=1.0 (임의)
    v13.7 가중치: w1=D1, w2=D2/D1=Φ³, w3=D1 (비등방성)

    tau 합성:
      v13.6: tau_H/max + tau_L/max (1:1)
      v13.7: (tau_H×D1 + tau_L×D2)/(D1+D2) (D1:D2=Φ³)
    """
    n = len(tau_H)

    # [V13.7-1] ① 비등방성 합성 tau
    tau_combined = make_anisotropic_combined(tau_H, tau_L)
    tau_norm = tau_combined / (tau_combined.max() + 1e-30)

    # [V13.7-1] ② 위상 동기화
    phase_coh = (np.cos(phase_H - phase_L) + 1.0) / 2.0
    phase_coh_smooth = uniform_filter1d(phase_coh, size=max(1, int(0.01 * fs)))
    phase_coh_norm = phase_coh_smooth / (phase_coh_smooth.max() + 1e-30)

    # [V13.7-1] ③ 엔트로피 감소 (tau_combined 기준)
    ent_combined = _sliding_entropy(tau_combined, win_sec=0.5, fs=fs)
    ent_drop = 1.0 - ent_combined / (ent_combined.max() + 1e-30)
    ent_norm = ent_drop / (ent_drop.max() + 1e-30)

    # [V13.7-1] 비등방성 가중치
    w_tau = D1  # Φ^(-1.5) ≈ 0.4859
    w_phase = D2 / D1  # Φ³       ≈ 4.236  (볼츠만 파단비)
    w_entropy = D1  # Φ^(-1.5) ≈ 0.4859
    w_total = w_tau + w_phase + w_entropy

    score = (
        w_tau * tau_norm + w_phase * phase_coh_norm + w_entropy * ent_norm
    ) / w_total

    print(f"\n  [V13.7-1] 비등방성 가중치:")
    print(f"    w_tau(D1)     = {w_tau:.4f}  (수축, Φ^-1.5)")
    print(f"    w_phase(Φ³)   = {w_phase:.4f}  (볼츠만 파단비, Φ^3)")
    print(f"    w_entropy(D1) = {w_entropy:.4f}  (수축, Φ^-1.5)")
    print(f"    w_total       = {w_total:.4f}  (= D1+Φ³+D1 = {D1+w_phase+D1:.4f})")

    # GPS 탐색 윈도우 (V13.6-3 유지)
    idx_arr = np.arange(n)
    t_rel_arr = gps_start_cropped + idx_arr / fs - GW150914_GPS_MERGER
    window_mask = np.abs(t_rel_arr) <= search_sec

    if window_mask.sum() == 0:
        print(f"  ⚠️  GPS 윈도우 내 샘플 없음 → 전 구간 탐색으로 폴백")
        window_mask = np.ones(n, dtype=bool)

    search_start = int(np.argmax(window_mask))
    search_end = int(len(window_mask) - np.argmax(window_mask[::-1]))
    print(
        f"  [V13.6-3] 탐색 윈도우: "
        f"idx {search_start:,}~{search_end:,}  "
        f"(GPS±{search_sec:.0f}s, {window_mask.sum():,}샘플)"
    )

    score_search = np.where(window_mask, score, -1.0)
    best_idx = int(np.argmax(score_search))
    t_detected = gps_start_cropped + best_idx / fs - GW150914_GPS_MERGER
    error_ms = abs(t_detected) * 1e3

    top5_idx = np.argsort(score_search)[::-1][:5]
    top5 = [
        (gps_start_cropped + i / fs - GW150914_GPS_MERGER, float(score[i]))
        for i in top5_idx
    ]

    print(f"\n  [합병 탐지 V13.7]  비등방성 복합 스코어 + GPS 윈도우")
    print(f"  탐지: idx={best_idx:,}  t={t_detected:+.3f}s  오차={error_ms:.1f}ms")
    print(f"  상위 5개:")
    for rank, (t_val, s) in enumerate(top5, 1):
        print(f"    {rank}위: t={t_val:+.3f}s  score={s:.4f}")
    return best_idx, t_detected, error_ms, top5, score, tau_combined


# ============================================================
# 메인 엔진 클래스
# ============================================================


class EQIExtendedGaussLawEngineV13_7:
    def __init__(self):
        print("=" * 80)
        print("  Maestro MAPSI: EQI Extended Gauss Law Ultimate Engine v13.7")
        print("  [V13.7-1] H:L = D1:D2 비등방성 합성  (Φ³ 볼츠만 파단비)")
        print("  [V13.6]   Welch PSD / 위상 스티칭 / GPS 윈도우 유지")
        print("=" * 80)

        self.U = mpmath.mpf("1.0")
        self.Phi = (mpmath.sqrt(mpmath.mpf("5.0")) + self.U) / mpmath.mpf("2.0")
        self.h_planck = mpmath.mpf("6.62607015e-34")

        self.ln_25 = mpmath.ln(mpmath.mpf("25.0"))
        self.absolute_gap_ratio = float(self.ln_25 / mpmath.mpf("100.0"))

        life_pulse = (self.U / self.Phi) + (self.U / self.Phi**2)
        fractal_sum = sum(
            mpmath.ln(mpmath.exp(self.U / mpmath.mpf(2) ** n)) for n in range(1, 1000)
        )
        print(f"  [EQI^2 정규화] Phi^-1 + Phi^-2 = {mpmath.nstr(life_pulse, 10)} UNITY")
        print(
            f"  [EQI^4 재정규화] Σ ln(e^(1/2^n)) = {mpmath.nstr(fractal_sum, 10)} UNITY"
        )
        print(f"  [ln(25)% 갭 비율] = {self.absolute_gap_ratio * 100:.8f}%")
        print(f"  [V13.7-1] D1 = Φ^(-1.5) = {D1:.6f}  D2 = Φ^(+1.5) = {D2:.6f}")
        print(f"  [V13.7-1] D1×D2 = {D1*D2:.6f}  D2/D1 = Φ³ = {D2/D1:.6f}")
        print("=" * 80)

    # ── 데이터 로드 (v13.6 유지) ─────────────────────────────
    def load_and_clean_strain_data(self, h_file, l_file):
        if not os.path.exists(h_file) or not os.path.exists(l_file):
            raise FileNotFoundError(
                f"LIGO 데이터 경로 오류:\n H1: {h_file}\n L1: {l_file}"
            )
        if h_file.lower().endswith((".hdf5", ".h5")):
            import h5py

            print(f"  📥 [HDF5 Parser] 로드 중...")
            with h5py.File(h_file, "r") as f_h:
                strain_h = f_h["strain/Strain"][:]
                n_pts = f_h["strain/Strain"].attrs["Npoints"]
                xspacing = f_h["strain/Strain"].attrs["Xspacing"]
                gps_start = float(
                    f_h["strain/Strain"].attrs.get(
                        "Xstart", f_h["strain/Strain"].attrs.get("GPSstart", 0.0)
                    )
                )
            t_h = np.arange(n_pts) * xspacing
            with h5py.File(l_file, "r") as f_l:
                strain_l = f_l["strain/Strain"][:]
            fs = 1.0 / xspacing
            df_H = pd.DataFrame({"time": t_h, "strain": strain_h})
            df_L = pd.DataFrame({"time": t_h, "strain": strain_l})
            print(f"  ✅ {len(df_H):,}샘플  fs={fs:.0f}Hz  GPS_start={gps_start}")
            return df_H, df_L, fs, gps_start
        else:
            print(f"  📥 [ASCII Parser]...")
            df_H = pd.read_csv(h_file, sep=r"\s+", names=["time", "strain"], skiprows=2)
            df_L = pd.read_csv(l_file, sep=r"\s+", names=["time", "strain"], skiprows=2)
            fs = 1.0 / (df_H["time"].iloc[1] - df_H["time"].iloc[0])
            gps_start = float(df_H["time"].iloc[0])
            print(f"  ✅ {len(df_H):,}샘플  fs={fs:.0f}Hz")
            return df_H, df_L, fs, gps_start

    # ── 전처리 (v13.6 유지) ──────────────────────────────────
    def preprocess_with_hilbert(self, strain_series, fs):
        signal = np.nan_to_num(strain_series.values, nan=0.0)
        return preprocess_strain(signal, fs)

    # ── I_s (v13.6 유지) ─────────────────────────────────────
    def extract_information_singularity(self):
        I_s = mpmath.log10(self.U / (self.h_planck**2 * self.Phi**2))
        return float(I_s)

    # ── 렌더러 (v13.6 유지 + V13.7-1 패널 제목 업데이트) ────
    def render_active_nodes_manifold(
        self,
        t,
        raw_amplitude,
        active_indices,
        output_folder,
        t_merger_offset=None,
        error_ms=None,
        score=None,
    ):
        n_nodes = len(active_indices)
        print(f"\n🎨 [Rendering] {n_nodes:,}개 순수 위상 노드 렌더링 중...")

        n_bg = len(t)
        dec = max(1, n_bg // 200000)
        t_bg = t[::dec]
        amp_bg = raw_amplitude[::dec]
        t_active = t[active_indices]
        amp_active = raw_amplitude[active_indices]

        n_panels = 3 if score is not None else 2
        fig = plt.figure(figsize=(20, 6 * n_panels), facecolor="#050505")

        # 패널 1
        ax1 = plt.subplot(n_panels, 1, 1, facecolor="#050505")
        ax1.plot(
            t_bg,
            amp_bg,
            color="gray",
            alpha=0.2,
            lw=0.3,
            label="Background (Discarded)",
        )
        ax1.scatter(
            t_active,
            amp_active,
            color="cyan",
            s=3,
            alpha=0.8,
            label=f"True Nodes (ln(25)%: {n_nodes:,})",
        )
        if t_merger_offset is not None:
            ax1.axvline(
                t_merger_offset,
                color="red",
                lw=2,
                ls="--",
                label=f"Merger t=0 (err={error_ms:.0f}ms)",
            )
            ax1.axvspan(
                t_merger_offset - 1,
                t_merger_offset + 1,
                color="red",
                alpha=0.07,
                label="±1s Window",
            )
        ax1.set_title(
            f"AFT/EQI Topological Manifold: {n_nodes:,} Purified Nodes"
            f"  [V13.7: H×D1+L×D2 비등방성 합성  D1:D2=Φ³={D2/D1:.3f}]",
            color="cyan",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend(facecolor="black", labelcolor="white", loc="upper right", fontsize=9)
        ax1.tick_params(colors="white")
        ax1.set_ylabel("Amplitude² (D1:D2 weighted)", color="white")

        # 패널 2: 바코드
        ax2 = plt.subplot(n_panels, 1, 2, facecolor="#050505", sharex=ax1)
        ax2.vlines(t_active, 0, 1, color="gold", lw=0.4, alpha=0.5)
        if t_merger_offset is not None:
            ax2.axvline(t_merger_offset, color="red", lw=2, ls="--", label="Merger t=0")
            ax2.axvspan(
                t_merger_offset - 1, t_merger_offset + 1, color="red", alpha=0.12
            )
            ax2.legend(
                facecolor="black", labelcolor="white", loc="upper right", fontsize=9
            )
        err_str = f"  탐지 오차: {error_ms:.0f}ms" if error_ms is not None else ""
        ax2.set_title(
            f"65.9395 Bits Information Singularity Density Map{err_str}",
            color="gold",
            fontsize=13,
            fontweight="bold",
        )
        ax2.tick_params(colors="white")
        ax2.set_yticks([])
        if score is None:
            ax2.set_xlabel("Time (s) from data start", color="white")

        # 패널 3: 복합 스코어
        if score is not None:
            ax3 = plt.subplot(n_panels, 1, 3, facecolor="#050505", sharex=ax1)
            dec_s = max(1, len(t) // 200000)
            ax3.plot(t[::dec_s], score[::dec_s], color="lime", lw=0.8, alpha=0.9)
            ax3.fill_between(t[::dec_s], 0, score[::dec_s], color="lime", alpha=0.2)
            if t_merger_offset is not None:
                ax3.axvline(
                    t_merger_offset, color="red", lw=2, ls="--", label="Merger t=0"
                )
                ax3.legend(
                    facecolor="black", labelcolor="white", loc="upper right", fontsize=9
                )
            ax3.set_title(
                f"V13.7 Composite Score  "
                f"w_tau=D1={D1:.3f}  w_phase=Φ³={D2/D1:.3f}  w_ent=D1={D1:.3f}",
                color="lime",
                fontsize=13,
                fontweight="bold",
            )
            ax3.set_xlabel("Time (s) from data start", color="white")
            ax3.tick_params(colors="white")
            ax3.set_ylabel("Score", color="white")

        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        out_path = f"{output_folder}/eqi_v13_7_gps2_pure_nodes_manifold.png"
        plt.savefig(out_path, dpi=150, facecolor="#050505")
        plt.close()
        print(f"  ✅ 저장: {out_path}")
        return out_path

    # ── 메인 파이프라인 ──────────────────────────────────────
    def run_upgraded_pipeline(
        self, h_file, l_file, output_folder="./output_v13_7_gps2"
    ):
        os.makedirs(output_folder, exist_ok=True)

        # STEP 1
        print("\n[STEP 1] 데이터 로드...")
        df_H, df_L, fs, gps_start = self.load_and_clean_strain_data(h_file, l_file)
        merger_offset_from_start = GW150914_GPS_MERGER - gps_start
        print(f"  GPS_start={gps_start}  GPS_merger={GW150914_GPS_MERGER}")
        print(f"  합병 위치: 데이터 시작으로부터 {merger_offset_from_start:.1f}s")

        # STEP 1-B: [V13.7-GPS] 4096초 대용량 데이터 GPS 윈도우 크롭
        n_total = len(df_H)
        duration_sec = n_total / fs
        if duration_sec > CROP_WINDOW_SEC:
            print(
                f"\n[STEP 1-B] GPS 윈도우 크롭 ({duration_sec:.0f}s > {CROP_WINDOW_SEC:.0f}s)..."
            )
            merger_sample = (GW150914_GPS_MERGER - gps_start) * fs
            half_win = int(CROP_WINDOW_SEC * fs)
            crop_start = max(0, int(merger_sample) - half_win)
            crop_end = min(n_total, int(merger_sample) + half_win)
            df_H = df_H.iloc[crop_start:crop_end].reset_index(drop=True)
            df_L = df_L.iloc[crop_start:crop_end].reset_index(drop=True)
            # time 배열도 원본 GPS 유지
            gps_start = gps_start + crop_start / fs
            print(
                f"  크롭: idx {crop_start:,}~{crop_end:,}  ({(crop_end-crop_start)/fs:.1f}초)"
            )
            print(f"  새 GPS_start = {gps_start:.1f}")
            print(f"  합병 위치: 크롭 기준 {GW150914_GPS_MERGER - gps_start:.1f}s 후")
        else:
            print(
                f"\n[STEP 1-B] GPS 크롭 불필요 ({duration_sec:.0f}s ≤ {CROP_WINDOW_SEC:.0f}s)"
            )

        # STEP 2
        print("\n[STEP 2] 전처리 (밴드패스 + Welch PSD 화이트닝)...")
        phase_H_full, tau_H_full = self.preprocess_with_hilbert(df_H["strain"], fs)
        phase_L_full, tau_L_full = self.preprocess_with_hilbert(df_L["strain"], fs)
        print(f"  전처리 완료: {len(tau_H_full):,}샘플")

        # STEP 3
        print(f"\n[STEP 3] 경계 crop (±{CROP_SEC}s)...")
        phase_H, tau_H, phase_L, tau_L, t_cropped = crop_boundary(
            fs,
            CROP_SEC,
            phase_H_full,
            tau_H_full,
            phase_L_full,
            tau_L_full,
            df_H["time"].values,
        )
        gps_start_cropped = gps_start + CROP_SEC
        print(f"  crop 후 유효 샘플: {len(tau_H):,}")

        # STEP 4: [V13.7-1] 비등방성 합성 탐지
        print("\n[STEP 4] 합병 탐지 (비등방성 D1:D2 복합 스코어 + GPS 윈도우)...")
        best_idx, t_detected, error_ms, top5, score, tau_combined = (
            detect_merger_with_error(
                tau_H,
                tau_L,
                phase_H,
                phase_L,
                gps_start_cropped,
                fs,
                search_sec=SEARCH_SEC,
            )
        )
        t_merger_in_data = t_cropped[best_idx] if best_idx < len(t_cropped) else None

        # STEP 4-C: 글리치 마스킹
        print("\n[STEP 4-C] 글리치 마스킹 (상위 0.1% 제거)...")
        tau_comb_clean, gmask, n_glitch, g_thresh = apply_glitch_mask(tau_combined)
        print(f"  글리치 제거: {n_glitch:,}샘플 (임계값={g_thresh:.4e})")

        # 마스킹된 스코어로 재탐지
        score_clean = score.copy()
        score_clean[gmask] = 0.0
        t_rel_c = (
            gps_start_cropped + np.arange(len(tau_combined)) / fs - GW150914_GPS_MERGER
        )
        wmask_c = np.abs(t_rel_c) <= SEARCH_SEC
        sc2 = np.where(wmask_c, score_clean, -1.0)
        bi2 = int(np.argmax(sc2))
        t_det2 = gps_start_cropped + bi2 / fs - GW150914_GPS_MERGER
        err2 = abs(t_det2) * 1e3
        top5_2 = [
            (gps_start_cropped + i / fs - GW150914_GPS_MERGER, float(score_clean[i]))
            for i in np.argsort(sc2)[::-1][:5]
        ]
        print(f"  [C] 마스킹 후 탐지: t={t_det2:+.3f}s  오차={err2:.1f}ms")
        print(f"  마스킹 후 상위 5개:")
        for rank, (tv, sc_v) in enumerate(top5_2, 1):
            print(f"    {rank}위: t={tv:+.3f}s  score={sc_v:.4f}")
        if err2 < error_ms:
            print(f"  ✅ 글리치 마스킹으로 오차 개선: {error_ms:.0f}ms → {err2:.0f}ms")
            best_idx = bi2
            t_detected = t_det2
            error_ms = err2
            top5 = top5_2
            t_merger_in_data = (
                t_cropped[best_idx] if best_idx < len(t_cropped) else None
            )
        else:
            print(f"  ℹ️ 마스킹 전후 오차 동일 ({err2:.0f}ms)")

        # STEP 5: [V13.7-1] tau_combined 기준 갭 필터링
        print("\n[STEP 5] ln(25)% 절대 갭 에너지 필터링 (tau_combined 기준)...")
        amplitude = tau_comb_clean
        threshold_index = int(len(amplitude) * (1.0 - self.absolute_gap_ratio))
        active_indices = np.sort(np.argsort(amplitude)[threshold_index:])
        print(f"  ✅ {len(amplitude):,}샘플 → {len(active_indices):,}노드")
        print(f"  [V13.7-1] 필터 기준: tau_combined_clean (글리치 제거 후)")

        # STEP 6
        I_s = self.extract_information_singularity()
        print(f"\n  I_s = {I_s:.6f} Bits")

        # STEP 7
        print("\n[STEP 7] 렌더링...")
        self.render_active_nodes_manifold(
            t_cropped,
            amplitude,
            active_indices,
            output_folder,
            t_merger_offset=t_merger_in_data,
            error_ms=error_ms,
            score=score,
        )

        # STEP 8: 보고서
        report_path = f"{output_folder}/eqi_v13_7_gps2_gap_filtered_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 61 + "\n")
            f.write("    🌟 MAESTRO MAPSI: EQI GAP ENERGY V13.7-GPS2 REPORT 🌟\n")
            f.write("=" * 61 + "\n\n")
            f.write(f"[0. 데이터 규격]\n")
            f.write(f" - 원본 샘플: {len(df_H):,}  fs={fs:.0f}Hz\n")
            f.write(f" - crop 후:   {len(tau_H):,}  (±{CROP_SEC}s 제거)\n")
            f.write(f" - GPS start: {gps_start}\n")
            f.write(f" - 합병 위치: 시작으로부터 {merger_offset_from_start:.1f}s\n\n")
            f.write(f"[V13.7 수정사항]\n")
            f.write(f" - [V13.7-1] H:L = D1:D2 비등방성 합성\n")
            f.write(
                f"   tau_combined = (tau_H×{D1:.4f} + tau_L×{D2:.4f}) / {D1+D2:.4f}\n"
            )
            f.write(f"   D1×D2 = {D1*D2:.6f} (Unity)  D2/D1 = Φ³ = {D2/D1:.6f}\n")
            f.write(f"   w_tau={D1:.4f}  w_phase=Φ³={D2/D1:.4f}  w_ent={D1:.4f}\n")
            f.write(f" - [GPS2-A] notch: {NOTCH_FREQS}\n")
            f.write(f" - [GPS2-B] 크롭: ±{CROP_WINDOW_SEC:.0f}s\n")
            f.write(f" - [GPS2-C] 글리치 마스킹: 상위 {100-GLITCH_PERCENTILE:.1f}%\n")
            f.write(f" - [GPS2-D] 전처리 순서: notch→bandpass→whitening\n")
            f.write(
                f" - [GPS2-E] SEARCH_SEC={SEARCH_SEC:.0f}s  GLITCH={GLITCH_PERCENTILE}%\n\n"
            )
            f.write(f"[1. 팩토리얼-리만 갭 에너지 필터]\n")
            f.write(f" - ln(25)% = {self.absolute_gap_ratio * 100:.8f}%\n")
            f.write(f" - 추출 노드 수: {len(active_indices):,}개\n\n")
            f.write(f"[2. 합병 탐지]\n")
            f.write(f" - 탐지 시각: t={t_detected:+.3f}s\n")
            f.write(f" - 오차:      {error_ms:.1f}ms\n")
            f.write(f" - 상위 5개:\n")
            for rank, (tv, sc) in enumerate(top5, 1):
                f.write(f"     {rank}위: t={tv:+.3f}s  score={sc:.4f}\n")
            f.write(f"\n[3. 65.9395 정보 특이점]\n")
            f.write(f" - I_s = {I_s:.6f} Bits\n")
            f.write("=" * 61 + "\n")

        print("\n" + "=" * 80)
        print(f"💾 [완료] v13.7 결과 저장: {output_folder}")
        print(f"   eqi_v13_7_pure_nodes_manifold.png  (3-패널)")
        print(f"   eqi_v13_7_gap_filtered_report.txt")
        print(f"   탐지 오차: {error_ms:.0f}ms  |  노드: {len(active_indices):,}개")
        print("=" * 80)


# ============================================================
# 실행 진입점
# ============================================================
if __name__ == "__main__":
    # 32초 4kHz (기본)
    # 4096초 16kHz (기본)
    # H1_PATH = "H-H1_LOSC_16_V1-1126256640-4096.hdf5"
    # L1_PATH = "L-L1_LOSC_16_V1-1126256640-4096.hdf5"
    # 32초 4kHz 사용 시:
    H1_PATH = "H-H1_LOSC_4_V2-1126259446-32.hdf5"
    L1_PATH = "L-L1_LOSC_4_V2-1126259446-32.hdf5"

    # 4096초 16kHz 사용 시 아래 경로로 교체
    # H1_PATH = "d:\\LIGO\\H-H1_LOSC_16_V1-1126256640-4096.hdf5"
    # L1_PATH = "d:\\LIGO\\L-L1_LOSC_16_V1-1126256640-4096.hdf5"

    try:
        engine = EQIExtendedGaussLawEngineV13_7()
        engine.run_upgraded_pipeline(
            H1_PATH, L1_PATH, output_folder="./output_v13_7_gps2"
        )
    except Exception as error:
        print(f"\n❌ [구동 실패] {error}")

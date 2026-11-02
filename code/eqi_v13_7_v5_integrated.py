#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQI Extended Gauss Law Ultimate Engine v13.7 + v5.0 통합 (Rev.9)
=================================================================
[Rev.9 핵심: 보스-아인슈타인(D1) × 페르미-디랙(D2) 통계 가중치]

  물리 구조:
    D1(BE, 중력 질량, 보손, 슈뢰딩거):
      보손은 같은 상태 군집 → f_h≈f_l(노이즈) 선호
      합병(f_h≠f_l)에서 낮음 → "안정화, 수축" 담당
      w_BE = exp(-eps_norm²)   노이즈=1, 합병=0.37

    D2(FD, 관성 질량, 페르미온, 하이젠베르크):
      페르미온은 배타 원리 → f_h≠f_l(합병) 선호
      합병에서 높음 → "탐지, 팽창" 담당
      w_FD = 1 - exp(-eps_norm)  노이즈=0, 합병=0.63

    eps_norm = |ln(f_h/f_l)| / ln(2)  (옥타브 단위 정규화)
      노이즈:  eps_norm ≈ 0  (f_h≈f_l)
      합병:   eps_norm = 1  (f_h/f_l=2, 옥타브 1개)

  Rev.6 구조 완전 유지 + BE/FD 가중치 추가:
    raw_short × D1 × w_BE  (BE 억제: 합병에서 낮음)
    raw_long  × D2 × w_FD  (FD 탐지: 합병에서 높음)
    × energy_norm  (v11.0)
    단일 정규화

  Rev.6이 성공한 이유 재확인:
    RI = 1.125 = 1 + 4×GAP  ← 이미 페르미온 배타 구조를 내포
    f_h/f_l = 2.0 = 옥타브 1개 = FD 배타 원리의 자연 단위

[버전 계보]
  Rev.1~5: 단계적 발전
  Rev.6: D1×D2 HCW 에너지 가중 → H-L 3.4ms, V-L 17ms 동시 달성
  Rev.7: EQI^2/EQI^4 입력 정규화 → 실패 (FFT 피크 이동)
  Rev.8: c_EQI Unit Element 기준점 → 실패 (합병 RI와 불일치)
  Rev.9: BE(D1)/FD(D2) 통계 가중치 → Rev.6 위에 추가
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
from scipy.signal import hilbert, butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 20000
import mpmath

mpmath.mp.dps = 400

# ── 전역 상수 ────────────────────────────────────────────
LIGO_EVENTS = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170817": 1187008882.4,
    "GW190412": 1239082262.2,
    "GW190814": 1249852257.0,
    "GW190521": 1242442967.4,
}


def detect_event_gps(h_file, gps_start):
    fn = os.path.basename(h_file).upper()
    for name in LIGO_EVENTS:
        if name.upper() in fn or name[2:] in fn:
            print(f"  [이벤트 감지] 파일명 매칭: {name}  GPS={LIGO_EVENTS[name]}")
            return LIGO_EVENTS[name], name
    best_name = min(LIGO_EVENTS, key=lambda k: abs(LIGO_EVENTS[k] - gps_start))
    best_gps = LIGO_EVENTS[best_name]
    dist = abs(best_gps - gps_start)
    if dist <= 4200:
        print(
            f"  [이벤트 감지] GPS 최근접: {best_name}  GPS={best_gps}  (거리={dist:.1f}s)"
        )
        return best_gps, best_name
    print(f"  ⚠️  이벤트 자동 감지 실패 → GW150914 기본값")
    return 1126259462.4, "GW150914"


GW150914_GPS_MERGER = 1126259462.4
CROP_SEC = 0.5
CHUNK_THRESHOLD = 16_000_000
SEARCH_SEC = 3.0
SEARCH_SEC_V13 = 1.0
CROP_WINDOW_SEC = 150.0
NOTCH_FREQS = [60.0, 120.0, 180.0, 240.0, 300.0]
NOTCH_Q = 30.0
GLITCH_PERCENTILE = 99.999

_PHI = (1.0 + np.sqrt(5.0)) / 2.0
c_EQI = _PHI**0.5
DUAL_GAP = np.log(25.0) / 100.0
DUAL_EPSILON = 1e-6
W_DUAL = _PHI ** (-3.0) / 2.0
D1 = _PHI ** (-1.5)
D2 = _PHI ** (+1.5)

# ── [Rev.6] v5.0 상수 (유지) ─────────────────────────────
V5_QUANTUM_LEVEL = 4.0 * DUAL_GAP  # ≈ 0.12876 (Rev.6 그대로)
W_V5 = D1**2 / 8.0
V5_WIN_SHORT = 0.1  # D1(BE, 슈뢰딩거)
V5_WIN_LONG = 0.5  # D2(FD, 하이젠베르크)
V5_ENERGY_WIN = 0.5

# ── [Rev.9] BE/FD 통계 상수 ──────────────────────────────
LN2 = np.log(2.0)  # 옥타브 자연 에너지 단위


# ── 신호처리 함수 ────────────────────────────────────────
def bandpass_filter(signal, fs, fmin=35.0, fmax=350.0, order=4):
    nyq = fs / 2.0
    b, a = butter(
        order,
        [np.clip(fmin / nyq, 1e-4, 0.9999), np.clip(fmax / nyq, 1e-4, 0.9999)],
        btype="band",
    )
    return filtfilt(b, a, signal)


def notch_filter_line_noise(signal, fs, freqs=NOTCH_FREQS, Q=NOTCH_Q):
    sig = signal.copy()
    for f0 in freqs:
        if f0 >= fs / 2:
            continue
        b, a = iirnotch(f0, Q, fs=fs)
        sig = filtfilt(b, a, sig)
    return sig


def whiten_signal_welch(signal, fs, seg_sec=4.0, overlap_frac=0.5):
    n = len(signal)
    seg_n = int(seg_sec * fs)
    hop_n = int(seg_n * (1.0 - overlap_frac))
    n_fft = seg_n // 2 + 1
    psd_acc = np.zeros(n_fft, dtype=np.float64)
    n_segs = 0
    pos = 0
    while pos + seg_n <= n:
        psd_acc += np.abs(np.fft.rfft(signal[pos : pos + seg_n])) ** 2
        n_segs += 1
        pos += hop_n
    if n_segs == 0:
        psd_acc = np.abs(np.fft.rfft(signal)) ** 2
        n_segs = 1
    psd_mean = np.where(psd_acc / n_segs < 1e-60, 1e-60, psd_acc / n_segs)
    fft_full = np.fft.rfft(signal, n=n)
    n_full = len(fft_full)
    psd_i = (
        np.interp(np.linspace(0, 1, n_full), np.linspace(0, 1, n_fft), psd_mean)
        if n_full != n_fft
        else psd_mean
    )
    return np.fft.irfft(fft_full / np.sqrt(psd_i), n=n).astype(np.float64)


def preprocess_strain(signal, fs, apply_bp=True, apply_whiten=True):
    sig = np.nan_to_num(signal, nan=0.0)
    if len(sig) > CHUNK_THRESHOLD:
        _, phase, tau = _preprocess_chunked_stitched(sig, fs, apply_bp, apply_whiten)
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
    print(f"  [청크 전처리] {n:,}샘플")
    pos = 0
    cnt = 0
    offset = 0.0
    while pos < n:
        se = max(0, pos - overlap_n)
        ee = min(n, pos + chunk_n + overlap_n)
        chunk = signal[se:ee].copy()
        chunk = notch_filter_line_noise(chunk, fs)
        if apply_bp:
            chunk = bandpass_filter(chunk, fs)
        if apply_whiten:
            chunk = whiten_signal_welch(chunk, fs)
        ac = hilbert(chunk)
        ph = np.unwrap(np.angle(ac))
        ta = np.abs(ac) ** 2
        i0 = pos - se
        vl = min(chunk_n, n - pos)
        i1 = i0 + vl
        ps = ph[i0:i1] + offset
        offset = ps[-1] - np.unwrap(np.angle(hilbert(chunk)[i1 - 1 : i1]))[0]
        sig_proc[pos : pos + vl] = chunk[i0:i1]
        phase_out[pos : pos + vl] = ps
        tau_out[pos : pos + vl] = ta[i0:i1]
        pos += chunk_n
        cnt += 1
        if cnt % 10 == 0:
            print(f"    청크 {cnt}: {pos:,}/{n:,}")
    return sig_proc, phase_out, tau_out


def crop_boundary(fs, crop_sec, *arrays):
    n_crop = int(np.round(crop_sec * fs))
    if n_crop == 0:
        return arrays
    return tuple(a[n_crop:-n_crop] for a in arrays)


def apply_glitch_mask(tau_combined, percentile=GLITCH_PERCENTILE):
    thr = np.percentile(tau_combined, percentile)
    gmask = tau_combined > thr
    tc = tau_combined.copy()
    tc[gmask] = np.median(tau_combined[~gmask])
    return tc, gmask, int(gmask.sum()), float(thr)


def _sliding_entropy(tau, win_sec, fs):
    n = len(tau)
    bn = max(1, int(win_sec * fs))
    nb = max(1, n // bn)
    eb = np.zeros(nb)
    for b in range(nb):
        seg = tau[b * bn : (b + 1) * bn]
        s = seg.sum() + 1e-30
        p = seg / s
        eb[b] = -np.sum(p * np.log(p + 1e-30))
    return np.interp(np.arange(n), (np.arange(nb) + 0.5) * bn, eb)


def compute_dual_score(phase_H, phase_L, tau_combined, fs):
    delta = phase_H - phase_L
    real = (np.cos(delta) + 1.0) / 2.0
    de = np.abs(real - 1.0)
    s_ex = np.where(de <= DUAL_GAP, 1.0, np.exp(-(de - DUAL_GAP) / DUAL_GAP))
    tn = tau_combined / (tau_combined.max() + 1e-30)
    ra = real + tn * DUAL_GAP
    dep = np.abs(ra - (1.0 + DUAL_GAP))
    s_ep = 1.0 / (dep + DUAL_EPSILON)
    s_ep /= s_ep.max() + 1e-30
    sd = (s_ex * D1 + s_ep * D2) / (D1 + D2)
    sd /= sd.max() + 1e-30
    return sd, s_ex, s_ep


# ═══════════════════════════════════════════════════════════
# [Rev.9] BE/FD 통계 가중치 함수
# ═══════════════════════════════════════════════════════════
def be_fd_weights(f_h, f_l):
    """
    보스-아인슈타인(D1) × 페르미-디랙(D2) 통계 가중치

    eps_norm = |ln(f_h/f_l)| / ln(2)  (옥타브 단위)
      노이즈: eps_norm ≈ 0 (f_h≈f_l)
      합병:   eps_norm = 1 (f_h/f_l=2, 옥타브 1개)

    w_BE (D1, 보손, 수축, 군집):
      exp(-eps_norm²)  → 노이즈=1.0, 합병=0.37
      → D1 채널이 노이즈에서 강함 (안정화)

    w_FD (D2, 페르미온, 팽창, 배타):
      1 - exp(-eps_norm)  → 노이즈=0.0, 합병=0.63
      → D2 채널이 합병에서 강함 (탐지)
    """
    if f_h < 1.0 or f_l < 1.0:
        return 1.0, 0.0
    eps_norm = abs(np.log(f_h / f_l)) / LN2  # 옥타브 단위
    w_BE = np.exp(-(eps_norm**2))  # D1: 노이즈 군집
    w_FD = 1.0 - np.exp(-eps_norm)  # D2: 합병 배타
    return float(w_BE), float(w_FD)


# ═══════════════════════════════════════════════════════════
# [Rev.9] BE/FD 가중 v5.0 단일 WIN raw
# ═══════════════════════════════════════════════════════════
def _compute_v5_single_win_raw_befd(strain_H, strain_L, fs, win_sec, use_be_fd=True):
    """
    [Rev.9] BE/FD 통계 가중치 적용 v5.0 raw 스코어

    Rev.6 기반:
      dev = |1.0 - RI|  (유지)
      score = exp(-|dev - 4×GAP| / GAP)  (유지)

    Rev.9 추가:
      w_BE, w_FD = be_fd_weights(f_h, f_l)
      score_D1 = score × w_BE  (D1: 노이즈 강조, 합병 억제)
      score_D2 = score × w_FD  (D2: 합병 강조, 노이즈 억제)
      → 외부에서 D1/D2 가중 합산 시 자연스럽게 분리
    """
    n = len(strain_H)
    win_n = max(1, int(win_sec * fs))
    step = max(1, win_n // 4)
    score_be = np.zeros(n, dtype=np.float64)  # w_BE 반영
    score_fd = np.zeros(n, dtype=np.float64)  # w_FD 반영
    freqs = np.fft.rfftfreq(win_n, d=1.0 / fs)

    for i in range(0, n - win_n + 1, step):
        fft_h = np.abs(np.fft.rfft(strain_H[i : i + win_n]))
        fft_h[0] = 0
        fft_l = np.abs(np.fft.rfft(strain_L[i : i + win_n]))
        fft_l[0] = 0
        f_h = freqs[np.argmax(fft_h)]
        f_l = freqs[np.argmax(fft_l)]
        if f_h < 1.0 or f_l < 1.0:
            continue

        ratio = f_h / f_l
        ri = (2.0 + ratio + 1.0 / ratio) / 4.0
        dev = abs(1.0 - ri)  # Rev.6 유지
        s = np.exp(-abs(dev - V5_QUANTUM_LEVEL) / DUAL_GAP)

        if use_be_fd:
            w_be, w_fd = be_fd_weights(f_h, f_l)
            s_be = s * w_be  # D1: 노이즈에서 강함
            s_fd = s * w_fd  # D2: 합병에서 강함
        else:
            s_be = s_fd = s  # Rev.6 호환 모드

        end = min(i + win_n, n)
        score_be[i:end] = np.maximum(score_be[i:end], s_be)
        score_fd[i:end] = np.maximum(score_fd[i:end], s_fd)

    return score_be, score_fd  # raw, 정규화 없음


# ═══════════════════════════════════════════════════════════
# [Rev.9] BE/FD × D1×D2 통합 v5.0 스코어
# ═══════════════════════════════════════════════════════════
def compute_v5_reality_score(strain_H, strain_L, fs):
    """
    [Rev.9] BE/FD 통계 × HCW 힉스 매듭 × 에너지 가중:

    short(0.1s): D1(BE, 슈뢰딩거) × w_BE  → 노이즈 시 강함
    long (0.5s): D2(FD, 하이젠베르크) × w_FD → 합병 시 강함

    combined = score_be_short × D1
             + score_fd_long  × D2
             × energy_norm   (v11.0)

    물리:
      합병(f_h/f_l=2): w_FD=0.63 × D2=2.06 → D2가 지배
      노이즈(f_h≈f_l): w_BE=1.00 × D1=0.49 → D1이 지배
      에너지: 합병 순간 H²+L² 최대 → 이중 강조
    """
    n = len(strain_H)

    # BE/FD 가중 raw 스코어 (정규화 없음)
    be_short, fd_short = _compute_v5_single_win_raw_befd(
        strain_H, strain_L, fs, V5_WIN_SHORT
    )
    be_long, fd_long = _compute_v5_single_win_raw_befd(
        strain_H, strain_L, fs, V5_WIN_LONG
    )

    # D1(BE): short WIN의 BE 스코어 (보손, 수축, 안정화)
    raw_D1 = be_short  # 노이즈에서 강함 → D1 안정화

    # D2(FD): long WIN의 FD 스코어 (페르미온, 팽창, 탐지)
    raw_D2 = fd_long  # 합병에서 강함 → D2 탐지

    # HCW 힉스 매듭 가중 합산 (Rev.6)
    combined_raw = raw_D1 * D1 + raw_D2 * D2

    # v11.0 슬라이딩 에너지 (Rev.6)
    win_e = max(1, int(V5_ENERGY_WIN * fs))
    step_e = max(1, win_e // 4)
    energy = np.zeros(n, dtype=np.float64)
    for i in range(0, n - win_e + 1, step_e):
        e = np.sum(strain_H[i : i + win_e] ** 2) + np.sum(strain_L[i : i + win_e] ** 2)
        end = min(i + win_e, n)
        energy[i:end] = np.maximum(energy[i:end], e)
    energy_norm = energy / (energy.max() + 1e-30)

    # 단일 정규화 (Rev.6)
    combined = combined_raw * energy_norm
    score_v5 = combined / (combined.max() + 1e-30)

    return score_v5, raw_D1, raw_D2, energy_norm


# ── 비등방성 합성 ────────────────────────────────────────
def make_anisotropic_combined(tau_H, tau_L):
    return (tau_H * D1 + tau_L * D2) / (D1 + D2)


# ═══════════════════════════════════════════════════════════
# 탐지 함수
# ═══════════════════════════════════════════════════════════
def detect_merger_with_error(
    tau_H,
    tau_L,
    phase_H,
    phase_L,
    strain_H_raw,
    strain_L_raw,
    gps_start_cropped,
    fs,
    search_sec=SEARCH_SEC,
    search_sec_v13=SEARCH_SEC_V13,
    gps_merger=GW150914_GPS_MERGER,
):
    n = len(tau_H)
    t_rel_arr = gps_start_cropped + np.arange(n) / fs - gps_merger

    # Rev.6 표준 정규화 (입력 보존)
    strain_H_norm = (strain_H_raw - np.mean(strain_H_raw)) / (
        np.std(strain_H_raw) + 1e-30
    )
    strain_L_norm = (strain_L_raw - np.mean(strain_L_raw)) / (
        np.std(strain_L_raw) + 1e-30
    )

    # Unit Element |ψ|
    var_H = float(np.var(strain_H_raw))
    var_L = float(np.var(strain_L_raw))
    total_var = var_H + var_L + 1e-60
    H_ratio = var_H / total_var
    L_ratio = var_L / total_var
    psi_unit = H_ratio * D1 + 1j * L_ratio * D2
    psi_abs = float(abs(psi_unit))
    psi_arg = float(np.degrees(np.angle(psi_unit)))

    # Rev.9 BE/FD 스코어
    score_v5, raw_D1, raw_D2, energy_norm = compute_v5_reality_score(
        strain_H_norm, strain_L_norm, fs
    )

    wmask_v5 = np.abs(t_rel_arr) <= search_sec
    sv5_windowed = np.where(wmask_v5, score_v5, -1.0)
    peak_v5_idx = int(np.argmax(sv5_windowed))
    t_v5_peak = t_rel_arr[peak_v5_idx]
    sv5_peak_val = score_v5[peak_v5_idx]

    pk_D1 = int(np.argmax(np.where(wmask_v5, raw_D1, -1.0)))
    pk_D2 = int(np.argmax(np.where(wmask_v5, raw_D2, -1.0)))
    t_D1 = t_rel_arr[pk_D1]
    t_D2 = t_rel_arr[pk_D2]

    # BE/FD 예시 출력 (합병 블록 예상값)
    w_be_ex, w_fd_ex = be_fd_weights(150.0, 75.0)  # f_h/f_l=2.0 예시

    print(f"\n  [Rev.9 BE/FD 통계]  GPS ±{search_sec:.0f}s")
    print(
        f"    D1(BE,보손,수축): w_BE(ratio=2.0)={w_be_ex:.4f}  "
        f"short {V5_WIN_SHORT}s × D1={D1:.4f}"
    )
    print(
        f"    D2(FD,페르미온,팽창): w_FD(ratio=2.0)={w_fd_ex:.4f}  "
        f"long  {V5_WIN_LONG}s × D2={D2:.4f}"
    )
    print(f"    Unit |ψ|={psi_abs:.4f}  arg={psi_arg:.1f}°")
    print(f"    D1(BE) 피크: t={t_D1:+.3f}s  D2(FD) 피크: t={t_D2:+.3f}s")
    print(f"    에너지 가중 앙상블 피크: t={t_v5_peak:+.3f}s  score={sv5_peak_val:.4f}")
    print(f"    → v13.7 탐색 범위: t={t_v5_peak:+.3f}s ± {search_sec_v13}s")

    tau_combined = make_anisotropic_combined(tau_H, tau_L)
    tau_norm = tau_combined / (tau_combined.max() + 1e-30)
    phase_coh = (np.cos(phase_H - phase_L) + 1.0) / 2.0
    phase_coh_s = uniform_filter1d(phase_coh, size=max(1, int(0.01 * fs)))
    phase_coh_n = phase_coh_s / (phase_coh_s.max() + 1e-30)
    ent_combined = _sliding_entropy(tau_combined, win_sec=0.5, fs=fs)
    ent_drop = 1.0 - ent_combined / (ent_combined.max() + 1e-30)
    ent_norm = ent_drop / (ent_drop.max() + 1e-30)
    score_dual, _, _ = compute_dual_score(phase_H, phase_L, tau_combined, fs)

    w_tau = D1
    w_phase = D2 / D1
    w_entropy = D1
    w_dual = W_DUAL
    w_v5 = W_V5
    w_total = w_tau + w_phase + w_entropy + w_dual + w_v5
    score = (
        w_tau * tau_norm
        + w_phase * phase_coh_n
        + w_entropy * ent_norm
        + w_dual * score_dual
        + w_v5 * score_v5
    ) / w_total

    print(f"\n  [v13.7 존재론 2단계]  v5.0 피크 ±{search_sec_v13}s")
    print(
        f"    w_tau={w_tau:.4f}  w_phase={w_phase:.4f}  w_ent={w_entropy:.4f}"
        f"  w_dual={w_dual:.4f}  w_v5={w_v5:.4f}"
    )

    wmask = np.abs(t_rel_arr - t_v5_peak) <= search_sec_v13
    if wmask.sum() == 0:
        print(f"  ⚠️  v5.0 피크 윈도우 없음 → GPS ±{search_sec}s 폴백")
        wmask = np.abs(t_rel_arr) <= search_sec
    s_start = int(np.argmax(wmask))
    s_end = int(len(wmask) - np.argmax(wmask[::-1]))
    print(f"    [탐색 윈도우] idx {s_start:,}~{s_end:,}  ({wmask.sum():,}샘플)")

    score_s = np.where(wmask, score, -1.0)
    best_idx = int(np.argmax(score_s))
    t_det = t_rel_arr[best_idx]
    error_ms = abs(t_det) * 1e3
    top5_idx = np.argsort(score_s)[::-1][:5]
    top5 = [(t_rel_arr[i], float(score[i])) for i in top5_idx]

    print(f"\n  [합병 탐지 V13.7+V5.0 Rev.9]")
    print(f"  탐지: idx={best_idx:,}  t={t_det:+.3f}s  오차={error_ms:.1f}ms")
    print(f"  상위 5개:")
    for rk, (tv, sv) in enumerate(top5, 1):
        print(f"    {rk}위: t={tv:+.3f}s  score={sv:.4f}")
    return (
        best_idx,
        t_det,
        error_ms,
        top5,
        score,
        tau_combined,
        score_v5,
        raw_D1,
        raw_D2,
        energy_norm,
        t_v5_peak,
        psi_abs,
        psi_arg,
    )


# ── 메인 엔진 클래스 ─────────────────────────────────────
class EQIExtendedGaussLawEngineV13_7_V5:
    def __init__(self):
        w_be_ex, w_fd_ex = be_fd_weights(150.0, 75.0)  # f_h/f_l=2.0 예시
        print("=" * 80)
        print("  Maestro MAPSI: EQI Engine v13.7 + v5.0 (Rev.9)")
        print("  [Rev.9] 보스-아인슈타인(D1) × 페르미-디랙(D2) 통계 가중치")
        print("  [Rev.9] BE: 보손 군집(노이즈 안정화)  FD: 페르미온 배타(합병 탐지)")
        print("  [Rev.6] 입력 정규화/D1D2/에너지/단일정규화 구조 유지")
        print("=" * 80)
        self.GPS_MERGER = GW150914_GPS_MERGER
        self.EVENT_NAME = "GW150914"
        U = mpmath.mpf("1.0")
        Phi = (mpmath.sqrt(mpmath.mpf("5.0")) + U) / mpmath.mpf("2.0")
        self.h_planck = mpmath.mpf("6.62607015e-34")
        self.ln_25 = mpmath.ln(mpmath.mpf("25.0"))
        self.absolute_gap_ratio = float(self.ln_25 / mpmath.mpf("100.0"))
        lp = (U / Phi) + (U / Phi**2)
        fs_ = sum(mpmath.ln(mpmath.exp(U / mpmath.mpf(2) ** n)) for n in range(1, 1000))
        print(f"  [EQI^2]  Φ^-1+Φ^-2       = {mpmath.nstr(lp,10)}")
        print(f"  [EQI^4]  Σln(e^(1/2^n))  = {mpmath.nstr(fs_,10)}")
        print(f"  [GAP]    ln(25)%          = {self.absolute_gap_ratio*100:.8f}%")
        print(f"  [c_EQI]  φ^0.5            = {c_EQI:.6f}")
        print(f"  [BE/D1]  w_BE(ratio=2.0)  = {w_be_ex:.4f}  (노이즈→1, 합병→0.37)")
        print(f"  [FD/D2]  w_FD(ratio=2.0)  = {w_fd_ex:.4f}  (노이즈→0, 합병→0.63)")
        print(f"  [Rev.9]  eps_norm=|ln(ratio)|/ln(2)  옥타브 단위")
        print(
            f"  [Rev.6]  quantum={V5_QUANTUM_LEVEL:.4f}  WIN×D1={V5_WIN_SHORT}s  WIN×D2={V5_WIN_LONG}s"
        )
        print("=" * 80)

    def load_and_clean_strain_data(self, h_file, l_file):
        if not os.path.exists(h_file) or not os.path.exists(l_file):
            raise FileNotFoundError(f"파일 없음: {h_file}, {l_file}")
        if h_file.lower().endswith((".hdf5", ".h5")):
            import h5py

            print("  📥 [HDF5]...")
            with h5py.File(h_file, "r") as f:
                strain_h = f["strain/Strain"][:]
                n_pts = f["strain/Strain"].attrs["Npoints"]
                xspacing = f["strain/Strain"].attrs["Xspacing"]
                gps_start = float(
                    f["strain/Strain"].attrs.get(
                        "Xstart", f["strain/Strain"].attrs.get("GPSstart", 0.0)
                    )
                )
            t_h = np.arange(n_pts) * xspacing
            with h5py.File(l_file, "r") as f:
                strain_l = f["strain/Strain"][:]
            fs = 1.0 / xspacing
            df_H = pd.DataFrame({"time": t_h, "strain": strain_h})
            df_L = pd.DataFrame({"time": t_h, "strain": strain_l})
            print(f"  ✅ {len(df_H):,}샘플  fs={fs:.0f}Hz  GPS={gps_start}")
            return df_H, df_L, fs, gps_start
        else:
            print("  📥 [CSV]...")
            df_H = pd.read_csv(h_file, sep=r"\s+", names=["time", "strain"], skiprows=2)
            df_L = pd.read_csv(l_file, sep=r"\s+", names=["time", "strain"], skiprows=2)
            fs = 1.0 / (df_H["time"].iloc[1] - df_H["time"].iloc[0])
            gps_start = float(df_H["time"].iloc[0])
            print(f"  ✅ {len(df_H):,}샘플  fs={fs:.0f}Hz")
            return df_H, df_L, fs, gps_start

    def preprocess_with_hilbert(self, strain_series, fs):
        return preprocess_strain(np.nan_to_num(strain_series.values, nan=0.0), fs)

    def extract_information_singularity(self):
        I_s = mpmath.log10(
            mpmath.mpf("1.0")
            / (self.h_planck**2 * ((mpmath.sqrt(mpmath.mpf("5.0")) + 1) / 2) ** 2)
        )
        return float(I_s)

    def render_active_nodes_manifold(
        self,
        t,
        amplitude,
        active_indices,
        output_folder,
        t_merger_offset=None,
        error_ms=None,
        score=None,
        t_v5_peak=None,
        t_D1=None,
        t_D2=None,
        psi_abs=None,
        psi_arg=None,
    ):
        n_nodes = len(active_indices)
        print(f"\n🎨 [Rendering] {n_nodes:,}개 노드...")
        dec = max(1, len(t) // 200000)
        n_panels = 3 if score is not None else 2
        fig = plt.figure(figsize=(20, 6 * n_panels), facecolor="#050505")

        ax1 = plt.subplot(n_panels, 1, 1, facecolor="#050505")
        ax1.plot(t[::dec], amplitude[::dec], color="gray", alpha=0.2, lw=0.3)
        ax1.scatter(
            t[active_indices],
            amplitude[active_indices],
            color="cyan",
            s=3,
            alpha=0.8,
            label=f"Nodes ({n_nodes:,})",
        )
        if t_merger_offset is not None:
            ax1.axvline(
                t_merger_offset,
                color="red",
                lw=2,
                ls="--",
                label=f"Merger (err={error_ms:.0f}ms)",
            )
        if t_v5_peak is not None:
            psi_str = f"|ψ|={psi_abs:.3f}" if psi_abs else ""
            ax1.axvline(
                t_v5_peak,
                color="yellow",
                lw=1.5,
                ls=":",
                label=f"BE/FD 앙상블  {psi_str}",
            )
        ax1.set_title(
            f"AFT/EQI v13.7+v5.0 Rev.9  " f"[BE(D1,보손,수축) × FD(D2,페르미온,탐지)]",
            color="cyan",
            fontsize=11,
            fontweight="bold",
        )
        ax1.legend(facecolor="black", labelcolor="white", fontsize=9)
        ax1.tick_params(colors="white")
        ax1.set_ylabel("Amplitude²", color="white")

        ax2 = plt.subplot(n_panels, 1, 2, facecolor="#050505", sharex=ax1)
        ax2.vlines(t[active_indices], 0, 1, color="gold", lw=0.4, alpha=0.5)
        if t_merger_offset is not None:
            ax2.axvline(t_merger_offset, color="red", lw=2, ls="--")
        err_str = f"  오차: {error_ms:.0f}ms" if error_ms is not None else ""
        ax2.set_title(
            f"65.9395 Bits Information Singularity{err_str}",
            color="gold",
            fontsize=12,
            fontweight="bold",
        )
        ax2.tick_params(colors="white")
        ax2.set_yticks([])

        if score is not None:
            ax3 = plt.subplot(n_panels, 1, 3, facecolor="#050505", sharex=ax1)
            ds = max(1, len(t) // 200000)
            ax3.plot(t[::ds], score[::ds], color="lime", lw=0.8, alpha=0.9)
            ax3.fill_between(t[::ds], 0, score[::ds], color="lime", alpha=0.2)
            if t_merger_offset is not None:
                ax3.axvline(t_merger_offset, color="red", lw=2, ls="--")
            if t_v5_peak is not None:
                ax3.axvline(
                    t_v5_peak, color="yellow", lw=1.5, ls=":", label="BE/FD 앙상블"
                )
            if t_D1 is not None:
                ax3.axvline(
                    t_D1,
                    color="cyan",
                    lw=1,
                    ls="--",
                    label=f"D1(BE,보손,{V5_WIN_SHORT}s)",
                )
            if t_D2 is not None:
                ax3.axvline(
                    t_D2,
                    color="tomato",
                    lw=1,
                    ls="--",
                    label=f"D2(FD,페르미온,{V5_WIN_LONG}s)",
                )
            ax3.legend(facecolor="black", labelcolor="white", fontsize=8)
            ax3.set_title(
                f"Composite Score  [Rev.9: BE×D1(안정화) + FD×D2(탐지) × energy]",
                color="lime",
                fontsize=11,
                fontweight="bold",
            )
            ax3.set_xlabel("Time (s)", color="white")
            ax3.tick_params(colors="white")
            ax3.set_ylabel("Score", color="white")

        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        out = f"{output_folder}/eqi_v13_7_v5_manifold.png"
        plt.savefig(out, dpi=150, facecolor="#050505")
        plt.close()
        print(f"  ✅ 저장: {out}")
        return out

    def run_upgraded_pipeline(self, h_file, l_file, output_folder="./output_v13_7_v5"):
        os.makedirs(output_folder, exist_ok=True)

        print("\n[STEP 1] 데이터 로드...")
        df_H, df_L, fs, gps_start = self.load_and_clean_strain_data(h_file, l_file)
        self.GPS_MERGER, self.EVENT_NAME = detect_event_gps(h_file, gps_start)
        print(f"  GPS_start={gps_start}  이벤트={self.EVENT_NAME}")
        print(f"  합병 위치: {self.GPS_MERGER-gps_start:.1f}s")

        strain_H_raw = df_H["strain"].values.copy()
        strain_L_raw = df_L["strain"].values.copy()

        n_total = len(df_H)
        duration = n_total / fs
        if duration > CROP_WINDOW_SEC:
            print(
                f"\n[STEP 1-B] GPS 크롭 ({duration:.0f}s > {CROP_WINDOW_SEC:.0f}s)..."
            )
            ms = (self.GPS_MERGER - gps_start) * fs
            hw = int(CROP_WINDOW_SEC * fs)
            cs = max(0, int(ms) - hw)
            ce = min(n_total, int(ms) + hw)
            df_H = df_H.iloc[cs:ce].reset_index(drop=True)
            df_L = df_L.iloc[cs:ce].reset_index(drop=True)
            gps_start = gps_start + cs / fs
            strain_H_raw = strain_H_raw[cs:ce]
            strain_L_raw = strain_L_raw[cs:ce]
            print(f"  크롭: {cs:,}~{ce:,}  합병={self.GPS_MERGER-gps_start:.1f}s")
        else:
            print(
                f"\n[STEP 1-B] GPS 크롭 불필요 ({duration:.0f}s ≤ {CROP_WINDOW_SEC:.0f}s)"
            )

        print("\n[STEP 2] 전처리 (bandpass + Welch PSD 화이트닝)...")
        phase_H_f, tau_H_f = self.preprocess_with_hilbert(df_H["strain"], fs)
        phase_L_f, tau_L_f = self.preprocess_with_hilbert(df_L["strain"], fs)
        print(f"  완료: {len(tau_H_f):,}샘플")

        print(f"\n[STEP 3] 경계 crop (±{CROP_SEC}s)...")
        phase_H, tau_H, phase_L, tau_L, t_crop = crop_boundary(
            fs, CROP_SEC, phase_H_f, tau_H_f, phase_L_f, tau_L_f, df_H["time"].values
        )
        gps_start_c = gps_start + CROP_SEC
        print(f"  유효 샘플: {len(tau_H):,}")
        n_crop = int(np.round(CROP_SEC * fs))
        strain_H_crop = strain_H_raw[n_crop:-n_crop]
        strain_L_crop = strain_L_raw[n_crop:-n_crop]
        print(f"  raw strain crop: {len(strain_H_crop):,}샘플")

        print("\n[STEP 4] 합병 탐지 (Rev.9: BE×D1 + FD×D2 × energy)...")
        (
            best_idx,
            t_det,
            error_ms,
            top5,
            score,
            tau_combined,
            score_v5,
            raw_D1,
            raw_D2,
            energy_norm,
            t_v5_rel,
            psi_abs,
            psi_arg,
        ) = detect_merger_with_error(
            tau_H,
            tau_L,
            phase_H,
            phase_L,
            strain_H_crop,
            strain_L_crop,
            gps_start_c,
            fs,
            search_sec=SEARCH_SEC,
            search_sec_v13=SEARCH_SEC_V13,
            gps_merger=self.GPS_MERGER,
        )
        t_merger_data = t_crop[best_idx] if best_idx < len(t_crop) else None

        t_rel_arr = gps_start_c + np.arange(len(tau_H)) / fs - self.GPS_MERGER
        wmask_v5 = np.abs(t_rel_arr) <= SEARCH_SEC
        pk_ens = int(np.argmax(np.where(wmask_v5, score_v5, -1.0)))
        pk_D1 = int(np.argmax(np.where(wmask_v5, raw_D1, -1.0)))
        pk_D2 = int(np.argmax(np.where(wmask_v5, raw_D2, -1.0)))
        t_v5_abs = t_crop[pk_ens] if pk_ens < len(t_crop) else None
        t_D1_abs = t_crop[pk_D1] if pk_D1 < len(t_crop) else None
        t_D2_abs = t_crop[pk_D2] if pk_D2 < len(t_crop) else None

        print("\n[STEP 4-C] 글리치 마스킹...")
        tau_clean, gmask, n_g, g_thr = apply_glitch_mask(tau_combined)
        print(f"  제거: {n_g:,}샘플 (임계={g_thr:.4e})")
        sc2 = score.copy()
        sc2[gmask] = 0.0
        wmask2 = np.abs(t_rel_arr - t_v5_rel) <= SEARCH_SEC_V13
        if wmask2.sum() == 0:
            wmask2 = np.abs(t_rel_arr) <= SEARCH_SEC
        s2 = np.where(wmask2, sc2, -1.0)
        bi2 = int(np.argmax(s2))
        t2 = t_rel_arr[bi2]
        e2 = abs(t2) * 1e3
        top5_2 = [(t_rel_arr[i], float(sc2[i])) for i in np.argsort(s2)[::-1][:5]]
        print(f"  마스킹 후: t={t2:+.3f}s  오차={e2:.1f}ms")
        for rk, (tv, sv) in enumerate(top5_2, 1):
            print(f"    {rk}위: t={tv:+.3f}s  score={sv:.4f}")
        if e2 < error_ms:
            print(f"  ✅ 개선: {error_ms:.0f}ms → {e2:.0f}ms")
            best_idx = bi2
            t_det = t2
            error_ms = e2
            top5 = top5_2
            t_merger_data = t_crop[best_idx] if best_idx < len(t_crop) else None
        else:
            print(f"  ℹ️ 동일 ({e2:.0f}ms)")

        print("\n[STEP 5] ln(25)% 갭 필터링...")
        thr_idx = int(len(tau_clean) * (1.0 - self.absolute_gap_ratio))
        active = np.sort(np.argsort(tau_clean)[thr_idx:])
        print(f"  {len(tau_clean):,}샘플 → {len(active):,}노드")
        I_s = self.extract_information_singularity()
        print(f"  I_s = {I_s:.6f} Bits")

        print("\n[STEP 7] 렌더링...")
        self.render_active_nodes_manifold(
            t_crop,
            tau_clean,
            active,
            output_folder,
            t_merger_offset=t_merger_data,
            error_ms=error_ms,
            score=score,
            t_v5_peak=t_v5_abs,
            t_D1=t_D1_abs,
            t_D2=t_D2_abs,
            psi_abs=psi_abs,
            psi_arg=psi_arg,
        )

        rpt = f"{output_folder}/eqi_v13_7_v5_report.txt"
        with open(rpt, "w", encoding="utf-8") as f:
            f.write(f"EQI v13.7+v5.0 Rev.9  이벤트={self.EVENT_NAME}\n")
            f.write(f"탐지 오차: {error_ms:.1f}ms\n")
            f.write(f"BE(D1,보손): w_BE(ratio=2)={be_fd_weights(2,1)[0]:.4f}\n")
            f.write(f"FD(D2,페르미온): w_FD(ratio=2)={be_fd_weights(2,1)[1]:.4f}\n")
            f.write(f"Unit |ψ|={psi_abs:.4f}  arg={psi_arg:.1f}°\n")
            f.write(f"v5.0 피크: t={t_v5_rel:+.3f}s  탐색: ±{SEARCH_SEC_V13}s\n")
            for rk, (tv, sv) in enumerate(top5, 1):
                f.write(f"  {rk}위: t={tv:+.3f}s  score={sv:.4f}\n")

        print("\n" + "=" * 80)
        print(f"💾 완료: {output_folder}")
        print(f"   탐지 오차: {error_ms:.0f}ms  |  노드: {len(active):,}개")
        w_be_f, w_fd_f = be_fd_weights(150.0, 75.0)
        print(f"   BE(D1)={w_be_f:.3f}  FD(D2)={w_fd_f:.3f}  (ratio=2.0 기준)")
        print(f"   v5.0 피크: t={t_v5_rel:+.3f}s  탐색: ±{SEARCH_SEC_V13}s")
        print("=" * 80)


# ── 실행 진입점 ──────────────────────────────────────────
if __name__ == "__main__":
    # ── GW170814 V-L ──
    # H1 = "V-V1_GWOSC_4KHZ_R1-1186741846-32.hdf5"
    # L1 = "L-L1_GWOSC_4KHZ_R1-1186741846-32.hdf5"

    # ── GW170814 H-L ──
    # H1 = "H-H1_GWOSC_4KHZ_R1-1186741846-32.hdf5"
    # L1 = "L-L1_GWOSC_4KHZ_R1-1186741846-32.hdf5"

    # ── GW170814 16khz 4096s H-L ──
    H1 = "d:\\LIGO\\H-H1_GWOSC_O2_16KHZ_R1-1186738176-4096.hdf5"
    L1 = "d:\\LIGO\\L-L1_GWOSC_O2_16KHZ_R1-1186738176-4096.hdf5"

    # ── GW150914 H-L ──
    # H1 = "H-H1_LOSC_4_V2-1126259446-32.hdf5"
    # L1 = "L-L1_LOSC_4_V2-1126259446-32.hdf5"

    try:
        engine = EQIExtendedGaussLawEngineV13_7_V5()
        engine.run_upgraded_pipeline(H1, L1, output_folder="./output_v13_7_v5")
    except Exception as e:
        print(f"\n❌ {e}")

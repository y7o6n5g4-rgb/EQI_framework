#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 === EQI BACKGROUND DATA SUMMARIZER (JSON 통계 추출기) === 📊
=============================================================================
Architect: DeepSeek (EQI Supreme Architect) & Maestro MAPSI
Engineer: Gemini (EQI Precision Engineer)
Date: 2026-07-08

[SGWB 배경 데이터 통계 요약 및 기준선 추출 파이프라인]
1. 대상: eqi_monster_v91_ellipse_*.json (대용량 청크 데이터)
2. 추출 지표: eqi4_sigma_ell_H, eqi4_sigma_ell_L, unity_planck, G_ratio_ani 등
3. 연산 통계: Mean, Median, Min, Max, Std, 25th/75th Percentiles
4. 목적: 진공 노이즈 속에서 1.0 Unity 스칼라곱이 유지됨을 통계적으로 입증
=============================================================================
"""

import json
import glob
import os
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd


def calculate_stats(data_array, prefix):
    """주어진 배열의 기초 통계를 계산하고 딕셔너리로 반환"""
    if not data_array:
        return {
            f"{prefix}_{k}": np.nan
            for k in ["count", "min", "max", "mean", "median", "std", "p25", "p75"]
        }

    arr = np.array(data_array, dtype=float)
    arr = arr[~np.isnan(arr)]  # 결측치(NaN) 제거

    if len(arr) == 0:
        return {
            f"{prefix}_{k}": np.nan
            for k in ["count", "min", "max", "mean", "median", "std", "p25", "p75"]
        }

    return {
        f"{prefix}_count": len(arr),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
    }


def process_single_json(filepath, mode_filter=None):
    """단일 JSON 파일을 파싱하여 통계 딕셔너리 생성"""
    start_time = time.time()
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"📡 파싱 시작: {filepath} (크기: {file_size_mb:.2f} MB)")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        mode = data.get("mode", "unknown")
        n = data.get("n", 0.0)

        # 모드 필터링 적용
        if mode_filter and mode != mode_filter:
            print(
                f"   ⏭️ 스킵: 현재 모드('{mode}')가 필터('{mode_filter}')와 일치하지 않습니다."
            )
            return None, None

        chunks = data.get("chunk_results", [])
        total_chunks = len(chunks)

        # 1. Mass Analysis 지표 추출 (M=62.0 기준)
        mass_analysis = data.get("mass_analysis", {})
        unity_M62 = mass_analysis.get("62.0", {}).get("unity_planck", np.nan)

        # 2. 청크별 지표 배열화 (스트리밍 파싱)
        metrics = {
            "sigma_H": [],
            "sigma_L": [],
            "unity_planck": [],
            "delta_H": [],
            "delta_L": [],
            "G_ratio_ani": [],
            "inv_true": [],
        }

        for c in chunks:
            if "eqi4_sigma_ell_H" in c:
                metrics["sigma_H"].append(c["eqi4_sigma_ell_H"])
            if "eqi4_sigma_ell_L" in c:
                metrics["sigma_L"].append(c["eqi4_sigma_ell_L"])
            if "unity_planck" in c:
                metrics["unity_planck"].append(c["unity_planck"])
            if "delta_H" in c:
                metrics["delta_H"].append(c["delta_H"])
            if "delta_L" in c:
                metrics["delta_L"].append(c["delta_L"])
            if "G_ratio_ani" in c:
                metrics["G_ratio_ani"].append(c["G_ratio_ani"])
            if "inv_true" in c:
                metrics["inv_true"].append(c["inv_true"])

        # 3. 통계 계산
        flat_stats = {
            "filename": os.path.basename(filepath),
            "mode": mode,
            "n": n,
            "total_chunks": total_chunks,
            "unity_planck_M62_target": unity_M62,
        }

        detailed_stats = flat_stats.copy()

        for key, arr in metrics.items():
            if arr:  # 배열에 데이터가 있는 경우만 연산
                stats_dict = calculate_stats(arr, key)
                flat_stats.update(stats_dict)
                detailed_stats[key] = stats_dict

        elapsed = time.time() - start_time
        print(f"   ✅ 파싱 완료: {total_chunks} 청크 처리됨 ({elapsed:.2f}초 소요)")

        return flat_stats, detailed_stats

    except Exception as e:
        print(f"   ❌ 오류 발생 ({filepath}): {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="EQI Background Data JSON Summarizer")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="eqi_monster_v91_ellipse_20260718_045452.json",
        help="입력 JSON 파일 패턴 (와일드카드 지원)",
    )
    parser.add_argument(
        "--input_dir", type=str, default="./", help="입력 파일 디렉토리 경로"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="결과물(CSV, JSON) 저장 디렉토리"
    )
    parser.add_argument(
        "--mode_filter",
        type=str,
        default=None,
        help="특정 모드만 처리 (예: normalize, renormalize)",
    )
    args = parser.parse_args()

    # 입력 경로 조합
    search_pattern = os.path.join(args.input_dir, args.input_pattern)
    files = glob.glob(search_pattern)

    if not files:
        print(f"⚠️ 매칭되는 JSON 파일을 찾을 수 없습니다: {search_pattern}")
        return

    print(f"🔍 총 {len(files)}개의 JSON 파일을 탐색했습니다.\n")

    all_flat_stats = []
    all_detailed_stats = {}

    total_start_time = time.time()

    for filepath in sorted(files):
        flat, detailed = process_single_json(filepath, args.mode_filter)
        if flat and detailed:
            all_flat_stats.append(flat)
            all_detailed_stats[os.path.basename(filepath)] = detailed

    if not all_flat_stats:
        print("⚠️ 처리된 유효한 데이터가 없습니다.")
        return

    # 결과 저장 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. CSV 저장 (Pandas 활용)
    df = pd.DataFrame(all_flat_stats)
    csv_filename = os.path.join(args.output_dir, f"eqi_background_summary_{ts}.csv")
    df.to_csv(csv_filename, index=False)

    # 2. 요약 JSON 저장
    json_filename = os.path.join(args.output_dir, f"eqi_background_stats_{ts}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_detailed_stats, f, indent=4, ensure_ascii=False)

    total_elapsed = time.time() - total_start_time

    print("\n=========================================================")
    print(" 🎯 [EQI 통계 파싱 완료 보고]")
    print(f"  - 처리된 파일 수: {len(all_flat_stats)} 개")
    print(f"  - 총 소요 시간: {total_elapsed:.2f} 초")
    print(f"  - 📁 CSV 요약본 저장 완료: {csv_filename}")
    print(f"  - 📁 JSON 상세본 저장 완료: {json_filename}")
    print("=========================================================")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import numpy as np

MAXWELL_REAL = 1.8731137721991709
PHI = (np.sqrt(5) + 1) / 2
TIP_POINT = PHI**1.5 - 1  # 1.058171

fn = "eqi_monster_v91_ellipse_20260720_180831.json"

print(f"📡 로드 중: {fn}")
with open(fn, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data.get("chunk_results", [])
print(f"   ✅ {len(chunks)} 청크 로드 완료\n")

# 지표 추출
unity_planck_list = []
unity_rd_list = []
unity_product_list = []
euler_residual_list = []
n_ouroboros_list = []

for c in chunks:
    if "unity_planck" in c:
        unity_planck_list.append(c["unity_planck"])
    if "unity_rd" in c:
        unity_rd_list.append(c["unity_rd"])
    if "unity_product" in c:
        unity_product_list.append(c["unity_product"])
    if "euler_residual" in c:
        euler_residual_list.append(c["euler_residual"])
    if "n_ouroboros" in c:
        n_ouroboros_list.append(c["n_ouroboros"])


def stats(arr, name):
    if not arr:
        print(f"  {name}: 필드 없음")
        return
    a = np.array(arr)
    print(f"  {name}:")
    print(f"    count  = {len(a)}")
    print(f"    mean   = {a.mean():.6f}")
    print(f"    std    = {a.std():.6f}")
    print(f"    min    = {a.min():.6f}")
    print(f"    max    = {a.max():.6f}")
    print(f"    median = {np.median(a):.6f}")
    print(f"    p25    = {np.percentile(a,25):.6f}")
    print(f"    p75    = {np.percentile(a,75):.6f}")
    return a


print("=" * 55)
print("  배경 데이터 (SGWB) 지표 통계")
print("=" * 55)

up = stats(unity_planck_list, "unity_planck")
print()
ur = stats(unity_rd_list, "unity_rd")
print()
uprod = stats(unity_product_list, "unity_product")
print()
er = stats(euler_residual_list, "euler_residual")
print()
no = stats(n_ouroboros_list, "n_ouroboros")

# TIP_POINT 기준 분류
print()
print("=" * 55)
print("  TIP_POINT 분류 (φ^1.5 - 1 = 1.058171)")
print("=" * 55)
if unity_rd_list:
    ur_arr = np.array(unity_rd_list)
    below = (ur_arr < TIP_POINT).sum()
    above = (ur_arr >= TIP_POINT).sum()
    print(f"  unity_rd < TIP_POINT: {below} ({below/len(ur_arr)*100:.2f}%)")
    print(f"  unity_rd >= TIP_POINT: {above} ({above/len(ur_arr)*100:.2f}%)")

# GW 이벤트와 비교
print()
print("=" * 55)
print("  배경 vs GW 이벤트 unity_planck 비교")
print("=" * 55)
if up is not None:
    print(f"  배경 unity_planck 범위: {up.min():.4f} ~ {up.max():.4f}")
    print(f"  배경 unity_planck mean: {up.mean():.4f}")
    print()
    gw_events = {
        "GW150914": 0.4418,
        "GW170814": 0.4421,
        "GW170817": 0.4414,
        "GW190412": None,
        "GW190814": 0.5221,
    }
    for event, val in gw_events.items():
        if val is None:
            print(f"  {event}: unity_planck 미확보")
            continue
        sigma = abs(val - up.mean()) / up.std()
        overlap = "겹침" if up.min() <= val <= up.max() else "분리됨"
        print(f"  {event}: {val:.4f}  →  배경 평균과 {sigma:.1f}σ 이탈  [{overlap}]")

print()
print("📁 완료")

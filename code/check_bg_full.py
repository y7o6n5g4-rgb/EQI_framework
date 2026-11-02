#!/usr/bin/env python3
"""
📊 EQI BACKGROUND FULL ANALYZER (v2.1 — FIXED ZONE & CONSTANTS)
"""

import json
import numpy as np

PHI = (np.sqrt(5) + 1) / 2

# 🔥 모든 임계값을 명시적 상수로 정의
TIP_POINT = PHI**1.5 - 1.0  # T1: 1.058171 (ZONE-2 시작)
THR_T4 = PHI**1.5 - PHI**0.5  # T4: 0.786151 (ZONE-0 끝)
T5 = PHI**1.5 - PHI ** (-1.5)  # T5: 1.572303 (ZONE-3 시작)
THR_T3 = PHI**1.5 - PHI ** (-2.5)  # T3: 1.757888 (ZONE-4 시작)

fn = "eqi_monster_v91_ellipse_20260720_180831.json"

print(f"📡 로드 중: {fn}")
with open(fn, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data.get("chunk_results", [])
print(f"   ✅ {len(chunks)} 청크 로드 완료\n")


# 🔥 ZONE 분류 함수 (상수 사용)
def classify_zone(ur):
    if ur < THR_T4:
        return 0
    elif ur < TIP_POINT:
        return 1
    elif ur < T5:
        return 2
    elif ur < THR_T3:
        return 3
    else:
        return 4


# ✅ E_D1_emission 추출
emission_list = []
for c in chunks:
    if "E_D1_emission" in c:
        emission_list.append(c["E_D1_emission"])

# ✅ ZONE 계산 (JSON의 zone 필드 무시하고 직접 계산)
zone_list = [classify_zone(c["unity_rd"]) for c in chunks]

arr_em = np.array(emission_list)
arr_zone = np.array(zone_list)

print("=" * 65)
print("  🔥 배경 데이터 (SGWB) 중력적 광전효과 방출량 통계")
print("=" * 65)
print(f"  E_D1_emission:")
print(f"    count  = {len(arr_em)}")
print(f"    mean   = {arr_em.mean():.6f}")
print(f"    std    = {arr_em.std():.6f}")
print(f"    min    = {arr_em.min():.6f}")
print(f"    max    = {arr_em.max():.6f}")
print(f"    median = {np.median(arr_em):.6f}")
print(f"    p25    = {np.percentile(arr_em, 25):.6f}")
print(f"    p75    = {np.percentile(arr_em, 75):.6f}")
print(
    f"  방출량 > 0 청크: {(arr_em > 0).sum()}/{len(arr_em)} ({((arr_em > 0).sum()/len(arr_em))*100:.2f}%)"
)

print("\n" + "=" * 65)
print("  📍 배경 데이터 ZONE 분포 (0~4)")
print("=" * 65)
unique, counts = np.unique(arr_zone, return_counts=True)
zone_map = {
    0: "ZONE-0 (안정)",
    1: "ZONE-1 (금지)",
    2: "ZONE-2 (암흑물질)",
    3: "ZONE-3 (암흑에너지)",
    4: "ZONE-4 (인과율 초과)",
}
for z, cnt in zip(unique, counts):
    print(f"  {zone_map.get(z, f'ZONE-{z}')}: {cnt} ({cnt/len(arr_zone)*100:.2f}%)")

print("\n" + "=" * 65)
print("  🎯 GW 이벤트 방출량과의 비교 (피크값)")
print("=" * 65)
print(f"  배경 최대 방출량     : {arr_em.max():.6f}")
print(f"  GW170817 방출 피크  : 0.521588 (과도 현상)")
print(f"  GW190412 방출 피크  : 0.767319 (ZONE-2, 암흑물질)")
print(f"  GW190814 방출 피크  : 0.824123 (ZONE-3, 암흑에너지)")
if arr_em.max() > 0.52:
    print(
        "  ⚠️ 배경에도 방출 피크가 존재합니다. 하지만 이는 과도 현상(Transient)이 아닌"
    )
    print("     우주가 ZONE-3(암흑에너지) 상태에 머물면서 가지는 정적 잉여 텐션입니다.")

print("\n📁 완료")

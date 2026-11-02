import json
import numpy as np

files = {
    "GW150914_H-L": "eqi_monster_v91_ellipse_20260722_035902.json",
    # "GW170817_V-L": "eqi_monster_v91_ellipse_20260719_041642.json",
}

PHI = (np.sqrt(5) + 1) / 2
TIP_POINT = PHI**1.5 - 1  # T1: 1.058171 (ZONE-2 시작)
T5 = PHI**1.5 - PHI**-1.5  # T5: 1.572303 (ZONE-3 시작)
THR_T3 = PHI**1.5 - PHI**-2.5  # T3: 1.758435 (ZONE-4 시작, 인과율 경계)

for event, fn in files.items():
    try:
        with open(fn, encoding="utf-8") as f:
            data = json.load(f)
        chunks = data["chunk_results"]

        ur = [c["unity_rd"] for c in chunks]
        em = [c["E_D1_emission"] for c in chunks]

        arr_ur = np.array(ur)
        arr_em = np.array(em)

        print(f"\n=== {event} ===")
        print(
            f"unity_rd: mean={arr_ur.mean():.4f} max={arr_ur.max():.4f} @ chunk {arr_ur.argmax()}"
        )
        print(f"E_D1_emission: max={arr_em.max():.6f} @ chunk {arr_em.argmax()}")
        print(f"방출 > 0 청크: {(arr_em > 0).sum()}/64")
        print(f"TIP_POINT (T1) = {TIP_POINT:.6f}")
        print(f"T5 (힉스 매듭)  = {T5:.6f}")
        print(f"THR_T3 (인과율)  = {THR_T3:.6f}")
        print()
        print(f"chunk | time  | unity_rd | E_D1_emission | 구간 (ZONE)")
        print("-" * 65)
        for c in chunks:
            ur_val = c["unity_rd"]
            em_val = c["E_D1_emission"]
            if ur_val > TIP_POINT or em_val > 0:
                if ur_val >= THR_T3:
                    zone = "T3 초과 (ZONE-4)"
                elif ur_val >= T5:
                    zone = "T5~T3 (ZONE-3)"
                elif ur_val >= TIP_POINT:
                    zone = "T1~T5 (ZONE-2)"
                else:
                    zone = "안정 (ZONE-0)"
                print(
                    f"  {c['chunk']:2d}  | {c['time_s']:4.1f}s | {ur_val:.6f} | {em_val:.6f}      | {zone}"
                )
    except FileNotFoundError:
        print(f"\n=== {event} ===")
        print(f"⚠️ 파일을 찾을 수 없습니다: {fn}")
        print("현재 디렉토리에 해당 JSON 파일이 있는지 확인해 주세요.")

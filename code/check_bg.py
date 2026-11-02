import json, numpy as np

with open("eqi_monster_v91_ellipse_20260718_045452.json") as f:
    data = json.load(f)

unity_rd_list = [c["unity_rd"] for c in data["chunk_results"] if "unity_rd" in c]

if unity_rd_list:
    arr = np.array(unity_rd_list)
    print(f"unity_rd 청크 수: {len(arr)}")
    print(f"mean  = {arr.mean():.6f}")
    print(f"std   = {arr.std():.6f}")
    print(f"min   = {arr.min():.6f}")
    print(f"max   = {arr.max():.6f}")
    print(f"TIP_POINT 초과 청크: {(arr >= 1.058171).sum()}")
else:
    print("unity_rd 필드 없음")
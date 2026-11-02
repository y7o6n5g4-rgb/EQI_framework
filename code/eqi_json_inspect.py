
import json, sys

fname = sys.argv[1] if len(sys.argv) > 1 else "eqi_monster_v92_ellipse_20260730_151314.json"
with open(fname) as f:
    data = json.load(f)

print("=== 최상위 키 ===")
for k,v in data.items():
    if k != "chunk_results":
        print(f"  {k}: {v}")

print()
print("=== chunk_results[0] 키 목록 ===")
chunks = data.get("chunk_results", [])
if chunks:
    c0 = chunks[0]
    for k,v in c0.items():
        print(f"  {k}: {v}")
    print()
    print(f"총 청크 수: {len(chunks)}")
    print()
    print("=== chunk_results[0..3] unity_rd, E_D1_emission ===")
    for c in chunks[:4]:
        print(f"  chunk={c.get('chunk')}  unity_rd={c.get('unity_rd')}  "
              f"E_D1={c.get('E_D1_emission', c.get('e_d1','?'))}  "
              f"unity_planck={c.get('unity_planck','?')}")

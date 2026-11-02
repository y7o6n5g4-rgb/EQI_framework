import json

with open("eqi_monster_v91_ellipse_20260716_193002.json") as f:
    data = json.load(f)

pt = data["v91_targets"]
print(f"상전이 청크 수: {pt['phase_transition_count']}")
print(f"최초 상전이 청크: {pt['first_phase_transition_chunk']}")
print(f"상전이 청크 목록: {pt['phase_transition_chunks']}")
print(f"MAXWELL_REAL = 1.8731137721991709")
print()

# 청크별 unity_product 출력
print("chunk | time_s | unity_product | phase_transition")
print("-" * 55)
for c in data["chunk_results"]:
    pt_flag = "⚡상전이" if c.get("phase_transition") else "  안정  "
    print(
        f"  {c['chunk']:2d}  | {c['time_s']:5.1f}s | {c['unity_product']:.6f}    | {pt_flag}"
    )
    print(
        f"  {c['chunk']:2d}  | {c['time_s']:5.1f}s | {c['unity_product']:.4f} | {c['euler_residual']:+.4f} | n={c['n_ouroboros']:.1f}"
    )

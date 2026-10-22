#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R8 접속 어댑터 회귀시험 — 입력·기대 상태는 code/rupert_r8_adapter_tests.json (실행 전 고정) 에서 읽는다.
결과는 기대와 대조만 하며, 표본·재시도로 덮어쓰지 않는다.
실행: python3 code/rupert_r8_adapter_test.py  (선행: python3 code/rupert_local_region_scope.py)
"""

import json
import sys
import time

sys.path.insert(0, "code")
from rupert_r8_adapter import LocalTheorem  # noqa: E402
from rupert_local_region_scope import evaluate_request  # noqa: E402

OUT = "code/rupert_r8_adapter_test_output.txt"
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


spec = json.load(open("code/rupert_r8_adapter_tests.json"))
LT = LocalTheorem()
log("=" * 100)
log(f"R8 접속 어댑터 회귀시험 (기대값 고정: {spec['registered']})")
log(f"  범위 기록 로드: verifier PASS & 증명서 해시 일치 = {LT.loaded_ok} ({LT.cert_hash[:16]}…)")
log(f"  검증 정리 θ_max = {LT.theta_max}, 임계 1 + 2cos(1/20) ∈ {LT.thr!r}")
log("=" * 100)
n_match = 0
rows = []
for t in spec["cell_tests"]:
    obj = dict(spec["object_default"])
    obj.update(t.get("object", {}))
    t0 = time.time()
    r = LT.classify(obj, t["units"], t["box"])
    dt = time.time() - t0
    ok = r["status"] == t["expect"] and ("expect_detail" not in t or r.get("detail") == t["expect_detail"])
    n_match += ok
    log(f"[{t['id']}] {t['desc']}")
    log(f"     기대 {t['expect']}{'/' + t['expect_detail'] if 'expect_detail' in t else ''}"
        f"  →  결과 {r['status']}/{r.get('detail')}   {'일치' if ok else '불일치'}  ({dt:.2f}s)")
    log(f"     사유: {r['reason']}")
    for k in ("q_box", "trace_box", "trace_center", "threshold", "H_upper", "M1_in", "M2_in", "M1_out", "M2_out"):
        if k in r:
            log(f"     {k}: {r[k]}")
    rows.append(dict(id=t["id"], expect=t["expect"], expect_detail=t.get("expect_detail"), result=r, match=ok))
log("")
scope = json.load(open("code/rupert_local_region_scope.json"))
for q in spec["request_tests"]:
    st, why = evaluate_request(scope, q["theta_max"])
    ok = st == q["expect"]
    n_match += ok
    log(f"[{q['id']}] 요청 θ_max = {q['theta_max']}: 기대 {q['expect']} → 결과 {st} {'일치' if ok else '불일치'} — {why}")
    rows.append(dict(id=q["id"], expect=q["expect"], result={"status": st, "reason": why}, match=ok))
total = len(spec["cell_tests"]) + len(spec["request_tests"])
log("")
log(f"기대 일치 {n_match}/{total}")
with open("code/rupert_r8_adapter_test_results.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")

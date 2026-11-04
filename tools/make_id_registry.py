#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID 등록부 1회 생성 (CLAUDE.md 규칙 7) — 기존 index/ledger.csv (L0001~L0265) 에서
각 ID 의 '주 원문 줄' 키를 뽑아 index/id_registry.csv 로 고정한다.
  키 = "<파일 약칭>:<행>"  (LOG = 기준 로그, ENG = 엔진 소스). 묶음 행은 묶음의 첫 원문 줄.
이미 등록부가 있으면 아무것도 하지 않는다 (덮어쓰기 금지). 이후 ID 는 tools/build_step2.py 가 등록부 끝에만 추가.
"""
import csv, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "index", "id_registry.csv")
LED = os.path.join(ROOT, "index", "ledger.csv")
ENG = "code/eqi_unified_engine_v9_7.py"

def key_of(r):
    if r["근거_파일"] == ENG:
        m = re.search(r"✓ (\d+)행", r["엔진_§"])
        if m: return f"LOG:{m.group(1)}"
        m = re.search(r"머리말 (\d+)행", r["근거_위치"])
        return f"ENG:{m.group(1)}"
    m = re.search(r"✓ (\d+)행", r["근거_위치"])          # z0 '대칭 이벤트'
    if m: return f"LOG:{m.group(1)}"
    m = re.search(r"로그 (\d+)", r["근거_위치"])
    return f"LOG:{m.group(1)}"

if __name__ == "__main__":
    if os.path.exists(REG):
        sys.exit("id_registry.csv 이미 있음 — 덮어쓰지 않음")
    rows = list(csv.DictReader(open(LED, encoding="utf-8")))
    out = [dict(id=r["id"], 키=key_of(r), 최초_항목=r["항목"], 최초_별칭=r["별칭"], 분리_원ID="") for r in rows]
    keys = [o["키"] for o in out]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup: sys.exit(f"키 중복: {sorted(dup)}")
    with open(REG, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "키", "최초_항목", "최초_별칭", "분리_원ID"]); w.writeheader(); w.writerows(out)
    print("등록부", len(out), "행")

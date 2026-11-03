#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 7 (후속_정정 채우기) 자동 대조 → index/verify_step7.md
  원장 CSV · map_step7.csv · 원문 파일 · git 기준판 (단계 7 직전 커밋 BASE 의 index/ledger.csv) 만 본다. 생성 스크립트 내부 값 미사용.
  W1 상태 열 불변      : 모든 행의 상태 = BASE 원장의 상태 = step7_status_snapshot.csv 의 상태
  W2 다른 열 불변      : 현행_상태·후속_정정 밖 모든 열 = BASE. 후속_정정은 바뀐 행만, BASE 값이 앞에 그대로 남고 '단계 7:' 덧붙임
  W3 현행_상태 원천    : 현행_상태가 있는 행마다 후속_정정에 원천 L-ID, 그 원천의 주 근거 원문 줄에 그 낱말 ('폐기'/'철회' 원문 그대로)
  W4 현행_상태 ≠ 상태  : 같으면 빈칸이어야 함 (재확정은 후속_정정에만)
  W5 반영 범위         : 후속_정정이 바뀐 행 = map_step7.csv 대상 행 = MAPSI 승인 6행 (L3335 L4621 L0505 L3641 L3653 L0809) + 확정 쌍 L1714
  W6 대응표 일치       : map_step7.csv 의 현행_상태·원천 ID 가 원장 후속_정정·현행_상태와 같음
실행: python3 tools/verify_step7.py [BASE커밋] (기본 5f705de)
"""
import csv, io, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = sys.argv[1] if len(sys.argv) > 1 else "5f705de"
APPROVED = {"L3335", "L4621", "L0505", "L3641", "L3653", "L0809", "L1714"}   # L1714 = MAPSI 확정 쌍 (2026-11-03)
TAGF = {"로그": "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt", "소스": "code/eqi_unified_engine_v9_7.py",
        "v3.4": "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"}
cur = list(csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8")))
base = list(csv.DictReader(io.StringIO(subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE}:index/ledger.csv"]).decode("utf-8"))))
snap = {r["id"]: r["상태"] for r in csv.DictReader(open(os.path.join(ROOT, "index", "step7_status_snapshot.csv"), encoding="utf-8"))}
mp = list(csv.DictReader(open(os.path.join(ROOT, "index", "map_step7.csv"), encoding="utf-8")))
B = {r["id"]: r for r in base}; C = {r["id"]: r for r in cur}
bad = []
def line_of(tok):
    m = re.match(r"(로그|소스|v3\.4|문서 (.+?\.md)) (\d+)", tok)
    if not m: return None
    p = m.group(2) or TAGF[m.group(1)]
    L = open(os.path.join(ROOT, p), encoding="utf-8", errors="replace").read().split("\n")
    return L[int(m.group(3)) - 1]
if set(B) - set(C): bad.append(("—", "W2", f"BASE 의 ID 가 사라짐: {sorted(set(B) - set(C))[:5]}"))
changed = set()
for i, r in C.items():
    b = B.get(i)
    if r["상태"] != snap.get(i): bad.append((i, "W1", f"상태 '{r['상태']}' ≠ 스냅샷 '{snap.get(i)}'"))
    if b is None: continue
    if r["상태"] != b["상태"]: bad.append((i, "W1", f"상태 '{r['상태']}' ≠ BASE '{b['상태']}'"))
    for k in b:
        if k in ("후속_정정",): continue
        if r.get(k) != b[k]: bad.append((i, "W2", f"열 '{k}' 바뀜"))
    if r["후속_정정"] != b["후속_정정"]:
        changed.add(i)
        if not r["후속_정정"].startswith(b["후속_정정"]) or "단계 7:" not in r["후속_정정"][len(b["후속_정정"]):]:
            bad.append((i, "W2", "후속_정정: BASE 값 보존·'단계 7:' 덧붙임 아님"))
    v = r.get("현행_상태", "")
    if v:
        if v == r["상태"]: bad.append((i, "W4", "현행_상태 = 상태"))
        srcs = re.findall(r"현행_상태 '" + re.escape(v) + r"' ← (L\d{4})", r["후속_정정"])
        fsrc = re.findall(r"현행_상태 '" + re.escape(v) + r"' ← ((?:code|성과정리문서)/\S+) (\d+)행 \(MAPSI 확정 쌍", r["후속_정정"])
        for fp_, fn_ in fsrc:
            if v not in open(os.path.join(ROOT, fp_), encoding="utf-8", errors="replace").read().split("\n")[int(fn_) - 1]:
                bad.append((i, "W3", f"원천 {fp_} {fn_}행에 '{v}' 없음"))
        if not srcs and not fsrc: bad.append((i, "W3", "후속_정정에 원천 ID·원천 파일 줄 없음"))
        for s in srcs:
            sr = C.get(s)
            if not sr: bad.append((i, "W3", f"원천 {s} 없음")); continue
            ln = line_of(sr["근거_위치"].split(" / ")[0])
            if ln is None or v not in ln: bad.append((i, "W3", f"원천 {s} 주 근거 원문 줄에 '{v}' 없음"))
mt = {m["대상_ID"] for m in mp}
if not (changed == mt == APPROVED): bad.append(("—", "W5", f"후속_정정 바뀐 행 {sorted(changed)} / map {sorted(mt)} / 승인 {sorted(APPROVED)}"))
for m in mp:
    r = C[m["대상_ID"]]
    if r.get("현행_상태", "") != m["현행_상태"]: bad.append((m["대상_ID"], "W6", "현행_상태 ≠ map"))
    if m["원천_ID"] not in r["후속_정정"]: bad.append((m["대상_ID"], "W6", f"후속_정정에 map 원천 {m['원천_ID']} 없음"))
filled = sorted(i for i, r in C.items() if r.get("현행_상태"))
O = ["# verify_step7.md — 단계 7 (후속_정정 채우기) 자동 대조\n",
     f"> 생성: `python3 tools/verify_step7.py` (BASE = git {BASE}, 단계 7 직전 원장). 원장·대응표·원문·BASE 만 비교.\n",
     "## 요약\n",
     f"- 원장 행: {len(C)} / BASE 행: {len(B)}",
     f"- 현행_상태 채운 행: {len(filled)} ({', '.join(f'{i} {C[i]['현행_상태']}' for i in filled)})",
     f"- 후속_정정 바뀐 행: {len(changed)} ({', '.join(sorted(changed))})",
     f"- 불일치: {len(bad)}" + (" — 통과" if not bad else ""),
     "- 항목: W1 상태 열 불변 (BASE·스냅샷) / W2 다른 열 불변·후속_정정 덧붙임만 / W3 현행_상태 원천 ID·원문 낱말 / W4 현행_상태 ≠ 상태 / W5 반영 범위 = 승인 6행 + 확정 쌍 / W6 대응표 일치\n"]
if bad:
    O += ["## 불일치\n", "| id | 항목 | 사유 |", "|---|---|---|"] + [f"| {a} | {b} | {c.replace('|', chr(92) + '|')} |" for a, b, c in bad[:200]]
open(os.path.join(ROOT, "index", "verify_step7.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print(f"verify_step7: 불일치 {len(bad)} | 현행_상태 {len(filled)} | 후속_정정 변경 {len(changed)}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 8·10 자동 대조 → index/verify_step10.md  (조립본·주제 지도 문자열과 원장·원문만 비교)
  X1 조립본·topics.md 에 적힌 L 번호가 모두 원장에 있음 (범위 'La~Lb' 는 양 끝)
  X2 Part B 표의 상태 칸 = 원장 상태 (+ ' → 현행_상태')
  X3 Part A 인용 줄 ('> ' 줄) 이 <sub> 에 적힌 파일의 적힌 행 원문의 부분 문자열 ('\\|' 는 '|' 로 되돌려 비교)
  X4 topics.csv 의 (키워드, id) 마다 원장 행이 그 키워드에 걸림 (build_def.kw_hits 또는 키워드 열)
"""
import csv, os, re, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IX = os.path.join(ROOT, "index")
sp = importlib.util.spec_from_file_location("bd", os.path.join(ROOT, "tools", "build_def.py")); bd = importlib.util.module_from_spec(sp); sp.loader.exec_module(bd)
R = {r["id"]: r for r in csv.DictReader(open(os.path.join(IX, "ledger.csv"), encoding="utf-8"))}
doc = open(os.path.join(IX, "EQI_색인_v2.md"), encoding="utf-8").read().split("\n")
top = open(os.path.join(IX, "topics.md"), encoding="utf-8").read()
bad = []
for name, txt in (("EQI_색인_v2.md", "\n".join(doc)), ("topics.md", top)):
    for m in re.finditer(r"(?<![A-Za-z0-9])L(\d{4})(?!\d)", txt):
        if f"L{m.group(1)}" not in R: bad.append(("X1", name, f"L{m.group(1)} 원장에 없음"))
inB = False; nb = 0
for l in doc:
    if l.startswith("## Part B"): inB = True
    elif l.startswith("## Part C"): inB = False
    if inB and re.match(r"^\| L\d{4} \|", l):
        c = [x.strip() for x in re.split(r"(?<!\\)\|", l)[1:-1]]
        r = R[c[0]]; want = (r["상태"] or "빈칸") + (f" → {r['현행_상태']}" if r["현행_상태"] else ""); nb += 1
        if c[3] != want: bad.append(("X2", c[0], f"상태 칸 '{c[3]}' ≠ 원장 '{want}'"))
inA = False; qs = []; na = 0
def lines_of(p):
    return open(os.path.join(ROOT, p if p == "CLAUDE.md" else "성과정리문서/" + p), encoding="utf-8").read().split("\n")
for l in doc:
    if l.startswith("## Part A"): inA = True
    elif l.startswith("## Part B"): inA = False
    if not inA: continue
    if l.startswith("> "): qs.append(l[2:].rstrip().replace("\\|", "|"))
    m = re.match(r"^<sub>(\S+) (.+)</sub>$", l)
    if m:
        p, spec = m.group(1), m.group(2); L = lines_of(p); ns = []
        for part in spec.split(", "):
            a = re.match(r"(\d+)(?:–(\d+))?행", part); ns += list(range(int(a.group(1)), int(a.group(2) or a.group(1)) + 1))
        if len(ns) != len(qs): bad.append(("X3", p, f"인용 {len(qs)}줄 ≠ 행 {len(ns)}"))
        for n, s in zip(ns, qs):
            na += 1
            if s not in L[n - 1]: bad.append(("X3", f"{p} {n}행", f"인용 '{s[:40]}' 원문에 없음"))
        qs = []
nt = 0
for t in csv.DictReader(open(os.path.join(IX, "topics.csv"), encoding="utf-8")):
    r = R[t["id"]]; nt += 1
    ok = t["키워드"] in bd.kw_hits(" ".join([r["항목"], r["식"], r["별칭"]])) or t["키워드"] in [x.strip() for x in r["키워드"].split(",")]
    if not ok: bad.append(("X4", t["id"], f"키워드 '{t['키워드']}' 에 걸리지 않음"))
O = ["# verify_step10.md — 주제 지도·조립본 자동 대조\n", "> 생성: `python3 tools/verify_step10.py`. 조립본·topics 문자열과 원장·원문만 비교.\n",
     f"- X1 L 번호 존재 / X2 Part B 상태 칸 {nb}행 / X3 Part A 인용 {na}줄 / X4 topics.csv {nt}행",
     f"- 불일치: {len(bad)}" + (" — 통과" if not bad else "")]
if bad: O += ["", "| 항목 | 위치 | 사유 |", "|---|---|---|"] + [f"| {a} | {b} | {c.replace('|', chr(92) + '|')} |" for a, b, c in bad[:200]]
open(os.path.join(IX, "verify_step10.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print("verify_step10 불일치", len(bad), "| B", nb, "A", na, "topics", nt)

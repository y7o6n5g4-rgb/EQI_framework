#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원문 문자 보존 점검 (MAPSI 정정 2026-11-03: 원문 문자 대체 금지, md 표에서는 '\\|' 이스케이프)
  C1 대체 문자   : ledger.csv 전 열에서 원문 '|' 등의 대체로 쓰일 수 있는 문자 (U+2223 ∣, U+FF5C ｜, U+00A6 ¦, U+2502 │, U+01C0 ǀ) 를 찾고,
                   같은 문자가 그 행의 인용 원문 줄에 있는지 대조 (원문에 있으면 원문 문자, 없으면 대체 의심)
  C2 식 원문     : 식 열의 ' ‖ ' 조각마다 인용 원문 줄 (앞뒤 공백·'║' 만 제거) 의 부분 문자열인지
  C3 항목 원문   : 항목이 인용 원문 줄 또는 식의 부분 문자열인지 (비고에 '항목명은 색인 부여' 인 행, 엔진 ✓ 라벨에서 '§N ' 를 뗀 행은 따로 표시)
  결과: index/check_chars.md
"""
import csv, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = {"로그": "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt", "소스": "code/eqi_unified_engine_v9_7.py",
     "v3.4": "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"}
class _L(dict):
    def __missing__(self, k):            # '문서 <경로>' (단계 5)
        b = open(os.path.join(ROOT, k[3:]), "rb").read()
        v = [l.rstrip("\r") for l in b.decode("utf-8", errors="replace").split("\n")]; self[k] = v; return v
L = _L({k: [l.rstrip("\r\n") for l in open(os.path.join(ROOT, v), encoding="utf-8", newline="").read().split("\n")] for k, v in F.items()})
SUB = {"∣": "∣ U+2223", "｜": "｜ U+FF5C", "¦": "¦ U+00A6", "│": "│ U+2502", "ǀ": "ǀ U+01C0"}
TOK = re.compile(r"(로그|소스|v3\.4|문서 \S+) (\d+)(?:–(\d+))?행(?: \(\d+행\))? \[([^\]]+)\]")
norm = lambda s: s.strip().strip("║").strip()
rows = list(csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8")))
c1, c2, c3, c3i = [], [], [], 0
for r in rows:
    cited = [norm(L[t][n - 1]) for t, a, b, _ in TOK.findall(r["근거_위치"]) for n in range(int(a), int(b or a) + 1)]
    raw = " ".join(L[t][n - 1] for t, a, b, _ in TOK.findall(r["근거_위치"]) for n in range(int(a), int(b or a) + 1))
    for col, v in r.items():
        for ch, nm in SUB.items():
            if ch in (v or ""): c1.append((r["id"], col, nm, "원문에도 있음" if ch in raw else "대체 의심"))
    for p in (r["식"].split(" ‖ ") if r["식"] else []):
        if not any(p.strip() in t for t in cited): c2.append((r["id"], p[:50]))
    it = r["항목"]
    p0 = r["식"].split(" ‖ ")[0] if r["식"] else ""
    if "(기대" in p0 and "(기대" not in it and it in p0: c3.append((r["id"], it[:60] + "  ← 식 첫 조각의 '(기대 …)' 가 빠짐")); continue
    if "항목명은 색인 부여" in r["비고"]: c3i += 1; continue
    if not (any(it in t for t in cited) or it in r["식"] or any(re.sub(r"^§\d+\s+", "", t.split("오차=")[0].replace("✓", "").strip()).startswith(it.split(" [")[0][:20]) for t in cited)):
        c3.append((r["id"], it[:60]))
O = ["# check_chars.md — 원장 원문 문자 보존 점검\n", f"> 생성: `python3 tools/check_chars.py`. 원장 {len(rows)}행.\n",
     f"- C1 대체 의심 문자 (∣ ｜ ¦ │ ǀ 가 인용 원문 줄에 없는데 원장에 있음): {sum(1 for x in c1 if x[3] == '대체 의심')}건"
     + "".join(f"; {a} {b} {c}" for a, b, c, d in c1 if d == "대체 의심"),
     f"- (참고) 같은 문자가 인용 원문 줄에도 있는 경우 = 원문 문자: {sum(1 for x in c1 if x[3] != '대체 의심')}건"
     + ("" if not any(x[3] != "대체 의심" for x in c1) else " — " + ", ".join(sorted({f"{a} {c}" for a, b, c, d in c1 if d != "대체 의심"}))),
     f"- C2 식 조각이 인용 원문의 부분 문자열이 아님: {len(c2)}건" + ("" if not c2 else " — " + "; ".join(f"{a} '{b}'" for a, b in c2)),
     f"- C3 항목이 원문·식에 없음: {len(c3)}건 (색인 부여 항목명 {c3i}행 제외)" + ("" if not c3 else "\n" + "\n".join(f"  - {a}: {b}" for a, b in c3)), ""]
open(os.path.join(ROOT, "index", "check_chars.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print(f"C1(대체 의심) {sum(1 for x in c1 if x[3] == '대체 의심')} / 원문 문자 {sum(1 for x in c1 if x[3] != '대체 의심')} / C2 {len(c2)} / C3 {len(c3)} (색인 부여 {c3i})"); [print(" ", x) for x in c3[:40]]

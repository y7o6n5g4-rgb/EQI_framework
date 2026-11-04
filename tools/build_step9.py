#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 9 — 충돌·중복·후보 목록 정리 (MAPSI 지시 2026-11-03: 판정 없이 현재 목록을 그대로 정리만) → index/step9_lists.md
  읽기만: index/conflicts.csv, index/ledger.csv, index/coverage.md, index/dup_candidates_2to5.md, index/step7_pair_candidates.md, index/reclass_candidates.md.
  coverage 보충 열 (위치 계산만, 연결·판정 아님): v3.4 머리글마다 그 머리글 행 ~ 다음 같은·상위 수준 머리글 앞 행 범위 안에
    근거 토큰 ('v3.4 N행' / 'v3.4 N–M행') 을 가진 원장 행 수 (등록 / 정의·원칙).
"""
import csv, os, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IX = os.path.join(ROOT, "index")
esc = lambda x: x.replace("|", "\\|")
rows = list(csv.DictReader(open(os.path.join(IX, "ledger.csv"), encoding="utf-8")))
CF = list(csv.DictReader(open(os.path.join(IX, "conflicts.csv"), encoding="utf-8")))
cov = open(os.path.join(IX, "coverage.md"), encoding="utf-8").read().split("\n")
first = lambda p, pat: next((m.group(1) for l in open(os.path.join(IX, p), encoding="utf-8") for m in [re.search(pat, l)] if m), "?")
n_dup = first("dup_candidates_2to5.md", r"후보 쌍 (\d+)")
n_pair = first("step7_pair_candidates.md", r"원천 행: (\d+)")
n_recl = first("reclass_candidates.md", r"걸린 행: \*\*(\d+)\*\*")
link = [r for r in rows if "연결 후보:" in r["비고"]]

# coverage A 표
A = []
for l in cov:
    m = re.match(r"^\| (\d+) \| (#+) \| (.*) \| (.*) \|$", l)
    if m: A.append((int(m.group(1)), len(m.group(2)), m.group(3), m.group(4).strip()))
nV = int(re.search(r"\((\d+)행, 머리글", "\n".join(cov)).group(1))
TOK = re.compile(r"(?:^| / )v3\.4 (\d+)(?:–(\d+))?행")
cites = []
for r in rows:
    for m in TOK.finditer(r["근거_위치"]): cites.append((int(m.group(1)), int(m.group(2) or m.group(1)), r))
cnt = {}
for i, (ln, lv, t, ids) in enumerate(A):
    end = next((A[j][0] - 1 for j in range(i + 1, len(A)) if A[j][1] <= lv), nV)
    rs = {r["id"]: r for a, b, r in cites if a <= end and b >= ln}
    cnt[ln] = (end, sum(1 for r in rs.values() if r["행_종류"] == "등록"), sum(1 for r in rs.values() if r["행_종류"] != "등록"))
blank = [x for x in A if not x[3]]
# 엔진 B 표 · §P 블록
Bl = [l for l in cov if re.match(r"^\| §\d+ \|", l)]
B_blank = [l.split("|")[1].strip() for l in Bl if not l.rstrip(" |").split("|")[-1].strip()]

st = Counter(r["상태"] for r in CF)
O = ["# step9_lists.md — 단계 9 충돌·중복·후보 목록 정리 (판정 없음)\n",
     "> 생성: `python3 tools/build_step9.py`. MAPSI 지시 2026-11-03: 판정 없이 현재 충돌·중복·후보 목록을 그대로 정리만. 원장·충돌 목록은 바꾸지 않음.\n",
     "## 1. 목록 현황\n",
     "| 목록 | 파일 | 건수 | 처리 상태 |", "|---|---|---|---|",
     f"| 충돌 (같은 항목 다른 값·상태) | index/conflicts.csv | {len(CF)} (K0001~K{len(CF):04d}) | 열림 {st['']} / 상태 열 기록 {len(CF) - st['']} (해소 사유) |",
     f"| 중복 후보 (단계 2~5 등록 행 쌍) | index/dup_candidates_2to5.md | {n_dup} 쌍 | 보고만 — 병합 안 함 |",
     f"| 연결 후보 (단계 6, 비고 '연결 후보:') | index/ledger.csv 비고 열 | {len(link)} 행 | 합치지 않음 |",
     f"| 단계 7 짝 후보 (이름으로만 적힌 정정) | index/step7_pair_candidates.md | 원천 {n_pair} 행 | perplexity 확인 후 확정 쌍만 반영 |",
     f"| 정의·원칙 행 O·I 표기 (재분류 후보) | index/reclass_candidates.md | {n_recl} 행 | 원장 반영 미결정 |",
     f"| v3.4 머리글 원장 ID 빈칸 (coverage A) | index/coverage.md | {len(blank)} / {len(A)} 머리글 | 빈칸 = 연결 안 함 (미등록 판정 아님) |",
     f"| 엔진 § 원장 ID 빈칸 (coverage B) | index/coverage.md | {len(B_blank)} / {len(Bl)} § | — |", "",
     "## 2. 충돌 목록 (conflicts.csv 그대로, 항목 앞 70자)\n", "| K | 항목 | 원장_ID | 상태 |", "|---|---|---|---|"]
for r in CF:
    O.append(f"| {r['id']} | {esc(r['항목'][:70])} | {esc(r['원장_ID']) or '—'} | {esc(r['상태'][:60]) or '열림'} |")
O += ["", "## 3. v3.4 머리글 — 원장 ID 빈칸 목록과 범위 안 근거 토큰 수 (위치 계산만)\n",
      "> 범위 = 머리글 행 ~ 다음 같은·상위 수준 머리글 앞 행. 근거 토큰 수는 '그 범위 줄을 근거로 든 원장 행 수' 이며 연결·누락 판정이 아니다.\n",
      "| 행 | 수준 | 제목 | 범위 | 등록 행 | 정의·원칙 행 |", "|---|---|---|---|---|---|"]
for ln, lv, t, ids in blank:
    e, a, b = cnt[ln]
    O.append(f"| {ln} | {'#' * lv} | {t} | {ln}–{e} | {a} | {b} |")
z = [x for x in blank if cnt[x[0]][1] == 0 and cnt[x[0]][2] == 0]
O += ["", f"- 빈칸 머리글 {len(blank)} 중 범위 안 근거 토큰이 하나도 없는 머리글: {len(z)} (" + ", ".join(f"{x[0]}행" for x in z) + ")"]
open(os.path.join(IX, "step9_lists.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print("충돌", len(CF), dict(st), "| 중복", n_dup, "| 연결 후보", len(link), "| 짝", n_pair, "| 재분류", n_recl, "| coverage 빈칸", len(blank), "/", len(A), "근거 0:", len(z), "| 엔진 빈칸", B_blank)

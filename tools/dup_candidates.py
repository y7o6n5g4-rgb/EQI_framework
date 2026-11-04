#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 2~5 등록 행 사이 중복 후보 보고 (MAPSI 보충 2026-11-03, 단계 9 에서 처리) — 원장은 바꾸지 않는다. → index/dup_candidates_2to5.md
  기준 = 단계 6 판 C 의 같은 항목 판정을 단계 2~5 등록 행 (L0001~L1236) 쌍에 적용:
    꼬리표 ('§R'·'[R]'·'7체[R]'·'5체[D1]'·'*'·'□') 를 뗀 뒤, (a) 한 행의 식 첫 조각 (공백 제거, 8자 이상, '=' 포함) 이 다른 행 식에 그대로 들어 있거나
    (b) 등호로 나눈 변을 2개 이상 공유하고 첫 변 (주어) 이 같으면 후보. 같은 원문 줄을 가리키는 쌍은 뺌.
  상태가 다르면 표시만 (판정 없음).
실행: python3 tools/dup_candidates.py
"""
import csv, os, re, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = importlib.util.spec_from_file_location("s4a", os.path.join(ROOT, "tools", "build_step4a.py")); s4a = importlib.util.module_from_spec(sp); sp.loader.exec_module(s4a)
LBL = re.compile(r"^\s*(?:\d체\[[A-Z0-9]+\]|§[A-Z]'?|\[[A-Z]'?\]|[*□]+)\s*")
nz = lambda s: re.sub(r"\s+", "", s)
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8"))
        if r["행_종류"] == "등록" and int(r["id"][1:]) <= 1236]
P = []
for r in rows:
    p0 = LBL.sub("", r["식"].split(" ‖ ")[0].strip().strip("*"))
    P.append((r, nz(p0), s4a.sides(p0), s4a.first_side(p0), nz(LBL.sub("", " ".join(r["식"].split(" ‖ ")).strip("*"))), r["근거_위치"].split(" / ")[0]))
pairs = []
for i in range(len(P)):
    ri, ki, si, fi, ji, li = P[i]
    for j in range(i + 1, len(P)):
        rj, kj, sj, fj, jj, lj = P[j]
        if li == lj: continue
        a = (len(ki) >= 8 and "=" in ki and ki in jj) or (len(kj) >= 8 and "=" in kj and kj in ji)
        b = len(si & sj) >= 2 and fi == fj and fi
        if (a or b) and fi == fj:
            pairs.append((ri, rj, "식 첫 조각 포함" if a else "공통 변 2개 이상: " + ", ".join(sorted(si & sj)), ri["상태"] != rj["상태"]))
O = ["# dup_candidates_2to5.md — 단계 2~5 등록 행 사이 중복 후보 (보고만, 단계 9 에서 처리)\n",
     "> 생성: `python3 tools/dup_candidates.py`. 기준: 단계 6 판 C 의 같은 항목 판정 (꼬리표 정규화 + 식 첫 조각 포함 또는 같은 첫 변 + 공통 변 2개 이상).",
     "> 단계 2~5 행은 수정하지 않음 (MAPSI 보충 (1)). 상태가 다른 쌍은 '상태 다름' 표시만 — 판정 없음.\n",
     f"- 대상: 등록 행 {len(rows)} (L0001~L1236) / **후보 쌍 {len(pairs)}** (상태 다름 {sum(1 for p in pairs if p[3])})\n",
     "| 행 A | 행 B | 근거 | A 상태 | B 상태 | A 식 (앞 50자) | B 식 (앞 50자) | A 근거 | B 근거 |", "|---|---|---|---|---|---|---|---|---|"]
esc = lambda x: x.replace("|", "\\|")
for a, b, why, d in pairs:
    O.append(f"| {a['id']} | {b['id']} | {esc(why)} | {a['상태'] or '빈칸'} | {b['상태'] or '빈칸'}{' (상태 다름)' if d else ''} | "
             f"{esc(a['식'][:50])} | {esc(b['식'][:50])} | {esc(a['근거_위치'].split(' / ')[0])} | {esc(b['근거_위치'].split(' / ')[0])} |")
open(os.path.join(ROOT, "index", "dup_candidates_2to5.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print("대상", len(rows), "후보 쌍", len(pairs), "상태 다름", sum(1 for p in pairs if p[3]))

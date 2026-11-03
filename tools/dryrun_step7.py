#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 7 (정정 반영) 시험 실행 — 원장은 바꾸지 않는다. → index/step7_dryrun_report.md
  정정 원천 행 = 원장 등록 행 중 상태가 철회·미적용이거나, 식에 정정·오기·폐기·철회·미적용 (뒤에 아님·않음·없음·하지 않 이 붙은 것 제외) 이 있는 행.
  대상 행 = 원천보다 이른 세션 (등록_세션 숫자) 의 등록 행, 아래 문자열 규칙으로만 연결 (해석 없음):
    규칙 F (같은 항목): 꼬리표 정규화 뒤 식 첫 조각 포함 (8자 이상, '=') 또는 같은 첫 변 + 공통 변 2개 이상.
    규칙 N (대상 이름): 원천 문장에서 '폐기'·'철회' (앞에 '표현'·'주장'·'가설 검증 →' 허용) 바로 앞 식 조각 (수식 기호 또는 § 포함, 3자 이상, '§N' 단독 제외)
                       을 공백 제거 뒤 그대로 포함하는 행 (대상 쪽 식에 정정 표기가 없는 행만).
    규칙 S (§ 번호): 원천 문장의 '§N … 폐기/철회' (같은 쉼표·'/'·'|' 구간) → 식이 '§N' 꼬리표로 시작하는 행.
"""
import csv, re, os, importlib.util
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = importlib.util.spec_from_file_location("s4a", os.path.join(ROOT, "tools", "build_step4a.py")); s4a = importlib.util.module_from_spec(sp); sp.loader.exec_module(s4a)
LBL = re.compile(r"^\s*(?:\d체\[[A-Z0-9]+\]|§[A-Z]'?|\[[A-Z]'?\]|[*□]+)\s*")
nz = lambda s: re.sub(r"\s+", "", s)
rows = list(csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8")))
CORR = re.compile(r"(정정|오기|폐기|철회|미적용)(?!\s*(?:아님|않음|없음|하지 않))")
OBJ = re.compile(r"([^\s,/(:→\"'|]*[=×÷^²³⁴§φ√/·][^\s,/(:→\"'|]*)\s*(?:표현\s*|주장\s*|가설\s*검증\s*→\s*)?(?:폐기|철회)")
def sess(r):
    m = re.search(r"\d+", r["등록_세션"] or ""); return int(m.group(0)) if m else None
reg = [r for r in rows if r["행_종류"] == "등록"]
src = [r for r in reg if r["상태"] in ("철회", "미적용") or CORR.search(r["식"])]
def feat(t):
    t0 = LBL.sub("", t.strip().strip("*")); return nz(t0), s4a.sides(t0), s4a.first_side(t0)
F = {r["id"]: feat(r["식"].split(" ‖ ")[0]) for r in reg}
links = []   # (원천, 대상, 규칙, 근거 문자열)
for s_ in src:
    ss = sess(s_); text = " ".join(s_["식"].split(" ‖ ")); parts = s_["식"].split(" ‖ ")
    early = [t for t in reg if t is not s_ and sess(t) is not None and ss is not None and sess(t) < ss]
    jn = nz(LBL.sub("", text)); sd = set().union(*[s4a.sides(LBL.sub("", x)) for x in parts]); fl = {s4a.first_side(LBL.sub("", x)) for x in parts if "=" in x}
    for t in early:
        k, ks, fr = F[t["id"]]
        if ((len(k) >= 8 and "=" in k and k in jn) or len(ks & sd) >= 2) and fr in fl: links.append((s_, t, "F", "같은 항목"))
    for m in OBJ.finditer(text):
        ob = m.group(1).strip("*.\"'“”").strip()
        if len(ob) < 3 or re.fullmatch(r"§\d+[a-z]?", ob): continue
        for t in early:
            if nz(ob) in nz(t["식"]) and not CORR.search(t["식"]): links.append((s_, t, "N", ob))
    for m in re.finditer(r"(§\d+[a-z]?)(?![\d-])[^,/|]*?(?:폐기|철회)", text):
        sec = m.group(1)
        for t in early:
            if re.match(r"^\s*\|?\s*\**" + re.escape(sec) + r"(?![\d-])", t["식"]): links.append((s_, t, "S", sec))
seen, L = set(), []
for s_, t, ru, why in links:
    if (s_["id"], t["id"]) in seen: continue
    seen.add((s_["id"], t["id"])); L.append((s_, t, ru, why))
linked_src = {s_["id"] for s_, t, ru, why in L}
esc = lambda x: x.replace("|", "\\|")
O = ["# step7_dryrun_report.md — 단계 7 (정정 반영) 시험 실행 (원장 미반영, MAPSI 판단용)\n",
     "> 생성: `python3 tools/dryrun_step7.py`. 규칙은 스크립트 머리말 (문자열 기준, 해석 없음).\n",
     "## 합계\n",
     f"- 정정 원천 행: **{len(src)}** (상태: " + ", ".join(f"{k} {v}" for k, v in Counter(r['상태'] or '빈칸' for r in src).most_common()) + ")",
     f"- 문자열 규칙으로 이른 세션 등록 행에 연결된 원천: **{len(linked_src)}** → 대상 행 **{len({t['id'] for s_, t, ru, why in L})}** (연결 {len(L)}건, 규칙별 " + ", ".join(f"{k} {v}" for k, v in Counter(ru for _, _, ru, _ in L).items()) + ")",
     f"- 연결 없음: {len(src) - len(linked_src)}", "",
     "## 연결 (후속_정정 · 상태 반영 후보)\n", "| 대상 행 | 대상 세션·상태 | 대상 식 | ← 원천 행 | 원천 세션·상태 | 규칙 | 근거 | 원천 식 |", "|---|---|---|---|---|---|---|---|"]
for s_, t, ru, why in L:
    O.append(f"| {t['id']} | {t['등록_세션']} · {t['상태'] or '빈칸'} | {esc(t['식'][:50])} | {s_['id']} | {s_['등록_세션']} · {s_['상태'] or '빈칸'} | {ru} | {esc(why)} | {esc(s_['식'][:70])} |")
O += ["", "## 연결 없는 원천 행 (원문 그대로 앞 110자)\n", "| 원천 행 | 세션 | 상태 | 근거 파일 | 식 |", "|---|---|---|---|---|"]
for s_ in src:
    if s_["id"] not in linked_src:
        O.append(f"| {s_['id']} | {s_['등록_세션']} | {s_['상태'] or '빈칸'} | {s_['근거_파일'].split('/')[-1]} | {esc(s_['식'].replace(' ‖ ', ' / ')[:110])} |")
open(os.path.join(ROOT, "index", "step7_dryrun_report.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print("원천", len(src), "연결 원천", len(linked_src), "대상", len({t['id'] for s_, t, ru, why in L}), "연결", len(L))

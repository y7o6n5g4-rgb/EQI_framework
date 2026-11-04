#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 8 — 주제 지도 → index/topics.md, index/topics.csv  (원장 index/ledger.csv 만 읽음, 원장 수정 없음)
  키워드: (1) TASKS §6 필수 키워드 (tools/build_def.py KWS, 같은 순서) 먼저, (2) 원장 키워드 열에만 있는 그 밖 키워드.
  행 걸기 (문자열 기준, 해석 없음): 필수 키워드 = build_def.kw_hits(항목 + 식 + 별칭) 에 걸리거나 원장 키워드 열에 그 키워드가 있는 행.
                                   그 밖 키워드 = 원장 키워드 열에 있는 행.
  정렬 (MAPSI 지시): 등록 행 먼저, 정의·원칙 행 뒤. 등록 행 안에서는 CLAUDE.md 규칙 4 순서
    0 엔진 (근거_파일 code/eqi_unified_engine_v9_7*) → 1 v3.4 §11 표 근거 토큰 있음 → 2 v3.4 → 3 세션 성과정리 → 4 인수인계 → 5 그 밖 (번역·설계 등), 같은 순위는 ID 순.
  대표 근거 위치 = 정렬 앞 3개 등록 행의 주 근거 (근거_위치 첫 토큰).
"""
import csv, os, re, importlib.util
from collections import OrderedDict, Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = importlib.util.spec_from_file_location("bd", os.path.join(ROOT, "tools", "build_def.py")); bd = importlib.util.module_from_spec(sp); sp.loader.exec_module(bd)
V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
RANKN = ["엔진", "v3.4 §11 표", "v3.4", "성과정리", "인수인계", "그 밖"]
esc = lambda x: x.replace("|", "\\|")


def rank(r):
    f = r["근거_파일"]
    if f.startswith("code/eqi_unified_engine_v9_7"): return 0
    if "[§11 표]" in r["근거_위치"]: return 1
    if f == V34: return 2
    if "성과정리" in f: return 3
    if "인수인계" in f: return 4
    return 5


def span(ids):
    ids = sorted(set(ids), key=lambda x: int(x[1:]))
    out, a = [], 0
    while a < len(ids):
        b = a
        while b + 1 < len(ids) and int(ids[b + 1][1:]) == int(ids[b][1:]) + 1: b += 1
        out.append(ids[a] if a == b else f"{ids[a]}~{ids[b]}"); a = b + 1
    return ", ".join(out)


def kwcol(r): return [x.strip() for x in r["키워드"].split(",") if x.strip()]


def build(rows):
    must = list(bd.KWS)
    hits = OrderedDict((k, []) for k in must)
    for r in rows:
        h = set(bd.kw_hits(" ".join([r["항목"], r["식"], r["별칭"]]))) | set(kwcol(r))
        for k in must:
            if k in h: hits[k].append(r)
    other = OrderedDict()
    for r in rows:
        for k in kwcol(r):
            if k not in hits: other.setdefault(k, []).append(r)
    for k in sorted(other, key=lambda k: (-len(other[k]), k)): hits[k] = other[k]
    out = OrderedDict()
    for k, rs in hits.items():
        reg = sorted([r for r in rs if r["행_종류"] == "등록"], key=lambda r: (rank(r), int(r["id"][1:])))
        dfn = sorted([r for r in rs if r["행_종류"] != "등록"], key=lambda r: int(r["id"][1:]))
        out[k] = (k in must, reg, dfn)
    return out


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8")))
    T = build(rows)
    with open(os.path.join(ROOT, "index", "topics.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["키워드", "필수", "순번", "id", "행_종류", "근거_순위", "상태", "현행_상태", "주_근거"])
        for k, (m, reg, dfn) in T.items():
            for i, r in enumerate(reg + dfn, 1):
                w.writerow([k, "필수" if m else "", i, r["id"], r["행_종류"], RANKN[rank(r)] if r["행_종류"] == "등록" else "—", r["상태"], r["현행_상태"], r["근거_위치"].split(" / ")[0]])
    nm = sum(1 for k, v in T.items() if v[0]); z = [k for k, v in T.items() if v[0] and not v[1] and not v[2]]
    O = ["# topics.md — 주제 지도 (단계 8)\n",
         "> 생성: `python3 tools/build_step8.py` (원장 index/ledger.csv 만 읽음). 행 걸기·정렬 규칙은 스크립트 머리말 (문자열 기준, 해석 없음).",
         "> 등록 행 먼저 (근거 순위: 엔진 → v3.4 §11 표 → v3.4 → 성과정리 → 인수인계 → 그 밖), 정의·원칙 행은 뒤에 ID 만. 전체 행 목록은 index/topics.csv.",
         "> '걸린 행 없음' 은 문자열 기준 결과이며 미등록 판정이 아니다.\n",
         "## 합계\n",
         f"- 필수 키워드 (TASKS §6): {nm}개 — 등록 행이 걸린 키워드 {sum(1 for k, v in T.items() if v[0] and v[1])}, 정의·원칙 행만 걸린 키워드 {sum(1 for k, v in T.items() if v[0] and not v[1] and v[2])}, 걸린 행 없음 {len(z)}" + (f" ({', '.join(z)})" if z else ""),
         f"- 그 밖 키워드 (원장 키워드 열): {len(T) - nm}개", "",
         "## 필수 키워드\n", "| 키워드 | 등록 | 정의·원칙 | 대표 근거 위치 (등록 앞 3행) | 등록 행 ID | 정의·원칙 행 ID |", "|---|---|---|---|---|---|"]
    def line(k, reg, dfn):
        rep = " / ".join(f"{r['id']} {r['근거_위치'].split(' / ')[0].replace('문서 성과정리문서/', '')}" for r in reg[:3]) or "—"
        return f"| {esc(k)} | {len(reg)} | {len(dfn)} | {esc(rep)} | {span([r['id'] for r in reg]) or '—'} | {span([r['id'] for r in dfn]) or '—'} |"
    for k, (m, reg, dfn) in T.items():
        if m: O.append(line(k, reg, dfn))
    O += ["", "## 그 밖 키워드 (원장 키워드 열, 행 수 순)\n", "| 키워드 | 등록 | 정의·원칙 | 대표 근거 위치 (등록 앞 3행) | 등록 행 ID | 정의·원칙 행 ID |", "|---|---|---|---|---|---|"]
    for k, (m, reg, dfn) in T.items():
        if not m: O.append(line(k, reg, dfn))
    open(os.path.join(ROOT, "index", "topics.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
    print("필수", nm, "그 밖", len(T) - nm, "걸린 행 없음", z)
    return T


if __name__ == "__main__":
    main()

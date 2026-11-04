#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAPSI 정정 규칙 (2026-11-04) — 새 행 L6469 이후에만 적용, 앞 행 (L0001~L6468) 상태 불변.
  (1) 상태 표기 줄에 '교체 확정 대기'·'확정 대기'·'확정 전' 이 있으면 상태 '결정 대기'.
  (2) 상위 머리글 (상태_출처 [머리글 범주] / [블록 등급]) 과 대상 줄 사이에 상태 표기 없는 굵은 소제목 줄 (줄 전체가 **…** 하나) 이 끼어 있으면
      상위 머리글 상태를 넘기지 않음 → 상태 빈칸, 비고 "머리글 N행 (…) — M행 굵은 소제목 개입으로 전달 안 함".
  apply(rows, ctx): build_step2 가 ID 부여 뒤 (상태 내려받기·단계 7 전) 에 부름 — id ≥ FROM 인 등록 행만.
  impact(rows, ctx): 앞 행 전체에 적용했을 때 상태가 바뀔 행 → index/rule_change_impact.md (반영 없음).
"""
import os, re
FROM = 6469
PH = re.compile(r"교체 확정 대기|확정 대기|확정 전")
TAGF = {"로그": "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt", "소스": "code/eqi_unified_engine_v9_7.py",
        "v3.4": "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"}
SRC = re.compile(r"^(로그|소스|v3\.4|문서 (\S+?)) (\d+)행 \[([^\]]+)\] '([^']*)'")
BOLD = re.compile(r"^\*\*[^*]+\*\*$")          # 줄 전체가 굵은 글자 하나
_cache = {}


def lines(root, p):
    if p not in _cache: _cache[p] = open(os.path.join(root, p), encoding="utf-8", errors="replace").read().split("\n")
    return _cache[p]


def src_of(r):
    m = SRC.match(r["상태_출처"])
    if not m: return None
    return (m.group(2) or TAGF[m.group(1)]), int(m.group(3)), m.group(4), m.group(5)


def prim_line(r):
    m = re.match(r"(로그|소스|v3\.4|문서 (\S+?)) (\d+)", r["근거_위치"])
    return ((m.group(2) or TAGF[m.group(1)]), int(m.group(3))) if m else None


def check(r, root, tokens):
    """(규칙, 새 상태, 근거 문구) 또는 None — 판정 없이 문자열 조건만."""
    if r.get("행_종류", "등록") != "등록" or not r["상태"]: return None
    s = src_of(r)
    if not s: return None
    p, n, role, tok = s
    V = lines(root, p)
    if role in ("머리글 범주", "블록 등급"):
        pl = prim_line(r)
        if pl and pl[0] == p and pl[1] > n:
            for k in range(pl[1] - 1, n, -1):              # 대상 줄에 가장 가까운 굵은 소제목
                t = V[k - 1].strip()
                if BOLD.match(t) and not any(tokens(t)):
                    h = V[n - 1]
                    ctxm = re.search(r"\(([^()]*" + re.escape(tok) + r"[^()]*)\)", h)
                    return ("(2)", "", f"머리글 {n}행 ({ctxm.group(1) if ctxm else tok}) — {k}행 굵은 소제목 개입으로 전달 안 함")
    m = PH.search(V[n - 1])
    if m and r["상태"] != "결정 대기":
        return ("(1)", "결정 대기", f"상태 표기 줄 ({p.replace('성과정리문서/', '')} {n}행) 의 '{m.group(0)}' → 결정 대기")
    return None


def apply(rows, ctx):
    root, tokens = ctx["ROOT"], ctx["tokens"]
    done = []
    for r in rows:
        if r.get("id") and int(r["id"][1:]) >= FROM:
            c = check(r, root, tokens)
            if not c: continue
            rule, st, why = c
            r["상태"] = st
            if st == "": r["등급"] = "없음"
            r["비고"] = re.sub(r"상태 = 머리글 범주 '[^']*' \([^)]*\), 줄 자체 표기 없음; ?", "", r["비고"])
            r["비고"] = (why + "; " + r["비고"]).rstrip("; ") if rule == "(2)" else (r["비고"] + "; " if r["비고"] else "") + why
            r["상태_출처"] = r["상태_출처"] + f" → MAPSI 정정 2026-11-04 {rule}: " + why
            done.append((r["id"], rule, st))
    ctx["rule_change_done"] = done
    return done


def impact(rows, ctx):
    root, tokens, out = ctx["ROOT"], ctx["tokens"], ctx["OUT"]
    hit = []
    for r in rows:
        if r.get("id") and int(r["id"][1:]) < FROM:
            c = check(r, root, tokens)
            if c: hit.append((r, c))
    esc = lambda x: x.replace("|", "\\|")
    c1 = [x for x in hit if x[1][0] == "(1)"]; c2 = [x for x in hit if x[1][0] == "(2)"]
    def span(ids):
        ids = sorted(ids, key=lambda x: int(x[1:])); o, a = [], 0
        while a < len(ids):
            b = a
            while b + 1 < len(ids) and int(ids[b + 1][1:]) == int(ids[b][1:]) + 1: b += 1
            o.append(ids[a] if a == b else f"{ids[a]}~{ids[b]}"); a = b + 1
        return ", ".join(o) or "없음"
    O = ["# rule_change_impact.md — MAPSI 정정 규칙 (2026-11-04) 을 앞 행 전체에 적용했을 때 (반영 없음)\n",
         "> 생성: `python3 tools/build_step2.py` (tools/rule_change.py impact). 규칙은 L6469 이후 새 행에만 적용 중. 앞 행 L0001~L6468 상태는 바꾸지 않았다.",
         "> (1) 상태 표기 줄에 '교체 확정 대기'·'확정 대기'·'확정 전' → 결정 대기. (2) 상위 머리글·블록 등급 줄과 대상 줄 사이에 상태 표기 없는 굵은 소제목 줄 → 상태 빈칸.",
         "> 상태 표기 줄 = 상태_출처 에 적힌 1순위 위치 (문자열 기준). 상태_출처 형식이 다른 행 (내려받기·무등급 표기 등) 은 대상 아님.\n",
         f"- 상태가 바뀔 행: **{len(hit)}** — (1) {len(c1)} / (2) {len(c2)}",
         f"- (1) ID: {span([r['id'] for r, c in c1])}", f"- (2) ID: {span([r['id'] for r, c in c2])}", "",
         "| ID | 규칙 | 현재 상태 | 적용 시 | 근거 |", "|---|---|---|---|---|"]
    for r, (ru, st, why) in hit:
        O.append(f"| {r['id']} | {ru} | {r['상태']} | {st or '빈칸'} | {esc(why)} |")
    open(os.path.join(out, "rule_change_impact.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
    print("규칙 영향 (앞 행):", len(hit), "(1)", len(c1), "(2)", len(c2), "| 새 행 적용:", ctx.get("rule_change_done"))

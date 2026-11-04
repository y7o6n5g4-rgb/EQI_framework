#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAPSI 정정 규칙 (2026-11-04) — 생성 규칙 정정으로 전 행 적용, 전 상태는 비고에 보존
  (처음 L6469 이후에만 적용 → MAPSI 2026-11-04 지시로 앞 행 20행에도 적용: 머리글 전달 규칙이 원문에 없는 상태를 붙인 생성 오류).
  (1) 상태 표기 줄에 '교체 확정 대기'·'확정 대기'·'확정 전' 이 있으면 상태 '결정 대기'.
  (2) 상위 머리글 (상태_출처 [머리글 범주] / [블록 등급]) 과 대상 줄 사이에 상태 표기 없는 굵은 소제목 줄 (줄 전체가 **…** 하나) 이 끼어 있으면
      상위 머리글 상태를 넘기지 않음 → 상태 빈칸, 비고 "머리글 N행 (…) — M행 굵은 소제목 개입으로 전달 안 함".
  apply(rows, ctx): build_step2 가 ID 부여 뒤 (상태 내려받기·단계 7 전) 에 부름 — 전 등록 행. 비고에 '규칙 정정 전 상태: X (S119 말미 rule_change)'.
  impact(rows, ctx): 바뀐 행 전후 목록 → index/rule_change_impact.md.
"""
import os, re
FROM = 6469          # 처음 적용 범위 (기록용) — 지금은 전 행 적용
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
        if r.get("id"):
            c = check(r, root, tokens)
            if not c: continue
            rule, st, why = c
            old = r["상태"]
            r["상태"] = st
            if st == "": r["등급"] = "없음"
            r["비고"] = re.sub(r"(?:; )?상태 = 머리글 범주 '[^']*' \([^)]*\), 줄 자체 표기 없음(?=;|$)", "", r["비고"]).lstrip("; ")
            r["비고"] = (why + "; " + r["비고"]).rstrip("; ") if rule == "(2)" else (r["비고"] + "; " if r["비고"] else "") + why
            r["비고"] += f"; 규칙 정정 전 상태: {old} (S119 말미 rule_change)"
            r["상태_출처"] = r["상태_출처"] + f" → MAPSI 정정 2026-11-04 {rule}: " + why
            done.append((r, rule, old, st, why))
    ctx["rule_change_done"] = done
    return done


def impact(rows, ctx):
    out = ctx["OUT"]; done = ctx.get("rule_change_done", [])
    esc = lambda x: x.replace("|", "\\|")
    old_ = [d for d in done if int(d[0]["id"][1:]) < FROM]; new_ = [d for d in done if int(d[0]["id"][1:]) >= FROM]
    O = ["# rule_change_impact.md — MAPSI 정정 규칙 (2026-11-04) 반영 기록 (전후 상태)\n",
         "> 생성: `python3 tools/build_step2.py` (tools/rule_change.py). (1) 상태 표기 줄의 '교체 확정 대기'·'확정 대기'·'확정 전' → 결정 대기.",
         "> (2) 상위 머리글·블록 등급 줄과 대상 줄 사이에 상태 표기 없는 굵은 소제목 줄 (줄 전체가 **…** 하나) → 상태 전달 안 함 (빈칸).",
         "> 처음엔 L6469 이후만 적용 (앞 행 20행은 보고만) → MAPSI 2026-11-04: 머리글 전달 규칙의 생성 오류이므로 앞 행에도 적용. 전 상태는 각 행 비고 '규칙 정정 전 상태: …' 에 보존.\n",
         f"- 바뀐 행: {len(done)} — 앞 행 (L{FROM - 1:04d} 이전) {len(old_)} / 새 행 (L{FROM:04d} 이후) {len(new_)}", "",
         "| ID | 규칙 | 정정 전 | 정정 후 | 근거 |", "|---|---|---|---|---|"]
    for r, ru, o, st, why in done:
        O.append(f"| {r['id']} | {ru} | {o} | {st or '빈칸'} | {esc(why)} |")
    open(os.path.join(out, "rule_change_impact.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
    print("규칙 정정:", len(done), "앞 행", len(old_), "새 행", len(new_))

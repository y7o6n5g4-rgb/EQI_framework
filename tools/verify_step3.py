#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 3 대조 — v3.4 §11 표 (999~1042행) ↔ 원장 / 대응표 index/map_step3.csv → index/verify_step3.md
원장·대응표·충돌 CSV 와 v3.4 원문 행만 본다 (생성 스크립트 내부 값 미사용). 해석·재판정 없음.
  W1 표 행 전수   : §11 코드 블록 안의 비어 있지 않은 표 행 (블록 머리 '[S48~S63 추가 확정]' 제외) 이 모두 대응표에 있음
  W2 조각 원문    : 대응표 조각이 해당 v3.4 행의 부분 문자열
  W3 근거 토큰    : '근거 위치 추가' 행은 원장 행 근거_위치에 'v3.4 N행 [§11 표]' 토큰이 있음.
                    새 행은 근거_파일 = v3.4, 식 = 조각 그대로, 근거_위치 첫 토큰 = 'v3.4 N행 [§11 표]'
  W4 § 번호       : 표 행 왼쪽 열 §N 과 대상 원장 행 별칭 '엔진 §N' 이 같음 (§ 없는 블록·v9.7 행은 제외)
  W5 대응 근거    : 공통 변이 있거나 근거 메모가 있음
  W6 상태 차이    : 등급_대조 '다름' 인 행은 conflicts.csv 에 같은 원장_ID 와 v3.4 행 번호로 기록돼 있고, 원장 상태는 엔진 표기 그대로
  W7 역방향       : 원장에서 'v3.4 N행 [§11 표]' 토큰을 가진 행이 모두 대응표에 있음
실행: python3 tools/verify_step3.py
"""
import csv, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
V = [l.rstrip("\r\n") for l in open(os.path.join(ROOT, V34), encoding="utf-8").read().split("\n")]
rd = lambda n: list(csv.DictReader(open(os.path.join(ROOT, "index", n), encoding="utf-8")))
led = {r["id"]: r for r in rd("ledger.csv")}
mp, cf = rd("map_step3.csv"), rd("conflicts.csv")
T0 = next(i for i, l in enumerate(V, 1) if l.startswith("## §11."))
T1 = next(i for i, l in enumerate(V, 1) if i > T0 and l.startswith("## §"))
fence = [i for i in range(T0, T1) if V[i - 1].startswith("```")]
table = [n for n in range(fence[0] + 1, fence[1]) if V[n - 1].strip() and not V[n - 1].startswith("[S48~S63")]
res, bad = [], []
def chk(tag, ok, msg):
    if not ok: bad.append(f"{tag} {msg}")
    return ok
mapped = {int(m["v34행"]) for m in mp}
for n in table: chk("W1", n in mapped, f"v3.4 {n}행 대응표에 없음")
for m in mp:
    n, piece, rid = int(m["v34행"]), m["조각"], m["원장_ID"]
    why = []
    r = led.get(rid)
    if piece not in V[n - 1]: why.append("W2 조각이 원문 행에 없음")
    if r is None: why.append("W3 원장 ID 없음")
    else:
        tok = f"v3.4 {n}행 [§11 표]"
        if m["대응"].startswith("새 행"):
            if m["대응"] == "새 행" and r["근거_파일"] != V34: why.append("W3 새 행 근거_파일 ≠ v3.4")
            if m["대응"] == "새 행" and not r["근거_위치"].startswith(tok): why.append("W3 새 행 첫 토큰 ≠ v3.4 행")
            if m["대응"] != "새 행" and not r["근거_위치"].startswith("소스 "): why.append("W3 엔진 머리말 근거 새 행의 첫 토큰 ≠ 소스 행")
            if r["식"] != piece: why.append("W3 새 행 식 ≠ 조각")
            if tok not in r["근거_위치"]: why.append("W3 새 행에 v3.4 토큰 없음")
        elif tok not in r["근거_위치"]: why.append("W3 근거_위치에 v3.4 토큰 없음")
        ms = re.match(r"^§(\d+)\s", V[n - 1]); ma = re.match(r"^엔진 §(\d+)$", r["별칭"])
        if ms and ma and ms.group(1) != ma.group(1): why.append(f"W4 § 불일치 (§{ms.group(1)} / {r['별칭']})")
        if not m["대응"].startswith("새 행") and m["공통변"].startswith("공통 변 없음") and not m["근거"]: why.append("W5 대응 근거 없음")
        if m["등급_대조"].startswith("다름"):
            if not any(c["원장_ID"] == rid and f"{n}행" in c["위치_B"] for c in cf): why.append("W6 상태 차이가 conflicts.csv 에 없음")
    bad.extend(f"{n}행 {rid}: {w}" for w in why)
    res.append((m, why))
rev = [(rid, t) for rid, r in led.items() for t in re.findall(r"v3\.4 (\d+)행 \[§11 표\]", r["근거_위치"])]
for rid, t in rev:
    chk("W7", any(m["원장_ID"] == rid and m["v34행"] == t for m in mp), f"{rid} 의 v3.4 {t}행 토큰이 대응표에 없음")
new = [m for m in mp if m["대응"].startswith("새 행")]; add = [m for m in mp if not m["대응"].startswith("새 행")]
O = ["# verify_step3.md — 단계 3 (v3.4 §11) 대응 자동 대조\n",
     f"> 생성: `python3 tools/verify_step3.py`. 원문: `{V34}` §11 {T0}~{T1 - 1}행 (표 {fence[0]}~{fence[1]}행).",
     "> 원장 전체 원문 대조 (식·상태·등급·오차·등록_세션·ID) 는 `index/verify_step2.md` (같은 검사기, v3.4 토큰 포함).\n",
     "## 요약\n",
     f"- §11 표 행: {len(table)} / 대응표 행: {len(mp)} (근거 위치 추가 {len(add)}, 새 행 {len(new)})",
     f"- 근거 위치가 추가된 원장 행: {len({m['원장_ID'] for m in add})} / 새 원장 행: {', '.join(m['원장_ID'] for m in new)}",
     f"- 상태 다름 (conflicts 기록): {sum(m['등급_대조'].startswith('다름') for m in mp)} / 오차 다름: {sum(m['오차_대조'].startswith('다름') for m in mp)}",
     f"- **불일치: {len(bad)}**" + ("" if not bad else " — " + "; ".join(bad)), "",
     "## 대응표 행별 결과\n", "| v3.4 행 | 조각 | 원장 ID | 대응 | 공통 변 / 근거 | 등급 대조 | 오차 대조 | 결과 |", "|---|---|---|---|---|---|---|---|"]
for m, why in res:
    basis = m["공통변"] if m["공통변"] not in ("—", "") and not m["공통변"].startswith("공통 변 없음") else (m["근거"] or m["공통변"])
    O.append(f"| {m['v34행']} | {m['조각'].replace('|', '\\|')} | {m['원장_ID']} | {m['대응']} | {basis.replace('|', '\\|')} | {m['등급_대조']} | {m['오차_대조']} | {'불일치: ' + '; '.join(why) if why else '일치'} |")
open(os.path.join(ROOT, "index", "verify_step3.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print(f"표 행 {len(table)} / 대응 {len(mp)} (추가 {len(add)}, 새 {len(new)}) / 불일치 {len(bad)}", bad[:5])

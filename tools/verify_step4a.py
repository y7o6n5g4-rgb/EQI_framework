#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 4a 대조 — v3.4 본문 §12~§30 (1043~2210행) ↔ 원장 / index/map_step4a.csv → index/verify_step4a.md
원장·대응표·충돌 CSV 와 v3.4 원문 행만 본다. 규칙 판정식은 build_step4a.py 와 독립으로 다시 씀. 해석·재판정 없음.
  X1 전수     : 범위 안 상태 표기 줄 (머리글·코드 울타리 제외) 이 모두 대응표에 있음 (새 행 / 근거 추가 / 제외 사유)
  X2 새 행    : 식 조각 각각이 인용된 v3.4 줄 그대로, 근거_파일 = v3.4, 첫 근거 토큰이 대응표의 행 번호
  X3 병합     : 대상 원장 행 근거_위치에 그 v3.4 줄 토큰이 있음. 병합 근거가 (a) 식 첫 조각 포함 또는 (b) 같은 첫 변 + 공통 변 2개 이상
  X4 § 제한   : 줄이 엔진 §N (≤42) 을 적고 있으면 병합 대상은 '엔진 §N' 행뿐
  X5 블록 등급 : R3 행의 '블록 등급' 토큰 줄에 'ALL TZ'
  X6 상태 차이 : 대응표 상태_대조 '다름' 인 줄은 conflicts.csv 위치_B 에 그 줄 번호로 기록
  X7 제외 사유 : '수식 문자 없는' 제외 줄에 수식 문자 없음, '굵은 블록 표제' 제외 줄은 ** 로 시작
  X5 보충     : R3 행 비고에 블록 표제 행 번호 (MAPSI 지시, 2026-11-03)
  X8 R5       : § 현황 줄은 '§' 로 시작   X9 보류 우선 : '보류' 가 들어 있는 행의 상태 = O 보류
실행: python3 tools/verify_step4a.py [4a|4b]
실행: python3 tools/verify_step4a.py
"""
import csv, os, re, sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "4a"      # 4a: §12~§30 / 4b: 캡슐 블록 S79~S108 (2211행 ~ 끝)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
V = [l.rstrip("\r\n") for l in open(os.path.join(ROOT, V34), encoding="utf-8").read().split("\n")]
rd = lambda n: list(csv.DictReader(open(os.path.join(ROOT, "index", n), encoding="utf-8")))
led = {r["id"]: r for r in rd("ledger.csv")}; mp = rd(f"map_step{TAG}.csv"); cf = rd("conflicts.csv")
if TAG == "4a":
    S0 = next(i for i, l in enumerate(V, 1) if l.startswith("## §12."))
    E0 = next(i for i, l in enumerate(V, 1) if l.startswith("## §S79/S80")) - 1
else:
    S0 = next(i for i, l in enumerate(V, 1) if l.startswith("## §S79/S80"))
    E0 = len(V) - (1 if V[-1] == "" else 0)
STAT = re.compile(r"True Zero|Near TZ|Real Zero|대수필연|(?<![A-Za-z])TZ(?![A-Za-z(])|결정 대기|O 보류|해석 I|설계 배치|(?<![불미])확정|철회|미적용")
MATH = re.compile(r"[=≈<>∝×φ^π√/]")
nz = lambda s: re.sub(r"\s+", "", s)
SEP = " ‖ "
bad = []
def rng(s):
    a, _, b = s.partition("–"); return int(a), int(b or a)
covered = set()
for m in mp:
    a, b = rng(m["v34행"]); covered.update(range(a, b + 1))
fence = False
for n in range(S0, E0 + 1):
    l = V[n - 1]
    if l.startswith("```"): fence = not fence; continue
    if l.startswith("#"): continue
    if STAT.search(l) and n not in covered: bad.append(f"X1 v3.4 {n}행 (상태 표기 줄) 대응표에 없음")
res = []
for m in mp:
    why = []
    a, b = rng(m["v34행"])
    if m["종류"] == "제외":
        t = V[a - 1].strip()
        if "수식 문자 없는" in m["대응"] and MATH.search(t): why.append("X7 수식 문자 있음")
        if "굵은 블록 표제" in m["대응"] and not t.startswith("**"): why.append("X7 굵은 표제 아님")
        res.append((m, why)); bad.extend(f"{m['v34행']}: {w}" for w in why); continue
    r = led.get(m["원장_ID"])
    if not r: why.append("원장 ID 없음")
    else:
        toks = re.findall(r"v3\.4 (\d+)(?:–(\d+))?행(?: \(\d+행\))? \[([^\]]+)\]", r["근거_위치"])
        body = [(int(x), int(y or x)) for x, y, ro in toks if ro == "본문"]
        if m["대응"] == "새 행":
            if r["근거_파일"] != V34: why.append("X2 근거_파일 ≠ v3.4")
            if not body or body[0][0] != a: why.append("X2 첫 근거 토큰 ≠ 대응표 행")
            lines_ = [k for x, y in body for k in range(x, y + 1)]
            parts = r["식"].split(SEP)
            if [V[k - 1].strip() for k in lines_] != parts: why.append("X2 식 ≠ 인용 줄 원문")
            if m["종류"] == "R3":
                bg = [int(x) for x, y, ro in toks if ro == "블록 등급"]
                if not bg or "ALL TZ" not in V[bg[0] - 1]: why.append("X5 블록 등급 줄에 'ALL TZ' 없음")
                elif f"v3.4 {bg[0]}행" not in r["비고"]: why.append("X5 비고에 블록 표제 행 번호 없음")
            if m["종류"] == "R5" and not V[a - 1].strip().startswith("§"): why.append("X8 R5 줄이 § 로 시작하지 않음")
            if "보류" in " ".join(parts) and r["상태"] != "O 보류": why.append("X9 '보류' 줄인데 상태 ≠ O 보류")
        else:
            if not any(x <= a <= y for x, y in body): why.append("X3 원장 행에 v3.4 토큰 없음")
            secs = {int(x) for k in range(a, b + 1) for x in re.findall(r"§(\d+)(?![-\d])", V[k - 1]) if int(x) <= 42}
            if secs and r["별칭"] not in {f"엔진 §{x}" for x in secs}: why.append(f"X4 줄은 §{sorted(secs)} 를 적는데 대상 {r['별칭']}")
            k = nz(r["식"].split(SEP)[0]); jn = nz(" ".join(V[x - 1] for x in range(a, b + 1)))
            if not (m["공통"].startswith("식 첫 조각 포함") and k in jn) and not m["공통"].startswith("같은 첫 변"):
                why.append("X3 병합 근거 형식 아님")
            if m["공통"].startswith("같은 첫 변"):
                fs = m["공통"].split("'")[1]
                if fs not in jn: why.append(f"X3 첫 변 '{fs}' 줄에 없음")
        if m["상태_대조"].startswith("다름") and not any(f"{a}행" in c["위치_B"] and r["id"] in c["원장_ID"] for c in cf):
            why.append("X6 상태 차이가 conflicts.csv 에 없음")
    bad.extend(f"{m['v34행']} {m['원장_ID']}: {w}" for w in why)
    res.append((m, why))
from collections import Counter
k = Counter(m["종류"] for m in mp); d = Counter(m["대응"] for m in mp if m["종류"] != "제외")
O = [f"# verify_step{TAG}.md — 단계 {TAG} (" + ("v3.4 본문 §12~§30" if TAG == "4a" else "v3.4 캡슐 블록 S79~S108") + ") 자동 대조\n",
     f"> 생성: `python3 tools/verify_step4a.py`. 원문: `{V34}` {S0}~{E0}행. 원장 전체 원문 대조는 `index/verify_step2.md`.\n",
     "## 요약\n",
     f"- 대응표 행: {len(mp)} — 종류 {dict(k)} / 대응 {dict(d)}",
     f"- 새 원장 행: {sum(1 for m in mp if m['대응'] == '새 행')} / 근거 위치 추가 (원장 행): {len({m['원장_ID'] for m in mp if m['대응'] == '근거 위치 추가'})}",
     f"- 상태 다름: {sum(1 for m in mp if m['상태_대조'].startswith('다름'))} 대응 → conflicts.csv",
     f"- **불일치: {len(bad)}**" + ("" if not bad else " — " + "; ".join(bad[:20])), "",
     "## 대응표 행별 결과\n", "| v3.4 행 | 종류 | 원장 ID | 대응 | 표기 | 상태 대조 | 근거 / 후보 | 원문 | 결과 |", "|---|---|---|---|---|---|---|---|---|"]
for m, why in res:
    O.append("| " + " | ".join(x.replace("|", "∣") for x in [m["v34행"], m["종류"], m["원장_ID"], m["대응"], m["표기"], m["상태_대조"], m["공통"], m["원문"][:60]])
             + f" | {'불일치: ' + '; '.join(why) if why else '일치'} |")
open(os.path.join(ROOT, "index", f"verify_step{TAG}.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print(f"대응표 {len(mp)} {dict(k)} / 불일치 {len(bad)}", bad[:5])

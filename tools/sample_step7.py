#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""단계 7 표본 → index/sample_check_7.md (반영 대상 6행 + 원천 4행 = 10행 전부). 고정: 파일이 있으면 쓰지 않음 (재생성은 REGEN_SAMPLE_7=1)."""
import csv, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "index", "sample_check_7.md")
if os.path.exists(OUT) and not os.environ.get("REGEN_SAMPLE_7"): raise SystemExit("sample_check_7.md 고정 — 쓰지 않음")
C = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8"))}
mp = list(csv.DictReader(open(os.path.join(ROOT, "index", "map_step7.csv"), encoding="utf-8")))
TAGF = {"로그": "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt", "소스": "code/eqi_unified_engine_v9_7.py", "v3.4": "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"}
esc = lambda x: x.replace("|", "\\|")
def orig(tok):
    m = re.match(r"(로그|소스|v3\.4|문서 (.+?\.md)) (\d+)", tok); p = m.group(2) or TAGF[m.group(1)]
    return p, int(m.group(3)), open(os.path.join(ROOT, p), encoding="utf-8").read().split("\n")[int(m.group(3)) - 1].strip()
tg = list(dict.fromkeys(m["대상_ID"] for m in mp)); sr = list(dict.fromkeys(m["원천_ID"] for m in mp))
S = ["# sample_check_7.md — 단계 7 표본 (반영 대상 6행 + 정정 원천 4행 = 10행 전부)\n",
     "> 생성: `python3 tools/sample_step7.py`. 대조 방법: 각 행의 주 근거 원문 줄을 열어 (1) 상태 열이 등록 당시 원문 표기 그대로인지,",
     "> (2) 현행_상태가 원천 원문 낱말 그대로인지 ('폐기' 는 철회·미적용으로 바꾸지 않음), (3) 후속_정정의 원천 ID·세션·위치가 맞는지,",
     "> (4) 재확정 (L0809) 은 현행_상태 빈칸·후속_정정만인지 확인. 원문 줄은 파일에서 그대로 옮김 (md 표 안 '|' 는 '\\|').\n",
     "## 대상 6행\n", "| ID | 등록_세션 | 상태 | 현행_상태 | 주 근거 (파일·행) | 원문 줄 (앞 120자) | 후속_정정 |", "|---|---|---|---|---|---|---|"]
for i in tg:
    r = C[i]; p, n, l = orig(r["근거_위치"].split(" / ")[0])
    S.append(f"| {i} | {r['등록_세션']} | {r['상태'] or '빈칸'} | {r['현행_상태'] or '빈칸'} | {p} {n}행 | {esc(l[:120])} | {esc(r['후속_정정'])} |")
S += ["", "## 정정 원천 4행\n", "| ID | 등록_세션 | 상태 | 주 근거 (파일·행) | 원문 줄 (앞 160자) | 연결 대상 (규칙·근거) |", "|---|---|---|---|---|---|"]
for i in sr:
    r = C[i]; p, n, l = orig(r["근거_위치"].split(" / ")[0])
    to = ", ".join(f"{m['대상_ID']} ({m['규칙']}·{m['근거']})" for m in mp if m["원천_ID"] == i)
    S.append(f"| {i} | {r['등록_세션']} | {r['상태'] or '빈칸'} | {p} {n}행 | {esc(l[:160])} | {esc(to)} |")
S += ["", "## 회신 양식\n", "| ID | 상태 원문 그대로 (O/X) | 현행_상태 원문 낱말 (O/X/해당 없음) | 후속_정정 원천·위치 (O/X) | 메모 |", "|---|---|---|---|---|"]
S += [f"| {i} | | | | |" for i in tg + sr]
open(OUT, "w", encoding="utf-8").write("\n".join(S) + "\n"); print("sample_check_7.md", len(tg), len(sr))

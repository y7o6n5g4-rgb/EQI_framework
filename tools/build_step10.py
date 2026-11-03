#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 10 — 조립본 index/EQI_색인_v2.md (TASKS §5, MAPSI 지시 2026-11-03: 새 세션 AI 가 처음 읽는 문서 → 짧게)
  Part A 원칙 = 원문 줄 인용만 (파일·행, 인용 문자열은 그 줄의 부분 문자열인지 스크립트가 확인 — 아니면 멈춤). 새 문장으로 요약하지 않음.
  Part B 주제별 = 단계 8 주제 지도 (tools/build_step8.py) 의 필수 키워드마다 등록 행 앞 3행 (근거 순위 정렬 그대로), 정의·원칙 행은 ID 만.
  Part C 현황판 = 원장 상태 열·현행_상태 열 집계, 결정 대기 행, 보류·검증 대기 표기 행, 코드 최신판 (원문 인용), TZ 누적 (원문 표기),
                   v1.1 현행 32 항목 (v1.1 표기 그대로 + 원장 연결 문자열 규칙), 목록 현황 (단계 9).
  Part D 문서 지도 = index/doc_map.csv 요약.
  상태 승격·판정 없음. 원장 읽기만.
"""
import csv, os, re, importlib.util
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IX = os.path.join(ROOT, "index")
D = "성과정리문서/"
sp = importlib.util.spec_from_file_location("s8", os.path.join(ROOT, "tools", "build_step8.py")); s8 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s8)
esc = lambda x: x.replace("|", "\\|")
_cache = {}
def line(p, n):
    if p not in _cache: _cache[p] = open(os.path.join(ROOT, p), encoding="utf-8").read().split("\n")
    return _cache[p][n - 1]
def q(p, n, sub=None):
    l = line(p, n).strip()
    s = sub if sub is not None else l
    if s not in l: raise SystemExit(f"인용 불일치: {p} {n}행 — '{s[:40]}'")
    return s
short = lambda p: p.replace(D, "")
rows = list(csv.DictReader(open(os.path.join(IX, "ledger.csv"), encoding="utf-8")))
reg = [r for r in rows if r["행_종류"] == "등록"]
V34 = D + "EQI_연구지침_프로젝트용_v3_4.md"; S120 = D + "EQI_Session120_인수인계_v1.md"; S119H = D + "EQI_Session119_인수인계_v1.md"; V11 = D + "EQI_클로드용_통합색인_v1_1.md"
_v11 = open(os.path.join(ROOT, V11), encoding="utf-8").read().split("\n")
V11_CUR01 = next(i for i, l in enumerate(_v11, 1) if l.startswith("| CUR01 |")); V11_NOW11 = next(i for i, l in enumerate(_v11, 1) if l.startswith("| NOW11 |"))

O = ["# EQI 색인 v2 (조립본)\n",
     "> 새 세션 AI 첫 진입 문서. 생성: `python3 tools/build_step10.py` (원장 `index/ledger.csv` 6468행에서 조립, 2026-11-03).",
     "> 이 문서는 기록 정리다 — 상태 승격·판정·새 유도 없음 (CLAUDE.md 절대 규칙). 원칙은 원문 줄 인용, 항목은 원장 ID 로 찾아간다.",
     "> 상세: 원장 `index/ledger.csv` · 주제 지도 `index/topics.md` (`topics.csv`) · 충돌 `index/conflicts.csv` · 목록 현황 `index/step9_lists.md` · 문서 지도 `index/doc_map.csv` · 누락 점검 `index/coverage.md`.\n",
     "**읽는 법.** 원장 `상태` = 등록 당시 원문 표기 (바꾸지 않음). `현행_상태` = 후속 정정으로 바뀐 행만 (빈칸 = 상태와 같음, '폐기' 는 원문 표기 그대로). "
     "등록 행 = 상태 표기가 있는 원문 줄, 정의·원칙 행 = 상태 표기 없이 필수 키워드와 '=' / '→' 를 가진 줄 (상태 빈칸 고정).\n",
     "---\n", "## Part A. 현행 원칙 (원문 인용)\n"]
A = [("A1 공리 (최상위)", [(V34, n) for n in range(48, 58) if line(V34, n).strip() and not line(V34, n).startswith("```")]),
     ("A2 GAP — MAPSI 확정 (S118)", [(S119H, 140, None), (S119H, 146, None)]),
     ("A3 핵심 상수", [(V34, n) for n in range(63, 72) if line(V34, n).strip() and not line(V34, n).startswith("```")]),
     ("A4 상수명 분리 (v1.1 CUR01)", [(V11, V11_CUR01, None)]),
     ("A5 작업 전 필수 5줄", [(S120, 18), (S120, 19)]),
     ("A6 설계 원칙 — 격자 먼저 (MAPSI S115·S118·S119 재확인)", [(S120, 48, "**격자 핵심을 먼저 세우고 법칙은 대입값만 바꿔 올린다.**")]),
     ("A7 관측 = tip 에서 지불 (S119 W-1, W-2)", [(S120, 76, "관측은 C 가 TIP 에 닿을 때만. 그 순간 1 = h·ν_p + W 지불, 방출, 이월 장부."),
                                            (S120, 77, "관측창 = 장전이 tip 에 닿기까지의 사이클 구간.")]),
     ("A8 역할·운영 (S118~S119)", [(S120, n) for n in range(21, 28)]),
     ("A9 운영 원칙 (v3.4 §13)", [(V34, n) for n in range(1080, 1087)]),
     ("A10 판정 등급 (CLAUDE.md 용어)", [("CLAUDE.md", n) for n in range(1, len(open(os.path.join(ROOT, "CLAUDE.md"), encoding="utf-8").read().split("\n")) + 1) if line("CLAUDE.md", n).startswith("- 등급:")])]
for lab, items in A:
    O.append(f"**{lab}**\n")
    loc = []
    for it in items:
        p, n = it[0], it[1]; s = q(p, n, it[2] if len(it) > 2 else None)
        O.append(f"> {esc(s)}  ")
        loc.append(n)
    rg = f"{loc[0]}–{loc[-1]}행" if len(loc) > 1 and loc[-1] - loc[0] == len(loc) - 1 else ", ".join(f"{x}행" for x in loc)
    O.append(f"\n<sub>{short(items[0][0])} {rg}</sub>\n")

# Part B
T = s8.build(rows)
O += ["---\n", "## Part B. 주제별 항목 색인 (필수 키워드, 등록 행 앞 3행)\n",
      "> 정렬 = 근거 순위 (엔진 → v3.4 §11 표 → v3.4 → 성과정리 → 인수인계 → 그 밖) → ID. 나머지 등록 행·정의·원칙 행 전체는 `index/topics.md`.",
      "> 상태 열의 '→ X' 는 현행_상태. 정정 열 '있음' = 후속_정정 기록 있음 (원장 열 참조).\n"]
def ids_short(rs, k=20):
    ids = [r["id"] for r in rs]
    return (s8.span(ids[:k]) if len(ids) <= k else s8.span(sorted(ids, key=lambda x: int(x[1:]))[:k]) + f" 외 {len(ids) - k}")
for k, (m, rg, df) in T.items():
    if not m: continue
    O.append(f"### {k} — 등록 {len(rg)} / 정의·원칙 {len(df)}\n")
    if rg:
        O += ["| ID | 항목 (앞 60자) | 별칭·세션 | 상태 | 근거 | 정정 |", "|---|---|---|---|---|---|"]
        for r in rg[:3]:
            stt = (r["상태"] or "빈칸") + (f" → {r['현행_상태']}" if r["현행_상태"] else "")
            O.append(f"| {r['id']} | {esc(r['항목'][:60])} | {esc(r['별칭'] if r['별칭'].startswith('엔진') else r['등록_세션'])} | {stt} | {esc(short(r['근거_위치'].split(' / ')[0].replace('문서 ', '')))} | {'있음' if r['후속_정정'] else ''} |")
        O.append("")
    else:
        O.append("- 등록 행 없음 (문자열 기준, 미등록 판정 아님)")
    if df: O.append(f"- 정의·원칙 ID: {ids_short(df)}")
    O.append("")
oth = [(k, rg, df) for k, (m, rg, df) in T.items() if not m]
O += ["### 그 밖 키워드 (원장 키워드 열)\n", "| 키워드 | 등록 | 정의·원칙 | 대표 등록 행 (앞 3) |", "|---|---|---|---|"]
for k, rg, df in oth:
    O.append(f"| {esc(k)} | {len(rg)} | {len(df)} | {', '.join(r['id'] for r in rg[:3]) or '—'} |")
O.append("")

# Part C
st = Counter(r["상태"] or "빈칸" for r in reg); cu = Counter(r["현행_상태"] for r in rows if r["현행_상태"])
dec = [r for r in reg if r["상태"] == "결정 대기"]
hold = [r for r in reg if r["상태"] == "O 보류"]
WAIT = re.compile(r"판단 대기|검증 대기|재현 대기|승인 대기|확인 대기|패치 대기|구현 대기|\| 대기 \|")
wait = [r for r in reg if WAIT.search(r["식"])]
BN = re.compile(r"(?<![A-Za-z0-9_\-])B(\d{1,2})(?![0-9A-Za-z_′'])")
bnum = {}
for r in reg:
    for m in BN.finditer(r["식"]): bnum.setdefault(int(m.group(1)), []).append(r["id"])
O += ["---\n", "## Part C. 현황판\n", "### C1 원장 상태 분포 (등록 행 4140)\n",
      "| 상태 | 행 수 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in st.most_common()] + [
      "", f"- 현행_상태 (단계 7): " + ", ".join(f"{k} {v}" for k, v in cu.items()) + f" — 행 {', '.join(r['id'] + ' ' + r['현행_상태'] for r in rows if r['현행_상태'])}",
      "- 행 수는 원문 줄 수다 (같은 항목이 여러 문서에 적히면 여러 행 — 중복 후보는 Part C6). 등록 카운트가 아니다.\n",
      f"### C2 결정 대기 ({len(dec)}행, 원장 상태 '결정 대기')\n", "| ID | 등록_세션 | 항목 | 근거 |", "|---|---|---|---|"]
for r in dec:
    O.append(f"| {r['id']} | {r['등록_세션']} | {esc(r['항목'][:70])} | {esc(short(r['근거_위치'].split(' / ')[0].replace('문서 ', '')))} |")
O += ["", f"### C3 보류·B 번호\n", f"- O 보류 {len(hold)}행: {s8.span([r['id'] for r in hold])}",
      "- 식에 B 번호 표기가 있는 등록 행 (문자열 'B<숫자>', 번호별): " + "; ".join(f"B{k} {s8.span(v)}" for k, v in sorted(bnum.items())),
      f"- v1.1 NOW11 원문 ({short(V11)} {V11_NOW11}행): {esc(q(V11, V11_NOW11))}", "",
      f"### C4 검증·승인·판단 대기\n",
      f"- 원장 등록 행 식에 '판단·검증·재현·승인·확인·패치·구현 대기' 또는 표 칸 '대기' 표기: {len(wait)}행 — " + (", ".join(f"{r['id']} ({r['등록_세션']}, 상태 {r['상태'] or '빈칸'})" for r in wait) or "없음"),
      "- v1.1 현행항목 상태에 '대기' 가 있는 항목: " + ", ".join(f"{c['ID']} '{c['상태']}'" for c in csv.DictReader(open(os.path.join(ROOT, D + "EQI_클로드용_현행항목_v1_1.csv"), encoding="utf-8-sig")) if "대기" in c["상태"]) + " (C7 표).", "",
      "### C5 코드 최신판 (원문 인용)\n"]
for n in range(33, 41): O.append(q(S120, n))
O += ["", f"<sub>{short(S120)} 33–40행</sub>\n", "### C6 TZ 누적 카운트\n",
      f"- **73건** — 원문 마지막 표기: {short(D + 'EQI_Session115_인수인계_v1.md')} 4행 \"{q(D + 'EQI_Session115_인수인계_v1.md', 4, 'TZ 73건')}\", "
      f"{short(D + 'EQI_Session114_성과정리_v1.md')} 5행 \"{q(D + 'EQI_Session114_성과정리_v1.md', 5)}\". S116~S120 성과정리·인수인계에는 'TZ 누적·TZ N건' 표기 없음 (grep).",
      f"- 이전 표기 (판정 없음, 나란히 적음): v3.4 4074행 \"{q(V34, 4074)}\" / v3.4 4115행 \"{q(V34, 4115)}\".",
      f"- 원장 상태 TZ 행 수 {st['TZ']} 는 원문 줄 수 (중복 포함) 이며 TZ 카운트가 아니다.\n",
      "### C7 v1.1 현행 32 항목 → 원장 연결\n",
      "> v1.1 상태 표기 그대로. 원장 연결 = v1.1 현행내용을 ';'·'.' 로 나눈 조각 중 '=' 를 가진 8자 이상 조각을 공백 제거 뒤 그대로 포함하는 등록 행 (앞 3, 문자열 기준). 연결 없음 = 문자열 기준 결과.\n",
      "| v1.1 ID | 주제 | v1.1 상태 | 원장 연결 |", "|---|---|---|---|"]
nz = lambda s: re.sub(r"\s+", "", s)
regn = [(nz(r["식"]), r) for r in reg]
for c in csv.DictReader(open(os.path.join(ROOT, D + "EQI_클로드용_현행항목_v1_1.csv"), encoding="utf-8-sig")):
    fr = [nz(x) for x in re.split(r"[;.]\s", c["현행내용"]) if "=" in x and len(nz(x)) >= 8]
    hit = []
    for f_ in fr:
        for t, r in regn:
            if f_ in t and r["id"] not in hit: hit.append(r["id"])
    O.append(f"| {c['ID']} | {esc(c['주제'])} | {esc(c['상태'])} | {', '.join(hit[:3]) + (f' 외 {len(hit) - 3}' if len(hit) > 3 else '') or '연결 없음'} |")
s9 = open(os.path.join(IX, "step9_lists.md"), encoding="utf-8").read().split("\n")
a = s9.index("| 목록 | 파일 | 건수 | 처리 상태 |"); b = next(i for i in range(a + 2, len(s9)) if not s9[i].startswith("|"))
O += ["", "### C8 목록 현황 (단계 9, 판정 없음)\n"] + s9[a:b] + [""]

# Part D
dm = list(csv.DictReader(open(os.path.join(IX, "doc_map.csv"), encoding="utf-8")))
kc = Counter(r["분류"] for r in dm)
O += ["---\n", "## Part D. 문서 지도\n",
      f"- `index/doc_map.csv` {len(dm)} 파일: " + ", ".join(f"{k} {v}" for k, v in kc.most_common()) + ". 세션별 최종판 표는 `index/coverage.md` C 절.",
      "- 등록 표 우선 순서 (CLAUDE.md 규칙 4): 엔진 v9.7 (`code/eqi_unified_engine_v9_7.py` 머리말 표·`code/eqi_unified_engine_v9_7_terminal_mpmath140.txt`) → v3.4 §11 → v1.1 현행항목 → 세션 성과정리 → 인수인계.",
      f"- 기준 문서: v3.4 `{short(V34)}` (S108 동결, 역사 기록) · 통합색인 v1.1 `{short(V11)}` · §118 이중 공리 v2 · 번역 S116 v4 / S117 v5 재료역학 / S117 확장번역 v4 / LIGO 설계서 v4.",
      "- docx 원문은 `derived/` pandoc 변환본으로 검색 (원문 아님, 인용 때 원문 docx 이름 병기).", "",
      "### 최근 세션 최종판 (S109~S120)\n", "| 세션 | 성과정리 | 인수인계 |", "|---|---|---|"]
by = {}
for r in dm:
    m = re.fullmatch(r"S(\d+)", r["세션"] or "")
    if m and 109 <= int(m.group(1)) <= 120 and r["최종판"] == "Y" and r["종류"] in ("성과정리", "인수인계"):
        by.setdefault(int(m.group(1)), {}).setdefault(r["종류"], []).append(r["파일명"])
for s in sorted(by):
    O.append(f"| S{s} | {', '.join(by[s].get('성과정리', [])) or '—'} | {', '.join(by[s].get('인수인계', [])) or '—'} |")
O += ["", "---\n", "*EQI 색인 v2 조립본 | 2026-11-03 | 원장 L0001~L6468 · 충돌 K0001~K0095 | 기록 정리 (판정 없음)*"]
open(os.path.join(IX, "EQI_색인_v2.md"), "w", encoding="utf-8").write("\n".join(x for x in O) + "\n")
print("조립본 줄", len(O))

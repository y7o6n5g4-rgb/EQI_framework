#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정의·원칙 행 재분류 후보 검색 (MAPSI 2026-11-03 지시, 원장 반영 없음 — 단계 9 에서 결정) → index/reclass_candidates.md
  대상: 원장 행_종류 '정의·원칙' 전부. 식 (원문 줄) 에서 줄 안의 낱글자 O·I 표기를 문자열로만 찾는다 (해석 없음).
  낱글자 = 앞뒤가 영문자·숫자·'_'·'^'·'′'·'·' 가 아닌 'O' / 'I'.
  꼴 (앞 경계 → 글자 → 뒤 경계):
    A  괄호 단독         : '(O)' '[O]' '(I)' '[I]'
    B  여는 괄호 + 쉼표/대시: '(O,' '(O —' '[O 재등록]' 처럼 '(' 또는 '[' 뒤, 뒤에 ',' '—' '–' ';' 또는 공백+한글
    C  쉼표 뒤 + 닫기/대시 : ', O)' ', O —' ', I]' 처럼 ',' ';' 뒤, 뒤에 ')' ']' '—' '–' ','
    D  대시 표기          : 앞이 공백·줄 처음, 뒤가 ' —' / ' –' (예 'O — 같은 기원 근거 없음')
    E  표 칸 단독         : '| O |' '| I |' (단계 5 토큰에 이미 있음 — 남아 있으면 기록)
    F  공백 뒤 + 닫기     : '(… 별도 I)' '(… O)' — 앞이 공백, 뒤가 ')' ']'
    G  대시 뒤            : '— I (…)' '… — I' (줄 끝) — 앞이 '—'/'–', 뒤가 '(' 또는 줄 끝
    H  굵은 글자 단독     : '**I**' '**O**'
  꼴 정의는 낱글자 O·I 전수 (앞뒤 2자 문맥 빈도) 를 보고 정함 — 그 밖 낱글자 쓰임은 'I =' 'I(' '(−I)' '⟨I⟩' 'H₂O' 'O(G' '[G]~[J] 절 표지 안 [I]' 등.
  제외: 'I =' 'O =' 'I(' 처럼 글자 뒤에 '=' '(' 가 바로 오는 변수 쓰임은 위 꼴에 걸리지 않음 (뒤 경계 조건).
"""
import csv, os, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = r"(?<![A-Za-z0-9_^′·'])"; R = r"(?![A-Za-z0-9_^′·'])"
PAT = [("A", re.compile(r"[(\[]\s*([OI])\s*[)\]]")),
       ("B", re.compile(r"[(\[]\s*([OI])" + R + r"(?=\s*[,—–;]|\s+[가-힣])")),
       ("C", re.compile(r"[,;]\s*([OI])" + R + r"(?=\s*[)\]—–,])")),
       ("D", re.compile(r"(?:^|(?<=\s))([OI])(?=\s+[—–])")),
       ("E", re.compile(r"\|\s*([OI])\s*\|")),
       ("F", re.compile(r"(?<=\s)([OI])(?=\s*[)\]])")),
       ("G", re.compile(r"[—–]\s*([OI])(?=\s*(?:\(|$))")),
       ("H", re.compile(r"\*\*([OI])\*\*"))]
rows = list(csv.DictReader(open(os.path.join(ROOT, "index", "ledger.csv"), encoding="utf-8")))
D = [r for r in rows if r["행_종류"] == "정의·원칙"]
hits = []
for r in D:
    t = r["식"]
    fs = []
    for k, p in PAT:
        for m in p.finditer(t):
            fs.append((k, m.group(1), m.start(), t[max(0, m.start() - 12): m.end() + 14]))
    if fs: hits.append((r, fs))
esc = lambda x: x.replace("|", "\\|")
ck = Counter(k for _, fs in hits for k in {f[0] for f in fs}); cl = Counter(l for _, fs in hits for l in {f[1] for f in fs})
O = ["# reclass_candidates.md — 정의·원칙 행 재분류 후보 (원장 반영 없음, 단계 9 에서 결정)\n",
     "> 생성: `python3 tools/reclass_scan.py`. MAPSI 지시 2026-11-03: L1714 처럼 정의·원칙 행에 섞인 줄 안의 O·I 표기를 전체 정의·원칙 행에서 문자열로 검색.",
     "> 원인: 단계 5·6 정의·원칙 판정의 상태 토큰 (tools/build_step5.py DEC_RE) 은 '(O)'·'(I)'·'| O |'·'| I |'·'O 보류'·'해석 I' 만 상태 표기로 봄 →",
     "> '(0.13 %, O — …)' · '(O, 0.38 %)' · '[O 재등록]' 같은 꼴은 상태 표기 없음으로 판정되어 정의·원칙 행이 됨. 꼴 정의는 스크립트 머리말.",
     "> 이 목록은 후보일 뿐이다. 낱글자 O·I 가 상태 표기인지 (예: 변수·항목 기호) 는 판정하지 않았다.\n",
     "## 첫 등록 사례\n",
     "- L1714 (EQI_S117_플랑크_양자에너지_확장번역_v4.md 195행) '(0.13 %, O — 같은 기원 근거 없음)' — 단계 7 MAPSI 확정 쌍으로 현행_상태 '철회'. 상태 열 빈칸 그대로. 재분류는 단계 9.\n",
     "## 합계\n",
     f"- 정의·원칙 행: {len(D)}",
     f"- O·I 표기 꼴이 걸린 행: **{len(hits)}** (글자별 행 수: " + ", ".join(f"{k} {v}" for k, v in sorted(cl.items())) + ")",
     "- 꼴별 행 수: " + ", ".join(f"{k} {v}" for k, v in sorted(ck.items())), "",
     "## 목록\n", "| ID | 근거 (주) | 꼴 | 글자 | 걸린 부분 | 식 (앞 110자) |", "|---|---|---|---|---|---|"]
for r, fs in hits:
    O.append(f"| {r['id']} | {esc(r['근거_위치'].split(' / ')[0].replace('문서 성과정리문서/', ''))} | {', '.join(dict.fromkeys(f[0] for f in fs))} | "
             f"{', '.join(dict.fromkeys(f[1] for f in fs))} | {esc(' / '.join(dict.fromkeys(f[3] for f in fs)))} | {esc(r['식'][:110])} |")
open(os.path.join(ROOT, "index", "reclass_candidates.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print("정의·원칙", len(D), "후보 행", len(hits), dict(ck), dict(cl))

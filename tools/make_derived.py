#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derived/ — 원문 docx 의 텍스트 변환본 (pandoc, gfm, --wrap=none). 원문이 아니다 (CLAUDE.md 규칙 1·3 보충).
  대상: doc_map.csv 의 분류 sessions·foundation 중 형식 docx 인 파일 전부.
  산출: derived/<원문 파일명 확장자 제외>.md  +  derived/README.md (원문 경로·sha256·pandoc 버전·행 수)
  인용: 'derived/<파일>.md N행 (원문 docx 변환본: 성과정리문서/<원문>.docx)'. 행 번호는 변환본 기준.
실행: python3 tools/make_derived.py (저장소 루트). 같은 pandoc 버전이면 결과 동일.
"""
import csv, hashlib, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "derived")
os.makedirs(OUT, exist_ok=True)
ver = subprocess.run(["pandoc", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "index", "doc_map.csv"), encoding="utf-8"))
        if r["형식"] == "docx" and r["분류"] in ("sessions", "foundation")]
R = ["# derived/ — 원문 docx 변환본 (원문 아님)\n",
     f"- 생성: `python3 tools/make_derived.py` / {ver} / `pandoc -f docx -t gfm --wrap=none`",
     "- 원문은 `성과정리문서/` 의 docx 그대로 (읽기 전용). 이 폴더의 행 번호는 변환본 기준이며, 색인에서 인용할 때는 원문 docx 파일명을 함께 적는다.",
     "- 변환 대상: doc_map.csv 분류 sessions·foundation 의 docx 전부. 그 밖의 docx (Framework v3~v17 등 역사 문서) 는 아직 변환하지 않음.\n",
     "| 변환본 | 원문 | 원문 sha256 | 변환본 행 수 | 최종판 |", "|---|---|---|---|---|"]
for r in sorted(rows, key=lambda x: x["파일명"]):
    src = os.path.join(ROOT, r["상대경로"])
    dst = os.path.join(OUT, os.path.splitext(r["파일명"])[0] + ".md")
    subprocess.run(["pandoc", "-f", "docx", "-t", "gfm", "--wrap=none", src, "-o", dst], check=True)
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()
    n = open(dst, encoding="utf-8").read().count("\n")
    R.append(f"| `derived/{os.path.basename(dst)}` | `{r['상대경로']}` | {h[:16]}… | {n} | {r['최종판']} |")
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
print(len(rows), "files")

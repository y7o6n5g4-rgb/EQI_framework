#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 2 원장 전체 재점검 — index/ledger.csv 각 행의 근거_위치 원문 행을 직접 읽어 자동 대조 → index/verify_step2.md
원장 생성 스크립트(build_step2.py)의 내부 값은 쓰지 않는다. 원장 CSV 문자열과 원문 파일만 본다.

검사 (행마다):
  V1 근거 행 존재   : 근거_위치의 모든 토큰 행이 원문 파일 범위 안에 있고 빈 줄이 아님
  V2 식 원문 일치   : 식 열을 ' ‖ ' 로 나눈 각 조각이 인용된 원문 행(앞뒤 공백·'║' 제거) 의 부분 문자열
  V3 여러 줄 완전성 : [항목]·[본문]·[표] 범위 토큰의 모든 줄이 식 조각으로 들어 있음 (첫 줄은 항목 라벨 부분)
  V4 상태·등급·오차 : 주 원문 줄 ([✓] 또는 [항목]) 의 등급 표기·오차 문자열과 원장 상태·등급·오차 대조
                      그 밖 ([머리말]·[본문]) 은 원장 상태의 원문 표기가 인용 행에 있는지
  V5 상태 올림 없음 : 원장 상태가 원문 표기에서 옮긴 값보다 높지 않음 (V4 결과에서 판정)
  V6 등록_세션      : 값이 있으면 인용 행 (원문 해당 줄·블록 표제) 의 부분 문자열. 없으면 비고에 '추정' 또는 빈칸 허용
  V7 ID             : L 번호 유일·연속, id_registry.csv 와 일치
실행: python3 tools/verify_step2.py (저장소 루트)
"""
import csv, os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = "code/eqi_unified_engine_v9_7.py"
LOG = "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt"
SEP = " ‖ "
def lines(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", newline="") as f:
        return [l.rstrip("\r\n") for l in f.read().split("\n")]
F = {"로그": lines(LOG), "소스": lines(ENG)}
FN = {"로그": LOG, "소스": ENG}
norm = lambda s: s.strip().strip("║").strip()

TOK = re.compile(r"(로그|소스) (\d+)(?:–(\d+))?행(?: \(\d+행\))? \[([^\]]+)\]")
CK = re.compile(r"^\s*✓\s+(.*?)\s*(?:오차=(\S+))?\s*(?:★\s*(.+?))?\s*$")
GRADE_MAP = [("TZ(대수필연)", "대수필연", "대수필연"), ("True Zero(대수필연)", "대수필연", "대수필연"),
             ("Near True Zero", "Near TZ", "Near TZ"), ("Near TZ", "Near TZ", "Near TZ"),
             ("True Zero", "TZ", "TZ"), ("Real Zero", "Real Zero", "Real Zero"),
             ("대수필연", "대수필연", "대수필연"), ("TZ", "TZ", "TZ")]
RANK = {"": 0, "미적용": 0, "Real Zero": 1, "Near TZ": 2, "TZ": 3, "대수필연": 4}
WORD = {"TZ": ["True Zero", "TZ"], "Near TZ": ["Near TZ", "Near True Zero"], "Real Zero": ["Real Zero"],
        "대수필연": ["대수필연"], "미적용": ["미적용"]}
GPH = re.compile(r"(True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|탐색 완료\(NTZ\)|True Zero|Real Zero|대수필연|미적용)")
SESS_RE = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*")
SESSW = re.compile(r"Session \d+(?:~\d+)?")
def map_grade(label):
    label = (label or "").strip()
    if label in ("정수", "-", "") or label.startswith(("통과", "귀속 검토", "탐색 완료")): return "", "없음"
    for k, st, gr in GRADE_MAP:
        if label.startswith(k): return st, gr
    return "", "없음"

LEDGER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "index", "ledger.csv")   # 자기 시험용 경로 인자
OUTMD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "index", "verify_step2.md")
rows = list(csv.DictReader(open(LEDGER, encoding="utf-8")))
reg = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "index", "id_registry.csv"), encoding="utf-8"))}
res = []
for r in rows:
    why = []
    toks = [(t, int(a), int(b or a), role) for t, a, b, role in TOK.findall(r["근거_위치"])]
    if not toks: why.append("V1 근거_위치 토큰 없음")
    texts = []
    for t, a, b, role in toks:
        for n in range(a, b + 1):
            if n < 1 or n > len(F[t]) or not F[t][n - 1].strip():
                why.append(f"V1 {t} {n}행 없음/빈 줄")
            else:
                texts.append((t, n, role, norm(F[t][n - 1])))
    alltext = [x[3] for x in texts]
    parts = r["식"].split(SEP) if r["식"] else []
    # V2
    for p_ in parts:
        if not any(p_.strip() in tx for tx in alltext):
            why.append(f"V2 식 조각이 원문에 없음: '{p_[:40]}'")
    # V3
    for t, a, b, role in toks:
        if role in ("항목", "본문", "표") and b > a:
            for n in range(a + 1, b + 1):
                tx = norm(F[t][n - 1])
                if tx and tx not in [p_.strip() for p_ in parts]:
                    why.append(f"V3 {t} {n}행이 식에 없음")
    # V4/V5
    prim = toks[0] if toks else None
    exp_st = None
    if prim:
        t, a, b, role = prim
        line = F[t][a - 1]
        if role == "✓":
            m = CK.match(line)
            err, grade = (m.group(2) or ""), (m.group(3) or "").strip()
            if not grade and "[별도 출력]" in line:      # §5 → 본문 줄 등급
                body = " ".join(x[3] for x in texts if x[2] == "본문 등급")
                gm = re.search(r"★\s*(.+)$", body); grade = gm.group(1).strip() if gm else ""
            exp_st, exp_gr = map_grade(grade)
            if err and err not in r["오차"]: why.append(f"V4 오차 '{err}' 원장에 없음")
            if grade and f"★ {grade}" not in r["오차"]: why.append(f"V4 등급 표기 '★ {grade}' 원장 오차 열에 없음")
        elif role == "항목":
            fs = [x.strip() for x in re.sub(r"^\s*\[[^\]]*\]\s*", "", line).split(" | ")]
            grade = fs[4] if len(fs) >= 6 else (fs[3] if len(fs) == 5 else (fs[2] if len(fs) == 4 else ""))
            err = fs[3] if len(fs) >= 6 else ""
            exp_st, exp_gr = map_grade(grade)
            if r["별칭"].startswith("엔진 v9.7 §C-주석"): exp_st = "미적용"     # 원문 줄 '현 체제 미적용'
            if parts and parts[0] != fs[0]: why.append(f"V2 식 첫 조각 ≠ 원문 라벨 '{fs[0][:40]}'")
            if err and err not in r["오차"]: why.append(f"V4 오차 '{err}' 원장에 없음")
            if grade and f"판정 열: {grade}" not in r["오차"]: why.append(f"V4 판정 열 '{grade}' 원장 오차 열에 없음")
        if exp_st is not None:
            if r["상태"] != exp_st:
                tag = "V5 상태 올림" if RANK.get(r["상태"], 0) > RANK.get(exp_st, 0) else "V4 상태 불일치"
                why.append(f"{tag}: 원문→'{exp_st or '빈칸'}', 원장 '{r['상태'] or '빈칸'}'")
            if r["등급"] != exp_gr: why.append(f"V4 등급 불일치: 원문→'{exp_gr}', 원장 '{r['등급']}'")
        else:
            # [머리말]·[본문] 주 원문: 인용 행의 첫 등급 표기 (본문 → 머리말 등급 → 머리말 순) 로 기대 상태
            order = ["본문", "본문 등급", "머리말 등급", "머리말"]
            found = ""
            for ro in order:
                for x in texts:
                    if x[2] == ro:
                        gm = GPH.search(x[3])
                        if gm: found = gm.group(1); break
                if found: break
            exp_st, exp_gr = map_grade(found) if found != "미적용" else ("미적용", "없음")
            if r["상태"] != exp_st:
                tag = "V5 상태 올림" if RANK.get(r["상태"], 0) > RANK.get(exp_st, 0) else "V4 상태 불일치"
                why.append(f"{tag}: 원문 표기 '{found or '없음'}'→'{exp_st or '빈칸'}', 원장 '{r['상태'] or '빈칸'}'")
            q = re.findall(r"'([^']+)'", r["오차"])
            for qq in q:
                if not any(qq in tx for tx in alltext): why.append(f"V4 오차 인용 '{qq}' 원문에 없음")
            m = re.match(r"머리말 등급 표기: (\S.*?)(?: \(|$)", r["오차"])
            if m and not any(m.group(1) in tx for tx in alltext): why.append(f"V4 머리말 등급 '{m.group(1)}' 원문에 없음")
    # V6
    if r["등록_세션"]:
        exprs = {e for tx in alltext for e in SESS_RE.findall(tx)} | {e for tx in alltext for e in SESSW.findall(tx)}
        if r["등록_세션"] not in exprs:
            why.append(f"V6 등록_세션 '{r['등록_세션']}' 이 인용 행의 세션 표기와 같지 않음")
    # V7
    g = reg.get(r["id"])
    if not g: why.append("V7 등록부에 없는 ID")
    res.append((r, why))

ids = [r["id"] for r in rows]
id_ok = len(set(ids)) == len(ids) and ids == [f"L{n:04d}" for n in range(1, len(ids) + 1)] and set(ids) == set(reg)
bad = [(r, w) for r, w in res if w]
cnt = Counter(w.split()[0] for _, ws in bad for w in ws)
O = ["# verify_step2.md — 단계 2 원장 전체 자동 대조\n",
     f"> 생성: `python3 tools/verify_step2.py`. 원문: `{ENG}` ({len(F['소스'])}행), `{LOG}` ({len(F['로그'])}행).",
     "> 원장 CSV 문자열과 원문 행만 비교 (생성 스크립트 내부 값 미사용). 해석·재판정 없음.\n",
     "## 요약\n",
     f"- 원장 행: {len(rows)} (L0001~L{len(rows):04d})",
     f"- 일치: {len(rows) - len(bad)} / 불일치: {len(bad)}",
     f"- ID 검사 (유일·연속·등록부 일치): {'통과' if id_ok else '실패'}",
     "- 불일치 사유별: " + (", ".join(f"{k} {v}건" for k, v in sorted(cnt.items())) if cnt else "없음"),
     "- 검사 항목: V1 근거 행 존재 / V2 식 원문 일치 / V3 여러 줄 완전성 / V4 상태·등급·오차 / V5 상태 올림 없음 / V6 등록_세션 / V7 ID\n"]
if bad:
    O.append("## 불일치 행\n\n| id | 항목 | 사유 |\n|---|---|---|")
    for r, w in bad:
        O.append(f"| {r['id']} | {r['항목'][:50].replace('|', '∣')} | {'; '.join(w).replace('|', '∣')} |")
    O.append("")
O.append("## 행별 결과\n\n| id | 판정 | 주 원문 줄 | 상태 | 등록_세션 | 사유 |\n|---|---|---|---|---|---|")
for r, w in res:
    m = TOK.search(r["근거_위치"])
    prim = f"{m.group(1)} {m.group(2)}" + (f"–{m.group(3)}" if m.group(3) else "") + f" [{m.group(4)}]" if m else ""
    O.append(f"| {r['id']} | {'불일치' if w else '일치'} | {prim} | {r['상태'] or '—'} | {r['등록_세션'] or '—'} | {'; '.join(w).replace('|', '∣')} |")
with open(OUTMD, "w", encoding="utf-8") as f:
    f.write("\n".join(O) + "\n")
print(f"rows={len(rows)} 일치={len(rows) - len(bad)} 불일치={len(bad)} id_ok={id_ok}", dict(cnt))

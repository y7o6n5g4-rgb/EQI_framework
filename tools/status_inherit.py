#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
상태 내려받기 (CLAUDE.md 규칙 4 보충, MAPSI 2026-11-03) — tools/build_step2.py 가 ID 부여 뒤 apply() 를 부른다.

  각 원장 행에 `상태_출처` 를 채운다.
  ① 1순위 근거 (행의 주 원문) 에 상태·등급 표기가 있으면 그대로 두고, 그 표기가 적힌 위치를 상태_출처에 적는다.
     (v3.4 와 다르면 충돌 목록 유지 — L0161·L0162 등)
  ② 1순위 근거에 '명시적 무등급 표기' (판정 열 정수·'-'·통과·귀속 검토 중, 본문 '값만'·'등급 없음'·'계산 없음'·'탐색 완료'·'폐기')
     가 있으면 그것이 1순위의 표기이므로 내려받지 않는다. 상태 빈칸 유지, 상태_출처에 그 표기와 위치.
  ③ 1순위 근거에 아무 표기도 없으면 등록 표 우선 순서의 다음 원문에서 찾는다:
     v3.4 §11 (행에 붙은 §11 표 토큰) → 색인 v1.1 현행항목 csv → 세션 성과정리 최종판 (세션 번호 순) → 인수인계 최종판 (세션 번호 순).
     조건 (문자열 기준, 해석 없음): 그 줄이 행의 식 첫 조각을 공백 제거 후 그대로 포함하고, 상태 목록 표기를 가짐.
     처음 찾은 줄의 표기를 상태로 쓰고 상태_출처에 위치·표기·'(내려받기)' 를 적는다. 끝까지 없으면 빈칸 + 그 사실.
  docx 최종판 (S29·S37·S109·S110 성과정리, S111 인수인계) 은 derived/ 변환본 (tools/make_derived.py) 으로 검색하고,
  상태_출처에 변환본 경로·행 + '원문 docx 변환본' 을 적는다 (2026-11-03 MAPSI 지시).
"""
import csv, os, re

DOCS = "성과정리문서"
V11 = f"{DOCS}/EQI_클로드용_현행항목_v1_1.csv"
TOKENS = [  # (원문 표기, 상태, 등급) — 긴 것 먼저
    ("True Zero(대수필연)", "대수필연", "대수필연"), ("TZ(대수필연)", "대수필연", "대수필연"),
    ("Near True Zero", "Near TZ", "Near TZ"), ("Near TZ", "Near TZ", "Near TZ"), ("True Zero", "TZ", "TZ"),
    ("Real Zero", "Real Zero", "Real Zero"), ("대수필연", "대수필연", "대수필연"), ("TZ", "TZ", "TZ"),
    ("결정 대기", "결정 대기", "없음"), ("O 보류", "O 보류", "없음"), ("해석 I", "해석 I", "없음"), ("설계 배치", "설계 배치", "없음"),
    ("확정", "확정", "없음"), ("철회", "철회", "없음"), ("미적용", "미적용", "없음"),
]
TOKMAP = {t: (s, g) for t, s, g in TOKENS}
TOK_RE = re.compile("|".join(
    (r"(?<![A-Za-z])TZ(?![A-Za-z(])" if t == "TZ" else
     r"(?<![불미])확정" if t == "확정" else re.escape(t)) for t, _, _ in TOKENS))
GRADE_RE = re.compile("|".join(
    (r"(?<![A-Za-z])TZ(?![A-Za-z(])" if t == "TZ" else re.escape(t)) for t, _, g in TOKENS if g != "없음"))
def last_grade(line):
    ms = list(GRADE_RE.finditer(line)); return ms[-1].group(0) if ms else None
NOGRADE = ["값만", "등급 없음", "계산 없음", "탐색 완료", "폐기"]
SEP = " ‖ "
nz = lambda s: re.sub(r"\s+", "", s)

BADENC = set()
def _lines(root, rel):
    # 일부 원문 (S50 성과정리 v1) 에 UTF-8 아닌 바이트가 섞여 있음 → 대체 문자로 읽음 (줄 번호 = '\n' 기준, grep -n 과 같음)
    b = open(os.path.join(root, rel), "rb").read()
    try: t = b.decode("utf-8")
    except UnicodeDecodeError:
        t = b.decode("utf-8", errors="replace"); BADENC.add(rel)
    return [l.rstrip("\r") for l in t.split("\n")]

def pick_token(line, key=None):
    ms = list(TOK_RE.finditer(line))
    if not ms: return None
    if key:
        pos = nz(line).find(key)
        # 공백 제거 위치 → 원문 위치 근사: 원문에서 key 의 마지막 글자 뒤 첫 표기
        after = [m for m in ms if len(nz(line[:m.start()])) >= pos + len(key)]
        if after: return after[0].group(0)
        return ms[-1].group(0)
    return ms[0].group(0)

def tiers(root):
    dm = list(csv.DictReader(open(os.path.join(root, "index", "doc_map.csv"), encoding="utf-8")))
    sess = lambda r: int(r["세션"][1:]) if r["세션"] else 0
    def path_of(r):     # docx 최종판은 derived/ 변환본 (tools/make_derived.py) 으로 검색
        if r["형식"] == "md": return r["상대경로"]
        d = os.path.join("derived", os.path.splitext(r["파일명"])[0] + ".md")
        return d if r["형식"] == "docx" and os.path.exists(os.path.join(root, d)) else None
    fin = [dict(r, 상대경로=path_of(r)) for r in dm if r["분류"] == "sessions" and r["최종판"] == "Y" and path_of(r)]
    so = sorted([r for r in fin if r["종류"] == "성과정리"], key=sess)
    ho = sorted([r for r in fin if r["종류"] == "인수인계"], key=sess)
    skipped = sorted(r["파일명"] for r in dm if r["분류"] == "sessions" and r["최종판"] == "Y" and not path_of(r)
                     and r["종류"] in ("성과정리", "인수인계"))
    return [("v1.1 현행항목", [V11])], [("성과정리", [r["상대경로"] for r in so]), ("인수인계", [r["상대경로"] for r in ho])], skipped

def apply(rows, ctx):
    root, FILES, render = ctx["ROOT"], ctx["FILES"], ctx["render"]
    TAG = ctx["TAG"]
    t_v11, t_sess, skipped = tiers(root)
    cache = {}
    def L(rel):
        if rel not in cache: cache[rel] = _lines(root, rel)
        return cache[rel]
    changed, nograde, none = [], [], []
    for r in rows:
        if r.get("상태_출처") and r["_key"].startswith("S5:"): continue     # 단계 5 행은 build_step5 가 상태_출처를 적음
        cites = r["_cites"]
        prim = cites[0]
        f, a, b, role = prim
        ptxt = [FILES[f][n - 1] for n in range(a, b + 1)]
        loc = lambda c: f"{TAG[c[0]]} {c[1]}행" + (f"–{c[2]}행" if c[2] != c[1] else "") + f" [{c[3]}]"
        if r["상태"]:
            # ① 표기 위치: 주 원문 역할 우선, 그다음 같은 행의 등급 역할 토큰
            src = None
            if r["상태"] == "O 보류":          # 보류 우선 행 (build_step4a): '보류' 표기 줄
                for c in cites:
                    for n in range(c[1], c[2] + 1):
                        if "보류" in FILES[c[0]][n - 1]: src = f"{TAG[c[0]]} {n}행 [{c[3]}] '보류'"; break
                    if src: break
            if src is None and role == "항목":
                fs = [x.strip() for x in re.sub(r"^\s*\[[^\]]*\]\s*", "", ptxt[0]).split(" | ")]
                g = fs[4] if len(fs) >= 6 else (fs[3] if len(fs) == 5 else (fs[2] if len(fs) == 4 else ""))
                if r["별칭"].startswith("엔진 v9.7 §C-주석"): g = "미적용"
                if g: src = f"{TAG[f]} {a}행 [{role}] '{g}'"     # 판정 열은 항목 첫 줄
            if src is None:
                for c in cites:
                    if c[3] not in (role, "✓", "본문 등급", "머리말 등급", "머리말", "본문", "§11 표", "블록 등급"): continue
                    if c[3] == "§11 표" and role != "§11 표": continue
                    if c[3] == "본문" and c[0] != f: continue           # 덧붙인 v3.4 본문 토큰은 다른 파일 주 원문 행의 상태 근거가 아님
                    for n in range(c[1], c[2] + 1):
                        t = FILES[c[0]][n - 1]
                        m = re.search(r"★\s*(.+?)\s*$", t) if c[3] == "✓" else None
                        if c[3] == "✓" and not m: continue          # ✓ 줄은 ★ 등급만 (라벨 속 문구 제외)
                        tok = m.group(1) if m else (last_grade(t) or pick_token(t))
                        if tok:
                            src = f"{TAG[c[0]]} {n}행 [{c[3]}] '{tok}'"; break
                    if src: break
            if src is None and r["상태"] == "O 보류":
                for c in cites:
                    for n in range(c[1], c[2] + 1):
                        if "보류" in FILES[c[0]][n - 1]: src = f"{TAG[c[0]]} {n}행 [{c[3]}] '보류'"; break
                    if src: break
            if src is None: raise SystemExit(f"상태 표기 위치를 찾지 못함: {r['id']}")
            r["상태_출처"] = src + " — 1순위"
            continue
        # 빈칸: ② 명시적 무등급 표기?
        note = None
        if role == "항목":
            fs = [x.strip() for x in re.sub(r"^\s*\[[^\]]*\]\s*", "", ptxt[0]).split(" | ")]
            g = fs[4] if len(fs) >= 6 else (fs[3] if len(fs) == 5 else (fs[2] if len(fs) == 4 else ""))
            if g: note = g
        if note is None:
            for t in ptxt:
                for w in NOGRADE:
                    if w in t: note = w; break
                if note: break
        if note is None and role == "머리말" and "탐색 완료" in " ".join(ptxt): note = "탐색 완료(NTZ)"
        if note is not None:
            r["상태_출처"] = f"1순위 표기 '{note}' ({loc(prim)}) — 상태 목록 밖 표기, 내려받기 미적용"
            nograde.append(r["id"]); continue
        # ③ 내려받기
        key = nz(r["식"].split(SEP)[0])
        hit = None
        if len(key) >= 6:
            for c in cites:                                # v3.4 §11 토큰
                if c[3] != "§11 표": continue
                t = FILES[c[0]][c[1] - 1]
                if key in nz(t):
                    tok = pick_token(t, key)
                    if tok: hit = (f"v3.4 {c[1]}행 [§11 표]", tok, "v3.4 §11"); break
            if not hit:
                for tier, files in t_v11 + t_sess:
                    for rel in files:
                        for n, t in enumerate(L(rel), 1):
                            if key in nz(t):
                                tok = pick_token(t, key)
                                if tok:
                                    hit = (f"문서 {rel} {n}행", tok, tier + (" (원문 docx 변환본)" if rel.startswith("derived/") else "")); break
                        if hit: break
                    if hit: break
        if hit:
            st, gr = TOKMAP[hit[1]]
            r["상태"], r["등급"] = st, gr
            r["상태_출처"] = f"{hit[0]} '{hit[1]}' (내려받기: 1순위 표기 없음 → {hit[2]})"
            r["비고"] = (r["비고"] + "; " if r["비고"] else "") + f"상태 내려받기 (규칙 4 보충): {hit[2]}"
            changed.append((r["id"], r["항목"], st, r["상태_출처"]))
        else:
            r["상태_출처"] = ("1순위 표기 없음 — 다음 순위 (v3.4 §11·v1.1 현행항목·성과정리·인수인계 최종판 md) 에 같은 식 문자열 + 상태 표기 줄 없음 (grep)"
                          if len(key) >= 6 else "1순위 표기 없음 — 식 첫 조각이 6자 미만이라 문자열 검색 생략")
            none.append(r["id"])
    ctx["inherit"] = dict(changed=changed, nograde=nograde, none=none, skipped=skipped, badenc=sorted(BADENC))

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
  V10 상태 열 불변  : 단계 7 은 상태 열을 바꾸지 않음 — 원장 상태 = index/step7_status_snapshot.csv (단계 7 직전) 의 상태
  V11 현행_상태     : 값이 있으면 상태와 다르고, 허용 목록 (상태 목록 + '폐기(원문 표기)') 안이며,
                      후속_정정에 '단계 7: 현행_상태 '<값>' ← L<원천>' (또는 MAPSI 확정 쌍의 원문 파일 줄) 이 있고, 그 원천 행의 식 (또는 그 줄) 에 그 낱말이 원문 그대로 있음.
                      정의·원칙 행은 MAPSI 확정 쌍일 때만
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
V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
class _LazyF(dict):        # '문서 <경로>' 토큰 (단계 5 세션 문서·derived 변환본) 은 필요할 때 읽음 (UTF-8 아닌 바이트는 대체 문자)
    def __missing__(self, k):
        if not k.startswith("문서 "): raise KeyError(k)
        b = open(os.path.join(ROOT, k[3:]), "rb").read()
        v = [l.rstrip("\r") for l in b.decode("utf-8", errors="replace").split("\n")]
        self[k] = v; return v
F = _LazyF({"로그": lines(LOG), "소스": lines(ENG), "v3.4": lines(V34)})
FN = {"로그": LOG, "소스": ENG, "v3.4": V34}
norm = lambda s: s.strip().strip("║").strip()

TOK = re.compile(r"(로그|소스|v3\.4|문서 [^\[\]]+?\.md) (\d+)(?:–(\d+))?행(?: \(\d+행\))? \[([^\]]+)\]")
CK = re.compile(r"^\s*✓\s+(.*?)\s*(?:오차=(\S+))?\s*(?:★\s*(.+?))?\s*$")
GRADE_MAP = [("TZ(대수필연)", "대수필연", "대수필연"), ("True Zero(대수필연)", "대수필연", "대수필연"),
             ("Near True Zero", "Near TZ", "Near TZ"), ("Near TZ", "Near TZ", "Near TZ"),
             ("True Zero", "TZ", "TZ"), ("Real Zero", "Real Zero", "Real Zero"),
             ("대수필연", "대수필연", "대수필연"), ("TZ", "TZ", "TZ")]
RANK = {"": 0, "미적용": 0, "O 보류": 0, "Real Zero": 1, "Near TZ": 2, "TZ": 3, "대수필연": 4}
WORD = {"TZ": ["True Zero", "TZ"], "Near TZ": ["Near TZ", "Near True Zero"], "Real Zero": ["Real Zero"],
        "대수필연": ["대수필연"], "미적용": ["미적용"]}
GPH = re.compile(r"(True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|탐색 완료\(NTZ\)|True Zero|Real Zero|대수필연|미적용)")
SESS_RE = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*")
SESSW = re.compile(r"Session ?\d+(?:~\d+)?")
LISTTOK = {"확정": "확정", "철회": "철회", "미적용": "미적용", "결정 대기": "결정 대기", "O 보류": "O 보류", "해석 I": "해석 I", "설계 배치": "설계 배치"}
STAT2 = re.compile(r"True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|True Zero|Real Zero|대수필연|(?<![A-Za-z])TZ(?![A-Za-z(])"
                   r"|결정 대기|O 보류|해석 I|설계 배치|(?<![불미])확정|철회|미적용")
GRD2 = re.compile(r"True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|True Zero|Real Zero|대수필연|(?<![A-Za-z])TZ(?![A-Za-z(])")
def body_token(t):
    g = GRD2.findall(t)
    if g: return g[-1]
    a = STAT2.findall(t); return a[0] if a else ""
SESSTOK = {"O": "O 보류", "(O)": "O 보류", "보류": "O 보류", "I": "해석 I", "(I)": "해석 I", "후보": ""}   # 단계 5 세션 문서 표기
def map_any(tok):
    if tok in SESSTOK: return SESSTOK[tok], "없음"
    if tok == "Near-TZ": return "Near TZ", "Near TZ"
    return (LISTTOK[tok], "없음") if tok in LISTTOK else map_grade(tok)
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
BYID = {r["id"]: r for r in rows}
_snp = os.path.join(ROOT, "index", "step7_status_snapshot.csv")
SNAP7 = {r["id"]: r["상태"] for r in csv.DictReader(open(_snp, encoding="utf-8"))} if os.path.exists(_snp) else None
CUR_OK = {"확정", "대수필연", "TZ", "Near TZ", "Real Zero", "O 보류", "해석 I", "결정 대기", "철회", "미적용", "설계 배치", "폐기"}
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
        if toks and t != toks[0][0]: continue          # 덧붙인 다른 원문의 근거 토큰 (단계 3~5 근거 위치 추가) 은 V3 대상 아님
        if role in ("항목", "본문", "표") and b > a:
            for n in range(a + 1, b + 1):
                tx = norm(F[t][n - 1])
                if tx and tx not in [p_.strip() for p_ in parts]:
                    why.append(f"V3 {t} {n}행이 식에 없음")
    # V4/V5
    prim = toks[0] if toks else None
    exp_st = None
    toks_ = toks
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
        if prim[0].startswith("문서 "):
            exp_st = None; doc_prim = True        # 상태 선택 규칙 대조는 tools/verify_step5.py (독립 재구현)
        if exp_st is not None:
            if r["상태"] != exp_st:
                tag = "V5 상태 올림" if RANK.get(r["상태"], 0) > RANK.get(exp_st, 0) else "V4 상태 불일치"
                why.append(f"{tag}: 원문→'{exp_st or '빈칸'}', 원장 '{r['상태'] or '빈칸'}'")
            if r["등급"] != exp_gr: why.append(f"V4 등급 불일치: 원문→'{exp_gr}', 원장 '{r['등급']}'")
        elif not prim[0].startswith("문서 "):
            # [머리말]·[본문] 주 원문: 인용 행의 첫 등급 표기 (본문 → 머리말 등급 → 머리말 순) 로 기대 상태
            # 주 원문 역할 먼저. 덧붙인 v3.4 §11 토큰은 주 원문일 때만 (엔진 행의 v3.4 표기 차이는 conflicts.csv 로 따로 기록)
            order = [prim[3]] + [x for x in ["본문", "본문 등급", "블록 등급", "머리말 등급", "머리말"] if x != prim[3]]
            found = ""
            for ro in order:
                if prim[0] == "v3.4" and ro in ("본문", "블록 등급"):     # v3.4 본문 (단계 4a): 줄들 합친 문자열의 마지막 등급, 없으면 첫 상태 표기
                    jt = " ".join(x[3] for x in texts if x[2] == ro and x[0] == "v3.4")
                    found = "보류" if "보류" in jt else body_token(jt)
                    if found: break
                    continue
                for x in texts:
                    if x[2] == ro and (x[0] == prim[0] or ro != "본문"):   # 덧붙인 다른 원문의 [본문] 근거 (단계 4~6 근거 추가) 는 상태 대조에 쓰지 않음
                        gm = GPH.findall(x[3]) if ro == "§11 표" else ([GPH.search(x[3]).group(1)] if GPH.search(x[3]) else [])
                        if gm: found = gm[-1]; break
                if found: break
            exp_st, exp_gr = (("O 보류", "없음") if found == "보류" else map_any(found)) if found != "미적용" else ("미적용", "없음")
            if r["상태"] != exp_st:
                tag = "V5 상태 올림" if RANK.get(r["상태"], 0) > RANK.get(exp_st, 0) else "V4 상태 불일치"
                why.append(f"{tag}: 원문 표기 '{found or '없음'}'→'{exp_st or '빈칸'}', 원장 '{r['상태'] or '빈칸'}'")
            q = re.findall(r"'([^']+)'", r["오차"])
            for qq in q:
                if not any(qq in tx for tx in alltext): why.append(f"V4 오차 인용 '{qq}' 원문에 없음")
            m = re.match(r"(?:머리말|v3\.4) (?:등급 )?표기: (\S.*?)(?: \(|$)", r["오차"]) or re.search(r"\(v3\.4 표기: (.+)\)$", r["오차"])
            if m and not any(m.group(1) in tx for tx in alltext): why.append(f"V4 머리말 등급 '{m.group(1)}' 원문에 없음")
    # V8 상태_출처 (CLAUDE.md 규칙 4 보충: 상태 내려받기)
    so = r.get("상태_출처", "")
    if not so: why.append("V8 상태_출처 빈칸")
    elif "(내려받기" in so:
        # 1순위 기대 상태가 빈칸일 때만 내려받기 허용 → 1순위 비교의 상태·등급 불일치 줄은 V8 로 대체
        if (exp_st or "") != "": why.append(f"V8 1순위 표기('{exp_st}') 가 있는데 내려받음")
        why[:] = [w for w in why if not re.match(r"V[45] (상태|등급)", w)]
        m = re.match(r"^(?:문서 (\S+) (\d+)행|v3\.4 (\d+)행 \[§11 표\]) '([^']+)'", so)
        if not m: why.append("V8 내려받기 위치 형식 오류")
        else:
            rel, n, tok = (m.group(1) or V34), int(m.group(2) or m.group(3)), m.group(4)
            try: t = (F["v3.4"] if rel == V34 else lines(rel))[n - 1]
            except Exception: t = ""; why.append(f"V8 {rel} {n}행 읽기 실패")
            key = re.sub(r"\s+", "", r["식"].split(SEP)[0])
            if key not in re.sub(r"\s+", "", t): why.append("V8 내려받기 줄에 식 첫 조각 없음")
            if tok not in t: why.append(f"V8 내려받기 줄에 표기 '{tok}' 없음")
            want = {"확정": "확정", "철회": "철회", "미적용": "미적용", "결정 대기": "결정 대기", "O 보류": "O 보류",
                    "해석 I": "해석 I", "설계 배치": "설계 배치"}.get(tok) or map_grade(tok)[0]
            if r["상태"] != want: why.append(f"V8 내려받은 표기 '{tok}'→'{want}' ≠ 원장 '{r['상태']}'")
            # 1순위에 아무 표기도 없어야 함
            pt = " ".join(x[3] for x in texts if x[2] == (prim[3] if prim else ""))
            jf = ""
            if prim and prim[3] == "항목":
                fs8 = [x.strip() for x in re.sub(r"^\s*\[[^\]]*\]\s*", "", F[prim[0]][prim[1] - 1]).split(" | ")]
                jf = fs8[4] if len(fs8) >= 6 else (fs8[3] if len(fs8) == 5 else (fs8[2] if len(fs8) == 4 else ""))
            if jf or GPH.search(pt) or any(w in pt for w in ["값만", "등급 없음", "계산 없음", "탐색 완료", "폐기"]):
                why.append("V8 1순위 원문에 표기가 있음 (내려받기 대상 아님)")
    elif so.startswith("정의·원칙 행"):
        pass                                           # V9 에서 검사
    elif so.startswith("1순위 표기 '"):
        w8 = so.split("'")[1]
        if w8 not in " ".join(alltext): why.append(f"V8 1순위 표기 '{w8}' 원문에 없음")
        if r["상태"]: why.append("V8 무등급 표기 행인데 상태 있음")
    elif so.startswith("1순위 표기 없음"):
        if r["상태"]: why.append("V8 표기 없음 행인데 상태 있음")
    else:
        so, rc_ = (so.split(" → MAPSI 정정 2026-11-04 ")[0], so.split(" → MAPSI 정정 2026-11-04 ")[1]) if " → MAPSI 정정 2026-11-04 " in so else (so, "")
        if rc_ and "규칙 정정 전 상태: " not in r["비고"]: why.append("V8 MAPSI 정정 규칙 (2026-11-04) 행인데 비고에 정정 전 상태 없음")
        m = re.match(r"^(로그|소스|v3\.4|문서 [^\[\]]+?\.md) (\d+)행 \[([^\]]+)\] '([^']+)' — 1순위$", so)
        if not m: why.append("V8 1순위 상태_출처 형식 오류")
        else:
            t = F[m.group(1)][int(m.group(2)) - 1]
            if m.group(4) not in t: why.append(f"V8 상태_출처 줄에 표기 '{m.group(4)}' 없음")
            if not any(c[0] == m.group(1) and c[1] <= int(m.group(2)) <= c[2] for c in toks): why.append("V8 상태_출처 줄이 근거_위치에 없음")
            tk = m.group(4); want = "미적용" if tk == "미적용" else ("O 보류" if tk == "보류" else map_any(tk)[0])
            if rc_.startswith("(1)"):          # MAPSI 정정 (1): 표기 줄의 '교체 확정 대기'·'확정 대기'·'확정 전' → 결정 대기
                if not re.search(r"교체 확정 대기|확정 대기|확정 전", t): why.append("V8 정정 (1) 인데 표기 줄에 '확정 대기'·'확정 전' 없음")
                want = "결정 대기"
            elif rc_.startswith("(2)"):        # MAPSI 정정 (2): 머리글과 대상 줄 사이 굵은 소제목 → 전달 안 함
                bm = re.search(r"— (\d+)행 굵은 소제목 개입으로 전달 안 함", rc_); pr = toks[0]
                if not bm or not (int(m.group(2)) < int(bm.group(1)) < pr[1]) or not re.match(r"^\*\*[^*]+\*\*$", F[m.group(1)][int(bm.group(1)) - 1].strip()):
                    why.append("V8 정정 (2) 굵은 소제목 줄 위치·형식 불일치")
                if bm and bm.group(0)[2:] not in r["비고"]: why.append("V8 정정 (2) 비고 문구 없음")
                want = ""
            if want != r["상태"]: why.append(f"V8 상태_출처 표기 '{tk}'→'{want}' ≠ 원장 '{r['상태']}'")
    # V9 정의·원칙 행 (MAPSI 승인 2026-11-03): 상태 빈칸 고정·내려받기 금지·상태 표기 없는 줄·'=' 또는 '→'·비고 키워드가 원문 줄에 있음
    kind = r.get("행_종류", "")
    if kind not in ("등록", "정의·원칙"): why.append(f"V9 행_종류 '{kind}' (등록 / 정의·원칙 아님)")
    if kind == "정의·원칙":
        if r["상태"]: why.append("V9 정의·원칙 행인데 상태 있음")
        if r["등급"] != "없음": why.append("V9 정의·원칙 행인데 등급 있음")
        if not so.startswith("정의·원칙 행 — 상태 빈칸 고정"): why.append("V9 정의·원칙 행 상태_출처 문구 아님 (내려받기 금지)")
        pl = alltext[0] if alltext else ""
        if STAT2.search(pl) or GRD2.search(pl): why.append("V9 정의·원칙 줄에 상태 표기 있음")
        if "=" not in pl and "→" not in pl: why.append("V9 정의·원칙 줄에 '=' / '→' 없음")
        km = re.search(r"걸린 키워드: ([^;]+)", r["비고"])
        if not km: why.append("V9 비고에 걸린 키워드 없음")
    elif so.startswith("정의·원칙 행"): why.append("V9 등록 행인데 정의·원칙 상태_출처")
    # V10 상태 열 불변 (단계 7) / V11 현행_상태 (MAPSI 결정 2026-11-03 방식 (나))
    if SNAP7 is None: why.append("V10 step7_status_snapshot.csv 없음")
    elif SNAP7.get(r["id"]) != r["상태"]: why.append(f"V10 상태 '{r['상태']}' ≠ 단계 7 직전 '{SNAP7.get(r['id'])}'")
    cur = r.get("현행_상태", None)
    if cur is None: why.append("V11 현행_상태 열 없음")
    elif cur:
        if cur == r["상태"]: why.append("V11 현행_상태가 상태와 같음 (빈칸이어야 함)")
        if cur not in CUR_OK: why.append(f"V11 현행_상태 '{cur}' 허용 목록 밖")
        fsrc = re.findall(r"현행_상태 '" + re.escape(cur) + r"' ← ((?:code|성과정리문서)/\S+) (\d+)행 \(MAPSI 확정 쌍", r["후속_정정"])
        if kind != "등록" and not fsrc: why.append("V11 정의·원칙 행에 현행_상태 (MAPSI 확정 쌍 아님)")
        srcs = re.findall(r"현행_상태 '" + re.escape(cur) + r"' ← (L\d{4})", r["후속_정정"])
        if not srcs and not fsrc: why.append("V11 현행_상태가 있는데 후속_정정에 원천 ID·원천 파일 줄 없음")
        for fp_, fn_ in fsrc:
            try: fl_ = open(os.path.join(ROOT, fp_), encoding="utf-8", errors="replace").read().split("\n")[int(fn_) - 1]
            except Exception: fl_ = ""
            if cur not in fl_: why.append(f"V11 원천 {fp_} {fn_}행에 '{cur}' 없음")
        for sid in srcs:
            sr = BYID.get(sid)
            if not sr: why.append(f"V11 원천 {sid} 원장에 없음")
            elif cur not in sr["식"]: why.append(f"V11 원천 {sid} 식에 '{cur}' 없음")
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
     "- 검사 항목: V1 근거 행 존재 / V2 식 원문 일치 / V3 여러 줄 완전성 / V4 상태·등급·오차 / V5 상태 올림 없음 / V6 등록_세션 / V7 ID / V8 상태_출처 (1순위 표기 위치·내려받기 줄·무등급 표기) / V9 정의·원칙 행은 상태 빈칸 (내려받기 금지·상태 표기 없음·키워드) / V10 상태 열은 단계 7 에서 바뀌지 않음 / V11 현행_상태가 있으면 상태와 다르고 후속_정정에 원천 ID 있음\n"]
if bad:
    O.append("## 불일치 행\n\n| id | 항목 | 사유 |\n|---|---|---|")
    for r, w in bad:
        O.append(f"| {r['id']} | {r['항목'][:50].replace('|', '\\|')} | {'; '.join(w).replace('|', '\\|')} |")
    O.append("")
O.append("## 행별 결과\n\n| id | 판정 | 주 원문 줄 | 상태 | 등록_세션 | 사유 |\n|---|---|---|---|---|---|")
for r, w in res:
    m = TOK.search(r["근거_위치"])
    prim = f"{m.group(1)} {m.group(2)}" + (f"–{m.group(3)}" if m.group(3) else "") + f" [{m.group(4)}]" if m else ""
    O.append(f"| {r['id']} | {'불일치' if w else '일치'} | {prim} | {r['상태'] or '—'} | {r['등록_세션'] or '—'} | {'; '.join(w).replace('|', '\\|')} |")
with open(OUTMD, "w", encoding="utf-8") as f:
    f.write("\n".join(O) + "\n")
print(f"rows={len(rows)} 일치={len(rows) - len(bad)} 불일치={len(bad)} id_ok={id_ok}", dict(cnt))

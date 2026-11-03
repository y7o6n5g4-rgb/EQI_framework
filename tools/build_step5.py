#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 5 (세션 문서 S109~S119) — TASKS_index_v3.md §4 단계 5
  tools/build_step2.py 가 단계 4 hook 뒤, ID 부여 전에 apply() 를 부른다. 단독 실행하지 않는다.
  대상 순서: 성과정리 최종판 (S109~S119) → 인수인계 최종판 (S109~S119) → 번역·설계 문서 (doc_map 분류 translations).
  docx 최종판 (S109·S110 성과정리, S111 인수인계) 은 derived/ 변환본 (원문 docx 파일명을 비고에).

행 단위 규칙 (문자열 기준, 해석 없음) — 줄마다 index/map_step5.csv:
  T1 표기 줄   : 상태 표기가 있는 줄 = 한 행. 표 행·목록 항목은 그대로, 문단·코드 줄은 수식 문자가 있을 때.
                 상태 표기 = MAPSI 결정 표기 (철회·미적용·보류·O 보류·'(O)'·표 칸 'O'·해석 I·'(I)'·표 칸 'I'·결정 대기·설계 배치·확정)
                 + 등급 표기 (True Zero·TZ·Near TZ·Real Zero·대수필연). 부정 꼬리 ('… 표기 안 함', '… 아님', '… 금지', '… 불가', '… 하지 않') 가 붙은 표기는 뺌.
  T2 등급만 줄 : 상태 표기·수치·★·괄호·[S…] 외 글자가 없는 줄은 앞 줄과 한 행 (단계 4 R2 와 같은 판정).
  T3 머리글 범주: 자체 표기가 없는 표 행·목록 항목은 가장 가까운 ##/### 머리글이 범주 표기 하나 (설계 배치 / 관찰 (O) / 해석 (I) / 철회 / 미적용 /
                 결정 대기 / 확정) 를 가지면 그 상태. 비고에 머리글 행 번호. 범주가 둘 이상인 머리글 (예 '관찰 (O) · 철회 기록') 은 넘기지 않음.
                 '확정' 머리글은 'MAPSI' 가 함께 적힌 것만 (예 '확정 결정 (MAPSI S116)'). '확정 서지'·'확정 설계'·'실행 전 확정안' 등은 넘기지 않음.
  T4 제외      : 머리글, 표 머리 행·구분 행, 상태 표기 없는 줄 (T3 해당 없음), 수식 없는 문단 표기 줄.
  T5 범주 둘   : 자체 표기 없는 표 행·목록 항목인데 머리글 범주가 둘 이상 (예 '관찰 (O) · 철회 기록') → 행으로 넣고 상태는 빈칸 (MAPSI 보충 2026-11-03).
                 표 머리에 '처리'·'상태'·'판정' 칸이 있으면 그 칸 글을 1순위 표기로 상태_출처에 원문 그대로 인용 (상태 목록 표기가 아니므로 상태는 빈칸).
                 비고 "머리글 N행 범주 둘 (…) — 상태 미전달". 다음 순위 내려받기 하지 않음.
  상태 선택    : 한 행에 표기가 여럿이면 MAPSI 결정 표기가 등급보다 우선 — 철회 > 미적용 > O 보류 > 해석 I > 결정 대기 > 설계 배치 > 확정,
                 결정 표기가 없으면 마지막 등급 표기 (단계 4 와 같음), 그것도 없으면 확정. 나머지 표기는 비고.
                 O·I 는 상태 목록 이름 (O 보류·해석 I) 으로 적고, 원문 표기는 상태_출처에 그대로 인용.
  같은 항목    : 단계 4 규칙에 더해, 식 첫 조각 포함인 경우에도 첫 변 (주어) 이 같아야 함 (표 행 'X-15 | E_p/(h·ν_p) | We = φ^(11/4)' 이
                 'We=φ^(11/4)' 행으로 붙던 문제). § 제한 같음. 새 행 없이 근거 위치만 추가, 상태 다르면 충돌.
  후보·정의    : '후보' 가 든 행은 상태 빈칸 (규칙 2, 상태_출처에 '후보'). 등급 정의 줄 ('**TZ** = 오차 < …') 과 등급 언급 ('TZ 임계', 'TZ 기준', 'TZ 카운트', 'TZ급') 은 표기로 보지 않음.
  등록_세션    : 줄 → 머리글 → 문서 제목 줄 (1행) 의 세션 표기 원문 그대로 ('S119', 'Session 119' 등).
"""
import csv, os, re

SEP = " ‖ "
DEC = [("철회", "철회"), ("미적용", "미적용"), ("O 보류", "O 보류"), ("보류", "O 보류"), ("(O)", "O 보류"), ("O", "O 보류"),
       ("해석 I", "해석 I"), ("(I)", "해석 I"), ("I", "해석 I"), ("결정 대기", "결정 대기"), ("설계 배치", "설계 배치"), ("확정", "확정")]
DEC_ORDER = ["철회", "미적용", "O 보류", "해석 I", "결정 대기", "설계 배치", "확정"]
NEG = r"(?!\s*(?:표기\s*)?(?:안 함|안함|아님|금지|불가|하지 않|않음|없음))(?!\s*(?:임계|기준|카운트|급\b|급 ))"   # 부정 꼬리, 등급 언급 (임계·기준·카운트·급)
DEC_RE = re.compile(r"(철회|미적용|O 보류|(?<![가-힣A-Za-z])보류|\(O\)|(?<=\| )O(?= \|)|해석 I|\(I\)|(?<=\| )I(?= \|)|결정 대기|설계 배치|(?<![불미])확정)" + NEG)
GRD_RE = re.compile(r"(True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near[- ]TZ|True Zero|Real Zero|대수필연|(?<![A-Za-z-])TZ(?![A-Za-z(]))" + NEG)
HEADCAT = [("설계 배치", "설계 배치"), ("관찰 (O)", "O 보류"), ("(O)", "O 보류"), ("해석 (I)", "해석 I"), ("(I)", "해석 I"),
           ("철회", "철회"), ("미적용", "미적용"), ("결정 대기", "결정 대기"), ("확정", "확정")]
MATH = re.compile(r"[=≈<>∝×φ^π√/]")
SESS = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*|Session ?\d+(?:~\d+)?")
GMAP = {"True Zero(대수필연)": ("대수필연", "대수필연"), "TZ(대수필연)": ("대수필연", "대수필연"), "Near True Zero": ("Near TZ", "Near TZ"),
        "Near TZ": ("Near TZ", "Near TZ"), "Near-TZ": ("Near TZ", "Near TZ"), "True Zero": ("TZ", "TZ"), "Real Zero": ("Real Zero", "Real Zero"), "대수필연": ("대수필연", "대수필연"), "TZ": ("TZ", "TZ")}
DMAP = dict(DEC)
nz = lambda s: re.sub(r"\s+", "", s)

def tokens(t):
    d = [m.group(1) for m in DEC_RE.finditer(t)]
    g = [m.group(1) for m in GRD_RE.finditer(t)]
    return d, g

def choose(t):
    """(표기 원문, 상태, 등급, 나머지 표기들) — 상태 선택 규칙."""
    d, g = tokens(t)
    if "후보" in t:      # 규칙 2: '후보' 는 확정·TZ 로 올리지 않음 → 상태 빈칸, 원문 표기 '후보' 를 상태_출처에
        return "후보", "", "없음", list(dict.fromkeys(d + g))
    if d:
        ds = sorted(set(DMAP[x] for x in d), key=DEC_ORDER.index)
        st = ds[0]
        tok = next(x for x in d if DMAP[x] == st)
        if st == "확정" and g:          # 결정 표기가 확정뿐이면 등급 표기 우선 (단계 4 와 같음)
            tok = g[-1]; st, gr = GMAP[tok]
            return tok, st, gr, [x for x in dict.fromkeys(d + g) if x != tok]
        return tok, st, "없음", [x for x in dict.fromkeys(d + g) if x != tok]
    if g:
        tok = g[-1]; st, gr = GMAP[tok]
        return tok, st, gr, [x for x in dict.fromkeys(g) if x != tok]
    return None, "", "없음", []

def head_category(h):
    hits = []
    for w, st in HEADCAT:
        if w in h and st not in [x[1] for x in hits]:
            if w == "확정" and (re.search(r"[불미]확정", h) or "MAPSI" not in h): continue   # '확정' 머리글은 MAPSI 확정만
            hits.append((w, st))
    # '관찰 (O)' 와 '(O)' 는 같은 범주
    return hits

def is_grade_only(t):
    if "=" in t: return False
    d, g = tokens(t)
    if not (d or g): return False
    u = GRD_RE.sub("", DEC_RE.sub("", t)); u = re.sub(r"\[S\d+[^\]]*\]", "", u); u = re.sub(r"e-?\d+", "", u)
    u = re.sub(r"오차|[\d.,~+\-−×⁻⁰¹²³⁴⁵⁶⁷⁸⁹%★\[\]():：\s*]", "", u)
    return u == ""

def files_of(root, lo=109, hi=119, trans=True):
    dm = list(csv.DictReader(open(os.path.join(root, "index", "doc_map.csv"), encoding="utf-8")))
    out = []
    def pth(r):
        if r["형식"] == "md": return r["상대경로"], ""
        d = os.path.join("derived", os.path.splitext(r["파일명"])[0] + ".md")
        return d, r["상대경로"]
    for kind in ("성과정리", "인수인계"):
        sel = [r for r in dm if r["분류"] == "sessions" and r["최종판"] == "Y" and r["종류"] == kind and r["세션"]
               and lo <= int(r["세션"][1:]) <= hi]
        for r in sorted(sel, key=lambda r: int(r["세션"][1:])):
            p, src = pth(r); out.append((kind, r["세션"], p, src))
    if trans:
        for r in sorted([r for r in dm if r["분류"] == "translations"], key=lambda r: r["파일명"]):
            out.append(("번역·설계", "", r["상대경로"], ""))
    return out

RANGES = {"5": (109, 119, True), "6": (21, 108, False)}   # 단계 → (세션 시작, 끝, 번역·설계 문서 포함)

def apply(ctx, tag="5"):
    rows, add, cite, render, FILES, TAG = ctx["rows"], ctx["add"], ctx["cite"], ctx["render"], ctx["FILES"], ctx["TAG"]
    root = ctx["ROOT"]; sides, first_side = ctx["sides"], ctx["first_side"]
    PFX = f"S{tag}:"
    _LBL = re.compile(r"^\s*(?:\d체\[[A-Z0-9]+\]|§[A-Z]'?|\[[A-Z]'?\]|[*□]+)\s*")
    if tag == "6" and ctx.get("NORM6"):        # 단계 6 시험: '§R'·'[R]'·'7체[R]'·'5체[D1]' 같은 꼬리표를 변 비교 전에 뗌 (단계 2~5 결과에는 적용 안 함)
        _LBL = re.compile(r"^\s*(?:\d체\[[A-Z0-9]+\]|§[A-Z]'?|\[[A-Z]'?\]|[*□]+)\s*")
        _s0, _f0 = sides, first_side
        sides = lambda t: _s0(_LBL.sub("", t.strip().strip("*")))
        first_side = lambda t: _f0(_LBL.sub("", t.strip().strip("*")))
        ctx["_norm_sides"], ctx["_norm_first"] = sides, first_side
    out, conf, excluded, files = [], [], [], files_of(root, *RANGES[tag])
    pool = []
    for r in rows:
        if r["_key"].startswith((PFX, "DEF:")): continue   # 같은 단계 새 행·정의·원칙 행에는 병합하지 않음
        p0 = r["식"].split(SEP)[0]
        k = nz(_LBL.sub("", p0.strip().strip("*"))) if (tag == "6" and ctx.get("NORM6")) else nz(p0)
        pool.append((k, sides(p0), r, first_side(p0)))
    for kind, ses, p, src in files:
        b = open(os.path.join(root, p), "rb").read()
        V = [l.rstrip("\r") for l in b.decode("utf-8", errors="replace").split("\n")]
        FILES[p] = V; TAG[p] = f"문서 {p}"
        title = V[0] if V else ""
        heads, h = {}, None
        fence, infence = {}, False
        for n, l in enumerate(V, 1):
            if l.startswith("```"): infence = not infence; fence[n] = None; continue
            fence[n] = infence
            if re.match(r"^#{2,4} ", l): h = n
            heads[n] = h
        used = set()
        def is_table(t): return t.startswith("|") and t.endswith("|")
        def is_sep(t): return bool(re.match(r"^\|[-\s|:]+\|$", t))
        def is_list(t): return bool(re.match(r"^([-*]|\d+\.|[①-⑳])\s", t))
        def emit(lines_, tok_info, rule, hline=None, t5=None):
            a, b_ = lines_[0], lines_[-1]
            text = [V[k - 1].strip() for k in lines_]
            tok, st, gr, others = tok_info
            cites = [cite(p, a, b_, "본문")]
            hl = heads.get(a)
            if hline: cites.append(cite(p, hline, role="머리글 범주"))
            # 등록_세션: 줄 → 머리글 → 제목
            sess, sline = "", None
            for k in list(lines_) + ([hl] if hl else []) + [1]:
                m = SESS.search(V[k - 1])
                if m: sess, sline = m.group(0), k; break
            if sline and not any(c[1] <= sline <= c[2] for c in cites):
                cites.append(cite(p, sline, role="문서 제목" if sline == 1 else "머리글"))
            jn = nz(" ".join(text)); sd = set().union(*[sides(t) for t in text])
            fs_line = {first_side(t) for t in text if "=" in t}
            secs = {int(x) for x in re.findall(r"§(\d+)(?![-\d])", " ".join(text)) if int(x) <= 42}
            hits, cands = [], []
            for k, ks, r, fs_row in pool:
                ok = ((len(k) >= 8 and "=" in k and k in jn) or len(ks & sd) >= 2) and fs_row in fs_line   # 단계 5: 포함이라도 첫 변 (주어) 같아야
                if ok and secs and not any(r["별칭"] == f"엔진 §{x}" for x in secs): cands.append(r); continue
                if ok: hits.append(r)
                elif any(len(x) >= 3 for x in ks & sd): cands.append(r)
            loc = f"{a}" + (f"–{b_}" if b_ != a else "")
            srcline = (f"{TAG[p]} {hline}행 [머리글 범주] '{tok}'" if hline else
                       f"{TAG[p]} {next(k for k in lines_ if tok and tok in V[k - 1])}행 [본문] '{tok}'") if tok else ""
            rec = dict(문서=p, 행=loc, 종류=rule, 표기=tok or "", 원문=text[0][:120])
            if hits:
                bad = []
                for r in hits:
                    t_ = cite(p, a, b_, "본문")
                    if t_ not in r["_cites"]: r["_cites"].append(t_)
                    r["근거_위치"] = render(r["_cites"])
                    d_ = "같음" if (not tok or st == r["상태"]) else f"다름 ('{tok}'→{st} / 원장 {r['상태'] or '빈칸'})"
                    out.append(dict(rec, _row=r, 대응="근거 위치 추가", 상태_대조=d_, 비고=""))
                    if tok and st != r["상태"]: bad.append(r)
                if bad: conf.append(dict(rows_=bad, p=p, ln=a, lb=b_, tok=tok, st=st, text=text[0], src=src))
                return
            note = []
            if t5: note.append(f"머리글 {t5[0]}행 범주 둘 ({t5[1]}) — 상태 미전달")
            if src: note.append(f"원문 docx 변환본: {src}")
            if hline: note.append(f"상태 = 머리글 범주 '{tok}' ({TAG[p]} {hline}행), 줄 자체 표기 없음")
            if others: note.append(f"같은 행 다른 표기 병기: {', '.join(others)}")
            if tok in ("O", "I", "(O)", "(I)", "보류", "Near-TZ"): note.append(f"원문 표기 '{tok}' → 상태 목록 '{st}'")
            if tok == "후보": note.append("원문 표기 '후보' — 규칙 2 에 따라 상태 올리지 않음 (상태 빈칸)")
            if not sess: note.append("등록_세션: 줄·머리글·문서 제목에 세션 표기 없음")
            if cands and tag != "6": note.append("같은 항목 후보 (병합 안 함): " + ", ".join(sorted({c["_key"] for c in cands})[:5]))
            r = add(f"{PFX}{p}:{a}", cites, 항목=text[0][:80], 식=SEP.join(text),
                    별칭=f"{kind} {ses or os.path.basename(p)[:30]} ({p.split('/')[-1]} {a}행)", 상태=st, 등급=gr,
                    오차=(f"원문 표기: {tok}" if tok else ""), 층=ctx["layer"](" ".join(text)), 등록_세션=sess,
                    근거_파일=p, 엔진_S="엔진 원장 행 없음 (식 문자열·변 대조)", 키워드=ctx["keywords"](" ".join(text)), 비고="; ".join(note))
            r["상태_출처"] = (srcline + " — 1순위") if tok else ""
            if t5:
                r["상태_출처"] = (f"1순위 표기 '{t5[2][1]}' ({TAG[p]} {a}행, 표 '{t5[2][0]}' 칸) — 상태 목록 밖 표기, 머리글 {t5[0]}행 범주 둘 — 상태 미전달"
                               if t5[2] else f"1순위 표기 없음 — 머리글 {t5[0]}행 범주 둘 ({t5[1]}) — 상태 미전달 (내려받기 안 함)")
            if tag == "6": r["_cands6"] = [c for c in cands if c.get("행_종류", "등록") != "정의·원칙"]   # ID 부여 뒤 비고 '연결 후보: L…'
            out.append(dict(rec, _row=r, 대응="새 행", 상태_대조="—", 비고="후보: " + ", ".join(sorted({c["_key"] for c in cands})[:3]) if cands else ""))
        n = 1
        while n <= len(V):
            l = V[n - 1]; t = l.strip()
            if n in used or not t or l.startswith("#") or l.startswith("```") or (is_table(t) and is_sep(t)):
                n += 1; continue
            if is_table(t) and n < len(V) and is_sep(V[n].strip()):       # 표 머리 행
                n += 1; continue
            d, g = tokens(t)
            has = bool(d or g)
            kindl = "표" if is_table(t) else ("목록" if is_list(t) else ("코드" if fence.get(n) else "문단"))
            if has and is_grade_only(t) and kindl in ("문단", "코드"):     # T2
                k = n - 1
                while k >= 1 and V[k - 1].strip() and not V[k - 1].startswith("#") and not V[k - 1].startswith("```") \
                        and not any(tokens(V[k - 1])): k -= 1
                para = [x for x in range(k + 1, n + 1) if x not in used]
                if len(para) > 1:
                    emit(para, choose(" ".join(V[x - 1] for x in para)), "T2"); used.update(para); n += 1; continue
            if has and re.search(r"(?:TZ|대수필연|Near TZ|Real Zero)\**\s*=\s*(?:오차|기호)", t):   # 등급 정의 줄
                excluded.append((p, n, "T4 등급 정의 줄 (예: '**TZ** = 오차 < …')", t)); n += 1; continue
            if has and (kindl in ("표", "목록") or MATH.search(t)):            # T1 (+ 이어진 등급만 줄)
                lines_ = [n]; j = n + 1
                while j <= len(V) and kindl in ("문단", "코드") and V[j - 1].strip() and is_grade_only(V[j - 1].strip()):
                    lines_.append(j); j += 1
                emit(lines_, choose(" ".join(V[x - 1] for x in lines_)), "T1"); used.update(lines_); n = j; continue
            if not has and kindl in ("표", "목록") and heads.get(n):           # T3
                hc = head_category(V[heads[n] - 1])
                if len(hc) == 1:
                    w, st = hc[0]
                    emit([n], (w, st, "없음", []), "T3", hline=heads[n]); used.add(n); n += 1; continue
                if len(hc) > 1:                                              # T5: 범주 둘 머리글 아래 행 — 행으로 넣되 상태 미전달
                    hl5 = heads[n]
                    cell = None
                    if is_table(t):        # 표 머리의 '처리'·'상태'·'판정' 칸 = 행 자체의 표기 (원문 그대로 인용)
                        k = n - 1
                        while k >= 1 and is_table(V[k - 1].strip()): k -= 1
                        hdr = V[k].strip() if k < n - 1 else ""
                        hc_ = [c.strip() for c in hdr.strip("|").split("|")]
                        cc_ = [c.strip() for c in re.split(r"(?<!\\)\|", t.strip("|"))]
                        for ci, nm in enumerate(hc_):
                            if nm in ("처리", "상태", "판정", "처리 결과", "결과") and ci < len(cc_) and cc_[ci]: cell = (nm, cc_[ci]); break
                    emit([n], (None, "", "없음", []), "T5", t5=(hl5, " · ".join(x[0] for x in hc), cell)); used.add(n); n += 1; continue
            elif has:
                excluded.append((p, n, "T4 수식 없는 문단 표기 줄", t))
            n += 1
    ctx[f"step{tag}_map"], ctx[f"step{tag}_conf"], ctx[f"step{tag}_excluded"], ctx[f"step{tag}_files"] = out, conf, excluded, files

def write_map(ctx, path, tag="5"):
    cols = ["문서", "행", "종류", "원장_ID", "대응", "표기", "상태_대조", "비고", "원문"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for m in ctx[f"step{tag}_map"]:
            m = dict(m); m["원장_ID"] = m["_row"]["id"]; w.writerow(m)
        for p, n, why, t in ctx[f"step{tag}_excluded"]:
            w.writerow(dict(문서=p, 행=n, 종류="제외", 원장_ID="", 대응=why, 원문=t[:120]))

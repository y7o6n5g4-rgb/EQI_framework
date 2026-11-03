#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 4a (v3.4 본문 §12~§30, 1043~2210행) — TASKS_index_v3.md §4 단계 4 의 앞부분
  tools/build_step2.py 가 단계 3 hook 뒤, ID 부여 전에 apply() 를 부른다. 단독 실행하지 않는다.
  범위: '## §12.' 머리글 행부터 '## §S79/S80 캡슐화' 머리글 직전까지 (원문 배치 그대로 — §5b·§5c·§2b 포함, §20 머리글 없음).

행 단위 규칙 (문자열 기준, 해석 없음) — 결과는 index/map_step4a.csv 에 줄마다 기록:
  R1 등록 줄   : 상태 목록 표기 (True Zero·TZ·Near TZ·Real Zero·대수필연·확정·철회·미적용·결정 대기·O 보류·해석 I·설계 배치;
                 '불확정'·'미확정'·'NTZ' 제외) 와 수식 문자 (= ≈ < > ∝ × φ ^ π √ /) 를 함께 가진 줄 = 원장 한 행.
  R2 등급만 줄 : 표기는 있으나 수식 문자가 없는 줄 중 '오차 …%' / '★' 형태의 등급 줄은 같은 문단 (빈 줄·코드 울타리까지)
                 의 앞 줄들과 한 행 (여러 줄 항목). 문단에 수식 줄이 없으면 코드 블록 바로 위의 굵은 블록 표제 줄을 첫 줄로 포함.
  R3 블록 등급 : 굵은 블록 표제 (**…:**) 또는 코드 블록 안 '…:' 표제 줄이 'ALL TZ' / '[TZ]' / '[… TZ]' 를 가지면
                 그 블록의 수식 줄 (자체 표기 없음) 을 각각 한 행으로, 상태 = 표제 표기 (상태_출처 = 표제 줄).
  R4 제외      : 머리글 (#), 굵은 블록 표제 줄 자체, 수식 문자 없는 표기 줄 (예: '- TZ 카운트 통일'), 단계 3 범위 밖 줄.
  R5 § 현황 줄 : '§N' 으로 시작하고 상태 표기가 있는 줄은 수식 문자가 없어도 한 행 (§ 번호 현황·등록 목록, 단계 4b 에서 추가).
  R1 이어진 줄: R1 줄 바로 뒤의 '=' 시작 줄과 '오차…%' 등급만 줄 ('=' 없음) 은 같은 행 (여러 줄 항목, MAPSI 정정 (2)).
  보류 우선    : 행 문자열에 '보류' 가 있으면 상태 = O 보류 (규칙 2), 다른 표기는 비고. '오차: 2.19×10⁻⁵% ★ …' 처럼 '오차' 로 시작하고
                 '=' 없는 줄은 '×10⁻ⁿ' 이 있어도 R2 등급만 줄.
  같은 항목    : (a) 기존 원장 행의 식 첫 조각 (공백 제거, 8자 이상, '=' 포함) 이 v3.4 줄 (공백 제거) 에 그대로 들어 있거나
                 (b) 등호로 나눈 변 (공백·[…]·'§48-A:' 같은 꼬리표·상태 표기 제거) 을 2개 이상 공유하고 첫 변 (주어) 이 같으면
                 새 행 없이 그 원장 행 근거_위치에 'v3.4 N행 [본문]' 토큰만 추가. 상태 표기가 다르면 충돌 (판정 없음).
                 줄이 엔진 §N (≤42) 을 적고 있으면 그 § 의 원장 행에만 병합. 그 밖에 변 1개 (3자 이상) 공유는 '후보' 로 기록만.
  R2 등급만 줄 은 '오차' 를 포함하거나 '★'·'전 k' 로 시작하는 줄, 또는 is_grade_only() (표기·수치·★·괄호·[S…] 외에 남는 글자 없음) 인 줄. '=' 로 시작하는 줄은 앞 줄과 한 행.
  상태         : 줄의 마지막 등급 표기 (없으면 첫 상태 표기). 같은 줄에 다른 표기가 더 있으면 비고에 병기.
  등록_세션    : 줄 → 블록 표제 → § 머리글 의 [S…]/S… 표기 원문 그대로. 없으면 빈칸 + 비고.
"""
import csv, os, re

V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
SEP = " ‖ "
MATH = re.compile(r"[=≈<>∝×φ^π√/]")
STAT = re.compile(r"True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|True Zero|Real Zero|대수필연|(?<![A-Za-z])TZ(?![A-Za-z(])"
                  r"|결정 대기|O 보류|해석 I|설계 배치|(?<![불미])확정|철회|미적용")
GRADE = re.compile(r"True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|True Zero|Real Zero|대수필연|(?<![A-Za-z])TZ(?![A-Za-z(])")
SESS = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*")
GRADEONLY = re.compile(r"^\s*(오차\s|★|전 k)")      # 이 범위에서 쓰인 등급만 줄 형태 (오차 …% ★ … / 전 k(…) ALL True Zero)
nz = lambda s: re.sub(r"\s+", "", s)

def is_grade_only(t):
    """등급만 줄: 상태 표기·오차 수치·★·괄호·[S…] 를 빼면 아무것도 남지 않고 '=' 가 없는 줄 (예: '[TZ 4.34×10⁻⁹%]  [S80]', '오차: ~5×10⁻⁸%  ★ True Zero')."""
    if "=" in t or not STAT.search(t): return False
    u = STAT.sub("", t); u = re.sub(r"\[S\d+[^\]]*\]", "", u); u = re.sub(r"e-?\d+", "", u)
    u = re.sub(r"오차|[\d.,~+\-−×⁻⁰¹²³⁴⁵⁶⁷⁸⁹%★\[\]():：\s]", "", u)
    return u == ""

def status_of(t, TOKMAP):
    if "보류" in t:     # 규칙 2: 보류를 확정·TZ 로 올리지 않음 → 줄에 '보류' 가 있으면 상태 O 보류, 다른 표기는 비고
        allt = list(dict.fromkeys(STAT.findall(t)))
        return "보류", ("O 보류", "없음"), allt
    g = GRADE.findall(t)
    tok = g[-1] if g else (STAT.findall(t) or [None])[0]
    allt = list(dict.fromkeys(STAT.findall(t)))
    return tok, (TOKMAP[tok] if tok else ("", "없음")), [x for x in allt if x != tok]

LABEL_PREFIX = re.compile(r"^\s*(§\d+[-A-Za-z0-9]*:|[^=:]{1,20}:)\s*")
LEAD_SEC = re.compile(r"^\s*(§\d+[-A-Za-z0-9]*|\[[^\]]*\])\s+")
def untable(t): return t.replace("|", " ") if t.lstrip().startswith("|") else t
def sides(t):
    t = untable(t); t = LEAD_SEC.sub("", t.strip())
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = STAT.sub("", t); t = re.sub(r"★+|오차\s*\S+%|\(400자리\)", "", t)
    t = LABEL_PREFIX.sub("", t)
    out = set()
    for x in t.split("="):
        x = nz(x)
        if len(x) >= 2 and not re.fullmatch(r"[-+0-9.,%]+", x): out.add(x)
    return out

def rng_first(s): return int(str(s).split("–")[0])

def first_side(t):
    t = untable(t); t = LEAD_SEC.sub("", t.strip())
    t = re.sub(r"\[[^\]]*\]", "", t); t = LABEL_PREFIX.sub("", t)
    x = nz(t.split("=")[0]); return x

RANGES = {   # 단계 → (시작 머리글, 끝 머리글 (그 직전까지) 또는 None = 파일 끝, 원장 키 접두, 별칭 구분)
    "4a": ("## §12.", "## §S79/S80", "V34B:", "본문"),
    "4b": ("## §S79/S80", None, "V34C:", "캡슐"),
}

def apply(ctx, tag="4a"):
    rows, add, cite, render, FILES, TOKMAP = ctx["rows"], ctx["add"], ctx["cite"], ctx["render"], ctx["FILES"], ctx["TOKMAP"]
    V = FILES[V34]
    st_m, en_m, PFX, KIND = RANGES[tag]
    S0 = next(i for i, l in enumerate(V, 1) if l.startswith(st_m))
    E0 = (next(i for i, l in enumerate(V, 1) if l.startswith(en_m)) - 1) if en_m else len(V) - (1 if V[-1] == "" else 0)
    head_of, h3_of, fence_of = {}, {}, {}
    cur_h, cur_h3, in_f, f_start = None, None, False, None
    for n in range(S0, E0 + 1):
        l = V[n - 1]
        if l.startswith("## "): cur_h, cur_h3 = n, None
        if l.startswith("### "): cur_h3 = n
        head_of[n], h3_of[n] = cur_h, cur_h3
        if l.startswith("```"):
            if not in_f: in_f, f_start = True, n
            else: in_f = False
            fence_of[n] = None; continue
        fence_of[n] = f_start if in_f else None
    def bold_above(fs):
        for k in (fs - 1, fs - 2):
            if k >= S0 and re.match(r"^\*\*.*\*\*", V[k - 1].strip()): return k
        return None
    pool = []
    for r in rows:
        if r["_key"].startswith(PFX): continue          # 같은 단계 안의 새 행끼리는 병합하지 않음
        k = nz(r["식"].split(SEP)[0])
        pool.append((k, sides(r["식"].split(SEP)[0]), r, first_side(r["식"].split(SEP)[0])))
    out, conf, excluded = [], [], []
    def sess_for(lines_, extra):
        for k in list(lines_) + [x for x in extra if x] + [h3_of.get(lines_[0]), head_of.get(lines_[0])]:
            if not k: continue
            t = V[k - 1]
            m = re.search(r"\[([^\]]*S\d[^\]]*)\]", t) or re.search(r"\(([^)]*S\d[^)]*)\)", t)
            s_ = SESS.search(m.group(1)) if m else (SESS.search(t) if k in lines_ else None)
            if s_: return s_.group(0), k
        return "", None
    def layer(t):
        h = re.findall(r"([357])체", t)
        return (h[0] + "체") if len(set(h)) == 1 and len(h) == 1 else "해당 없음"
    def emit(lines_, n_stat, tok_info, extra_cites, kind):
        a, b = lines_[0], lines_[-1]
        text = [V[k - 1].strip() for k in lines_]
        formula = SEP.join(text)
        tok, (st, gr), others = tok_info
        bt = bold_above(fence_of[a]) if fence_of.get(a) else None
        sess, sline = sess_for(lines_, [n_stat if n_stat not in lines_ else None, bt])
        runs, cur = [], [lines_[0]]
        for x in lines_[1:]:
            if x == cur[-1] + 1: cur.append(x)
            else: runs.append(cur); cur = [x]
        runs.append(cur)
        cites = [cite(V34, r_[0], r_[-1], "본문") for r_ in runs] + list(extra_cites)
        if sline and not any(c[1] <= sline <= c[2] for c in cites):
            cites.append(cite(V34, sline, role="§ 머리글" if sline in (head_of.get(a), h3_of.get(a)) else "블록 표제"))
        rec = dict(v34행=f"{a}" + (f"–{b}" if b != a else ""), 종류=kind, 표기=tok or "", 원문=text[0][:120])
        jn = nz(" ".join(text)); sd = set().union(*[sides(t) for t in text])
        hits, cands = [], []
        secs = {int(x) for x in re.findall(r"§(\d+)(?![-\d])", " ".join(text)) if int(x) <= 42}   # 줄이 가리키는 엔진 §
        fs_line = {first_side(t) for t in text if "=" in t}
        for k, ks, r, fs_row in pool:
            ok = (len(k) >= 8 and "=" in k and k in jn) or (len(ks & sd) >= 2 and fs_row in fs_line)
            if ok and secs and not any(r["별칭"] == f"엔진 §{n_}" for n_ in secs):
                cands.append(r); continue                      # 줄이 다른 § 를 가리키면 병합하지 않음 (후보로만)
            if ok: hits.append(r)
            elif any(len(x) >= 3 for x in ks & sd): cands.append(r)
        if hits:
            ids = []
            for r in hits:
                for r_ in runs:
                    t_ = cite(V34, r_[0], r_[-1], "본문")
                    if t_ not in r["_cites"]:
                        r["_cites"].append(t_)
                r["근거_위치"] = render(r["_cites"])
                d = "같음" if (not tok or st == r["상태"]) else f"다름 (v3.4 '{tok}'→{st} / 원장 {r['상태'] or '빈칸'})"
                k_, ks_, _, fs_ = next(x for x in pool if x[2] is r)
                basis = "식 첫 조각 포함" if (len(k_) >= 8 and "=" in k_ and k_ in jn) else f"같은 첫 변 '{fs_}' + 공통 변: " + ", ".join(sorted(ks_ & sd))
                out.append(dict(rec, _row=r, 대응="근거 위치 추가", 공통=basis, 상태_대조=d))
                if tok and st != r["상태"]: ids.append(r)
            if ids: conf.append(dict(rows_=ids, ln=a, lb=b, tok=tok, st=st, text=text[0]))
            return
        note = []
        if others: note.append(f"같은 줄 다른 표기 병기: {', '.join(others)}")
        if not sess: note.append("등록_세션: 줄·블록 표제·§ 머리글에 세션 표기 없음")
        if kind == "R3": note.append(f"상태 = 블록 표제 'ALL TZ' (v3.4 {n_stat}행), 줄 자체 표기 없음")
        if tok == "보류": note.append("상태 = 'O 보류' (줄에 '보류' 표기 — 규칙 2, 다른 표기보다 우선)")
        if cands: note.append("같은 항목 후보 (변 1개 공유, 병합 안 함): " + ", ".join(sorted({c["_key"] for c in cands})[:5]))
        hsec = re.match(r"^## (§[0-9]+[a-z]?)", V[head_of[a] - 1]) if head_of.get(a) else None
        h3 = re.match(r"^### (\[S[^\]]+\])", V[h3_of[a] - 1]) if h3_of.get(a) else None
        lab = h3.group(1) if h3 else (hsec.group(1) if hsec else KIND)
        r = add(f"{PFX}{a}", cites, 항목=text[0][:80], 식=formula,
                별칭=f"v3.4 {lab} ({KIND} {a}행)", 상태=st, 등급=gr,
                오차=(f"v3.4 표기: {tok}" if tok else ""), 층=layer(formula), 등록_세션=sess, 근거_파일=V34,
                엔진_S="엔진 원장 행 없음 (식 문자열·변 대조)", 키워드=ctx["keywords"](formula), 비고="; ".join(note))
        out.append(dict(rec, _row=r, 대응="새 행", 공통=("후보: " + ", ".join(sorted({c["_key"] for c in cands})[:5])) if cands else "—", 상태_대조="—"))
    def para_back(n):
        k = n - 1
        lo = fence_of.get(n) or S0
        while k > lo and V[k - 1].strip() and not STAT.search(V[k - 1]): k -= 1
        return list(range(k + 1, n + 1))
    used = set()
    for n in range(S0, E0 + 1):
        if n in used: continue
        l = V[n - 1]; t = l.strip()
        if not t or l.startswith("#") or l.startswith("```"): continue
        has_s, has_m = bool(STAT.search(t)), bool(MATH.search(t))
        if re.match(r"^\*\*.*\*\*", t):                         # 굵은 블록 표제
            if has_s and "=" in t: emit([n], n, status_of(t, TOKMAP), [], "R1"); used.add(n)
            elif has_s: excluded.append((n, "R4 굵은 블록 표제 (표기는 블록 근거·세션 근거로만)", t))
            continue
        if t.endswith(":") and not (has_s and has_m and "=" in t.rstrip(":")):
            if has_s: excluded.append((n, "R4 블록 안 표제 줄", t))
            continue
        if re.match(r"^§\d", t) and has_s and not has_m:          # R5 § 번호 현황 줄 (§N … 상태 표기, 수식 없음)
            emit([n], n, status_of(t, TOKMAP), [], "R5"); used.add(n); continue
        grade_only = is_grade_only(t)
        if (has_s and not has_m) or grade_only:
            if fence_of.get(n) and (grade_only or "오차" in t or t.startswith("★") or t.startswith("전 k")):   # R2 등급만 줄
                para = para_back(n)
                if not any(MATH.search(V[x - 1]) for x in para if x != n):
                    ba = bold_above(fence_of[n]); para = ([ba] if ba else []) + para
                if len(para) > 1:
                    emit(para, n, status_of(" ".join(V[x - 1] for x in para), TOKMAP), [], "R2"); used.update(para); continue
            excluded.append((n, "R4 수식 문자 없는 표기 줄", t)); continue
        if has_s and has_m:                                         # R1 (+ '=' 로 시작하는 이어진 줄)
            lines_ = para_back(n) if t.startswith("=") else [n]
            j = n + 1
            def _go(x):   # 이어진 줄: '=' 로 시작하거나 '오차…%' 등급만 줄 ('=' 없음)
                u = V[x - 1].strip()
                return u.startswith("=") or is_grade_only(u)
            while j <= E0 and _go(j) and not V[j - 1].startswith("```"): lines_.append(j); j += 1
            emit(lines_, n, status_of(" ".join(V[x - 1] for x in lines_), TOKMAP), [], "R1"); used.update(lines_); continue
        if has_m and "=" in t and fence_of.get(n):                  # R3: 'ALL TZ' 블록 표제 아래 수식 줄
            bt = bold_above(fence_of[n])
            sub = None
            k = n - 1
            while k > fence_of[n] and V[k - 1].strip():
                if V[k - 1].strip().endswith(":") and "ALL TZ" in V[k - 1]: sub = k; break
                k -= 1
            src = sub or (bt if bt and "ALL TZ" in V[bt - 1] else None)
            if src:
                lines_ = [n]; j = n + 1
                while j <= E0 and V[j - 1].strip().startswith("="): lines_.append(j); j += 1
                emit(lines_, src, ("TZ", TOKMAP["TZ"], []), [cite(V34, src, role="블록 등급")], "R3"); used.update(lines_)
    ctx[f"step{tag}_sec"] = [(head_of[rng_first(m_["v34행"])], m_["_row"]) for m_ in out]
    ctx[f"step{tag}_map"], ctx[f"step{tag}_conf"], ctx[f"step{tag}_excluded"] = out, conf, excluded
    ctx[f"step{tag}_range"] = (S0, E0)

def write_map(ctx, path, tag="4a"):
    cols = ["v34행", "종류", "원장_ID", "대응", "표기", "상태_대조", "공통", "원문"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for m in ctx[f"step{tag}_map"]:
            m = dict(m); m["원장_ID"] = m["_row"]["id"]; w.writerow(m)
        for n, why, t in ctx[f"step{tag}_excluded"]:
            w.writerow(dict(v34행=n, 종류="제외", 원장_ID="", 대응=why, 표기="", 상태_대조="", 공통="", 원문=t[:120]))

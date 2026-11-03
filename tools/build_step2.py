#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 2 (엔진 §) — TASKS_index_v3.md §4 단계 2  (S1 정정판: 원문 한 줄 = 원장 한 행)
  index/ledger.csv        : 엔진 v9.7 머리말 표 §1~§42 + 출력 로그 ✓ 줄 + v9.7 §P 블록
  index/conflicts.csv     : 엔진 안에서 같은 항목이 다른 등급·귀속·값으로 적힌 곳 (판정 없음)
  index/sample_check_2.md : 무작위 10행 + 원문 해당 행 인용
  index/coverage.md       : 엔진 § 원장 ID 연결 (tools/build_step1.py 재사용)

규칙 (MAPSI 단계 2 정정):
  (1) 원문 한 줄 = 원장 한 행. 묶음 행 없음. 식 열 = 원문 문자열 그대로 (덧붙인 문구 금지).
  (2) 여러 줄 항목은 모든 하위 줄을 식 열에 담는다. 줄 구분자 = " ‖ " (원문에 없는 문자). 근거_위치는 행 범위.
  (3) 등록_세션 = 원문 해당 줄 또는 블록 표제에 적힌 세션 표기만 원문 그대로. 그 밖은 빈칸 + 비고 "추정: …".
  ID: index/id_registry.csv (주 원문 줄 키 → ID) 로 고정. 새 키는 끝 번호 뒤에만 추가 (CLAUDE.md 규칙 7).

근거_위치 형식 (tools/verify_step2.py 가 읽음): "<로그|소스> N행 [역할]" 또는 "<로그|소스> N–M행 (k행) [역할]" 을 " / " 로 연결.
  첫 토큰 = 주 원문 줄 (식·상태·등급·오차가 적힌 줄). 역할: ✓, 항목, 본문, 표, 머리말, 머리말 등급, 계산 블록, 블록 표제.
원문 읽기 전용. 실행: python3 tools/build_step2.py (저장소 루트)
"""
import csv, os, re, random
from collections import OrderedDict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = "code/eqi_unified_engine_v9_7.py"                       # v9.7 완성판 (2,662행)
LOG = "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt"     # 완성판 mpmath 1.4.0 출력 (81,311 B) — 기준 로그
OUT = os.path.join(ROOT, "index")
REG = os.path.join(OUT, "id_registry.csv")
SEP = " ‖ "
COLS = ["id", "행_종류", "항목", "식", "별칭", "상태", "현행_상태", "등급", "상태_출처", "오차", "층", "등록_세션", "근거_파일", "근거_위치",
        "엔진_§", "후속_정정", "키워드", "비고"]

def lines(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", newline="") as f:
        return [l.rstrip("\r\n") for l in f.read().split("\n")]

V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"               # 단계 3 (v3.4 §11)
SRC, LG = lines(ENG), lines(LOG)
FILES = {ENG: SRC, LOG: LG, V34: lines(V34)}
TAG = {LOG: "로그", ENG: "소스", V34: "v3.4"}
def norm(s):            # 비교용 정규화: 앞뒤 공백·상자 테두리 '║' 제거
    return s.strip().strip("║").strip()
def L_(f, n): return FILES[f][n - 1]

# ─────────────── 키워드 (문자열 그대로 일치하는 것만, 해석 없음) ───────────────
KW = OrderedDict([
    ("MAXWELL", ["MAXWELL"]), ("X7", ["X7"]), ("We", ["We"]), ("St", ["St"]), ("TH", ["TH_D", "TH"]),
    ("허블", ["허블"]), ("로렌츠", ["로렌츠"]), ("보른", ["born", "보른"]), ("격자", ["격자", "lattice"]),
    ("포논", ["포논"]), ("membrane", ["membrane", "R_mem"]), ("란다우", ["란다우"]),
    ("볼츠만", ["Boltzmann", "볼츠만", "k_B"]), ("오일러", ["오일러", "γ_EQI"]), ("ħ_ani", ["ħ_ani"]),
    ("[x]·{x}", ["[n1²]", "{n1²}", "{√disc}", "int_x"]), ("상전이", ["상전이"]), ("π탈피", ["π탈피", "π 탈피"]),
    ("리만", ["리만"]), ("Feigenbaum", ["Feigenbaum"]), ("z0", ["z0"]), ("GW", ["GW", "SGWB"]),
    ("D1·D2", ["D1", "D2"]), ("팩토리얼", ["4!", "Γ("]), ("뤼드베리", ["뤼드베리", "n1²"]),
    ("등가원리", ["m_g", "m_i", "등가원리"]), ("양성자", ["양성자", "proton"]), ("중성자", ["중성자", "neutron"]),
    ("Re", ["Re"]), ("Ro", ["Ro"]), ("Bi", ["Bi"]), ("GAP", ["GAP"]), ("LC²GD", ["LC²GD"]), ("α⁻¹", ["α⁻¹", "137"]),
    ("μ", ["μ"]), ("n_c", ["n_c"]), ("L_p", ["L_p"]), ("σ", ["σ"]), ("Stefan–Boltzmann", ["Stefan"]),
    ("빛", ["I_p", "Z_p", "V_p", "h/k_B"]), ("메르센", ["메르센"]), ("콜모고로프", ["콜모고로프", "η×GAP"]),
    ("오비탈", ["오비탈"]), ("브루스터", ["θ_B"]), ("분산", ["분산"]), ("브래그", ["브래그"]),
])
def keywords(text, extra=()):
    ks = [k for k, pats in KW.items() if any(re.search(r"(?<![A-Za-z])" + re.escape(p) + r"(?![a-z])", text) for p in pats)]
    for e in extra:
        if e and e not in ks: ks.append(e)
    return ", ".join(ks)

# ─────────────── 등급 표기 옮김 (원문 표기 그대로, 승격 없음) ───────────────
GRADE_MAP = [("TZ(대수필연)", "대수필연", "대수필연"), ("True Zero(대수필연)", "대수필연", "대수필연"),
             ("Near True Zero", "Near TZ", "Near TZ"), ("Near TZ", "Near TZ", "Near TZ"),
             ("True Zero", "TZ", "TZ"), ("Real Zero", "Real Zero", "Real Zero"),
             ("대수필연", "대수필연", "대수필연"), ("TZ", "TZ", "TZ")]
def map_grade(label):
    label = (label or "").strip()
    for k, st, gr in GRADE_MAP:
        if label.startswith(k): return st, gr
    return "", "없음"

SESS_RE = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*")
def sess_in(text):
    m = SESS_RE.search(text or "")
    return m.group(0) if m else ""

# ─────────────── 근거 토큰 ───────────────
def cite(f, a, b=None, role=""):
    return (f, a, b if b else a, role)
def render(cites):
    out = []
    for f, a, b, role in cites:
        rng = f"{a}행" if a == b else f"{a}–{b}행 ({b - a + 1}행)"
        out.append(f"{TAG[f]} {rng} [{role}]")
    return " / ".join(out)

rows = []
def add(key, cites, **kw):
    if "엔진_S" in kw: kw["엔진_§"] = kw.pop("엔진_S")
    r = {c: "" for c in COLS}; r.update({k: v for k, v in kw.items() if k in COLS})
    r["근거_위치"] = render(cites)
    r["_key"], r["_cites"], r["_bline"], r["_grp"] = key, cites, kw.get("_bline"), kw.get("_grp")
    rows.append(r); return r

# ─────────────── 머리말 표 (소스 docstring) ───────────────
HEAD = OrderedDict()
cur = None
for i in range(1, 100):
    l = SRC[i - 1]
    m = re.match(r"^║\s+§(\d+):\s*(.*?)\s*║?\s*$", l)
    if m:
        cur = int(m.group(1)); HEAD[cur] = dict(line=i, text=norm(m.group(2)), cont=[]); continue
    if cur and re.match(r"^║\s{4,}\S", l) and "──" not in l:
        HEAD[cur]["cont"].append((i, norm(l)))
    elif "──" in l or l.startswith("╚"):
        cur = None
def head_last(s): return HEAD[s]["cont"][-1][0] if HEAD[s]["cont"] else HEAD[s]["line"]
def head_grade(s):
    for pat in ["True Zero(대수필연)", "True Zero", "Near TZ", "탐색 완료(NTZ)"]:
        if pat in HEAD[s]["text"]: return pat
    return ""

SRCPOS = {}
for i in range(140, 1090):
    m = re.match(r"^\s*#\s*(?:──\s*)?§(\d+)(?:,\s*§(\d+))?\s*[:：]", SRC[i - 1])
    if m:
        for g in m.groups():
            if g and int(g) not in SRCPOS: SRCPOS[int(g)] = i
LOGTITLE = {}
for i, l in enumerate(LG, 1):
    m = re.match(r"^\s{2}§(\d+)\s", l)
    if m and int(m.group(1)) not in LOGTITLE: LOGTITLE[int(m.group(1))] = i
SEC0L = next(i for i, l in enumerate(LG, 1) if "SECTION 0 —" in l)          # 로그 블록 표제 (§1~§16)
SESSW = lambda t: (re.search(r"Session \d+(?:~\d+)?", t) or re.search(r"$^", "")).group(0) if re.search(r"Session \d+(?:~\d+)?", t) else ""
SEC0_SESS = SESSW(LG[SEC0L - 1])
NOTE_SEC0 = f"등록_세션 = 블록 표제 표기 (로그 {SEC0L}행 'SECTION 0 … (Session 38~47 §1~§16)')"

def sess_for_sec(s):
    """§ 등록_세션: 로그 표제 [S..] → 머리말 행 S.. (원문 그대로). 없으면 ('', 추정 비고)."""
    if s in LOGTITLE:
        m = re.search(r"\[(S[^\]]*)\]", LG[LOGTITLE[s] - 1])
        if m: return sess_in(m.group(1)), cite(LOG, LOGTITLE[s], role="블록 표제")
    if s in HEAD and sess_in(HEAD[s]["text"]):
        return sess_in(HEAD[s]["text"]), cite(ENG, HEAD[s]["line"], role="머리말")
    if s <= 16 and SEC0_SESS:
        return SEC0_SESS, cite(LOG, SEC0L, role="블록 표제")
    return "", None

def eng_ref(s, extra=""):
    parts = [f"§{s}"]
    if s in HEAD: parts.append(f"머리말 {HEAD[s]['line']}행")
    if s in LOGTITLE: parts.append(f"로그 표제 {LOGTITLE[s]}행")
    if extra: parts.append(extra)
    return " / ".join(parts)

# ─────────────── A. 출력 로그 ✓ 줄 (§1~§42 + z0) ───────────────
CK = re.compile(r"^\s*✓\s+(.*?)\s*(?:오차=(\S+))?\s*(?:★\s*(.+?))?\s*$")
Z0T = next(i for i, l in enumerate(LG, 1) if "전채널 z0 분류자 계단 구조" in l)
def z0_table_line(label):
    return next(j for j, x in enumerate(LG, 1) if x.strip().startswith(label) and "✓" not in x)

for i, l in enumerate(LG, 1):
    if not re.match(r"^\s*✓", l) or i > 470: continue          # 470행 이후 = 요약 블록 (범위 줄)
    m = CK.match(l)
    label, err, grade = m.group(1).strip(), m.group(2) or "", (m.group(3) or "").strip()
    ms = re.search(r"§(\d+)", label)
    z0_sess = SESSW(LG[Z0T - 1])
    z0_note = f"등록_세션 = 블록 표제 표기 (로그 {Z0T}행 '전채널 z0 분류자 계단 구조 (Session 38~47 §1~§16)')"
    if not ms:   # z0 표 '대칭 이벤트 (BNS/BBH)' — § 번호 없음
        tl = z0_table_line(label)
        f0 = re.split(r"\s{2,}", LG[tl - 1].strip())[1]
        add(f"LOG:{i}", [cite(LOG, i, role="✓"), cite(LOG, tl, tl + 1, "표"), cite(LOG, Z0T, role="블록 표제")],
            항목=label, 식=f0 + SEP + norm(LG[tl]), 별칭="엔진 SECTION 1 z0 분류자 (§ 번호 없음)", 상태="", 등급="없음",
            오차=f"오차={err}", 층="해당 없음", 등록_세션=z0_sess, 근거_파일=LOG,
            엔진_S=f"SECTION 1 (소스 1530행) / 로그 표 {tl}행 / ✓ {i}행", 키워드=keywords(label, ["z0", "리만"]),
            비고=z0_note)
        continue
    s = int(ms.group(1))
    name = re.sub(r"\s*\[별도 출력\]$", "", re.sub(r"^§\d+\s+", "", label).strip())
    cites = [cite(LOG, i, role="✓")]
    errtxt = err
    if s == 5 and not grade:       # '[별도 출력]' → 로그 23–24행
        grade = "True Zero"; errtxt = "7.2232098e-15 (로그 24행 'SI대비 오차%')"; cites = [cite(LOG, i, role="✓"), cite(LOG, 23, 24, "본문 등급")]
    if grade: errtxt = (errtxt + f" (★ {grade})").strip()
    st, gr = map_grade(grade)
    note = ""
    if re.match(r"GW|SGWB", label):
        tl = z0_table_line(label)
        formula = re.split(r"\s{2,}", LG[tl - 1].strip())[1]
        cites.append(cite(LOG, tl, role="표"))
        sess = z0_sess; note = z0_note
        cites.append(cite(LOG, Z0T, role="블록 표제"))
    else:
        if "=" in name or "<" in name:
            formula = name
        else:
            h = HEAD[s]; formula = SEP.join([h["text"]] + [t for _, t in h["cont"]])
        sess, sc = sess_for_sec(s)
        if sc: cites.append(sc)
        if sc and sc[1] == SEC0L: note = NOTE_SEC0
    if s in HEAD: cites.append(cite(ENG, HEAD[s]["line"], head_last(s), "머리말"))
    if s in SRCPOS: cites.append(cite(ENG, SRCPOS[s], role="계산 블록"))
    layer = {19: "5체", 21: "7체", 24: "5체;7체"}.get(s, "해당 없음")
    add(f"LOG:{i}", cites, 항목=name, 식=formula, 별칭=f"엔진 §{s}", 상태=st, 등급=gr, 오차=errtxt, 층=layer,
        등록_세션=sess, 근거_파일=ENG, 엔진_S=eng_ref(s, f"✓ {i}행 ({LOG})"), 키워드=keywords(label + " " + formula), 비고=note)

# ─────────────── B. 머리말 표에만 있는 항목 (✓ 줄 없음) ───────────────
EXTRA = [(2, None, "MAXWELL 4차항 c4×GD⁴"), (9, None, "γ 3항 구조 (오일러-란다우어)"), (29, None, "전달함수 Q/MAXWELL≈34"),
         (10, "κ×Φ₀=1", "κ×Φ₀=1"), (17, "Fr=1/(2πφ)", "Fr=1/(2πφ)"), (27, "|ln(GAP)|/GAP_D2=213.5", "|ln(GAP)|/GAP_D2=213.5"),
         (30, "Sh=MAXWELL", "Sh=MAXWELL")]
for s, needle, name in EXTRA:
    h = HEAD[s]
    cand = [(h["line"], h["text"])] + h["cont"]
    ln, txt = next((a, b) for a, b in cand if needle in b) if needle else (h["line"], h["text"])
    formula = needle if needle else txt
    hg = head_grade(s)
    st, gr = map_grade(hg)
    if hg == "탐색 완료(NTZ)": st, gr = "", "없음"
    cites = [cite(ENG, ln, role="머리말")]
    if ln != h["line"]: cites.append(cite(ENG, h["line"], role="머리말 등급"))
    if s in SRCPOS: cites.append(cite(ENG, SRCPOS[s], role="계산 블록"))
    sess, sc = sess_for_sec(s)
    if sc and sc[1] != ln: cites.append(sc)
    nt = f"로그에 '§{s}' 표제·✓ 줄 없음 (grep) — 머리말 표·소스 위치만 기록" if not needle else "머리말 표 기재, 로그 ✓ 줄에는 이 식 없음"
    add(f"ENG:{ln}", cites, 항목=name, 식=formula, 별칭=f"엔진 §{s}", 상태=st, 등급=gr,
        오차=(f"머리말 등급 표기: {hg}" if hg else ""), 층="해당 없음", 등록_세션=sess, 근거_파일=ENG,
        엔진_S=eng_ref(s, nt), 키워드=keywords(name + " " + txt),
        비고=((NOTE_SEC0 if sc and sc[1] == SEC0L else "") + ("" if needle else "; 항목명은 색인 부여 (머리말 행 전체가 식)")).lstrip("; "))

# §15 메르센 수열 (로그 본문 73–78행)
m1 = next(i for i, l in enumerate(LG, 1) if "D2 지수 메르센 수열" in l)
m2 = m1
while LG[m2].strip().startswith("k="): m2 += 1
add(f"LOG:{m1}", [cite(LOG, m1, m2, "본문"), cite(ENG, HEAD[15]["line"], role="머리말 등급"), cite(LOG, LOGTITLE[15], role="블록 표제")],
    항목="D2 지수 메르센 수열 (k=3~5 ✓ / k=6~7 ✗ X7 금지)", 식=SEP.join(norm(LG[j - 1]) for j in range(m1, m2 + 1)),
    별칭="엔진 §15", 상태="TZ", 등급="TZ", 오차="머리말 등급 표기: True Zero (§15 전체)", 층="해당 없음",
    등록_세션=sess_for_sec(15)[0], 근거_파일=LOG, 엔진_S=eng_ref(15, "로그 본문 ✓·✗ 표기"), 키워드=keywords("메르센 X7 포논 D2"),
    비고="항목명은 색인 부여 (로그 본문 줄 묶음)")
# §13 ε (로그 본문 등급 Real Zero)
e1 = next(i for i, l in enumerate(LG, 1) if "ε = σ³/N³×K_EQI" in l)
add(f"LOG:{e1}", [cite(LOG, e1, role="본문"), cite(LOG, LOGTITLE[13], role="블록 표제"), cite(ENG, HEAD[13]["line"], role="머리말")],
    항목="ε = σ³/N³×K_EQI", 식=norm(LG[e1 - 1]), 별칭="엔진 §13", 상태="Real Zero", 등급="Real Zero",
    오차="0.005% (로그 원문 'Real Zero 0.005%')", 층="해당 없음", 등록_세션=sess_for_sec(13)[0], 근거_파일=LOG,
    엔진_S=eng_ref(13, "✓ 줄 없음, 로그 본문 등급"), 키워드=keywords("ε σ³ K_EQI Leibniz", ["포논"]))

# ─────────────── C. v9.7 §P 블록 (원문 한 줄 = 한 행) ───────────────
PB = re.compile(r"^\s{2}(§P\S*)\s+(.*)$")
IT = re.compile(r"^\s{2}\[(§[A-Z][^\]]*)\]\s*(.*)$")
items, block = [], None
i = 528
while i <= len(LG):
    l = LG[i - 1]
    if l.startswith("  등급 집계"): break
    m = PB.match(l)
    if m:
        block = (m.group(1), m.group(2).strip(), i); i += 1; continue
    if block is None or not l.strip() or l.strip().startswith("─") or l.strip().startswith("═"):
        i += 1; continue
    m = IT.match(l)
    if m:
        j = i + 1
        while j <= len(LG) and LG[j - 1].startswith("     ") and not IT.match(LG[j - 1]) \
                and not re.match(r"^\s{5}\[S\d+", LG[j - 1]):
            j += 1
        items.append(dict(id=m.group(1), body=m.group(2), line=i, end=j - 1, block=block, kind="항목")); i = j; continue
    if l.startswith("     "):        # 블록 머리 값·주석 줄 / '[S..] 계보' 줄 (항목 밖)
        j = i + 1
        if not re.match(r"^\s{5}\[S\d+", l):
            while j <= len(LG) and LG[j - 1].startswith("     ") and not IT.match(LG[j - 1]) \
                    and not re.match(r"^\s{5}\[S\d+", LG[j - 1]):
                j += 1
        items.append(dict(id=None, body=l.strip(), line=i, end=j - 1, block=block, kind="본문")); i = j; continue
    i += 1     # '  ※ …' 같은 2칸 들여쓴 블록 끝 주석은 해당 행 엔진_§ 에 병기

def parse(body):
    f = [x.strip() for x in body.split(" | ")]
    d = dict(label=f[0], calc="", tgt="", err="", grade="", src="")
    if len(f) >= 6: d.update(calc=f[1], tgt=f[2], err=f[3], grade=f[4], src=f[5])
    elif len(f) == 5: d.update(calc=f[1], tgt=f[2], grade=f[3], src=f[4])
    elif len(f) == 4: d.update(calc=f[1], grade=f[2], src=f[3])
    return d

def layer_of(btitle, label):
    m = re.match(r"^(\d)체 층", btitle)
    if m: return m.group(1) + "체"
    if "체 계층" in btitle:
        hits = set(re.findall(r"([357])체", label))
        if len(hits) == 1 and not re.search(r"[0-9]체.*[0-9]체", label): return hits.pop() + "체"
    return "해당 없음"

_bc = {}
for it in items: _bc.setdefault(it["block"][0], set()).add(it["block"][2])
DUP_B = {b for b, v in _bc.items() if len(v) > 1}
UNOTE = next(i for i, l in enumerate(LG, 1) if "U-1~U-3 '귀속 확정' 판정 없음" in l)
S114_NOTE = "엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인')"
EST_S114 = "추정: S114 (v9.7 신규 블록, S114 성과정리 1. 표) — 원문 줄·블록 표제에 세션 표기 없음"

def gkey(i):     # 옛 묶음 행 (L0208 등) 의 구성원 판별용 — 묶음 생성에는 쓰지 않음
    for pat, rep in [(r"^(§P3-[12])\.k\d+$", r"\1"), (r"^(§P3-3)\.k\d+([a-f])$", r"\1.k*\2"), (r"^(§C-8)\.\S+$", r"\1"),
                     (r"^(§O-1)\.[a-z]$", r"\1"), (r"^(§L-7b)\.k\S+$", r"\1")]:
        if i and re.match(pat, i): return re.sub(pat, rep, i)
    return None

for it in items:
    bid, btitle, bline = it["block"]
    subs = [(j, norm(LG[j - 1])) for j in range(it["line"] + 1, it["end"] + 1)]
    bt_cite = cite(LOG, bline, role="블록 표제")
    loc_block = (f"{bid} '{btitle.split(' [')[0]}'" if bid in DUP_B else bid)
    if it["kind"] == "본문":
        lab = it["body"]
        formula = SEP.join([lab] + [t for _, t in subs])
        pre = not re.match(r"\[S\d+", lab)
        name = f"{bid} 블록 머리 (값·주석)" if pre else lab
        sess = sess_in(lab) if not pre else ""
        note = "항목명은 색인 부여 (원문 표제 없음)" if pre else ""
        if not sess:
            bs = re.search(r"\[(S[^\]]*)\]", btitle)
            if bs and sess_in(bs.group(1)):
                sess = sess_in(bs.group(1)); note = (note + "; 등록_세션 = 블록 표제 표기").lstrip("; ")
            else:
                note = (note + "; " + EST_S114).lstrip("; ")
        add(f"LOG:{it['line']}", [cite(LOG, it["line"], it["end"], "본문"), bt_cite],
            항목=name, 식=formula, 별칭=f"엔진 v9.7 {bid} 주석 (로그 {it['line']}행)", 상태="", 등급="없음", 오차="",
            층=layer_of(btitle, lab), 등록_세션=sess, 근거_파일=LOG,
            엔진_S=f"v9.7 {bid} 블록 머리 (표제 {bline}행) / {S114_NOTE} / 소스 class EQIPhononLattice",
            키워드=keywords(btitle + " " + formula, [bid]), 비고=note, _bline=bline)
        continue
    p = parse(it["body"]); xid = it["id"]
    st, gr = map_grade(p["grade"])
    if p["grade"] in ("정수", "-", "") or p["grade"].startswith("통과") or p["grade"].startswith("귀속 검토"):
        st, gr = "", "없음"
    err = p["err"] + (f" (판정 열: {p['grade']})" if p["grade"] else "")
    err = err.strip()
    formula = SEP.join([p["label"]] + [t for _, t in subs])
    name = p["label"]          # 원문 라벨 그대로 (2026-11-03 MAPSI 정정: '(기대 …)' 삭제 등 원문 문자 변경 금지)
    fix, note = "", ""
    if xid == "§L-4":
        fix = "S114 정정: 기대값 −7/4. 구 기대값 +1/4 (S77 §48-E, 단위계 철학 v3 §11.1, v3.4 2104행) 산술 오기 — 엔진 소스 2325–2326행"
    if xid == "§P7-3":
        fix = "f×We²×Re 폐기(S92) — 원문 표기 '폐기' (엔진 로그 696행)"
    if xid == "§C-주석":
        st = "미적용"
        fix = "S114 v9.7: 현 체제(단일 항등원) 미적용, 계산하지 않음 — 엔진 소스 2408–2409행 / 로그 743행"
        name = "S64 §6.2 '중력질량 = E_topo² = φ^1.5 = 관성질량' (v15 이중 항등원 표기)"
        note = "항목명은 색인 부여 (원문 줄 요약)"
    sess = sess_in(p["src"]) or sess_in(p["label"])
    if not sess:
        bs = re.search(r"\[(S[^\]]*)\]", btitle)
        if bs and sess_in(bs.group(1)):
            sess = sess_in(bs.group(1)); note = (note + "; 등록_세션 = 블록 표제 표기").lstrip("; ")
        else:
            note = (note + "; " + EST_S114).lstrip("; ")
    eng = f"v9.7 {bid} {xid} / {S114_NOTE} / 소스 class EQIPhononLattice"
    if bid.startswith("§P-DS") and xid.startswith("§U"):
        eng += f" / 로그 {UNOTE}행: '※ U-1~U-3 '귀속 확정' 판정 없음 (Architect/MAPSI 판단).'"
    add(f"LOG:{it['line']}", [cite(LOG, it["line"], it["end"], "항목"), bt_cite],
        항목=name, 식=formula, 별칭=f"엔진 v9.7 {xid}" + (f"; 출처 표기 '{p['src']}'" if p["src"] else ""),
        상태=st, 등급=gr, 오차=err, 층=layer_of(btitle, p["label"]), 등록_세션=sess, 근거_파일=LOG,
        엔진_S=eng, 후속_정정=fix, 키워드=keywords(btitle + " " + formula, [bid]), 비고=note,
        _bline=bline, _grp=(bid, gkey(xid)) if gkey(xid) else None)

# ─────────────── 단계 3 hook: v3.4 §11 (tools/build_step3.py) ───────────────
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("s3", os.path.join(ROOT, "tools", "build_step3.py"))
s3 = _ilu.module_from_spec(_sp); _sp.loader.exec_module(s3)
CTX = dict(rows=rows, add=add, cite=cite, render=render, FILES=FILES, map_grade=map_grade, keywords=keywords, ENG=ENG)
s3.apply(CTX)
# ─────────────── 단계 4a hook: v3.4 본문 §12~§30 (tools/build_step4a.py) ───────────────
_sp4 = _ilu.spec_from_file_location("s4a", os.path.join(ROOT, "tools", "build_step4a.py"))
s4a = _ilu.module_from_spec(_sp4); _sp4.loader.exec_module(s4a)
_spt = _ilu.spec_from_file_location("si0", os.path.join(ROOT, "tools", "status_inherit.py"))
_si0 = _ilu.module_from_spec(_spt); _spt.loader.exec_module(_si0)
CTX["TOKMAP"] = _si0.TOKMAP
s4a.apply(CTX, "4a")
s4a.apply(CTX, "4b")      # 단계 4b: v3.4 캡슐 블록 S79~S108 (2211행 ~ 파일 끝), 같은 규칙
# ─────────────── 단계 5 hook: 세션 문서 S109~S119 + 번역·설계 (tools/build_step5.py) ───────────────
_sp5 = _ilu.spec_from_file_location("s5", os.path.join(ROOT, "tools", "build_step5.py"))
s5 = _ilu.module_from_spec(_sp5); _sp5.loader.exec_module(s5)
def _layer5(t):
    h = re.findall(r"([357])체", t)
    return (h[0] + "체") if len(h) == 1 else "해당 없음"
CTX.update(ROOT=ROOT, TAG=TAG, sides=s4a.sides, first_side=s4a.first_side, layer=_layer5)
s5.apply(CTX)
# ─────────────── '정의·원칙' 행 hook (tools/build_def.py, MAPSI 승인 2026-11-03) ───────────────
_spd = _ilu.spec_from_file_location("sdef", os.path.join(ROOT, "tools", "build_def.py"))
sdef = _ilu.module_from_spec(_spd); _spd.loader.exec_module(sdef)
CTX["s5"] = s5
sdef.apply(CTX)
# ─────────────── 단계 6 (S21~S108) — 지금은 건수 보고용 시험 실행만 (STEP6_DRYRUN=1): 원장·등록부 등 파일 쓰기 전에 멈춤 ───────────────
# 단계 6 (S21~S108) — MAPSI 승인 판 C: 꼬리표 정규화 + 정의·원칙→등록 병합 + 머리글 사슬 기준어 (단계 6 에만 적용)
STEP6 = os.environ.get("STEP6_DRYRUN") == "1"          # 1 이면 건수 보고만 쓰고 멈춤
CTX["NORM6"] = True; CTX["CHAIN6"] = True
s5.apply(CTX, "6")
sdef.apply(CTX, "6")
for r in rows:
    if not r.get("행_종류"): r["행_종류"] = "등록"
for r in rows:
    if not r.get("행_종류"): r["행_종류"] = "등록"

# ─────────────── ID 부여 (등록부 고정, 규칙 7) ───────────────
reg = list(csv.DictReader(open(REG, encoding="utf-8")))
key2id = {r["키"]: r["id"] for r in reg}
bykey = {}
for r in rows:
    if r["_key"] in bykey: raise SystemExit(f"주 원문 줄 키 중복: {r['_key']}")
    bykey[r["_key"]] = r
missing = [r["id"] for r in reg if r["키"] not in bykey]
if missing: raise SystemExit(f"ID 불변 규칙 위반: 등록부 ID 의 원문 줄이 생성되지 않음 {missing[:10]}")
nxt = max(int(r["id"][1:]) for r in reg) + 1
new_reg = []
old_led = {r["id"]: r for r in csv.DictReader(open(os.path.join(OUT, "ledger.csv"), encoding="utf-8"))}
# 옛 묶음 행 (별칭에 '~§' 범위) → (블록, 묶음키) → 원 ID
grp_origin = {}
for rid, r in old_led.items():
    m = re.match(r"^엔진 v9\.7 (§\S+)~§\S+", r["별칭"])
    if m:
        k = next(x["키"] for x in reg if x["id"] == rid)
        g = bykey[k]["_grp"]
        if g: grp_origin[g] = rid
for r in rows:
    if r["_key"] in key2id:
        r["id"] = key2id[r["_key"]]
    else:
        r["id"] = f"L{nxt:04d}"; nxt += 1
        new_reg.append(dict(id=r["id"], 키=r["_key"], 최초_항목=r["항목"], 최초_별칭=r["별칭"],
                            분리_원ID=grp_origin.get(r["_grp"], "")))
split_of = {x["id"]: x.get("분리_원ID", "") for x in reg + new_reg}
for r in rows:
    if split_of.get(r["id"]):
        r["비고"] = (f"원 묶음 {split_of[r['id']]}에서 분리; " + r["비고"]).rstrip("; ")
SPLIT_FIRST = set(grp_origin.values()) | {v for v in split_of.values() if v}   # 묶음 첫 줄만 남긴 원 ID (MAPSI 정정 (1))

def id_guard(path, new_rows, keys, allow=()):
    """ID 불변 규칙 (CLAUDE.md 규칙 7): 기존 파일의 모든 ID 가 남아 있고, 허용 목록 밖에서는 같은 항목이어야 한다."""
    if not os.path.exists(path): return
    with open(path, encoding="utf-8") as f:
        old = {r["id"]: r for r in csv.DictReader(f)}
    new = {r["id"]: r for r in new_rows}
    bad = [i for i, r in old.items() if i not in new or (i not in allow and any(r.get(k, "") != new[i].get(k, "") for k in keys))]
    if bad:
        raise SystemExit(f"ID 불변 규칙 위반 ({os.path.basename(path)}): {bad[:10]} … — 기존 ID 를 바꾸거나 지우지 말 것")

rows.sort(key=lambda r: int(r["id"][1:]))
# 항목 원문 복원 (2026-11-03 MAPSI 정정: 원문 문자 대체·삭제 금지) — 이전 판이 '(기대 …)' 를 지운 12행. ID 유지, 항목만 원문으로.
RESTORED_ITEM = {"L0114", "L0115", "L0116", "L0145", "L0278", "L0284", "L0290", "L0296", "L0302", "L0308", "L0314", "L0320"}
id_guard(os.path.join(OUT, "ledger.csv"), rows, ("항목", "별칭"), allow=SPLIT_FIRST | RESTORED_ITEM)
for r in rows:
    if r["id"] in SPLIT_FIRST:
        r["비고"] = ("묶음 해체: 첫 원문 줄만 유지 (MAPSI 단계 2 정정 (1)); " + r["비고"]).rstrip("; ")

# 단계 6 '연결 후보' (MAPSI 보충 (2)): 짧거나 표기만 다른 연결은 합치지 않고 비고에 ID
for r in rows:
    cs = [c["id"] for c in r.get("_cands6", []) if c is not r and c.get("id")]
    if cs:
        cs = sorted(set(cs), key=lambda x: int(x[1:]))[:5]
        if "연결 후보:" not in r["비고"]: r["비고"] = (r["비고"] + "; " if r["비고"] else "") + "연결 후보: " + ", ".join(cs)

# ─────────────── 상태 내려받기 (CLAUDE.md 규칙 4 보충) — tools/status_inherit.py ───────────────
_sp2 = _ilu.spec_from_file_location("si", os.path.join(ROOT, "tools", "status_inherit.py"))
si = _ilu.module_from_spec(_sp2); _sp2.loader.exec_module(si)
SI_CTX = dict(ROOT=ROOT, FILES=FILES, render=render, TAG=TAG)
si.apply(rows, SI_CTX)
for m in CTX["step3_map"]:          # 대응표: 내려받기 뒤 원장 상태 반영
    r = m["_row"]
    if m["등급_대조"].startswith("다름") and "(내려받기" in r.get("상태_출처", "") and m["v34_등급"]:
        if map_grade(m["v34_등급"])[0] == r["상태"]:
            m["원장_상태"] = r["상태"]; m["등급_대조"] = "같음 (내려받기 후 — 1순위 엔진 표기 없음, 원장 상태 = v3.4 표기)"

# ─────────────── 단계 7 (후속_정정 채우기, MAPSI 결정 2026-11-03 방식 (나)) — tools/build_step7.py ───────────────
_sp7 = _ilu.spec_from_file_location("s7", os.path.join(ROOT, "tools", "build_step7.py"))
s7 = _ilu.module_from_spec(_sp7); _sp7.loader.exec_module(s7)
S7_CTX = dict(OUT=OUT, s4a=s4a)
if not STEP6: s7.apply(rows, S7_CTX)

if STEP6:
    _sp6 = _ilu.spec_from_file_location("r6", os.path.join(ROOT, "tools", "report_step6.py"))
    r6 = _ilu.module_from_spec(_sp6); _sp6.loader.exec_module(r6)
    r6.report(CTX, rows, OUT)
    raise SystemExit("STEP6_DRYRUN: 건수 보고만 쓰고 멈춤 (원장·등록부·대응표 등은 쓰지 않음)")
with open(os.path.join(OUT, "ledger.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
s7.write(rows, S7_CTX)
if new_reg:
    with open(REG, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "키", "최초_항목", "최초_별칭", "분리_원ID"])
        w.writeheader(); w.writerows([{**x, "분리_원ID": x.get("분리_원ID", "")} for x in reg + new_reg])

s3.write_map(CTX, os.path.join(OUT, "map_step3.csv"))
s4a.write_map(CTX, os.path.join(OUT, "map_step4a.csv"), "4a")
s4a.write_map(CTX, os.path.join(OUT, "map_step4b.csv"), "4b")
s5.write_map(CTX, os.path.join(OUT, "map_step5.csv"))
s5.write_map(CTX, os.path.join(OUT, "map_step6.csv"), "6")
with open(os.path.join(OUT, "map_def6.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["문서", "행", "원장_ID", "대응", "키워드", "원문"], extrasaction="ignore"); w.writeheader()
    for m in CTX["def_rows6"]: w.writerow(dict(m, 원장_ID=m["_row"]["id"], 대응="새 행"))
    for p_, n_, hs in CTX.get("def6_merged", []):
        w.writerow(dict(문서=p_, 행=n_, 원장_ID=", ".join(r["id"] for r in hs), 대응="등록 행 근거 위치 추가", 원문=FILES[p_][n_ - 1].strip()[:120]))
with open(os.path.join(OUT, "map_def.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["문서", "행", "원장_ID", "키워드", "원문"], extrasaction="ignore"); w.writeheader()
    for m in CTX["def_rows"]: w.writerow(dict(m, 원장_ID=m["_row"]["id"]))

# ─────────────── conflicts.csv (K 번호 고정, 새 충돌은 끝에만) ───────────────
def find(pred): return next(r["id"] for r in rows if pred(r))
TK = lines("code/TASKS.md")
T7C = next(i for i, l in enumerate(TK, 1) if "L-7c 브래그" in l)
L7C = next(i for i, l in enumerate(LG, 1) if "TASKS 문구 '7차부터 < 1'" in l)
CF = [
    dict(항목="§12 ratio(34)×34 vs 2/R_mem 등급", 원장_ID=find(lambda r: r["항목"].startswith("ratio(34)")),
         위치_A=f"{ENG} 머리말 24행", 내용_A="'S43  True Zero'",
         위치_B=f"{LOG} 46행 / 395행", 내용_B=f"46행 '{LG[45].strip()}' / 395행 '★ Near TZ'",
         비고="같은 오차값(5.4e-3%)에 등급 표기 셋 (머리말·로그 본문 True Zero, ✓ 줄 Near TZ). 판정 없음"),
    dict(항목="a₅_eff = 5σ⁵/4 의 § 귀속", 원장_ID=find(lambda r: r["항목"].startswith("a₅_eff")),
         위치_A=f"{ENG} 머리말 26행 / {LOG} 57행", 내용_A="§14 (머리말 'a₅_eff=5σ⁵/4, 포논 굴절', 로그 §14 블록 57행)",
         위치_B=f"{LOG} 399행 / 71–72행", 내용_B="'✓ §15 a₅_eff = 5/4×σ⁵' (로그 §15 블록 71–72행에도 a₅_eff 식)",
         비고="같은 식이 §14·§15 양쪽에 귀속 표기. 판정 없음"),
    dict(항목="§L-6 H&Q 2.95μm 등급", 원장_ID=find(lambda r: r["별칭"].startswith("엔진 v9.7 §L-6;")),
         위치_A=f"{LOG} 709행", 내용_A="v9.7 판정 'Near TZ'", 위치_B=f"{LOG} 709행 (원문 인용)", 내용_B="'원문 TZ(S99)'",
         비고="엔진이 두 등급 병기. S99 원문 대조는 단계 6. 판정 없음"),
    dict(항목="§14 n_ref,6 (k=6 포논13차) 표시값", 원장_ID="",
         위치_A="code/eqi_unified_engine_v9_7_output.txt 67행 (이전 판, git 74b469a)", 내용_A="k=6(포논13차): 0.0007934570312",
         위치_B="code/eqi_unified_engine_v9_7_terminal.txt 67행 (이전 판, git 74b469a)", 내용_B="k=6(포논13차): 0.0007934570313",
         상태="근거 파일 교체로 해소, 이전 판 git 74b469a",
         비고="같은 엔진 출력 두 파일의 마지막 자리 차이. code/CLAUDE.md '기준 환경' (mpmath 1.4.0 / 1.3.0 반올림 경계값 13/16384) 참조. "
              f"현 기준 로그 {LOG} 67행 = '{LG[66].strip()}'. "
              "perplexity 독립 대조 (2026-11-02): mpmath 1.3.0 재실행 시 k=6 표시 마지막 자리 1줄 차이 (13/16384 = 0.00079345703125, 반올림 방식 차이) — 같은 현상. 판정 없음"),
    dict(항목="Stefan–Boltzmann 지수 기대값", 원장_ID=find(lambda r: r["별칭"].startswith("엔진 v9.7 §L-4;")),
         위치_A="성과정리문서/EQI_연구지침_프로젝트용_v3_4.md 2104행", 내용_A="§48-E: Stefan-Boltzmann σ ∝ φ^(+1/4) = M_p = 수명 지수  [TZ]",
         위치_B=f"{ENG} 2325–2326행 / {LOG} 707행", 내용_B="기대값 −7/4, '+1/4 … 산술 오기' (정정 S114)",
         비고="엔진 측 정정 기록 있음. 원장 후속_정정 열에 옮김. v3.4 원문 미수정"),
    dict(항목="§L-7c 브래그 N·σ^(2k+1)/2 < 1 시작 차수 (문구)", 원장_ID=find(lambda r: r["별칭"].startswith("엔진 v9.7 §L-7c")),
         위치_A=f"code/TASKS.md {T7C}행", 내용_A=TK[T7C - 1].strip(),
         위치_B=f"{LOG} {L7C}행 (같은 줄, 엔진 계산 표기)", 내용_B=LG[L7C - 1].strip(),
         비고="엔진 출력 '※ TASKS 문구 … 불일치' 줄. 판정 없음"),
]
# 단계 3: 같은 항목의 상태가 엔진 원장과 v3.4 §11 에서 다르게 적힌 곳 (판정 없음, 원장 상태는 엔진 기준 그대로)
for c3 in CTX["step3_conf"]:
    r = c3["row"]
    _res = ("해소: 상태 내려받기 (CLAUDE.md 규칙 4 보충, 2026-11-03) — 1순위 엔진 표기 없음, 원장 상태 = " + r["상태_출처"].split(" (내려받기")[0])
    _res = _res if ("(내려받기" in r.get("상태_출처", "") and r["상태"] == c3["st"]) else ""
    CF.append(dict(상태=_res, 항목=f"{r['항목']} 상태 (엔진 / v3.4 §11)", 원장_ID=r["id"],
                   위치_A=r["근거_위치"].split(" / ")[0] + f" ({r['근거_파일']})", 내용_A=f"원장 상태 '{r['상태'] or '빈칸'}' / 등급 '{r['등급']}' (오차 열: {r['오차'] or '—'})",
                   위치_B=f"{V34} {c3['ln']}행 (§11 표)", 내용_B=FILES[V34][c3['ln'] - 1].strip(),
                   비고=f"v3.4 표기 '{c3['g']}' (옮기면 '{c3['st']}'). 등록 표 우선 순서 (CLAUDE.md 규칙 4) 에 따라 원장 상태는 엔진 표기 유지. 판정 없음"))
# 단계 4a: v3.4 본문 줄 표기와 원장 상태가 다른 곳 (판정 없음)
for c5 in CTX["step5_conf"] + CTX["step6_conf"]:
    ids = ", ".join(r["id"] for r in c5["rows_"]); r0 = c5["rows_"][0]
    CF.append(dict(항목=f"{c5['p'].split('/')[-1]} {c5['ln']}행 '{c5['text'][:40]}' 상태 (원장 / 세션 문서)", 원장_ID=ids,
                   위치_A=f"{r0['근거_위치'].split(' / ')[0]} ({r0['근거_파일']})", 내용_A=f"원장 상태 '{r0['상태'] or '빈칸'}'",
                   위치_B=f"{c5['p']} {c5['ln']}행" + (f"–{c5['lb']}행" if c5['lb'] != c5['ln'] else "") + (f" (원문 docx: {c5['src']})" if c5['src'] else ""),
                   내용_B=c5["text"], 비고=f"세션 문서 표기 '{c5['tok']}' (옮기면 '{c5['st']}'). 등록 표 우선 순서상 원장 상태 유지. 판정 없음"))
for c4 in CTX["step4a_conf"] + CTX["step4b_conf"]:
    ids = ", ".join(r["id"] for r in c4["rows_"])
    r0 = c4["rows_"][0]
    CF.append(dict(항목=f"v3.4 본문 {c4['ln']}행 '{c4['text'][:40]}' 상태 (원장 / v3.4 본문)", 원장_ID=ids,
                   위치_A=f"{r0['근거_위치'].split(' / ')[0]} ({r0['근거_파일']})" + (f" 외 {len(c4['rows_'])-1}행" if len(c4["rows_"]) > 1 else ""),
                   내용_A=f"원장 상태 '{r0['상태'] or '빈칸'}' (상태_출처: {r0.get('상태_출처', '')[:60]})",
                   위치_B=f"{V34} {c4['ln']}행" + (f"–{c4['lb']}행" if c4['lb'] != c4['ln'] else "") + " (본문)", 내용_B=c4["text"],
                   비고=f"v3.4 본문 표기 '{c4['tok']}' (옮기면 '{c4['st']}'). v3.4 본문은 등록 표 우선 순서 밖 — 원장 상태 유지. 판정 없음"))
# K 번호: 기존 conflicts.csv 의 항목 → ID 유지, 새 항목만 끝 번호 뒤 (CLAUDE.md 규칙 7)
_cfp = os.path.join(OUT, "conflicts.csv")
_old = {r["항목"]: r["id"] for r in csv.DictReader(open(_cfp, encoding="utf-8"))} if os.path.exists(_cfp) else {}
_nk = max([int(v[1:]) for v in _old.values()] or [0]) + 1
for c in CF:
    if c["항목"] in _old: c["id"] = _old[c["항목"]]
    else: c["id"] = f"K{_nk:04d}"; _nk += 1
CF.sort(key=lambda c: int(c["id"][1:]))
ccols = ["id", "항목", "원장_ID", "위치_A", "내용_A", "위치_B", "내용_B", "상태", "비고"]
id_guard(os.path.join(OUT, "conflicts.csv"), CF, ("항목",))
with open(os.path.join(OUT, "conflicts.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ccols); w.writeheader(); w.writerows(CF)

# ─────────────── sample_check_2.md (추가 행 2개 이상 포함) ───────────────
# perplexity 독립 대조 통과본 (2026-11-02) — 고정. 파일이 없을 때만 생성 (재생성하면 대조 기록이 바뀜)
if not os.path.exists(os.path.join(OUT, "sample_check_2.md")):
    added = {r["id"] for r in rows if int(r["id"][1:]) > 265}
    seed = 20261103
    while True:
        random.seed(seed)
        pick = sorted(random.sample(rows, 10), key=lambda r: int(r["id"][1:]))
        if sum(r["id"] in added for r in pick) >= 2: break
        seed += 1
    S = [f"# sample_check_2.md — 단계 2 표본 대조 (무작위 10행, seed {seed}, L0266 이후 추가 행 2개 이상 조건)\n",
         "> 원장 행 + 근거_위치의 원문 행 인용. 식 열의 ' ‖ ' 는 원문 줄 구분자. MAPSI / perplexity 대조용.",
         "> 틀린 행이 있으면 단계 2 전체 재점검 (TASKS §7). 자동 대조 결과는 `index/verify_step2.md`.\n"]
    for r in pick:
        S.append(f"## {r['id']} — {r['항목']}" + ("  (추가 행)" if r["id"] in added else "") + "\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', '\\|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b}" if b != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_3.md (단계 3 묶음: v3.4 §11 토큰이 있는 행, 새 행 2개 이상) ───────────────
# perplexity 독립 대조 통과본 (2026-11-03) — 고정. 파일이 없을 때만 생성
if not os.path.exists(os.path.join(OUT, "sample_check_3.md")):
    b3 = [r for r in rows if any(c[0] == V34 for c in r["_cites"])]
    new3 = {r["id"] for r in b3 if r["근거_파일"] == V34}
    seed3 = 20261104
    while True:
        random.seed(seed3)
        pick3 = sorted(random.sample(b3, 10), key=lambda r: int(r["id"][1:]))
        if sum(r["id"] in new3 for r in pick3) >= 2: break
        seed3 += 1
    S = [f"# sample_check_3.md — 단계 3 표본 대조 (v3.4 §11 묶음 {len(b3)}행 중 무작위 10행, seed {seed3}, 새 행 2개 이상 조건)\n",
         "> 원장 행 + 근거_위치의 원문 행 인용 (엔진·로그·v3.4). 식 열의 ' ‖ ' 는 원문 줄 구분자.",
         "> 엔진 행은 근거_위치 끝의 'v3.4 N행 [§11 표]' 토큰이 단계 3 에서 추가된 것. 대응 근거는 `index/map_step3.csv`, 자동 대조는 `index/verify_step3.md`.\n"]
    for r in pick3:
        S.append(f"## {r['id']} — {r['항목']}" + ("  (단계 3 새 행)" if r["id"] in new3 else "  (엔진 행 + v3.4 근거 추가)") + "\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', '\\|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b}" if b != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_3.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_4a.md (단계 4a 묶음: 새 행 + 근거 위치 추가 행, 각 2개 이상) ───────────────
b4 = sorted({id(m_["_row"]): m_["_row"] for m_ in CTX["step4a_map"]}.values(), key=lambda r: int(r["id"][1:]))
new4 = {m_["_row"]["id"] for m_ in CTX["step4a_map"] if m_["대응"] == "새 행"}
if not os.path.exists(os.path.join(OUT, "sample_check_4a.md")) or os.environ.get("REGEN_SAMPLE_4A"):
    seed4 = 20261105
    while True:
        random.seed(seed4)
        pick4 = sorted(random.sample(b4, 10), key=lambda r: int(r["id"][1:]))
        nn = sum(r["id"] in new4 for r in pick4)
        if nn >= 2 and 10 - nn >= 2: break
        seed4 += 1
    S = [f"# sample_check_4a.md — 단계 4a 표본 대조 (v3.4 본문 §12~§30 묶음 {len(b4)}행 중 무작위 10행, seed {seed4}, 새 행·근거 추가 행 각 2개 이상)\n",
         "> 원장 행 + 근거_위치의 원문 행 인용. 식 열의 ' ‖ ' 는 원문 줄 구분자.",
         "> 행 단위 규칙 R1~R4·같은 항목 판정은 `tools/build_step4a.py` 머리말, 줄별 결과는 `index/map_step4a.csv`, 자동 대조는 `index/verify_step4a.md`.\n"]
    for r in pick4:
        S.append(f"## {r['id']} — {r['항목']}" + ("  (단계 4a 새 행)" if r["id"] in new4 else "  (기존 행 + v3.4 본문 근거 추가)") + "\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', '\\|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b}" if b != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_4a.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_4b.md (단계 4a 묶음: 새 행 + 근거 위치 추가 행, 각 2개 이상) ───────────────
b4b_ = sorted({id(m_["_row"]): m_["_row"] for m_ in CTX["step4b_map"]}.values(), key=lambda r: int(r["id"][1:]))
new4b_ = {m_["_row"]["id"] for m_ in CTX["step4b_map"] if m_["대응"] == "새 행"}
if not os.path.exists(os.path.join(OUT, "sample_check_4b.md")) or os.environ.get("REGEN_SAMPLE_4B"):
    seed4 = 20261106
    while True:
        random.seed(seed4)
        pick4 = sorted(random.sample(b4b_, 10), key=lambda r: int(r["id"][1:]))
        nn = sum(r["id"] in new4b_ for r in pick4)
        kind_ = {m_["_row"]["id"]: m_["종류"] for m_ in CTX["step4b_map"]}
        sp = sum(1 for r in pick4 if kind_.get(r["id"]) in ("R3", "R5") or r["상태"] == "O 보류")
        if nn >= 2 and sp >= 2 and 10 - nn >= 1: break
        seed4 += 1
    S = [f"# sample_check_4b.md — 단계 4b 표본 대조 (v3.4 캡슐 블록 S79~S108, 2211행~끝 묶음 {len(b4b_)}행 중 무작위 10행, seed {seed4}, 새 행 2개 이상 · R3/R5/보류 행 2개 이상 · 근거 추가 행 1개)\n",
         "> 원장 행 + 근거_위치의 원문 행 인용. 식 열의 ' ‖ ' 는 원문 줄 구분자.",
         "> 행 단위 규칙 R1~R4·같은 항목 판정은 `tools/build_step4a.py` 머리말, 줄별 결과는 `index/map_step4b.csv`, 자동 대조는 `index/verify_step4b.md`.\n"]
    for r in pick4:
        S.append(f"## {r['id']} — {r['항목']}" + ("  (단계 4b 새 행)" if r["id"] in new4b_ else "  (기존 행 + v3.4 본문 근거 추가)") + "\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', '\\|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b}" if b != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_4b.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_5.md (단계 5: 세션 문서 S109~S119 + 번역·설계) ───────────────
b5 = sorted({id(m_["_row"]): m_["_row"] for m_ in CTX["step5_map"]}.values(), key=lambda r: int(r["id"][1:]))
k5 = {m_["_row"]["id"]: m_ for m_ in CTX["step5_map"]}
if not os.path.exists(os.path.join(OUT, "sample_check_5.md")) or os.environ.get("REGEN_SAMPLE_5"):
    WD = [r for r in b5 if r["근거_파일"].endswith("Session119_성과정리_v1.md") and re.match(r"^\| [WD]-[1-4] \|", r["식"])]
    seed5 = 20261107
    while True:
        random.seed(seed5)
        pick5 = random.sample(WD, 1) + random.sample([r for r in b5 if r not in WD], 9)
        c_t3 = sum(1 for r in pick5[1:] if k5[r["id"]]["종류"] == "T3")
        c_der = sum(1 for r in pick5 if r["근거_파일"].startswith("derived/"))
        c_oi = sum(1 for r in pick5 if k5[r["id"]]["표기"] in ("O", "I", "(O)", "(I)", "후보", "보류"))
        if c_t3 >= 1 and c_der >= 1 and c_oi >= 1: break
        seed5 += 1
    pick5 = sorted(pick5, key=lambda r: int(r["id"][1:]))
    S = [f"# sample_check_5.md — 단계 5 표본 대조 (세션 문서 S109~S119 + 번역·설계 묶음 {len(b5)}행 중 10행, seed {seed5})\n",
         "> 조건: S119 W-1~W-4·D-1~D-4 중 1행, 그 밖의 머리글 범주 (T3) 행 1개 이상, derived/ 변환본 행 1개 이상, O·I·후보·보류 표기 행 1개 이상.",
         "> 원장 행 + 근거_위치의 원문 행 인용. 식 열의 ' ‖ ' 는 원문 줄 구분자. md 표의 원문 '|' 는 '\\|' 로 이스케이프 (원문 문자 대체 없음).",
         "> 규칙 T1~T4·상태 선택·같은 항목 판정은 `tools/build_step5.py` 머리말, 줄별 결과는 `index/map_step5.csv`, 자동 대조는 `index/verify_step5.md`.\n"]
    for r in pick5:
        mk = k5[r["id"]]
        S.append(f"## {r['id']} — {r['항목'].replace('|', chr(92) + '|')}  ({mk['종류']}, {mk['대응']})\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', chr(92) + '|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b_, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b_}" if b_ != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b_, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_5.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_5b.md (단계 5 보충 표본, MAPSI 지정 범주 포함) — sample_check_5.md 는 그대로 ───────────────
if not os.path.exists(os.path.join(OUT, "sample_check_5b.md")) or os.environ.get("REGEN_SAMPLE_5B"):
    random.seed(20261109)
    s119 = "성과정리문서/EQI_Session119_성과정리_v1.md"; s117x = "성과정리문서/EQI_S117_플랑크_양자에너지_확장번역_v4.md"
    V17 = FILES[s117x]
    sec0 = next(i for i, l in enumerate(V17, 1) if l.startswith("## 0.")); sec1 = next(i for i, l in enumerate(V17, 1) if l.startswith("## 1."))
    CATS = [
        ("설계 배치 D 행 (S119 §4)", [r for r in b5 if r["근거_파일"] == s119 and re.match(r"^\| D-[1-4] \|", r["식"])]),
        ("철회 행", [r for r in b5 if r["상태"] == "철회"]),
        ("결정 대기 행", [r for r in b5 if r["상태"] == "결정 대기"]),
        ("범주 둘 머리글 아래 새 행 (T5, 보충 (1))", [r for r in b5 if k5[r["id"]]["종류"] == "T5"]),
        (f"S117 확장번역 v4 §0 ({sec0}~{sec1 - 1}행) — 관측·tip·부정적분 상수",
         [r for r in b5 if r["근거_파일"] == s117x and sec0 <= r["_cites"][0][1] < sec1 and re.search(r"TIP|tip|관측|부정적분|C =", r["식"])]),
    ]
    pick5b, why5b = [], {}
    for nm, cand in CATS:
        cand = [r for r in cand if r not in pick5b]
        if not cand: raise SystemExit(f"sample_check_5b: 후보 없음 — {nm}")
        r = random.choice(sorted(cand, key=lambda r: int(r["id"][1:]))); pick5b.append(r); why5b[r["id"]] = f"{nm} — 후보 {len(cand)}행 중 1"
    rest = sorted([r for r in b5 if r not in pick5b], key=lambda r: int(r["id"][1:]))
    for r in random.sample(rest, 5): pick5b.append(r); why5b[r["id"]] = "무작위 (단계 5 묶음)"
    pick5b.sort(key=lambda r: int(r["id"][1:]))
    S = [f"# sample_check_5b.md — 단계 5 보충 표본 (10행, seed 20261109)\n",
         "> MAPSI 지정 (2026-11-03): 설계 배치 D 행 1 · 철회 행 1 · 결정 대기 행 1 · 보충 (1) 로 새로 들어온 행 1 이상 · S117 확장번역 v4 §0 (관측·tip·부정적분 상수) 행 1 + 무작위 5.",
         "> 각 범주는 후보 중 seed 고정 무작위 1행. sample_check_5.md 는 수정하지 않음. md 표의 원문 '|' 는 '\\|' 이스케이프.\n",
         "| 원장 ID | 뽑힌 범주 |", "|---|---|"] + [f"| {r['id']} | {why5b[r['id']]} |" for r in pick5b] + [""]
    for r in pick5b:
        mk = k5[r["id"]]
        S.append(f"## {r['id']} — {r['항목'].replace('|', chr(92) + '|')}  ({mk['종류']}, {mk['대응']})\n")
        S.append(f"> 뽑힌 범주: {why5b[r['id']]}\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', chr(92) + '|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b_, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b_}" if b_ != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b_, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_5b.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_5c.md ('정의·원칙' 행 10개, MAPSI 지정 조건) ───────────────
if not os.path.exists(os.path.join(OUT, "sample_check_5c.md")) or os.environ.get("REGEN_SAMPLE_5C"):
    random.seed(20261110)
    dr = sorted([m_["_row"] for m_ in CTX["def_rows"]], key=lambda r: int(r["id"][1:]))
    s117x = "성과정리문서/EQI_S117_플랑크_양자에너지_확장번역_v4.md"
    must = [r for r in dr if r["근거_파일"] == s117x and r["_cites"][0][1] == 40]
    if len(must) != 1: raise SystemExit("sample_check_5c: S117 확장번역 §0 40행 정의·원칙 행 없음")
    why5c = {must[0]["id"]: "지정: S117 확장번역 v4 §0 40행"}
    pick5c = list(must)
    v34r = [r for r in dr if r["근거_파일"] == V34 and r not in pick5c]
    for r in random.sample(v34r, 3): pick5c.append(r); why5c[r["id"]] = f"v3.4 출처 (후보 {len(v34r)}행 중)"
    com = [r for r in dr if r not in pick5c and r["식"].startswith("|") and re.search(r"MAXWELL|π_EQI", r["식"])]
    for r in random.sample(com, 2): pick5c.append(r); why5c[r["id"]] = f"흔한 키워드 (MAXWELL·π_EQI) 표 행 (후보 {len(com)}행 중)"
    rest = [r for r in dr if r not in pick5c]
    for r in random.sample(rest, 10 - len(pick5c)): pick5c.append(r); why5c[r["id"]] = "무작위 (정의·원칙 행)"
    pick5c.sort(key=lambda r: int(r["id"][1:]))
    S = [f"# sample_check_5c.md — '정의·원칙' 행 표본 (정의·원칙 {len(dr)}행 중 10행, seed 20261110)\n",
         "> 조건 (MAPSI 2026-11-03): S117 확장번역 v4 §0 40행 포함 · v3.4 출처 3개 · 흔한 키워드 (MAXWELL·π_EQI 등) 표 행 2개 · 나머지 무작위.",
         "> 정의·원칙 행 = 상태 표기 없음 + TASKS §6 필수 키워드 1개 이상 + '=' 또는 '→'. 상태 빈칸 고정, 내려받기 금지, 비고에 걸린 키워드.",
         "> md 표의 원문 '|' 는 '\\|' 로 이스케이프 (원문 문자 대체 없음). 규칙: `tools/build_def.py`, 줄 목록: `index/map_def.csv`.\n",
         "| 원장 ID | 뽑힌 조건 |", "|---|---|"] + [f"| {r['id']} | {why5c[r['id']]} |" for r in pick5c] + [""]
    for r in pick5c:
        S.append(f"## {r['id']} — {r['항목'].replace('|', chr(92) + '|')}\n")
        S.append(f"> 뽑힌 조건: {why5c[r['id']]}\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', chr(92) + '|')} |")
        S.append("\n원문 인용:\n")
        for f, a, b_, role in r["_cites"]:
            S.append(f"- `{f}:{a}" + (f"-{b_}" if b_ != a else "") + f"` [{role}]\n  ```")
            for n in range(a, min(b_, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_5c.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── sample_check_6.md (단계 6 판 C, MAPSI 지정 조건) ───────────────
if not os.path.exists(os.path.join(OUT, "sample_check_6.md")) or os.environ.get("REGEN_SAMPLE_6"):
    random.seed(20261111)
    s63 = "성과정리문서/EQI_Session63_성과정리_v1.md"; s64 = "성과정리문서/EQI_Session64_성과정리_v1.md"
    V64 = FILES[s64]
    h3 = next(i for i, l in enumerate(V64, 1) if l.startswith("## §3.")); h4 = next(i for i, l in enumerate(V64, 1) if l.startswith("## §4."))
    h7 = next(i for i, l in enumerate(V64, 1) if l.startswith("## §7.")); h8 = next(i for i, l in enumerate(V64, 1) if l.startswith("## §8."))
    b6 = [m_["_row"] for m_ in CTX["step6_map"] if m_["대응"] == "새 행"] + [d["_row"] for d in CTX["def_rows6"]]
    byid = {r["id"]: r for r in rows}
    cf6 = [c for c in CF if int(c["id"][1:]) >= 17]
    CATS6 = [
        ("S63 §21 병합 (L0034·L0035 의 S63 새 근거 위치)", [byid[i] for i in ("L0034", "L0035")]),
        (f"S64 §3 ({h3}~{h4 - 1}행) 행", [r for r in b6 if r["근거_파일"] == s64 and h3 <= r["_cites"][0][1] < h4]),
        (f"S64 §3 ({h3}~{h4 - 1}행) 행", None),
        (f"S64 §7 H_ratio ({h7}~{h8 - 1}행) 행", [r for r in b6 if r["근거_파일"] == s64 and h7 <= r["_cites"][0][1] < h8 and "H_ratio" in r["식"]]),
        ("충돌 K 행 (K0017~) 에 걸린 등록 행", [byid[i] for i in sorted({i for c in cf6 for i in c["원장_ID"].split(", ")}, key=lambda x: int(x[1:]))]),
    ]
    pick6, why6, last = [], {}, None
    for nm, cand in CATS6:
        src_ = cand if cand is not None else last
        last = src_
        cand = [r for r in src_ if r not in pick6]
        if not cand: raise SystemExit(f"sample_check_6: 후보 없음 — {nm}")
        r = random.choice(cand); pick6.append(r)
        extra = ""
        if nm.startswith("충돌"):
            extra = " — " + ", ".join(c["id"] for c in cf6 if r["id"] in c["원장_ID"].split(", "))
        why6[r["id"]] = f"{nm} (후보 {len(src_)}행 중){extra}"
    rest = sorted([r for r in b6 if r not in pick6], key=lambda r: int(r["id"][1:]))
    for r in random.sample(rest, 5): pick6.append(r); why6[r["id"]] = "무작위 (단계 6 새 행)"
    pick6.sort(key=lambda r: int(r["id"][1:]))
    S = [f"# sample_check_6.md — 단계 6 표본 (판 C, 10행, seed 20261111)\n",
         "> 조건 (MAPSI 2026-11-03): S63 §21 병합 행 1 · S64 §3 행 2 · S64 §7 H_ratio 행 1 · 충돌 K 행에 걸린 등록 행 1 · 무작위 5.",
         "> 판 C = 단계 5 규칙 + 꼬리표 정규화 (단계 6 만) + 정의·원칙→등록 병합 + 좁힌 기준 머리글 사슬. md 표의 원문 '|' 는 '\\|' 이스케이프.",
         "> 규칙: `tools/build_step5.py`·`tools/build_def.py`, 줄 목록: `index/map_step6.csv`·`index/map_def6.csv`, 자동 대조: `index/verify_step6.md`.\n",
         "| 원장 ID | 뽑힌 조건 |", "|---|---|"] + [f"| {r['id']} | {why6[r['id']]} |" for r in pick6] + [""]
    for r in pick6:
        S.append(f"## {r['id']} — {r['항목'].replace('|', chr(92) + '|')}  ({r['행_종류']})\n")
        S.append(f"> 뽑힌 조건: {why6[r['id']]}\n")
        S.append("| 열 | 값 |\n|---|---|")
        for c in COLS[1:]:
            if r[c]: S.append(f"| {c} | {r[c].replace('|', chr(92) + '|')} |")
        S.append("\n원문 인용:\n")
        cites_ = r["_cites"] if r["id"] not in ("L0034", "L0035") else [c for c in r["_cites"] if c[0] in (ENG, LOG) or c[0] == s63][:12]
        for f, a, b_, role in cites_:
            S.append(f"- `{f}:{a}" + (f"-{b_}" if b_ != a else "") + f"` [{role}]" + (" ← 단계 6 새 근거 위치" if f == s63 else "") + "\n  ```")
            for n in range(a, min(b_, a + 9) + 1): S.append(f"  {n}: {FILES[f][n - 1].strip()}")
            S.append("  ```")
        S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
    with open(os.path.join(OUT, "sample_check_6.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")

# ─────────────── coverage.md 원장 ID 연결 ───────────────
import importlib.util
spec = importlib.util.spec_from_file_location("s1", os.path.join(ROOT, "tools", "build_step1.py"))
s1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
def span(ids):
    ids = sorted(ids, key=lambda x: int(x[1:]))
    if not ids: return ""
    out, a = [], 0
    while a < len(ids):
        b = a
        while b + 1 < len(ids) and int(ids[b + 1][1:]) == int(ids[b][1:]) + 1: b += 1
        out.append(ids[a] if a == b else f"{ids[a]}~{ids[b]}"); a = b + 1
    return ", ".join(out) + (f" ({len(ids)})" if len(ids) > 1 else "")
links = {}
for sec in HEAD: links[("eng", sec)] = span([r["id"] for r in rows if r["별칭"] == f"엔진 §{sec}"])
pl = {}
for r in rows:
    if r.get("_bline"): pl.setdefault(r["_bline"], []).append(r["id"])
for k, v in pl.items(): links[("p", k)] = span(v)
for h_, r_ in CTX["step4a_sec"] + CTX["step4b_sec"]:
    links.setdefault(("v34l", h_), []).append(r_["id"])
for k_ in [k for k in links if k[0] == "v34l"]:
    links[("v34", k_[1])] = span(sorted(set(links.pop(k_)), key=lambda x: int(x[1:])))
links[("v34", s3.T0)] = span([r["id"] for r in rows if any(c[0] == V34 and c[3] == "§11 표" for c in r["_cites"])])
s1.build_coverage(s1.build_doc_map(), links)

_ih = SI_CTX["inherit"]
print("상태 내려받기: 변경", [(c[0], c[2]) for c in _ih["changed"]], "| 무등급 표기 유지", len(_ih["nograde"]), "| 다음 순위에도 없음", _ih["none"], "| docx 미검색", _ih["skipped"], "| 비UTF-8", _ih["badenc"])
print("ledger rows:", len(rows), "| 새 행:", len(new_reg), "| 묶음 해체 원 ID:", sorted(SPLIT_FIRST), "| conflicts:", len(CF))
print("상태:", dict(Counter(r["상태"] for r in rows)))
print("엔진 § (§1~§42) 원장 미연결:", [s for s in HEAD if not any(r["별칭"] == f"엔진 §{s}" for r in rows)])

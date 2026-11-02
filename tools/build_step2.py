#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 2 (엔진 §) — TASKS_index_v3.md §4 단계 2
  index/ledger.csv        : 원장 첫 묶음 (엔진 v9.7 머리말 표 §1~§42 + 출력 로그 ✓ 줄 + v9.7 §P 블록)
  index/conflicts.csv     : 엔진 안에서 같은 항목이 다른 등급·귀속·값으로 적힌 곳 (판정 없음)
  index/sample_check_2.md : 무작위 10행 + 원문 해당 행 인용
원문 읽기 전용. 실행: python3 tools/build_step2.py (저장소 루트)
"""
import csv, os, re, random
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = "code/eqi_unified_engine_v9_7.py"                       # v9.7 완성판 (2,662행)
LOG = "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt"     # 완성판 mpmath 1.4.0 출력 (81,311 B) — 기준 로그
TERM = "code/eqi_unified_engine_v9_7_terminal.txt"              # 같은 실행의 Windows 콘솔 사본 (CRLF, 프롬프트 줄 포함)
OUT = os.path.join(ROOT, "index")
COLS = ["id", "항목", "식", "별칭", "상태", "등급", "오차", "층", "등록_세션", "근거_파일", "근거_위치",
        "엔진_§", "후속_정정", "키워드"]

def lines(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", newline="") as f:
        return [l.rstrip("\r\n") for l in f.read().split("\n")]

SRC, LG = lines(ENG), lines(LOG)
SRC_L = lambda n: SRC[n - 1]
LOG_L = lambda n: LG[n - 1]

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
])

def keywords(text, extra=()):
    ks = [k for k, pats in KW.items() if any(re.search(r"(?<![A-Za-z])" + re.escape(p) + r"(?![a-z])", text) for p in pats)]
    for e in extra:
        if e and e not in ks: ks.append(e)
    return ", ".join(ks)

# ─────────────── 머리말 표 (소스 docstring) ───────────────
HEAD = OrderedDict()           # § → dict(line, text, cont=[(line,text)])
cur = None
for i in range(1, 100):
    l = SRC_L(i)
    m = re.match(r"^║\s+§(\d+):\s*(.*?)\s*║?\s*$", l)
    if m:
        cur = int(m.group(1)); HEAD[cur] = dict(line=i, text=m.group(2).rstrip("║ ").strip(), cont=[])
        continue
    if cur and re.match(r"^║\s{4,}\S", l) and "──" not in l:
        HEAD[cur]["cont"].append((i, l.strip("║ ").strip()))
    elif "──" in l or l.startswith("╚"):
        cur = None

def head_session(s):
    m = re.search(r"\bS(\d+)(?:~S?(\d+))?\b", HEAD[s]["text"])
    if m: return "S" + m.group(1) + ("~S" + m.group(2) if m.group(2) else "")
    return ""

def head_grade(s):
    t = HEAD[s]["text"]
    for pat in ["True Zero(대수필연)", "True Zero", "Near TZ", "탐색 완료(NTZ)", "ALL True Zero"]:
        if pat in t: return pat
    return ""

# 소스 계산 블록 위치 (EQIConstants.__init__ 주석 145~1087행)
SRCPOS = {}
for i in range(140, 1090):
    m = re.match(r"^\s*#\s*(?:──\s*)?§(\d+)(?:,\s*§(\d+))?\s*[:：]", SRC_L(i))
    if m:
        for g in m.groups():
            if g and int(g) not in SRCPOS: SRCPOS[int(g)] = i
SRCSESS = {}
for s, i in SRCPOS.items():
    m = re.search(r"Session (\d+)(?:~(\d+))?", SRC_L(i))
    if m: SRCSESS[s] = "S" + m.group(1) + ("~S" + m.group(2) if m.group(2) else "")

# 로그 표제 행
LOGTITLE = {}
for i, l in enumerate(LG, 1):
    m = re.match(r"^\s{2}§(\d+)\s", l)
    if m and int(m.group(1)) not in LOGTITLE: LOGTITLE[int(m.group(1))] = i

def sess_for(s):
    h = head_session(s)
    if h: return h, f"머리말 {HEAD[s]['line']}행"
    if s in SRCSESS: return SRCSESS[s], f"소스 주석 {SRCPOS[s]}행"
    return "S38~S47 (묶음 표기)", "엔진 122행 'SECTION 0 … (Session 38~47 §1~§16)' — §별 세션 표기 없음"

GRADE_MAP = [("TZ(대수필연)", "대수필연", "대수필연"), ("True Zero(대수필연)", "대수필연", "대수필연"),
             ("Near True Zero", "Near TZ", "Near TZ"), ("Near TZ", "Near TZ", "Near TZ"),
             ("True Zero", "TZ", "TZ"), ("Real Zero", "Real Zero", "Real Zero"),
             ("대수필연", "대수필연", "대수필연"), ("TZ", "TZ", "TZ")]

def map_grade(label):
    label = (label or "").strip()
    for k, st, gr in GRADE_MAP:
        if label.startswith(k): return st, gr
    return "", "없음"

rows = []
def add(**kw):
    if "엔진_S" in kw: kw["엔진_§"] = kw.pop("엔진_S")
    r = {c: "" for c in COLS}; r.update({k: v for k, v in kw.items() if k in COLS})
    r["_refs"] = kw.get("_refs", []); r["_bline"] = kw.get("_bline")
    rows.append(r); return r

def eng_ref(s, extra=""):
    h = HEAD.get(s)
    parts = [f"§{s}"]
    if h: parts.append(f"머리말 {h['line']}행")
    if s in LOGTITLE: parts.append(f"로그 표제 {LOGTITLE[s]}행")
    if extra: parts.append(extra)
    return " / ".join(parts)

# ─────────────── A. 출력 로그 ✓ 줄 (§1~§42) ───────────────
CK = re.compile(r"^\s*✓\s+(.*?)\s*(?:오차=(\S+))?\s*(?:★\s*(.+?))?\s*$")
seen = set()
for i, l in enumerate(LG, 1):
    if not re.match(r"^\s*✓", l) or i > 470: continue          # 470행 이후 = 요약 블록 (범위 줄)
    m = CK.match(l)
    label, err, grade = m.group(1).strip(), m.group(2) or "", (m.group(3) or "").strip()
    ms = re.search(r"§(\d+)", label)
    if not ms:   # z0 표 '대칭 이벤트' — § 번호 없음
        tl = next(j for j, x in enumerate(LG, 1) if x.strip().startswith(label) and "✓" not in x)
        add(항목=label, 식=re.split(r"\s{2,}", LG[tl - 1].strip())[1] + " ‖ " + LG[tl].strip(),
            별칭="엔진 SECTION 1 z0 분류자 (§ 번호 없음)", 상태="", 등급="없음", 오차=f"오차={err}", 층="해당 없음",
            등록_세션="S38~S47 (묶음 표기)", 근거_파일=LOG, 근거_위치=f"로그 z0 표 {tl}행 / ✓ {i}행",
            엔진_S=f"SECTION 1 (소스 1530행) / 로그 표 {tl}행 / ✓ {i}행", 키워드=keywords(label, ["z0", "리만"]),
            _refs=[(LOG, tl), (LOG, i)])
        continue
    s = int(ms.group(1))
    seen.add(s)
    name = re.sub(r"^§\d+\s+", "", label).strip()
    name = re.sub(r"\s*\[별도 출력\]$", "", name)
    refs = [(LOG, i)]
    errtxt = f"{err}" if err else ""
    if s == 5 and not grade:      # [별도 출력] → 로그 24행
        grade = "True Zero"; errtxt = "7.2232098e-15 (로그 24행 'SI대비 오차%')"; refs.append((LOG, 24))
    if grade: errtxt = (errtxt + f" (★ {grade})").strip()
    st, gr = map_grade(grade)
    if not grade and re.match(r"GW|SGWB", label):   # z0 전채널 ✓ — 등급 표기 없음, 머리말 § 표기 따름
        st, gr = "", "없음"
    h = HEAD.get(s, {})
    formula = name if "=" in name or "<" in name else " / ".join([h.get("text", "")] + [t for _, t in h.get("cont", [])])
    if re.match(r"GW|SGWB", label):   # z0 표 수식 열
        tl = next(j for j, x in enumerate(LG, 1) if x.strip().startswith(label) and "✓" not in x)
        formula = re.split(r"\s{2,}", LG[tl - 1].strip())[1]; refs.append((LOG, tl))
    sess, sess_src = sess_for(s)
    layer = {19: "5체", 21: "7체", 24: "5체;7체"}.get(s, "해당 없음")
    if s in HEAD: refs.append((ENG, HEAD[s]["line"]))
    add(항목=name, 식=formula, 별칭=f"엔진 §{s}", 상태=st, 등급=gr, 오차=errtxt, 층=layer,
        등록_세션=sess, 근거_파일=ENG,
        근거_위치=f"소스 계산 블록 {SRCPOS.get(s, '')}행 (주석 '§{s}') / 세션 근거: {sess_src}",
        엔진_S=eng_ref(s, f"✓ {i}행 ({LOG})"), 키워드=keywords(label + " " + formula), _refs=refs)

# ─────────────── B. 머리말 표에만 있는 항목 (✓ 줄 없음) ───────────────
EXTRA = [  # (§, 머리말에서 찾을 원문 문자열 또는 None=머리말 행 전체, 항목명)
    (2, None, "MAXWELL 4차항 c4×GD⁴"), (9, None, "γ 3항 구조 (오일러-란다우어)"), (29, None, "전달함수 Q/MAXWELL≈34"),
    (10, "κ×Φ₀=1", "κ×Φ₀=1"), (17, "Fr=1/(2πφ)", "Fr=1/(2πφ)"), (27, "|ln(GAP)|/GAP_D2=213.5", "|ln(GAP)|/GAP_D2=213.5"),
    (30, "Sh=MAXWELL", "Sh=MAXWELL"),
]
for s, needle, name in EXTRA:
    h = HEAD[s]
    cand = [(h["line"], h["text"])] + h["cont"]
    if needle:
        ln, txt = next((a, b) for a, b in cand if needle in b)
        formula = needle
    else:
        ln, txt = h["line"], h["text"]; formula = txt
    hg = head_grade(s)
    st, gr = map_grade(hg)
    if hg == "탐색 완료(NTZ)": st, gr = "", "없음"
    sess, sess_src = sess_for(s)
    note = f"로그에 '§{s}' 표제·✓ 줄 없음 (grep) — 머리말 표·소스 위치만 기록" if not needle else "머리말 표 기재, 로그 ✓ 줄에는 이 식 없음"
    add(항목=name, 식=formula, 별칭=f"엔진 §{s}", 상태=st, 등급=gr,
        오차=(f"머리말 등급 표기: {hg}" if hg else ""), 층="해당 없음", 등록_세션=sess, 근거_파일=ENG,
        근거_위치=f"머리말 {ln}행 / " + (f"소스 계산 블록 {SRCPOS[s]}행" if s in SRCPOS else f"소스 주석 '§{s}:' grep 없음") + f" / 세션 근거: {sess_src}",
        엔진_S=eng_ref(s, note), 키워드=keywords(name + " " + txt), _refs=[(ENG, ln)])
# §15 메르센 수열 (로그 본문 ✓/✗ 표기, 줄 머리 ✓ 아님)
m1 = next(i for i, l in enumerate(LG, 1) if "D2 지수 메르센 수열" in l)
add(항목="D2 지수 메르센 수열 (k=3~5 ✓ / k=6~7 ✗ X7 금지)", 식=" ‖ ".join(LG[j - 1].strip() for j in (m1 + 1, m1 + 4)),
    별칭="엔진 §15", 상태="TZ", 등급="TZ", 오차="머리말 등급 표기: True Zero (§15 전체)", 층="해당 없음",
    등록_세션="S46", 근거_파일=LOG, 근거_위치=f"로그 {m1}–{m1+5}행 / 머리말 {HEAD[15]['line']}행 (S46)",
    엔진_S=eng_ref(15, "로그 본문 ✓·✗ 표기"), 키워드=keywords("메르센 X7 포논 D2"),
    _refs=[(LOG, m1 + 1), (LOG, m1 + 4)])
# §13 ε (로그 등급 Real Zero, ✓ 줄 없음)
ln = next(i for i, l in enumerate(LG, 1) if "ε = σ³/N³×K_EQI" in l)
add(항목="ε = σ³/N³×K_EQI", 식=LOG_L(ln).strip(), 별칭="엔진 §13", 상태="Real Zero", 등급="Real Zero",
    오차="0.005% (로그 원문 'Real Zero 0.005%')", 층="해당 없음", 등록_세션="S44", 근거_파일=LOG,
    근거_위치=f"로그 {ln}행 / 머리말 {HEAD[13]['line']}행 (S44)", 엔진_S=eng_ref(13, "✓ 줄 없음, 로그 본문 등급"),
    키워드=keywords("ε σ³ K_EQI Leibniz", ["포논"]), _refs=[(LOG, ln)])

# ─────────────── C. v9.7 §P 블록 ───────────────
PB = re.compile(r"^\s{2}(§P\S*)\s+(.*)$")
IT = re.compile(r"^\s{2}\[(§[A-Z][^\]]*)\]\s*(.*)$")
items, block = [], None
for i, l in enumerate(LG, 1):
    if i < 528: continue
    m = PB.match(l)
    if m: block = (m.group(1), m.group(2).strip(), i); continue
    if block and not IT.match(l) and l.strip() and not l.strip().startswith("─"):
        # 블록 머리 값·주석 (첫 항목 전) 또는 '[S..] 계보' 단독 주석 줄 → 별도 행 (값만)
        first_item = not any(x["block"] == block and not x.get("pre") for x in items)
        if (first_item and l.startswith("     ")) or re.match(r"^\s{5}\[S\d+[^\]]*\]", l):
            if items and items[-1]["block"] == block and items[-1].get("pre") and first_item:
                items[-1]["sub"].append((i, l.strip())); continue
            items.append(dict(id=f"{block[0]} 주석@{i}", body=l.strip(), line=i, block=block, sub=[],
                              pre=first_item and not re.match(r"^\s{5}\[S\d+", l)))
            continue
    m = IT.match(l)
    if m and block:
        sub = []
        j = i + 1
        while j <= len(LG) and LG[j - 1].startswith("     ") and not IT.match(LG[j - 1]):
            sub.append((j, LG[j - 1].strip())); j += 1
        items.append(dict(id=m.group(1), body=m.group(2), line=i, block=block, sub=sub))
    if l.startswith("  등급 집계"): break

def layer_of(btitle, label):
    m = re.match(r"^(\d)체 층", btitle)
    if m: return m.group(1) + "체"
    if "체 계층" in btitle:
        hits = set(re.findall(r"([357])체", label))
        if len(hits) == 1 and not re.search(r"[0-9]체.*[0-9]체", label): return hits.pop() + "체"
    return "해당 없음"
def parse(it):
    f = [x.strip() for x in it["body"].split(" | ")]
    d = dict(label=f[0], calc="", tgt="", err="", grade="", src="")
    if len(f) >= 6: d.update(calc=f[1], tgt=f[2], err=f[3], grade=f[4], src=f[5])
    elif len(f) == 4: d.update(calc=f[1], grade=f[2], src=f[3])
    elif len(f) == 5: d.update(calc=f[1], tgt=f[2], grade=f[3], src=f[4])
    return d

def gkey(i):
    for pat, rep in [(r"^(§P3-[12])\.k\d+$", r"\1"), (r"^(§P3-3)\.k\d+([a-f])$", r"\1.k*\2"), (r"^(§C-8)\.\S+$", r"\1"),
                     (r"^(§O-1)\.[a-z]$", r"\1"), (r"^(§L-7b)\.k\S+$", r"\1")]:
        if re.match(pat, i): return re.sub(pat, rep, i)
    return i

_bc = {}
for it in items: _bc.setdefault(it["block"][0], set()).add(it["block"][2])
DUP_B = {b for b, v in _bc.items() if len(v) > 1}
groups = OrderedDict()
for it in items:
    it["p"] = parse(it); groups.setdefault(gkey(it["id"]), []).append(it)

def src_session(src, label=""):
    m = re.search(r"S(\d+)(?:~S?(\d+))?", src) or re.search(r"\bS(\d+)\b()", label)
    if not m: return "S114 (v9.7 신규 블록)"
    return "S" + m.group(1) + (("~S" + m.group(2)) if m.group(2) else "")

UNOTE = next(i for i, l in enumerate(LG, 1) if "U-1~U-3 '귀속 확정' 판정 없음" in l)
S114_NOTE = "엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인')"
for key, g in groups.items():
    first = g[0]; p = first["p"]; bid, btitle, bline = first["block"]
    grades = {x["p"]["grade"] for x in g}
    members = [g] if len(grades) == 1 else [[x] for x in g]
    for mem in members:
        x0 = mem[0]; p = x0["p"]
        gid = key if len(mem) > 1 else x0["id"]
        st, gr = map_grade(p["grade"])
        if p["grade"] in ("정수", "-", "") or p["grade"].startswith("통과"): st, gr = "", "없음"
        if len(mem) > 1:
            errs = [m_["p"]["err"] for m_ in mem if m_["p"]["err"]]
            err = f"{len(mem)}행: " + ", ".join(errs) if errs else ""
            ids = f"{mem[0]['id']}~{mem[-1]['id']}"
            lab = p["label"] + f" 외 {len(mem)-1}행 ({ids})"
            loc = f"로그 {mem[0]['line']}–{mem[-1]['line']}행 ({len(mem)}행)"
        else:
            err = p["err"]; lab = p["label"]; ids = x0["id"]; loc = f"로그 {x0['line']}행"
        if p["grade"]: err = (err + f" (판정 열: {p['grade']})").strip()
        formula = lab
        if not p["calc"] and x0["sub"]:
            formula = lab + " ‖ " + " ‖ ".join(t for _, t in x0["sub"][:3])
        fix = ""
        if x0["id"] == "§L-4":
            fix = "S114 정정: 기대값 −7/4. 구 기대값 +1/4 (S77 §48-E, 단위계 철학 v3 §11.1, v3.4 2104행) 산술 오기 — 엔진 소스 2325–2326행"
        if x0["id"] == "§P7-3":
            fix = "f×We²×Re 폐기(S92) — 원문 표기 '폐기' (엔진 로그 696행)"
        reg = src_session(p["src"], p["label"])
        if x0["id"] == "§C-주석":
            st = "미적용"
            fix = "S114 v9.7: 현 체제(단일 항등원) 미적용, 계산하지 않음 — 엔진 소스 2408–2409행 / 로그 743행"
        name = re.sub(r"\s*\(기대[^)]*\)", "", lab).strip()
        if x0.get("pre") is not None and "주석@" in x0["id"]:
            ids = f"{bid} 주석 (로그 {x0['line']}행)"
            name = (f"{bid} 블록 머리 (값·주석)" if x0["pre"] else lab)
            if x0["pre"]: formula = lab + "".join(" ‖ " + t for _, t in x0["sub"])
            if x0["pre"] and x0["sub"]: loc = f"로그 {x0['line']}–{x0['sub'][-1][0]}행 ({len(x0['sub'])+1}행)"
        if x0["id"] == "§C-주석": name = "S64 §6.2 '중력질량 = E_topo² = φ^1.5 = 관성질량' (v15 이중 항등원 표기)"
        layer = layer_of(btitle, lab)
        if bid.startswith("§P-DS") and re.match(r"§U", x0["id"]):
            fix_u = f"로그 {UNOTE}행: '※ U-1~U-3 \'귀속 확정\' 판정 없음 (Architect/MAPSI 판단).'"
        else:
            fix_u = ""
        add(항목=name, 식=formula, 별칭=f"엔진 v9.7 {ids}" + (f"; 출처 표기 '{p['src']}'" if p["src"] else ""),
            상태=st, 등급=gr, 오차=err, 층=layer, 등록_세션=reg, 근거_파일=LOG,
            근거_위치=(f"{bid} '{btitle.split(' [')[0]}' 블록" if bid in DUP_B else f"{bid} 블록") + f" (표제 {bline}행) / {loc}",
            엔진_S=f"v9.7 {bid} {ids} / {S114_NOTE} / 소스 class EQIPhononLattice" + (f" / {fix_u}" if fix_u else ""),
            후속_정정=fix, _bline=bline, 키워드=keywords(btitle + " " + formula, [bid]),
            _refs=[(LOG, m_["line"]) for m_ in mem][:3])

for n, r in enumerate(rows, 1): r["id"] = f"L{n:04d}"

with open(os.path.join(OUT, "ledger.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

# ─────────────── conflicts.csv ───────────────
def find(pred):
    return next(r["id"] for r in rows if pred(r))
TL = lines(TERM)
TK = lines("code/TASKS.md")
T7C = next(i for i, l in enumerate(TK, 1) if "L-7c 브래그" in l)
L7C = next(i for i, l in enumerate(LG, 1) if "TASKS 문구 '7차부터 < 1'" in l)
CF = [
    dict(항목="§12 ratio(34)×34 vs 2/R_mem 등급", 원장_ID=find(lambda r: r["항목"].startswith("ratio(34)")),
         위치_A=f"{ENG} 머리말 24행", 내용_A="'S43  True Zero'",
         위치_B=f"{LOG} 46행 / 395행", 내용_B=f"46행 '{LOG_L(46).strip()}' / 395행 '★ Near TZ'",
         비고="같은 오차값(5.4e-3%)에 등급 표기 셋 (머리말·로그 본문 True Zero, ✓ 줄 Near TZ). 판정 없음"),
    dict(항목="a₅_eff = 5σ⁵/4 의 § 귀속", 원장_ID=find(lambda r: r["항목"].startswith("a₅_eff")),
         위치_A=f"{ENG} 머리말 26행 / {LOG} 57행", 내용_A="§14 (머리말 'a₅_eff=5σ⁵/4, 포논 굴절', 로그 §14 블록 57행)",
         위치_B=f"{LOG} 399행 / 71–72행", 내용_B="'✓ §15 a₅_eff = 5/4×σ⁵' (로그 §15 블록 71–72행에도 a₅_eff 식)", 비고="같은 식이 §14·§15 양쪽에 귀속 표기. 판정 없음"),
    dict(항목="§L-6 H&Q 2.95μm 등급", 원장_ID=find(lambda r: "§L-6" in r["별칭"] and "주석" not in r["별칭"]),
         위치_A=f"{LOG} 709행", 내용_A="v9.7 판정 'Near TZ'", 위치_B=f"{LOG} 709행 (원문 인용)", 내용_B="'원문 TZ(S99)'",
         비고="엔진이 두 등급 병기. S99 원문 대조는 단계 6. 판정 없음"),
    dict(항목="Stefan–Boltzmann 지수 기대값", 원장_ID=find(lambda r: "§L-4" in r["별칭"]),
         위치_A="성과정리문서/EQI_연구지침_프로젝트용_v3_4.md 2104행", 내용_A="§48-E: Stefan-Boltzmann σ ∝ φ^(+1/4) = M_p = 수명 지수  [TZ]",
         위치_B=f"{ENG} 2325–2326행 / {LOG} 707행", 내용_B="기대값 −7/4, '+1/4 … 산술 오기' (정정 S114)",
         비고="엔진 측 정정 기록 있음. 원장 후속_정정 열에 옮김. v3.4 원문 미수정"),
    dict(항목="§L-7c 브래그 N·σ^(2k+1)/2 < 1 시작 차수 (문구)", 원장_ID=find(lambda r: "§L-7c" in r["별칭"]),
         위치_A=f"code/TASKS.md {T7C}행", 내용_A=TK[T7C - 1].strip(),
         위치_B=f"{LOG} {L7C}행 (같은 줄, 엔진 계산 표기)", 내용_B=LOG_L(L7C).strip(),
         비고="엔진 출력 '※ TASKS 문구 … 불일치' 줄. 판정 없음"),
]
for n, c in enumerate(CF, 1): c["id"] = f"K{n:04d}"
ccols = ["id", "항목", "원장_ID", "위치_A", "내용_A", "위치_B", "내용_B", "비고"]
with open(os.path.join(OUT, "conflicts.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ccols); w.writeheader(); w.writerows(CF)

# ─────────────── sample_check_2.md ───────────────
random.seed(20261102)
pick = sorted(random.sample(rows, 10), key=lambda r: r["id"])
S = ["# sample_check_2.md — 단계 2 표본 대조 (무작위 10행, seed 20261102)\n",
     "> 원장 행 + 원문 해당 행 인용. MAPSI / perplexity 대조용. 틀린 행이 있으면 단계 2 전체 재점검 (TASKS §7).\n"]
src_cache = {ENG: SRC, LOG: LG}
for r in pick:
    S.append(f"## {r['id']} — {r['항목']}\n")
    S.append("| 열 | 값 |\n|---|---|")
    for c in COLS[1:]:
        if r[c]: S.append(f"| {c} | {r[c].replace('|', '∣')} |")
    S.append("\n원문 인용:\n")
    for fpath, ln in r["_refs"]:
        S.append(f"- `{fpath}:{ln}`\n  ```\n  {src_cache[fpath][ln-1].strip()}\n  ```")
    S.append("\n대조 결과: [ ] 일치  [ ] 불일치 → 메모:\n")
with open(os.path.join(OUT, "sample_check_2.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(S) + "\n")

# coverage.md 원장 ID 연결 (단계 1 스크립트 재사용)
import importlib.util
spec = importlib.util.spec_from_file_location("s1", os.path.join(ROOT, "tools", "build_step1.py"))
s1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
def span(ids):
    return "" if not ids else (ids[0] if len(ids) == 1 else f"{ids[0]}~{ids[-1]} ({len(ids)})" if
            int(ids[-1][1:]) - int(ids[0][1:]) == len(ids) - 1 else ", ".join(ids))
links = {}
for sec in HEAD:
    links[("eng", sec)] = span([r["id"] for r in rows if r["별칭"] == f"엔진 §{sec}"])
for r in rows:
    if r.get("_bline"): links.setdefault(("p", r["_bline"]), []).append(r["id"])
for k in list(links):
    if k[0] == "p": links[k] = span(links[k])
s1.build_coverage(s1.build_doc_map(), links)

from collections import Counter
print("ledger rows:", len(rows), "| conflicts:", len(CF))
print("상태:", Counter(r["상태"] for r in rows)); print("등급:", Counter(r["등급"] for r in rows))
print("엔진 § (§1~§42) 원장 미연결:", [s for s in HEAD if not any(r["별칭"] == f"엔진 §{s}" for r in rows)])

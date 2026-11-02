#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 1 (골격) 산출물 재생성 스크립트 — TASKS_index_v3.md §4 단계 1
  index/doc_map.csv   : 원문 폴더 전체 파일 목록 + 세션·종류·판본·최종판·동명판본·바이트
  index/coverage.md   : v3.4 § 제목 목록, 엔진 v9.7 § 목록 (연결 열은 빈칸 — 단계 2 이후 채움)
원문은 읽기만 한다. 실행: python3 tools/build_step1.py  (저장소 루트에서)
"""
import csv, os, re, unicodedata
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = "성과정리문서"           # TASKS §1 의 docs/ 에 해당 (저장소 실제 폴더, 평면 구조)
V34  = f"{DOCS}/EQI_연구지침_프로젝트용_v3_4.md"
ENG  = "code/eqi_unified_engine_v9_7.py"
ENGO = "code/eqi_unified_engine_v9_7_terminal_mpmath140.txt"   # v9.7 완성판 mpmath 1.4.0 출력 (81,311 B)
OUT  = os.path.join(ROOT, "index")

nfc = lambda s: unicodedata.normalize("NFC", s)

def read_lines(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read().split("\n")

# ─────────────────────────── doc_map ───────────────────────────
SESS = re.compile(r"^EQI_Session(\d+)_(성과정리|인수인계|추가논의)(?:_v(\d+))?(_final)?(?: \((\d+)\))?\.(md|docx)$")
SESS_OLD = re.compile(r"^session_(\d+)_(.+)\.md$")
VER = re.compile(r"_v(\d+)(?:_(\d+))?")

def classify(name):
    if name == "EQI_연구지침_프로젝트용_v3_4.md": return "v3_4"
    if SESS.match(name) or SESS_OLD.match(name): return "sessions"
    if name.startswith(("EQI_S116_", "EQI_S117_")): return "translations"
    if name.startswith(("EQI_이론적_기반", "EQI_Section118_", "EQI_단위계_철학",
                        "EQI_SESSION_PREAMBLE", "EQI_SESSION_START_TRIGGER", "EQI_119_압축본")):
        return "foundation"
    if name.startswith(("EQI_클로드용_", "EQI_클로드_전달메모")): return "index_v1_1"
    return "기타(역사문서·메모)"

def base_and_ver(name):
    stem, ext = os.path.splitext(name)
    dup = re.search(r" \((\d+)\)$", stem)
    if dup: stem = stem[:dup.start()]
    m = None
    for m in VER.finditer(stem): pass
    if m:
        ver = m.group(1) + ("." + m.group(2) if m.group(2) else "")
        base = stem[:m.start()] + stem[m.end():]
    else:
        ver, base = "", stem
    return base, ver, (dup.group(1) if dup else "")

# MAPSI 보충 지시 (단계 1 확인 후): 최종판 판정은 S118 색인 작업지시서 v2 기록 우선, 그 밖은 v 숫자 최대.
V2DOC = f"{DOCS}/EQI_색인_작업지시서_v2.md"
FINAL_OVERRIDE = {   # 파일명 → 근거 (작업지시서 v2 70·87·89·93행)
    "EQI_Session111_성과정리_v5.md": "S118 색인 작업지시서 v2 70행·87행 (S111 v5)",
    "EQI_Session113_성과정리_v6.md": "S118 색인 작업지시서 v2 70행·89행 (S113 v6, 파일 1행 헤더 'v5' 는 오기)",
    "EQI_Session117_성과정리_v2.md": "S118 색인 작업지시서 v2 70행·93행 (S117 성과정리 v2)",
}
CURRENT = {          # MAPSI 지시: 현행 표시
    "EQI_SESSION_PREAMBLE_v6.md": "현행 (MAPSI 지시, 색인 단계 1 확인 후 보충)",
    "EQI_SESSION_START_TRIGGER_v7.md": "현행 (MAPSI 지시, 색인 단계 1 확인 후 보충)",
}

def vkey(v):
    return tuple(int(x) for x in v.split(".")) if v else (0,)

def build_doc_map():
    rows = []
    for name in sorted(os.listdir(os.path.join(ROOT, DOCS))):
        p = os.path.join(ROOT, DOCS, name)
        if not os.path.isfile(p): continue
        n = nfc(name)
        ext = os.path.splitext(n)[1].lstrip(".").lower()
        sess = kind = ""
        m = SESS.match(n)
        if m:
            sess, kind = f"S{int(m.group(1))}", m.group(2)
            base, ver = f"S{int(m.group(1))}_{kind}", (m.group(3) or "")
            dupn = m.group(5) or ""
            note = []
            if m.group(4): note.append("파일명 _final")
        else:
            m2 = SESS_OLD.match(n)
            base, ver, dupn = base_and_ver(n)
            note = []
            if m2:
                sess, kind = f"S{int(m2.group(1))}", "세션기록(구형식)"
                base = f"S{int(m2.group(1))}_{m2.group(2)}"
        if dupn: note.append(f"파일명 ' ({dupn})' 사본 표기")
        if ext in ("docx", "pdf"): note.append(f"{ext} — 텍스트 추출 필요")
        rows.append(dict(세션=sess, 종류=kind, 분류=classify(n), 파일명=n,
                         판본=("v" + ver) if ver else "", 바이트=os.path.getsize(p), 형식=ext,
                         _base=base, _ver=ver, 비고="; ".join(note),
                         상대경로=f"{DOCS}/{n}"))
    groups = defaultdict(list)
    for r in rows: groups[r["_base"]].append(r)
    for base, g in groups.items():
        top = max(vkey(r["_ver"]) for r in g)
        forced = [r for r in g if r["파일명"] in FINAL_OVERRIDE]
        for r in g:
            r["동명판본수"] = len(g)
            if forced:
                r["최종판"] = "Y" if r in forced else "N"
                r["최종판_근거"] = FINAL_OVERRIDE.get(r["파일명"], "작업지시서 v2 지정판 아님")
            else:
                r["최종판"] = "Y" if vkey(r["_ver"]) == top else "N"
                r["최종판_근거"] = ("v 숫자 최대" if len(g) > 1 else "단일 판본") if r["최종판"] == "Y" else "v 숫자 최대 아님"
            r["현행"] = CURRENT.get(r["파일명"], "")
            if len(g) > 1:
                others = [x["파일명"] for x in g if x is not r]
                r["동명판본"] = " | ".join(f'{o} ({next(x["바이트"] for x in g if x["파일명"]==o)}B)' for o in others)
            else:
                r["동명판본"] = ""
    def sk(r):
        s = int(r["세션"][1:]) if r["세션"] else 10**6
        return (s, r["분류"], r["_base"], vkey(r["_ver"]))
    rows.sort(key=sk)
    cols = ["세션", "종류", "분류", "파일명", "판본", "최종판", "최종판_근거", "현행", "동명판본수", "동명판본", "바이트", "형식", "비고", "상대경로"]
    with open(os.path.join(OUT, "doc_map.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return rows

# ─────────────────────────── v3.4 § 제목 ───────────────────────────
def v34_headings():
    out, in_code = [], False
    for i, line in enumerate(read_lines(V34), 1):
        if line.lstrip().startswith("```"): in_code = not in_code; continue
        if in_code: continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m: out.append((i, len(m.group(1)), m.group(2).strip()))
    return out

# ─────────────────────────── 엔진 § ───────────────────────────
def engine_sections():
    src, log = read_lines(ENG), read_lines(ENGO)
    head = {}
    for i, line in enumerate(src[:120], 1):
        m = re.match(r"^║\s+§(\d+):\s*(.*?)\s*║?\s*$", line)
        if m: head[int(m.group(1))] = (i, m.group(2).rstrip("║ ").strip())
    title, checks = defaultdict(list), defaultdict(list)
    for i, line in enumerate(log, 1):
        if "✓" in line:
            for m in re.finditer(r"§(\d+)\b", line):
                checks[int(m.group(1))].append(i); break
            continue
        m = re.match(r"^\s{2}§(\d+)\s", line)
        if m: title[int(m.group(1))].append(i)
    # v9.7 추가 블록 (§P*, §L*, §C*)
    pblocks, cur = [], None
    for i, line in enumerate(log, 1):
        m = re.match(r"^\s{2}(§P\S*)\s+(.*)$", line)
        if m:
            cur = dict(id=m.group(1), title=m.group(2).strip(), line=i, items=[]); pblocks.append(cur); continue
        m = re.match(r"^\s{2}\[(§[A-Z][^\]]*)\]", line)
        if m and cur: cur["items"].append((m.group(1), i))
    psrc = defaultdict(list)
    for i, line in enumerate(src, 1):
        for m in re.finditer(r"(§P[0-9]+|§P-[A-Za-z]+)(?![A-Za-z0-9])", line):
            psrc[m.group(1)].append(i)
    return head, title, checks, pblocks, psrc

def rng(ls):
    if not ls: return ""
    if len(ls) == 1: return str(ls[0])
    return f"{ls[0]}–{ls[-1]} ({len(ls)}행)" if ls == list(range(ls[0], ls[-1] + 1)) else ", ".join(map(str, ls))

def build_coverage(rows, links=None):
    links = links or {}
    hs = v34_headings()
    head, title, checks, pblocks, psrc = engine_sections()
    L = []
    L.append("# coverage.md — 누락 점검표" + (" (단계 2: 엔진 § 연결)" if links else " (단계 1 초안: 목록만, 원장 연결 없음)") + "\n")
    L.append("> 생성: `python3 tools/build_step1.py`. 행 번호는 스크립트의 행 단위 grep 결과 그대로.")
    L.append("> `원장 ID` 열: B 표(엔진)는 단계 2 에서 연결, A 표(v3.4)는 단계 3~4 에서 연결. 빈칸 = 아직 연결 안 함 (\"미등록\" 판정 아님).\n")
    # A. v3.4
    L.append(f"## A. v3.4 § 제목 목록 — `{V34}` ({len(read_lines(V34))}행, 머리글 {len(hs)}개)\n")
    L.append("| 행 | 수준 | 제목 | 원장 ID |")
    L.append("|---|---|---|---|")
    for i, lv, t in hs:
        L.append(f"| {i} | {'#'*lv} | {t.replace('|', '∣')} | {links.get(('v34', i), '')} |")
    L.append("")
    L.append("### A-1. 구조 메모 (원문 열람 확인 사항, 판정 아님)\n")
    caps = [(i, t) for i, lv, t in hs if t.startswith("[S9") and lv == 3]
    L.append("- TASKS v3 §4 단계 4 선행 확인: **S92~S97 캡슐 블록 존재** — " +
             ", ".join(f"`{t.split(']')[0]}]` {i}행" for i, t in caps if re.match(r"\[S9[2-7]\]", t)) +
             " (S88~S91 캡슐화 `■` 블록(2701행) 안에 위치. 별도 `■ S92~S97` 머리글은 없음)")
    order = [re.match(r"(§[0-9]+[a-z]?)", t).group(1) for i, lv, t in hs if lv == 2 and re.match(r"§[0-9]", t)]
    L.append(f"- `##` 수준 § 머리글 순서 (원문 배치 그대로): {' → '.join(order)}")
    L.append("- `## §20` 머리글 행은 없음 (grep `^## §20`). §20 은 §11 표 안의 행(1008행)으로 존재 — 단계 3 에서 연결")
    L.append("")
    # B. engine
    L.append(f"## B. 엔진 § 목록 — `{ENG}` 머리말 표 + `{ENGO}` 출력 로그\n")
    L.append("열: 머리말 행 = 엔진 소스 머리말 표(docstring) 행 / 출력 표제 행 = 로그의 `  §N ...` 줄 / ✓ 행 = 로그 검증 줄\n")
    L.append("| § | 머리말 표 내용 (원문) | 머리말 행 | 출력 표제 행 | ✓ 행 | 원장 ID |")
    L.append("|---|---|---|---|---|---|")
    for s in sorted(set(head) | set(title) | set(checks)):
        h = head.get(s, ("", ""))
        L.append(f"| §{s} | {h[1].replace('|', '∣')} | {h[0]} | {rng(title.get(s, []))} | {rng(checks.get(s, []))} | {links.get(('eng', s), '')} |")
    nck = sum(len(v) for v in checks.values())
    L.append(f"\n- 머리말 표 § 수: {len(head)} (§{min(head)}~§{max(head)}), ✓ 줄 중 §번호 포함: {nck}")
    allck = sum(1 for l in read_lines(ENGO) if "✓" in l)
    L.append(f"- 로그 전체 ✓ 줄: {allck} (z0 전채널 블록 379–382행 ✓ 은 `[§N]` 표기로 위 표에 포함)")
    L.append("- §3 의 480행, §11 의 481행은 최종 요약 블록의 범위 줄 (`§3~§7 z0 전채널 ✓`, `§11~§35 ALL VERIFIED ✓`) — 개별 검증 줄 아님")
    noout = [s for s in sorted(head) if s not in title and s not in checks]
    if noout:
        L.append(f"- 머리말 표에는 있으나 출력 로그에서 `§N` 표제·✓ 줄이 grep 되지 않는 §: "
                 + ", ".join(f"§{s}" for s in noout) + " — 단계 2 에서 소스 본문 위치로 확인 (판정 아님)")
    L.append("")
    L.append("### B-1. v9.7 추가 블록 (§P0~, S114) — 출력 로그 기준 (완성판 2,662행 / mpmath 1.4.0 출력)\n")
    L.append("| 블록 | 표제 (원문) | 로그 표제 행 | 하위 항목 수 | 하위 항목 ID (로그 행) | 소스 언급 행 | 원장 ID |")
    L.append("|---|---|---|---|---|---|---|")
    for b in pblocks:
        ids = ", ".join(f"{i}({ln})" for i, ln in b["items"]) if len(b["items"]) <= 12 else \
              f"{b['items'][0][0]}({b['items'][0][1]}) … {b['items'][-1][0]}({b['items'][-1][1]})"
        L.append(f"| {b['id']} | {b['title'].replace('|','∣')} | {b['line']} | {len(b['items'])} | {ids} | {rng(psrc.get(b['id'], []))} | {links.get(('p', b['line']), '')} |")
    L.append("")
    # C. sessions
    L.append("## C. 세션 문서 보유 현황 (doc_map.csv 요약)\n")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["세션"]: by[int(r["세션"][1:])][r["종류"]].append(r)
    L.append("| 세션 | 성과정리 (최종판) | 인수인계 (최종판) | 기타 | 동명 판본 |")
    L.append("|---|---|---|---|---|")
    def fin(lst):
        return ", ".join(r["파일명"] for r in lst if r["최종판"] == "Y")
    for s in sorted(by):
        d = by[s]
        dup = "; ".join(f'{r["파일명"]}({r["바이트"]}B)' for k in d for r in d[k] if r["동명판본수"] > 1)
        etc = ", ".join(r["파일명"] for k in d if k not in ("성과정리", "인수인계") for r in d[k])
        L.append(f"| S{s} | {fin(d.get('성과정리', [])) or '—'} | {fin(d.get('인수인계', [])) or '—'} | {etc} | {dup} |")
    have = sorted(by)
    miss = [s for s in range(min(have), max(have) + 1) if s not in by]
    L.append(f"\n- 저장소 세션 문서 범위: S{min(have)}~S{max(have)} (S1~S{min(have)-1} 세션 문서는 저장소에 없음)")
    L.append("- MAPSI 보충 지시: S1~S20·S23 은 v3.4 본문을 원천으로 한다 (단계 4). 단계 6 은 저장소에 있는 S21 이후만 보충")
    if miss: L.append(f"- 범위 내 파일 없는 세션 번호: {', '.join('S'+str(s) for s in miss)}")
    no_s = [f"S{s}" for s in have if "성과정리" not in by[s] and s >= 24]
    no_h = [f"S{s}" for s in have if "인수인계" not in by[s] and s >= 24]
    L.append(f"- 성과정리 파일 없음 (S24 이후): {', '.join(no_s) or '없음'}")
    L.append(f"- 인수인계 파일 없음 (S24 이후): {', '.join(no_h) or '없음'}")
    L.append("- '—' 은 저장소에 파일이 없다는 뜻이다. 세션 내용 부재를 뜻하지 않는다 (v3.4 캡슐 블록 등에 수록 가능)")
    L.append("")
    with open(os.path.join(OUT, "coverage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return hs, head, pblocks

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = build_doc_map()
    hs, head, pblocks = build_coverage(rows)
    print(f"doc_map rows={len(rows)}  v3.4 headings={len(hs)}  engine §={len(head)}  v9.7 P-blocks={len(pblocks)}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 7 (후속_정정 채우기) — MAPSI 결정 (2026-11-03) 방식 (나). tools/build_step2.py 가 상태 내려받기 뒤, 원장 쓰기 전에 apply() 를 부른다.
  · 상태 열은 바꾸지 않는다 (원문 등록 당시 표기 유지). 새 열 '현행_상태' = 후속 정정으로 바뀐 행만 채움. 빈칸 = 상태와 같음.
  · 현행_상태를 채운 행은 후속_정정 열에 정정 원천 행 ID·세션·원문 표기·위치를 적는다.
  · 원래 상태와 같은 재확정 (같은 항목, 같은 상태) 은 현행_상태를 비우고 후속_정정에만 적는다.
  · '폐기' 는 원문 표기 그대로 현행_상태에 적는다 (철회·미적용으로 바꾸지 않음).
  · 이번 반영 = 문자열 규칙 (tools/dryrun_step7.py 와 같은 규칙 F / N / S) 으로 연결된 대상 6행만.
    규칙 결과가 MAPSI 승인 목록 (APPROVED) 과 다르면 쓰지 않고 멈춘다.
  · 연결 없는 원천 행: 키워드 (따옴표 문자열 · 수식 기호 조각 · §N) 가 이른 세션 등록 행 어디에도 없으면 '대상 없음',
    있으면 '이름으로만 적힌 정정 → 짝 후보 (미반영)' — index/step7_pair_candidates.md 에 목록만 (MAPSI·perplexity 확인 후 확정 쌍만 다음에 반영).
  산출: index/map_step7.csv, index/step7_report.md, index/step7_pair_candidates.md, index/step7_status_snapshot.csv (단계 7 직전 상태 열).
"""
import csv, os, re
from collections import Counter, OrderedDict

LBL = re.compile(r"^\s*(?:\d체\[[A-Z0-9]+\]|§[A-Z]'?|\[[A-Z]'?\]|[*□]+)\s*")
nz = lambda s: re.sub(r"\s+", "", s)
CORR = re.compile(r"(정정|오기|폐기|철회|미적용)(?!\s*(?:아님|않음|없음|하지 않))")
OBJ = re.compile(r"([^\s,/(:→\"'|]*[=×÷^²³⁴§φ√/·][^\s,/(:→\"'|]*)\s*(?:표현\s*|주장\s*|가설\s*검증\s*→\s*)?(폐기|철회)")
SECW = re.compile(r"(§\d+[a-z]?)(?![\d-])[^,/|]*?(폐기|철회)")
# MAPSI 승인 (2026-11-03): 대상 6행 / 원천 4행. 현행_상태 값은 원천 원문의 낱말 그대로.
APPROVED = {"L3335": "폐기", "L4621": "폐기", "L0505": "철회", "L3641": "철회", "L3653": "철회", "L0809": ""}
# MAPSI 확정 쌍 (2026-11-03, 짝 후보 확인 1차): 원천은 원문 파일 줄 (원장 행 아님). 원천 줄은 인용 문자열로 grep 해 1줄일 때만, 대상은 주 근거 위치로 원장에서 찾음.
CONFIRMED = [dict(일자="2026-11-03", 원천_파일="code/eqi_extended_v2_7_terminal.txt", 원천_문자열="S117 '≈ h·ν_p·ln2 (0.13 %)' 는 단위 혼합 → 철회",
                  MAPSI_행=166, 대상_위치="문서 성과정리문서/EQI_S117_플랑크_양자에너지_확장번역_v4.md 195행", 현행_상태="철회",
                  같은_기록_행="L0855")]      # L0855 = 같은 철회를 적은 원장 행 (식에 '확장 번역 v4 195행' · '철회')
# MAPSI 확정 '대상 없음' (2026-11-03)
NO_TARGET = {"L0869": "S119 성과정리 §5 n* 행 — 이전 세션 등록 없음",
             "L0870": "HKr 'n̄ ≫ σ 극한' 문장 — 코드 출력 문장, 원장 항목 아님"}
QUOTE = re.compile(r"['‘’\"“”`]([^'‘’\"“”`]{3,60})['‘’\"“”`]")
SYMTOK = re.compile(r"[^\s,()（）\[\]{}:;|→\"'“”‘’`*]*[=×÷^²³⁴φ√·≈][^\s,()（）\[\]{}:;|→\"'“”‘’`*]*")
SECT = re.compile(r"§\d+[a-z]?(?:-[A-Za-z0-9]+)?(?![\d])")
esc = lambda x: x.replace("|", "\\|")


def sess(r):
    m = re.search(r"\d+", r["등록_세션"] or ""); return int(m.group(0)) if m else None


def prim(r):
    return r["근거_위치"].split(" / ")[0]


def compute(rows, s4a):
    reg = [r for r in rows if r.get("행_종류", "등록") == "등록"]
    src = [r for r in reg if r["상태"] in ("철회", "미적용") or CORR.search(r["식"])]
    def feat(t):
        t0 = LBL.sub("", t.strip().strip("*")); return nz(t0), s4a.sides(t0), s4a.first_side(t0)
    F = {r["id"]: feat(r["식"].split(" ‖ ")[0]) for r in reg}
    links = []   # (원천, 대상, 규칙, 근거 조각, 원천 낱말)
    for s_ in src:
        ss = sess(s_); text = " ".join(s_["식"].split(" ‖ ")); parts = s_["식"].split(" ‖ ")
        early = [t for t in reg if t is not s_ and sess(t) is not None and ss is not None and sess(t) < ss]
        jn = nz(LBL.sub("", text)); sd = set().union(*[s4a.sides(LBL.sub("", x)) for x in parts]); fl = {s4a.first_side(LBL.sub("", x)) for x in parts if "=" in x}
        for t in early:
            k, ks, fr = F[t["id"]]
            if ((len(k) >= 8 and "=" in k and k in jn) or len(ks & sd) >= 2) and fr in fl: links.append((s_, t, "F", "같은 항목", ""))
        for m in OBJ.finditer(text):
            ob = m.group(1).strip("*.\"'“”").strip()
            if len(ob) < 3 or re.fullmatch(r"§\d+[a-z]?", ob): continue
            for t in early:
                if nz(ob) in nz(t["식"]) and not CORR.search(t["식"]): links.append((s_, t, "N", ob, m.group(2)))
        for m in SECW.finditer(text):
            sec = m.group(1)
            for t in early:
                if re.match(r"^\s*\|?\s*\**" + re.escape(sec) + r"(?![\d-])", t["식"]): links.append((s_, t, "S", sec, m.group(2)))
    seen, L = set(), []
    for x in links:
        if (x[0]["id"], x[1]["id"]) in seen: continue
        seen.add((x[0]["id"], x[1]["id"])); L.append(x)
    return reg, src, L


def keywords(text):
    ks = []
    for m in QUOTE.finditer(text): ks.append(m.group(1).strip())
    for m in SYMTOK.finditer(text):
        t = m.group(0).strip(".·")
        if len(t) >= 3 and not re.fullmatch(r"[=×÷^·≈\d.%]+", t): ks.append(t)
    for m in SECT.finditer(text): ks.append(m.group(0))
    out = []
    for k in ks:
        if k not in out: out.append(k)
    return out


def apply(rows, ctx):
    OUT, s4a = ctx["OUT"], ctx["s4a"]
    snap = [(r["id"], r["상태"]) for r in rows]
    for r in rows: r["현행_상태"] = ""
    reg, src, L = compute(rows, s4a)
    by_t = OrderedDict()
    for s_, t, ru, why, word in L: by_t.setdefault(t["id"], (t, []))[1].append((s_, ru, why, word))
    if set(by_t) != set(APPROVED):
        raise SystemExit(f"단계 7: 규칙 연결 대상 {sorted(by_t)} ≠ 승인 목록 {sorted(APPROVED)} — 쓰지 않고 멈춤")
    mp = []
    for tid, (t, ss) in by_t.items():
        words = {w for _, _, _, w in ss if w}
        same = all(ru == "F" and s_["상태"] == t["상태"] for s_, ru, _, _ in ss)
        if same:
            cur = ""
        else:
            if len(words) != 1: raise SystemExit(f"단계 7: {tid} 원천 낱말 {words} — 하나가 아님")
            cur = words.pop()
        if cur != APPROVED[tid]: raise SystemExit(f"단계 7: {tid} 현행_상태 '{cur}' ≠ 승인 '{APPROVED[tid]}'")
        if cur and cur == t["상태"]: raise SystemExit(f"단계 7: {tid} 현행_상태가 상태와 같음")
        notes = []
        for s_, ru, why, word in ss:
            frag = s_["식"].split(" ‖ ")[0].strip()
            q = (f"원문 앞 90자 '{frag[:90]}'" if len(frag) > 90 else f"원문 '{frag}'")
            if ru == "F":
                notes.append(f"재확정 (같은 항목, 같은 상태 '{s_['상태']}') ← {s_['id']} ({s_['등록_세션']}) {q} ({prim(s_)}); 현행_상태 비움")
            else:
                notes.append(f"현행_상태 '{cur}' ← {s_['id']} ({s_['등록_세션']}) 원문 표기 '{word}' — {q} ({prim(s_)}; 규칙 {ru}: '{why}')")
            mp.append(dict(대상_ID=tid, 대상_세션=t["등록_세션"], 상태=t["상태"], 현행_상태=cur, 원천_ID=s_["id"], 원천_세션=s_["등록_세션"],
                           원천_상태=s_["상태"], 규칙=ru, 근거=why, 원천_낱말=word, 원천_위치=prim(s_), 원천_식=frag))
        t["현행_상태"] = cur
        t["후속_정정"] = (t["후속_정정"] + "; " if t["후속_정정"] else "") + "단계 7: " + " / ".join(notes)
    # MAPSI 확정 쌍 (원천 = 원문 파일 줄)
    ROOT = os.path.dirname(OUT); cf_out = []
    for c in CONFIRMED:
        V = open(os.path.join(ROOT, c["원천_파일"]), encoding="utf-8", errors="replace").read().split("\n")
        ln = [n for n, l in enumerate(V, 1) if c["원천_문자열"] in l]
        if len(ln) != 1: raise SystemExit(f"단계 7 확정 쌍: 원천 문자열 줄 {ln} — 1줄 아님")
        n = ln[0]; word = c["현행_상태"]
        if word not in V[n - 1]: raise SystemExit("단계 7 확정 쌍: 원천 줄에 낱말 없음")
        tg = [r for r in rows if r["근거_위치"].split(" / ")[0].startswith(c["대상_위치"] + " ")]
        rec = dict(c, 원천_행=n, 대상=[r["id"] for r in tg])
        cf_out.append(rec)
        if len(tg) != 1: continue                    # 원장 행이 아니면 (또는 하나가 아니면) 반영하지 않고 보고만
        t = tg[0]
        if t["현행_상태"]: raise SystemExit(f"단계 7 확정 쌍: {t['id']} 현행_상태 이미 있음")
        if word == t["상태"]: raise SystemExit(f"단계 7 확정 쌍: {t['id']} 현행_상태가 상태와 같음")
        sr = next(r for r in rows if r["id"] == c["같은_기록_행"])
        if "철회" not in sr["식"] or "확장 번역 v4 195행" not in sr["식"]: raise SystemExit("단계 7 확정 쌍: 같은 기록 행 문자열 불일치")
        note = (f"현행_상태 '{word}' ← {c['원천_파일']} {n}행 (MAPSI 확정 쌍 {c['일자']}; 원장 행 아님, 세션 표기 없음) "
                f"원문 표기 '{word}' — 원문 '{c['원천_문자열']}' (grep -n {n}행, MAPSI 지시 {c['MAPSI_행']}행); "
                f"같은 철회를 적은 원장 행 {sr['id']} ({sr['등록_세션']}, {prim(sr)})")
        t["현행_상태"] = word
        t["후속_정정"] = (t["후속_정정"] + "; " if t["후속_정정"] else "") + "단계 7: " + note
        mp.append(dict(대상_ID=t["id"], 대상_세션=t["등록_세션"], 상태=t["상태"], 현행_상태=word, 원천_ID=f"{c['원천_파일']} {n}행", 원천_세션="세션 표기 없음",
                       원천_상태="원장 행 아님", 규칙="MAPSI 확정 쌍", 근거=f"{c['일자']}; 같은 기록 행 {sr['id']}", 원천_낱말=word, 원천_위치=f"{c['원천_파일']} {n}행",
                       원천_식=c["원천_문자열"]))
    # 상태 열 불변 확인
    if [(r["id"], r["상태"]) for r in rows] != snap: raise SystemExit("단계 7: 상태 열이 바뀜 — 쓰지 않고 멈춤")
    ctx["step7"] = dict(src=src, L=L, map=mp, snap=snap, reg=reg, cf=cf_out)


def write(rows, ctx):
    OUT = ctx["OUT"]; d = ctx["step7"]; src, L, mp, reg, cf = d["src"], d["L"], d["map"], d["reg"], d["cf"]
    cf_same = {c["같은_기록_행"]: c for c in cf if len(c["대상"]) == 1}
    with open(os.path.join(OUT, "step7_status_snapshot.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "상태"]); w.writerows(d["snap"])
    cols = ["대상_ID", "대상_세션", "상태", "현행_상태", "원천_ID", "원천_세션", "원천_상태", "규칙", "근거", "원천_낱말", "원천_위치", "원천_식"]
    with open(os.path.join(OUT, "map_step7.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(mp)
    linked = {s_["id"] for s_, *_ in L}
    cls, pairs = [], []
    for s_ in src:
        if s_["id"] in linked: continue
        ss = sess(s_)
        text = " ".join(s_["식"].split(" ‖ "))
        ks = keywords(text)
        if s_["id"] in cf_same:
            c = cf_same[s_["id"]]; cls.append((s_, f"MAPSI 확정 쌍 반영 → {c['대상'][0]} (원천 {c['원천_파일']} {c['원천_행']}행; 이 행은 같은 철회 기록)", ks, {})); continue
        if s_["id"] in NO_TARGET:
            cls.append((s_, f"대상 없음 (MAPSI 확정 2026-11-03: {NO_TARGET[s_['id']]})", ks, {})); continue
        if ss is None:
            cls.append((s_, "대상 없음 (등록_세션 숫자 없음 — 이른 세션 비교 불가)", ks, {})); continue
        early = [t for t in reg if t is not s_ and sess(t) is not None and sess(t) < ss]
        hits = OrderedDict()
        for k in ks:
            kk = nz(k)
            if SECT.fullmatch(k): pat = re.compile(re.escape(kk) + r"(?![\d.a-z-])")      # §6 은 §60·§6.1 과 맞추지 않음
            else: pat = re.compile(re.escape(kk))
            hs = [t for t in early if pat.search(nz(t["식"] + " " + t["항목"] + " " + t["별칭"]))]
            if hs: hits[k] = hs
        if not ks: cls.append((s_, "대상 없음 (키워드 없음)", ks, {}))
        elif not hits: cls.append((s_, "대상 없음 (키워드가 이른 세션 원장 항목에 없음)", ks, {}))
        else:
            cls.append((s_, "이름으로만 적힌 정정 → 짝 후보 (미반영)", ks, hits)); pairs.append((s_, ks, hits))
    cc = Counter(c.split(" (")[0] if c.startswith("대상 없음") else (c.split(" →")[0] if c.startswith("MAPSI") else c) for _, c, _, _ in cls)
    n_conf_no = sum(1 for _, c, _, _ in cls if c.startswith("대상 없음 (MAPSI 확정"))
    O = ["# step7_report.md — 단계 7 (후속_정정 채우기) 반영 보고\n",
         "> 생성: `python3 tools/build_step2.py` (tools/build_step7.py). MAPSI 결정 2026-11-03 방식 (나): 상태 열 유지, 현행_상태 열 신설.",
         "> 현행_상태 = 후속 정정으로 바뀐 행만 채움 (빈칸 = 상태와 같음). '폐기' 는 원문 표기 그대로. 문자열 규칙 연결 6행 + MAPSI 확정 쌍만 반영.\n",
         "## 합계\n",
         f"- 정정 원천 행: {len(src)} (규칙은 tools/dryrun_step7.py 와 같음)",
         f"- 반영: 원천 {len(linked)} (규칙) + 원문 파일 줄 {sum(1 for c in cf if len(c['대상']) == 1)} (MAPSI 확정 쌍) → 대상 {len({m['대상_ID'] for m in mp})}행 (연결 {len(mp)}건)",
         f"  - 현행_상태 채움: {sum(1 for r in rows if r.get('현행_상태'))}행 (" + ", ".join(f"{k} {v}" for k, v in Counter(r['현행_상태'] for r in rows if r.get('현행_상태')).items()) + ")",
         f"  - 재확정 (현행_상태 비움, 후속_정정만): {len({m['대상_ID'] for m in mp if not m['현행_상태']})}행",
         f"  - 그중 MAPSI 확정 쌍 (원천 = 원문 파일 줄): " + ("; ".join(f"{c['원천_파일']} {c['원천_행']}행 → " + (c['대상'][0] if len(c['대상']) == 1 else f"원장 행 {len(c['대상'])}개 — 반영 안 함") for c in cf) or "없음"),
         f"- 상태 열 변경: 0 (단계 7 직전 스냅샷 index/step7_status_snapshot.csv 와 같음)",
         f"- 연결 없는 원천 {len(src) - len(linked)}: " + ", ".join(f"{k} {v}" for k, v in cc.items()) + f" (대상 없음 중 MAPSI 확정 {n_conf_no})", "",
         "## 반영 (map_step7.csv)\n", "| 대상 | 대상 세션 | 상태 | 현행_상태 | ← 원천 | 원천 세션·상태 | 규칙 | 원천 낱말 | 원천 위치 |", "|---|---|---|---|---|---|---|---|---|"]
    for m in mp:
        O.append(f"| {m['대상_ID']} | {m['대상_세션']} | {m['상태'] or '빈칸'} | {m['현행_상태'] or '(빈칸 — 재확정)'} | {m['원천_ID']} | {m['원천_세션']} · {m['원천_상태'] or '빈칸'} | {m['규칙']} | {m['원천_낱말'] or '—'} | {esc(m['원천_위치'])} |")
    O += ["", "## 연결 없는 원천 행 분류\n",
          "> '대상 없음' = 원천 문장의 키워드 (따옴표 문자열 · 수식 기호 조각 · §N) 가 이른 세션 등록 행의 식·항목·별칭 어디에도 없음 (원장 항목이 아닌 원천).",
          "> '짝 후보' = 키워드가 이른 세션 등록 행에 있음 — index/step7_pair_candidates.md, 반영하지 않음.\n",
          "| 원천 | 세션 | 상태 | 분류 | 키워드 | 식 (앞 90자) |", "|---|---|---|---|---|---|"]
    for s_, c, ks, hits in cls:
        O.append(f"| {s_['id']} | {s_['등록_세션'] or '—'} | {s_['상태'] or '빈칸'} | {c} | {esc(', '.join(ks)[:80]) or '—'} | {esc(s_['식'].replace(' ‖ ', ' / ')[:90])} |")
    open(os.path.join(OUT, "step7_report.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
    P = ["# step7_pair_candidates.md — 단계 7 짝 후보 (반영 없음, MAPSI·perplexity 확인용)\n",
         "> 생성: `python3 tools/build_step2.py` (tools/build_step7.py). 이름으로만 적힌 정정 (철회·폐기·미적용·정정·오기) 원천 행과,",
         "> 그 키워드 (따옴표 문자열 · 수식 기호 조각 · §N, 공백 제거 뒤 그대로 포함) 를 식·항목·별칭에 가진 이른 세션 등록 행. 문자열 기준, 해석 없음.",
         "> 원장에는 반영하지 않았다. 확인 후 확정된 쌍만 다음 반영에서 현행_상태·후속_정정에 옮긴다. 원천마다 후보 행을 일치 키워드 수 → 가장 긴 일치 키워드 길이 → ID 순으로 앞 12행까지 (전체 건수 병기). '§N' 키워드는 §N0·§N.x 와 맞추지 않음.\n",
         "> MAPSI 확인 1차 (2026-11-03) 로 목록에서 빠진 원천: " + "; ".join([f"{c['같은_기록_행']} (확정 쌍 반영 → {', '.join(c['대상']) or '원장 행 없음'})" for c in cf] + [f"{k} (대상 없음 확정)" for k in NO_TARGET]) + ". 나머지는 perplexity 확인 후.\n",
         f"- 원천 행: {len(pairs)} / 후보 쌍 (원천×대상, 중복 제외): {len({(s_['id'], t['id']) for s_, ks, h in pairs for v in h.values() for t in v})}\n"]
    for s_, ks, hits in pairs:
        P += [f"## {s_['id']} — {s_['등록_세션']} · {s_['상태'] or '빈칸'}\n",
              f"- 원천 식: {esc(s_['식'].replace(' ‖ ', ' / ')[:200])}",
              f"- 원천 위치: {esc(prim(s_))}",
              f"- 키워드: {esc(', '.join(ks))}\n",
              "| 후보 행 | 세션 | 상태 | 일치 키워드 (수) | 식 (앞 80자) |", "|---|---|---|---|---|"]
        cand = OrderedDict()
        for k, hs in hits.items():
            for t in hs: cand.setdefault(t["id"], (t, []))[1].append(k)
        order = sorted(cand.values(), key=lambda x: (-len(x[1]), -max(len(k) for k in x[1]), int(x[0]["id"][1:])))
        for t, kk in order[:12]:
            P.append(f"| {t['id']} | {t['등록_세션']} | {t['상태'] or '빈칸'} | {esc(', '.join(kk))} ({len(kk)}) | {esc(t['식'].replace(' ‖ ', ' / ')[:80])} |")
        if len(order) > 12: P.append(f"| … 외 {len(order) - 12}행 (전체 {len(order)}) | | | | |")
        P.append("")
    open(os.path.join(OUT, "step7_pair_candidates.md"), "w", encoding="utf-8").write("\n".join(P) + "\n")
    print(f"단계7 반영 대상 {len({m['대상_ID'] for m in mp})} 연결 {len(mp)} | 연결 없는 원천 {len(src) - len(linked)} {dict(cc)} | 짝 후보 원천 {len(pairs)}")

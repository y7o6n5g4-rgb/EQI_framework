#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'정의·원칙' 행 (MAPSI 승인 2026-11-03) — tools/build_step2.py 가 단계 5 hook 뒤, ID 부여 전에 apply() 를 부른다.
  행_종류 = '정의·원칙'. 기준 (tools/count_def_candidates.py 와 같음, 문자열 기준·해석 없음):
    상태 표기 없음 (build_step5.tokens) + TASKS §6 필수 키워드 1개 이상 + '=' 또는 '→' 를 가진 줄.
    머리글·표 머리 행·구분 행, 이미 원장 행인 줄은 제외 (단계 5 문서: map_step5 의 행 범위, v3.4: 원장 근거 토큰 전부).
  범위: (1) 단계 5 대상 26개 문서 전부, (2) v3.4 전부. 단계 6 범위는 좁힌 기준 (줄 또는 가장 가까운 상위 머리글에
        MAPSI·원칙·정의·공리·선언·확정) 으로 단계 6 때 적용 — 여기서는 하지 않음.
  상태 빈칸 고정·내려받기 금지 (상태_출처 '정의·원칙 행 — 상태 빈칸 고정'), 비고에 걸린 키워드. 원장 키 DEF:<경로>:<행>.
  키워드 매칭: '정적분·부정적분' → 정적분 / 부정적분, '[x]·{x}' → [x] / {x}, 'D1·D2 배정' → D1·D2 배정 / D1/D2 배정,
             영문 키워드는 앞뒤가 영문자가 아닐 때만, 'tip' 만 대소문자 무시.
"""
import os, re

V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
KW_RAW = ("관측 / 관측창 / tip / 지불 / 정적분·부정적분 / C = GAP / 상전이 / 확장 번역 / 비율 1 / 보른 / 3체 / 5체 / 7체 / 9체 / X7 / TH / We / St / "
          "MAXWELL / 홀로그래픽 투영 / 암흑물질 / 암흑에너지 / 허블 / H_ratio / 로렌츠 / 상대론 / 중력 적색편이 / 우주론 / 프리드만 / 뉴턴 중력 / G 배치 / "
          "§4d / ħ_ani / π_EQI / 슈냐 / 오일러 / 격자 / [x]·{x} / 포논 / membrane / 란다우 / 볼츠만 / 흑체 / 훅 / 페르미 / D1·D2 배정 / 스케일 반전 / "
          "원칙K / 세 층 구조 / 정보 특이점 / family / CMB / LIGO / Solar")
KWS = [k.strip() for k in KW_RAW.split(" / ")]
SPLIT = {"정적분·부정적분": ["정적분", "부정적분"], "[x]·{x}": ["[x]", "{x}"], "D1·D2 배정": ["D1·D2 배정", "D1/D2 배정"]}
SESS = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*|Session ?\d+(?:~\d+)?")

def kw_hits(t):
    out = []
    for k in KWS:
        for p in SPLIT.get(k, [k]):
            if re.fullmatch(r"[A-Za-z_]+", p):
                if re.search(r"(?<![A-Za-z])" + re.escape(p) + r"(?![A-Za-z])", t, re.I if p == "tip" else 0): out.append(k); break
            elif p in t: out.append(k); break
    return out

def candidates(V, used_lines, tokens):
    out = []
    for n, l in enumerate(V, 1):
        t = l.strip()
        if l.startswith("```"): continue
        if not t or l.startswith("#") or re.match(r"^\|[-\s|:]+\|$", t): continue
        if t.startswith("|") and n < len(V) and re.match(r"^\|[-\s|:]+\|$", V[n].strip()): continue
        if n in used_lines: continue
        d, g = tokens(t)
        if d or g: continue
        if "=" not in t and "→" not in t: continue
        h = kw_hits(t)
        if h: out.append((n, h))
    return out

NARROW = re.compile(r"MAPSI|원칙|정의|공리|선언|(?<![불미])확정")   # 단계 6 좁힌 기준 (MAPSI 2026-11-03): 줄 또는 가장 가까운 상위 머리글

def apply(ctx, tag="5"):
    rows, add, cite, FILES, TAG, s5 = ctx["rows"], ctx["add"], ctx["cite"], ctx["FILES"], ctx["TAG"], ctx["s5"]
    # (1) 그 단계 문서: map_step{tag} 행 범위 (새 행·근거 추가) 제외
    used5 = {}
    for m in ctx[f"step{tag}_map"]:
        a, _, b = m["행"].partition("–"); used5.setdefault(m["문서"], set()).update(range(int(a), int(b or a) + 1))
    targets = [(p, used5.get(p, set()) | ((set(range(1, len(FILES[p]) + 1)) - s5.LINE_ONLY[p]) if p in getattr(s5, "LINE_ONLY", {}) else set()), kind)
               for kind, ses, p, src in ctx[f"step{tag}_files"]]
    if tag == "5":
        # (2) v3.4: 원장 근거 토큰 전부 제외 (단계 5 때 함께 반영)
        u34 = set()
        for r in rows:
            for c in r["_cites"]:
                if c[0] == V34: u34.update(range(c[1], c[2] + 1))
        targets.append((V34, u34, "v3.4"))
    out = []
    for p, used, kind in targets:
        V = FILES[p]
        heads, h = {}, None
        lv = {}
        for n, l in enumerate(V, 1):
            if re.match(r"^#{1,4} ", l): h = n; lv[n] = len(l) - len(l.lstrip("#"))
            heads[n] = h
        def chain_of(n):       # 가장 가까운 머리글과 그 위 (수준이 더 높은) 머리글들
            out, h_ = [], heads.get(n)
            while h_:
                out.append(h_)
                h_ = next((k for k in range(h_ - 1, 0, -1) if k in lv and lv[k] < lv[out[-1]]), None)
            return out
        src = next((s for k_, ses, pp, s in ctx[f"step{tag}_files"] if pp == p), "")
        pool6 = []
        if tag == "6" and ctx.get("NORM6"):
            sd_, fs_ = ctx["_norm_sides"], ctx["_norm_first"]
            nz_ = lambda x: re.sub(r"\s+", "", x)
            _LB = re.compile(r"^\s*(?:\d체\[[A-Z0-9]+\]|§[A-Z]'?|\[[A-Z]'?\]|[*□]+)\s*")
            pool6 = [(nz_(_LB.sub("", r["식"].split(" ‖ ")[0].strip().strip("*"))), sd_(r["식"].split(" ‖ ")[0]), fs_(r["식"].split(" ‖ ")[0]), r) for r in rows if r.get("행_종류", "등록") != "정의·원칙" and not r["_key"].startswith("DEF:")]
        for n, hits in candidates(V, used, s5.tokens):
            cand6 = []
            if pool6:
                tt = V[n - 1].strip(); jn = re.sub(r"\s+", "", _LB.sub("", tt.strip("*"))); sd = sd_(tt); fl = fs_(tt) if "=" in tt else None
                hit = [r for k, ks, fr, r in pool6 if (((len(k) >= 8 and "=" in k and k in jn) or len(ks & sd) >= 2) and fr == fl)]
                if hit:     # 정의·원칙 후보가 등록 행과 같은 항목 → 새 행 대신 그 등록 행 근거 위치에 추가
                    for r_ in hit:
                        t_ = cite(p, n, role="본문")
                        if t_ not in r_["_cites"]: r_["_cites"].append(t_); r_["근거_위치"] = ctx["render"](r_["_cites"])
                    ctx.setdefault("def6_merged", []).append((p, n, hit)); continue
                cand6 = [r for k, ks, fr, r in pool6 if any(len(x) >= 3 for x in ks & sd)]
            if tag == "6":                                 # 단계 6: 좁힌 기준 (줄 또는 가장 가까운 상위 머리글; 시험 C 는 상위 머리글 사슬 전체)
                chain = [heads.get(n)] if not ctx.get("CHAIN6") else chain_of(n)
                if not (NARROW.search(V[n - 1]) or any(h_ and NARROW.search(V[h_ - 1]) for h_ in chain)):
                    continue
            t = V[n - 1].strip()
            cites = [cite(p, n, role="정의·원칙")]
            hl = heads.get(n)
            sess, sl = "", None
            for k in [n] + ([hl] if hl else []) + [1]:
                m = SESS.search(V[k - 1])
                if m: sess, sl = m.group(0), k; break
            if hl: cites.append(cite(p, hl, role="머리글"))
            if sl and not any(c[1] <= sl <= c[2] for c in cites): cites.append(cite(p, sl, role="문서 제목" if sl == 1 else "머리글"))
            note = [f"걸린 키워드: {', '.join(hits)}"]
            if tag == "6":
                nw = NARROW.search(V[n - 1])
                if nw: note.append(f"좁힌 기준: 줄의 '{nw.group(0)}'")
                else:
                    ch = chain_of(n) if ctx.get("CHAIN6") else [heads.get(n)]
                    k_ = next(i_ for i_, h_ in enumerate(ch) if h_ and NARROW.search(V[h_ - 1])); hh = ch[k_]
                    up = "가장 가까운 머리글" if k_ == 0 else f"머리글 사슬 {k_}단계 위"
                    note.append(f"좁힌 기준: 머리글 {hh}행의 '{NARROW.search(V[hh - 1]).group(0)}' ({up})")
            if src: note.append(f"원문 docx 변환본: {src}")
            if not sess: note.append("등록_세션: 줄·머리글·문서 제목에 세션 표기 없음")
            r = add(f"DEF:{p}:{n}", cites, 항목=t[:80], 식=t, 별칭=f"정의·원칙 ({p.split('/')[-1]} {n}행)", 상태="", 등급="없음",
                    오차="", 층=ctx["layer"](t), 등록_세션=sess, 근거_파일=p, 엔진_S="—", 키워드=", ".join(hits), 비고="; ".join(note))
            r["행_종류"] = "정의·원칙"
            if cand6: r["_cands6"] = cand6
            r["상태_출처"] = "정의·원칙 행 — 상태 빈칸 고정, 내려받기 금지"
            if p in getattr(s5, "EXTRA_KW", {}):
                kw_ = [x.strip() for x in r["키워드"].split(",") if x.strip()]
                r["키워드"] = ", ".join(kw_ + [x for x in s5.EXTRA_KW[p] if x not in kw_]); r["비고"] += "; " + s5.EXTRA_NOTE[p]
            out.append(dict(문서=p, 행=n, 키워드=", ".join(hits), _row=r, 원문=t[:120]))
    ctx["def_rows" if tag == "5" else f"def_rows{tag}"] = out

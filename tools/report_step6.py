#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 6 건수 보고 (시험 실행) — `STEP6_DRYRUN=1 python3 tools/build_step2.py` 가 ID 부여·상태 정리 뒤, 파일 쓰기 전에 부른다.
  원장·등록부·대응표·충돌 목록은 쓰지 않고 index/step6_count_report.md 만 쓴다.
  보고: 문서별 등록 행 (단계 5 규칙 T1/T2/T3/T5) · 근거 위치 추가 · 제외 · 정의·원칙 행 (좁힌 기준) 건수, 상태 분포, 예상 충돌 수,
        S63·S64 성과정리에서 나오는 행 (예정 ID — 시험 실행 순서 기준, 실제 반영 때 같은 규칙이면 같은 번호).
"""
import os, re
from collections import Counter, OrderedDict

def report(ctx, rows, out_dir):
    m6 = ctx["step6_map"]; d6 = ctx["def_rows6"]; ex6 = ctx["step6_excluded"]; files = ctx["step6_files"]
    new = [m for m in m6 if m["대응"] == "새 행"]; mer = [m for m in m6 if m["대응"] == "근거 위치 추가"]
    per = OrderedDict((p, Counter()) for k, s, p, src in files)
    for m in new: per[m["문서"]]["등록_" + m["종류"]] += 1
    for m in mer: per[m["문서"]]["근거추가"] += 1
    for p, n, why, t in ex6: per[p]["제외"] += 1
    for d in d6: per[d["문서"]]["정의·원칙"] += 1
    ids_new = sorted({m["_row"]["id"] for m in new}, key=lambda x: int(x[1:]))
    ids_def = sorted({d["_row"]["id"] for d in d6}, key=lambda x: int(x[1:]))
    st = Counter(m["_row"]["상태"] or "빈칸" for m in new)
    kinds = Counter(m["종류"] for m in new)
    conf = ctx["step6_conf"]
    big = [(sum(v for k, v in c.items() if k.startswith("등록_")), p) for p, c in per.items()]
    bigd = [(c["정의·원칙"], p) for p, c in per.items()]
    O = ["# step6_count_report.md — 단계 6 건수 보고 (시험 실행, 원장 미반영)\n",
         "> 생성: `STEP6_DRYRUN=1 python3 tools/build_step2.py` (원장·등록부·대응표·충돌 목록은 쓰지 않음). 대상: S21~S108 성과정리·인수인계 최종판 "
         f"{len(files)}개 (S29·S37 성과정리는 derived/ 변환본). 등록 행 = 단계 5 규칙 그대로, 정의·원칙 행 = 좁힌 기준.\n",
         "## 합계\n",
         f"- **등록 행 (새 행): {len(new)}** — 규칙별 {dict(kinds)} / 예정 ID {ids_new[0] if ids_new else '—'}~{ids_new[-1] if ids_new else '—'}",
         f"  - 상태: {dict(st)}",
         f"- 기존 원장 행에 근거 위치 추가: {len(mer)} 대응 (원장 행 {len({m['_row']['id'] for m in mer})}개)",
         f"- 상태가 원장과 다른 대응 → 예상 충돌: {len(conf)}건 (K0017 부터)",
         f"- 제외 줄: {len(ex6)} (" + ", ".join(f"{k} {v}" for k, v in Counter(w.split(' (')[0] for p, n, w, t in ex6).most_common()) + ")",
         f"- **정의·원칙 행 (좁힌 기준: 줄 또는 가장 가까운 상위 머리글에 MAPSI·원칙·정의·공리·선언·확정): {len(d6)}** — 예정 ID {ids_def[0] if ids_def else '—'}~{ids_def[-1] if ids_def else '—'}",
         f"- 등록 행 30 초과 문서 {sum(1 for c, p in big if c > 30)}개, 정의·원칙 행 30 초과 문서 {sum(1 for c, p in bigd if c > 30)}개", "",
         "## 문서별\n", "| 문서 | 등록 T1 | T2 | T3 | T5 | 근거 추가 | 제외 | 정의·원칙 |", "|---|---|---|---|---|---|---|---|"]
    for p, c in per.items():
        O.append(f"| {p.split('/')[-1]} | {c['등록_T1']} | {c['등록_T2']} | {c['등록_T3']} | {c['등록_T5']} | {c['근거추가']} | {c['제외']} | {c['정의·원칙']} |")
    O += ["", "## S63·S64 성과정리에서 나오는 행 (예정 ID)\n",
          "> S63 = 7체 §21 관련, S64 = §3 암흑물질·암흑에너지, §7 H_ratio (MAPSI 지정). 예정 ID 는 시험 실행 순서 기준.\n"]
    for ses in ("S63", "S64"):
        fp = next((p for k, s, p, src in files if s == ses and k == "성과정리"), None)
        if not fp: O.append(f"### {ses}: 성과정리 최종판 없음\n"); continue
        V = ctx["FILES"][fp]
        heads, h = {}, None
        for n, l in enumerate(V, 1):
            if re.match(r"^#{1,4} ", l): h = n
            heads[n] = h
        O.append(f"### {ses} — `{fp}`\n")
        O.append("| 예정 ID | 행_종류 | 행 | 머리글 | 규칙 | 대응 | 상태 | 원문 (앞 90자) |")
        O.append("|---|---|---|---|---|---|---|---|")
        items = [(int(m["행"].split("–")[0]), m["행"], "등록", m["종류"], m["대응"], m["_row"]) for m in m6 if m["문서"] == fp]
        items += [(d["행"], str(d["행"]), "정의·원칙", "DEF", "새 행", d["_row"]) for d in d6 if d["문서"] == fp]
        for a, loc, kind, rule, dd, r in sorted(items, key=lambda x: x[0]):
            hl = heads.get(a); ht = V[hl - 1].strip()[:40] if hl else ""
            O.append(f"| {r['id']} | {kind} | {loc} | {ht.replace('|', chr(92) + '|')} | {rule} | {dd} | {r['상태'] or '빈칸'} | {V[a - 1].strip()[:90].replace('|', chr(92) + '|')} |")
        O.append("")
    open(os.path.join(out_dir, "step6_count_report.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
    if ctx.get("NORM6"):
        dm6 = ctx.get("def6_merged", [])
        open(os.path.join(out_dir, "step6_count_report.md"), "a", encoding="utf-8").write(
            f"\n## (시험 B) 정의·원칙 후보 중 등록 행과 같은 항목 → 새 행 대신 근거 위치 추가: {len(dm6)}줄\n\n"
            + "\n".join(f"- {p.split('/')[-1]} {n}행 → {', '.join(r['id'] for r in hs)}" for p, n, hs in dm6) + "\n")
        print("정의·원칙 → 등록 행 병합", len(dm6))
    print(f"단계6 등록 {len(new)} {dict(kinds)} | 근거추가 {len(mer)} | 충돌 {len(conf)} | 제외 {len(ex6)} | 정의·원칙 {len(d6)}")

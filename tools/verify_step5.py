#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 5 대조 — 세션 문서 S109~S119 + 번역·설계 문서 ↔ 원장 / index/map_step5.csv → index/verify_step5.md
원장·대응표·충돌 CSV 와 원문 (derived 변환본 포함) 줄만 본다. 상태 선택 규칙은 build_step5.py 와 독립으로 다시 씀. 해석·재판정 없음.
  Y1 전수     : 대상 문서의 상태 표기 줄 (머리글·표 구분 행 제외) 이 모두 대응표에 있음 (새 행 / 근거 추가 / 제외 사유)
  Y2 새 행    : 식 조각 = 인용 줄 원문, 근거_파일 = 그 문서, derived 행은 비고에 원문 docx 경로
  Y3 상태     : 독립 재구현한 선택 규칙 (결정 표기 우선 → 등급 → 확정, 부정·등급 언급 제외, '후보' → 빈칸) 결과 = 원장 상태
                T3 행은 비고·근거에 머리글 행 번호, 그 머리글의 범주 표기가 하나, '확정' 머리글은 MAPSI 포함
  Y4 병합     : 대상 행 근거_위치에 그 문서 줄 토큰
  Y5 상태 차이 : 대응표 '다름' 은 conflicts.csv 에 기록
  Y6 지정 행  : S119 성과정리 W-1~W-4 (미적용) · D-1~D-4 (설계 배치) 가 각각 별도 행
  Y7 T5 행    : 범주 둘 머리글 아래 행 — 상태 빈칸, 비고 '머리글 N행 범주 둘 — 상태 미전달', 그 머리글 범주 둘 이상, 처리 칸 인용이 원문 줄에 있음, 내려받기 없음
실행: python3 tools/verify_step5.py
"""
import csv, os, re, sys
from collections import Counter
TAG = sys.argv[1] if len(sys.argv) > 1 else "5"     # 5: S109~S119+번역·설계 / 6: S21~S108 (판 C)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda n: list(csv.DictReader(open(os.path.join(ROOT, "index", n), encoding="utf-8")))
led = {r["id"]: r for r in rd("ledger.csv")}; mp = rd(f"map_step{TAG}.csv"); cf = rd("conflicts.csv")
def L(p):
    return [l.rstrip("\r") for l in open(os.path.join(ROOT, p), "rb").read().decode("utf-8", errors="replace").split("\n")]
NEG = r"(?!\s*(?:표기\s*)?(?:안 함|안함|아님|금지|불가|하지 않|않음|없음))(?!\s*(?:임계|기준|카운트|급\b|급 ))"
D = re.compile(r"(철회|미적용|O 보류|(?<![가-힣A-Za-z])보류|\(O\)|(?<=\| )O(?= \|)|해석 I|\(I\)|(?<=\| )I(?= \|)|결정 대기|설계 배치|(?<![불미])확정)" + NEG)
G = re.compile(r"(True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near[- ]TZ|True Zero|Real Zero|대수필연|(?<![A-Za-z-])TZ(?![A-Za-z(]))" + NEG)
DM = {"철회": "철회", "미적용": "미적용", "O 보류": "O 보류", "보류": "O 보류", "(O)": "O 보류", "O": "O 보류", "해석 I": "해석 I", "(I)": "해석 I",
      "I": "해석 I", "결정 대기": "결정 대기", "설계 배치": "설계 배치", "확정": "확정"}
ORD = ["철회", "미적용", "O 보류", "해석 I", "결정 대기", "설계 배치", "확정"]
GM = lambda t: {"True Zero(대수필연)": "대수필연", "TZ(대수필연)": "대수필연", "Near True Zero": "Near TZ", "Near TZ": "Near TZ", "Near-TZ": "Near TZ",
                "True Zero": "TZ", "Real Zero": "Real Zero", "대수필연": "대수필연", "TZ": "TZ"}[t]
def expect(t):
    if "후보" in t: return ""
    d = [m.group(1) for m in D.finditer(t)]; g = [m.group(1) for m in G.finditer(t)]
    if d:
        st = min({DM[x] for x in d}, key=ORD.index)
        if st == "확정" and g: return GM(g[-1])
        return st
    return GM(g[-1]) if g else None
bad = []
files = sorted({m["문서"] for m in mp})
for p in files:
    V = L(p); rows_ = {int(m["행"].split("–")[0]): m for m in mp if m["문서"] == p}
    cov = set()
    for m in mp:
        if m["문서"] == p:
            a, _, b = m["행"].partition("–"); cov.update(range(int(a), int(b or a) + 1))
    for n, l in enumerate(V, 1):
        t = l.strip()
        if not t or l.startswith("#") or l.startswith("```") or re.match(r"^\|[-\s|:]+\|$", t): continue
        if t.startswith("|") and n < len(V) and re.match(r"^\|[-\s|:]+\|$", V[n].strip()): continue
        if (D.search(t) or G.search(t)) and n not in cov: bad.append(f"Y1 {p} {n}행 대응표에 없음")
res = []
for m in mp:
    why = []
    if m["종류"] == "제외": res.append((m, why)); continue
    r = led.get(m["원장_ID"]); p = m["문서"]; V = L(p)
    a, _, b = m["행"].partition("–"); a, b = int(a), int(b or a)
    if not r: why.append("원장 ID 없음")
    elif m["대응"] == "새 행":
        if r["근거_파일"] != p: why.append("Y2 근거_파일 ≠ 문서")
        if r["식"].split(" ‖ ") != [V[k - 1].strip() for k in range(a, b + 1)]: why.append("Y2 식 ≠ 인용 줄")
        if p.startswith("derived/") and "원문 docx 변환본" not in r["비고"]: why.append("Y2 derived 행 비고에 원문 docx 없음")
        if m["종류"] == "T3":
            hm = re.search(r"\(문서 \S+ (\d+)행\)", r["비고"])
            if not hm: why.append("Y3 T3 비고에 머리글 행 번호 없음")
            else:
                h = V[int(hm.group(1)) - 1]
                if not re.match(r"^#{2,4} ", h): why.append("Y3 T3 머리글 행이 머리글 아님")
                if m["표기"] == "확정" and "MAPSI" not in h: why.append("Y3 '확정' 머리글에 MAPSI 없음")
                if m["표기"] not in h: why.append("Y3 머리글에 범주 표기 없음")
                if DM.get(m["표기"], m["표기"]) != r["상태"]: why.append("Y3 T3 상태 ≠ 머리글 범주")
        elif m["종류"] == "T5":
            hm = re.search(r"머리글 (\d+)행 범주 둘", r["비고"])
            if not hm or "상태 미전달" not in r["비고"]: why.append("Y7 T5 비고에 '머리글 N행 범주 둘 — 상태 미전달' 없음")
            else:
                h = V[int(hm.group(1)) - 1]
                cats = [w for w in ("설계 배치", "(O)", "(I)", "철회", "미적용", "결정 대기") if w in h] + (["확정"] if "확정" in h and "MAPSI" in h else [])
                if len(cats) < 2: why.append(f"Y7 T5 머리글 범주 {cats} (둘 이상 아님)")
            if r["상태"]: why.append("Y7 T5 상태가 빈칸이 아님")
            q = re.match(r"^1순위 표기 '([^']+)'", r["상태_출처"])
            if q and q.group(1) not in V[a - 1]: why.append("Y7 상태_출처 인용 칸 글이 원문 줄에 없음")
            if "(내려받기" in r["상태_출처"]: why.append("Y7 T5 행이 내려받음")
        else:
            e = expect(" ".join(V[k - 1] for k in range(a, b + 1)))
            if e is not None and e != r["상태"]: why.append(f"Y3 상태 기대 '{e or '빈칸'}' ≠ 원장 '{r['상태'] or '빈칸'}'")
    else:
        if f"문서 {p} {a}" not in r["근거_위치"]: why.append("Y4 대상 행에 문서 줄 토큰 없음")
        if m["상태_대조"].startswith("다름") and not any(r["id"] in c["원장_ID"] and f"{p} {a}행" in c["위치_B"] for c in cf):
            why.append("Y5 상태 차이 conflicts.csv 에 없음")
    bad.extend(f"{p.split('/')[-1]} {m['행']} {m['원장_ID']}: {w}" for w in why)
    res.append((m, why))
s119 = "성과정리문서/EQI_Session119_성과정리_v1.md"
WD8 = [("W-1", "미적용"), ("W-2", "미적용"), ("W-3", "미적용"), ("W-4", "미적용"), ("D-1", "설계 배치"), ("D-2", "설계 배치"), ("D-3", "설계 배치"), ("D-4", "설계 배치")]
for lab, want in (WD8 if TAG == "5" else []):
    hit = [x for x in led.values() if x["근거_파일"] == s119 and x["식"].startswith(f"| {lab} |")]
    if len(hit) != 1 or hit[0]["상태"] != want: bad.append(f"Y6 S119 {lab}: 행 {len(hit)}개, 상태 {[h['상태'] for h in hit]} (기대 {want})")
k = Counter(m["종류"] for m in mp); dd = Counter(m["대응"][:12] for m in mp)
O = [f"# verify_step{TAG}.md — 단계 {TAG} (" + ("세션 문서 S109~S119 + 번역·설계" if TAG == "5" else "세션 문서 S21~S108, 판 C") + ") 자동 대조\n",
     f"> 생성: `python3 tools/verify_step5.py`. 대상 {len(files)}개 문서. 원장 전체 원문 대조는 `index/verify_step2.md`, 문자 보존은 `index/check_chars.md`.\n",
     "## 요약\n", f"- 대응표 행: {len(mp)} — 종류 {dict(k)} / 대응 {dict(dd)}",
     ("- Y6 지정 행 (S119 W-1~W-4 미적용, D-1~D-4 설계 배치): " + ("통과" if not any(x.startswith("Y6") for x in bad) else "실패")) if TAG == "5"
     else "- Y8 정의·원칙 좁힌 기준·머리글 사슬 단계 수 / Y9 정의·원칙 → 등록 병합 토큰 (단계 6)",
     f"- **불일치: {len(bad)}**" + ("" if not bad else " — " + "; ".join(bad[:20])), "",
     "## 대응표 행별 결과\n", "| 문서 | 행 | 종류 | 원장 ID | 대응 | 표기 | 상태 대조 | 원문 | 결과 |", "|---|---|---|---|---|---|---|---|---|"]
for m, why in res:
    O.append("| " + " | ".join(x.replace("|", "\\|") for x in [m["문서"].split("/")[-1], m["행"], m["종류"], m["원장_ID"], m["대응"], m["표기"], m["상태_대조"], m["원문"][:60]])
             + f" | {'불일치: ' + '; '.join(why) if why else '일치'} |")
if TAG == "6":   # Y8 정의·원칙 (단계 6 좁힌 기준·머리글 사슬) / Y9 정의·원칙 → 등록 병합 토큰
    NW = re.compile(r"MAPSI|원칙|정의|공리|선언|(?<![불미])확정")
    for d in rd("map_def6.csv"):
        p, n = d["문서"], int(d["행"]); V = L(p)
        if d["대응"] == "새 행":
            r = led[d["원장_ID"]]
            if r["행_종류"] != "정의·원칙" or r["상태"]: bad.append(f"Y8 {d['원장_ID']} 행_종류·상태")
            m = re.search(r"좁힌 기준: (?:줄의 '([^']+)'|머리글 (\d+)행의 '([^']+)' \((가장 가까운 머리글|머리글 사슬 (\d+)단계 위)\))", r["비고"])
            if not m: bad.append(f"Y8 {r['id']} 비고에 좁힌 기준 없음"); continue
            if m.group(1):
                if m.group(1) not in V[n - 1]: bad.append(f"Y8 {r['id']} 줄에 '{m.group(1)}' 없음")
            else:
                h = int(m.group(2)); ht = V[h - 1]
                if not re.match(r"^#{1,4} ", ht) or m.group(3) not in ht or h >= n: bad.append(f"Y8 {r['id']} 머리글 {h}행 확인 실패")
                if NW.search(V[n - 1]): bad.append(f"Y8 {r['id']} 줄에 기준어가 있는데 머리글로 기록")
                # 단계 수: n 위의 머리글을 수준 낮아지는 순으로 따라가 h 가 몇 번째인지
                chain, kk, lv = [], n, None
                while kk > 1:
                    kk -= 1
                    mm = re.match(r"^(#{1,4}) ", V[kk - 1])
                    if mm and (lv is None or len(mm.group(1)) < lv): chain.append(kk); lv = len(mm.group(1))
                if h not in chain: bad.append(f"Y8 {r['id']} 머리글 {h}행이 사슬에 없음")
                else:
                    want = 0 if m.group(4) == "가장 가까운 머리글" else int(m.group(5))
                    if chain.index(h) != want: bad.append(f"Y8 {r['id']} 단계 수 {want} ≠ 실제 {chain.index(h)}")
                    if any(NW.search(V[c - 1]) for c in chain[:chain.index(h)]): bad.append(f"Y8 {r['id']} 더 가까운 머리글에 기준어")
        else:
            for i in d["원장_ID"].split(", "):
                if f"문서 {p} {n}행" not in led[i]["근거_위치"]: bad.append(f"Y9 {i} 에 {p.split('/')[-1]} {n}행 토큰 없음")
open(os.path.join(ROOT, "index", f"verify_step{TAG}.md"), "w", encoding="utf-8").write("\n".join(O) + "\n")
print(f"문서 {len(files)} / 대응표 {len(mp)} {dict(k)} / 불일치 {len(bad)}", bad[:8])

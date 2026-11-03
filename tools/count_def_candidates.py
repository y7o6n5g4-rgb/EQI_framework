#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'정의·원칙' 행 후보 건수 보고 (MAPSI 2026-11-03 제안 기준) — 원장은 바꾸지 않는다. 결과: index/def_candidates_report.md
  기준: 상태 표기 없음 (build_step5.tokens 기준) + TASKS §6 필수 키워드 1개 이상 + '=' 또는 '→' 포함.
        머리글·표 머리·구분 행, 이미 원장 행인 줄은 제외. 코드 울타리 안 줄도 셈.
  키워드: '정적분·부정적분' → 정적분 / 부정적분, '[x]·{x}' → [x] / {x}, 'D1·D2 배정' → D1·D2 배정 / D1/D2 배정.
          영문 키워드는 앞뒤가 영문자가 아닐 때만 (We·St·TH 가 단어 일부일 때 제외), 'tip' 만 대소문자 무시 (TIP 포함).
  단계 6 범위 수는 단계 6 등록 행이 아직 없으므로 상한값.
실행: python3 tools/count_def_candidates.py
"""
import csv, os, re, sys, importlib.util
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp=importlib.util.spec_from_file_location("s5",ROOT+"/tools/build_step5.py"); s5=importlib.util.module_from_spec(sp); sp.loader.exec_module(s5)
KW_RAW="관측 / 관측창 / tip / 지불 / 정적분·부정적분 / C = GAP / 상전이 / 확장 번역 / 비율 1 / 보른 / 3체 / 5체 / 7체 / 9체 / X7 / TH / We / St / MAXWELL / 홀로그래픽 투영 / 암흑물질 / 암흑에너지 / 허블 / H_ratio / 로렌츠 / 상대론 / 중력 적색편이 / 우주론 / 프리드만 / 뉴턴 중력 / G 배치 / §4d / ħ_ani / π_EQI / 슈냐 / 오일러 / 격자 / [x]·{x} / 포논 / membrane / 란다우 / 볼츠만 / 흑체 / 훅 / 페르미 / D1·D2 배정 / 스케일 반전 / 원칙K / 세 층 구조 / 정보 특이점 / family / CMB / LIGO / Solar"
kws=[k.strip() for k in KW_RAW.split(" / ")]
def kw_patterns(k):
    if k=="정적분·부정적분": return ["정적분","부정적분"]
    if k=="[x]·{x}": return ["[x]","{x}"]
    if k=="D1·D2 배정": return ["D1·D2 배정","D1/D2 배정"]
    return [k]
def kw_hits(t):
    out=[]
    for k in kws:
        for p in kw_patterns(k):
            if re.fullmatch(r"[A-Za-z_]+",p):
                pat=r"(?<![A-Za-z])"+re.escape(p)+r"(?![A-Za-z])"
                if re.search(pat,t, re.I if p=="tip" else 0): out.append(k); break
            elif p in t: out.append(k); break
    return out
used=set()
for r in csv.DictReader(open(ROOT+"/index/map_step5.csv",encoding="utf-8")):
    if r["종류"]!="제외":
        a,_,b=r["행"].partition("–"); used.update((r["문서"],n) for n in range(int(a),int(b or a)+1))
tot=0; per=[]
for kind,ses,p,src in s5.files_of(ROOT):
    V=[l.rstrip("\r") for l in open(ROOT+"/"+p,"rb").read().decode("utf-8",errors="replace").split("\n")]
    c=0; fence=False; ex=[]
    for n,l in enumerate(V,1):
        t=l.strip()
        if l.startswith("```"): fence=not fence; continue
        if not t or l.startswith("#") or re.match(r"^\|[-\s|:]+\|$",t): continue
        if t.startswith("|") and n<len(V) and re.match(r"^\|[-\s|:]+\|$",V[n].strip()): continue
        if (p,n) in used: continue
        d,g=s5.tokens(t)
        if d or g: continue
        if "=" not in t and "→" not in t: continue
        h=kw_hits(t)
        if not h: continue
        c+=1; ex.append((n,h,t[:120]))
    tot+=c; per.append((c,p,ex))

def count_file(p, usedset):
    V=[l.rstrip("\r") for l in open(ROOT+"/"+p,"rb").read().decode("utf-8",errors="replace").split("\n")]
    c=0; fence=False
    for n,l in enumerate(V,1):
        t=l.strip()
        if l.startswith("```"): fence=not fence; continue
        if not t or l.startswith("#") or re.match(r"^\|[-\s|:]+\|$",t): continue
        if t.startswith("|") and n<len(V) and re.match(r"^\|[-\s|:]+\|$",V[n].strip()): continue
        if (p,n) in usedset: continue
        d,g=s5.tokens(t)
        if d or g: continue
        if "=" not in t and "→" not in t: continue
        if kw_hits(t): c+=1
    return c, len(V)
# v3.4: 원장 v3.4 행 제외
v34="성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"; u34=set()
for r in csv.DictReader(open(ROOT+"/index/ledger.csv",encoding="utf-8")):
    for m in re.finditer(r"v3\.4 (\d+)(?:–(\d+))?행", r["근거_위치"]):
        u34.update((v34,n) for n in range(int(m.group(1)),int(m.group(2) or m.group(1))+1))
c34,nl=count_file(v34,u34)
V=open(ROOT+"/"+v34,encoding="utf-8").read().split("\n")
b1=next(i for i,l in enumerate(V,1) if l.startswith("## §12.")); b2=next(i for i,l in enumerate(V,1) if l.startswith("## §S79/S80"))
# section split counts
def cnt_range(a,b):
    cc=0; fence=False
    for n in range(a,b+1):
        l=V[n-1]; t=l.strip()
        if l.startswith("```"): fence=not fence; continue
        if not t or l.startswith("#") or re.match(r"^\|[-\s|:]+\|$",t): continue
        if (v34,n) in u34: continue
        d,g=s5.tokens(t)
        if d or g or ("=" not in t and "→" not in t) or not kw_hits(t): continue
        cc+=1
    return cc
r1,r2,r3=cnt_range(1,b1-1),cnt_range(b1,b2-1),cnt_range(b2,len(V))
dm=list(csv.DictReader(open(ROOT+"/index/doc_map.csv",encoding="utf-8")))
tot6=0; big=[]
for r in dm:
    if r["분류"]=="sessions" and r["최종판"]=="Y" and r["세션"] and 21<=int(r["세션"][1:])<=108:
        p=r["상대경로"] if r["형식"]=="md" else "derived/"+os.path.splitext(r["파일명"])[0]+".md"
        if not os.path.exists(ROOT+"/"+p): continue
        c,_=count_file(p,set()); tot6+=c
        if c>30: big.append((c,r["파일명"]))


from collections import Counter
kwc=Counter(k for c,p,ex in per for n,h,t in ex for k in h)
O=["# def_candidates_report.md — '정의·원칙' 행 후보 건수 (원장 미반영, MAPSI 판단용)\n",
   "> 생성: `python3 tools/count_def_candidates.py`. 기준: 상태 표기 없음 + TASKS §6 필수 키워드 1개 이상 + '=' 또는 '→'. 이미 원장 행인 줄 제외.\n",
   "## 건수\n",
   f"- **단계 5 대상 26개 문서: {tot}줄**, 문서당 30줄 초과 {sum(1 for c,_,_ in per if c>30)}개 문서",
   f"- v3.4 (단계 4 범위 포함 전체): {c34}줄 — 1~1042행 (머리말·§1~§11) {r1} / 4a §12~§30 {r2} / 4b 캡슐 {r3}",
   f"- 단계 6 범위 (S21~S108 최종판, 아직 등록 행 없음 → 상한): {tot6}줄, 30줄 초과 {len(big)}개 문서",
   "", "| 문서 | 줄 수 |", "|---|---|"] + [f"| `{p}` | {c}{' **(30 초과)**' if c>30 else ''} |" for c,p,ex in per] + [
   "", "## 키워드별 (단계 5 대상, 한 줄에 여러 키워드면 각각 셈)\n", ", ".join(f"{k} {v}" for k,v in kwc.most_common()), "",
   "## 후보 줄 전체 (단계 5 대상)\n", "| 문서 | 행 | 걸린 키워드 | 원문 (앞 120자) |", "|---|---|---|---|"]
for c,p,ex in per:
    for n,h,t in ex:
        O.append(f"| {p.split('/')[-1]} | {n} | {', '.join(h)} | {t.replace('|', chr(92)+'|')} |")
open(os.path.join(ROOT,"index","def_candidates_report.md"),"w",encoding="utf-8").write("\n".join(O)+"\n")
print("단계5",tot,"| v3.4",c34,"| 단계6 상한",tot6)

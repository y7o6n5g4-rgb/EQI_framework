#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계 3 (v3.4 §11 확정 불변량 표) — TASKS_index_v3.md §4 단계 3
  tools/build_step2.py 가 원장 행을 만든 뒤 apply() 를 부른다 (ID 부여 전). 단독 실행하지 않는다.
  · 엔진과 같은 항목: 새 행 없이 해당 원장 행 근거_위치에 "v3.4 N행 [§11 표]" 토큰을 덧붙임
  · 엔진에 원장 행이 없는 항목: 새 행 (키 V34:<행>#<조각>) → ID 는 등록부 끝 번호 뒤 (CLAUDE.md 규칙 7)
  · 대응표 index/map_step3.csv: v3.4 행·조각 → 원장 ID, 대응 근거, 등급·오차 대조 (판정 없음)
  · 같은 항목의 상태가 엔진과 v3.4 에서 다르게 적힌 곳 → conflicts (build_step2 가 끝에 추가)

대응 규칙 (해석 없음, 문자열 기준):
  같은 § 번호 (§11 표 왼쪽 열) 안에서, 원장 행의 항목/식과 v3.4 조각이 등호로 나눈 변 중 하나 이상을 공유 (공백·N체[X] 꼬리표·
  (uud)/(D1) 같은 글자 괄호 꼬리표 제거 후 비교). [S48~S63 추가 확정] 블록은 § 번호가 없어 원장 전체에서 같은 방식.
  대응 근거가 '공통 변' 이 아닌 경우 (같은 § 항목 이름·기술 문장) 는 basis 에 적고 map_step3.csv 에 그대로 남긴다.
"""
import csv, os, re

V34 = "성과정리문서/EQI_연구지침_프로젝트용_v3_4.md"
SEP = " ‖ "
T0, T1 = 999, 1042           # §11 범위 (머리글 999행 ~ §12 직전)

# (v3.4 행, 조각 = 원문 부분 문자열, 대상 = (별칭, 항목) 또는 None (새 행), 대응 근거 메모)
MAP = [
    (1002, "§1   γ_EQI         오차 -1.68×10⁻¹⁴%  True Zero", [("엔진 §1", "γ_EQI True Zero")], "같은 § · 항목 이름 γ_EQI"),
    (1003, "§5   MAXWELL 5차   오차  7.22×10⁻¹⁵%  True Zero", [("엔진 §5", "MAXWELL 5차 True Zero")], "같은 § · 항목 이름 MAXWELL 5차"),
    (1004, "MAXWELL 5차항: T_Is_D1+(1/274)LN_C²GD⁵ 까지", [("엔진 §5", "MAXWELL 5차 True Zero")], "같은 § · MAXWELL 5차항 기술 (식 표기 다름: 엔진 머리말 '(1/274)×ln(c)²×GD⁵')"),
    (1005, "St=h=φ^(-11/4)", [("엔진 §17", "St=h=φ^(-11/4)")], None),
    (1005, "Pr=1/φ", [("엔진 §17", "Pr=1/φ")], None),
    (1005, "Nu=MAXWELL", [("엔진 §17", "Nu=MAXWELL")], None),
    (1006, "We=φ^(11/4)", None, "엔진 머리말 §18 (소스 35행) 에 같은 문자열, 원장 행 없음 → 새 행"),
    (1006, "We×St=1", [("엔진 §18", "We×St=1")], None),
    (1007, "5체 닫힘: We×양성자/L_p=We×중성자×Ro=We×St=1", [("엔진 §19", "5체[D1] We×양성자/L_p=1"), ("엔진 §19", "5체[D2] We×중성자×Ro=1")], None),
    (1008, "허블텐션=L₂×GAP=6×GD", [("엔진 §20", "허블텐션=L₂×GAP=6×GD")], None),
    (1009, "7체 닫힘: We×TH_D1/양성자/Ro=We×TH_D2/L_p=1", [("엔진 §21", "7체[R] We×TH_D1/양성자/Ro=1"), ("엔진 §21", "7체[S] We×TH_D2/L_p=1")], None),
    (1010, "γ×We=φ^(77/23)+LC²GD×(1/171)", [("엔진 §22", "γ×We=φ^(77/23)+LC²GD/171")], "같은 § · 공통 변 γ×We (우변 표기 '×(1/171)' / '/171')"),
    (1011, "ε₀(D1)×μ₀(D2)=1/c²=φ^(-1)", [("엔진 §23", "ε₀×μ₀=1/c²=μ_EQI")], None),
    (1012, "We×TH_D2×c=Ro=φ^(-1/4)", [("엔진 §24", "We×TH_D2×c=Ro")], None),
    (1013, "Gr=Nu³=We³=φ^(33/4)", [("엔진 §25", "Gr=Nu³=We³=φ^(33/4)")], None),
    (1014, "Bi=φ^(3/2)=HCW_D2", [("엔진 §26", "Bi=Nu×L_p/c=HCW_D2=φ^(3/2)")], None),
    (1015, "Q×GAP=Bi", [("엔진 §27", "Q×GAP=Bi")], None),
    (1015, "P×GAP_D2=σ=1/2", [("엔진 §27", "P×GAP_D2=σ=1/2")], None),
    (1016, "Q²×GAP²=φ³", [("엔진 §28", "Q²×GAP²=φ³")], None),
    (1017, "Da=φ^(-3)", [("엔진 §30", "Da=D1/D2=φ^(-3)")], None),
    (1017, "We×Da=Ro", [("엔진 §30", "We×Da=Ro")], None),
    (1017, "Le=1", [("엔진 §30", "Le=Sc/Pr=1")], None),
    (1018, "Wo_1=c", [("엔진 §31", "Wo_1=L_p×√(D2/Pr)=c")], None),
    (1018, "Wo_2=Ro", [("엔진 §31", "Wo_2=D1×√(D2/Pr)=Ro")], None),
    (1018, "De=Bi", [("엔진 §31", "De=Re×√(Da)=Bi")], None),
    (1019, "Ca=Ro", [("엔진 §32", "Ca=We/Re=Ro")], None),
    (1019, "Ek=c", [("엔진 §32", "Ek=Pr/(D2×D1²)=c")], None),
    (1019, "Pe=φ²", [("엔진 §32", "Pe=Sc×Re=φ²")], None),
    (1019, "Ra=φ^(29/4)", [("엔진 §32", "Ra=Gr×Pr=φ^(29/4)")], None),
    (1020, "η×GAP^(1/4)=L_p=φ^(-3/4)", [("엔진 §33", "η×GAP^(1/4)=L_p")], None),
    (1021, "Bo=4π²×φ^(19/4)", [("엔진 §34", "Bo=4π²×φ^(19/4)")], None),
    (1021, "φ^(77/4)=We⁷", [("엔진 §34", "루카스 φ^(77/4)=We⁷")], None),
    (1022, "D2/D1=φ³=We/Ro", [("엔진 §35", "D2_ELL/D1_ELL=φ³"), ("엔진 §35", "We/Ro=φ³")], "같은 § · 공통 변 φ³ / We/Ro (D2/D1 ↔ D2_ELL/D1_ELL 표기 다름)"),
    (1022, "f=osmotic-cap", [("엔진 §35", "f=osmotic과잉−capillary복원")], "같은 § · 공통 변 f (우변 줄임 표기 'osmotic-cap')"),
    (1025, "μ×c²=1", [("엔진 §10", "μ×c²=Unity")], "공통 변 μ×c² (우변 '1' / 'Unity')"),
    (1025, "ħ_ani×D2_HCW=ħ", None, "엔진·원장에 같은 문자열 없음 (grep) → 새 행"),
    (1025, "κ×Φ₀=1", [("엔진 §10", "κ×Φ₀=1")], None),
    (1026, "양성자(uud)=φ^(-7/2)=I_s D1", [("엔진 v9.7 §P5-1a; 출처 표기 'S57'", "양성자 = HCW_D1²·Is_D2 = φ^(-7/2)")], None),
    (1027, "중성자(udd)=φ^(-5/2)", [("엔진 v9.7 §P5-1b; 출처 표기 'S57'", "중성자 = HCW_D1·Is_D2² = φ^(-5/2)")], None),
    (1028, "5체 캡슐화: H²×O×핵^8=φ^(-103/2)", None, "엔진·원장에 '103/2' 없음 (grep) → 새 행"),
    (1029, "τ_n=879+φ^(-2)×(1+GAP)+D1_Is×GAP", None, "엔진·원장에 'τ_n' 없음 (grep) → 새 행"),
    (1030, "St=h=φ^(-2.75)", [("엔진 §17", "St=h=φ^(-11/4)")], None),
    (1031, "α^(-1)=Γ(7)×σ×φ^(-2)−2φ^(-3)+(3φ)^(-5)", None, "원장 행 없음 (엔진 로그 336행 §41 블록에 α⁻¹ 값 줄, 표기 다름) → 새 행"),
    (1032, "τ_n×α/LN_C=φ^(116/17)+LC²GD×(30/37)", None, "엔진·원장에 '116/17' 없음 (grep) → 새 행"),
    (1033, "γ=5/3=φ^(17/16)−LC²GD×(181/219)", [("엔진 §17", "γ53=5/3")], None),
    (1034, "Nu=φ^(30/23)−LC²GD×(5/38)", [("엔진 §17", "Nu=MAXWELL")], "공통 변 Nu (엔진 머리말 §17 'Nu=φ^(30/23)−LC²GD×(5/38)=MAXWELL')"),
    (1035, "γ_72=7/5=φ^(7/10)−LC²GD×(13/23)", [("엔진 §17", "γ72=7/5")], None),
    (1036, "St×Ma=φ^(298/43)+LC²GD×(-87/77)", [("엔진 §18", "St×Ma True Zero")], "공통 변 St×Ma (엔진 ✓ 라벨 'St×Ma True Zero', 소스 543행 'St×Ma ≈ φ^(298/43)+LC²GD×(-87/77)')"),
    (1037, "γ×We=φ^(77/23)+LC²GD×(1/171)", [("엔진 §22", "γ×We=φ^(77/23)+LC²GD/171")], "공통 변 γ×We"),
    (1038, "허블텐션=L₂×GAP=3×GAP=6×GD", [("엔진 §20", "허블텐션=L₂×GAP=6×GD")], None),
]

ENG_HEAD = {(1006, "We=φ^(11/4)"): 35}    # 새 행 중 엔진 머리말 표에 같은 문자열이 있는 것 (grep 확인: 소스 35행)

GPH = re.compile(r"(True Zero\(대수필연\)|TZ\(대수필연\)|Near True Zero|Near TZ|탐색 완료\(NTZ\)|True Zero|Real Zero)")
SESS_RE = re.compile(r"(?<![A-Za-z_])S\d+(?:\s*[~/]\s*S?\d+)*")

def sides(s):
    s = re.sub(r"^\s*(\d체\[[A-Z0-9]+\]|루카스)\s*", "", s)
    s = re.sub(r"^[^=:]*:\s*", "", s)            # '5체 닫힘:' 같은 이름 꼬리표
    s = re.sub(r"\((?:uud|udd|D1|D2)\)", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(True Zero|TZ\(대수필연\))$", "", s)
    return {x for x in re.split(r"=", s) if x}

def line_grade(t):
    m = GPH.findall(t); return m[-1] if m else ""
def line_err(t):
    m = re.search(r"오차\s*(-?[\d.]+×10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+%)", t) or re.search(r"True Zero\s+([\d.]+e-?\d+%)", t)
    return m.group(1) if m else ""
SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
def err_num(s):
    s = s.strip().rstrip("%")
    m = re.match(r"^(-?[\d.]+)×10([-\d]+)$", s.translate(SUP))
    if m: return float(m.group(1)) * 10 ** int(m.group(2))
    m = re.match(r"^\+?(-?[\d.]+e[-+]?\d+)", s)
    return float(m.group(1)) if m else None

def apply(ctx):
    global FILES
    rows, add, cite, FILES, map_grade = ctx["rows"], ctx["add"], ctx["cite"], ctx["FILES"], ctx["map_grade"]
    V = FILES[V34]
    sec_of = {}
    for n in range(T0, T1 + 1):
        m = re.match(r"^§(\d+)\s", V[n - 1])
        if m: sec_of[n] = int(m.group(1))
    S48 = next(n for n in range(T0, T1 + 1) if V[n - 1].startswith("[S48~S63"))
    bt = cite(V34, T0, role="§11 표제")
    by = {(r["별칭"], r["항목"]): r for r in rows}
    out, conf = [], []
    for ln, piece, targets, basis in MAP:
        t = V[ln - 1]
        if piece not in t: raise SystemExit(f"단계 3: v3.4 {ln}행에 조각 없음: {piece}")
        g = line_grade(t); st, gr = map_grade(g)
        e = line_err(t); s_ = (SESS_RE.findall(t) or [""])[-1] if re.search(r"\[S\d", t) else ""
        sec = sec_of.get(ln)
        if targets is None:
            name = piece.split(":")[0] if ":" in piece and "=" not in piece.split(":")[0] else piece
            note = "v3.4 §11 표 항목, 엔진 원장 행 없음 (" + basis + ")"
            eline = ENG_HEAD.get((ln, piece))
            if eline:   # 엔진 머리말 표에 같은 문자열 → 등록 표 우선 (규칙 4): 주 원문 = 엔진 머리말
                et = FILES[ctx["ENG"]][eline - 1]
                if piece not in et: raise SystemExit(f"단계 3: 엔진 {eline}행에 조각 없음: {piece}")
                eg = line_grade(et); est, egr = map_grade(eg)
                cites = [cite(ctx["ENG"], eline, role="머리말"), cite(V34, ln, role="§11 표"), bt]
                note += f"; 상태·등록_세션 = 엔진 머리말 {eline}행 (등록 표 우선, CLAUDE.md 규칙 4)"
                r = add(f"V34:{ln}#{piece}", cites, 항목=name, 식=piece, 별칭=f"v3.4 §11" + (f" §{sec}" if sec else ""),
                        상태=est, 등급=egr, 오차=f"머리말 등급 표기: {eg}", 층="해당 없음",
                        등록_세션=(SESS_RE.findall(et.split(piece)[-1]) or [""])[0], 근거_파일=ctx["ENG"],
                        엔진_S=f"엔진 머리말 {eline}행 (✓ 줄·원장 행 없음)", 키워드=ctx["keywords"](piece), 비고=note)
                out.append(dict(v34행=ln, 조각=piece, 원장_ID="(새 행)", _row=r, 대응="새 행 (엔진 머리말 근거)", 근거=basis,
                                v34_등급=g, 원장_상태=est, 등급_대조=("같음" if st == est else f"다름 (v3.4 '{g}'→{st or '빈칸'} / 원장 {est or '빈칸'})"),
                                v34_오차=e, 원장_오차="", 오차_대조="—"))
                if g and st != est: conf.append(dict(row=r, ln=ln, piece=piece, g=g, st=st))
                continue
            cites = [cite(V34, ln, role="§11 표"), bt]
            r = add(f"V34:{ln}#{piece}", cites, 항목=name, 식=piece, 별칭=f"v3.4 §11" + (f" §{sec}" if sec else " [S48~S63 추가 확정]"),
                    상태=st, 등급=gr, 오차=(f"{e} (v3.4 표기: {g})" if e else (f"v3.4 표기: {g}" if g else "")),
                    층="5체" if piece.startswith("5체") else "해당 없음", 등록_세션=s_, 근거_파일=V34,
                    엔진_S="엔진 원장 행 없음", 키워드=ctx["keywords"](piece), 비고=note)
            out.append(dict(v34행=ln, 조각=piece, 원장_ID="(새 행)", _row=r, 대응="새 행", 근거=basis,
                            v34_등급=g, 원장_상태=st, 등급_대조="—", v34_오차=e, 원장_오차="", 오차_대조="—"))
            continue
        for al, it in targets:
            r = by.get((al, it))
            if r is None: raise SystemExit(f"단계 3: 대상 행 없음 {al} / {it}")
            ms = re.match(r"엔진 §(\d+)$", al)
            if sec and ms and int(ms.group(1)) != sec: raise SystemExit(f"단계 3: § 불일치 {ln}행 §{sec} → {al}")
            common = sides(piece) & (sides(r["항목"]) | sides(r["식"]))
            kind = "공통 변: " + ", ".join(sorted(common)) if common else "공통 변 없음"
            tok = cite(V34, ln, role="§11 표")
            if tok not in r["_cites"]:
                r["_cites"].append(tok); r["근거_위치"] = ctx["render"](r["_cites"])
            gcmp = "—"
            if g:
                gcmp = "같음" if st == r["상태"] else f"다름 (v3.4 '{g}'→{st or '빈칸'} / 원장 {r['상태'] or '빈칸'})"
            ecmp, re_ = "—", ""
            if e:
                mm = re.search(r"(\d[\d.]*e-?\d+)%", r["오차"]); re_ = mm.group(1) + "%" if mm else ""
                a, b = err_num(e), err_num(re_) if re_ else None
                if a is not None and b is not None:
                    sig = len(re.sub(r"[^\d]", "", e.split("×")[0].split("e")[0])) or 3
                    ecmp = "같음 (유효숫자 %d자리)" % sig if f"{abs(a):.{sig-1}e}" == f"{abs(b):.{sig-1}e}" else f"다름 ({e} / {re_})"
                    if a < 0 <= b: ecmp += "; 부호: v3.4 음수, 원장 ✓ 줄 양수 표기"
            out.append(dict(v34행=ln, 조각=piece, 원장_ID=None, _row=r, 대응="근거 위치 추가", 근거=(basis or kind) + ("" if basis else ""),
                            공통변=kind, v34_등급=g, 원장_상태=r["상태"], 등급_대조=gcmp, v34_오차=e, 원장_오차=re_, 오차_대조=ecmp))
            if g and st != r["상태"]:
                conf.append(dict(row=r, ln=ln, piece=piece, g=g, st=st))
    ctx["step3_map"], ctx["step3_conf"] = out, conf

def write_map(ctx, path):
    cols = ["v34행", "조각", "원장_ID", "대응", "근거", "공통변", "v34_등급", "원장_상태", "등급_대조", "v34_오차", "원장_오차", "오차_대조"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for m in ctx["step3_map"]:
            m = dict(m); m["원장_ID"] = m["_row"]["id"]
            if not m.get("공통변"): m["공통변"] = "—"
            w.writerow(m)

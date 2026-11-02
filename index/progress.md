# progress.md — EQI 색인 v2 진행표

작업 지시: `TASKS_index_v3.md` (S119) / 규칙: `CLAUDE.md` (저장소 루트, 업로드본 사본)

## 저장소 경로 대응 (TASKS §1 ↔ 실제)

TASKS §1 은 `docs/v3_4`, `docs/sessions` … 하위 폴더 구조를 가정하나, 실제 저장소는 평면 구조다.
원문 파일은 이동하지 않고 (CLAUDE.md 규칙 1) 아래처럼 대응시킨다. 분류는 `doc_map.csv` 의 `분류` 열.

| TASKS §1 | 실제 위치 | doc_map `분류` |
|---|---|---|
| docs/v3_4 | `성과정리문서/EQI_연구지침_프로젝트용_v3_4.md` | v3_4 |
| docs/sessions | `성과정리문서/EQI_Session*`, `session_21_*`, `session_22_*` | sessions |
| docs/translations | `성과정리문서/EQI_S116_*`, `EQI_S117_*` | translations |
| docs/foundation | `성과정리문서/` 이론적_기반, Section118, 단위계_철학, PREAMBLE, TRIGGER, 119_압축본 | foundation |
| docs/index_v1_1 | `성과정리문서/EQI_클로드용_*`, `EQI_클로드_전달메모_v1_1.md` | index_v1_1 |
| (그 외) | Framework v3~v17, 핵심 구조 문서, pdf, txt 메모, monster spec, 색인 작업지시서 v2 등 | 기타(역사문서·메모) |
| code/ | `code/` (엔진 `eqi_unified_engine_v9_7.py` 완성판 2,662행 + 기준 로그 `eqi_unified_engine_v9_7_terminal_mpmath140.txt`) | — |

- `code/CLAUDE.md` 는 엔진 v9.7 확장 작업용 별도 지시서 (색인 작업 지시 아님).
- "협업 지침" 독립 파일은 없음 — v3.4 §1·§13 이 해당 내용.

## 단계 현황

| 단계 | 상태 | 새 원장 행 | 충돌 | 산출물 | 비고 |
|---|---|---|---|---|---|
| 1 골격 | **완료** (2026-11-02, 보충 반영) | 0 | 0 | doc_map.csv, coverage.md 초안, tools/build_step1.py | 원장 행 없음 → 표본 대조(§7) 해당 없음 |
| 2 엔진 § | **완료 (완성판 재실행)** (2026-11-02) | 265 (L0001~L0265; 완성판 신규 58) | 6 (K0001~K0006, K0004 해소) | ledger.csv, conflicts.csv, sample_check_2.md (재출력), coverage.md B 연결, tools/build_step2.py | 표본 10행 대조 대기 (MAPSI / perplexity) |
| 3 v3.4 §11 | 대기 | | | | |
| 4 v3.4 본문 | 대기 | | | | |
| 5 S109~S119 | 대기 | | | | |
| 6 S1~S108 | 대기 | | | | |
| 7 정정 반영 | 대기 | | | | |
| 8 주제 지도 | 대기 | | | | |
| 9 충돌·누락 | 대기 | | | | |
| 10 조립 | 대기 | | | | |

## 단계 1 결과

- `doc_map.csv`: `성과정리문서/` 전체 227 파일 (sessions 170 / 기타 38 / foundation 9 / index_v1_1 5 / translations 4 / v3_4 1).
  형식: md 186, docx 25, txt 11, csv 2, pdf 2, json 1.
- 동명 판본 (둘 다 기록): S98 성과정리 v4·v5, S113 성과정리 v3·v6, PREAMBLE v5·v6, TRIGGER v6·v7, Section118 v1·v2,
  Framework v3~v17 (15개), Core_Structure 무판본·v2.
- 사본 표기 ` (N)` 파일: `EQI_Session30_성과정리_v1 (2).md`, `EQI_Session32_성과정리_v1 (1).md`, `EQI_Framework_v16 (1).docx`.
- 세션 문서 범위 S21~S120 (S23 없음, S1~S20 없음). S120 은 인수인계만 존재 (S120 성과정리 파일 없음).
- 텍스트 추출이 필요한 세션 문서 (docx): S29 성과정리 v9_final, S37 성과정리 v1, **S109 성과정리 v4, S110 성과정리 v4, S111 인수인계 v3** — 단계 5 에서 pandoc 으로 읽기 (원문 미변경).
- `coverage.md`: v3.4 머리글 105개 (4122행) / 엔진 머리말 표 §1~§42 (✓ 91줄) / v9.7 추가 블록 7개 (§P0, §P1, §P3, §P5, §P7, §P-light, §P-chain).
- 확인 사항
  - **S92~S97 캡슐 블록 존재** (v3.4 2957~3425행, `■ S88~S91 캡슐화` 블록 내부). 단계 4 에서 정상 처리 가능.
  - v3.4 에 `## §20` 머리글 없음 — §11 표 1008행에 §20 행. 단계 3 에서 연결.
  - 엔진 §2, §9, §29 는 출력 로그에 `§N` 표제·✓ 줄이 grep 되지 않음 → 단계 2 에서 소스 본문 위치로 근거 확보 (미등록 판정 아님).
  - 엔진 머리말 docstring 은 `v9.6` 표기 (§1~§42). v9.7 추가분은 머리말 표가 아니라 1909행 이후 블록과 출력 로그 528행 이후에만 있음.

## 단계 1 보충 (MAPSI 지시 반영)

- doc_map 에 `최종판_근거`, `현행` 열 추가. 최종판 판정은 S118 색인 작업지시서 v2 (`EQI_색인_작업지시서_v2.md` 70·87·89·93행) 우선:
  S111 성과정리 v5, S113 성과정리 v6 (파일 1행 헤더 "v5" 는 오기), S117 성과정리 v2. 그 밖은 v 숫자 최대. 동명 판본 모두 유지.
- 프리앰블 v6·트리거 v7: `현행` 표시.
- S1~S20·S23 은 v3.4 본문을 원천으로 함 (단계 4). 단계 6 은 저장소에 있는 S21 이후만 보충 (coverage.md C 절에 기록).

## 단계 2 결과 (v9.7 완성판 기준 재실행)

기준 파일 (MAPSI 교체, main c8b9662 병합):
- 엔진 `code/eqi_unified_engine_v9_7.py` = 완성판 2,662행 (161,083 B, `ccr-6a187b58-00312y` 브랜치 blob 과 동일)
- 기준 로그 `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt` = 완성판 mpmath 1.4.0 출력 863행 (81,311 B, ccr 브랜치 `_output.txt` blob 과 동일)
- `code/eqi_unified_engine_v9_7_terminal.txt` (82,254 B) = 같은 실행의 Windows 콘솔 사본. mpmath140 과 `diff -w` 차이는 첫 프롬프트 줄·끝 프롬프트 줄뿐 (CRLF·줄끝 공백 제외) → 충돌 아님

원장 265행:
- 엔진 §1~§42 = L0001~L0095 (95행), v9.7 §P 블록 = L0096~L0265 (170행)
  - 완성판 신규 블록 (L0208~L0265, 58행): §P6 양자 오비탈 L0208~L0213 / §P3 빛 보강 (§L-7·§L-8·§L-9) L0214~L0228 /
    §P3·§P5·§P7 체 계층 보강 (§B) L0229~L0233 / §P-MAXWELL (§PM-0~§PM-7) L0234~L0249 / §P-DS (§DS-1′·§DS-1″·§U1~§U3) L0250~L0265.
    신규 58행 상태: 대수필연 34 / TZ 2 (§PM-7b, §PM-6) / 빈칸 22 (값만·정수·'귀속 검토 중').
  - §DS-1′, §DS-1′r, §DS-1″, §DS-1″r (L0251~L0254): 판정 열 '귀속 검토 중(S113, 승격 없음)' → 상태 빈칸, 등급 없음 (원문 판정 열은 오차 열에 그대로).
  - §U 행 엔진_§ 열에 로그 860행 '※ U-1~U-3 '귀속 확정' 판정 없음 (Architect/MAPSI 판단).' 병기.
  - 블록 머리 값·주석 줄도 행으로 둠: §P3 빛 보강 [S46 HCW 계보] (L0214) / [S50 계보] "두 계보 병기 (어느 쪽도 폐기 아님)" (L0219) / §P-DS 머리 R·잔류·D′·D″·좌표 구분 (L0250).
- 행 단위·상태 옮김 규칙은 이전과 같음: ✓ 줄 1줄 = 1행, `[§…]` 1줄 = 1행, k 만 바뀌고 판정 같은 묶음은 1행 (§P3-1, §P3-2, §P3-3 a~f, §C-8, 신규 §O-1, §L-7b).
  ★ True Zero → TZ, ★ TZ(대수필연) → 대수필연, Near (True) Zero → Near TZ. 정수·'-'·통과·귀속 검토 중·표기 없음 → 상태 빈칸.
- 집계 (265행): 대수필연 154 / TZ 57 / Near TZ 10 / Real Zero 1 / 미적용 1 / 빈칸 42.
- 등급 기준 주의 (이전과 같음): 엔진 원문 등급 그대로. v9.7 §P 판정 열은 code/CLAUDE.md 기준 (TZ < 1e-6 %), 색인 CLAUDE.md (TZ < 1e-4 %) 와 다름 — 재판정 없음.
  §P 블록 4열(오차) 단위 % (S115 성과정리 62행).
- 후속_정정 (엔진 안에 적힌 것만): §L-4 (S114 정정), §P7-3 (S92 폐기), §C-주석 (S114 미적용). 나머지는 단계 7.

### 이전 판 (2,429행 / output.txt 746행) 기준 행과의 차이

- 소스: 1~2412행 동일 (`diff` = 2412a2413,2642 / 2423a2654,2656 — 삽입만). 원장이 인용한 소스 행 (머리말 13~92, 122, 145~1087, 1530, 2325–2326, 2408–2409) 번호 변화 없음.
- 로그: 이전 output.txt 1~744행 = mpmath140 1~744행 (동일). 745행 '등급 집계' 자리에 신규 블록 삽입, 등급 집계는 862행으로 이동.
- L0001~L0207: id·항목·식·별칭·상태·등급·오차·층·등록_세션·후속_정정·키워드 전부 동일 (스크립트 대조 207/207).
  - 근거_파일 115행, 엔진_§ 85행: 로그 경로 `code/eqi_unified_engine_v9_7_output.txt` → `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt` (행 번호 동일).
  - 근거_위치 22행 (L0139~L0160): §P3 블록 ID 가 두 번 나와 (588행 '3체 층', 759행 '빛 보강') 블록 표제 병기 — "§P3 '3체 층 (빛 속성 + 뤼드베리)' 블록 (표제 588행)".
- 신규 L0208~L0265 (58행) 추가.
- conflicts (단계 2 재실행 확인 후 MAPSI 정정 (1) 반영 — ID 불변 규칙, CLAUDE.md 규칙 7):
  재실행 때 K0004 를 지우고 K0005 → K0004 로 당겼던 것을 원래 번호로 복원.
  K0004 = 옛 output.txt / terminal.txt 67행 마지막 자리 차이 (0.0007934570312 / …313) — 행 유지, `상태` 열 "근거 파일 교체로 해소, 이전 판 git 74b469a"
  (위치는 이전 판 경로 그대로 + "(이전 판, git 74b469a)" 표기. 현 기준 로그 67행 = 0.0007934570312 를 비고에 병기).
  K0005 = Stefan–Boltzmann (원래 번호). K0006 = §L-7c 브래그 '불일치(문구)' (code/TASKS.md 350행 '7차부터 < 1' / 로그 775행 '계산상 5차(k=2) 0.53125 부터 < 1').
  conflicts.csv 에 `상태` 열 추가 (해소 사유 기록용, 나머지 행은 빈칸).
- ID 불변 검사: `tools/build_step2.py` `id_guard` — 실행 시 기존 ledger.csv (id → 항목·별칭), conflicts.csv (id → 항목) 와 대조, 어긋나면 파일을 쓰지 않고 중단.
  복원은 git 74b469a 의 conflicts.csv 를 기준으로 재생성해 검사 통과 확인. 원장 L0001~L0207 은 74b469a 판과 id·항목·별칭 일치 (재실행 때 207/207 대조).

충돌 6건 (판정 없음): K0001 §12 등급 3중 표기 / K0002 a₅_eff §14·§15 귀속 / K0003 §L-6 Near TZ·원문 TZ(S99) /
K0004 옛 output·terminal 67행 마지막 자리 (해소, 이전 판 git 74b469a) / K0005 Stefan–Boltzmann v3.4 2104행 vs 엔진 정정 / K0006 §L-7c 문구.

### code/ 파일 변경 확인 (MAPSI 정정 (2), git 대조 결과)

- 이 작업 브랜치의 커밋 (b12ef1d, 74b469a, 07acf04, bd898f6) 이 바꾼 파일은 `CLAUDE.md`, `TASKS_index_v3.md`, `.gitignore`, `index/*`, `tools/*` 뿐. **code/ 아래 파일은 하나도 수정하지 않음** (`git show --name-only` 확인).
- `git diff origin/main HEAD -- code/` = 차이 없음. 병합 커밋 de18b5e 는 main c8b9662 의 code/ 를 그대로 가져옴.
- blob 대조 (내용 동일 = 같은 해시):
  - `code/eqi_unified_engine_v9_7.py` HEAD = main = ccr-6a187b58-00312y = 557bd9c
  - `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt` HEAD = main = ccr `code/eqi_unified_engine_v9_7_output.txt` = a8530e0
  - `code/eqi_unified_engine_v9_7_terminal.txt` HEAD = main = 56fe44b
- code/ 를 바꾼 커밋은 모두 MAPSI 업로드·삭제: e7ce1ce·c78ae37 (단계 1 이전, terminal 교체) / 703fc85·1edf640·0c9a688 (이전 판 엔진·output·terminal 삭제) / c8b9662 (완성판 3파일 업로드). 교체 외 내용 수정 없음.
- 재실행 보고의 "블록 머리 주석 처리 수정" 은 **추출 스크립트 `tools/build_step2.py` 쪽** 수정이다 (블록 첫 항목 앞 값 줄과 `[S46 …]`·`[S50 …]` 계보 주석 줄을 원장 행으로 읽도록 파서 변경). 엔진·로그 원문은 손대지 않음.


## 다음 단계 시작 지점

- 단계 2 표본 대조: `index/sample_check_2.md` (재출력) 10행 (L0023, L0034, L0063, L0082, L0104, L0194, L0208, L0213, L0240, L0265) 원문 대조 후 결과 회신.
- 단계 3: `성과정리문서/EQI_연구지침_프로젝트용_v3_4.md` §11 확정 불변량 표 (999~1042행) → 엔진과 같은 항목은 기존 행 근거_위치에 추가, 새 항목만 L0266 부터.

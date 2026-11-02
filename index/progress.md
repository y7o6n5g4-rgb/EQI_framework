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
| code/ | `code/` (엔진 `eqi_unified_engine_v9_7.py` + `_output.txt`) | — |

- `code/CLAUDE.md` 는 엔진 v9.7 확장 작업용 별도 지시서 (색인 작업 지시 아님).
- "협업 지침" 독립 파일은 없음 — v3.4 §1·§13 이 해당 내용.

## 단계 현황

| 단계 | 상태 | 새 원장 행 | 충돌 | 산출물 | 비고 |
|---|---|---|---|---|---|
| 1 골격 | **완료** (2026-11-02) | 0 | 0 | doc_map.csv, coverage.md 초안, tools/build_step1.py | 원장 행 없음 → 표본 대조(§7) 해당 없음 |
| 2 엔진 § | 대기 | | | | |
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

## 다음 단계 시작 지점

- 단계 2: `code/eqi_unified_engine_v9_7.py` 머리말 표 13~92행 → `code/eqi_unified_engine_v9_7_output.txt` ✓ 줄 379~468행 → v9.7 블록 528~746행 (§P0~§P-chain).
  원장 ID L0001 부터. 단계 끝에 `index/sample_check_2.md` (무작위 10행) 출력.

# EQI_framework

EQI 색인 저장소. 작업 규칙은 `CLAUDE.md`, 작업 지시는 `TASKS_index_v3.md`, 진행 기록은 `index/progress.md`.

## 원장 (`index/ledger.csv`) 의 상태 열

| 열 | 뜻 |
|---|---|
| `상태` | 그 행 근거 원문의 등록 당시 표기 (CLAUDE.md 규칙 4 순서). 후속 정정이 있어도 바꾸지 않는다. |
| `현행_상태` | 후속 세션 정정으로 바뀐 행만 채운다. **빈칸 = 상태와 같음.** 값은 원천 원문 낱말 그대로 ('폐기' 는 철회·미적용으로 바꾸지 않음). |
| `후속_정정` | 현행_상태를 채운 행: 정정 원천 행 ID·세션·원문 표기·위치. 같은 상태 재확정은 현행_상태 없이 여기에만 기록. |

상태 목록: 확정 (MAPSI) / 대수필연 / TZ / Near TZ / O 보류 / 해석 I / 결정 대기 / 철회 / 미적용 / 설계 배치 / 폐기(원문 표기).

재생성: `python3 tools/build_step2.py` (원장·대응표·보고서), 검사: `python3 tools/verify_step2.py`, `python3 tools/verify_step7.py`.

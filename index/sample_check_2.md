# sample_check_2.md — 단계 2 표본 대조 (무작위 10행, seed 20261103, L0266 이후 추가 행 2개 이상 조건)

> 원장 행 + 근거_위치의 원문 행 인용. 식 열의 ' ‖ ' 는 원문 줄 구분자. MAPSI / perplexity 대조용.
> 틀린 행이 있으면 단계 2 전체 재점검 (TASKS §7). 자동 대조 결과는 `index/verify_step2.md`.

## L0028 — St×Ma True Zero

| 열 | 값 |
|---|---|
| 항목 | St×Ma True Zero |
| 식 | We=φ^(11/4), St×Ma 확정, We×St=1  S61  True Zero |
| 별칭 | 엔진 §18 |
| 상태 | TZ |
| 등급 | TZ |
| 오차 | 1.935393e-07% (★ True Zero) |
| 층 | 해당 없음 |
| 등록_세션 | S61 |
| 근거_파일 | code/eqi_unified_engine_v9_7.py |
| 근거_위치 | 로그 410행 [✓] / 로그 111행 [블록 표제] / 소스 35행 [머리말] / 소스 540행 [계산 블록] |
| 엔진_§ | §18 / 머리말 35행 / 로그 표제 111행 / ✓ 410행 (code/eqi_unified_engine_v9_7_terminal_mpmath140.txt) |
| 키워드 | We, St |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:410` [✓]
  ```
  410: ✓ §18 St×Ma True Zero                      오차=1.935393e-07%  ★ True Zero
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:111` [블록 표제]
  ```
  111: §18 We=φ^(11/4), St×Ma, We×St=1 [S61]
  ```
- `code/eqi_unified_engine_v9_7.py:35` [머리말]
  ```
  35: ║  §18: We=φ^(11/4), St×Ma 확정, We×St=1  S61  True Zero                   ║
  ```
- `code/eqi_unified_engine_v9_7.py:540` [계산 블록]
  ```
  540: # §18: We=φ^(11/4), St×Ma 확정, We×St=1  (Session 61 신규)
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0046 — Q²×GAP²/We=1/Ro

| 열 | 값 |
|---|---|
| 항목 | Q²×GAP²/We=1/Ro |
| 식 | Q²×GAP²/We=1/Ro |
| 별칭 | 엔진 §28 |
| 상태 | 대수필연 |
| 등급 | 대수필연 |
| 오차 | 0.000000e+00% (★ TZ(대수필연)) |
| 층 | 해당 없음 |
| 등록_세션 | S65 |
| 근거_파일 | code/eqi_unified_engine_v9_7.py |
| 근거_위치 | 로그 428행 [✓] / 로그 188행 [블록 표제] / 소스 55–56행 (2행) [머리말] / 소스 707행 [계산 블록] |
| 엔진_§ | §28 / 머리말 55행 / 로그 표제 188행 / ✓ 428행 (code/eqi_unified_engine_v9_7_terminal_mpmath140.txt) |
| 키워드 | We, Ro, GAP |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:428` [✓]
  ```
  428: ✓ §28 Q²×GAP²/We=1/Ro                      오차=0.000000e+00%  ★ TZ(대수필연)
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:188` [블록 표제]
  ```
  188: §28 임피던스 Q² 구조 [S65]
  ```
- `code/eqi_unified_engine_v9_7.py:55-56` [머리말]
  ```
  55: ║  §28: 임피던스 Q² 구조             S65  True Zero(대수필연)               ║
  56: ║       Q²×GAP²=φ³, Q²×GAP²/We=1/Ro                                        ║
  ```
- `code/eqi_unified_engine_v9_7.py:707` [계산 블록]
  ```
  707: # §28: 임피던스 Q² 구조 (Session 65 신규)
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0069 — D2_ELL/D1_ELL=φ³

| 열 | 값 |
|---|---|
| 항목 | D2_ELL/D1_ELL=φ³ |
| 식 | D2_ELL/D1_ELL=φ³ |
| 별칭 | 엔진 §35 |
| 상태 | 대수필연 |
| 등급 | 대수필연 |
| 오차 | 0.000000e+00% (★ TZ(대수필연)) |
| 층 | 해당 없음 |
| 등록_세션 | S68 |
| 근거_파일 | code/eqi_unified_engine_v9_7.py |
| 근거_위치 | 로그 451행 [✓] / 로그 264행 [블록 표제] / 소스 71–76행 (6행) [머리말] / 소스 877행 [계산 블록] |
| 엔진_§ | §35 / 머리말 71행 / 로그 표제 264행 / ✓ 451행 (code/eqi_unified_engine_v9_7_terminal_mpmath140.txt) |
| 키워드 | D1·D2 |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:451` [✓]
  ```
  451: ✓ §35 D2_ELL/D1_ELL=φ³                     오차=0.000000e+00%  ★ TZ(대수필연)
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:264` [블록 표제]
  ```
  264: §35 EQI 로렌츠 비등방성 공식 [S68]
  ```
- `code/eqi_unified_engine_v9_7.py:71-76` [머리말]
  ```
  71: ║  §35: EQI 로렌츠 비등방성 공식     S68  True Zero(대수필연)               ║
  72: ║       D1_ELL=(1+GAP)/√[(1+GAP)²+φ⁶]  [capillary/종파/qE]                ║
  73: ║       D2_ELL=φ³×D1_ELL             [osmotic/횡파/q(v×B)]                 ║
  74: ║       f_lattice=D2²−1=[φ⁶×GAP×(2+GAP)−(1+GAP)²]/[(1+GAP)²+φ⁶]          ║
  75: ║       φ⁶−1=4φ³  피보나치 항등식  GAP=0→f<0, GAP≠0→f>0                   ║
  76: ║       보른이탈분해: f=osmotic과잉−capillary복원  GAP≠0 직접 증거           ║
  ```
- `code/eqi_unified_engine_v9_7.py:877` [계산 블록]
  ```
  877: # §35: EQI 로렌츠 비등방성 공식  (Session 68 신규)
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0074 — f×We×Re 대수 닫힘

| 열 | 값 |
|---|---|
| 항목 | f×We×Re 대수 닫힘 |
| 식 | f×We×Re 대수 닫힘            S70  True Zero 1.11e-7% ‖ LC²GD×(100/171) + LC²GD²×(2/285)  3×57→5×57 진화 |
| 별칭 | 엔진 §36 |
| 상태 | TZ |
| 등급 | TZ |
| 오차 | 1.107661e-07% (★ True Zero) |
| 층 | 해당 없음 |
| 등록_세션 | S70 |
| 근거_파일 | code/eqi_unified_engine_v9_7.py |
| 근거_위치 | 로그 456행 [✓] / 로그 298행 [블록 표제] / 소스 78–79행 (2행) [머리말] / 소스 946행 [계산 블록] |
| 엔진_§ | §36 / 머리말 78행 / 로그 표제 298행 / ✓ 456행 (code/eqi_unified_engine_v9_7_terminal_mpmath140.txt) |
| 키워드 | We, Re, LC²GD |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:456` [✓]
  ```
  456: ✓ §36 f×We×Re 대수 닫힘                        오차=1.107661e-07%  ★ True Zero
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:298` [블록 표제]
  ```
  298: §36 f×We×Re 대수 닫힘 (97 유령 분모 해소) [S70]
  ```
- `code/eqi_unified_engine_v9_7.py:78-79` [머리말]
  ```
  78: ║  §36: f×We×Re 대수 닫힘            S70  True Zero 1.11e-7%                ║
  79: ║       LC²GD×(100/171) + LC²GD²×(2/285)  3×57→5×57 진화                   ║
  ```
- `code/eqi_unified_engine_v9_7.py:946` [계산 블록]
  ```
  946: # §36: f × We × Re 대수 닫힘 (Session 70 이식)
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0084 — Is(k=+3/2)×We 마스터

| 열 | 값 |
|---|---|
| 항목 | Is(k=+3/2)×We 마스터 |
| 식 | Is(k=+3/2)×We 마스터 |
| 별칭 | 엔진 §42 |
| 상태 | TZ |
| 등급 | TZ |
| 오차 | 4.644310e-09% (★ True Zero) |
| 층 | 해당 없음 |
| 등록_세션 | S71 |
| 근거_파일 | code/eqi_unified_engine_v9_7.py |
| 근거_위치 | 로그 466행 [✓] / 로그 345행 [블록 표제] / 소스 92–95행 (4행) [머리말] / 소스 1050행 [계산 블록] |
| 엔진_§ | §42 / 머리말 92행 / 로그 표제 345행 / ✓ 466행 (code/eqi_unified_engine_v9_7_terminal_mpmath140.txt) |
| 키워드 | We |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:466` [✓]
  ```
  466: ✓ §42 Is(k=+3/2)×We 마스터                    오차=4.644310e-09%  ★ True Zero
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:345` [블록 표제]
  ```
  345: §42 Is(k/2)×We 마스터 공식 [S71] ★ 전 k ALL True Zero
  ```
- `code/eqi_unified_engine_v9_7.py:92-95` [머리말]
  ```
  92: ║  §42: Is(k/2)×We 마스터 공식       S71  전 k(-3~+7) ALL True Zero ★       ║
  93: ║       =φ^((2k+11)/4)+LC²GD×(1/437)                                       ║
  94: ║        −LC²GD²×(24/171)−LC²GD³×(32/285)                                  ║
  95: ║       I_s 계단 전체 단일 마스터 닫힘  4!=24 팩토리얼 언어 공유             ║
  ```
- `code/eqi_unified_engine_v9_7.py:1050` [계산 블록]
  ```
  1050: # §42: Is(k/2) × We 마스터 공식 (Session 71 신규)
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0100 — R_pi(34) 오일러 3항

| 열 | 값 |
|---|---|
| 항목 | R_pi(34) 오일러 3항 |
| 식 | R_pi(34) 오일러 3항 |
| 별칭 | 엔진 v9.7 §P0-3.3; 출처 표기 'S44/S45 (Task 1a)' |
| 상태 | TZ |
| 등급 | TZ |
| 오차 | 9.62914e-10 (판정 열: TZ) |
| 층 | 해당 없음 |
| 등록_세션 | S44/S45 |
| 근거_파일 | code/eqi_unified_engine_v9_7_terminal_mpmath140.txt |
| 근거_위치 | 로그 536행 [항목] / 로그 530행 [블록 표제] |
| 엔진_§ | v9.7 §P0 §P0-3.3 / 엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인') / 소스 class EQIPhononLattice |
| 키워드 | 격자, 포논, 오일러, §P0 |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:536` [항목]
  ```
  536: [§P0-3.3] R_pi(34) 오일러 3항 | 0.05882034993362227675 | 0.05882034993305588742 | 9.62914e-10 | TZ | S44/S45 (Task 1a)
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:530` [블록 표제]
  ```
  530: §P0 π 격자 포논 (Leibniz L(M) ↔ 오일러 수) [S43/S44/S45/S88]
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0131 — S_D1 − H_D2 = φ^(-7/2)

| 열 | 값 |
|---|---|
| 항목 | S_D1 − H_D2 = φ^(-7/2) |
| 식 | S_D1 − H_D2 = φ^(-7/2) |
| 별칭 | 엔진 v9.7 §P1-7a; 출처 표기 'S88 §83' |
| 상태 | 대수필연 |
| 등급 | 대수필연 |
| 오차 | 5.33354e-402 (판정 열: 대수필연) |
| 층 | 해당 없음 |
| 등록_세션 | S88 |
| 근거_파일 | code/eqi_unified_engine_v9_7_terminal_mpmath140.txt |
| 근거_위치 | 로그 578행 [항목] / 로그 565행 [블록 표제] |
| 엔진_§ | v9.7 §P1 §P1-7a / 엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인') / 소스 class EQIPhononLattice |
| 키워드 | 격자, D1·D2, §P1 |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:578` [항목]
  ```
  578: [§P1-7a] S_D1 − H_D2 = φ^(-7/2) | 0.1855851657558680703 | 0.1855851657558680703 | 5.33354e-402 | 대수필연 | S88 §83
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:565` [블록 표제]
  ```
  565: §P1 격자 기저 (S24 / S27 / S88) — 판정 열 = 400자리 절대 차
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0271 — 1−cos θ_3 = n_ref,3  (θ=13.4295440951°)  (추가 행)

| 열 | 값 |
|---|---|
| 항목 | 1−cos θ_3 = n_ref,3  (θ=13.4295440951°) |
| 식 | 1−cos θ_3 = n_ref,3  (θ=13.4295440951°) |
| 별칭 | 엔진 v9.7 §P3-2.k3; 출처 표기 'S45/S50' |
| 상태 | 대수필연 |
| 등급 | 대수필연 |
| 오차 | 0.0 (판정 열: 대수필연) |
| 층 | 3체 |
| 등록_세션 | S45/S50 |
| 근거_파일 | code/eqi_unified_engine_v9_7_terminal_mpmath140.txt |
| 근거_위치 | 로그 597행 [항목] / 로그 588행 [블록 표제] |
| 엔진_§ | v9.7 §P3 §P3-2.k3 / 엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인') / 소스 class EQIPhononLattice |
| 키워드 | 뤼드베리, §P3 |
| 비고 | 원 묶음 L0140에서 분리 |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:597` [항목]
  ```
  597: [§P3-2.k3] 1−cos θ_3 = n_ref,3  (θ=13.4295440951°) | 0.02734375 | 0.02734375 | 0.0 | 대수필연 | S45/S50
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:588` [블록 표제]
  ```
  588: §P3 3체 층 (빛 속성 + 뤼드베리) [S45/S50~S53/S73/S92/S99]
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0316 — n1²+n2² = √disc  (추가 행)

| 열 | 값 |
|---|---|
| 항목 | n1²+n2² = √disc |
| 식 | n1²+n2² = √disc |
| 별칭 | 엔진 v9.7 §P3-3.k9a; 출처 표기 'S51~S53' |
| 상태 | 대수필연 |
| 등급 | 대수필연 |
| 오차 | 0.0 (판정 열: 대수필연) |
| 층 | 3체 |
| 등록_세션 | S51~S53 |
| 근거_파일 | code/eqi_unified_engine_v9_7_terminal_mpmath140.txt |
| 근거_위치 | 로그 648행 [항목] / 로그 588행 [블록 표제] |
| 엔진_§ | v9.7 §P3 §P3-3.k9a / 엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인') / 소스 class EQIPhononLattice |
| 키워드 | 뤼드베리, §P3 |
| 비고 | 원 묶음 L0141에서 분리 |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:648` [항목]
  ```
  648: [§P3-3.k9a] n1²+n2² = √disc | 2048.0881328692864627 | 2048.0881328692864627 | 0.0 | 대수필연 | S51~S53
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:588` [블록 표제]
  ```
  588: §P3 3체 층 (빛 속성 + 뤼드베리) [S45/S50~S53/S73/S92/S99]
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:

## L0318 — 1/n1²−1/n2² = n_ref,9  (추가 행)

| 열 | 값 |
|---|---|
| 항목 | 1/n1²−1/n2² = n_ref,9 |
| 식 | 1/n1²−1/n2² = n_ref,9 |
| 별칭 | 엔진 v9.7 §P3-3.k9c; 출처 표기 'S51~S53' |
| 상태 | 대수필연 |
| 등급 | 대수필연 |
| 오차 | 0.0 (판정 열: 대수필연) |
| 층 | 3체 |
| 등록_세션 | S51~S53 |
| 근거_파일 | code/eqi_unified_engine_v9_7_terminal_mpmath140.txt |
| 근거_위치 | 로그 650행 [항목] / 로그 588행 [블록 표제] |
| 엔진_§ | v9.7 §P3 §P3-3.k9c / 엔진 v9.7 블록 (S114 TASKS, S115 'Task 2~6 승인') / 소스 class EQIPhononLattice |
| 키워드 | 뤼드베리, §P3 |
| 비고 | 원 묶음 L0143에서 분리 |

원문 인용:

- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:650` [항목]
  ```
  650: [§P3-3.k9c] 1/n1²−1/n2² = n_ref,9 | 0.00001811981201171875 | 0.00001811981201171875 | 0.0 | 대수필연 | S51~S53
  ```
- `code/eqi_unified_engine_v9_7_terminal_mpmath140.txt:588` [블록 표제]
  ```
  588: §P3 3체 층 (빛 속성 + 뤼드베리) [S45/S50~S53/S73/S92/S99]
  ```

대조 결과: [ ] 일치  [ ] 불일치 → 메모:


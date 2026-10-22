# 국소정리의 안전한 R8 접속 어댑터 — 보고서

- 작성: 2026-10-22.
- 하지 않은 것: 영역 확대, 전역 탐색, 대칭 복제.
- 수정하지 않은 것: legacy R8 (`taskR8_Pt_lattice.py`), 보존 파일 (`rupert_local_region_verify.py`, 증명서 등).
- GAP·μ·Λ 는 사용하지 않았다.
- **산출물 (미커밋)**:

| 파일 | 역할 |
|---|---|
| `code/rupert_local_region_scope.py` | 보존 검증기를 runpy 로 무수정 재실행하고, 검증한 정리를 기계 판독 레코드로 기록 |
| `code/rupert_local_region_scope.json` | 범위 레코드. 판정 필드 (`verified_theorem`) 와 무시 주석 (`annotations_ignored_for_decision`) 분리 |
| `code/rupert_local_region_scope_output.txt` | 실행 출력 |
| `code/rupert_r8_adapter.py` | 보조 모듈 `LocalTheorem.classify(obj, units, box)` |
| `code/rupert_r8_adapter_tests.json` | 시험 입력·기대 상태 (**실행 전 고정**) |
| `code/rupert_r8_adapter_test.py` | 회귀시험 실행기 |
| `code/rupert_r8_adapter_test_output.txt` | 회귀시험 출력 |
| `code/rupert_r8_adapter_test_results.json` | 회귀시험 결과 |
| 본 보고서 | |

- **실행 순서**:
  1. `python3 code/rupert_local_region_scope.py`
  2. `python3 code/rupert_r8_adapter_test.py`
  - 합계 약 1 s.
- 환경: Python 3.13.16, 표준 라이브러리만 사용.
  - 산술은 Fraction 구간이다 (2^−200 격자 바깥 반올림).
  - π 는 Machin 공식으로 감싼다. cos/sin 은 테일러 부분합과 라그랑주 잔차로 감싼다.

## A. 검증 결과의 범위 결합 (`rupert_local_region_scope.json`)
- 보존 검증기를 증명서 경로 인자로 실행한다 (출력 파일 미기록). `SystemExit(1)` 이면 FAIL_OR_UNKNOWN 이다.
- 레코드는 검증기가 스스로 재계산한 전역값만 쓴다 (dmin, amin, DV, CERTS).
- `verified_theorem` 필드:

| 필드 | 값 |
|---|---|
| verifier_status | PASS |
| object | Zeng_Pt, conv{p_i, −t p_i}, P0, t_shape "11/20", λ = 1 |
| Q | q1 [2043/5120, 2053/5120], q2 [4603/5120, 4613/5120] |
| rotation_convention | R_out(q) = Ry(−q1)·Rz(−q2), R_in = R_out(q)·expm(θ[u]×), Ry/Rz 정의, 그림자 = 처음 두 좌표, Δ = R_outᵀR_in |
| theta_range | ["0", "1/20"] |
| certificate_sha256 | afce2b74d9cd3e5464c1b6da6bfe903385e4ac8a711dc2a45f923403a20ac3a6 |
| verifier_sha256 | 861149c6dbf9e8b652de3b7e3442ca331880bf0cd243c4ffe20d81a087da517e |
| analytic_assumptions | 8항목 (삼각 부등식 2, 약쌍대, 고정 접촉 하계, Rodrigues, θ=0 보조정리, 정육면체-구면 피복, (0, 3/2) 단조성) |
| generator_summary_consistent | true |

- 생성기 요약 일치의 근거:
  - 구조 일치: 쌍대 꼭짓점 집합, 증명서 24개, 칸 96개.
  - 생성기 δ (0.057393) 와 검증기 δ (0.060360) 모두 1/20 이상이다.
  - 생성기 하한 ≤ 검증기 하한 (a, δ 모두).
- **검증 정리와 요청 정리 구별** (`evaluate_request`):

| 요청 θ_max | 결과 |
|---|---|
| 9/10 | **NOT_VERIFIED** (범위 밖, PASS 미발행) |
| 1/20 | VERIFIED |
| 3/50 | NOT_VERIFIED |

  - 3/50 은 생성기·검증기 δ 하한보다 작지만, 검증한 정리는 1/20 이다. 그래서 NOT_VERIFIED 다.
- 생성기의 더 큰 δ 와 기존 δ_Q = 0.0573932756328 은 `annotations_ignored_for_decision` 에만 둔다. 판정에는 쓰지 않는다.

## B. 칸 전체의 적용 조건
- **R8 인코딩 (실제 코드 `taskR8_Pt_lattice.py` 확인)**:
  - x = (α, φ₂, θ₂, θ₁, φ₁)
  - R_out = Ry(−θ₁)·Rz(−φ₁), R_in = Rz(α)·Ry(−θ₂)·Rz(−φ₂), 투영 = 회전 후 (x, y)
  - 물체 Pt(t) = vstack(P0, −t·P0), λ = 1
  - 격자 칸 = LO + W0·[k, k+1]/2^L, W0 = (2π, 2π, π, π/2, π/2)
  - → q = (θ₁, φ₁) 가 국소정리의 q 와 같은 convention 이다.
- **조건 1 (물체)**:
  - P0 정확 일치, t 정확 유리수 = 11/20, λ = 1, encoding R8_v1. 하나라도 다르면 NOT_APPLICABLE.
  - float t 는 정확 정의가 아니므로 NOT_APPLICABLE 이다. R8 코드의 t = 0.55 float 은 2476979795053773/2^52 ≠ 11/20 이다.
  - R8 호출자는 t 를 유리수 문자열 ("11/20" 또는 "0.55") 로 선언해야 한다.
- **조건 2 (q ⊆ Q)**:
  - θ₁, φ₁ 구간 끝점을 Q 의 유리수 끝점과 비교한다. π 배 칸은 π 구간을 곱해 바깥으로 감싼다.
  - 확실히 밖이면 UNKNOWN/OUTSIDE_Q, 걸침이면 UNKNOWN/STRADDLES_Q 다.
- **조건 3 (angle(Δ) ≤ 1/20, acos 미사용)**:
  - Δ ∈ SO(3) 는 정확 회전들의 곱이다. trace Δ = 1 + 2cosθ 이고, θ ∈ [0, π] 에서 cos 는 감소한다.
  - 따라서 θ ≤ 1/20 ⇔ trace Δ ≥ 1 + 2cos(1/20) (임계 구간 2.99750052078993) 이다.
  - **M1**: 칸 전체 구간으로 trace 를 직접 계산한다. lo(trace) ≥ hi(임계) 이면 인증이다.
  - **M2**: 쌍불변 거리 d(R,S) = angle(RᵀS) 를 쓴다.
    - Δ(x) = Rz(φ₁)Ry(θ₁)Rz(α)Ry(−θ₂)Rz(−φ₂) 의 각 인수는 단위속도 1-매개 부분군이다.
    - 따라서 angle(Δ(x)) ≤ angle(Δ(c)) + H 다 (H = 반폭 합의 상계).
    - 중심 trace 구간에서 cos angle(Δ(c)) ≥ κ 를 얻는다. κ ≥ hi(cos(1/20 − H)) 이면 칸 전체가 인증된다.
    - 중심점만 확인하는 방식이 아니다. H 가 칸 전체를 덮는다.
  - **밖 인증**: M1 hi(trace) < lo(임계) 이거나, M2 에서 angle(Δ(c)) − H > 1/20 이면 UNKNOWN/OUTSIDE_THETA 다.
  - 그 외는 UNKNOWN/STRADDLES_THETA (경계 걸침 또는 과대평가로 미확정) 다.
- **θ = 0**: Δ = I (trace = 3) 은 검증 정리 범위 [0, 1/20] 에 포함된다. 근거는 동일 투영 보조정리다.

## C. 반환 상태
- LOCAL_NO_STRICT_CONTAINMENT, NOT_APPLICABLE, UNKNOWN 세 가지에 detail 과 근거 구간을 붙인다.
- 범위 기록이 PASS 가 아니거나 증명서 해시가 불일치하면 UNKNOWN/SCOPE_NOT_VERIFIED 다.
- UNKNOWN 은 통과나 불통과로 바꾸지 않는다. legacy R8 판정기를 대체하지 않는 보조 모듈이다.

## D. 회귀시험 (기대값은 실행 전 `rupert_r8_adapter_tests.json` 에 고정)

| ID | 입력 | 기대 | 결과 | 근거 |
|---|---|---|---|---|
| T1 | 점, Δ = Rz(−1/100) | LOCAL | LOCAL (M1) | trace 2.9999000008 ≥ 2.9975005208 |
| T2 | θ = 0 동일 방향 | LOCAL | LOCAL (M1) | trace = 3 |
| T3 | 실제 R8 dyadic L=11 칸 | LOCAL | **LOCAL (M2)** | M1 은 과대평가로 미확정 (trace ∈ [2.99185, 3.00814]; 사전 노트대로). M2: 중심 trace 2.9999984, H ≤ 0.004602 |
| T4 | q1 이 Q 경계를 걸침 | UNKNOWN/STRADDLES_Q | 일치 | θ₁ ∈ [0.398779, 0.399268] ∋ 2043/5120 |
| T5 | α ∈ 1/20 ± 1/1000 | UNKNOWN/STRADDLES_THETA | 일치 | 중심 trace = 임계 |
| T6 | q 안, 회전 1/2 | UNKNOWN/OUTSIDE_THETA | 일치 | trace 2.7551651 < 임계 (M1·M2 모두 밖) |
| T7 | t = 3/5 | NOT_APPLICABLE | 일치 | 물체 불일치 |
| T8 | t = float 0.55 | NOT_APPLICABLE | 일치 | 정확 정의 아님 |
| T9 | 실제 R8 L=6 칸 (q0 포함) | UNKNOWN | UNKNOWN/STRADDLES_Q | 칸 폭 π/128 ≈ 0.0245 > Q 폭 1/512 |
| Q1–Q3 | 요청 θ_max 9/10, 1/20, 3/50 | NOT_VERIFIED, VERIFIED, NOT_VERIFIED | 일치 | |

- **기대 일치 12/12**.
- 과대평가 사례: T3 의 직접 trace (M1) 는 내부 칸인데도 미확정이었다. 사전 노트대로이며, M2 로 인증되었다.
- 표본으로 덮어쓴 결과는 없다.

## 관찰 (R8 접속 시 유의점)
- **R8 의 세분 상한 L ≤ 6 칸은 이 국소정리에 구조적으로 들어올 수 없다.**
  - θ₁·φ₁ 칸 폭 π/128 이 Q 폭 1/512 보다 크다.
  - Q 안에 들어가는 θ₁·φ₁ 격자 칸은 L ≥ 11 부터 존재한다 (T3).
  - 접속은 R8 이 그 수준까지 세분할 때만 효력이 있다. 세분 정책 변경은 이번 범위 밖이다.
- R8 은 t 와 LO/HI 를 float 으로 쓴다. 어댑터에는 정확 t 와 π 배 칸 끝점 (`units: "pi"`) 으로 전달해야 한다.
- 대칭 복제 (T_d, ±) 는 하지 않았다. 대칭 동치 칸은 UNKNOWN 으로 남는다.

## 범위
- 이 어댑터는 검증된 국소 영역 하나 (Q × angle(Δ) ≤ 1/20) 에 대한 칸 적용 판정이다. 전역 비루퍼트 판정이 아니다.
- 범위 밖: 전체 탐색, 영역 확대, 대칭 복제, EQI 동역학.

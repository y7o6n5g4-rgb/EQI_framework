# index/evidence/ — 독립 대조 증빙 (단계별)

> 원문·원장과 별개로, 각 단계 표본의 독립 대조 (perplexity) 결과를 보관. 파일명 앞 `stepN_` = 단계.
> CSV 는 MAPSI 업로드 파일을 바이트 그대로 복사 (sha256 앞 16자 표기). md 는 MAPSI 가 대화로 전달한 결과 문구를 원문 그대로 옮긴 것 (원본 파일 없음).

| 단계 | 파일 | 대상 표본 | 내용 |
|---|---|---|---|
| 2 | `step2_independent_check.md` | sample_check_2.md | 전달 문구 (인용 40줄 불일치 0, blob SHA, mpmath 1.3.0 재실행) |
| 3 | `step3_independent_check.md` | sample_check_3.md | 전달 문구 (인용 61줄, 이름 대응 4건, v3.4 blob SHA) |
| 4a | `step4a_independent_check.md` | sample_check_4a.md | 전달 문구 (인용 31줄, 근거 추가 3행, 내려받기 4행, 블록 표제 전달) |
| 4b | `step4b_independent_check.md` | sample_check_4b.md | 전달 문구 (인용 29줄, L0442, § 현황 줄, 보류 3행) |
| 5 | `step5_independent_check.md` | sample_check_5·5b·5c.md | 1차 통보 + perplexity 응답 붙여넣기 원문 + MAPSI 정리 |
| 5 | `step5_independent_sample_5_5b_5c_quotes.csv` | 〃 | 원문 인용 58줄 대조 (sha256 57089e468bd6a046…) |
| 5 | `step5_independent_sample_5_5b_5c_ledger_fields.csv` | 〃 | 표본 30행 실제 원장 필드 (sha256 95cf96eab76904d7…) |
| 5 | `step5_samples5_5b_5c_line_checks.csv` | 〃 | 줄 대조 43행 (sha256 0e56fef89dcedd20…) |

- 단계 6 (`sample_check_6.md`) 대조 결과가 오면 `step6_…` 로 추가.

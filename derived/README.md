# derived/ — 원문 docx 변환본 (원문 아님)

- 생성: `python3 tools/make_derived.py` / pandoc 3.1.3 / `pandoc -f docx -t gfm --wrap=none`
- 원문은 `성과정리문서/` 의 docx 그대로 (읽기 전용). 이 폴더의 행 번호는 변환본 기준이며, 색인에서 인용할 때는 원문 docx 파일명을 함께 적는다.
- 변환 대상: doc_map.csv 분류 sessions·foundation 의 docx 전부. 그 밖의 docx (Framework v3~v17 등 역사 문서) 는 아직 변환하지 않음.

| 변환본 | 원문 | 원문 sha256 | 변환본 행 수 | 최종판 |
|---|---|---|---|---|
| `derived/EQI_119_압축본_v10.md` | `성과정리문서/EQI_119_압축본_v10.docx` | f02cb398f99d9ffb… | 635 | Y |
| `derived/EQI_Session109_성과정리_v4.md` | `성과정리문서/EQI_Session109_성과정리_v4.docx` | f66778c8cdbf8cc4… | 187 | Y |
| `derived/EQI_Session110_성과정리_v4.md` | `성과정리문서/EQI_Session110_성과정리_v4.docx` | 9005c89fef1d4686… | 179 | Y |
| `derived/EQI_Session111_인수인계_v3.md` | `성과정리문서/EQI_Session111_인수인계_v3.docx` | 35e097cf7e85497f… | 299 | Y |
| `derived/EQI_Session29_성과정리_v9_final.md` | `성과정리문서/EQI_Session29_성과정리_v9_final.docx` | d0b87df0f3f5e584… | 190 | Y |
| `derived/EQI_Session37_성과정리_v1.md` | `성과정리문서/EQI_Session37_성과정리_v1.docx` | 48643301f54794b2… | 121 | Y |

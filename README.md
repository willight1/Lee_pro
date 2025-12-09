# Steel Tariff Analysis System

철강제품 관세(반덤핑) 전문 분석 시스템입니다. PDF 문서에서 HS Code 및 관세 정보를 자동으로 추출하고 구조화된 데이터로 변환합니다.

## 주요 기능

- **PDF 파싱**: pdfplumber를 사용한 고품질 텍스트 추출
- **HS Code 추출**: 정규식 기반 자동 HS Code 탐지 (예: 7209.15.0000)
- **AI 기반 데이터 구조화**: OpenAI GPT-4o-mini를 활용한 지능형 정보 추출
- **CSV 관리**: 추출 데이터를 CSV로 저장 및 누적 관리
- **웹 인터페이스**: Streamlit 기반 사용자 친화적 UI
- **데이터 시각화**: 추출된 데이터의 통계 및 시각화
- **다국가 확장 가능**: 국가별 프롬프트 템플릿 지원

## 프로젝트 구조

```
lee_pro2/
├── PDF/                                    # 입력 PDF 파일 저장 폴더
│   ├── USA_CR_Antidumping_A-580-881_2016.pdf
│   └── USA_CR_Antidumping_A-580-881.pdf
├── CSV/                                    # 출력 CSV 파일 저장 폴더
│   └── tariff_data.csv                     # 추출된 데이터 (자동 생성)
├── prompt_templet/                         # 국가별 프롬프트 템플릿
│   └── usa_prompt.txt                      # USA 프롬프트
├── .env                                    # API 키 설정 (직접 생성)
├── .env.example                            # API 키 설정 예시
├── requirements.txt                        # Python 의존성
├── llm.py                                  # 핵심 분석 로직
├── streamlit_app.py                        # 웹 인터페이스
└── README.md                               # 이 파일
```

## 설치 방법

### 1. 가상환경 생성 및 활성화

```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. OpenAI API 키 설정

1. `.env.example` 파일을 복사하여 `.env` 파일 생성:
   ```bash
   cp .env.example .env
   ```

2. `.env` 파일을 편집하여 실제 API 키 입력:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. OpenAI API 키는 [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급받을 수 있습니다.

### 4. PDF 파일 준비

분석할 PDF 파일을 `PDF/` 폴더에 저장합니다:
- **HS Code PDF**: HS Code 목록이 포함된 PDF
- **Detail PDF**: 상세 관세 정보가 포함된 PDF

## 사용 방법

### 방법 1: Streamlit 웹 인터페이스 (권장)

```bash
streamlit run streamlit_app.py
```

브라우저가 자동으로 열리며 웹 인터페이스가 표시됩니다.

#### 웹 인터페이스 사용법

1. **PDF Processing 페이지**:
   - 두 개의 PDF 파일 업로드 (HS Code PDF, Detail PDF)
   - 국가 선택 (현재 USA 지원)
   - "Extract Data" 버튼 클릭
   - 추출된 데이터 확인 및 CSV 다운로드

2. **Data Visualization 페이지**:
   - `CSV/tariff_data.csv` 파일의 데이터 시각화
   - **Data Table 탭**: 전체 데이터 테이블 보기
   - **Statistics 탭**: 통계 및 차트
   - **Filter 탭**: 국가, 회사, 케이스 번호, HS Code로 필터링
   - **Export 탭**: CSV/Excel 형식으로 다운로드

### 방법 2: Python 스크립트 직접 실행

```bash
python llm.py
```

스크립트가 `PDF/` 폴더의 파일을 자동으로 처리하고 결과를 `CSV/` 폴더에 저장합니다.

## 출력 데이터 형식

추출된 데이터는 다음 컬럼을 포함합니다:

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| `hs_code` | HS Code | 7209.15.0000 |
| `issuing_country` | 발행 국가 | USA |
| `country` | 대상 국가 | Korea |
| `tariff_type` | 관세 유형 | Antidumping |
| `tariff_rate` | 관세율 (%) | 0 |
| `effective_date_from` | 발효 시작일 | 2025-01-15 |
| `effective_date_to` | 발효 종료일 | |
| `investigation_period_from` | 조사 기간 시작일 | 2022-09-01 |
| `investigation_period_to` | 조사 기간 종료일 | 2023-08-31 |
| `company` | 회사명 | Hyundai Steel Company |
| `case_number` | 케이스 번호 | A-580-881 |
| `product_description` | 제품 설명 | Certain Cold-Rolled Steel Flat Products |
| `note` | 비고 | Final results of AD administrative review |

## 주요 기능 설명

### 1. HS Code 추출
- 정규식 패턴 `\d{4}\.\d{2}\.\d{4}`을 사용하여 자동 추출
- 중복 제거 및 정렬
- 예시: 7209.15.0000, 7225.50.6000

### 2. LLM 기반 정보 추출
- OpenAI GPT-4o-mini 모델 사용
- JSON 형식으로 구조화된 데이터 반환
- 재시도 로직으로 안정성 보장
- 국가별 맞춤형 프롬프트 템플릿

### 3. CSV 누적 관리
- 새 데이터를 기존 CSV에 자동 추가
- 중복 제거 자동 처리
- UTF-8 BOM 인코딩으로 한글 지원

### 4. 확장 가능한 구조
- 국가별 프롬프트 템플릿 분리
- 향후 다른 국가 문서 처리 가능
- `prompt_templet/` 폴더에 새 템플릿 추가

## 로깅

모든 작업은 상세하게 로깅됩니다:
- INFO: 일반 작업 진행 상황
- WARNING: 경고 사항
- ERROR: 오류 발생

로그 예시:
```
2025-01-15 10:30:00 - __main__ - INFO - Extracting text from PDF/USA_CR_Antidumping_A-580-881_2016.pdf
2025-01-15 10:30:05 - __main__ - INFO - Found 15 unique HS codes
2025-01-15 10:30:10 - __main__ - INFO - Calling OpenAI API (attempt 1/3)
2025-01-15 10:30:15 - __main__ - INFO - Successfully parsed 2 records
```

## 오류 해결

### API 키 오류
```
ValueError: OpenAI API key not found
```
→ `.env` 파일에 `OPENAI_API_KEY`가 올바르게 설정되었는지 확인

### PDF 파일을 찾을 수 없음
```
ERROR: HS Code PDF not found
```
→ `PDF/` 폴더에 PDF 파일이 있는지 확인

### JSON 파싱 오류
- LLM 응답이 올바른 JSON 형식이 아닐 경우 자동으로 재시도
- 3회 재시도 후에도 실패하면 오류 로그 확인

## 개발 환경

- Python 3.8 이상
- macOS, Linux, Windows 지원
- 권장 메모리: 4GB 이상

## 라이센스

이 프로젝트는 내부 사용을 위한 것입니다.

## 기술 지원

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.

## 향후 개발 계획

- [ ] 더 많은 국가 지원 (중국, EU 등)
- [ ] 배치 처리 기능
- [ ] 데이터 검증 강화
- [ ] API 엔드포인트 제공
- [ ] 더 많은 차트 유형 지원
# Lee_pro

"""
SQLite-based Steel Tariff Analysis - LLM Integration Module
기존 CSV 기반 `llm.py`를 참고하여, 동일한 분석 파이프라인을 SQLite DB 저장 방식으로 확장한 버전입니다.
주의: 이 파일은 `llm.py`를 수정하지 않고, 별도의 DB 버전을 제공합니다.
"""

import re
import json
import logging
import os
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path

import pdfplumber
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging (기존 llm.py와 동일한 포맷 유지)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TariffAnalyzer:
    """Main class for analyzing steel tariff PDF documents (SQLite 버전)"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TariffAnalyzer

        Args:
            api_key: OpenAI API key. If None, loads from environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")

        # Validate API key format
        if not self.api_key.startswith(("sk-", "sk-proj-")):
            logger.warning(f"API 키 형식이 예상과 다릅니다. (시작: {self.api_key[:10]}...)")

        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            raise

        # HS Code patterns (llm.py와 동일 로직)
        self.hs_code_pattern = re.compile(r"\b\d{4}\.\d{2}\.\d{2,4}(?:\s?\d{2})?\b")
        self.eu_taric_pattern = re.compile(r"\b(722[56])\s*(\d{2})\s*(\d{2})\s*(\d{2})\b")

    # ------------------------------------------------------------------
    # PDF/LLM 관련 메서드: llm.py와 동일 구조를 유지
    # ------------------------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        try:
            logger.info(f"Extracting text from {pdf_path}")
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}"

            logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

    def extract_hs_codes(self, text: str, country: str = "usa") -> List[str]:
        """
        Extract HS Codes from text using regex pattern
        (llm.py와 동일한 정규식 및 국가별 처리 로직)
        """
        try:
            logger.info("Extracting HS codes from text")
            normalized_codes: List[str] = []

            # Standard pattern (USA, Malaysia, Australia, generic HS with two dots)
            raw_codes = self.hs_code_pattern.findall(text)
            for code in raw_codes:
                clean_code = code.replace(" ", "")
                parts = clean_code.split(".")
                if len(parts) == 3:
                    if len(parts[2]) == 2:
                        parts[2] = parts[2] + "00"
                    elif len(parts[2]) == 4:
                        pass
                    normalized_code = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    normalized_codes.append(normalized_code)

            # Pakistan PCT codes
            if country == "pakistan":
                pct_pattern = re.compile(r"\b\d{4}\.\d{4}\b")
                pct_codes = pct_pattern.findall(text)
                for code in pct_codes:
                    clean_code = code.replace(" ", "")
                    left, right = clean_code.split(".")
                    subheading = right[:2]
                    national = right[2:]
                    national_padded = national.ljust(4, "0")
                    normalized_code = f"{left}.{subheading}.{national_padded}"
                    normalized_codes.append(normalized_code)

            # EU TARIC pattern
            if country == "eu" or len(normalized_codes) == 0:
                eu_codes = self.eu_taric_pattern.findall(text)
                for match in eu_codes:
                    part1 = match[0]
                    part2 = match[1]
                    part3 = match[2] + match[3]
                    normalized_code = f"{part1}.{part2}.{part3}"
                    normalized_codes.append(normalized_code)

            unique_codes = sorted(set(normalized_codes))
            logger.info(f"Found {len(unique_codes)} unique HS codes")
            return unique_codes
        except Exception as e:
            logger.error(f"Error extracting HS codes: {e}")
            return []

    def load_prompt_template(self, country: str = "usa") -> str:
        """Load system prompt template for a specific country"""
        try:
            prompt_path = Path("prompt_templet") / f"{country}_prompt.txt"
            logger.info(f"Loading prompt template from {prompt_path}")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            return prompt
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            raise

    def extract_structured_data(
        self,
        text: str,
        country: str = "usa",
        max_retries: int = 3,
        max_chunk_size: int = 50000,
    ) -> List[Dict]:
        """Extract structured data from text using OpenAI API"""
        try:
            system_prompt = self.load_prompt_template(country)

            if len(text) > max_chunk_size:
                logger.warning(
                    f"텍스트가 너무 깁니다 ({len(text)} chars). 여러 청크로 나누어 처리합니다."
                )
                chunks = self._split_text_into_chunks(text, max_chunk_size)
                all_results: List[Dict] = []

                for i, chunk in enumerate(chunks):
                    logger.info(f"청크 {i + 1}/{len(chunks)} 처리 중...")
                    chunk_result = self._extract_from_chunk(
                        chunk, system_prompt, max_retries
                    )
                    if chunk_result:
                        all_results.extend(chunk_result)

                logger.info(f"총 {len(all_results)} 개의 레코드를 추출했습니다.")
                return all_results

            return self._extract_from_chunk(text, system_prompt, max_retries)
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            raise

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """텍스트를 청크로 나눕니다"""
        chunks: List[str] = []
        current_chunk = ""

        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_from_chunk(
        self,
        text: str,
        system_prompt: str,
        max_retries: int,
    ) -> List[Dict]:
        """단일 텍스트 청크에서 구조화된 데이터를 추출합니다"""
        try:
            for attempt in range(max_retries):
                try:
                    text_length = len(text)
                    prompt_length = len(system_prompt)
                    logger.info(
                        f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})"
                    )
                    logger.info(
                        f"Text length: {text_length} chars, Prompt length: {prompt_length} chars"
                    )

                    if text_length > 100000:
                        logger.warning(
                            f"텍스트가 매우 깁니다 ({text_length} chars). API 호출이 실패할 수 있습니다."
                        )

                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text},
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"},
                        timeout=120,
                    )

                    content = response.choices[0].message.content
                    logger.info("Received response from OpenAI API")

                    try:
                        data = json.loads(content)

                        if isinstance(data, list):
                            result = data
                        elif isinstance(data, dict):
                            if "companies" in data:
                                result = data["companies"]
                            elif "data" in data:
                                result = data["data"]
                            elif "results" in data:
                                result = data["results"]
                            elif "result" in data:
                                result = (
                                    data["result"]
                                    if isinstance(data["result"], list)
                                    else [data["result"]]
                                )
                            else:
                                result = [data]
                        else:
                            raise ValueError(f"Unexpected response format: {type(data)}")

                        logger.info(f"Successfully parsed {len(result)} records")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"JSON parse error (attempt {attempt + 1}): {e}"
                        )
                        if attempt == max_retries - 1:
                            raise
                        continue
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    logger.warning(
                        f"API call error (attempt {attempt + 1}/{max_retries}): {error_type} - {error_msg}"
                    )

                    if (
                        "authentication" in error_msg.lower()
                        or "api key" in error_msg.lower()
                        or "invalid" in error_msg.lower()
                    ):
                        logger.error(
                            "API 키 인증 오류: .env 파일의 OPENAI_API_KEY를 확인해주세요."
                        )
                    elif (
                        "rate limit" in error_msg.lower()
                        or "quota" in error_msg.lower()
                    ):
                        logger.error("API 할당량 초과: 잠시 후 다시 시도해주세요.")
                    elif (
                        "timeout" in error_msg.lower()
                        or "connection" in error_msg.lower()
                    ):
                        logger.error("네트워크 연결 오류: 인터넷 연결을 확인해주세요.")

                    if attempt == max_retries - 1:
                        raise Exception(
                            f"API 호출 실패 (최대 재시도 횟수 초과): {error_type} - {error_msg}"
                        )
                    continue

            return []
        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
            raise

    def create_records(
        self,
        hs_codes: List[str],
        structured_data: List[Dict],
        country: str = "usa",
    ) -> pd.DataFrame:
        """
        HS 코드와 LLM이 추출한 구조화 데이터를 결합하여 DataFrame 생성
        (기존 llm.py의 create_records 로직을 그대로 사용)
        """
        try:
            logger.info("Creating records DataFrame")

            records = []
            for hs_code in hs_codes:
                for company_data in structured_data:
                    case_number = company_data.get("case_number", "")
                    normalized_case = case_number.replace("–", "-").replace("—", "-")

                    # 데이터 품질 필터(미국 580 케이스만 허용)는 제거하고,
                    # 단순히 포맷 정규화 및 기본값 설정만 수행합니다.
                    if country == "usa":
                        if normalized_case.startswith("A-"):
                            tariff_type = "Antidumping"
                        elif normalized_case.startswith("C-"):
                            tariff_type = "Countervailing"
                        else:
                            tariff_type = company_data.get("tariff_type", "Antidumping")

                        case_number = normalized_case
                        issuing_country = "USA"
                    elif country == "malaysia":
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case
                        issuing_country = "Malaysia"
                    elif country == "australia":
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case
                        issuing_country = "Australia"
                    elif country == "pakistan":
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case
                        issuing_country = "Pakistan"
                    elif country == "eu":
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case
                        issuing_country = "EU"
                    else:
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        issuing_country = company_data.get(
                            "issuing_country", country.upper()
                        )

                    record = {
                        "hs_code": hs_code,
                        "issuing_country": company_data.get(
                            "issuing_country", issuing_country
                        ),
                        "country": company_data.get("country", ""),
                        "tariff_type": tariff_type,
                        "tariff_rate": company_data.get("tariff_rate", ""),
                        "effective_date_from": company_data.get(
                            "effective_date_from", ""
                        ),
                        "effective_date_to": company_data.get("effective_date_to", ""),
                        "investigation_period_from": company_data.get(
                            "investigation_period_from", ""
                        ),
                        "investigation_period_to": company_data.get(
                            "investigation_period_to", ""
                        ),
                        "company": company_data.get("company", ""),
                        "case_number": case_number,
                        "product_description": company_data.get(
                            "product_description", ""
                        ),
                        "note": company_data.get("note", ""),
                    }
                    records.append(record)

            df = pd.DataFrame(records)
            logger.info(f"Created DataFrame with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error creating records: {e}")
            raise

    def save_to_csv(
        self,
        df: pd.DataFrame,
        output_path: str = "CSV/tariff_data.csv",
        append: bool = True,
    ):
        """
        기존 CSV 저장 메서드 (llm.py와 동일, 호환성 유지용)
        주 로직은 SQLite DB를 사용하지만, 필요 시 CSV 저장도 지원하기 위해 남겨둡니다.
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if append and output_file.exists():
                logger.info(f"Appending to existing CSV: {output_path}")
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates()
                combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(
                    f"Appended {len(df)} records (total: {len(combined_df)} records)"
                )
            else:
                logger.info(f"Writing new CSV: {output_path}")
                df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved {len(df)} records")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise

    # ------------------------------------------------------------------
    # SQLite 저장/업데이트 및 DB → CSV export 메서드
    # ------------------------------------------------------------------
    def _ensure_db_schema(
        self,
        conn: sqlite3.Connection,
        table_name: str = "tariff_data",
    ) -> None:
        """
        필요한 경우 tariff_data 테이블을 생성하고, 스키마/UNIQUE 제약을 보장합니다.
        스키마:
            id INTEGER PRIMARY KEY AUTOINCREMENT
            hs_code TEXT
            issuing_country TEXT
            country TEXT
            tariff_type TEXT
            tariff_rate TEXT
            effective_date_from TEXT
            effective_date_to TEXT
            investigation_period_from TEXT
            investigation_period_to TEXT
            company TEXT
            case_number TEXT
            product_description TEXT
            note TEXT
            UNIQUE(hs_code, case_number, company, tariff_type, effective_date_from)
        """
        logger.info(f"Ensuring DB schema for table '{table_name}'")
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hs_code TEXT,
            issuing_country TEXT,
            country TEXT,
            tariff_type TEXT,
            tariff_rate TEXT,
            effective_date_from TEXT,
            effective_date_to TEXT,
            investigation_period_from TEXT,
            investigation_period_to TEXT,
            company TEXT,
            case_number TEXT,
            product_description TEXT,
            note TEXT,
            UNIQUE(hs_code, case_number, company, tariff_type, effective_date_from)
        );
        """
        conn.execute(create_table_sql)
        conn.commit()

    def save_to_db(
        self,
        df: pd.DataFrame,
        db_path: str = "DB/tariff_data.db",
        table_name: str = "tariff_data",
    ) -> None:
        """
        DataFrame을 SQLite DB에 저장합니다.

        - DB 폴더가 없으면 생성
        - 테이블이 없으면 스키마/UNIQUE 제약과 함께 생성
        - 복합 UNIQUE 키(hs_code, case_number, company, tariff_type, effective_date_from)
          기준으로 INSERT OR REPLACE upsert 수행

        이 메서드는 CSV 대신 DB를 메인 저장소로 사용하기 위한 핵심 엔트리 포인트입니다.
        """
        if df is None or df.empty:
            logger.info("save_to_db 호출됨: 빈 DataFrame, DB에 저장하지 않습니다.")
            return

        try:
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Connecting to SQLite DB: {db_path}")
            with sqlite3.connect(db_path) as conn:
                # 스키마 보장
                self._ensure_db_schema(conn, table_name=table_name)

                # upsert용 SQL (UNIQUE 제약에 의해 기존 레코드는 교체)
                insert_sql = f"""
                INSERT OR REPLACE INTO {table_name} (
                    hs_code,
                    issuing_country,
                    country,
                    tariff_type,
                    tariff_rate,
                    effective_date_from,
                    effective_date_to,
                    investigation_period_from,
                    investigation_period_to,
                    company,
                    case_number,
                    product_description,
                    note
                ) VALUES (
                    :hs_code,
                    :issuing_country,
                    :country,
                    :tariff_type,
                    :tariff_rate,
                    :effective_date_from,
                    :effective_date_to,
                    :investigation_period_from,
                    :investigation_period_to,
                    :company,
                    :case_number,
                    :product_description,
                    :note
                );
                """

                records = df.to_dict(orient="records")
                logger.info(f"Upserting {len(records)} records into DB...")
                conn.executemany(insert_sql, records)
                conn.commit()

                logger.info(
                    f"DB 저장 완료: {db_path} / 테이블 '{table_name}'에 {len(records)} 개 레코드 upsert"
                )
        except Exception as e:
            logger.error(f"Error saving to DB: {e}")
            raise

    def export_db_to_csv(
        self,
        db_path: str = "DB/tariff_data.db",
        table_name: str = "tariff_data",
        output_csv: str = "CSV/tariff_data_from_db.csv",
    ) -> None:
        """
        SQLite DB 내용을 CSV로 추출하는 메서드.

        - SELECT * FROM table_name 으로 전체 데이터를 읽어 DataFrame 생성
        - output_csv 경로의 폴더가 없으면 생성
        - UTF-8-SIG 인코딩으로 CSV 저장
        """
        try:
            if not Path(db_path).exists():
                logger.error(f"DB 파일이 존재하지 않습니다: {db_path}")
                return

            logger.info(
                f"Exporting DB table '{table_name}' from {db_path} to CSV: {output_csv}"
            )
            with sqlite3.connect(db_path) as conn:
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, conn)

            output_file = Path(output_csv)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

            logger.info(
                f"DB → CSV export 완료: {len(df)} records written to {output_csv}"
            )
        except Exception as e:
            logger.error(f"Error exporting DB to CSV: {e}")
            raise

    # ------------------------------------------------------------------
    # PDF 처리 파이프라인 (DB 중심으로 확장된 process_pdfs)
    # ------------------------------------------------------------------
    def process_pdfs(
        self,
        hs_code_pdf: str,
        detail_pdf: str,
        country: str = "usa",
        save_csv: bool = False,
        output_csv: str = "CSV/tariff_data.csv",
        append: bool = True,
        save_db: bool = True,
        db_path: str = "DB/tariff_data.db",
        table_name: str = "tariff_data",
    ) -> pd.DataFrame:
        """
        두 개의 PDF 파일을 처리하여 구조화된 데이터를 생성하고,
        기본적으로 SQLite DB에 저장하는 확장된 버전의 process_pdfs.

        - save_db=True: self.save_to_db(df, db_path, table_name) 호출 (기본값)
        - save_csv 옵션은 기존과 동일하게 유지하되, 기본값은 False로 변경
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting PDF processing (SQLite 버전)")
            logger.info("=" * 80)

            # Step 1: Extract HS codes
            logger.info(f"Step 1: Extracting HS codes from {hs_code_pdf}")
            hs_code_text = self.extract_text_from_pdf(hs_code_pdf)
            hs_codes = self.extract_hs_codes(hs_code_text, country)

            if not hs_codes:
                logger.warning("No HS codes found in the first PDF")
                return pd.DataFrame()

            # Step 2: Extract detailed information
            logger.info(f"Step 2: Extracting detailed information from {detail_pdf}")
            detail_text = self.extract_text_from_pdf(detail_pdf)

            # Step 3: Use LLM to structure the data
            logger.info("Step 3: Extracting structured data using LLM")
            structured_data = self.extract_structured_data(detail_text, country)

            if not structured_data:
                logger.warning("No structured data extracted from the second PDF")
                return pd.DataFrame()

            # Step 4: Combine HS codes with structured data
            logger.info("Step 4: Creating final records")
            df = self.create_records(hs_codes, structured_data, country)

            # Step 5: Save to DB (기본) 및 선택적 CSV 저장
            if save_db:
                logger.info("Step 5(DB): Saving to SQLite DB")
                self.save_to_db(df, db_path=db_path, table_name=table_name)

            if save_csv:
                logger.info("Step 5(CSV): Saving to CSV")
                self.save_to_csv(df, output_path=output_csv, append=append)

            logger.info("=" * 80)
            logger.info("PDF processing completed successfully (SQLite 버전)")
            logger.info("=" * 80)

            return df
        except Exception as e:
            logger.error(f"Error processing PDFs (SQLite 버전): {e}")
            raise


# ----------------------------------------------------------------------
# 보조 함수들: llm.py와 동일한 인터페이스를 유지하되, TariffAnalyzer_SQLite를 사용
# ----------------------------------------------------------------------
def detect_country_from_filename(filename: str) -> str:
    """
    Detect country code from PDF filename (llm.py와 동일 로직)
    """
    filename_upper = filename.upper()

    if "MALAYSIA" in filename_upper:
        return "malaysia"
    elif "AUSTRALIA" in filename_upper:
        return "australia"
    elif "PAKISTAN" in filename_upper:
        return "pakistan"
    elif "EU" in filename_upper or "COMMISSION" in filename_upper or re.search(
        r"EU[_\s]*\d{4}[/_]\d+", filename_upper
    ):
        return "eu"
    elif "USA" in filename_upper or re.match(r".*[AC]-580-\d{3}.*", filename_upper):
        return "usa"
    else:
        return "usa"


def find_pdf_pairs(pdf_folder: str = "PDF") -> List[tuple]:
    """
    Automatically find and pair HS Code PDFs with Detail PDFs based on case number
    (llm.py의 find_pdf_pairs를 그대로 복사하여 사용)
    """
    import re as _re

    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        logger.error(f"PDF folder not found: {pdf_folder}")
        return []

    all_pdfs = sorted(pdf_path.glob("*.pdf"))

    usa_pdfs = []
    malaysia_pdfs = []
    australia_pdfs = []
    pakistan_pdfs = []
    eu_pdfs = []

    for pdf_file in all_pdfs:
        country = detect_country_from_filename(pdf_file.name)
        if country == "malaysia":
            malaysia_pdfs.append(pdf_file)
        elif country == "australia":
            australia_pdfs.append(pdf_file)
        elif country == "pakistan":
            pakistan_pdfs.append(pdf_file)
        elif country == "eu":
            eu_pdfs.append(pdf_file)
        else:
            usa_pdfs.append(pdf_file)

    case_pattern = _re.compile(r"([AC]-580-\d{3})")

    case_groups: Dict[str, List[Path]] = {}
    for pdf_file in usa_pdfs:
        match = case_pattern.search(pdf_file.name)
        if match:
            case_num = match.group(1)
            if case_num not in case_groups:
                case_groups[case_num] = []
            case_groups[case_num].append(pdf_file)

    pairs: List[tuple] = []
    for case_num, files in case_groups.items():
        hs_code_files = [
            f
            for f in files
            if _re.search(
                r"_\d{4}\.pdf$|_F_\d{4}\.pdf$|_Pre_\d{4}\.pdf$|_연장_\d{4}\.pdf$", f.name
            )
        ]
        detail_files = [f for f in files if f not in hs_code_files]

        if hs_code_files and detail_files:
            hs_code_pdf = str(hs_code_files[0])
            detail_pdf = str(detail_files[0])
            pairs.append((hs_code_pdf, detail_pdf, case_num, "usa"))
            logger.info(f"[USA] Paired: {hs_code_files[0].name} <-> {detail_files[0].name}")
        elif hs_code_files and not detail_files:
            import pdfplumber as _pdfplumber

            hs_code_pattern = _re.compile(r"\b\d{4}\.\d{2}\.\d{4}\b")

            best_file: Optional[Path] = None
            max_hs_codes = 0

            for pdf_file in hs_code_files:
                try:
                    with _pdfplumber.open(str(pdf_file)) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text

                        hs_codes = hs_code_pattern.findall(text)
                        unique_count = len(set(hs_codes))

                        if unique_count > max_hs_codes:
                            max_hs_codes = unique_count
                            best_file = pdf_file

                        logger.info(
                            f"  {pdf_file.name}: {unique_count} unique HS codes"
                        )
                except Exception as e:
                    logger.warning(f"  Error reading {pdf_file.name}: {e}")
                    continue

            if best_file and max_hs_codes > 0:
                selected_file = str(best_file)
                pairs.append((selected_file, selected_file, case_num, "usa"))
                logger.info(
                    f"[USA] Selected (contains both HS and Detail): {best_file.name} ({max_hs_codes} HS codes)"
                )
            else:
                logger.warning(
                    f"[USA] No valid file found for case {case_num}: all files have 0 HS codes"
                )
        else:
            logger.warning(
                f"[USA] Could not pair case {case_num}: HS={len(hs_code_files)}, Detail={len(detail_files)}"
            )

    for pdf_file in malaysia_pdfs:
        pdf_path_str = str(pdf_file)
        pu_match = _re.search(
            r"PU\s*\(?A\)?\s*(\d+)|P\.U\.\s*\(A\)\s*(\d+)", pdf_file.name, _re.IGNORECASE
        )
        if pu_match:
            pu_num = pu_match.group(1) or pu_match.group(2)
            case_num = f"PUA{pu_num}"
        else:
            case_num = pdf_file.stem

        pairs.append((pdf_path_str, pdf_path_str, case_num, "malaysia"))
        logger.info(f"[Malaysia] Added: {pdf_file.name} (case: {case_num})")

    for pdf_file in australia_pdfs:
        pdf_path_str = str(pdf_file)
        adn_match = _re.search(
            r"ADN[_\s]*(\d{4})[_\s]*(\d+)", pdf_file.name, _re.IGNORECASE
        )
        if adn_match:
            year = adn_match.group(1)
            num = adn_match.group(2)
            case_num = f"ADN {year}/{num}"
        else:
            rep_match = _re.search(r"REP[_\s]*(\d+)", pdf_file.name, _re.IGNORECASE)
            if rep_match:
                case_num = f"REP {rep_match.group(1)}"
            else:
                case_num = pdf_file.stem

        pairs.append((pdf_path_str, pdf_path_str, case_num, "australia"))
        logger.info(f"[Australia] Added: {pdf_file.name} (case: {case_num})")

    for pdf_file in eu_pdfs:
        pdf_path_str = str(pdf_file)
        eu_match = _re.search(
            r"EU[_\s]*(\d{4})[/_\s]*(\d+)", pdf_file.name, _re.IGNORECASE
        )
        if eu_match:
            year = eu_match.group(1)
            num = eu_match.group(2)
            case_num = f"EU {year}/{num}"
        else:
            ad_match = _re.search(
                r"AD(\d+)[_\s]*R(\d+)", pdf_file.name, _re.IGNORECASE
            )
            if ad_match:
                ad_num = ad_match.group(1)
                r_num = ad_match.group(2)
                case_num = f"AD{ad_num}/R{r_num}"
            else:
                case_num = pdf_file.stem

        pairs.append((pdf_path_str, pdf_path_str, case_num, "eu"))
        logger.info(f"[EU] Added: {pdf_file.name} (case: {case_num})")

    for pdf_file in pakistan_pdfs:
        pdf_path_str = str(pdf_file)
        adc_match = _re.search(
            r"A\.?D\.?C[_.\s-]*No[_.\s-]*_?(\d+)", pdf_file.name, _re.IGNORECASE
        )
        if adc_match:
            adc_num = adc_match.group(1)
            case_num = f"ADC {adc_num}"
        else:
            case_num = pdf_file.stem

        pairs.append((pdf_path_str, pdf_path_str, case_num, "pakistan"))
        logger.info(f"[Pakistan] Added: {pdf_file.name} (case: {case_num})")

    return pairs


# ----------------------------------------------------------------------
# 모듈 레벨 헬퍼 함수: TariffAnalyzer 인스턴스를 이용한 래퍼
# 사용자가 요구한 함수 이름을 직접 호출할 수 있도록 제공
# ----------------------------------------------------------------------
def save_to_db(
    df: pd.DataFrame,
    db_path: str = "DB/tariff_data.db",
    table_name: str = "tariff_data",
) -> None:
    """
    모듈 레벨 함수: 내부적으로 TariffAnalyzer().save_to_db()를 호출합니다.
    간단한 스크립트나 다른 모듈에서 편하게 재사용할 수 있도록 제공.
    """
    analyzer = TariffAnalyzer()
    analyzer.save_to_db(df, db_path=db_path, table_name=table_name)


def export_db_to_csv(
    db_path: str = "DB/tariff_data.db",
    table_name: str = "tariff_data",
    output_csv: str = "CSV/tariff_data_from_db.csv",
) -> None:
    """
    모듈 레벨 함수: TariffAnalyzer().export_db_to_csv() 래퍼.
    Streamlit 등에서 간단히 호출할 수 있도록 제공.
    """
    analyzer = TariffAnalyzer()
    analyzer.export_db_to_csv(db_path=db_path, table_name=table_name, output_csv=output_csv)


def process_pdfs(
    hs_code_pdf: str,
    detail_pdf: str,
    country: str = "usa",
    save_csv: bool = False,
    output_csv: str = "CSV/tariff_data.csv",
    append: bool = True,
    save_db: bool = True,
    db_path: str = "DB/tariff_data.db",
    table_name: str = "tariff_data",
) -> pd.DataFrame:
    """
    모듈 레벨 함수 버전의 process_pdfs.
    내부적으로 TariffAnalyzer().process_pdfs(...)를 호출합니다.
    """
    analyzer = TariffAnalyzer()
    return analyzer.process_pdfs(
        hs_code_pdf=hs_code_pdf,
        detail_pdf=detail_pdf,
        country=country,
        save_csv=save_csv,
        output_csv=output_csv,
        append=append,
        save_db=save_db,
        db_path=db_path,
        table_name=table_name,
    )


def main():
    """
    SQLite DB 버전 메인 함수.

    - PDF 폴더에서 모든 PDF pair를 찾고
    - 각 pair마다 TariffAnalyzer.process_pdfs를 호출하여 즉시 DB에 upsert
    - 마지막에 DB를 직접 읽어 요약 통계를 출력
      (필요 시 export_db_to_csv로 CSV도 한 번에 생성 가능)
    """
    try:
        analyzer = TariffAnalyzer()

        logger.info("Scanning PDF folder for file pairs (SQLite 버전)...")
        pdf_pairs = find_pdf_pairs("PDF")

        if not pdf_pairs:
            logger.error("No PDF pairs found in PDF folder")
            return

        logger.info(f"Found {len(pdf_pairs)} PDF pair(s) to process (SQLite 버전)")

        all_dfs: List[pd.DataFrame] = []
        for idx, (hs_code_pdf, detail_pdf, case_num, country) in enumerate(
            pdf_pairs, 1
        ):
            logger.info("\n" + "=" * 80)
            logger.info(
                f"Processing pair {idx}/{len(pdf_pairs)}: {case_num} ({country.upper()}) [SQLite]"
            )
            logger.info("=" * 80)

            try:
                # 각 pair별로 즉시 DB에 반영 (save_db=True)
                df = analyzer.process_pdfs(
                    hs_code_pdf=hs_code_pdf,
                    detail_pdf=detail_pdf,
                    country=country,
                    save_csv=False,
                    save_db=True,
                    db_path="DB/tariff_data.db",
                    table_name="tariff_data",
                )

                all_dfs.append(df)
                logger.info(
                    f"✓ Successfully processed {case_num} (SQLite): {len(df)} records"
                )
            except Exception as e:
                logger.error(f"✗ Failed to process {case_num} (SQLite): {e}")
                continue

        # (옵션) DB → CSV로 최종 데이터 한 번에 추출
        try:
            analyzer.export_db_to_csv(
                db_path="DB/tariff_data.db",
                table_name="tariff_data",
                output_csv="CSV/tariff_data.csv",
            )
        except Exception as e:
            logger.error(f"DB → CSV export 중 오류 발생 (main 요약 단계): {e}")

        # 요약 출력: DB에서 직접 읽거나, all_dfs로 계산
        try:
            # 여기서는 DB를 직접 읽어서 요약 (중복 제거 후 최종 상태 기준)
            with sqlite3.connect("DB/tariff_data.db") as conn:
                df_db = pd.read_sql_query("SELECT * FROM tariff_data", conn)

            print("\n" + "=" * 80)
            print("FINAL RESULTS (SQLite DB 기준)")
            print("=" * 80)
            print(f"\nTotal pairs processed: {len(pdf_pairs)}")
            print(f"Total records in DB: {len(df_db)}")
            print(f"\nUnique case numbers: {df_db['case_number'].nunique()}")
            print(f"Unique companies: {df_db['company'].nunique()}")
            print(f"Unique HS codes: {df_db['hs_code'].nunique()}")
            print("\nFirst 5 records:")
            print(df_db.head().to_string())
            print("\nCase number distribution:")
            print(df_db["case_number"].value_counts().to_string())
        except Exception as e:
            logger.error(f"요약 통계 출력 중 오류 발생: {e}")

    except Exception as e:
        logger.error(f"Error in main (SQLite 버전): {e}")
        raise


if __name__ == "__main__":
    main()



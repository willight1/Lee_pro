"""
Steel Tariff Analysis - LLM Integration Module
Handles PDF parsing, HS Code extraction, and OpenAI API integration
"""

import re
import json
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path

import pdfplumber
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TariffAnalyzer:
    """Main class for analyzing steel tariff PDF documents"""

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
        
        # HS Code patterns:
        # USA format: 7219.31.0000
        # Malaysia format: 7219.31.00 00 or 7219.31.00
        # EU TARIC format: 7225 11 00 11 (with spaces) or 72251100
        self.hs_code_pattern = re.compile(r'\b\d{4}\.\d{2}\.\d{2,4}(?:\s?\d{2})?\b')
        # EU TARIC code pattern (with or without spaces: "7225 11 00 11" or "72251100")
        self.eu_taric_pattern = re.compile(r'\b(722[56])\s*(\d{2})\s*(\d{2})\s*(\d{2})\b')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
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

        Args:
            text: Text to search for HS codes
            country: Country code to determine pattern (usa, malaysia, eu, etc.)

        Returns:
            Sorted list of unique HS codes (normalized to XXXX.XX.XXXX format)
        """
        try:
            logger.info("Extracting HS codes from text")
            normalized_codes: List[str] = []
            
            # Standard pattern (USA, Malaysia, Australia, generic HS with two dots)
            raw_codes = self.hs_code_pattern.findall(text)
            for code in raw_codes:
                # Remove spaces and normalize
                clean_code = code.replace(" ", "")
                parts = clean_code.split(".")
                if len(parts) == 3:
                    # Ensure last part is 4 digits (normalize to XXXX.XX.XXXX)
                    if len(parts[2]) == 2:
                        parts[2] = parts[2] + "00"
                    elif len(parts[2]) == 4:
                        pass  # Already correct
                    normalized_code = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    normalized_codes.append(normalized_code)

            # Pakistan PCT codes often appear as "7209.1510" (4 digits . 4 digits)
            # Convert "XXXX.YYYY" -> "XXXX.YY.ZZZZ" (YYYY[0:2] as subheading, YYYY[2:4] padded to 4)
            if country == "pakistan":
                pct_pattern = re.compile(r'\b\d{4}\.\d{4}\b')
                pct_codes = pct_pattern.findall(text)
                for code in pct_codes:
                    clean_code = code.replace(" ", "")
                    left, right = clean_code.split(".")
                    # right is 4 digits: AB CD -> AB as subheading, CD for national line
                    subheading = right[:2]
                    national = right[2:]
                    # Pad national part to 4 digits (e.g., "10" -> "1000", "99" -> "9900")
                    national_padded = national.ljust(4, "0")
                    normalized_code = f"{left}.{subheading}.{national_padded}"
                    normalized_codes.append(normalized_code)
            
            # EU TARIC pattern (with spaces: "7225 11 00 11" or without: "72251100")
            if country == "eu" or len(normalized_codes) == 0:
                eu_codes = self.eu_taric_pattern.findall(text)
                for match in eu_codes:
                    # match is tuple: ('7225', '11', '00', '11')
                    # Convert TARIC code to standard format: 7225.11.0011
                    part1 = match[0]  # 7225
                    part2 = match[1]  # 11
                    part3 = match[2] + match[3]  # 00 + 11 = 0011
                    normalized_code = f"{part1}.{part2}.{part3}"
                    normalized_codes.append(normalized_code)
            
            unique_codes = sorted(set(normalized_codes))
            logger.info(f"Found {len(unique_codes)} unique HS codes")
            return unique_codes
        except Exception as e:
            logger.error(f"Error extracting HS codes: {e}")
            return []

    def load_prompt_template(self, country: str = "usa") -> str:
        """
        Load system prompt template for a specific country

        Args:
            country: Country code (lowercase, e.g., "usa")

        Returns:
            Prompt template string
        """
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
        max_chunk_size: int = 50000  # 대략적인 문자 제한
    ) -> List[Dict]:
        """
        Extract structured data from text using OpenAI API

        Args:
            text: PDF text content
            country: Country code for prompt template
            max_retries: Maximum number of retry attempts
            max_chunk_size: Maximum text size per API call (characters)

        Returns:
            List of dictionaries containing extracted data
        """
        try:
            system_prompt = self.load_prompt_template(country)
            
            # 텍스트가 너무 길면 여러 청크로 나누기
            if len(text) > max_chunk_size:
                logger.warning(f"텍스트가 너무 깁니다 ({len(text)} chars). 여러 청크로 나누어 처리합니다.")
                chunks = self._split_text_into_chunks(text, max_chunk_size)
                all_results = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"청크 {i+1}/{len(chunks)} 처리 중...")
                    chunk_result = self._extract_from_chunk(chunk, system_prompt, max_retries)
                    if chunk_result:
                        all_results.extend(chunk_result)
                
                logger.info(f"총 {len(all_results)} 개의 레코드를 추출했습니다.")
                return all_results
            
            # 텍스트가 짧으면 일반 처리
            return self._extract_from_chunk(text, system_prompt, max_retries)
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """텍스트를 청크로 나눕니다"""
        chunks = []
        current_chunk = ""
        
        # 문단 단위로 나누기
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_from_chunk(
        self,
        text: str,
        system_prompt: str,
        max_retries: int
    ) -> List[Dict]:
        """단일 텍스트 청크에서 구조화된 데이터를 추출합니다"""
        try:

            for attempt in range(max_retries):
                try:
                    # Log text length for debugging
                    text_length = len(text)
                    prompt_length = len(system_prompt)
                    logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")
                    logger.info(f"Text length: {text_length} chars, Prompt length: {prompt_length} chars")
                    
                    # Check if text is too long (approximate token limit)
                    if text_length > 100000:  # Rough estimate
                        logger.warning(f"텍스트가 매우 깁니다 ({text_length} chars). API 호출이 실패할 수 있습니다.")

                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"},
                        timeout=120  # 120초 타임아웃 설정
                    )

                    content = response.choices[0].message.content
                    logger.info("Received response from OpenAI API")

                    # Parse JSON response
                    try:
                        data = json.loads(content)

                        # Handle different response formats
                        if isinstance(data, list):
                            result = data
                        elif isinstance(data, dict):
                            # Check for common keys that might contain the list
                            if "companies" in data:
                                result = data["companies"]
                            elif "data" in data:
                                result = data["data"]
                            elif "results" in data:
                                result = data["results"]
                            elif "result" in data:
                                result = data["result"] if isinstance(data["result"], list) else [data["result"]]
                            else:
                                # Assume the dict itself is a single result
                                result = [data]
                        else:
                            raise ValueError(f"Unexpected response format: {type(data)}")

                        logger.info(f"Successfully parsed {len(result)} records")
                        return result

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                        if attempt == max_retries - 1:
                            raise
                        continue

                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    logger.warning(f"API call error (attempt {attempt + 1}/{max_retries}): {error_type} - {error_msg}")
                    
                    # Provide more specific error messages
                    if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                        logger.error("API 키 인증 오류: .env 파일의 OPENAI_API_KEY를 확인해주세요.")
                    elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                        logger.error("API 할당량 초과: 잠시 후 다시 시도해주세요.")
                    elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                        logger.error("네트워크 연결 오류: 인터넷 연결을 확인해주세요.")
                    
                    if attempt == max_retries - 1:
                        raise Exception(f"API 호출 실패 (최대 재시도 횟수 초과): {error_type} - {error_msg}")
                    continue

            return []

        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
            raise

    def create_records(
        self,
        hs_codes: List[str],
        structured_data: List[Dict],
        country: str = "usa"
    ) -> pd.DataFrame:
        """
        Create DataFrame combining HS codes with structured data

        Args:
            hs_codes: List of HS codes
            structured_data: List of company/tariff data
            country: Country code for determining validation rules

        Returns:
            DataFrame with complete records
        """
        try:
            logger.info("Creating records DataFrame")

            records = []
            for hs_code in hs_codes:
                for company_data in structured_data:
                    # Determine tariff_type from case_number
                    case_number = company_data.get("case_number", "")

                    # Normalize case_number: replace em dash, en dash with regular hyphen
                    normalized_case = case_number.replace("–", "-").replace("—", "-")

                    # Validation based on country
                    if country == "usa":
                        # Filter: Only process 580 cases (skip 583, 891, etc.)
                        import re
                        if not re.match(r'^[AC]-580-\d{3}$', normalized_case):
                            logger.debug(f"Skipping non-580 case: {normalized_case}")
                            continue

                        # Determine tariff type based on first character
                        if normalized_case.startswith("A-"):
                            tariff_type = "Antidumping"
                        elif normalized_case.startswith("C-"):
                            tariff_type = "Countervailing"
                        else:
                            tariff_type = company_data.get("tariff_type", "Antidumping")
                        
                        case_number = normalized_case
                        issuing_country = "USA"
                    elif country == "malaysia":
                        # Malaysia uses P.U. (A) xxx format
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case  # Keep original format
                        issuing_country = "Malaysia"
                    elif country == "australia":
                        # Australia uses ADN YYYY/NNN or REP NNN format
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case  # Keep original format
                        issuing_country = "Australia"
                    elif country == "pakistan":
                        # Pakistan NTC anti-dumping cases (A.D.C No. XX/20XX/NTC/CRC)
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case
                        issuing_country = "Pakistan"
                    elif country == "eu":
                        # EU uses EU YYYY/XX or Regulation number format
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        case_number = normalized_case  # Keep original format (e.g., EU 2022/58)
                        issuing_country = "EU"
                    else:
                        # Default handling
                        tariff_type = company_data.get("tariff_type", "Antidumping")
                        issuing_country = company_data.get("issuing_country", country.upper())

                    record = {
                        "hs_code": hs_code,
                        "issuing_country": company_data.get("issuing_country", issuing_country),
                        "country": company_data.get("country", ""),
                        "tariff_type": tariff_type,
                        "tariff_rate": company_data.get("tariff_rate", ""),
                        "effective_date_from": company_data.get("effective_date_from", ""),
                        "effective_date_to": company_data.get("effective_date_to", ""),
                        "investigation_period_from": company_data.get("investigation_period_from", ""),
                        "investigation_period_to": company_data.get("investigation_period_to", ""),
                        "company": company_data.get("company", ""),
                        "case_number": case_number,
                        "product_description": company_data.get("product_description", ""),
                        "note": company_data.get("note", "")
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
        append: bool = True
    ):
        """
        Save DataFrame to CSV file

        Args:
            df: DataFrame to save
            output_path: Output CSV file path
            append: If True, append to existing file; if False, overwrite
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if append and output_file.exists():
                logger.info(f"Appending to existing CSV: {output_path}")
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Remove duplicates based on all columns
                combined_df = combined_df.drop_duplicates()
                combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"Appended {len(df)} records (total: {len(combined_df)} records)")
            else:
                logger.info(f"Writing new CSV: {output_path}")
                df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved {len(df)} records")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise

    def process_pdfs(
        self,
        hs_code_pdf: str,
        detail_pdf: str,
        output_csv: str = "CSV/tariff_data.csv",
        country: str = "usa",
        save_csv: bool = True,
        append: bool = True
    ) -> pd.DataFrame:
        """
        Process two PDF files and generate structured CSV output

        Args:
            hs_code_pdf: Path to PDF containing HS codes
            detail_pdf: Path to PDF containing detailed information
            output_csv: Output CSV file path
            country: Country code for prompt template
            save_csv: Whether to save to CSV (default True)
            append: Whether to append to existing CSV (default True)

        Returns:
            DataFrame with extracted data
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting PDF processing")
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

            # Step 5: Save to CSV (optional)
            if save_csv:
                logger.info("Step 5: Saving to CSV")
                self.save_to_csv(df, output_csv, append=append)

            logger.info("=" * 80)
            logger.info("PDF processing completed successfully")
            logger.info("=" * 80)

            return df

        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            raise


def detect_country_from_filename(filename: str) -> str:
    """
    Detect country code from PDF filename
    
    Args:
        filename: PDF filename
        
    Returns:
        Country code (lowercase): "usa", "malaysia", "australia", "eu", etc.
    """
    filename_upper = filename.upper()
    
    if "MALAYSIA" in filename_upper:
        return "malaysia"
    elif "AUSTRALIA" in filename_upper:
        return "australia"
    elif "PAKISTAN" in filename_upper:
        return "pakistan"
    elif "EU" in filename_upper or "COMMISSION" in filename_upper or re.search(r'EU[_\s]*\d{4}[/_]\d+', filename_upper):
        return "eu"
    elif "USA" in filename_upper or re.match(r'.*[AC]-580-\d{3}.*', filename_upper):
        return "usa"
    else:
        # Default to usa for backwards compatibility
        return "usa"


def find_pdf_pairs(pdf_folder: str = "PDF") -> List[tuple]:
    """
    Automatically find and pair HS Code PDFs with Detail PDFs based on case number

    Args:
        pdf_folder: Path to folder containing PDFs

    Returns:
        List of tuples: (hs_code_pdf, detail_pdf, case_number, country)
    """
    import re

    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        logger.error(f"PDF folder not found: {pdf_folder}")
        return []

    # Get all PDF files
    all_pdfs = sorted(pdf_path.glob("*.pdf"))
    
    # Separate USA, Malaysia, Australia, Pakistan, and EU PDFs
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

    # Pattern to extract case number (A-580-XXX or C-580-XXX only)
    case_pattern = re.compile(r'([AC]-580-\d{3})')

    # Group USA PDFs by case number
    case_groups = {}
    for pdf_file in usa_pdfs:
        match = case_pattern.search(pdf_file.name)
        if match:
            case_num = match.group(1)
            if case_num not in case_groups:
                case_groups[case_num] = []
            case_groups[case_num].append(pdf_file)

    # Pair HS Code and Detail files for USA
    pairs = []
    for case_num, files in case_groups.items():
        # Files with year indicators are HS Code files
        hs_code_files = [f for f in files if re.search(r'_\d{4}\.pdf$|_F_\d{4}\.pdf$|_Pre_\d{4}\.pdf$|_연장_\d{4}\.pdf$', f.name)]
        # Others are Detail files
        detail_files = [f for f in files if f not in hs_code_files]

        # Match pairs (prefer most recent HS Code file with corresponding Detail file)
        if hs_code_files and detail_files:
            # Use the first HS Code file and first Detail file for each case
            hs_code_pdf = str(hs_code_files[0])
            detail_pdf = str(detail_files[0])
            pairs.append((hs_code_pdf, detail_pdf, case_num, "usa"))
            logger.info(f"[USA] Paired: {hs_code_files[0].name} <-> {detail_files[0].name}")
        elif hs_code_files and not detail_files:
            # No detail file, select file with most HS codes
            import pdfplumber
            hs_code_pattern = re.compile(r'\b\d{4}\.\d{2}\.\d{4}\b')

            best_file = None
            max_hs_codes = 0

            for pdf_file in hs_code_files:
                try:
                    with pdfplumber.open(str(pdf_file)) as pdf:
                        text = ''
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text

                        hs_codes = hs_code_pattern.findall(text)
                        unique_count = len(set(hs_codes))

                        if unique_count > max_hs_codes:
                            max_hs_codes = unique_count
                            best_file = pdf_file

                        logger.info(f"  {pdf_file.name}: {unique_count} unique HS codes")
                except Exception as e:
                    logger.warning(f"  Error reading {pdf_file.name}: {e}")
                    continue

            if best_file and max_hs_codes > 0:
                selected_file = str(best_file)
                pairs.append((selected_file, selected_file, case_num, "usa"))
                logger.info(f"[USA] Selected (contains both HS and Detail): {best_file.name} ({max_hs_codes} HS codes)")
            else:
                logger.warning(f"[USA] No valid file found for case {case_num}: all files have 0 HS codes")
        else:
            logger.warning(f"[USA] Could not pair case {case_num}: HS={len(hs_code_files)}, Detail={len(detail_files)}")

    # Process Malaysia PDFs (each file is self-contained)
    for pdf_file in malaysia_pdfs:
        pdf_path_str = str(pdf_file)
        # Extract P.U. (A) number from filename for case_number
        pu_match = re.search(r'PU\s*\(?A\)?\s*(\d+)|P\.U\.\s*\(A\)\s*(\d+)', pdf_file.name, re.IGNORECASE)
        if pu_match:
            pu_num = pu_match.group(1) or pu_match.group(2)
            case_num = f"PUA{pu_num}"
        else:
            case_num = pdf_file.stem  # Use filename without extension
        
        pairs.append((pdf_path_str, pdf_path_str, case_num, "malaysia"))
        logger.info(f"[Malaysia] Added: {pdf_file.name} (case: {case_num})")

    # Process Australia PDFs (each file is self-contained)
    for pdf_file in australia_pdfs:
        pdf_path_str = str(pdf_file)
        # Extract ADN number from filename for case_number (e.g., ADN_2023_035)
        adn_match = re.search(r'ADN[_\s]*(\d{4})[_\s]*(\d+)', pdf_file.name, re.IGNORECASE)
        if adn_match:
            year = adn_match.group(1)
            num = adn_match.group(2)
            case_num = f"ADN {year}/{num}"
        else:
            # Try REP number (e.g., REP 611)
            rep_match = re.search(r'REP[_\s]*(\d+)', pdf_file.name, re.IGNORECASE)
            if rep_match:
                case_num = f"REP {rep_match.group(1)}"
            else:
                case_num = pdf_file.stem  # Use filename without extension
        
        pairs.append((pdf_path_str, pdf_path_str, case_num, "australia"))
        logger.info(f"[Australia] Added: {pdf_file.name} (case: {case_num})")

    # Process EU PDFs (each file is self-contained)
    for pdf_file in eu_pdfs:
        pdf_path_str = str(pdf_file)
        # Extract EU regulation number from filename (e.g., EU_2022_58, EU 2022/58)
        eu_match = re.search(r'EU[_\s]*(\d{4})[/_\s]*(\d+)', pdf_file.name, re.IGNORECASE)
        if eu_match:
            year = eu_match.group(1)
            num = eu_match.group(2)
            case_num = f"EU {year}/{num}"
        else:
            # Try to extract from AD/R format (e.g., AD608_R728)
            ad_match = re.search(r'AD(\d+)[_\s]*R(\d+)', pdf_file.name, re.IGNORECASE)
            if ad_match:
                ad_num = ad_match.group(1)
                r_num = ad_match.group(2)
                case_num = f"AD{ad_num}/R{r_num}"
            else:
                case_num = pdf_file.stem  # Use filename without extension
        
        pairs.append((pdf_path_str, pdf_path_str, case_num, "eu"))
        logger.info(f"[EU] Added: {pdf_file.name} (case: {case_num})")

    # Process Pakistan PDFs (each file is self-contained)
    for pdf_file in pakistan_pdfs:
        pdf_path_str = str(pdf_file)
        # Try to extract A.D.C number from filename for case_number (e.g., PAKISTAN_CR_Antidumping_A.D.C_No._60.pdf)
        adc_match = re.search(r'A\.?D\.?C[_.\s-]*No[_.\s-]*_?(\d+)', pdf_file.name, re.IGNORECASE)
        if adc_match:
            adc_num = adc_match.group(1)
            case_num = f"ADC {adc_num}"
        else:
            case_num = pdf_file.stem  # Use filename without extension

        pairs.append((pdf_path_str, pdf_path_str, case_num, "pakistan"))
        logger.info(f"[Pakistan] Added: {pdf_file.name} (case: {case_num})")

    return pairs


def main():
    """Main function for testing - processes all PDF pairs in folder"""
    try:
        # Initialize analyzer
        analyzer = TariffAnalyzer()

        # Find all PDF pairs
        logger.info("Scanning PDF folder for file pairs...")
        pdf_pairs = find_pdf_pairs("PDF")

        if not pdf_pairs:
            logger.error("No PDF pairs found in PDF folder")
            return

        logger.info(f"Found {len(pdf_pairs)} PDF pair(s) to process")

        # Delete existing CSV to start fresh
        csv_path = Path("CSV/tariff_data.csv")
        if csv_path.exists():
            csv_path.unlink()
            logger.info("Deleted existing CSV file")

        # Process each pair
        all_dfs = []
        for idx, (hs_code_pdf, detail_pdf, case_num, country) in enumerate(pdf_pairs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing pair {idx}/{len(pdf_pairs)}: {case_num} ({country.upper()})")
            logger.info(f"{'='*80}")

            try:
                # Process PDFs without saving
                df = analyzer.process_pdfs(
                    hs_code_pdf=hs_code_pdf,
                    detail_pdf=detail_pdf,
                    country=country,
                    save_csv=False
                )

                all_dfs.append(df)
                logger.info(f"✓ Successfully processed {case_num}: {len(df)} records")

            except Exception as e:
                logger.error(f"✗ Failed to process {case_num}: {e}")
                continue

        # Save all data at once (overwrite mode)
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv("CSV/tariff_data.csv", index=False, encoding="utf-8-sig")
            logger.info(f"\n✓ Saved all data to CSV: {len(combined_df)} total records")

        # Display summary
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)

            print("\n" + "=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)
            print(f"\nTotal pairs processed: {len(all_dfs)}")
            print(f"Total records: {len(combined_df)}")
            print(f"\nUnique case numbers: {combined_df['case_number'].nunique()}")
            print(f"Unique companies: {combined_df['company'].nunique()}")
            print(f"Unique HS codes: {combined_df['hs_code'].nunique()}")
            print("\nFirst 5 records:")
            print(combined_df.head().to_string())
            print("\nCase number distribution:")
            print(combined_df['case_number'].value_counts().to_string())

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()

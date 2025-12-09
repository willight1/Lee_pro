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

        self.client = OpenAI(api_key=self.api_key)
        self.hs_code_pattern = re.compile(r'\b\d{4}\.\d{2}\.\d{4}\b')

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

    def extract_hs_codes(self, text: str) -> List[str]:
        """
        Extract HS Codes from text using regex pattern

        Args:
            text: Text to search for HS codes

        Returns:
            Sorted list of unique HS codes
        """
        try:
            logger.info("Extracting HS codes from text")
            hs_codes = self.hs_code_pattern.findall(text)
            unique_codes = sorted(set(hs_codes))
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
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Extract structured data from text using OpenAI API

        Args:
            text: PDF text content
            country: Country code for prompt template
            max_retries: Maximum number of retry attempts

        Returns:
            List of dictionaries containing extracted data
        """
        try:
            system_prompt = self.load_prompt_template(country)

            for attempt in range(max_retries):
                try:
                    logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")

                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"}
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
                    logger.warning(f"API call error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    continue

            return []

        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            raise

    def create_records(
        self,
        hs_codes: List[str],
        structured_data: List[Dict]
    ) -> pd.DataFrame:
        """
        Create DataFrame combining HS codes with structured data

        Args:
            hs_codes: List of HS codes
            structured_data: List of company/tariff data

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
                        # Fallback to LLM-provided value
                        tariff_type = company_data.get("tariff_type", "Antidumping")

                    # Also normalize the case_number for storage
                    case_number = normalized_case

                    record = {
                        "hs_code": hs_code,
                        "issuing_country": company_data.get("issuing_country", "USA"),
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
            hs_codes = self.extract_hs_codes(hs_code_text)

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
            df = self.create_records(hs_codes, structured_data)

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


def find_pdf_pairs(pdf_folder: str = "PDF") -> List[tuple]:
    """
    Automatically find and pair HS Code PDFs with Detail PDFs based on case number

    Args:
        pdf_folder: Path to folder containing PDFs

    Returns:
        List of tuples: (hs_code_pdf, detail_pdf, case_number)
    """
    import re

    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        logger.error(f"PDF folder not found: {pdf_folder}")
        return []

    # Get all PDF files
    all_pdfs = sorted(pdf_path.glob("*.pdf"))

    # Pattern to extract case number (A-580-XXX or C-580-XXX only)
    case_pattern = re.compile(r'([AC]-580-\d{3})')

    # Group PDFs by case number
    case_groups = {}
    for pdf_file in all_pdfs:
        match = case_pattern.search(pdf_file.name)
        if match:
            case_num = match.group(1)
            if case_num not in case_groups:
                case_groups[case_num] = []
            case_groups[case_num].append(pdf_file)

    # Pair HS Code and Detail files
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
            pairs.append((hs_code_pdf, detail_pdf, case_num))
            logger.info(f"Paired: {hs_code_files[0].name} <-> {detail_files[0].name}")
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
                pairs.append((selected_file, selected_file, case_num))
                logger.info(f"Selected (contains both HS and Detail): {best_file.name} ({max_hs_codes} HS codes)")
            else:
                logger.warning(f"No valid file found for case {case_num}: all files have 0 HS codes")
        else:
            logger.warning(f"Could not pair case {case_num}: HS={len(hs_code_files)}, Detail={len(detail_files)}")

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
        for idx, (hs_code_pdf, detail_pdf, case_num) in enumerate(pdf_pairs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing pair {idx}/{len(pdf_pairs)}: {case_num}")
            logger.info(f"{'='*80}")

            try:
                # Process PDFs without saving
                df = analyzer.process_pdfs(
                    hs_code_pdf=hs_code_pdf,
                    detail_pdf=detail_pdf,
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

"""
Steel Tariff Analysis - Streamlit Web Application
Web interface for PDF upload, data extraction, and visualization
"""

import os
import sys
from pathlib import Path
from typing import Optional
import tempfile

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from llm import TariffAnalyzer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Steel Tariff Analysis System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_api_key() -> bool:
    """Check if OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key != "your_openai_api_key_here"


def load_existing_csv(csv_path: str = "CSV/tariff_data.csv") -> Optional[pd.DataFrame]:
    """Load existing CSV file if it exists"""
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def save_uploaded_file(uploaded_file, directory: str) -> str:
    """Save uploaded file to temporary location"""
    try:
        temp_dir = Path(directory)
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        raise


def main():
    """Main Streamlit application"""

    # Header
    st.title("🏭 Steel Tariff Analysis System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Select Page",
            ["PDF Processing", "Conversational Search", "Data Visualization"],
            index=1
        )

        st.markdown("---")
        st.header("⚙️ Configuration")

        # Check API key
        if check_api_key():
            st.success("✓ OpenAI API Key configured")
        else:
            st.error("✗ OpenAI API Key not configured")
            st.info("Please set OPENAI_API_KEY in .env file")

        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **Steel Tariff Analysis System**

        Extract and analyze steel product
        tariff data from PDF documents.

        **Features:**
        - PDF parsing
        - HS Code extraction
        - LLM-based data structuring
        - Conversational search
        - CSV export
        - Data visualization
        """)

    # Main content
    if page == "PDF Processing":
        show_pdf_processing_page()
    elif page == "Conversational Search":
        show_conversational_search_page()
    else:
        show_data_visualization_page()


def show_conversational_search_page():
    """Show conversational search page with step-by-step filtering"""

    st.header("💬 Conversational Tariff Search")
    st.markdown("대화형으로 관세 정보를 검색합니다. 단계별로 필터링하여 원하는 정보를 찾아보세요.")

    # Load CSV data
    csv_path = "CSV/tariff_data.csv"
    df = load_existing_csv(csv_path)

    if df is None or df.empty:
        st.warning("📂 데이터가 없습니다. PDF Processing 페이지에서 먼저 데이터를 추출해주세요.")
        return

    st.info(f"💾 총 **{len(df)}** 개의 레코드가 로드되었습니다.")
    st.markdown("---")

    # Initialize session state for conversation flow
    if 'search_step' not in st.session_state:
        st.session_state.search_step = 1
        st.session_state.selected_issuing = None
        st.session_state.selected_target = None
        st.session_state.filtered_df = df

    # Step 1: Select issuing country (관세 부과 국가)
    st.subheader("🌍 Step 1: 관세를 부과한 국가를 선택하세요")

    issuing_countries = sorted(df["issuing_country"].unique().tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_issuing = st.selectbox(
            "관세 부과 국가 (Issuing Country)",
            issuing_countries,
            key="issuing_country_select"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("다음 단계로 →", type="primary", key="step1_next"):
            st.session_state.selected_issuing = selected_issuing
            st.session_state.search_step = 2
            st.rerun()

    if st.session_state.search_step >= 2 and st.session_state.selected_issuing:
        st.success(f"✓ 선택됨: **{st.session_state.selected_issuing}**")

        # Filter by issuing country
        filtered_by_issuing = df[df["issuing_country"] == st.session_state.selected_issuing]

        st.markdown("---")

        # Step 2: Select target country (관세 대상 국가)
        st.subheader("🎯 Step 2: 관세 대상 국가를 선택하세요")

        target_countries = sorted(filtered_by_issuing["country"].unique().tolist())

        if len(target_countries) == 0:
            st.warning("해당 관세 부과 국가에 대한 데이터가 없습니다.")
            if st.button("← 처음으로 돌아가기", key="reset_step2"):
                st.session_state.search_step = 1
                st.session_state.selected_issuing = None
                st.rerun()
            return

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_target = st.selectbox(
                "관세 대상 국가 (Target Country)",
                target_countries,
                key="target_country_select"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("다음 단계로 →", type="primary", key="step2_next"):
                st.session_state.selected_target = selected_target
                st.session_state.search_step = 3
                st.rerun()

        if st.button("← 이전 단계로", key="back_step2"):
            st.session_state.search_step = 1
            st.session_state.selected_issuing = None
            st.rerun()

    if st.session_state.search_step >= 3 and st.session_state.selected_target:
        st.success(f"✓ 선택됨: **{st.session_state.selected_target}**")

        # Filter by both countries
        filtered_by_both = df[
            (df["issuing_country"] == st.session_state.selected_issuing) &
            (df["country"] == st.session_state.selected_target)
        ]

        st.markdown("---")

        # Step 3: Enter HS Code
        st.subheader("🔢 Step 3: HS Code를 입력하세요")

        available_hs_codes = sorted(filtered_by_both["hs_code"].unique().tolist())

        if len(available_hs_codes) == 0:
            st.warning("해당 조건에 맞는 HS Code가 없습니다.")
            if st.button("← 처음으로 돌아가기", key="reset_step3"):
                st.session_state.search_step = 1
                st.session_state.selected_issuing = None
                st.session_state.selected_target = None
                st.rerun()
            return

        st.info(f"💡 이용 가능한 HS Code: {len(available_hs_codes)}개")

        with st.expander("📋 이용 가능한 HS Code 목록 보기"):
            st.write(available_hs_codes)

        col1, col2 = st.columns([3, 1])
        with col1:
            hs_code_input = st.text_input(
                "HS Code 입력 (예: 7209.15.0000)",
                placeholder="HS Code를 입력하세요",
                key="hs_code_input"
            )

            # Also provide selectbox as alternative
            st.markdown("**또는 선택:**")
            selected_hs_code = st.selectbox(
                "HS Code 선택",
                ["선택하세요..."] + available_hs_codes,
                key="hs_code_select"
            )

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🔍 검색", type="primary", key="search_button", use_container_width=True):
                # Use input or selectbox
                final_hs_code = hs_code_input if hs_code_input else (selected_hs_code if selected_hs_code != "선택하세요..." else None)

                if not final_hs_code:
                    st.error("HS Code를 입력하거나 선택해주세요.")
                else:
                    # Search for the HS Code
                    result_df = filtered_by_both[filtered_by_both["hs_code"] == final_hs_code]

                    if result_df.empty:
                        st.error(f"❌ HS Code '{final_hs_code}'에 대한 정보를 찾을 수 없습니다.")
                    else:
                        st.markdown("---")
                        st.success(f"✅ **{len(result_df)}** 건의 정보를 찾았습니다!")

                        # Display results
                        st.subheader(f"📊 검색 결과: {final_hs_code}")

                        # Show summary cards
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("관세 부과국", st.session_state.selected_issuing)
                        with col2:
                            st.metric("대상 국가", st.session_state.selected_target)
                        with col3:
                            st.metric("회사 수", result_df["company"].nunique())

                        st.markdown("---")

                        # Display detailed information for each record
                        for idx, row in result_df.iterrows():
                            with st.container():
                                st.markdown(f"""
                                ### 📋 레코드 {idx + 1}

                                **기본 정보:**
                                - **HS Code**: `{row['hs_code']}`
                                - **회사명**: {row['company']}
                                - **제품**: {row['product_description']}

                                **관세 정보:**
                                - **관세 유형**: {row['tariff_type']}
                                - **관세율**: {row['tariff_rate']}%
                                - **케이스 번호**: {row['case_number']}

                                **기간 정보:**
                                - **발효일**: {row['effective_date_from']} ~ {row['effective_date_to'] if row['effective_date_to'] else '진행 중'}
                                - **조사 기간**: {row['investigation_period_from']} ~ {row['investigation_period_to']}

                                **비고:**
                                {row['note']}
                                """)
                                st.markdown("---")

                        # Download option
                        csv = result_df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button(
                            label="📥 검색 결과 다운로드 (CSV)",
                            data=csv,
                            file_name=f"tariff_result_{final_hs_code.replace('.', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        if st.button("← 처음으로 돌아가기", key="reset_all"):
            st.session_state.search_step = 1
            st.session_state.selected_issuing = None
            st.session_state.selected_target = None
            st.rerun()


def show_pdf_processing_page():
    """Show PDF processing and extraction page"""

    st.header("📄 PDF Processing & Data Extraction")

    if not check_api_key():
        st.warning("⚠️ Please configure OpenAI API key in .env file to use this feature")
        st.code("""
# Create .env file in project root:
OPENAI_API_KEY=your_actual_api_key_here
        """)
        return

    st.markdown("""
    Upload two PDF files:
    1. **HS Code PDF**: Contains HS code listings (e.g., USA_CR_Antidumping_A-580-881_2016.pdf)
    2. **Detail PDF**: Contains detailed tariff information (e.g., USA_CR_Antidumping_A-580-881.pdf)
    """)

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ HS Code PDF")
        hs_code_file = st.file_uploader(
            "Upload HS Code PDF",
            type=["pdf"],
            key="hs_code_pdf",
            help="PDF containing HS code listings"
        )

    with col2:
        st.subheader("2️⃣ Detail PDF")
        detail_file = st.file_uploader(
            "Upload Detail PDF",
            type=["pdf"],
            key="detail_pdf",
            help="PDF containing detailed tariff information"
        )

    # Country selection
    st.subheader("3️⃣ Select Country")
    country = st.selectbox(
        "Country",
        ["USA"],
        index=0,
        help="Select the issuing country for prompt template"
    )

    # Process button
    st.markdown("---")

    if st.button("🚀 Extract Data", type="primary", use_container_width=True):
        if not hs_code_file or not detail_file:
            st.error("❌ Please upload both PDF files")
            return

        try:
            with st.spinner("Processing PDFs... This may take a few minutes."):
                # Save uploaded files
                temp_dir = Path("temp_uploads")
                hs_code_path = save_uploaded_file(hs_code_file, str(temp_dir))
                detail_path = save_uploaded_file(detail_file, str(temp_dir))

                # Initialize analyzer
                analyzer = TariffAnalyzer()

                # Create progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Extract HS codes
                status_text.text("Step 1/5: Extracting HS codes...")
                progress_bar.progress(20)
                hs_code_text = analyzer.extract_text_from_pdf(hs_code_path)
                hs_codes = analyzer.extract_hs_codes(hs_code_text)

                # Step 2: Extract detail text
                status_text.text("Step 2/5: Extracting detail text...")
                progress_bar.progress(40)
                detail_text = analyzer.extract_text_from_pdf(detail_path)

                # Step 3: LLM extraction
                status_text.text("Step 3/5: Analyzing with AI (this may take a minute)...")
                progress_bar.progress(60)
                structured_data = analyzer.extract_structured_data(
                    detail_text,
                    country=country.lower()
                )

                # Step 4: Create records
                status_text.text("Step 4/5: Creating records...")
                progress_bar.progress(80)
                df = analyzer.create_records(hs_codes, structured_data)

                # Step 5: Save to CSV (overwrite mode to prevent duplicates)
                status_text.text("Step 5/5: Saving to CSV...")
                progress_bar.progress(90)
                analyzer.save_to_csv(df, "CSV/tariff_data.csv", append=False)

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Cleanup temp files
                try:
                    os.remove(hs_code_path)
                    os.remove(detail_path)
                except:
                    pass

                # Display results
                st.success(f"✅ Successfully extracted {len(df)} records!")

                st.subheader("📊 Extracted Data Preview")
                st.dataframe(df, use_container_width=True, height=400)

                # Download button
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Download Extracted Data as CSV",
                    data=csv,
                    file_name="extracted_tariff_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Unique HS Codes", df["hs_code"].nunique())
                with col3:
                    st.metric("Companies", df["company"].nunique())
                with col4:
                    st.metric("Case Numbers", df["case_number"].nunique())

        except Exception as e:
            st.error(f"❌ Error processing PDFs: {str(e)}")
            st.exception(e)


def show_data_visualization_page():
    """Show data visualization page for existing CSV files"""

    st.header("📊 Data Visualization")

    # Load CSV from CSV folder
    csv_path = "CSV/tariff_data.csv"
    df = load_existing_csv(csv_path)

    if df is None or df.empty:
        st.info("📂 No data available. Please process PDFs first.")
        st.markdown("""
        **To get started:**
        1. Go to "PDF Processing" page
        2. Upload your PDF files
        3. Extract data
        4. Return here to view the results
        """)
        return

    # Display dataset info
    st.subheader(f"📁 Dataset: {csv_path}")
    st.info(f"Total records: **{len(df)}** | Last updated: **{Path(csv_path).stat().st_mtime}**")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📈 Statistics", "🔍 Filter", "💾 Export"])

    with tab1:
        st.subheader("Complete Dataset")

        # Display full dataframe
        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )

        # Show data types
        with st.expander("ℹ️ Column Information"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-Null Count": df.count().values,
                "Null Count": df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)

    with tab2:
        st.subheader("Dataset Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(df))
            st.metric("Unique HS Codes", df["hs_code"].nunique())

        with col2:
            st.metric("Issuing Countries", df["issuing_country"].nunique())
            st.metric("Target Countries", df["country"].nunique())

        with col3:
            st.metric("Companies", df["company"].nunique())
            st.metric("Case Numbers", df["case_number"].nunique())

        with col4:
            st.metric("Tariff Types", df["tariff_type"].nunique())
            avg_rate = pd.to_numeric(df["tariff_rate"], errors='coerce').mean()
            st.metric("Avg Tariff Rate", f"{avg_rate:.2f}%")

        # Group statistics
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Records by Company")
            company_counts = df["company"].value_counts().head(10)
            st.bar_chart(company_counts)

        with col2:
            st.subheader("Records by HS Code")
            hs_counts = df["hs_code"].value_counts().head(10)
            st.bar_chart(hs_counts)

        # Tariff rate distribution
        st.subheader("Tariff Rate Distribution")
        numeric_rates = pd.to_numeric(df["tariff_rate"], errors='coerce').dropna()
        if len(numeric_rates) > 0:
            st.bar_chart(numeric_rates.value_counts().sort_index())
        else:
            st.info("No numeric tariff rates available")

    with tab3:
        st.subheader("Filter Data")

        # Filter options
        col1, col2 = st.columns(2)

        with col1:
            # Country filter
            countries = ["All"] + sorted(df["country"].unique().tolist())
            selected_country = st.selectbox("Filter by Country", countries)

            # Company filter
            companies = ["All"] + sorted(df["company"].unique().tolist())
            selected_company = st.selectbox("Filter by Company", companies)

        with col2:
            # Case number filter
            case_numbers = ["All"] + sorted(df["case_number"].unique().tolist())
            selected_case = st.selectbox("Filter by Case Number", case_numbers)

            # HS Code filter
            hs_codes = ["All"] + sorted(df["hs_code"].unique().tolist())
            selected_hs = st.selectbox("Filter by HS Code", hs_codes)

        # Apply filters
        filtered_df = df.copy()

        if selected_country != "All":
            filtered_df = filtered_df[filtered_df["country"] == selected_country]

        if selected_company != "All":
            filtered_df = filtered_df[filtered_df["company"] == selected_company]

        if selected_case != "All":
            filtered_df = filtered_df[filtered_df["case_number"] == selected_case]

        if selected_hs != "All":
            filtered_df = filtered_df[filtered_df["hs_code"] == selected_hs]

        # Display filtered results
        st.markdown("---")
        st.info(f"Showing **{len(filtered_df)}** of **{len(df)}** records")

        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download filtered data
        csv = filtered_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_tariff_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab4:
        st.subheader("Export Options")

        st.markdown("**Available Export Formats:**")

        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="tariff_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Excel export
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Tariff Data')
            excel_data = output.getvalue()

            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name="tariff_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Data summary
        st.markdown("---")
        st.subheader("Data Summary")

        summary_text = f"""
        **Dataset Information:**
        - Total Records: {len(df)}
        - Columns: {len(df.columns)}
        - File Size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB

        **Coverage:**
        - HS Codes: {df['hs_code'].nunique()}
        - Companies: {df['company'].nunique()}
        - Case Numbers: {df['case_number'].nunique()}
        - Countries: {df['country'].nunique()}
        """

        st.markdown(summary_text)


if __name__ == "__main__":
    main()

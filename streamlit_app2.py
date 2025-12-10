"""
Steel Tariff Dashboard - 반덤핑/상계관세 조회 전용 Streamlit 앱

tariff_data.csv를 기반으로 수입국 / 생산국 / HS Code 필터링 및 상세 조회를 제공합니다.
"""

import re
from datetime import date
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Steel Tariff Dashboard",
    page_icon="🛃",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_tariff_data(
    csv_path: str = "CSV/tariff_data.csv",
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    반덤핑/상계관세 대시보드용 CSV 로더

    - CSV를 읽고
    - 문자열 컬럼의 앞뒤 공백을 제거하되 원래 결측은 보존하고
    - (DataFrame, 누락된_필수_컬럼_리스트)를 반환합니다.
    """
    required_columns = [
        "hs_code",
        "issuing_country",
        "country",
        "tariff_type",
        "tariff_rate",
        "effective_date_from",
        "effective_date_to",
        "investigation_period_from",
        "investigation_period_to",
        "company",
        "case_number",
        "product_description",
        "note",
    ]

    path = Path(csv_path)
    if not path.exists():
        return None, required_columns

    try:
        df = pd.read_csv(path)

        if not df.empty:
            obj_cols = df.select_dtypes(include=["object"]).columns
            for col in obj_cols:
                col_series = df[col]
                not_null = col_series.notna()
                # 결측은 그대로 두고, 값이 있는 것만 문자열 변환 + strip
                cleaned = col_series[not_null].astype(str).str.strip()
                # 빈 문자열이나 'nan' 류 표현은 결측으로 처리
                cleaned = cleaned.replace(
                    {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "N/A": pd.NA, "NA": pd.NA}
                )
                df.loc[not_null, col] = cleaned

        missing_cols = [c for c in required_columns if c not in df.columns]
        return df, missing_cols
    except Exception as e:
        st.error(f"CSV 로드 중 오류가 발생했습니다: {e}")
        return None, required_columns


def normalize_hs_digits(value: object) -> str:
    """hs_code에서 숫자만 추출하여 digits-only 문자열로 반환"""
    if pd.isna(value):
        return ""
    return re.sub(r"\D", "", str(value))


def compute_status_column(df: pd.DataFrame, as_of: date) -> pd.Series:
    """effective_date_from/to 기준으로 상태(status) 컬럼 계산"""
    as_of_ts = pd.Timestamp(as_of)

    from_dt = pd.to_datetime(df["effective_date_from"], errors="coerce")
    to_dt = pd.to_datetime(df["effective_date_to"], errors="coerce")

    status_values: List[str] = []
    for f, t in zip(from_dt, to_dt):
        if pd.isna(f) and pd.isna(t):
            status_values.append("기간불명")
            continue

        if not pd.isna(f) and as_of_ts < f:
            status_values.append("예정")
            continue

        if not pd.isna(t) and as_of_ts > t:
            status_values.append("만료")
            continue

        # 유효 조건들
        if not pd.isna(f) and not pd.isna(t) and f <= as_of_ts <= t:
            status_values.append("유효")
        elif not pd.isna(f) and pd.isna(t) and as_of_ts >= f:
            status_values.append("유효")
        elif pd.isna(f) and not pd.isna(t) and as_of_ts <= t:
            status_values.append("유효")
        else:
            status_values.append("기간불명")

    return pd.Series(status_values, index=df.index, name="status")


def compute_data_quality(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    데이터 품질 플래그/사유 계산

    - data_quality: "정상" 또는 "검토 필요"
    - quality_reason: 사유를 ;로 join
    """
    required_fields = ["issuing_country", "country", "hs_code", "tariff_type", "tariff_rate"]

    qualities: List[str] = []
    reasons_list: List[str] = []

    from_dt = pd.to_datetime(df["effective_date_from"], errors="coerce")
    to_dt = pd.to_datetime(df["effective_date_to"], errors="coerce")

    for idx, row in df.iterrows():
        reasons: List[str] = []

        # 필수 컬럼 누락 체크
        for col in required_fields:
            val = row.get(col)
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                reasons.append(f"필수컬럼누락:{col}")

        # 기간 불명: from/to 모두 NaT
        if pd.isna(from_dt.loc[idx]) and pd.isna(to_dt.loc[idx]):
            reasons.append("기간불명")

        # 관세율 파싱 실패
        rate = row.get("tariff_rate")
        rate_str = "" if pd.isna(rate) else str(rate).strip()
        m = re.search(r"(\d+(\.\d+)?)", rate_str)
        if not m:
            reasons.append("관세율파싱실패")

        if reasons:
            qualities.append("검토 필요")
            reasons_list.append(";".join(reasons))
        else:
            qualities.append("정상")
            reasons_list.append("")

    return (
        pd.Series(qualities, index=df.index, name="data_quality"),
        pd.Series(reasons_list, index=df.index, name="quality_reason"),
    )


def status_style(val: str) -> str:
    """status 컬럼용 스타일"""
    colors = {
        "유효": "#d4edda",
        "만료": "#f8d7da",
        "예정": "#fff3cd",
        "기간불명": "#e2e3e5",
    }
    color = colors.get(val, "")
    return f"background-color: {color}" if color else ""


def quality_style(val: str) -> str:
    """data_quality 컬럼용 스타일"""
    if val == "검토 필요":
        return "background-color: #f8d7da; font-weight: bold;"
    return ""


# POSCO 그룹사 판별용 키워드
# Company 값은 모두 영어로 기재되어 있다고 가정하고,
# 영문명에 'POSCO'가 포함되어 있으면 모두 POSCO 계열로 간주합니다.
POSCO_KEYWORDS = [
    "POSCO",
    # 필요 시 세부 계열사 영문명을 추가할 수 있지만,
    # 'POSCO' 부분 문자열 매칭만으로 대부분 식별 가능합니다.
]


def is_posco_group_company(value: object) -> bool:
    """company 문자열이 POSCO 또는 POSCO 그룹사인지 여부"""
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    lower = text.lower()
    for kw in POSCO_KEYWORDS:
        if kw.lower() in lower:
            return True
    return False


def posco_row_style(row: pd.Series) -> list:
    """
    POSCO 그룹사 및 데이터 품질에 따른 행 스타일 적용

    - POSCO 계열: 행 전체 옅은 하늘색
    - 그 외 + data_quality == '검토 필요': 행 전체 옅은 빨간색
    """
    is_posco = bool(row.get("is_posco_group", False))
    is_issue = row.get("data_quality") == "검토 필요"

    styles: list[str] = []
    for _ in row.index:
        if is_posco:
            styles.append("background-color: #E8F4FF")
        elif is_issue:
            styles.append("background-color: #f8d7da")
        else:
            styles.append("")
    return styles


# 가격 비교용 보조 함수들 -----------------------------------------------------

AD_KEYWORDS = ["AD", "ANTI-DUMPING", "ANTIDUMPING", "반덤핑"]
CVD_KEYWORDS = ["CVD", "COUNTERVAILING", "상계"]


def is_ad_tariff(tariff_type: object) -> bool:
    if pd.isna(tariff_type):
        return False
    text = str(tariff_type).strip()
    upper = text.upper()
    if "반덤핑" in text:
        return True
    return any(kw in upper for kw in AD_KEYWORDS)


def is_cvd_tariff(tariff_type: object) -> bool:
    if pd.isna(tariff_type):
        return False
    text = str(tariff_type).strip()
    upper = text.upper()
    if "상계" in text:
        return True
    return any(kw in upper for kw in CVD_KEYWORDS)


def parse_tariff_rate(rate_value: object) -> Tuple[Optional[float], bool]:
    """
    관세율 문자열에서 첫 번째 숫자만 추출하여 float(%)로 반환.
    매칭 실패 시 (None, False)를 반환.
    """
    if pd.isna(rate_value):
        return None, False
    rate_str = str(rate_value).strip()
    m = re.search(r"(\d+(\.\d+)?)", rate_str)
    if not m:
        return None, False
    try:
        return float(m.group(1)), True
    except Exception:
        return None, False


def select_best_tariff_record(candidates: pd.DataFrame) -> Optional[pd.Series]:
    """
    AD/CVD 후보 레코드 중에서
    - status == '유효' 우선
    - effective_date_from 최신 우선
    - 그 외 첫 번째
    """
    if candidates is None or candidates.empty:
        return None

    # 유효 상태 우선
    valid = candidates[candidates["status"] == "유효"]
    if not valid.empty:
        candidates = valid

    tmp = candidates.copy()
    tmp["_from"] = pd.to_datetime(tmp["effective_date_from"], errors="coerce")
    tmp = tmp.sort_values(by="_from", ascending=False, na_position="last")
    return tmp.iloc[0]


def compute_candidate_tariff(
    df: pd.DataFrame,
    company: str,
    export_country: Optional[str],
    import_country: str,
    hs_prefix: Optional[str],
) -> Dict[str, Any]:
    """
    단일 후보(자사/경쟁사)에 대해 AD/CVD 매칭 및 관세율/상태/품질/관세 포함 추정가 계산.

    반환 딕셔너리 키:
        ad_rate, cvd_rate, total_rate, status,
        data_quality, quality_reason, matched,
        ad_row, cvd_row
    """
    result: Dict[str, Any] = {
        "ad_rate": 0.0,
        "cvd_rate": 0.0,
        "total_rate": np.nan,
        "status": "",
        "data_quality": "정상",
        "quality_reason": "",
        "matched": False,
        "ad_row": None,
        "cvd_row": None,
    }

    # 기본 매칭 조건
    mask = (df["issuing_country"] == import_country) & (df["company"] == company)
    if export_country:
        mask &= df["country"] == export_country

    candidates = df[mask].copy()

    # HS prefix 필터 (4/6/8 중 가장 구체)
    if hs_prefix:
        candidates = candidates[candidates["hs_digits"].str.startswith(hs_prefix)]

    if candidates.empty:
        # 완전 매칭 없음: 0% 가정 + 검토 필요
        result["total_rate"] = 0.0
        result["data_quality"] = "검토 필요"
        result["quality_reason"] = "매칭데이터없음(0%가정)"
        return result

    result["matched"] = True

    ad_candidates = candidates[candidates["tariff_type"].apply(is_ad_tariff)]
    cvd_candidates = candidates[candidates["tariff_type"].apply(is_cvd_tariff)]

    ad_row = select_best_tariff_record(ad_candidates)
    cvd_row = select_best_tariff_record(cvd_candidates)

    result["ad_row"] = ad_row
    result["cvd_row"] = cvd_row

    reasons = set()
    quality = "정상"

    # 상태(status): AD 우선, 없으면 CVD
    if ad_row is not None:
        result["status"] = ad_row.get("status", "")
    elif cvd_row is not None:
        result["status"] = cvd_row.get("status", "")

    # 개별 관세율 파싱
    ad_rate, ad_ok = (0.0, True)
    if ad_row is not None:
        ad_rate, ad_ok = parse_tariff_rate(ad_row.get("tariff_rate"))
        if not ad_ok:
            quality = "검토 필요"
            reasons.add("관세율파싱실패(AD)")
    else:
        ad_rate, ad_ok = 0.0, True  # AD 자체가 없으면 0%로 간주

    cvd_rate, cvd_ok = (0.0, True)
    if cvd_row is not None:
        cvd_rate, cvd_ok = parse_tariff_rate(cvd_row.get("tariff_rate"))
        if not cvd_ok:
            quality = "검토 필요"
            reasons.add("관세율파싱실패(CVD)")
    else:
        cvd_rate, cvd_ok = 0.0, True  # CVD 자체가 없으면 0%로 간주

    result["ad_rate"] = ad_rate if ad_rate is not None else np.nan
    result["cvd_rate"] = cvd_rate if cvd_rate is not None else np.nan

    # row 단위 데이터 품질 반영
    for r in [ad_row, cvd_row]:
        if r is None:
            continue
        if r.get("data_quality") == "검토 필요":
            quality = "검토 필요"
        qr = str(r.get("quality_reason") or "").strip()
        if qr:
            for token in qr.split(";"):
                token = token.strip()
                if token:
                    reasons.add(token)

    # 총 관세율 및 관세 포함 추정가 계산 가능 여부
    if not ad_ok or not cvd_ok or ad_rate is None or cvd_rate is None:
        result["total_rate"] = np.nan
    else:
        result["total_rate"] = float(ad_rate) + float(cvd_rate)

    result["data_quality"] = quality
    result["quality_reason"] = ";".join(sorted(reasons)) if reasons else ""
    return result


def reset_filters():
    """사이드바 필터 초기화"""
    st.session_state["import_country"] = "선택하세요"
    st.session_state["origin_countries"] = []
    st.session_state["hs_code_input"] = ""


def main():
    st.title("🛃 철강 반덤핑/상계관세 조회 대시보드")
    st.markdown(
        "수입 국가(Import Country), 생산 국가(Origin Country), HS Code를 조합하여 "
        "**반덤핑/상계관세** 정보를 조회할 수 있습니다."
    )
    st.markdown("---")

    # 세션 상태 기본값 설정
    if "import_country" not in st.session_state:
        st.session_state["import_country"] = "선택하세요"
    if "origin_countries" not in st.session_state:
        st.session_state["origin_countries"] = []
    if "hs_code_input" not in st.session_state:
        st.session_state["hs_code_input"] = ""
    if "only_valid" not in st.session_state:
        st.session_state["only_valid"] = False
    if "posco_only" not in st.session_state:
        st.session_state["posco_only"] = False

    # 데이터 로드
    csv_path = "CSV/tariff_data.csv"
    df, missing_cols = load_tariff_data(csv_path)

    required_columns = [
        "hs_code",
        "issuing_country",
        "country",
        "tariff_type",
        "tariff_rate",
        "effective_date_from",
        "effective_date_to",
        "investigation_period_from",
        "investigation_period_to",
        "company",
        "case_number",
        "product_description",
        "note",
    ]

    if df is None:
        st.error(
            "📂 `CSV/tariff_data.csv` 파일을 찾을 수 없습니다. "
            "먼저 기존 앱에서 PDF를 처리하여 CSV를 생성한 뒤 다시 실행해주세요."
        )
        return

    if missing_cols:
        st.error(
            "다음 필수 컬럼이 `tariff_data.csv`에 없습니다. 데이터 생성/전처리를 확인해주세요:\n\n"
            + ", ".join(f"`{c}`" for c in missing_cols)
        )
        return

    if df.empty:
        st.warning("데이터가 비어 있습니다. PDF Processing 파이프라인을 통해 데이터를 먼저 생성해주세요.")
        return

    st.info(f"💾 총 **{len(df)}** 개의 레코드가 로드되었습니다.")

    # 공통 전처리: HS digits 및 prefix 생성
    df["hs_digits"] = df["hs_code"].apply(normalize_hs_digits)
    df["hs_prefix4"] = df["hs_digits"].str.slice(0, 4).where(df["hs_digits"].str.len() >= 4)
    df["hs_prefix6"] = df["hs_digits"].str.slice(0, 6).where(df["hs_digits"].str.len() >= 6)
    df["hs_prefix8"] = df["hs_digits"].str.slice(0, 8).where(df["hs_digits"].str.len() >= 8)

    # 데이터 품질 계산 (상태와는 무관)
    data_quality, quality_reason = compute_data_quality(df)
    df["data_quality"] = data_quality
    df["quality_reason"] = quality_reason

    # POSCO 그룹사 여부
    df["is_posco_group"] = df["company"].apply(is_posco_group_company)

    # 메인 화면 상단: 공통 옵션 (기준일, 유효 필터, POSCO 전용 보기)
    st.subheader("⚙️ 공통 옵션")
    col_opt1, col_opt2, col_opt3 = st.columns(3)

    with col_opt1:
        as_of_date = st.date_input(
            "기준일 (as-of date)",
            value=date.today(),
            key="as_of_date",
        )

    with col_opt2:
        only_valid = st.checkbox(
            "오늘 기준 유효 관세만 보기 (관세 조회 탭에만 적용)",
            value=st.session_state.get("only_valid", False),
            key="only_valid",
        )

    with col_opt3:
        posco_only = st.checkbox(
            "POSCO 계열만 보기 (관세 조회 탭)",
            value=st.session_state.get("posco_only", False),
            key="posco_only",
        )

    st.caption("※ 기준일 및 공통 옵션은 두 탭(관세 조회/가격 비교)에 공통으로 적용됩니다.")

    # 상태(status) 계산 (두 탭 공통 사용)
    df["status"] = compute_status_column(df, as_of_date)

    # 탭 구성: 관세 조회 / 가격 비교
    tab_search, tab_compare = st.tabs(["관세 조회", "가격 비교"])

    # ------------------------------------------------------------------
    # 탭 1: 관세 조회
    # ------------------------------------------------------------------
    with tab_search:
        st.subheader("🔍 관세 조회 - 검색 필터")

        col_f1, col_f_dummy = st.columns([2, 1])

        # 수입 국가 & 수출 국가 필터
        with col_f1:
            import_countries = sorted(df["issuing_country"].dropna().unique().tolist())
            import_options = ["선택하세요"] + import_countries
            selected_import = st.selectbox(
                "수입 국가 (Issuing Country / Import Country)",
                import_options,
                key="import_country",
            )

            origin_countries = sorted(df["country"].dropna().unique().tolist())
            selected_origins = st.multiselect(
                "수출 국가 (Export Country)",
                origin_countries,
                key="origin_countries",
            )

        # 1차 필터: 수입/수출 국가 (HS, 유효만 보기 제외)
        base_df = df.copy()

        # 수입국 필터 (실제 필수 여부는 아래 검색 버튼 처리에서 강제)
        if selected_import != "선택하세요":
            base_df = base_df[base_df["issuing_country"] == selected_import]

        if selected_origins:
            base_df = base_df[base_df["country"].isin(selected_origins)]

        # POSCO 계열만 보기
        if posco_only:
            base_df = base_df[base_df["is_posco_group"]]

        # HS 계층 드릴다운 옵션 구성 (4 → 6 → 8자리)
        st.markdown("HS Code 계층 탐색 (4 → 6 → 8자리)")

        hs4_values = (
            sorted(base_df["hs_prefix4"].dropna().unique().tolist())
            if not base_df.empty
            else []
        )
        hs4_options = ["(선택 안 함)"] + hs4_values

        col_h1, col_h2, col_h3 = st.columns(3)

        with col_h1:
            selected_hs4 = st.selectbox(
                "HS 4자리",
                hs4_options,
                key="hs_prefix4",
                disabled=not hs4_values,
            )

        if selected_hs4 != "(선택 안 함)":
            hs6_source = base_df[base_df["hs_prefix4"] == selected_hs4]
            hs6_values = sorted(hs6_source["hs_prefix6"].dropna().unique().tolist())
        else:
            hs6_values = []

        hs6_options = ["(선택 안 함)"] + hs6_values

        with col_h2:
            selected_hs6 = st.selectbox(
                "HS 6자리",
                hs6_options,
                key="hs_prefix6",
                disabled=(selected_hs4 == "(선택 안 함)") or not hs6_values,
            )

        if selected_hs6 != "(선택 안 함)":
            hs8_source = base_df[base_df["hs_prefix6"] == selected_hs6]
            hs8_values = sorted(hs8_source["hs_prefix8"].dropna().unique().tolist())
        else:
            hs8_values = []

        hs8_options = ["(선택 안 함)"] + hs8_values

        with col_h3:
            selected_hs8 = st.selectbox(
                "HS 8자리",
                hs8_options,
                key="hs_prefix8",
                disabled=(selected_hs6 == "(선택 안 함)") or not hs8_values,
            )

        # 필터 초기화 버튼
        st.button("필터 초기화", on_click=reset_filters, use_container_width=True)
        st.caption("※ 조회를 위해 수입 국가(필수), 수출 국가 또는 HS Code 중 하나 이상을 선택한 뒤 '검색' 버튼을 눌러주세요.")

        # 검색 버튼
        search_clicked = st.button("검색", type="primary", use_container_width=True)

        if not search_clicked:
            st.info("검색 조건을 설정한 뒤 '검색' 버튼을 누르면 결과가 표시됩니다.")
        else:
            # 필수 선택 조건: 수입국은 반드시 선택, 그 외 수출국/HS는 선택하지 않아도 됨
            has_import = selected_import != "선택하세요"
            has_hs = any(
                sel != "(선택 안 함)" for sel in [selected_hs4, selected_hs6, selected_hs8]
            )
            has_export = bool(selected_origins)

            if not has_import:
                st.error("수입 국가는 필수입니다. 수입 국가를 선택해주세요.")
            elif not (has_export or has_hs):
                st.warning(
                    "수입 국가는 선택되었지만, 수출 국가 또는 HS Code 중 하나 이상을 선택하는 것을 권장합니다."
                )

            # 실제 조회용 데이터: base_df를 기반으로 유효/HS 필터 차례로 적용
            filtered_df = base_df.copy()

            # '오늘 기준 유효 관세만 보기' 적용
            if only_valid:
                filtered_df = filtered_df[filtered_df["status"] == "유효"]

            # HS 계층 드릴다운 적용 (8 > 6 > 4)
            if selected_hs8 != "(선택 안 함)":
                filtered_df = filtered_df[filtered_df["hs_digits"].str.startswith(selected_hs8)]
            elif selected_hs6 != "(선택 안 함)":
                filtered_df = filtered_df[filtered_df["hs_digits"].str.startswith(selected_hs6)]
            elif selected_hs4 != "(선택 안 함)":
                filtered_df = filtered_df[filtered_df["hs_digits"].str.startswith(selected_hs4)]

            # 정렬: case_number, company, hs_code 순
            sort_cols = [c for c in ["case_number", "company", "hs_code"] if c in filtered_df.columns]
            if sort_cols:
                filtered_df = filtered_df.sort_values(by=sort_cols, na_position="last")

            st.markdown("---")

            # 결과 수 / 품질 요약 표시
            st.subheader("📊 조회 결과")
            total_count = len(filtered_df)
            issue_count = (filtered_df["data_quality"] == "검토 필요").sum()
            normal_count = (filtered_df["data_quality"] == "정상").sum()

            st.info(f"필터 조건에 맞는 레코드 수: **{total_count}** 건")

            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                st.metric("총 레코드", total_count)
            with col_q2:
                st.metric("정상", int(normal_count))
            with col_q3:
                st.metric("검토 필요", int(issue_count))

            if filtered_df.empty:
                st.warning("해당 조건에 맞는 데이터가 없습니다. 필터를 완화하여 다시 시도해보세요.")
            else:
                # 테이블 표시 (index는 새로 리셋해서 UI용으로 사용)
                display_df_full = filtered_df.reset_index(drop=True)

                # 조회 결과에서 숨길 컬럼들
                hidden_columns = [
                    "hs_digits",
                    "hs_prefix4",
                    "hs_prefix6",
                    "hs_prefix8",
                    "data_quality",
                    "quality_reason",
                    "is_posco_group",
                    "status",
                ]
                visible_columns = [
                    c for c in display_df_full.columns if c not in hidden_columns
                ]
                display_df = display_df_full[visible_columns]

                def row_style_visible(row: pd.Series) -> list:
                    """
                    화면에 표시되는 컬럼 수(visible_columns)에 맞춰 스타일 리스트를 반환하되,
                    POSCO 여부 / 데이터 품질은 hidden 컬럼이 포함된 전체 행(display_df_full)을 기준으로 계산.
                    """
                    full_row = display_df_full.loc[row.name]
                    is_posco = bool(full_row.get("is_posco_group", False))
                    is_issue = full_row.get("data_quality") == "검토 필요"

                    styles: list[str] = []
                    for _ in row.index:
                        if is_posco:
                            styles.append("background-color: #E8F4FF")
                        elif is_issue:
                            styles.append("background-color: #f8d7da")
                        else:
                            styles.append("")
                    return styles

                styled = display_df.style.apply(row_style_visible, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)

                # 상세 조회 섹션
                st.markdown("---")
                st.subheader("🔎 상세 조회")

                index_options = display_df.index.tolist()

                def format_record(idx: int) -> str:
                    row = display_df.loc[idx]
                    company = row.get("company", "")
                    case_no = row.get("case_number", "")
                    hs = row.get("hs_code", "")
                    return f"{company} | {case_no} | {hs}"

                selected_idx = st.selectbox(
                    "상세 조회할 레코드를 선택하세요",
                    index_options,
                    format_func=format_record,
                )

                row = display_df.loc[selected_idx]

                with st.expander("📄 선택된 레코드 상세 정보", expanded=True):
                    st.markdown(
                        f"""
                        - **수입 국가 (Issuing / Import Country)**: `{row.get('issuing_country', '')}`  
                        - **수출 국가 (Export Country)**: `{row.get('country', '')}`  
                        - **회사명 (Company)**: `{row.get('company', '')}`  
                        - **사건번호 (Case Number)**: `{row.get('case_number', '')}`  
                        - **HS Code**: `{row.get('hs_code', '')}`  
                        - **관세 유형 (Tariff Type)**: `{row.get('tariff_type', '')}`  
                        - **관세율 (Tariff Rate)**: `{row.get('tariff_rate', '')}`  

                        ---
                        **적용 기간 (Effective Period)**  
                        - `{row.get('effective_date_from', '')}` ~ `{row.get('effective_date_to', '')}`  

                        **조사 기간 (Investigation Period)**  
                        - `{row.get('investigation_period_from', '')}` ~ `{row.get('investigation_period_to', '')}`  

                        ---
                        **제품 설명 (Product Description)**  
                        {row.get('product_description', '')}

                        ---
                        **비고 (Note)**  
                        {row.get('note', '')}
                        """
                    )

                # CSV 다운로드 (현재 필터 결과만)
                st.markdown("---")
                csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig")
                as_of_str = as_of_date.strftime("%Y%m%d")
                safe_import = str(selected_import).replace(" ", "_")
                st.download_button(
                    label="📥 현재 필터 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"tariff_result_{safe_import}_{as_of_str}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ------------------------------------------------------------------
    # 탭 2: 가격 비교
    # ------------------------------------------------------------------
    with tab_compare:
        st.subheader("💲 가격 비교 (자사 vs 경쟁사)")
        st.caption(
            "자사 및 경쟁사 후보의 수출 국가/생산자/HS Code/CIF를 기준으로 "
            "반덤핑(AD) + 상계(CVD) 관세율을 매칭하여 관세 포함 추정가를 비교합니다."
        )

        export_countries = sorted(df["country"].dropna().unique().tolist())
        import_countries_pc = sorted(df["issuing_country"].dropna().unique().tolist())
        import_options_pc = ["선택하세요"] + import_countries_pc

        # 공통 수입 국가 (자사/경쟁사 모두 동일하게 적용)
        st.markdown("#### 공통 조건")
        common_import = st.selectbox(
            "수입 국가 (공통, Issuing Country / Import Country)",
            import_options_pc,
            key="pc_common_import",
        )
        st.caption("※ 자사(A)와 모든 경쟁사 후보(B, C)는 위에서 선택한 동일한 수입 국가를 기준으로 비교합니다.")

        def render_hs_drilldown_for_candidate(key_prefix: str) -> Tuple[str, str, str]:
            """후보별 HS 4/6/8자리 드릴다운 UI"""
            hs4_vals = sorted(df["hs_prefix4"].dropna().unique().tolist())
            hs4_options_local = ["(선택 안 함)"] + hs4_vals

            col1, col2, col3 = st.columns(3)
            with col1:
                hs4 = st.selectbox(
                    "HS 4자리",
                    hs4_options_local,
                    key=f"{key_prefix}_hs4",
                    disabled=not hs4_vals,
                )

            if hs4 != "(선택 안 함)":
                hs6_src = df[df["hs_prefix4"] == hs4]
                hs6_vals = sorted(hs6_src["hs_prefix6"].dropna().unique().tolist())
            else:
                hs6_vals = []

            hs6_options_local = ["(선택 안 함)"] + hs6_vals
            with col2:
                hs6 = st.selectbox(
                    "HS 6자리",
                    hs6_options_local,
                    key=f"{key_prefix}_hs6",
                    disabled=(hs4 == "(선택 안 함)") or not hs6_vals,
                )

            if hs6 != "(선택 안 함)":
                hs8_src = df[df["hs_prefix6"] == hs6]
                hs8_vals = sorted(hs8_src["hs_prefix8"].dropna().unique().tolist())
            else:
                hs8_vals = []

            hs8_options_local = ["(선택 안 함)"] + hs8_vals
            with col3:
                hs8 = st.selectbox(
                    "HS 8자리",
                    hs8_options_local,
                    key=f"{key_prefix}_hs8",
                    disabled=(hs6 == "(선택 안 함)") or not hs8_vals,
                )

            st.caption("※ HS Code는 8자리까지 선택하는 것을 권장합니다.")
            return hs4, hs6, hs8

        # (A) 자사
        st.markdown("### (A) 자사 (필수)")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            export_a = st.selectbox(
                "수출 국가 (Export Country)",
                export_countries,
                key="pc_A_export",
            )
            companies_a = sorted(
                df.loc[df["country"] == export_a, "company"].dropna().unique().tolist()
            )
            company_a = st.selectbox(
                "생산자 (Company)",
                companies_a,
                key="pc_A_company",
            )
        with col_a2:
            st.empty()

        st.markdown("**자사(A) HS Code 선택**")
        hs4_a, hs6_a, hs8_a = render_hs_drilldown_for_candidate("pc_A")

        cif_a = st.number_input(
            "자사(A) CIF 가격",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="pc_A_cif",
        )

        # (B) 경쟁사 후보 1
        st.markdown("### (B) 경쟁사 후보 1")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            export_b = st.selectbox(
                "수출 국가 (Export Country)",
                export_countries,
                key="pc_B_export",
            )
            companies_b = sorted(
                df.loc[df["country"] == export_b, "company"].dropna().unique().tolist()
            )
            company_b = st.selectbox(
                "생산자 (Company)",
                companies_b,
                key="pc_B_company",
            )
        with col_b2:
            st.empty()

        st.markdown("**경쟁사1(B) HS Code 선택**")
        hs4_b, hs6_b, hs8_b = render_hs_drilldown_for_candidate("pc_B")

        cif_b = st.number_input(
            "경쟁사1(B) CIF 가격",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="pc_B_cif",
        )

        # (C) 경쟁사 후보 2 (선택)
        use_c = st.checkbox("경쟁사 후보 2 사용하기", value=False, key="pc_use_C")
        company_c = export_c = None
        cif_c = 0.0
        hs4_c = hs6_c = hs8_c = "(선택 안 함)"

        if use_c:
            st.markdown("### (C) 경쟁사 후보 2")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                export_c = st.selectbox(
                    "수출 국가 (Export Country)",
                    export_countries,
                    key="pc_C_export",
                )
                companies_c = sorted(
                    df.loc[df["country"] == export_c, "company"].dropna().unique().tolist()
                )
                company_c = st.selectbox(
                    "생산자 (Company)",
                    companies_c,
                    key="pc_C_company",
                )
            with col_c2:
                st.empty()

            st.markdown("**경쟁사2(C) HS Code 선택**")
            hs4_c, hs6_c, hs8_c = render_hs_drilldown_for_candidate("pc_C")

            cif_c = st.number_input(
                "경쟁사2(C) CIF 가격",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="pc_C_cif",
            )

        compare_clicked = st.button("비교", type="primary")

        if compare_clicked:
            # 공통 수입국 검증
            if common_import == "선택하세요":
                st.error("공통 수입 국가를 선택해주세요.")
                return

            def choose_hs_prefix(h4: str, h6: str, h8: str) -> Optional[str]:
                if h8 != "(선택 안 함)":
                    return h8
                if h6 != "(선택 안 함)":
                    return h6
                if h4 != "(선택 안 함)":
                    return h4
                return None

            results: list[Dict[str, Any]] = []

            # 자사(A)
            hs_prefix_a = choose_hs_prefix(hs4_a, hs6_a, hs8_a)
            res_a = compute_candidate_tariff(
                df=df,
                company=company_a,
                export_country=export_a,
                import_country=common_import,
                hs_prefix=hs_prefix_a,
            )
            res_a.update(
                {
                    "label": "자사(A)",
                    "company": company_a,
                    "export_country": export_a,
                    "import_country": common_import,
                    "hs_display": hs8_a
                    if hs8_a != "(선택 안 함)"
                    else hs6_a
                    if hs6_a != "(선택 안 함)"
                    else hs4_a
                    if hs4_a != "(선택 안 함)"
                    else "",
                    "cif": cif_a,
                }
            )
            results.append(res_a)

            # 경쟁사(B)
            hs_prefix_b = choose_hs_prefix(hs4_b, hs6_b, hs8_b)
            res_b = compute_candidate_tariff(
                df=df,
                company=company_b,
                export_country=export_b,
                import_country=common_import,
                hs_prefix=hs_prefix_b,
            )
            res_b.update(
                {
                    "label": "경쟁사1(B)",
                    "company": company_b,
                    "export_country": export_b,
                    "import_country": common_import,
                    "hs_display": hs8_b
                    if hs8_b != "(선택 안 함)"
                    else hs6_b
                    if hs6_b != "(선택 안 함)"
                    else hs4_b
                    if hs4_b != "(선택 안 함)"
                    else "",
                    "cif": cif_b,
                }
            )
            results.append(res_b)

            # 경쟁사(C)
            if use_c:
                hs_prefix_c = choose_hs_prefix(hs4_c, hs6_c, hs8_c)
                res_c = compute_candidate_tariff(
                    df=df,
                    company=company_c,
                    export_country=export_c,
                    import_country=common_import,
                    hs_prefix=hs_prefix_c,
                )
                res_c.update(
                    {
                        "label": "경쟁사2(C)",
                        "company": company_c,
                        "export_country": export_c,
                        "import_country": common_import,
                        "hs_display": hs8_c
                        if hs8_c != "(선택 안 함)"
                        else hs6_c
                        if hs6_c != "(선택 안 함)"
                        else hs4_c
                        if hs4_c != "(선택 안 함)"
                        else "",
                        "cif": cif_c,
                    }
                )
                results.append(res_c)

            if not results:
                st.warning("비교할 후보가 없습니다. 수입 국가를 다시 확인해주세요.")
                return

            # 관세 포함 추정가 및 관세액 계산
            for r in results:
                total_rate = r.get("total_rate")
                cif_val = r.get("cif", 0.0) or 0.0
                if total_rate is None or (isinstance(total_rate, float) and np.isnan(total_rate)):
                    r["duty_amount"] = np.nan
                    r["landed_price"] = np.nan
                else:
                    r["duty_amount"] = cif_val * (total_rate / 100.0)
                    r["landed_price"] = cif_val * (1 + total_rate / 100.0)

                # 데이터 품질이 이미 "검토 필요"가 아닌데 HS 미입력인 경우 참고 사유 추가
                hs_disp = r.get("hs_display", "")
                if not hs_disp:
                    if r.get("data_quality") == "정상":
                        r["data_quality"] = "검토 필요"
                    reasons = set(
                        (r.get("quality_reason") or "").split(";")
                    ) if r.get("quality_reason") else set()
                    reasons.add("HS코드미입력")
                    r["quality_reason"] = ";".join(sorted(x for x in reasons if x))

            # 결과 테이블 요약
            summary_rows = []
            for r in results:
                summary_rows.append(
                    {
                        "구분": r["label"],
                        "Company": r["company"],
                        "Export": r["export_country"],
                        "Import": r["import_country"],
                        "HS 선택": r["hs_display"] or "-",
                        "CIF": r["cif"],
                        "AD 관세율(%)": r.get("ad_rate"),
                        "CVD 관세율(%)": r.get("cvd_rate"),
                        "Total 관세율(%)": r.get("total_rate"),
                        "상태(status)": r.get("status", ""),
                        "데이터 품질": r.get("data_quality", ""),
                        "품질 사유": r.get("quality_reason", ""),
                        "관세액": r.get("duty_amount"),
                        "관세 포함 추정가": r.get("landed_price"),
                    }
                )

            st.markdown("---")
            st.subheader("📑 후보별 요약 결과")
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(
                summary_df.style.format(
                    {
                        "CIF": "{:,.2f}",
                        "AD 관세율(%)": "{:.2f}",
                        "CVD 관세율(%)": "{:.2f}",
                        "Total 관세율(%)": "{:.2f}",
                        "관세액": "{:,.2f}",
                        "관세 포함 추정가": "{:,.2f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
            )

            # 경쟁력 판정
            st.markdown("---")
            st.subheader("⚖️ 경쟁력 판정")

            base = results[0]
            base_lp = base.get("landed_price")

            if base_lp is None or (isinstance(base_lp, float) and np.isnan(base_lp)):
                st.warning("자사(A)의 총 관세율 또는 관세 포함 추정가 계산이 불가능합니다.")
            else:
                for comp in results[1:]:
                    comp_lp = comp.get("landed_price")
                    if comp_lp is None or (isinstance(comp_lp, float) and np.isnan(comp_lp)):
                        st.info(f"{comp['label']}의 관세 포함 추정가를 계산할 수 없어 비교에서 제외됩니다.")
                        continue

                    diff_amt = comp_lp - base_lp
                    diff_pct = (diff_amt / comp_lp * 100.0) if comp_lp != 0 else np.nan

                    if diff_amt > 0:
                        verdict = "경쟁력 있음 (자사가 더 저렴)"
                    elif diff_amt < 0:
                        verdict = "경쟁력 낮음 (자사가 더 비쌈)"
                    else:
                        verdict = "가격 동일"

                    st.markdown(
                        f"- **{comp['label']} 대비**: {verdict}  \n"
                        f"  - 가격 차이: {diff_amt:,.2f} (자사 기준), "
                        f"{'' if np.isnan(diff_pct) else f'{diff_pct:.2f}%'}"
                    )

            # 간단 차트 (landed_price 막대그래프)
            st.markdown("---")
            st.subheader("📈 관세 포함 추정가 비교 차트")
            chart_rows = [
                {"구분": r["label"], "관세 포함 추정가": r.get("landed_price")}
                for r in results
                if r.get("landed_price") is not None
                and not (isinstance(r.get("landed_price"), float) and np.isnan(r.get("landed_price")))
            ]
            if chart_rows:
                chart_df = pd.DataFrame(chart_rows).set_index("구분")
                st.bar_chart(chart_df)
            else:
                st.info("유효한 관세 포함 추정가가 없어 차트를 표시할 수 없습니다.")


if __name__ == "__main__":
    main()



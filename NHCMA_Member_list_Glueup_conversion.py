import io
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="GlueUp Member Upload Cleaner", layout="wide")

st.title("GlueUp Member Upload Cleaner")
st.caption("Clean and prepare member lists for GlueUp import — with previews and exports.")

with st.expander("How this works", expanded=True):
    st.markdown("""
- Upload your **member list** and **specialty answer list** (Excel).
- The app will:
  - Rename common columns (Primary Email→Email, Member Date→Start Date, Address1→Address, Expiration Date→End Date)
  - Normalize ZIPs to 5 digits
  - Lowercase **Code** and address types
  - Map **specialties** by matching a chosen members column to the **second column** in the answer list; returns the **first column** from the answer list as the mapped code/value
  - Set **End Date = 12/31/2099** if **Code** starts with 'L'
  - Fill missing **Start Date** with **Feb 1 two years before End Date**
  - Split outputs into **Cleaned_Members_for_GlueUp.xlsx** and **Members_Missing_Emails.xlsx**
""")

st.sidebar.header("Upload files")
members_file = st.sidebar.file_uploader("Member list (.xlsx)", type=["xlsx"])
answer_file  = st.sidebar.file_uploader("Specialty answer list (.xlsx)", type=["xlsx"])

def load_excel(uploaded) -> pd.DataFrame:
    if not uploaded:
        return None
    return pd.read_excel(uploaded)

def clean_and_process(members_df: pd.DataFrame, answer_df: pd.DataFrame,
                      members_specialty_col: str | None,
                      answer_key_col: str, answer_match_col: str):
    # 1) Rename common columns
    rename_map = {}
    for col in members_df.columns:
        low = str(col).strip().lower()
        if low == 'primary email':
            rename_map[col] = 'Email'
        elif low == 'member date':
            rename_map[col] = 'Start Date'
        elif low == 'address1':
            rename_map[col] = 'Address'
        elif low == 'expiration date':
            rename_map[col] = 'End Date'
    members_df = members_df.rename(columns=rename_map).copy()

    # 2) Format ZIP as 5-char with leading zeros if present
    if 'Zip code' in members_df.columns:
        members_df['Zip code'] = members_df['Zip code'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(5)

    # 3) Lowercase select text fields
    if 'Code' in members_df.columns:
        members_df['Code'] = members_df['Code'].astype(str).str.lower()
    for col in ['Address type', 'Address type description']:
        if col in members_df.columns:
            members_df[col] = members_df[col].astype(str).str.lower()
    if 'Address type description' in members_df.columns:
        members_df['Address type description'] = members_df['Address type description'].str.replace(' ', '-', regex=False)

    # 4) Specialty mapping (VLOOKUP-style). Default behavior = use members col index 18 (“column S”) if the user doesn’t choose.
    # Answer list: return answer_key_col by matching members_specialty_col to answer_match_col.
    # Drop NA in match col to avoid bad index entries.
    answer_df_clean = answer_df.dropna(subset=[answer_match_col]).copy()

    if members_specialty_col is None:
        # Fall back to column S (0-based index 18) if it exists
        if members_df.shape[1] > 18:
            specialty_series = members_df.iloc[:, 18]
        else:
            specialty_series = pd.Series([None] * len(members_df))
    else:
        specialty_series = members_df[members_specialty_col] if members_specialty_col in members_df.columns else pd.Series([None]*len(members_df))

    mapper = answer_df_clean.set_index(answer_match_col)[answer_key_col]
    members_df['Specialty Description - 1'] = specialty_series.map(mapper)

    # 5) End Date to 2099-12-31 where Code starts with 'l'
    if 'Code' in members_df.columns and 'End Date' in members_df.columns:
        mask_l = members_df['Code'].astype(str).str.startswith('l', na=False)
        members_df.loc[mask_l, 'End Date'] = pd.Timestamp('2099-12-31')

    # 6) Fill Start Date if blank -> Feb 1 two years prior to End Date
    def fill_start(row):
        raw = row.get('Start Date')
        is_blank = (pd.isna(raw) or str(raw).strip().lower() in ['', 'none', 'nan'])
        if is_blank:
            end_date = pd.to_datetime(row.get('End Date'), errors='coerce')
            if pd.notnull(end_date):
                return pd.Timestamp(year=end_date.year - 2, month=2, day=1)
            return ''
        parsed = pd.to_datetime(raw, errors='coerce')
        return parsed if pd.notnull(parsed) else ''

    if 'Start Date' in members_df.columns:
        members_df['Start Date'] = members_df.apply(fill_start, axis=1)

    # 7) Require Email, split outputs
    if 'Email' not in members_df.columns:
        raise KeyError("Expected 'Email' column is missing after renaming. Make sure your file has 'Primary Email' or 'Email'.")

    is_blank_email = members_df['Email'].isna() | (members_df['Email'].astype(str).str.strip() == '')
    missing_email_df = members_df[is_blank_email].copy()
    cleaned_df = members_df[~is_blank_email].copy()

    # Return dataframes + bytes for download
    def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        return buf.getvalue()

    return cleaned_df, missing_email_df, to_xlsx_bytes(cleaned_df), to_xlsx_bytes(missing_email_df)

if members_file and answer_file:
    members_df = load_excel(members_file)
    answer_df  = load_excel(answer_file)

    st.subheader("Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Members (first 200 rows)**")
        st.dataframe(members_df.head(200), use_container_width=True)
    with c2:
        st.markdown("**Answer List (first 200 rows)**")
        st.dataframe(answer_df.head(200), use_container_width=True)

    st.divider()
    st.subheader("Mapping Options (optional)")

    # Members specialty column selection (optional)
    members_specialty_col = st.selectbox(
        "Members column to match against the Answer List’s match column (defaults to column S if not selected)",
        options=["(auto: column S / index 18)"] + list(members_df.columns),
        index=0
    )
    members_specialty_col = None if members_specialty_col.startswith("(auto") else members_specialty_col

    # Answer list columns: first = returned value (key), second = match column (default = first two columns)
    if len(answer_df.columns) < 2:
        st.error("Answer list must have at least two columns: first is the value to return, second is the value to match.")
    else:
        answer_key_col = st.selectbox("Answer list column to RETURN (default = first column)", options=list(answer_df.columns), index=0)
        answer_match_col = st.selectbox("Answer list column to MATCH ON (default = second column)", options=list(answer_df.columns), index=1)

    st.divider()
    if st.button("Run Cleaning", type="primary"):
        try:
            cleaned_df, missing_df, cleaned_bytes, missing_bytes = clean_and_process(
                members_df, answer_df, members_specialty_col, answer_key_col, answer_match_col
            )
            st.success(f"Done! Retained {len(cleaned_df)} rows with Email; {len(missing_df)} rows missing Email.")
            st.markdown("**Cleaned Output Preview**")
            st.dataframe(cleaned_df.head(200), use_container_width=True)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download Cleaned_Members_for_GlueUp.xlsx",
                    data=cleaned_bytes,
                    file_name="Cleaned_Members_for_GlueUp.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with d2:
                st.download_button(
                    "Download Members_Missing_Emails.xlsx",
                    data=missing_bytes,
                    file_name="Members_Missing_Emails.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Processing failed: {e}")

else:
    st.info("Upload both files in the sidebar to begin.")

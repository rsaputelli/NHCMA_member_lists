import io
import re
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="GlueUp Member Upload Cleaner", layout="wide")

st.title("GlueUp Member Upload Cleaner")
st.caption("Clean and prepare member lists for GlueUp import — with previews, mapping, and exports.")

with st.expander("How this works", expanded=True):
    st.markdown("""
- Upload your **member list** and **specialty answer list** (Excel).
- Choose the correct **sheet** (and optionally header **skiprows**) in the sidebar.
- Confirm/adjust column mappings (auto-detected).
- Click **Run Cleaning** to:
  - Canonicalize key columns (Email, Start Date, End Date, Address, Zip, Code)
  - Normalize ZIPs to 5 digits
  - Lowercase **Code** and address types (hyphenate Address type description)
  - Map **specialties** (choose members column to match; choose return/match columns from answer list)
  - Set **End Date = 12/31/2099** if **Code** starts with “L”
  - Backfill missing **Start Date** with **Feb 1 two years before End Date**
  - Split outputs into **Cleaned_Members_for_GlueUp.xlsx** and **Members_Missing_Emails.xlsx**
""")

# ---------- Helpers ----------
def _normalize_cols(cols):
    def norm(c):
        c = str(c).strip()
        return re.sub(r'[^a-z0-9]+', ' ', c.lower()).strip()
    return {c: norm(c) for c in cols}

def _guess(colmap, patterns):
    for orig, norm in colmap.items():
        for pat in patterns:
            if re.search(pat, norm):
                return orig
    return None

def read_excel_with_options(uploaded_file, sheet_name=None, skiprows=0):
    """Read Excel allowing sheet selection and header rows to skip."""
    if not uploaded_file:
        return None
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skiprows)

def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buf.getvalue()

# ---------- Uploads ----------
st.sidebar.header("Upload files")
members_file = st.sidebar.file_uploader("Member list (.xlsx)", type=["xlsx"])
answer_file  = st.sidebar.file_uploader("Specialty answer list (.xlsx)", type=["xlsx"])

# New: sheet pickers + skiprows
members_df = None
answer_df  = None

if members_file:
    try:
        xls_m = pd.ExcelFile(members_file)
        st.sidebar.subheader("Members file options")
        m_sheet = st.sidebar.selectbox("Members sheet", xls_m.sheet_names, index=0, key="m_sheet")
        m_skip  = st.sidebar.number_input("Rows to skip before header (members)", min_value=0, max_value=200, value=0, step=1, key="m_skip")
        members_df = read_excel_with_options(members_file, sheet_name=m_sheet, skiprows=m_skip)
    except Exception as e:
        st.sidebar.error(f"Members file error: {e}")

if answer_file:
    try:
        xls_a = pd.ExcelFile(answer_file)
        st.sidebar.subheader("Answer list options")
        a_sheet = st.sidebar.selectbox("Answer list sheet", xls_a.sheet_names, index=0, key="a_sheet")
        a_skip  = st.sidebar.number_input("Rows to skip before header (answer list)", min_value=0, max_value=200, value=0, step=1, key="a_skip")
        answer_df = read_excel_with_options(answer_file, sheet_name=a_sheet, skiprows=a_skip)
    except Exception as e:
        st.sidebar.error(f"Answer list error: {e}")

if members_df is not None and answer_df is not None:
    st.subheader("Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Members (first 200 rows)**")
        st.dataframe(members_df.head(200), use_container_width=True)
    with c2:
        st.markdown("**Answer List (first 200 rows)**")
        st.dataframe(answer_df.head(200), use_container_width=True)

    # ---------- Auto-detect columns ----------
    norm_map = _normalize_cols(members_df.columns)
    email_guess   = _guess(norm_map, [r'\bemail\b', r'\bprimary\s*email\b', r'\bemail\s*address\b'])
    start_guess   = _guess(norm_map, [r'\bmember\s*date\b', r'\bstart\b'])
    end_guess     = _guess(norm_map, [r'\bexpiration\b|\bexpire\b|\bend\s*date\b'])
    zip_guess     = _guess(norm_map, [r'\bzip\b|\bpostal\b'])
    code_guess    = _guess(norm_map, [r'\bcode\b|\bmember\s*type\b|\bcategory\b'])
    addr_type_guess = _guess(norm_map, [r'\baddress\s*type\b'])
    addr_type_desc_guess = _guess(norm_map, [r'\baddress\s*type\s*description\b|\baddress\s*desc'])

    st.sidebar.header("Column mapping (auto-detected; you can override)")
    def idx(opt_list, val):  # helper for selectbox default
        return 0 if val is None else opt_list.index(val)

    opts = [None] + list(members_df.columns)
    email_col   = st.sidebar.selectbox("Email column", options=opts, index=idx(opts, email_guess))
    start_col   = st.sidebar.selectbox("Start Date column", options=opts, index=idx(opts, start_guess))
    end_col     = st.sidebar.selectbox("End Date column", options=opts, index=idx(opts, end_guess))
    zip_col     = st.sidebar.selectbox("ZIP/Postal column", options=opts, index=idx(opts, zip_guess))
    code_col    = st.sidebar.selectbox("Code / Member Type column", options=opts, index=idx(opts, code_guess))
    addr_type_col = st.sidebar.selectbox("Address type column (optional)", options=opts, index=idx(opts, addr_type_guess))
    addr_type_desc_col = st.sidebar.selectbox("Address type description column (optional)", options=opts, index=idx(opts, addr_type_desc_guess))

    st.sidebar.divider()
    st.sidebar.subheader("Specialty mapping")
    members_specialty_default = members_df.columns[18] if members_df.shape[1] > 18 else None
    members_specialty_col = st.sidebar.selectbox("Members column to MATCH against Answer List",
                                                 options=opts, index=idx(opts, members_specialty_default))
    if len(answer_df.columns) < 2:
        st.error("Answer list must have at least two columns: one to return, one to match on.")
        st.stop()
    answer_return_col = st.sidebar.selectbox("Answer list column to RETURN", options=list(answer_df.columns), index=0)
    answer_match_col  = st.sidebar.selectbox("Answer list column to MATCH ON", options=list(answer_df.columns), index=min(1, len(answer_df.columns)-1))

    st.divider()
    if st.button("Run Cleaning", type="primary"):
        df = members_df.copy()
        log = {}

        # Canonical headers
        if email_col: df.rename(columns={email_col: "Email"}, inplace=True)
        if start_col: df.rename(columns={start_col: "Start Date"}, inplace=True)
        if end_col:   df.rename(columns={end_col: "End Date"}, inplace=True)
        if zip_col:   df.rename(columns={zip_col: "Zip code"}, inplace=True)
        if code_col:  df.rename(columns={code_col: "Code"}, inplace=True)
        if addr_type_col: df.rename(columns={addr_type_col: "Address type"}, inplace=True)
        if addr_type_desc_col: df.rename(columns={addr_type_desc_col: "Address type description"}, inplace=True)

        # Normalize ZIP
        if "Zip code" in df.columns:
            before = df["Zip code"].notna().sum()
            df["Zip code"] = df["Zip code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            log["ZIP normalized"] = before

        # Lowercase address type(s)
        for col in ["Address type", "Address type description"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
        if "Address type description" in df.columns:
            df["Address type description"] = df["Address type description"].str.replace(" ", "-", regex=False)

        # Lowercase Code
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str.lower()

        # Specialty mapping
        spec_mapped = 0
        if members_specialty_col and (members_specialty_col in df.columns):
            ans = answer_df.dropna(subset=[answer_match_col]).copy()
            mapper = ans.set_index(answer_match_col)[answer_return_col]
            df["Specialty Description - 1"] = df[members_specialty_col].map(mapper)
            spec_mapped = df["Specialty Description - 1"].notna().sum()
        log["Specialties mapped"] = spec_mapped

        # End Date = 2099-12-31 if Code starts with 'l'
        if "Code" in df.columns and "End Date" in df.columns:
            mask_l = df["Code"].astype(str).str.startswith("l", na=False)
            count_l = int(mask_l.sum())
            df.loc[mask_l, "End Date"] = pd.Timestamp("2099-12-31")
            log["End dates set to 2099 (Code=L*)"] = count_l

        # Fill Start Date if missing -> Feb 1 two years prior to End Date
        backfilled_counter = {"count": 0}
        if "Start Date" in df.columns:
            def fill_start(row):
                raw = row.get("Start Date")
                is_blank = (pd.isna(raw) or str(raw).strip().lower() in ["", "none", "nan"])
                if is_blank:
                    end_dt = pd.to_datetime(row.get("End Date"), errors="coerce")
                    if pd.notnull(end_dt):
                        backfilled_counter["count"] += 1
                        return pd.Timestamp(year=end_dt.year - 2, month=2, day=1)
                    return pd.NaT
                parsed = pd.to_datetime(raw, errors="coerce")
                return parsed
            df["Start Date"] = df.apply(fill_start, axis=1)
        log["Start dates backfilled"] = int(backfilled_counter["count"])

        # Split by Email presence
        if "Email" not in df.columns:
            st.error("No Email column found. Use the sidebar to map your email column.")
            st.stop()

        missing_mask = df["Email"].isna() | (df["Email"].astype(str).str.strip() == "")
        missing_df = df[missing_mask].copy()
        cleaned_df = df[~missing_mask].copy()

        # Summary
        st.success("Processing complete.")
        st.markdown(f"""
**Rows total:** {len(df)}  
**Rows with email (cleaned):** {len(cleaned_df)}  
**Rows missing email:** {len(missing_df)}  
**Start dates backfilled:** {log.get("Start dates backfilled", 0)}  
**End dates → 2099 (Code starts with 'L'):** {log.get("End dates set to 2099 (Code=L*)", 0)}  
**Specialties mapped:** {log.get("Specialties mapped", 0)}  
**ZIPs normalized (had a value):** {log.get("ZIP normalized", 0)}
""")

        st.markdown("**Cleaned Output Preview (first 200)**")
        st.dataframe(cleaned_df.head(200), use_container_width=True)

        # Downloads (always present)
        cdl, mdl = to_xlsx_bytes(cleaned_df), to_xlsx_bytes(missing_df)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download Cleaned_Members_for_GlueUp.xlsx",
                               data=cdl, file_name="Cleaned_Members_for_GlueUp.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with d2:
            st.download_button("Download Members_Missing_Emails.xlsx",
                               data=mdl, file_name="Members_Missing_Emails.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload both files to begin. Then choose the correct sheet(s) in the sidebar.")

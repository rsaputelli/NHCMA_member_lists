import io
import re
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from email_enrichment_util import email_enrichment_sidebar

st.set_page_config(page_title="GlueUp Member Upload Cleaner", layout="wide")

# === Branding (logo left of title) ===
LOGO_PATH = "logo.png"
hdr_l, hdr_r = st.columns([1, 8])
with hdr_l:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=140)  # no use_column_width param
with hdr_r:
    st.title("GlueUp Member Upload Cleaner")
    st.caption("Clean and prepare member lists for GlueUp import — with previews, mapping, and exports.")
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)  # OK: new param name


with st.expander("How this works", expanded=True):
    st.markdown("""
- Upload your **member list** and **specialty answer list** (Excel).
- Choose the correct **sheet** (and optionally header **skiprows**) in the sidebar.
- Confirm/adjust column mappings (auto-detected).
- Click **Run Cleaning** to:
  - Canonicalize key columns (Email, Start Date, End Date, Address, Zip, **code**)
  - Normalize ZIPs to 5 digits
  - Lowercase **code** and address types (hyphenate Address type description)
  - Map **specialties**
  - Set **End Date = 12/31/2099** if **code** starts with “l”
  - Backfill missing **Start Date** with **Feb 1 two years before End Date**
  - Split outputs into **Cleaned_Members_for_GlueUp.xlsx** and **Members_Missing_Emails.xlsx**
""")

# ---------- Helpers ----------
def _normalize_cols(cols):
    def norm(c):
        c = str(c).strip()
        return re.sub(r'[^a-z0-9]+', ' ', c.lower()).strip()
    return {c: norm(c) for c in cols}

def _guess_with_exclusions(colmap, include_patterns, exclude_contains=None):
    """Return first original column whose normalized form matches any include pattern,
    skipping any whose normalized name contains an excluded token (e.g., 'zip')."""
    exclude_contains = exclude_contains or []
    for orig, norm in colmap.items():
        if any(ex in norm for ex in exclude_contains):
            continue
        for pat in include_patterns:
            if re.search(pat, norm):
                return orig
    return None

def read_excel_with_options(uploaded_file, sheet_name=None, skiprows=0):
    if not uploaded_file:
        return None
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skiprows)

def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buf.getvalue()

def coalesce_duplicate_named_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c == name]
    if len(cols) <= 1:
        return df
    dup = df.loc[:, cols]
    ser = dup.bfill(axis=1).iloc[:, 0]
    df = df.drop(columns=[name])
    df[name] = ser
    return df

# ---------- Persisted results (so both download buttons stick around) ----------
if "results" not in st.session_state:
    st.session_state.results = None  # dict with cleaned_bytes, missing_bytes, cleaned_head, summary_md, ts

# ---------- Uploads ----------
st.sidebar.header("Upload files")
members_file = st.sidebar.file_uploader("Member list (.xlsx)", type=["xlsx"])
answer_file  = st.sidebar.file_uploader("Specialty answer list (.xlsx)", type=["xlsx"])

members_df = None
answer_df  = None

if members_file:
    try:
        xls_m = pd.ExcelFile(members_file)
        st.sidebar.subheader("Members file options")
        m_sheet = st.sidebar.selectbox("Members sheet", xls_m.sheet_names, index=0, key="m_sheet")
        m_skip  = st.sidebar.number_input("Rows to skip before header (members)",
                                          min_value=0, max_value=200, value=0, step=1, key="m_skip")
        members_df = read_excel_with_options(members_file, sheet_name=m_sheet, skiprows=m_skip)
    except Exception as e:
        st.sidebar.error(f"Members file error: {e}")

if answer_file:
    try:
        xls_a = pd.ExcelFile(answer_file)
        st.sidebar.subheader("Answer list options")
        a_sheet = st.sidebar.selectbox("Answer list sheet", xls_a.sheet_names, index=0, key="a_sheet")
        a_skip  = st.sidebar.number_input("Rows to skip before header (answer list)",
                                          min_value=0, max_value=200, value=0, step=1, key="a_skip")
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

    # Auto-detect columns
    norm_map = _normalize_cols(members_df.columns)

    email_guess = _guess_with_exclusions(norm_map, [r'\bemail\b', r'\bprimary\s*email\b', r'\bemail\s*address\b'])
    start_guess = _guess_with_exclusions(norm_map, [r'\bmember\s*date\b', r'\bstart\b'])
    end_guess   = _guess_with_exclusions(norm_map, [r'\bexpiration\b|\bexpire\b|\bend\s*date\b'])
    zip_guess   = _guess_with_exclusions(norm_map, [r'\bzip\b|\bpostal\b'])

    # Prefer "member type" / "category" over a bare "code", and exclude any with "zip" or "postal"
    code_guess = (
        _guess_with_exclusions(norm_map, [r'\bmember\s*type\b', r'\bcategory\b'], exclude_contains=["zip", "postal"])
        or _guess_with_exclusions(norm_map, [r'\bcode\b'], exclude_contains=["zip", "postal"])
    )

    addr_type_guess = _guess_with_exclusions(norm_map, [r'\baddress\s*type\b'])
    addr_type_desc_guess = _guess_with_exclusions(norm_map, [r'\baddress\s*type\s*description\b|\baddress\s*desc'])

    st.sidebar.header("Column mapping (auto-detected; you can override)")
    def idx(opt_list, val): return 0 if val is None else opt_list.index(val)
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
    members_specialty_col = st.sidebar.selectbox(
        "Members column to MATCH against Answer List",
        options=opts, index=idx(opts, members_specialty_default)
    )
    if len(answer_df.columns) < 2:
        st.error("Answer list must have at least two columns: one to return, one to match on.")
        st.stop()
    answer_return_col = st.sidebar.selectbox("Answer list column to RETURN", options=list(answer_df.columns), index=0)
    answer_match_col  = st.sidebar.selectbox("Answer list column to MATCH ON", options=list(answer_df.columns),
                                             index=min(1, len(answer_df.columns)-1))
# === Email Enrichment (beta) — place just BEFORE the divider that precedes "Run Cleaning" ===
with st.sidebar.expander("🔎 Email Enrichment (beta)", expanded=False):
    if "members_df" in locals() and members_df is not None:
        # Work on a view with canonical 'Email' column, regardless of mapped name
        view_df = members_df.copy()
        try:
            if 'email_col' in locals() and email_col and email_col != "Email" and email_col in view_df.columns:
                view_df["Email"] = view_df[email_col]
            elif "Email" not in view_df.columns:
                view_df["Email"] = ""
        except Exception:
            # Ensure Email column exists even if mapping not set yet
            if "Email" not in view_df.columns:
                view_df["Email"] = ""

        # Launch enrichment UI (requires SerpAPI or Google CSE secrets)
        email_enrichment_sidebar(view_df)

        # Sync enriched emails back to whichever column the user mapped as Email
        if st.button("↩️ Sync enriched emails to selected Email column"):
            target_col = email_col if ('email_col' in locals() and email_col) else "Email"
            if target_col not in members_df.columns:
                members_df[target_col] = ""
            # Copy back
            if "Email" in view_df.columns:
                members_df[target_col] = view_df["Email"]
                st.success(f"Synced enriched emails into '{target_col}'. You can now click 'Run Cleaning'.")
            else:
                st.warning("No 'Email' column present in enrichment view.")
    else:
        st.info("Upload your Members file first to enable enrichment.")
# === End Email Enrichment (beta) ===


    st.divider()
    if st.button("Run Cleaning", type="primary"):
        df = members_df.copy()
        log = {}

        # Canonical headers (note: 'code' is lowercase)
        if email_col: df.rename(columns={email_col: "Email"}, inplace=True)
        if start_col: df.rename(columns={start_col: "Start Date"}, inplace=True)
        if end_col:   df.rename(columns={end_col: "End Date"}, inplace=True)
        if zip_col:   df.rename(columns={zip_col: "Zip code"}, inplace=True)
        if code_col:  df.rename(columns={code_col: "code"}, inplace=True)
        if addr_type_col: df.rename(columns={addr_type_col: "Address type"}, inplace=True)
        if addr_type_desc_col: df.rename(columns={addr_type_desc_col: "Address type description"}, inplace=True)

        # Coalesce any duplicates created by mapping
        dupe_counts = pd.Series(df.columns).value_counts()
        dupes = dupe_counts[dupe_counts > 1]
        if not dupes.empty:
            st.warning(f"Found duplicate columns after mapping: {', '.join(dupes.index.tolist())}. Coalescing.")
            for name in dupes.index.tolist():
                df = coalesce_duplicate_named_column(df, name)

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

        # Lowercase code
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.lower()

        # Specialty mapping
        spec_mapped = 0
        if members_specialty_col and (members_specialty_col in df.columns):
            ans = answer_df.dropna(subset=[answer_match_col]).copy()
            mapper = ans.set_index(answer_match_col)[answer_return_col]
            df["Specialty Description - 1"] = df[members_specialty_col].map(mapper)
            spec_mapped = df["Specialty Description - 1"].notna().sum()
        log["Specialties mapped"] = spec_mapped

        # End Date = 2099-12-31 if code starts with 'l'
        if "code" in df.columns and "End Date" in df.columns:
            mask_l = df["code"].astype(str).str.startswith("l", na=False)
            count_l = int(mask_l.sum())
            df.loc[mask_l, "End Date"] = pd.Timestamp("2099-12-31")
            log["End dates set to 2099 (code=l*)"] = count_l

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

        # Build outputs and store in session_state (persist after downloads)
        cleaned_bytes = to_xlsx_bytes(cleaned_df)
        missing_bytes = to_xlsx_bytes(missing_df)
        summary_md = f"""
**Rows total:** {len(df)}  
**Rows with email (cleaned):** {len(cleaned_df)}  
**Rows missing email:** {len(missing_df)}  
**Start dates backfilled:** {log.get("Start dates backfilled", 0)}  
**End dates → 2099 (code starts with 'l'):** {log.get("End dates set to 2099 (code=l*)", 0)}  
**Specialties mapped:** {log.get("Specialties mapped", 0)}  
**ZIPs normalized (had a value):** {log.get("ZIP normalized", 0)}
"""
        st.session_state.results = dict(
            cleaned_bytes=cleaned_bytes,
            missing_bytes=missing_bytes,
            cleaned_head=cleaned_df.head(200),
            summary_md=summary_md,
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

# --- Always show persisted results, even if uploaders reset after a download ---
if st.session_state.results:
    st.success(f"Processing complete. (Generated {st.session_state.results['ts']})")
    st.markdown(st.session_state.results["summary_md"])
    st.markdown("**Cleaned Output Preview (first 200)**")
    st.dataframe(st.session_state.results["cleaned_head"], use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Cleaned_Members_for_GlueUp.xlsx",
            data=st.session_state.results["cleaned_bytes"],
            file_name="Cleaned_Members_for_GlueUp.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_cleaned"
        )
    with d2:
        st.download_button(
            "Download Members_Missing_Emails.xlsx",
            data=st.session_state.results["missing_bytes"],
            file_name="Members_Missing_Emails.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_missing"
        )
else:
    st.info("Upload both files to begin. Then choose the correct sheet(s) in the sidebar.")

import os, io, pandas as pd, streamlit as st
from email_enrichment_util import email_enrichment_sidebar, apply_email_enrichment_results

st.set_page_config(page_title="Utilities • Email Search (Standalone)", layout="wide")

st.title("Utilities: Email Search (Standalone)")
st.caption("Upload any CSV/XLSX, enrich emails from public sources, and download the enriched file — no need to run the main cleaner.")

uploaded = st.file_uploader("Upload list to enrich (Excel .xlsx or CSV .csv)", type=["xlsx","csv"])
df = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            sheet = 0
            try:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Choose sheet", xls.sheet_names, index=0)
                skip = st.number_input("Rows to skip before header", min_value=0, max_value=200, value=0, step=1)
                df = pd.read_excel(uploaded, sheet_name=sheet, skiprows=int(skip))
            except Exception as e:
                st.error(f"Excel read error: {e}")
                df = None
    except Exception as e:
        st.error(f"File read error: {e}")
        df = None

if df is not None and not df.empty:
    st.subheader("Preview")
    st.dataframe(df.head(200), use_container_width=True)

    st.subheader("Map required columns")
    opts = [None] + list(df.columns)

    def pick_guess(parts):
        for c in df.columns:
            s = str(c).lower().replace("_"," ").strip()
            if any(p in s for p in parts):
                return c
        return None

    first_col  = st.selectbox("First name", options=opts, index=(opts.index(pick_guess(['first'])) if pick_guess(['first']) in opts else 0))
    last_col   = st.selectbox("Last name",  options=opts, index=(opts.index(pick_guess(['last'])) if pick_guess(['last']) in opts else 0))
    prac_col   = st.selectbox("Practice Name", options=opts, index=(opts.index(pick_guess(['practice','group','clinic','hospital'])) if pick_guess(['practice','group','clinic','hospital']) in opts else 0))
    city_col   = st.selectbox("City", options=opts, index=(opts.index(pick_guess(['city'])) if pick_guess(['city']) in opts else 0))
    state_col  = st.selectbox("State", options=opts, index=(opts.index(pick_guess(['state','province'])) if pick_guess(['state','province']) in opts else 0))
    email_col  = st.selectbox("Email (if exists)", options=opts, index=(opts.index(pick_guess(['email','e-mail'])) if pick_guess(['email','e-mail']) in opts else 0))

    work = df.copy()
    alias = {}
    if first_col: alias[first_col] = "First name"
    if last_col:  alias[last_col]  = "Last name"
    if prac_col:  alias[prac_col]  = "Practice Name"
    if city_col:  alias[city_col]  = "City"
    if state_col: alias[state_col] = "State"
    if email_col: alias[email_col] = "Email"
    if alias:
        work.rename(columns=alias, inplace=True)
    if "Email" not in work.columns:
        work["Email"] = ""

    # Sidebar enrichment tool
    with st.sidebar.expander("🔎 Email Enrichment (beta)", expanded=False):
        email_enrichment_sidebar(work)

    st.info("After running the search in the sidebar and approving rows, click Apply to write results into your data.")

    if st.button("✅ Apply approved (from sidebar)", use_container_width=False):
        applied, msg = apply_email_enrichment_results(work, overwrite=False)
        if applied:
            st.success(f"Applied {applied} emails to the dataset.")
        else:
            st.warning(msg)

    st.subheader("Download")
    c1, c2 = st.columns([1,1])
    with c1:
        csv_bytes = work.to_csv(index=False).encode("utf-8")
        st.download_button("Download Enriched CSV", data=csv_bytes, file_name="enriched_members.csv", mime="text/csv")
    with c2:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            work.to_excel(writer, index=False, sheet_name="Enriched")
        st.download_button("Download Enriched Excel", data=bio.getvalue(), file_name="enriched_members.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload a CSV/XLSX to get started.")


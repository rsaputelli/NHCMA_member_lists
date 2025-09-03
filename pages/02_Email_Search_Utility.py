
# pages/02_Email_Search_Utility.py
# Standalone utility page for email enrichment (no dependency on main cleaning flow)
import os
import io
import pandas as pd
import streamlit as st
from email_enrichment_util import email_enrichment_sidebar

st.set_page_config(page_title="Utilities • Email Search (Standalone)", layout="wide")

# --- Branding (reuse logo if present) ---
LOGO_PATH = "logo.png"
hdr_l, hdr_r = st.columns([1, 8])
with hdr_l:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
with hdr_r:
    st.title("Utilities: Email Search (Standalone)")
    st.caption("Find public, professional emails for physicians/practices from a file you upload. No need to run the main cleaner.")

st.markdown("""
**How to use**
1. Upload the list you want to enrich (Excel or CSV).
2. Map the required columns on the right.
3. Open **🔎 Email Enrichment (beta)** in the sidebar, run search, approve, then **↩️ Sync**.
4. Download the enriched file here.
""")

st.divider()

# --- Uploads ---
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Upload list to enrich (Excel .xlsx or CSV .csv)", type=["xlsx", "csv"])

df = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            # Excel: allow sheet + skiprows selection
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
    st.markdown("**Preview (first 200 rows)**")
    st.dataframe(df.head(200), use_container_width=True)

    # --- Column mapping (standalone) ---
    st.subheader("Map required columns")
    opts = [None] + list(df.columns)
    def idx(opt_list, val): 
        try:
            return opt_list.index(val)
        except Exception:
            return 0

    # Quick guesses
    def guess(name_parts):
        for c in df.columns:
            lc = str(c).lower().replace("_"," ").strip()
            if any(p in lc for p in name_parts):
                return c
        return None

    first_col  = st.selectbox("First name", options=opts, index=idx(opts, guess(["first"])))
    last_col   = st.selectbox("Last name", options=opts, index=idx(opts, guess(["last"])))
    prac_col   = st.selectbox("Practice Name", options=opts, index=idx(opts, guess(["practice","group","clinic","hospital"])))
    city_col   = st.selectbox("City", options=opts, index=idx(opts, guess(["city"])))
    state_col  = st.selectbox("State", options=opts, index=idx(opts, guess(["state","province"])))
    email_col  = st.selectbox("Email (if exists)", options=opts, index=idx(opts, guess(["email","e-mail","e mail"])))

    # Create a working copy and normalize mappings
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

    # Ensure Email column exists
    if "Email" not in work.columns:
        work["Email"] = ""

    # --- Sidebar enrichment tool (standalone) ---
    with st.sidebar.expander("🔎 Email Enrichment (beta)", expanded=False):
        email_enrichment_sidebar(work)

    # --- Downloads ---
    st.divider()
    st.subheader("Download")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        csv_bytes = work.to_csv(index=False).encode("utf-8")
        st.download_button("Download Enriched CSV", data=csv_bytes, file_name="enriched_members.csv", mime="text/csv")
    with c2:
        # Excel writer to bytes
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            work.to_excel(writer, index=False, sheet_name="Enriched")
        st.download_button("Download Enriched Excel", data=bio.getvalue(), file_name="enriched_members.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Upload a CSV/XLSX to get started.")

# Footer note
st.caption("This utility is standalone and does not require the main cleaning workflow. It uses only public, professional emails.")

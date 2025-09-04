# Utilities • Email Search (Standalone)
# - Persists uploaded dataset across reruns in st.session_state["_master_df"]
# - Runs email_enrichment_sidebar(df) on the SAME DataFrame instance
# - After sidebar "Apply approved", changes persist (we save df back to session)
# - Optional: buttons to load last saved enriched CSV/XLSX written by the util's
#             "Persist full dataset to disk on Apply" toggle

import streamlit as st
st.set_page_config(page_title="Utilities • Email Search (Standalone)", layout="wide")

import os
import io
import pandas as pd
from io import BytesIO
from email_enrichment_util import email_enrichment_sidebar  # apply handled inside the util

st.title("Utilities: Email Search (Standalone)")
st.caption("Upload any CSV/XLSX, map columns, run the email search in the sidebar, approve, apply, and download the enriched file.")

# ---------------------------
# Upload + session persistence
# ---------------------------

def _load_df_from_upload(uploaded):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(data))
    return pd.read_csv(BytesIO(data))

uploaded = st.file_uploader("Upload list to enrich (Excel .xlsx or CSV .csv)", type=["xlsx", "csv"])
if uploaded and st.session_state.get("_uploaded_token") != uploaded.name:
    st.session_state["_uploaded_token"] = uploaded.name
    st.session_state["_uploaded_bytes"] = uploaded.getvalue()
    st.session_state["_master_df"] = _load_df_from_upload(uploaded)

if "_master_df" not in st.session_state or st.session_state["_master_df"] is None:
    st.info("Upload a CSV/XLSX to get started.")
    st.stop()

df = st.session_state["_master_df"]

# --------
# Preview
# --------
st.subheader("Preview")
st.dataframe(df.head(200), use_container_width=True)

# ---------------------
# Column mapping (light)
# ---------------------
opts = [None] + list(df.columns)

def _pick_guess(parts):
    for c in df.columns:
        s = str(c).lower().replace("_", " ").strip()
        if any(p in s for p in parts):
            return c
    return None

first_col  = st.selectbox("First name",       options=opts, index=(opts.index(_pick_guess(['first'])) if _pick_guess(['first']) in opts else 0))
last_col   = st.selectbox("Last name",        options=opts, index=(opts.index(_pick_guess(['last'])) if _pick_guess(['last']) in opts else 0))
prac_col   = st.selectbox("Practice Name",    options=opts, index=(opts.index(_pick_guess(['practice','group','clinic','hospital'])) if _pick_guess(['practice','group','clinic','hospital']) in opts else 0))
city_col   = st.selectbox("City",             options=opts, index=(opts.index(_pick_guess(['city'])) if _pick_guess(['city']) in opts else 0))
state_col  = st.selectbox("State",            options=opts, index=(opts.index(_pick_guess(['state','province'])) if _pick_guess(['state','province']) in opts else 0))
email_col  = st.selectbox("Email (if exists)",options=opts, index=(opts.index(_pick_guess(['email','e-mail'])) if _pick_guess(['email','e-mail']) in opts else 0))

# Work on the SAME DataFrame instance
work = df

# Normalize headers expected by the enrichment util
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

st.divider()

# ----------------------------
# Sidebar: Email enrichment UI
# ----------------------------
with st.sidebar.expander("🔎 Email Enrichment", expanded=True):
    email_enrichment_sidebar(work)
st.info("After running the search in the sidebar and approving rows, click **Apply** in the results area (or sidebar).")

# Persist any in-place edits made by the util’s “Apply approved”
st.session_state["_master_df"] = work

# ------------------------------------------------------------
# Optional: reload last saved enriched dataset written on Apply
# (only useful if you turned on “Persist full dataset to disk on Apply”)
# ------------------------------------------------------------
base = st.session_state.get("_persist_dataset_name", "enriched_dataset")
c1, c2 = st.columns(2)
with c1:
    if st.button("Load last saved enriched CSV"):
        path = f"{base}.csv"
        if os.path.exists(path):
            st.session_state["_master_df"] = pd.read_csv(path)
            st.success(f"Loaded {path}")
        else:
            st.warning(f"{path} not found.")
with c2:
    if st.button("Load last saved enriched Excel"):
        path = f"{base}.xlsx"
        if os.path.exists(path):
            st.session_state["_master_df"] = pd.read_excel(path)
            st.success(f"Loaded {path}")
        else:
            st.warning(f"{path} not found.")

st.divider()

# -------------------
# Manual downloads too
# -------------------
st.subheader("Download (current in-memory dataset)")
c1, c2 = st.columns(2)
with c1:
    csv_bytes = work.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Enriched CSV",
        data=csv_bytes,
        file_name="enriched_members.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        work.to_excel(writer, index=False, sheet_name="Enriched")
    st.download_button(
        "Download Enriched Excel",
        data=bio.getvalue(),
        file_name="enriched_members.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )



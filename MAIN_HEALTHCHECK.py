import streamlit as st, sys, platform, pandas, requests, bs4, tldextract, rapidfuzz, openpyxl, xlsxwriter
st.set_page_config(page_title="Healthcheck", layout="wide")
st.title("Streamlit Cloud Healthcheck")
st.write("✅ App boot OK")
st.json({
  "python": sys.version,
  "platform": platform.platform(),
  "streamlit": st.__version__,
  "pandas": pandas.__version__,
  "requests": requests.__version__,
  "bs4": bs4.__version__,
  "tldextract": tldextract.__version__,
  "rapidfuzz": rapidfuzz.__version__,
  "openpyxl": openpyxl.__version__,
})
st.success("If you can see this, the environment and dependencies are installed correctly.")

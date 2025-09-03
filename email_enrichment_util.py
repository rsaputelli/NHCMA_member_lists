# email_enrichment_util.py
# Drop-in sidebar utility for finding public, professional emails for physicians/practices.
# - Supports SerpAPI OR Google Programmable Search (CSE)
# - Optional free-tier cap for CSE (default 100/day, ET)
# - Writes back suggested emails into your DataFrame after manual review
#
# Dependencies (add to requirements.txt if needed):
#   requests, beautifulsoup4, tldextract, rapidfuzz

import os
import re
import time
import json
import requests
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from bs4 import BeautifulSoup
import tldextract
from rapidfuzz import fuzz

EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
GENERIC_LOCALPART = {"info","contact","office","frontdesk","help","support","admin","billing","media","press","jobs","hr","recruit","webmaster"}

# --- Daily quota helpers (ET) for Google CSE only ---
USAGE_PATH = Path(".email_enrich_usage.json")  # lightweight local persistence

def _today_et_str():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

def _load_usage():
    today = _today_et_str()
    usage = {"date": today, "count": 0}
    try:
        if USAGE_PATH.exists():
            data = json.loads(USAGE_PATH.read_text())
            if data.get("date") == today:
                usage = data
    except Exception:
        pass
    return usage

def _save_usage(usage: dict):
    try:
        USAGE_PATH.write_text(json.dumps(usage))
    except Exception:
        pass

def _get_free_cap():
    try:
        return int(st.secrets.get("FREE_TIER_DAILY_LIMIT", 100))
    except Exception:
        return 100

# --- Search backends ---
def _search_serpapi(query: str, num: int = 5) -> List[Dict]:
    key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
    if not key:
        return []
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": num, "api_key": key}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in (data.get("organic_results") or [])[:num]:
            link = item.get("link") or ""
            title = item.get("title") or ""
            if link:
                out.append({"title": title, "url": link})
        return out
    except Exception:
        return []

def _search_google_cse(query: str, num: int = 5) -> List[Dict]:
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    cse_id  = st.secrets.get("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CSE_ID")
    if not (api_key and cse_id):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query, "num": min(num, 10)}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("items", [])[:num]:
            link = item.get("link") or ""
            title = item.get("title") or ""
            if link:
                out.append({"title": title, "url": link})
        return out
    except Exception:
        return []

def _search_web(provider: str, query: str, num: int = 5) -> List[Dict]:
    """provider: 'Auto', 'SerpAPI', 'Google CSE'"""
    if provider == "SerpAPI":
        return _search_serpapi(query, num=num) or []
    if provider == "Google CSE":
        return _search_google_cse(query, num=num) or []
    # Auto preference: SerpAPI -> CSE
    hits = _search_serpapi(query, num=num)
    if hits:
        return hits
    return _search_google_cse(query, num=num)

# --- Scraping / heuristics ---
def _fetch_html(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def _extract_emails_from_html(html: str) -> List[str]:
    return list(set(EMAIL_REGEX.findall(html)))

def _domain(url: str) -> str:
    ext = tldextract.extract(url)
    domain = ".".join(part for part in [ext.domain, ext.suffix] if part)
    return domain.lower()

def _score_email(email: str, first: str, last: str, practice_domain: str) -> Tuple[int, str]:
    local, _, domain = email.lower().partition("@")
    score = 0
    typ = "unknown"

    # Domain match bonus
    if practice_domain and domain == practice_domain:
        score += 20

    # Name similarity (favor last name + first initial combos)
    if last and last.lower() in local:
        score += 15
    if first:
        fi = first[0].lower()
        if fi and (fi in local or f"{fi}{last.lower()}" in local or f"{fi}.{last.lower()}" in local):
            score += 8

    # Penalties / bonuses
    if local in GENERIC_LOCALPART:
        score -= 4
        typ = "practice inbox"
    if any(bad in local for bad in ["billing","media","press","jobs","hr","recruit"]):
        score -= 6
        typ = "non-clinical inbox"

    if typ == "unknown":
        if last and last.lower() in local:
            typ = "likely direct"
        elif local in GENERIC_LOCALPART:
            typ = "practice inbox"
        else:
            typ = "other"

    return score, typ

def _best_guess_practice_domain(candidates: List[Dict]) -> str:
    for hit in candidates[:3]:
        try:
            return _domain(hit["url"])
        except Exception:
            continue
    return ""

def _build_query(row: dict) -> str:
    first = str(row.get("First name","")).strip()
    last  = str(row.get("Last name","")).strip()
    prac  = str(row.get("Practice Name","")).strip()
    city  = str(row.get("City","")).strip()
    state = str(row.get("State","")).strip()
    base  = " ".join([x for x in [first, last, prac, city, state, "physician email contact"] if x])
    return base + " site:org OR site:com OR site:edu"

def _enrich_single(row: dict, provider: str) -> Dict:
    first = str(row.get("First name","")).strip()
    last  = str(row.get("Last name","")).strip()
    prac  = str(row.get("Practice Name","")).strip()

    query = _build_query(row)
    hits = _search_web(provider, query, num=6)
    if not hits:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"No search results"}

    # Try top N pages and common sibling paths
    candidates = []
    for h in hits[:6]:
        candidates.append(h["url"])
        base = h["url"].rstrip("/")
        for suffix in ("/contact", "/contact-us", "/team", "/providers", "/physicians", "/about"):
            candidates.append(base + suffix)

    emails_found = []
    scanned = set()
    for url in candidates:
        if url in scanned:
            continue
        scanned.add(url)
        html = _fetch_html(url)
        if not html:
            continue
        for email in _extract_emails_from_html(html):
            emails_found.append((email, url))
        time.sleep(0.25)

    practice_domain = _best_guess_practice_domain(hits) if prac else ""

    ranked = []
    for email, src in set(emails_found):
        score, typ = _score_email(email, first, last, practice_domain)
        ranked.append({"email": email, "score": score, "type": typ, "src": src})

    if not ranked:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":hits[0]["url"], "Notes":"No emails on scanned pages"}

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[0]
    confidence = max(0, min(100, 40 + top["score"]))

    return {
        "Found Email": top["email"],
        "Email Type": top["type"],
        "Confidence": confidence,
        "Source URL": top["src"],
        "Notes": f"Practice domain guess: {practice_domain}" if practice_domain else ""
    }

def email_enrichment_sidebar(df):
    """
    Sidebar UI: run targeted searches, review results, apply approved emails into df['Email'].
    Expects df to have columns: First name, Last name, Practice Name, City, State, Email
    """

    st.caption("Searches public web pages for practice/provider emails. Review suggested emails before applying.")

    # Detect configured providers
    serp_key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
    g_api    = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    g_cx     = st.secrets.get("GOOGLE_CSE_ID")  or os.getenv("GOOGLE_CSE_ID")

    available = []
    if serp_key:
        available.append("SerpAPI")
    if g_api and g_cx:
        available.append("Google CSE")
    if not available:
        st.error("No search API configured. Add SERPAPI_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID to Streamlit secrets.")
        return

    provider = st.selectbox("Search provider", ["Auto"] + available, index=0,
                            help="Auto = prefers SerpAPI (if present), else Google CSE.")

    using_cse = (provider == "Google CSE") or (provider == "Auto" and not serp_key and (g_api and g_cx))
    enforce_free = False
    remaining = None
    if using_cse:
        st.info("Using Google CSE • Free tier is 100 queries/day. Capped by default.")
        enforce_free = st.toggle("Never exceed free tier (100/day)", value=True,
                                 help="Disable to allow paid overage after 100 daily queries.")
        usage = _load_usage()
        cap   = _get_free_cap()
        remaining = max(0, cap - int(usage.get("count", 0)))
        st.write(f"**Remaining today (ET): {remaining} / {cap}**")

    # Column presence
    needed = ["First name","Last name","Practice Name","City","State","Email"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        return

    overwrite = st.checkbox("Overwrite existing Email values", value=False)

    # Candidate rows
    if overwrite:
        candidates_df = df.copy()
    else:
        candidates_df = df[df["Email"].isna() | (df["Email"].astype(str).str.strip() == "")]

    default_limit = min(25, len(candidates_df))
    max_limit = len(candidates_df)

    if using_cse and enforce_free:
        usage = _load_usage()
        cap   = _get_free_cap()
        remaining = max(0, cap - int(usage.get("count", 0)))
        if remaining == 0:
            st.warning("Daily free CSE quota reached. Try again tomorrow or disable the cap to allow paid overage.")
        max_limit = min(max_limit, remaining)

    limit = st.number_input("Max records to scan", min_value=0, max_value=int(max_limit), value=int(min(default_limit, max_limit)))

    conf_thresh = st.slider("Auto-approve at confidence ≥", 0, 100, 70,
                            help="You can still uncheck individual rows.")

    if st.button("Run email search", disabled=(max_limit == 0)):
        total = int(limit)
        if total <= 0:
            st.info("Nothing to process with current settings.")
            return

        subset = candidates_df.head(total).copy()
        progress = st.progress(0.0)
        results = []
        for i, (_, row) in enumerate(subset.iterrows(), start=1):
            res = _enrich_single(row.to_dict(), provider)
            rec = {
                "First name": row["First name"],
                "Last name": row["Last name"],
                "Practice Name": row.get("Practice Name",""),
                "City": row.get("City",""),
                "State": row.get("State",""),
                "Suggested Email": res.get("Found Email",""),
                "Type": res.get("Email Type",""),
                "Confidence": int(res.get("Confidence",0)),
                "Source URL": res.get("Source URL",""),
                "Notes": res.get("Notes",""),
                "Approve": bool(res.get("Found Email") and int(res.get("Confidence",0)) >= conf_thresh),
            }
            results.append(rec)
            progress.progress(i/total)

        if not results:
            st.info("No results.")
            return

        edited = st.data_editor(
            results,
            use_container_width=True,
            num_rows="fixed",
            key="email_enrich_editor"
        )

        st.write("Tip: uncheck **Approve** for generic inboxes you don’t want to use.")

        # Apply button
        if st.button("Apply approved emails to Data"):
            applied = 0
            for rec in edited:
                if not rec.get("Approve") or not rec.get("Suggested Email"):
                    continue
                mask = (
                    (df["First name"].astype(str).str.strip() == str(rec["First name"]).strip()) &
                    (df["Last name"].astype(str).str.strip() == str(rec["Last name"]).strip())
                )
                if "Practice Name" in df.columns and str(rec.get("Practice Name","")).strip():
                    mask &= (df["Practice Name"].fillna("").astype(str).str.strip() == str(rec["Practice Name"]).strip())

                if overwrite:
                    df.loc[mask, "Email"] = rec["Suggested Email"]
                    applied += int(mask.sum())
                else:
                    blanks = mask & (df["Email"].isna() | (df["Email"].astype(str).str.strip() == ""))
                    df.loc[blanks, "Email"] = rec["Suggested Email"]
                    applied += int(blanks.sum())

            st.success(f"Applied {applied} emails into the dataset.")

            # Download delta for audit
            try:
                import pandas as pd
                delta = pd.DataFrame(edited)
                st.download_button(
                    "Download enrichment results (CSV)",
                    delta.to_csv(index=False).encode("utf-8"),
                    file_name="email_enrichment_results.csv",
                    mime="text/csv",
                )
            except Exception:
                pass

            # Update usage tracker for CSE if enforced
            if using_cse and enforce_free and total:
                usage = _load_usage()
                if usage.get("date") != _today_et_str():
                    usage = {"date": _today_et_str(), "count": 0}
                usage["count"] = int(usage.get("count", 0)) + total
                usage["count"] = min(usage["count"], _get_free_cap())
                _save_usage(usage)

    with st.expander("Setup & Notes"):
        st.markdown("""
- **Data used**: only publicly visible, professional emails from practice/provider sites or reputable directories.
- **Privacy**: do not add personal emails; keep to practice/institutional or clearly professional addresses.
- **API**: configure **SerpAPI** (`SERPAPI_KEY`) or **Google CSE** (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`) in Streamlit **Secrets**.
- **Confidence**: higher when the email domain matches the practice and the local-part includes the physician's last name.
- **Caveats**: Large health systems often hide emails behind forms; you may only get a practice inbox.
        """)


# email_enrichment_util.py (quality-first + external apply helper)
import os, re, time, json, html, requests
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from bs4 import BeautifulSoup
import tldextract
from rapidfuzz import fuzz, process

# --- Regexes ---
EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
OBFUSCATED_REGEXES = [
    re.compile(r"([A-Z0-9._%+-]+)\s*(?:\(|\[)?\s*(?:at|@)\s*(?:\)|\])?\s*([A-Z0-9.-]+)\s*(?:\(|\[)?\s*(?:dot|\.)\s*(?:\)|\])?\s*([A-Z]{2,})", re.I),
    re.compile(r"([A-Z0-9._%+-]+)\s*(?:\(|\[)?\s*(?:at|@)\s*(?:\)|\])?\s*([A-Z0-9.-]+\.[A-Z]{2,})", re.I),
]
GENERIC_LOCALPART = {"info","contact","office","frontdesk","help","support","admin","billing","media","press","jobs","hr","recruit","webmaster","noreply","no-reply","donotreply","do-not-reply"}

# --- Daily quota helpers (ET) for Google CSE only ---
USAGE_PATH = Path(".email_enrich_usage.json")

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
    if provider == "SerpAPI":
        return _search_serpapi(query, num=num) or []
    if provider == "Google CSE":
        return _search_google_cse(query, num=num) or []
    hits = _search_serpapi(query, num=num)
    if hits:
        return hits
    return _search_google_cse(query, num=num)

# --- Fetching / crawling ---
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}

def _fetch_html(url: str, timeout=20) -> str:
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        r.raise_for_status()
        ct = r.headers.get("Content-Type","").lower()
        if "text" in ct or "html" in ct or "xml" in ct or "json" in ct:
            return r.text
        try:
            return r.content.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    except Exception:
        return ""

def _fetch_html_parallel(urls, timeout=20, max_workers=6):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_html, u, timeout): u for u in urls}
        for f in as_completed(futs):
            u = futs[f]
            try:
                html = f.result()
            except Exception:
                html = ""
            results.append((u, html))
    return results

def _same_domain(url_a: str, url_b: str) -> bool:
    try:
        a = urlparse(url_a).netloc.split(":")[0].lower()
        b = urlparse(url_b).netloc.split(":")[0].lower()
        return a == b
    except Exception:
        return False

def _collect_internal_links(base_url: str, html_text: str, limit=30) -> List[str]:
    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception:
        return []
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        if _same_domain(base_url, full):
            links.append(full)
        if len(links) >= limit:
            break
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# --- Extraction / scoring ---
def _extract_emails_from_html(html_text: str) -> List[str]:
    emails = set(EMAIL_REGEX.findall(html_text))
    for rx in OBFUSCATED_REGEXES:
        for match in rx.findall(html_text):
            if len(match) == 3:
                local, domain, tld = match
                emails.add(f"{local}@{domain}.{tld}")
            elif len(match) == 2:
                local, domain = match
                emails.add(f"{local}@{domain}")
    return list(emails)

def _domain(url: str) -> str:
    ext = tldextract.extract(url)
    domain = ".".join(part for part in [ext.domain, ext.suffix] if part)
    return domain.lower()

def _score_email(email: str, first: str, last: str, practice_domain: str) -> Tuple[int, str]:
    local, _, domain = email.lower().partition("@")
    score = 0
    typ = "unknown"
    if practice_domain and domain == practice_domain:
        score += 24
    lname = (last or "").lower()
    fname = (first or "").lower()
    if lname and lname in local:
        score += 18
    if fname:
        fi = fname[:1]
        for p in {f"{fi}{lname}", f"{fi}.{lname}", f"{fname}.{lname}", f"{fname}{lname}"}:
            if p in local:
                score += 6
    if local in GENERIC_LOCALPART:
        score -= 6
        typ = "practice inbox"
    if any(bad in local for bad in ["billing","media","press","jobs","hr","recruit","marketing","career","privacy","webmaster","noreply","no-reply","donotreply"]):
        score -= 8
        typ = "non-clinical inbox"
    if typ == "unknown":
        if lname and lname in local:
            typ = "likely direct"
        elif local in GENERIC_LOCALPART:
            typ = "practice inbox"
        else:
            typ = "other"
    return score, typ

def _best_guess_practice_domain(candidates: List[Dict]) -> str:
    for hit in candidates[:5]:
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
    parts = [first, last, prac, city, state, "physician email contact"]
    base  = " ".join([x for x in parts if x])
    return base + " site:org OR site:com OR site:edu"

def _viable_page(url: str) -> bool:
    bad = ["privacy","terms","login","careers","jobs","press","media","marketing","search?","sso","account"]
    path = urlparse(url).path.lower()
    return not any(b in path for b in bad)

def _enrich_single(row: dict, provider: str, mode: str = "Thorough") -> Dict:
    if mode == "Thorough":
        search_num = 10
        suffixes = ("/contact","/contact-us","/team","/providers","/physicians","/about","/faculty","/locations")
        parallel = 8
        timeout = 18
        internal_limit = 40
    elif mode == "Fast":
        search_num = 3
        suffixes = ("/contact","/team","/providers")
        parallel = 8
        timeout = 8
        internal_limit = 12
    else:  # Balanced
        search_num = 6
        suffixes = ("/contact","/contact-us","/team","/providers","/physicians")
        parallel = 6
        timeout = 12
        internal_limit = 20

    first = str(row.get("First name","")).strip()
    last  = str(row.get("Last name","")).strip()
    prac  = str(row.get("Practice Name","")).strip()

    query = _build_query(row)
    hits = _search_web(provider, query, num=search_num)
    if not hits:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"No search results"}

    candidates = []
    for h in hits:
        base = h["url"].rstrip("/")
        if _viable_page(base):
            candidates.append(base)
        for s in suffixes:
            u = base + s
            if _viable_page(u):
                candidates.append(u)

    candidates = list(dict.fromkeys(candidates))
    fetched = _fetch_html_parallel(candidates, timeout=timeout, max_workers=parallel)

    if mode in ("Thorough","Balanced"):
        more = []
        for url, html_text in fetched:
            if not html_text:
                continue
            more += [u for u in _collect_internal_links(url, html_text, limit=internal_limit) if _viable_page(u)]
        more = list(dict.fromkeys(more))
        fetched += _fetch_html_parallel(more, timeout=timeout, max_workers=parallel)

    emails_found = []
    for url, html_text in fetched:
        if not html_text:
            continue
        text = html.unescape(html_text)
        for email in _extract_emails_from_html(text):
            emails_found.append((email, url))

    practice_domain = _best_guess_practice_domain(hits) if prac else ""

    ranked = []
    for email, src in set(emails_found):
        score, typ = _score_email(email, first, last, practice_domain)
        ranked.append({"email": email, "score": score, "type": typ, "src": src})

    if not ranked:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":hits[0]["url"], "Notes":"No emails on scanned pages"}

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[0]
    confidence = max(0, min(100, 42 + top["score"]))
    return {
        "Found Email": top["email"],
        "Email Type": top["type"],
        "Confidence": confidence,
        "Source URL": top["src"],
        "Notes": (f"Practice domain guess: {practice_domain}" if practice_domain else "") + (f" • mode={mode}" if mode else ""),
    }

def email_enrichment_sidebar(df):
    st.caption("Searches public web pages for practice/provider emails. Review suggested emails before applying.")

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
    mode = st.radio("Mode", ["Thorough", "Balanced", "Fast"], index=0,
                    help="Thorough: best chance to find direct emails (slow). Fast: fewer pages & shorter timeouts.")

    using_cse = (provider == "Google CSE") or (provider == "Auto" and not serp_key and (g_api and g_cx))
    if using_cse:
        st.info("Using Google CSE • Free tier is 100 queries/day. Capped by default.")
        enforce_free = st.toggle("Never exceed free tier (100/day)", value=True)
        usage = _load_usage()
        cap   = _get_free_cap()
        remaining = max(0, cap - int(usage.get("count", 0)))
        st.write(f"**Remaining today (ET): {remaining} / {cap}**")
    else:
        enforce_free = False

    needed = ["First name","Last name","Practice Name","City","State","Email"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        return

    overwrite = st.checkbox("Overwrite existing Email values", value=False)

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
    conf_thresh = st.slider("Auto-approve at confidence ≥", 0, 100, 75)

    autosave_every = st.number_input("Autosave results every N rows", min_value=1, max_value=100, value=10, step=1)

    if st.button("Run email search", disabled=(max_limit == 0)):
        total = int(limit)
        if total <= 0:
            st.info("Nothing to process with current settings.")
            return
        subset = candidates_df.head(total).copy()
        progress = st.progress(0.0)
        results = []
        for i, (_, row) in enumerate(subset.iterrows(), start=1):
            res = _enrich_single(row.to_dict(), provider, mode=mode)
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
            if i % int(autosave_every) == 0:
                st.session_state["_enrich_partial"] = results.copy()
                st.session_state["_email_enrich_results"] = results.copy()
            progress.progress(i/total)

        if not results:
            st.info("No results.")
            return

        st.session_state["_enrich_partial"] = results.copy()
        edited = st.data_editor(
            results,
            use_container_width=True,
            num_rows="fixed",
            key="email_enrich_editor"
        )
        # Persist edited to session so an external button can apply
        st.session_state["_email_enrich_results"] = edited

        st.write("Tip: uncheck **Approve** for generic inboxes you don’t want to use.")

        # Offer CSV download of the review grid
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

            if using_cse and enforce_free and total:
                usage = _load_usage()
                if usage.get("date") != _today_et_str():
                    usage = {"date": _today_et_str(), "count": 0}
                usage["count"] = int(usage.get("count", 0)) + total
                usage["count"] = min(usage["count"], _get_free_cap())
                _save_usage(usage)

    with st.expander("Setup & Notes"):
        st.markdown("""- **Data used**: only publicly visible, professional emails.
- **Modes**: *Thorough* scans more pages (best results, slow). *Fast* scans fewer pages.
- **Confidence**: higher when domain matches practice and local-part includes the last name.
- **API**: use SERPAPI_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID in secrets.""")

def apply_email_enrichment_results(df, overwrite=False):
    """
    Apply the most recent enrichment results saved in session_state
    to 'df'. Reads from:
      - st.session_state["_email_enrich_results"] if present,
        otherwise falls back to st.session_state["_enrich_partial"].
    Returns (applied_count, message).
    """
    try:
        results = st.session_state.get("_email_enrich_results") or st.session_state.get("_enrich_partial")
        if not results:
            return 0, "No enrichment results found in session."
        applied = 0
        for rec in results:
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
        return applied, "Applied enrichment results from session."
    except Exception as e:
        return 0, f"Failed to apply enrichment results: {e}"


# email_enrichment_util.py — FINAL (import-safe)
# Features:
# • Modes: Fast / Balanced / Thorough
# • Heartbeat status + row mini progress bar
# • ⏹ Stop run button
# • Per-row time budget (sec)
# • Optional internal-link exploration (toggle) with cap
# • Bounded parsing (caps pages & text; yields heartbeat)
# • tldextract offline extractor (no network)
# • Google CSE free-tier limiter counted PER ROW before search
# • Streamlit new width API: width='stretch'|'content'

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

# --- Regexes / constants ---
EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
OBFUSCATED_REGEXES = [
    re.compile(r"([A-Z0-9._%+-]+)\s*(?:\(|\[)?\s*(?:at|@)\s*(?:\)|\])?\s*([A-Z0-9.-]+)\s*(?:\(|\[)?\s*(?:dot|\.)\s*(?:\)|\])?\s*([A-Z]{2,})", re.I),
    re.compile(r"([A-Z0-9._%+-]+)\s*(?:\(|\[)?\s*(?:at|@)\s*(?:\)|\])?\s*([A-Z0-9.-]+\.[A-Z]{2,})", re.I),
]
GENERIC_LOCALPART = {"info","contact","office","frontdesk","help","support","admin","billing","media","press","jobs","hr","recruit","webmaster","noreply","no-reply","donotreply","do-not-reply"}
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}
USAGE_PATH = Path(".email_enrich_usage.json")

# --- tldextract offline (prevents network on first call) ---
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None, cache_dir=False)

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
def _search_google_cse(query: str, num: int = 5):
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    cse_id  = st.secrets.get("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CSE_ID")
    if not (api_key and cse_id):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query, "num": min(num, 10)}
    try:
        r = requests.get(url, params=params, timeout=12)
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

def _search_serpapi(query: str, num: int = 5):
    # Kept for optional provider; returns [] if no key configured
    key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
    if not key:
        return []
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": num, "api_key": key}
    try:
        r = requests.get(url, params=params, timeout=12)
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

def _search_web(provider: str, query: str, num: int = 5):
    if provider == "SerpAPI":
        return _search_serpapi(query, num=num) or []
    if provider == "Google CSE":
        return _search_google_cse(query, num=num) or []
    # Auto: prefer SerpAPI if available, else CSE
    hits = _search_serpapi(query, num=num)
    if hits:
        return hits
    return _search_google_cse(query, num=num)

# --- Fetch / crawl (resource-safe) ---
def _fetch_html(url: str, timeout=10, max_bytes=750_000) -> str:
    try:
        with requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            content = b""
            for chunk in r.iter_content(chunk_size=32_768):
                content += chunk
                if len(content) >= max_bytes:
                    break
            try:
                return content.decode("utf-8", errors="ignore")
            except Exception:
                return ""
    except Exception:
        return ""

def _fetch_html_parallel(urls, timeout=10, max_workers=4, cap_total=None, logger=None):
    urls = urls[:cap_total] if cap_total else urls
    results = []
    if not urls:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_html, u, timeout): u for u in urls}
        for idx, f in enumerate(as_completed(futs), start=1):
            u = futs[f]
            try:
                html_text = f.result()
            except Exception:
                html_text = ""
            results.append((u, html_text))
            if logger:
                logger(f"Fetched {idx}/{len(urls)} pages", min(0.1 + 0.7*idx/max(1,len(urls)), 0.85))
    return results

def _same_domain(url_a: str, url_b: str) -> bool:
    try:
        a = urlparse(url_a).netloc.split(":")[0].lower()
        b = urlparse(url_b).netloc.split(":")[0].lower()
        return a == b
    except Exception:
        return False

def _collect_internal_links(base_url: str, html_text: str, per_page_limit=6, global_seen=None):
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
            if global_seen is None or full not in global_seen:
                links.append(full)
        if len(links) >= per_page_limit:
            break
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# --- Extraction / scoring ---
def _extract_emails_from_html(html_text: str):
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
    ext = EXTRACTOR(url)
    domain = ".".join(part for part in [ext.domain, ext.suffix] if part)
    return domain.lower()

def _score_email(email: str, first: str, last: str, practice_domain: str):
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

def _best_guess_practice_domain(candidates):
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

# --- Core enrichment for a single row ---
def _enrich_single(row: dict, provider: str, mode: str = "Balanced", logger=None, deadline_ts=None,
                   explore_internal: bool = False, max_internal_pages: int = 24) -> Dict:
    if mode == "Thorough":
        search_num = 10; initial_cap = 24; internal_per_page = 8; internal_total_cap = min(120, max_internal_pages)
        max_workers = 6; timeout = 12
    elif mode == "Fast":
        search_num = 3; initial_cap = 10; internal_per_page = 0; internal_total_cap = 0
        max_workers = 6; timeout = 6
    else:
        search_num = 6; initial_cap = 18; internal_per_page = 6; internal_total_cap = min(36, max_internal_pages)
        max_workers = 4; timeout = 10

    if not explore_internal:
        internal_per_page = 0
        internal_total_cap = 0

    def log(msg, frac=None):
        if logger:
            logger(msg, frac)

    first = str(row.get("First name","")).strip()
    last  = str(row.get("Last name","")).strip()
    prac  = str(row.get("Practice Name","")).strip()

    if deadline_ts and time.time() > deadline_ts:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"Row time budget exceeded (pre-search)"}

    query = _build_query(row)
    log("Searching...", 0.02)
    hits = _search_web(provider, query, num=search_num)
    if not hits:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"No search results"}

    candidates = []
    suffixes = ("/contact","/contact-us","/team","/providers","/physicians","/about","/faculty","/locations")
    for h in hits:
        base = h["url"].rstrip("/")
        if _viable_page(base):
            candidates.append(base)
        for s in suffixes:
            u = base + s
            if _viable_page(u):
                candidates.append(u)
    candidates = list(dict.fromkeys(candidates))[:initial_cap]

    if deadline_ts and time.time() > deadline_ts:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"Row time budget exceeded (before fetch)"}

    log(f"Fetching up to {len(candidates)} pages...", 0.05)
    fetched = _fetch_html_parallel(candidates, timeout=timeout, max_workers=max_workers, cap_total=len(candidates), logger=logger)

    # Optional internal crawl
    if internal_per_page > 0 and internal_total_cap > 0 and (not deadline_ts or time.time() < deadline_ts):
        seen = set()
        more = []
        for url, html_text in fetched:
            if not html_text:
                continue
            links = _collect_internal_links(url, html_text, per_page_limit=internal_per_page, global_seen=seen)
            for u in links:
                if u not in seen and _viable_page(u):
                    seen.add(u)
                    more.append(u)
                if len(more) >= internal_total_cap:
                    break
            if len(more) >= internal_total_cap:
                break
        if more:
            log(f"Following up to {len(more)} internal links...", 0.15)
            if deadline_ts and time.time() > deadline_ts:
                return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"Row time budget exceeded (before internal)"}
            fetched += _fetch_html_parallel(more, timeout=timeout, max_workers=max_workers, cap_total=internal_total_cap, logger=logger)

    if deadline_ts and time.time() > deadline_ts:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"Row time budget exceeded (before parsing)"}

    # --- Bounded parsing with heartbeat ---
    log("Parsing pages for emails...", 0.30)
    emails_found = []
    parsed = 0
    max_parse = min(len(fetched), 60)  # safety cap
    for url, html_text in fetched[:max_parse]:
        if deadline_ts and time.time() > deadline_ts:
            return {"Found Email":"", "Email Type":"", "Confidence":0,
                    "Source URL":"", "Notes":"Row time budget exceeded (during parsing)"}
        if not html_text:
            continue
        try:
            txt = html.unescape(html_text[:300_000])  # cap text size
            for email in set(EMAIL_REGEX.findall(txt)):
                emails_found.append((email, url))
            for rx in OBFUSCATED_REGEXES:
                for match in rx.findall(txt):
                    if len(match) == 3:
                        local, domain, tld = match
                        emails_found.append((f"{local}@{domain}.{tld}", url))
                    elif len(match) == 2:
                        local, domain = match
                        emails_found.append((f"{local}@{domain}", url))
        except Exception:
            pass
        parsed += 1
        if logger and parsed % 5 == 0:
            logger(f"Parsing pages… {parsed}/{max_parse}", 0.30 + 0.60*parsed/max(1, max_parse))

    practice_domain = _best_guess_practice_domain(hits) if prac else ""

    ranked = []
    for email, src in set(emails_found):
        sc, typ = _score_email(email, first, last, practice_domain)
        ranked.append({"email": email, "score": sc, "type": typ, "src": src})

    if not ranked:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":hits[0]["url"], "Notes":"No emails on scanned pages"}

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[0]
    conf = max(0, min(100, 42 + top["score"]))
    return {"Found Email": top["email"], "Email Type": top["type"], "Confidence": conf, "Source URL": top["src"], "Notes": ""}

# --- UI entry point (sidebar widget) ---
def email_enrichment_sidebar(df):
    st.caption("Searches public web pages for practice/provider emails. Review suggested emails before applying.")

    serp_key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
    g_api    = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    g_cx     = st.secrets.get("GOOGLE_CSE_ID")  or os.getenv("GOOGLE_CSE_ID")

    available = []
    if serp_key: available.append("SerpAPI")
    if g_api and g_cx: available.append("Google CSE")
    if not available:
        st.error("No search API configured. Add SERPAPI_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID to Streamlit secrets.")
        return

    provider = st.selectbox("Search provider", ["Auto"] + available, index=0,
                            help="Auto = prefers SerpAPI (if present), else Google CSE.")
    mode = st.radio("Mode", ["Balanced","Thorough","Fast"], index=0)

    # Controls / status
    status = st.empty()
    row_status = st.empty()
    row_bar = st.progress(0.0)
    c1, c2 = st.columns([2,2])
    with c1:
        if st.button("⏹️ Stop run", key="stop_run_btn", width='content'):
            st.session_state["_abort_enrich"] = True
    st.session_state.setdefault("_abort_enrich", False)
    row_time_budget = st.number_input("Per-row time budget (sec)", min_value=60, max_value=600, value=180, step=30,
                                      help="Skip to next row if a site is too slow.")
    explore_internal = st.checkbox("Explore internal links (slower, finds more)", value=False)
    max_internal_pages = st.slider("Max internal pages", 0, 120, 24, step=6,
                                   help="Upper bound when exploring internal links. Set 0 to disable.")

    using_cse = (provider == "Google CSE") or (provider == "Auto" and not serp_key and (g_api and g_cx))
    if using_cse:
        st.info("Using Google CSE • Free tier ~100 queries/day. Capped by default.")
        enforce_free = st.toggle("Never exceed free tier (100/day)", value=True)
        usage = _load_usage(); cap = _get_free_cap()
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
        usage = _load_usage(); cap = _get_free_cap()
        remaining = max(0, cap - int(usage.get("count", 0)))
        if remaining == 0:
            st.warning("Daily free CSE quota reached.")
        max_limit = min(max_limit, remaining)

    limit = st.number_input("Max records to scan", min_value=0, max_value=int(max_limit), value=int(min(default_limit, max_limit)))
    conf_thresh = st.slider("Auto-approve at confidence ≥", 0, 100, 75)
    autosave_every = st.number_input("Autosave every N rows", min_value=1, max_value=100, value=10, step=1)

    if st.button("Run email search", key="run_search_btn", width='content', disabled=(max_limit == 0)):
        total = int(limit)
        if total <= 0:
            st.info("Nothing to process with current settings.")
            return
        subset = candidates_df.head(total).copy()
        progress = st.progress(0.0)
        results = []
        for i, (_, row) in enumerate(subset.iterrows(), start=1):
            if st.session_state.get("_abort_enrich"):
                st.warning("Run stopped by user.")
                break

            status.write(f"Row {i}/{total}: {row['First name']} {row['Last name']} — {row.get('Practice Name','')}")
            start_ts = time.time()
            deadline_ts = start_ts + int(row_time_budget)

            # --- Count CSE usage PER ROW (before querying), if limiter is on ---
            if using_cse and enforce_free:
                usage = _load_usage()
                if usage.get("date") != _today_et_str():
                    usage = {"date": _today_et_str(), "count": 0}
                cap = _get_free_cap()
                if int(usage.get("count", 0)) >= cap:
                    st.warning("Daily free CSE quota reached — stopping.")
                    break
                usage["count"] = int(usage.get("count", 0)) + 1
                _save_usage(usage)

            def logger(msg, frac=None):
                row_status.write(msg)
                if frac is not None:
                    row_bar.progress(min(max(frac, 0.0), 0.95))

            res = _enrich_single(
                row.to_dict(), provider, mode=mode, logger=logger, deadline_ts=deadline_ts,
                explore_internal=explore_internal, max_internal_pages=int(max_internal_pages)
            )
            row_bar.progress(1.0)
            row_status.write("Row done.")

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
            progress.progress(i/total)

        if not results:
            st.info("No results.")
            return

        st.session_state["_enrich_partial"] = results.copy()
        edited = st.data_editor(
            results,
            width='stretch',
            num_rows="fixed",
            key="email_enrich_editor"
        )
        st.session_state["_email_enrich_results"] = edited

        st.write("Tip: uncheck **Approve** for generic inboxes you don’t want to use.")

        try:
            import pandas as pd
            delta = pd.DataFrame(edited)
            st.download_button(
                "Download enrichment results (CSV)",
                delta.to_csv(index=False).encode("utf-8"),
                file_name="email_enrichment_results.csv",
                mime="text/csv",
                width='content'
            )
        except Exception:
            pass

        if st.button("Apply approved emails to Data", key="apply_btn", width='content'):
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

# External helper to apply last reviewed results to a df
def apply_email_enrichment_results(df, overwrite=False):
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


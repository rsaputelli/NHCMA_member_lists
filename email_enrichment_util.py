
# email_enrichment_util.py — FULL + Live SerpAPI usage (Account API)
# This build includes everything from the previous drop plus:
# • Live SerpAPI usage via https://serpapi.com/account.json (free; not counted against quota)
# • Sidebar shows: used this month, plan limit, remaining, last-hour searches, hourly cap
# • Optional limiter uses LIVE remaining instead of a local counter (falls back if API unreachable)
#
# Notes:
# - Keep SERPAPI_KEY in Streamlit secrets or env
# - We cache the account call for 60s to avoid extra HTTP
#
# Other features retained:
# - Persistent results UI
# - Session & persisted skip list
# - Start-at-row-N
# - Disk-persisted autosave + recovery
# - Google CSE daily cap tracker
#
# If you prefer the limiter to exclusively use LIVE counts, turn on "Respect SerpAPI monthly limit"—
# we’ll prefer live plan_searches_left and only fall back to the local JSON when the API fails.

import os, re, time, json, html, requests, io
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from bs4 import BeautifulSoup
import tldextract
import pandas as pd

# --- Regexes / constants ---
EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
OBFUSCATED_REGEXES = [
    re.compile(r"([A-Z0-9._%+-]+)\s*(?:\(|\[)?\s*(?:at|@)\s*(?:\)|\])?\s*([A-Z0-9.-]+)\s*(?:\(|\[)?\s*(?:dot|\.)\s*(?:\)|\])?\s*([A-Z]{2,})", re.I),
    re.compile(r"([A-Z0-9._%+-]+)\s*(?:\(|\[)?\s*(?:at|@)\s*(?:\)|\])?\s*([A-Z0-9.-]+\.[A-Z]{2,})", re.I),
]
GENERIC_LOCALPART = {"info","contact","office","frontdesk","help","support","admin","billing","media","press","jobs","hr","recruit","webmaster","noreply","no-reply","donotreply","do-not-reply"}
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}
USAGE_PATH = Path(".email_enrich_usage.json")              # CSE daily
SERP_USAGE_PATH = Path(".serpapi_usage.json")              # SerpAPI monthly (fallback only when Account API not used)
SKIPLIST_PATH = Path(".email_enrich_skiplist.json")        # persisted skip keys
AUTOSAVE_PATH = Path(".email_enrich_autosave.json")        # persisted partials
AUTOSAVE_TMP = Path(".email_enrich_autosave.json.tmp")

# --- tldextract offline (prevents network on first call) ---
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None, cache_dir=False)

def _today_et_str():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

def _month_key():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m")

def _load_usage():
    # Google CSE daily usage (local)
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

def _load_serp_month_usage():
    key = _month_key()
    usage = {"month": key, "count": 0}
    try:
        if SERP_USAGE_PATH.exists():
            data = json.loads(SERP_USAGE_PATH.read_text())
            if data.get("month") == key:
                usage = data
    except Exception:
        pass
    return usage

def _save_serp_month_usage(usage: dict):
    try:
        SERP_USAGE_PATH.write_text(json.dumps(usage))
    except Exception:
        pass

def _get_free_cap():
    # Google CSE free tier daily cap (configurable via secrets)
    try:
        return int(st.secrets.get("FREE_TIER_DAILY_LIMIT", 100))
    except Exception:
        return 100

def _row_key3(first, last, practice):
    return (str(first).strip().lower(), str(last).strip().lower(), str(practice or "").strip().lower())

# --- SerpAPI Account API (live usage) ---
@st.cache_data(ttl=60, show_spinner=False)
def _serpapi_account_info(api_key: str) -> Optional[dict]:
    if not api_key:
        return None
    url = "https://serpapi.com/account.json"
    try:
        r = requests.get(url, params={"api_key": api_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Expected keys: searches_per_month, plan_searches_left, this_month_usage, last_hour_searches, account_rate_limit_per_hour, total_searches_left, extra_credits
        return data
    except Exception:
        return None

# --- Fetch / crawl helpers ---
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

EMAIL_RX = EMAIL_REGEX
OBF_RXS = OBFUSCATED_REGEXES

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
    hits = _search_web(provider, query, num=6 if mode!="Thorough" else 10)
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
    candidates = list(dict.fromkeys(candidates))[:(24 if mode=="Thorough" else (10 if mode=="Fast" else 18))]

    if deadline_ts and time.time() > deadline_ts:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"Row time budget exceeded (before fetch)"}

    log(f"Fetching up to {len(candidates)} pages...", 0.05)
    fetched = _fetch_html_parallel(candidates, timeout=(12 if mode=="Thorough" else (6 if mode=="Fast" else 10)), max_workers=(6 if mode!="Balanced" else 4), cap_total=len(candidates), logger=logger)

    # Optional internal crawl
    internal_per_page = 8 if (mode=="Thorough" and explore_internal) else (6 if (mode=="Balanced" and explore_internal) else 0)
    internal_total_cap = min(120 if mode=="Thorough" else 36, max_internal_pages) if explore_internal else 0
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
            fetched += _fetch_html_parallel(more, timeout=(12 if mode=="Thorough" else 10), max_workers=(6 if mode!="Balanced" else 4), cap_total=internal_total_cap, logger=logger)

    if deadline_ts and time.time() > deadline_ts:
        return {"Found Email":"", "Email Type":"", "Confidence":0, "Source URL":"", "Notes":"Row time budget exceeded (before parsing)"}

    # Parsing
    log("Parsing pages for emails...", 0.30)
    emails_found = []
    parsed = 0
    max_parse = min(len(fetched), 60)
    for url, html_text in fetched[:max_parse]:
        if deadline_ts and time.time() > deadline_ts:
            return {"Found Email":"", "Email Type":"", "Confidence":0,
                    "Source URL":"", "Notes":"Row time budget exceeded (during parsing)"}
        if not html_text:
            continue
        try:
            txt = html.unescape(html_text[:300_000])
            for email in set(EMAIL_RX.findall(txt)):
                emails_found.append((email, url))
            for rx in OBF_RXS:
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

# --- External helper to apply last reviewed results to a df (persist-safe) ---
def apply_email_enrichment_results(df, overwrite=False, persist_skip=False):
    try:
        results = st.session_state.get("_email_enrich_results") or st.session_state.get("_enrich_partial")
        if not results:
            return 0, "No enrichment results found in session."
        applied = 0
        st.session_state.setdefault("_email_enrich_blacklist", [])
        blacklist = st.session_state["_email_enrich_blacklist"]

        for rec in results:
            approved = bool(rec.get("Approve"))
            suggested = str(rec.get("Suggested Email") or "").strip()
            fn = rec.get("First name","")
            ln = rec.get("Last name","")
            pn = rec.get("Practice Name","")
            key = list(_row_key3(fn, ln, pn))

            if approved and suggested:
                mask = (
                    (df["First name"].astype(str).str.strip() == str(fn).strip()) &
                    (df["Last name"].astype(str).str.strip() == str(ln).strip())
                )
                if "Practice Name" in df.columns and str(pn).strip():
                    mask &= (df["Practice Name"].fillna("").astype(str).str.strip() == str(pn).strip())
                if overwrite:
                    df.loc[mask, "Email"] = suggested
                    applied += int(mask.sum())
                else:
                    blanks = mask & (df["Email"].isna() | (df["Email"].astype(str).str.strip() == ""))
                    df.loc[blanks, "Email"] = suggested
                    applied += int(blanks.sum())
            else:
                if key not in blacklist:
                    blacklist.append(key)

        st.session_state["_email_enrich_blacklist"] = blacklist

        if persist_skip:
            existing = set(_load_persisted_skiplist())
            merged = existing.union(set(tuple(x) for x in blacklist))
            _save_persisted_skiplist(list(merged))

        return applied, f"Applied enrichment results. Skipped rows (not approved) now tracked; session skip list size: {len(blacklist)}."
    except Exception as e:
        return 0, f"Failed to apply enrichment results: {e}"

# --- Skip list persistence ---
def _load_persisted_skiplist():
    try:
        if SKIPLIST_PATH.exists():
            data = json.loads(SKIPLIST_PATH.read_text())
            if isinstance(data, list):
                return [tuple(x) for x in data]
    except Exception:
        pass
    return []

def _save_persisted_skiplist(keys):
    try:
        data = [list(t) for t in keys]
        SKIPLIST_PATH.write_text(json.dumps(data))
    except Exception:
        pass

# --- Autosave persistence helpers ---
def _persist_autosave(results: list):
    try:
        AUTOSAVE_TMP.write_text(json.dumps(results))
        AUTOSAVE_TMP.replace(AUTOSAVE_PATH)
    except Exception:
        pass

def _load_autosave():
    try:
        if AUTOSAVE_PATH.exists():
            data = json.loads(AUTOSAVE_PATH.read_text())
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def _clear_autosave():
    try:
        if AUTOSAVE_PATH.exists():
            AUTOSAVE_PATH.unlink()
        if AUTOSAVE_TMP.exists():
            AUTOSAVE_TMP.unlink()
        return True
    except Exception:
        return False

# --- UI entry point (sidebar widget) ---
def email_enrichment_sidebar(df):
    st.session_state.setdefault("_results_rendered", False)
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
                            help="Auto = prefers SerpAPI if present, else Google CSE.")
    mode = st.radio("Mode", ["Balanced","Thorough","Fast"], index=0)

    # Controls / status
    status = st.empty()
    row_status = st.empty()
    row_bar = st.progress(0.0)
    c1, c2 = st.columns([2,2])
    with c1:
        if st.button("⏹️ Stop run", key="stop_run_btn", type="secondary"):
            st.session_state["_abort_enrich"] = True
    st.session_state.setdefault("_abort_enrich", False)
    row_time_budget = st.number_input("Per-row time budget (sec)", min_value=60, max_value=600, value=180, step=30,
                                      help="Skip to next row if a site is too slow.")
    explore_internal = st.checkbox("Explore internal links (slower, finds more)", value=False)
    max_internal_pages = st.slider("Max internal pages", 0, 120, 24, step=6,
                                   help="Upper bound when exploring internal links. Set 0 to disable.")

    # Determine which caps to display
    using_serp_pref = (provider == "SerpAPI") or (provider == "Auto" and bool(serp_key))
    using_cse = (provider == "Google CSE") or (provider == "Auto" and not bool(serp_key) and (g_api and g_cx))

    enforce_free = False
    respect_serp_cap = False

    # Google CSE
    if using_cse:
        st.info("Using Google CSE • Free tier ~100 queries/day. Capped by default.")
        enforce_free = st.toggle("Never exceed free tier (100/day)", value=True, key="cse_cap_toggle")
        usage = _load_usage(); cap = _get_free_cap()
        remaining = max(0, cap - int(usage.get("count", 0)))
        st.write(f"**CSE Remaining today (ET): {remaining} / {cap}**")

    # SerpAPI (live)
    live_remaining_mo = None
    live_monthly_cap = None
    if using_serp_pref:
        st.info("Using SerpAPI • Free plan ~250 searches/month (varies by plan).")
        respect_serp_cap = st.toggle("Respect SerpAPI monthly limit", value=True, key="serp_cap_toggle")
        acct = _serpapi_account_info(serp_key) if serp_key else None
        if acct:
            live_monthly_cap = acct.get("searches_per_month")
            live_remaining_mo = acct.get("plan_searches_left", acct.get("total_searches_left"))
            used = acct.get("this_month_usage")
            last_hr = acct.get("last_hour_searches")
            hr_cap = acct.get("account_rate_limit_per_hour")
            st.write(f"**SerpAPI (live): used {used} / {live_monthly_cap}, remaining {live_remaining_mo}**")
            if last_hr is not None and hr_cap is not None:
                st.caption(f"Last hour: {last_hr} • Hourly cap: {hr_cap}")
        else:
            su = _load_serp_month_usage()
            remaining_mo = max(0, 250 - int(su.get("count", 0)))
            st.write(f"**SerpAPI (local est.): remaining {remaining_mo} / 250**")

    needed = ["First name","Last name","Practice Name","City","State","Email"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        return

    # --- Skip list controls ---
    st.session_state.setdefault("_email_enrich_blacklist", [])
    persist_skip = st.toggle("Persist skip list across restarts", value=False,
                             help="Writes not-approved rows to a local JSON so they remain skipped after reloads.")
    if persist_skip and not st.session_state.get("_skip_loaded", False):
        persisted = _load_persisted_skiplist()
        if persisted:
            merged = set(tuple(x) for x in st.session_state["_email_enrich_blacklist"]).union(set(persisted))
            st.session_state["_email_enrich_blacklist"] = [list(t) for t in merged]
        st.session_state["_skip_loaded"] = True

    cols = st.columns(2)
    with cols[0]:
        if st.button("Reset PERSISTED skipped list", key="reset_persist_blacklist", type="secondary"):
            try:
                if SKIPLIST_PATH.exists():
                    SKIPLIST_PATH.unlink()
                st.success("Persisted skip list cleared (session skip list unchanged).")
            except Exception as e:
                st.warning(f"Could not clear persisted skip list: {e}")
    with cols[1]:
        if st.button("Reset skipped list (SESSION)", key="reset_blacklist", type="secondary"):
            st.session_state["_email_enrich_blacklist"] = []
            st.success("Session skip list cleared.")

    # --- Autosave persistence controls ---
    persist_autosave = st.toggle("Persist autosave to disk", value=False,
                                 help="Writes partial results to disk during a run so you can recover after a restart.")
    autosave_persist_every = 5
    if persist_autosave:
        autosave_persist_every = st.number_input("Autosave persist frequency (rows)", min_value=1, max_value=100, value=5, step=1)
        if AUTOSAVE_PATH.exists():
            try:
                data = json.loads(AUTOSAVE_PATH.read_text())
                count = len(data) if isinstance(data, list) else 0
            except Exception:
                count = 0
            recov_cols = st.columns(2)
            with recov_cols[0]:
                if st.button(f"Recover autosave ({count} rows)", key="recover_autosave"):
                    recovered = _load_autosave()
                    if recovered:
                        st.session_state["_enrich_partial"] = recovered
                        st.session_state["_email_enrich_results"] = recovered
                        st.success(f"Recovered {len(recovered)} rows from autosave.")
                    else:
                        st.info("No recoverable autosave found.")
            with recov_cols[1]:
                if st.button("Clear autosave file", key="clear_autosave"):
                    if _clear_autosave():
                        st.success("Autosave file cleared.")
                    else:
                        st.warning("Could not clear autosave file.")

    overwrite = st.checkbox("Overwrite existing Email values", value=False)

    # Candidate filtering
    if overwrite:
        candidates_df = df.copy()
    else:
        candidates_df = df[df["Email"].isna() | (df["Email"].astype(str).str.strip() == "")]

    # Skip list (session & persisted)
    skip_attempted = st.toggle("Skip already attempted this session", value=True,
                               help="Skips rows you did not approve in prior runs this session (created when you click Apply).")
    if skip_attempted:
        bl = set(tuple(x) for x in st.session_state["_email_enrich_blacklist"])
        def _key_from_row(r):
            return _row_key3(r["First name"], r["Last name"], r.get("Practice Name",""))
        mask_skip = candidates_df.apply(lambda r: _key_from_row(r) in bl, axis=1)
        if mask_skip.any():
            st.caption(f"Skipping {int(mask_skip.sum())} previously attempted rows this session.")
            candidates_df = candidates_df[~mask_skip]

    # --- Batching & offsets ---
    usage = _load_usage(); cap = _get_free_cap()
    remaining = max(0, cap - int(usage.get("count", 0)))
    default_limit = min(25, len(candidates_df))
    max_limit = len(candidates_df)
    if using_cse and enforce_free:
        max_limit = min(max_limit, remaining)

    start_offset = st.number_input("Start at row N (within current candidates)", min_value=0, max_value=int(max_limit), value=0,
                                   help="Offsets into the remaining candidate pool after filters & skip list are applied.")
    limit = st.number_input("Max records to scan", min_value=0, max_value=int(max(0, max_limit - start_offset)), value=int(min(default_limit, max(0, max_limit - start_offset))))
    conf_thresh = st.slider("Auto-approve at confidence ≥", 0, 100, 75)
    autosave_every = st.number_input("Autosave every N rows (session)", min_value=1, max_value=100, value=10, step=1)


    # --- Full dataset persistence on Apply ---
    persist_dataset = st.toggle("Persist full dataset to disk on Apply", value=False,
                                help="When you click Apply, save the entire updated dataset to CSV/XLSX on disk.")
    out_base = st.text_input("Output filename base", value="enriched_dataset",
                             help="Used if enabled (creates enriched_dataset.csv and .xlsx).")
    st.session_state["_persist_dataset_on_apply"] = persist_dataset
    st.session_state["_persist_dataset_name"] = out_base

    # Sidebar persistent actions if results exist
    have_results = bool(st.session_state.get("_email_enrich_results") or st.session_state.get("_enrich_partial"))
    if have_results:
        st.markdown("After running, you can apply or download again here:")
        if st.button("Apply approved (from sidebar)", key="apply_from_sidebar"):
            applied, msg = apply_email_enrichment_results(df, overwrite=overwrite, persist_skip=persist_skip)
            st.success(f"{msg} Updated rows: {applied}")
        
            # Stage full-dataset downloads and optional disk write
            try:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                    df.to_excel(xw, index=False, sheet_name="Data")
                xlsx_bytes = buf.getvalue()
                st.session_state["_last_enriched_csv"] = csv_bytes
                st.session_state["_last_enriched_xlsx"] = xlsx_bytes
                if st.session_state.get("_persist_dataset_on_apply", False):
                    base = st.session_state.get("_persist_dataset_name", "enriched_dataset") or "enriched_dataset"
                    Path(f"{base}.csv").write_bytes(csv_bytes)
                    Path(f"{base}.xlsx").write_bytes(xlsx_bytes)
                    st.info(f"Saved to disk: {base}.csv and {base}.xlsx")
            except Exception as e:
                st.warning(f"Could not prepare full dataset downloads: {e}")
        try:
            res = st.session_state.get("_email_enrich_results") or st.session_state.get("_enrich_partial")
            if res:
                df_out = pd.DataFrame(res)
                st.download_button(
                    "Download Enrichment Results (CSV)",
                    df_out.to_csv(index=False).encode("utf-8"),
                    file_name="email_enrichment_results.csv",
                    mime="text/csv",
                    key="dl_sidebar_csv",
                )
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                    df_out.to_excel(xw, index=False, sheet_name="Results")
                st.download_button(
                    "Download Enrichment Results (Excel)",
                    data=buf.getvalue(),
                    file_name="email_enrichment_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_sidebar_xlsx",
                )
        except Exception:
            pass
            pass

    # --- RUN BUTTON ---
    if st.button("Run email search", key="run_search_btn", disabled=(max_limit == 0)):
        # Effective total based on offset
        avail = max(0, len(candidates_df) - int(start_offset))
        total = min(int(limit), avail)
        if total <= 0:
            st.info("Nothing to process with current settings (check offset / limit).")
            return

        # Clear old results (fresh run)
        st.session_state["_email_enrich_results"] = None
        st.session_state["_enrich_partial"] = None
        st.session_state["_results_rendered"] = False

        subset = candidates_df.iloc[int(start_offset):int(start_offset)+total].copy()
        progress = st.progress(0.0)
        results = []

        # Determine caps once per run
        using_serp_pref = (provider == "SerpAPI") or (provider == "Auto" and bool(serp_key))
        using_cse = (provider == "Google CSE") or (provider == "Auto" and not bool(serp_key) and (g_api and g_cx))

        # Pull LIVE SerpAPI counters once per run (cache 60s)
        live_plan_remaining = None
        if using_serp_pref and respect_serp_cap and serp_key:
            acct = _serpapi_account_info(serp_key)
            if acct:
                live_plan_remaining = acct.get("plan_searches_left", acct.get("total_searches_left"))

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

            # --- SerpAPI monthly limiter (prefer LIVE; fallback to local) ---
            if using_serp_pref and respect_serp_cap:
                if live_plan_remaining is not None:
                    if live_plan_remaining <= 0:
                        st.warning("SerpAPI monthly limit reached (live) — stopping.")
                        break
                    live_plan_remaining -= 1
                else:
                    su = _load_serp_month_usage()
                    if su.get("month") != _month_key():
                        su = {"month": _month_key(), "count": 0}
                    if int(su.get("count", 0)) >= 250:
                        st.warning("SerpAPI monthly limit reached (local est.) — stopping.")
                        break
                    su["count"] = int(su.get("count", 0)) + 1
                    _save_serp_month_usage(su)

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

            # Session autosave
            if i % int(autosave_every) == 0:
                st.session_state["_enrich_partial"] = results.copy()

            # Disk autosave (optional)
            if persist_autosave and i % int(autosave_persist_every) == 0:
                _persist_autosave(results)

            progress.progress(i/total)

        if not results:
            st.info("No results.")
            render_email_enrichment_results(df, overwrite=overwrite, persist_skip=persist_skip)
            st.session_state["_results_rendered"] = True
            return

        st.session_state["_enrich_partial"] = results.copy()
        if persist_autosave:
            _persist_autosave(results)

        # Initial editor view (immediate), then persistently render
        edited = st.data_editor(
            results,
            width='stretch',
            num_rows="fixed",
            key="email_enrich_editor_first"
        )
        st.session_state["_email_enrich_results"] = edited

        st.write("Tip: leave **Approve** unchecked for generic/low-confidence inboxes. On Apply, those rows are added to the skip list.")
        render_email_enrichment_results(df, overwrite=overwrite, persist_skip=persist_skip)
        st.session_state["_results_rendered"] = True

    # Show results area once per run
    if not st.session_state.get("_results_rendered", False):
        render_email_enrichment_results(df, overwrite=overwrite, persist_skip=persist_skip)
        st.session_state["_results_rendered"] = True

# --- Persistent results renderer ---
def render_email_enrichment_results(df=None, overwrite=False, persist_skip=False):
    try:
        results = st.session_state.get("_email_enrich_results") or st.session_state.get("_enrich_partial")
        if not results:
            return
        st.subheader("Email Enrichment Results")
        edited = st.data_editor(
            results,
            width='stretch',
            num_rows="fixed",
            key="email_enrich_editor_persistent"
        )
        st.session_state["_email_enrich_results"] = edited

        df_out = pd.DataFrame(edited)

        st.download_button(
            "Download enrichment results (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="email_enrichment_results.csv",
            mime="text/csv",
            key="dl_results_csv_persist",
        )

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            df_out.to_excel(xw, index=False, sheet_name="Results")
        st.download_button(
            "Download enrichment results (Excel)",
            data=buf.getvalue(),
            file_name="email_enrichment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_results_xlsx_persist",
        )


        # Full dataset downloads if staged (survives reruns)
        full_csv = st.session_state.get("_last_enriched_csv")
        full_xlsx = st.session_state.get("_last_enriched_xlsx")
        if full_csv or full_xlsx:
            st.subheader("Download Enriched Data (Full Dataset)")
            if full_csv:
                st.download_button(
                    "Download Enriched (CSV)",
                    data=full_csv,
                    file_name="enriched_dataset.csv",
                    mime="text/csv",
                    key="dl_full_dataset_csv_persist",
                )
            if full_xlsx:
                st.download_button(
                    "Download Enriched (Excel)",
                    data=full_xlsx,
                    file_name="enriched_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_full_dataset_xlsx_persist",
                )
        if df is not None:
            if st.button("Apply approved emails to Data", key="apply_btn_persist"):
                applied, msg = apply_email_enrichment_results(df, overwrite=overwrite, persist_skip=persist_skip)
                st.success(f"{msg} Updated rows: {applied}")
    
                # Stage full-dataset downloads and optional persist, plus immediate download buttons
                try:
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                        df.to_excel(xw, index=False, sheet_name="Data")
                    xlsx_bytes = buf.getvalue()
                    st.session_state["_last_enriched_csv"] = csv_bytes
                    st.session_state["_last_enriched_xlsx"] = xlsx_bytes
                    if st.session_state.get("_persist_dataset_on_apply", False):
                        base = st.session_state.get("_persist_dataset_name", "enriched_dataset") or "enriched_dataset"
                        Path(f"{base}.csv").write_bytes(csv_bytes)
                        Path(f"{base}.xlsx").write_bytes(xlsx_bytes)
                        st.info(f"Saved to disk: {base}.csv and {base}.xlsx")
                    # Immediate buttons
                    st.download_button(
                        "Download Enriched (CSV)",
                        data=csv_bytes,
                        file_name="enriched_dataset.csv",
                        mime="text/csv",
                        key="dl_full_dataset_csv_inline",
                        help="Full dataset with applied emails"
                    )
                    st.download_button(
                        "Download Enriched (Excel)",
                        data=xlsx_bytes,
                        file_name="enriched_dataset.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_full_dataset_xlsx_inline",
                        help="Full dataset with applied emails"
                    )
                except Exception as e:
                    st.warning(f"Could not prepare full dataset downloads: {e}")
    except Exception as e:
        st.warning(f"Could not render results: {e}")



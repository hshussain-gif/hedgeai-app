"""HedgeAI — standalone Streamlit application. Requires Python 3.10+.

Install: python -m pip install streamlit yfinance vaderSentiment pandas
No API keys, paid services, or sentiment-model downloads are required.
Run:     python -m streamlit run app.py
"""
from __future__ import annotations

import html
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

UTC = timezone.utc


def numeric(value):
    """Missing values remain unknown rather than silently becoming zero."""
    try:
        result = float(value)
        return result if math.isfinite(result) and not isinstance(value, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


def clean(value, limit=2000):
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(str(value or ""))).split())[:limit]


def parse_time(value):
    try:
        if isinstance(value, (float, int)):
            return datetime.fromtimestamp(value, UTC)
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def extract_revenues(income):
    """Three years of growth require four consecutive annual observations."""
    rows = []
    if income is not None and not income.empty:
        for key in ("TotalRevenue", "Total Revenue"):
            if key in income.index:
                series = income.loc[key].sort_index(ascending=False).iloc[:4]
                rows = [{"fiscal_year_end": str(date.date()), "revenue": numeric(value)}
                        for date, value in series.items()]
                break
    cagr = None
    if len(rows) == 4 and all(row["revenue"] is not None and row["revenue"] > 0 for row in rows):
        dates = [datetime.fromisoformat(row["fiscal_year_end"]) for row in rows]
        if all(330 <= (dates[i] - dates[i + 1]).days <= 400 for i in range(3)):
            cagr = (rows[0]["revenue"] / rows[3]["revenue"]) ** (1 / 3) - 1
    return rows, cagr


def extract_news(raw, symbol, now):
    """Handle both current nested ticker.news records and older flat records."""
    candidates = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item
        if not isinstance(content, dict):
            continue
        related = content.get("relatedTickers") or item.get("relatedTickers")
        if isinstance(related, list) and related and symbol not in related:
            continue
        title = clean(content.get("title"), 400)
        published = parse_time(content.get("pubDate") or item.get("providerPublishTime"))
        if not title or published is None or not now - timedelta(days=30) <= published <= now:
            continue
        link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        url = link.get("url", "") if isinstance(link, dict) else ""
        url = clean(url or item.get("link"), 2000)
        if not url.startswith(("https://", "http://")):
            url = ""
        candidates.append({"headline": title, "published_utc": published.isoformat(), "url": url,
                           "source": content.get("feed_source", "yfinance")})
    candidates.sort(key=lambda row: row["published_utc"], reverse=True)
    selected, titles, urls = [], set(), set()
    for row in candidates:
        title, url = row["headline"].casefold(), row["url"].split("?")[0]
        if title in titles or (url and url in urls):
            continue
        titles.add(title)
        if url:
            urls.add(url)
        selected.append({"id": f"N{len(selected) + 1}", **row})
        if len(selected) == 5:
            break
    return selected


def parse_rss(payload, symbol, source, now):
    """Parse RSS 2.0 titles/dates; reuse normal sorting and deduplication."""
    if len(payload) > 2_000_000 or b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("Feed too large or contains unsupported XML declarations.")
    root = ET.fromstring(payload)
    raw = []
    for item in root.findall("./channel/item"):
        title = clean(item.findtext("title"), 400)
        publisher = clean(item.findtext("source"), 200)
        # Google appends the publisher to titles; do not score that suffix.
        if publisher and title.endswith(" - " + publisher):
            title = title[:-(len(publisher) + 3)].strip()
        value = item.findtext("pubDate")
        try:
            published = parsedate_to_datetime(value) if value else None
            if published and published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
        except (TypeError, ValueError, OverflowError):
            published = parse_time(value)
        if published is None:
            continue
        raw.append({"content": {"title": title, "pubDate": published.isoformat(),
                    "canonicalUrl": {"url": item.findtext("link") or ""}, "feed_source": source}})
    return extract_news(raw, symbol, now)


def fetch_rss_news(url, symbol, source):
    """Bounded HTTPS request; retry only transient errors, never indefinitely."""
    request = Request(url, headers={"User-Agent": "HedgeAI/1.0 (RSS reader)",
                                   "Accept": "application/rss+xml, application/xml, text/xml"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=8) as response:
                payload = response.read(2_000_001)
            return parse_rss(payload, symbol, source, datetime.now(UTC))
        except HTTPError as exc:
            if attempt or exc.code not in (408, 429, 500, 502, 503, 504):
                raise
        except (URLError, TimeoutError, OSError):
            if attempt:
                raise
        time.sleep(0.5)


@st.cache_data(ttl=120, max_entries=128, show_spinner=False)
def fetch_news(symbol, company=""):
    """Primary ticker.news → Yahoo RSS → independent Google News RSS."""
    warnings = []
    try:
        news = extract_news(yf.Ticker(symbol).news or [], symbol, datetime.now(UTC))
    except Exception as exc:
        warnings.append(f"yfinance news failed ({type(exc).__name__}); trying RSS.")
        news = []
    if news:
        return news, "yfinance", warnings
    warnings.append("No usable yfinance headlines; RSS fallback activated.")
    yahoo = "https://feeds.finance.yahoo.com/rss/2.0/headline?" + urlencode(
        {"s": symbol, "region": "US", "lang": "en-US"})
    name = clean(company, 200).replace('"', '')
    query = f'("{symbol}" OR "{name}") (stock OR shares OR earnings) when:30d' if name else f'"{symbol}" stock when:30d'
    google = "https://news.google.com/rss/search?" + urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    for source, url in [("Yahoo Finance RSS", yahoo), ("Google News RSS", google)]:
        try:
            news = fetch_rss_news(url, symbol, source)
        except Exception as exc:
            warnings.append(f"{source} unavailable ({type(exc).__name__}).")
            continue
        if news:
            return news, source, warnings
        warnings.append(f"{source} returned no usable headlines from the past 30 days.")
    return [], "Unavailable", warnings


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def fetch_data(symbol):
    """Cache public Yahoo data for five minutes to reduce request limits."""
    stock, warnings = yf.Ticker(symbol), []

    def optional(label, call, default):
        try:
            return call()
        except Exception:
            warnings.append(f"{label} unavailable; Yahoo may be rate limiting requests.")
            return default

    info = optional("Quote and company information", stock.get_info, {}) or {}
    if info.get("quoteType") not in (None, "EQUITY"):
        raise ValueError("Please enter an individual equity ticker.")
    price, quote_time = numeric(info.get("regularMarketPrice")), parse_time(info.get("regularMarketTime"))
    price_source = "Yahoo regular-market quote (may be delayed)"
    if price is None or price <= 0:
        history = optional("Price history", lambda: stock.history(period="5d", auto_adjust=False, timeout=20), None)
        if history is not None and not history.empty:
            closes = history["Close"].dropna()
            if not closes.empty:
                price = numeric(closes.iloc[-1])
                quote_time = None
                price_source = f"Daily close dated {closes.index[-1].date()}"
    if price is None or price <= 0:
        raise ValueError("No valid price found. Check the ticker or try again later.")
    if quote_time is None:
        warnings.append("Quote time unavailable; the price is not guaranteed live.")
    elif datetime.now(UTC) - quote_time > timedelta(days=5):
        warnings.append("Quote is more than five calendar days old.")

    income = optional("Annual revenues", lambda: stock.get_income_stmt(freq="yearly"), None)
    revenues, cagr = extract_revenues(income)
    if cagr is None:
        warnings.append("3-year revenue growth needs four valid consecutive annual revenues; shown as N/A.")
    pe = numeric(info.get("forwardPE"))
    pe = pe if pe is not None and pe > 0 else None
    # Yahoo's percentage-style 150 becomes 1.50x. Prefer direct book values.
    raw_de = numeric(info.get("debtToEquity"))
    de = raw_de / 100 if raw_de is not None and raw_de >= 0 else None
    de_source, balance_date = "Yahoo debtToEquity / 100", None
    balance = optional("Quarterly balance sheet", lambda: stock.get_balance_sheet(freq="quarterly"), None)
    if balance is not None and not balance.empty:
        date = sorted(balance.columns, reverse=True)[0]
        balance_date = str(date.date())
        debt = numeric(balance.loc["TotalDebt", date]) if "TotalDebt" in balance.index else None
        equity = numeric(balance.loc["StockholdersEquity", date]) if "StockholdersEquity" in balance.index else None
        if equity is not None and equity <= 0:
            de = None
            warnings.append("Book equity is nonpositive; conventional debt/equity is not meaningful.")
        elif debt is not None and debt >= 0 and equity is not None:
            de, de_source = debt / equity, "Quarterly TotalDebt / StockholdersEquity"
    if pe is None:
        warnings.append("Positive forward P/E unavailable.")
    if de is None:
        warnings.append("Meaningful debt/equity unavailable.")
    now = datetime.now(UTC)
    return {"ticker": symbol, "company": clean(info.get("longName") or symbol, 200),
            "sector": clean(info.get("sector"), 100), "currency": info.get("currency") or "Unknown",
            "reporting_currency": info.get("financialCurrency") or "Unknown", "price": price,
            "price_source": price_source, "quote_time_utc": quote_time.isoformat() if quote_time else None,
            "forward_pe": pe, "debt_to_equity_x": de, "debt_equity_source": de_source,
            "balance_sheet_date": balance_date, "revenue_cagr_3y": cagr,
            "annual_revenue": revenues, "warnings": warnings,
            "fetched_at_utc": now.isoformat()}


@st.cache_resource(show_spinner=False)
def sentiment_engine():
    """VADER bundles its lexicon: no credentials or runtime corpus downloads."""
    return SentimentIntensityAnalyzer()


def strategy_for_score(score):
    """Apply the requested strict cutoffs, including HOLD at +/-0.05."""
    if score > 0.05:
        return "LONG"
    if score < -0.05:
        return "SHORT"
    return "HOLD"


def score_news(news, analyzer):
    """Equal-weight mean of actual headlines; never divide by a fixed five."""
    scored = []
    for article in news:
        compound = float(analyzer.polarity_scores(article["headline"])["compound"])
        scored.append({**article, "compound": compound,
                       "sentiment": strategy_for_score(compound)})
    # With no news, 0 is a UI placeholder, not evidence of balanced sentiment.
    # Decimal arithmetic preserves exact threshold cases such as (0.4 - 0.3)/2.
    aggregate = float(sum((Decimal(str(row["compound"])) for row in scored), Decimal(0))
                      / Decimal(len(scored))) if scored else 0.0
    return {"strategy": strategy_for_score(aggregate), "score": aggregate,
            "articles": scored, "has_news": bool(scored)}


def progress_value(score):
    """Map signed sentiment [-1, 1] to Streamlit's progress range [0, 1]."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def metric_rows(data):
    def fmt(value, suffix="", percent=False):
        return "N/A" if value is None else (f"{value:.1%}" if percent else f"{value:,.2f}{suffix}")
    return [
        {"Metric": "Current / latest available price", "Value": f"{data['currency']} {fmt(data['price'])}"},
        {"Metric": "Forward P/E", "Value": fmt(data["forward_pe"], "x")},
        {"Metric": "Debt-to-equity", "Value": fmt(data["debt_to_equity_x"], "x")},
        {"Metric": "3-year revenue growth (CAGR)", "Value": fmt(data["revenue_cagr_3y"], percent=True)},
    ]


def main():
    st.set_page_config(page_title="HedgeAI — Free Sentiment", page_icon="📊", layout="wide")
    st.title("HedgeAI: 2-Year Long/Short Analysis Dashboard")
    st.caption("Free local VADER sentiment • Yahoo Finance data • No API key required")
    with st.sidebar:
        st.header("Research controls")
        with st.form("ticker_form"):
            symbol = st.text_input("Input Stock Ticker", value="AAPL", max_chars=20).strip().upper()
            submitted = st.form_submit_button("Analyze ticker", type="primary")
        st.caption("Financials cached for five minutes; news for two. Sentiment runs locally at no API cost.")
        st.caption("LONG: score > +0.05 | SHORT: score < −0.05 | HOLD: otherwise")

    if submitted:
        st.session_state.pop("hedge_report", None)
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,19}", symbol):
            st.error("Enter an equity ticker such as AAPL, BRK-B or 7203.T.")
            st.stop()
        with st.spinner(f"Fetching financials and news for {symbol}…"):
            try:
                data = fetch_data(symbol)
            except Exception:
                st.error("Could not retrieve a valid equity quote. Check the ticker or try again later.")
                st.stop()
            news, source, news_warnings = fetch_news(symbol, data["company"])
            data.update(news=news, news_source=source)
            data["warnings"].extend(news_warnings)
            if len(news) < 5:
                data["warnings"].append(f"Only {len(news)} recent usable headlines available across the selected source.")
        analysis = score_news(data["news"], sentiment_engine())
        st.session_state["hedge_report"] = {"data": data, "analysis": analysis}

    report = st.session_state.get("hedge_report")
    if not report:
        st.info("Enter a ticker in the sidebar and select Analyze ticker.")
        return
    data, analysis = report["data"], report["analysis"]
    st.subheader(f"{data['company']} ({data['ticker']})")
    st.caption(f"Data fetched: {data['fetched_at_utc']} | Quote time: {data['quote_time_utc'] or 'unknown'}")
    rows = metric_rows(data)
    for column, row in zip(st.columns(4), rows):
        column.metric(row["Metric"], row["Value"])
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"STRATEGY: {analysis['strategy']}")
        score = analysis["score"]
        st.metric("Aggregate news sentiment", f"{score:+.4f}" if analysis["has_news"] else "Unavailable")
        if analysis["has_news"]:
            st.progress(progress_value(score))
            st.caption("Bearish −1 ← Neutral 0 → Bullish +1. The midpoint is neutral; this is not confidence.")
        if not analysis["has_news"]:
            st.warning("HOLD: insufficient news. All news sources failed or returned no recent usable headlines; no measured sentiment is shown.")
        else:
            st.write(f"Equal-weight average of {len(analysis['articles'])} headline compound scores.")
        st.markdown("**How this signal is calculated**")
        st.write("• Score each headline locally with VADER, then average the compound scores.")
        st.write("• Apply strict thresholds: above +0.05 = LONG; below −0.05 = SHORT; otherwise HOLD.")
        st.write("• Financial metrics are context and do not change this sentiment-only rule.")
        st.markdown("**Core financial metrics**")
        st.table(pd.DataFrame(rows).set_index("Metric"))
    with right:
        st.markdown("**Latest usable headlines**")
        st.caption(f"News source: {data['news_source']}")
        if not analysis["articles"]:
            st.info("No dated headlines available from the past 30 days.")
        for article in analysis["articles"]:
            st.text(f"[{article['id']}] {article['headline']}")
            tone = {"LONG": "Positive", "SHORT": "Negative", "HOLD": "Neutral"}[article["sentiment"]]
            st.caption(f"{article['published_utc']} | {tone} | Score: {article['compound']:+.4f}")
            if article["url"]:
                st.link_button("Read source", article["url"])
    if analysis["articles"]:
        with st.expander("Headline sentiment scores"):
            st.dataframe(pd.DataFrame([
                {"ID": row["id"], "Headline": row["headline"], "Compound score": row["compound"]}
                for row in analysis["articles"]
            ]), hide_index=True, use_container_width=True)
    with st.expander("Data sources, annual revenue and limitations"):
        st.write(f"Price source: {data['price_source']}")
        st.write(f"Debt/equity source: {data['debt_equity_source']} | Statement date: {data['balance_sheet_date'] or 'unknown'}")
        st.write(f"Revenue reporting currency: {data['reporting_currency']}")
        if data["annual_revenue"]:
            st.table(pd.DataFrame(data["annual_revenue"]))
        for warning in data["warnings"]:
            st.write("• " + warning)
        st.write("Sources: ticker.news → Yahoo Finance RSS → Google News RSS. "
                 "Headlines are dated, deduplicated and sorted. "
                 "Coverage may be incomplete. VADER is a general English sentiment tool, "
                 "not a finance-specific model, and can misread financial wording or sarcasm.")
    st.caption("The title retains the 2-year dashboard context. Headline sentiment is not a "
               "24-month forecast or a complete valuation; the LONG/SHORT/HOLD signal is an unvalidated heuristic.")
    st.caption("⚠️ **Disclaimer:** This application is built strictly as an academic portfolio project by a Master of Finance student at UMass. All output, including the automated 'LONG', 'SHORT', or 'HOLD' directives and text sentiment scoring, is generated via an open-source data pipeline for educational and demonstration purposes only. This tool does not provide licensed investment, financial, tax, or legal advice. Do not use this application to make real-world trading or investment decisions.")



if __name__ == "__main__":
    main()

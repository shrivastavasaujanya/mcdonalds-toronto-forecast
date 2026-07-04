"""
Multi-agent stock briefing system.

Agents:
  Orchestrator — coordinates the pipeline
  Data Agent   — fetches & interprets price/volume per ticker (parallel)
  News Agent   — searches news for notable movers (parallel)
  Writer Agent — drafts the final briefing
  Critic Agent — checks quality, requests revision if needed
"""

import json
import re
import time
import concurrent.futures
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date
import anthropic
import pandas as pd
import yfinance as yf

client = anthropic.Anthropic()


def call_with_retry(fn, *args, retries=3, **kwargs):
    """Call an Anthropic API function, retrying on rate limit errors with backoff."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except anthropic.RateLimitError:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"Rate limit hit, retrying in {wait}s...", flush=True)
            time.sleep(wait)


EMAIL_FROM = "sauj.shrivastava@gmail.com"
EMAIL_TO = "sauj.shrivastava@gmail.com"
EMAIL_APP_PASSWORD = "hpog qohq ffml gsxs"
WATCHLIST_FALLBACK = ["AAPL", "MSFT", "NVDA"]
MAX_CRITIC_ROUNDS = 2


# ── Shared utilities ──────────────────────────────────────────────────────────

def get_top_movers(n=5):
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)["Close"]
        change = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100).dropna()
        return change.abs().nlargest(n).index.tolist()
    except Exception:
        return WATCHLIST_FALLBACK


def fetch_price_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        return {"error": f"no data for {ticker}"}
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    return {
        "ticker": ticker,
        "close": round(float(latest["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "volume": int(latest["Volume"]),
        "avg_volume_5d": int(hist["Volume"].mean()),
    }


def send_email(subject, body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())


# ── Agent 1: Data Agent ───────────────────────────────────────────────────────

def data_agent(ticker):
    """Fetches and interprets price data for one ticker. Runs in parallel."""
    raw = fetch_price_data(ticker)
    response = call_with_retry(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="You are a data analyst. Given stock price data as JSON, write one sentence summarizing the price action and whether volume is notable. Be factual, no filler.",
        messages=[{"role": "user", "content": json.dumps(raw)}],
    )
    return {
        "ticker": ticker,
        "raw": raw,
        "summary": response.content[0].text,
    }


# ── Agent 2: News Agent ───────────────────────────────────────────────────────

SEARCH_TOOLS = [{"type": "web_search_20250305", "name": "web_search"}]

def news_agent(data):
    """Searches for the news catalyst behind a notable mover. Runs in parallel."""
    ticker, price_summary = data["ticker"], data["summary"]
    messages = [{
        "role": "user",
        "content": f"Search for news explaining today's move in {ticker}. Context: {price_summary}. Return 1-2 sentences on the catalyst only.",
    }]
    while True:
        response = call_with_retry(
            client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system="You are a financial news researcher. Find the catalyst behind today's move in the given stock. Be concise and factual.",
            messages=messages,
            tools=SEARCH_TOOLS,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    return {"ticker": ticker, "news": text}


# ── Agent 3: Writer Agent ─────────────────────────────────────────────────────

def writer_agent(data_results, news_results, feedback=None):
    """Drafts the full briefing from aggregated data and news."""
    context = "Price & volume summaries:\n"
    for d in data_results:
        context += f"- {d['ticker']}: {d['summary']}\n"
    context += "\nNews catalysts for notable movers:\n"
    for n in news_results:
        context += f"- {n['ticker']}: {n['news']}\n"

    prompt = context
    if feedback:
        prompt += f"\n\nYour previous draft was rejected by the critic. Feedback: {feedback}\nPlease revise."

    response = call_with_retry(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system="""You are a stock market analyst writing a daily briefing for one reader.
For each notable name write 2-3 sentences: what happened, why if known, noise or worth tracking.
Skip names with no notable move. Format each entry with ticker, price, and % change in the header.
Be direct, no hedging.""",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Agent 4: Critic Agent ─────────────────────────────────────────────────────

def critic_agent(briefing, tickers):
    """Reviews the briefing. Returns (approved: bool, feedback: str)."""
    response = call_with_retry(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""You are a quality reviewer for stock briefings. Check:
1. Each notable mover has a reason (not just price stats)?
2. Tone is direct, no filler phrases?
3. Under 400 words?
Reply with JSON only: {"approved": true/false, "feedback": "..."}""",
        messages=[{"role": "user", "content": f"Tickers analyzed: {tickers}\n\nBriefing:\n{briefing}"}],
    )
    try:
        match = re.search(r'\{.*\}', response.content[0].text, re.DOTALL)
        result = json.loads(match.group()) if match else {}
        return result.get("approved", True), result.get("feedback", "")
    except Exception:
        return True, ""


# ── Orchestrator ──────────────────────────────────────────────────────────────

def orchestrate():
    # 1. Fetch top movers
    print("Fetching top S&P 500 movers...", flush=True)
    tickers = get_top_movers(5)
    print(f"Tickers: {', '.join(tickers)}\n", flush=True)

    # 2. Run all data agents in parallel
    print("Data agents running in parallel...", flush=True)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        data_results = list(pool.map(data_agent, tickers))

    # 3. Identify notable movers for news search
    notable = [
        d for d in data_results
        if abs(d["raw"].get("change_pct", 0)) > 2
        or d["raw"].get("volume", 0) > d["raw"].get("avg_volume_5d", 1) * 1.3
    ]
    print(f"Notable movers: {[d['ticker'] for d in notable]}", flush=True)

    # 4. Run all news agents in parallel
    print("News agents running in parallel...", flush=True)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        news_results = list(pool.map(news_agent, notable))

    # 5. Writer → Critic loop
    feedback = None
    briefing = None
    for round_num in range(MAX_CRITIC_ROUNDS + 1):
        print(f"Writer drafting (round {round_num + 1})...", flush=True)
        briefing = writer_agent(data_results, news_results, feedback)

        print("Critic reviewing...", flush=True)
        approved, feedback = critic_agent(briefing, tickers)

        if approved:
            print("Critic approved.\n", flush=True)
            break
        print(f"Critic rejected — feedback: {feedback}\n", flush=True)

    return briefing


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    briefing = orchestrate()
    print(briefing)
    subject = f"Stock Briefing (Multi-Agent) — {date.today().strftime('%b %d, %Y')}"
    send_email(subject, briefing)
    print("\nEmail sent.", flush=True)

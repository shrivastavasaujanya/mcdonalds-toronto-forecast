"""
Stock analyst agent, daily watchlist briefing.

Inner loop: the model decides which tools to call (price data, news search)
and in what order, looping until it judges it has enough to write the briefing.
You are not in this loop, the model drives it.

Outer loop (scheduling) is intentionally left out of this file, run it via
cron, Windows Task Scheduler, or just manually each morning to start.
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import anthropic
import pandas as pd
import yfinance as yf

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

EMAIL_FROM = "sauj.shrivastava@gmail.com"
EMAIL_TO = "sauj.shrivastava@gmail.com"
EMAIL_APP_PASSWORD = "hpog qohq ffml gsxs"


def send_email(subject: str, body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

WATCHLIST = ["AAPL", "MSFT", "NVDA"]  # fallback if mover fetch fails


def get_top_movers(n: int = 5) -> list[str]:
    """Return the n S&P 500 tickers with the largest absolute % move today."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)["Close"]
        change = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100).dropna()
        top = change.abs().nlargest(n).index.tolist()
        return top
    except Exception:
        return WATCHLIST


def get_price_data(ticker: str) -> str:
    """Fetch latest close, daily % change, and volume for a ticker."""
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        return json.dumps({"error": f"no data for {ticker}"})
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    return json.dumps({
        "ticker": ticker,
        "close": round(float(latest["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "volume": int(latest["Volume"]),
        "avg_volume_5d": int(hist["Volume"].mean()),
    })


TOOLS = [
    {
        "name": "get_price_data",
        "description": "Get latest closing price, daily percent change, and volume for a stock ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}
            },
            "required": ["ticker"],
        },
    },
    {"type": "web_search_20250305", "name": "web_search"},
]

SYSTEM_PROMPT = """You are a stock market analyst writing a daily briefing for one reader.

For each ticker on the watchlist:
1. Pull price data with get_price_data.
2. If the move is larger than 2 percent or volume is well above the 5 day average, search news for what happened.
3. Write 2 to 3 sentences per name covering what happened, why if known, and whether it looks like noise or something worth tracking.

Skip names with no notable move rather than padding with generic commentary. Be direct, no hedging."""


def run_briefing(watchlist=None):
    watchlist = watchlist or WATCHLIST
    messages = [{
        "role": "user",
        "content": f"Give me today's briefing on: {', '.join(watchlist)}",
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        client_tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "get_price_data":
                result = get_price_data(block.input["ticker"])
                client_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            # web_search runs server side automatically, no client handling needed

        if client_tool_results:
            messages.append({"role": "user", "content": client_tool_results})
        else:
            # only server side tool calls happened, nothing left for us to send back
            break

    final_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return final_text


if __name__ == "__main__":
    from datetime import date
    print("Fetching top 5 S&P 500 movers...", flush=True)
    movers = get_top_movers(5)
    print(f"Analyzing: {', '.join(movers)}\n", flush=True)
    briefing = run_briefing(movers)
    print(briefing)
    subject = f"Stock Briefing — {date.today().strftime('%b %d, %Y')}"
    send_email(subject, briefing)
    print("\nEmail sent.", flush=True)

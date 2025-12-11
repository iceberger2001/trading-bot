import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import *

# ----------------------------
# Bot universe (top 25)
# ----------------------------
UNIVERSE = [
    "NVDA","AXON","NFLX","ENPH","ANET",
    "AVGO","TTWO","PWR","FSLR","MU",
    "ALB","LLY","CMA","DRI","HWM",
    "CPRT","FITB","NTAP","TSLA","TYL",
    "LRCX","DECK","DVN","RCL","IDXX"
]

# ----------------------------
# Email function
# ----------------------------
def send_email(subject, message):
    if not EMAIL_ENABLED:
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())


# ----------------------------
# Indicator Calculation
# ----------------------------
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    if len(df) < 50:
        return None
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ----------------------------
# Strategy: breakout + RSI pullback
# ----------------------------
def generate_signal(df):
    if df is None:
        return None

    today = df.iloc[-1]

    uptrend = today["MA20"] > today["MA50"]
    rsi_buy = today["RSI"] < 55
    breakout = today["Close"] > df["Close"].rolling(20).max().iloc[-2]

    if uptrend and (rsi_buy or breakout):
        return "BUY"
    if today["Close"] < today["MA50"]:
        return "SELL"
    return None


# ----------------------------
# Main Bot
# ----------------------------
def run_bot():
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

    account = api.get_account()
    cash = float(account.cash)
    portfolio_value = float(account.portfolio_value)

    allocation = portfolio_value * 0.10  # 10% per position

    logs = ["Bot Run Starting...\n"]

    positions = {p.symbol: float(p.qty) for p in api.list_positions()}

    for ticker in UNIVERSE:
        df = load_data(ticker)
        signal = generate_signal(df)

        logs.append(f"{ticker}: {signal}")

        if signal == "BUY":
            price = df["Close"].iloc[-1]
            shares = int(allocation / price)

            if shares > 0:
                api.submit_order(
                    symbol=ticker,
                    side="buy",
                    type="market",
                    qty=shares
                )
                logs.append(f"BUY → {ticker} ({shares} shares)")

        elif signal == "SELL":
            if ticker in positions:
                qty = positions[ticker]
                api.submit_order(
                    symbol=ticker,
                    side="sell",
                    type="market",
                    qty=qty
                )
                logs.append(f"SELL → {ticker} (ALL {qty} shares)")

    log_text = "\n".join(logs)
    print(log_text)

    send_email(
        "Trading Bot Daily Report",
        log_text
    )


if __name__ == "__main__":
    run_bot()


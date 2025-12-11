import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.mime.text import MIMEText

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import *

# ----------------------------
# Universe of tickers (top 25)
# ----------------------------
UNIVERSE = [
    "NVDA","AXON","NFLX","ENPH","ANET",
    "AVGO","TTWO","PWR","FSLR","MU",
    "ALB","LLY","CMA","DRI","HWM",
    "CPRT","FITB","NTAP","TSLA","TYL",
    "LRCX","DECK","DVN","RCL","IDXX"
]

# ----------------------------
# OPTIONAL EMAIL NOTIFICATIONS
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
# Load 6 months of data
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
# BUY/SELL Signal Generator
# ----------------------------
def generate_signal(df):
    if df is None or len(df) < 50:
        return None

    today = df.iloc[-1]

    if (
        pd.isna(today["MA20"]) or 
        pd.isna(today["MA50"]) or 
        pd.isna(today["RSI"])
    ):
        return None

    # Force booleans
    uptrend = bool(today["MA20"] > today["MA50"])
    rsi_buy = bool(today["RSI"] < 55)

    prev_20_high = float(df["Close"].rolling(20).max().iloc[-2])
    breakout = bool(today["Close"] > prev_20_high)

    if uptrend and (rsi_buy or breakout):
        return "BUY"

    # SELL if trend breaks
    if today["Close"] < today["MA50"]:
        return "SELL"

    return None

# ----------------------------
# Main Trading Bot
# ----------------------------
def run_bot():
    logs = ["🚀 BOT RUN STARTED\n"]

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    account = client.get_account()

    portfolio_value = float(account.portfolio_value)
    allocation = portfolio_value * 0.10  # 10% per stock

    # Current positions
    positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}

    for ticker in UNIVERSE:
        df = load_data(ticker)
        signal = generate_signal(df)

        logs.append(f"{ticker} → {signal}")

        # ---------- BUY ----------
        if signal == "BUY":
            price = df["Close"].iloc[-1]
            qty = int(allocation / price)

            if qty > 0:
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                client.submit_order(order)
                logs.append(f"   🟢 BUY {qty} shares @ {price}")

        # ---------- SELL ----------
        elif signal == "SELL":
            if ticker in positions:
                qty = int(positions[ticker])

                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                client.submit_order(order)
                logs.append(f"   🔴 SELL {qty} shares")

    # Print logs to GitHub Actions output
    final_log = "\n".join(logs)
    print(final_log)

    # Send email summary
    send_email("Trading Bot Report", final_log)


if __name__ == "__main__":
    run_bot()

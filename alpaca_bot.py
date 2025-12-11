import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.mime.text import MIMEText

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import *

# -------------------------------------
# Bot Universe (Top 25 Backtest Winners)
# -------------------------------------
UNIVERSE = [
    "NVDA","AXON","NFLX","ENPH","ANET",
    "AVGO","TTWO","PWR","FSLR","MU",
    "ALB","LLY","CMA","DRI","HWM",
    "CPRT","FITB","NTAP","TSLA","TYL",
    "LRCX","DECK","DVN","RCL","IDXX"
]


# -------------------------------------
# Email Notifications
# -------------------------------------
def send_email(subject, message):
    if not EMAIL_ENABLED:
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    except Exception as e:
        print("Email failed:", e)


# -------------------------------------
# RSI Calculation
# -------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# -------------------------------------
# Download Data
# -------------------------------------
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

    if df is None or len(df) < 50:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])

    return df


# -------------------------------------
# Trading Signal Logic
# -------------------------------------
def generate_signal(df):
    if df is None or len(df) < 50:
        return None

    today = df.iloc[-1]

    # Extract safe scalar values (prevents "Series is ambiguous" errors)
    try:
        ma20 = float(today["MA20"])
        ma50 = float(today["MA50"])
        rsi_val = float(today["RSI"])
        close = float(today["Close"])
        prev20 = float(df["Close"].rolling(20).max().iloc[-2])
    except:
        return None

    # Skip NaN rows
    if any(np.isnan([ma20, ma50, rsi_val, close, prev20])):
        return None

    uptrend = ma20 > ma50
    rsi_pullback = rsi_val < 55
    breakout = close > prev20

    # BUY LOGIC
    if uptrend and (rsi_pullback or breakout):
        return "BUY"

    # SELL LOGIC
    if close < ma50:
        return "SELL"

    return None


# -------------------------------------
# Main Trading Bot
# -------------------------------------
def run_bot():
    logs = ["Bot run starting...\n"]

    # Initialize API
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)

    allocation = portfolio_value * 0.10  # 10% per position

    # Current positions
    positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}

    for ticker in UNIVERSE:
        df = load_data(ticker)
        signal = generate_signal(df)

        logs.append(f"{ticker}: {signal}")

        if signal == "BUY":
            price = df["Close"].iloc[-1]
            qty = int(allocation / price)

            if qty > 0:
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                client.submit_order(order)
                logs.append(f"BUY → {ticker} ({qty} shares)")

        elif signal == "SELL":
            if ticker in positions:
                qty = int(positions[ticker])
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                client.submit_order(order)
                logs.append(f"SELL → {ticker} (ALL {qty} shares)")

    # Output logs
    log_text = "\n".join(logs)
    print("\n----- BOT LOGS -----\n")
    print(log_text)

    # Email results
    send_email("Daily Trading Bot Report", log_text)


# -------------------------------------
# Run Bot
# -------------------------------------
if __name__ == "__main__":
    run_bot()


import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.mime.text import MIMEText

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

from config import *

# -------------------------------------
# Universe (Top 25 from your backtest)
# -------------------------------------
UNIVERSE = [
    "NVDA","AXON","NFLX","ENPH","ANET",
    "AVGO","TTWO","PWR","FSLR","MU",
    "ALB","LLY","CMA","DRI","HWM",
    "CPRT","FITB","NTAP","TSLA","TYL",
    "LRCX","DECK","DVN","RCL","IDXX"
]


# -------------------------------------
# EMAIL SENDER
# -------------------------------------
def send_email(subject: str, message: str):
    if not EMAIL_ENABLED:
        print("Email disabled — skipping send.")
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("Email sent successfully.")
    except Exception as e:
        print("Email failed:", e)


# -------------------------------------
# DATA LOADING
# -------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

    if df is None or df.empty:
        return None
    if len(df) < 50:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])

    return df


# -------------------------------------
# SIGNAL GENERATION (fixed version)
# -------------------------------------
def generate_signal(df):
    if df is None or len(df) < 60:
        return None

    # Extract scalars — FIXES ambiguous truth-value errors
    ma20 = float(df["MA20"].iloc[-1])
    ma50 = float(df["MA50"].iloc[-1])
    rsi_val = float(df["RSI"].iloc[-1])
    close = float(df["Close"].iloc[-1])
    prev20 = float(df["Close"].rolling(20).max().iloc[-2])

    # Skip if any indicator is missing
    if any(np.isnan([ma20, ma50, rsi_val, close, prev20])):
        return None

    uptrend = ma20 > ma50
    rsi_pullback = rsi_val < 55
    breakout = close > prev20

    if uptrend and (rsi_pullback or breakout):
        return "BUY"

    if close < ma50:
        return "SELL"

    return None


# -------------------------------------
# TRADING BOT
# -------------------------------------
def run_bot():
    print("Bot starting...")

    # Alpaca client
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    acct = client.get_account()
    portfolio_value = float(acct.portfolio_value)
    
    allocation = portfolio_value * 0.10  # Use 10% per stock

    logs = ["----- BOT LOGS -----", "Bot run starting...\n"]

    # Current positions
    positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}

    for ticker in UNIVERSE:
        df = load_data(ticker)
        signal = generate_signal(df)

        logs.append(f"{ticker}: {signal}")

        # No action
        if signal is None:
            continue

        price = float(df["Close"].iloc[-1])

        # BUY
        if signal == "BUY":
            qty = allocation / price  # fractional shares allowed

            order = OrderRequest(
                symbol=ticker,
                notional=allocation,  # fractional notional order!
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            logs.append(f"BUY → {ticker} (${allocation:.2f} notional)")

        # SELL
        elif signal == "SELL" and ticker in positions:
            qty = positions[ticker]

            order = OrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            logs.append(f"SELL → {ticker} ({qty} shares)")

    # Print logs
    log_text = "\n".join(logs)
    print(log_text)

    # Send email report
    send_email("Daily Trading Bot Report", log_text)


# -------------------------------------
# ENTRY POINT
# -------------------------------------
if __name__ == "__main__":
    run_bot()

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
# Email Alerts
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
# RSI Calculation
# ----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ----------------------------
# Load historical data
# ----------------------------
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    if len(df) < 50:
        return None
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    return df


# ----------------------------
# Strategy logic
# ----------------------------
def generate_signal(df):
    if df is None:
        return None

    today = df.iloc[-1]

    # ensure these are scalar booleans
    uptrend = bool(today["MA20"] > today["MA50"])
    rsi_buy = bool(today["RSI"] < 55)

    # breakout: close > previous 20-day high
    prev_20_high = df["Close"].rolling(20).max().iloc[-2]
    breakout = bool(today["Close"] > prev_20_high)

    if uptrend and (rsi_buy or breakout):
        return "BUY"

    # SELL when price loses trend
    if today["Close"] < today["MA50"]:
        return "SELL"

    return None



# ----------------------------
# Main Bot Logic
# ----------------------------
def run_bot():

    # Connect to Alpaca
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)

    allocation = portfolio_value * 0.10

    logs = ["Bot Run Starting...\n"]

    # Get open positions
    positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}

    for ticker in UNIVERSE:

        df = load_data(ticker)
        signal = generate_signal(df)

        logs.append(f"{ticker}: {signal}")

        # BUY logic
        if signal == "BUY":
            price = df["Close"].iloc[-1]
            shares = int(allocation / price)

            if shares > 0:
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                client.submit_order(order)
                logs.append(f"BUY → {ticker} ({shares} shares)")

        # SELL logic
        elif signal == "SELL" and ticker in positions:
            qty = int(float(positions[ticker]))
            order = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            logs.append(f"SELL → {ticker} (ALL {qty} shares)")

    # Log output
    log_text = "\n".join(logs)
    print(log_text)

    # Email daily report
    send_email("Trading Bot Daily Report", log_text)


if __name__ == "__main__":
    run_bot()


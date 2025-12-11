import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

import os
import traceback

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
EMAIL_ENABLED = False  # set True if you later configure email

STOCKS = [
    "NVDA","AXON","NFLX","ENPH","ANET","AVGO","TTWO","PWR","FSLR","MU",
    "ALB","LLY","CMA","DRI","HWM","CPRT","FITB","NTAP","TSLA","TYL",
    "LRCX","DECK","DVN","RCL","IDXX"
]

# ──────────────────────────────────────────────
# EMAIL (disabled unless EMAIL_ENABLED=True)
# ──────────────────────────────────────────────

def send_email(subject, body):
    if not EMAIL_ENABLED:
        return  # Skip email if disabled
    print("\n[EMAIL WOULD BE SENT HERE]\n")

# ──────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────

def compute_indicators(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # RSI Calculation
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────
# GENERATE TRADING SIGNAL
# ──────────────────────────────────────────────

def generate_signal(df):
    """
    This version pulls the last row cleanly and avoids Series ambiguity.
    """
    today = df.iloc[-1]

    ma20 = float(today["MA20"])
    ma50 = float(today["MA50"])
    rsi = float(today["RSI"])
    close = float(today["Close"])

    # Prior 20-day high for breakout detection
    prev20 = float(df["Close"].rolling(20).max().iloc[-2])

    # Conditions
    uptrend = close > ma20 > ma50
    rsi_buy = rsi < 35
    breakout = close > prev20

    # SELL signals
    downtrend = close < ma50
    rsi_sell = rsi > 70

    if uptrend and (rsi_buy or breakout):
        return "BUY"
    if downtrend or rsi_sell:
        return "SELL"
    return None


# ──────────────────────────────────────────────
# MAIN BOT EXECUTION
# ──────────────────────────────────────────────

def run_bot():
    logs = []
    logs.append("Bot starting...")

    try:
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    except Exception as e:
        logs.append(f"Failed to connect to Alpaca: {e}")
        print("\n".join(logs))
        return

    # Get account info
    try:
        account = client.get_account()
        portfolio_value = float(account.portfolio_value)
    except Exception as e:
        logs.append(f"Could not fetch Alpaca account: {e}")
        print("\n".join(logs))
        return

    if portfolio_value <= 0:
        logs.append("Portfolio value is $0 → cannot trade. Reset Alpaca paper account.")
        print("\n".join(logs))
        return

    allocation = portfolio_value * 0.10  # 10% notional per BUY trade

    for ticker in STOCKS:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if df.empty:
                logs.append(f"{ticker}: No data")
                continue

            df = compute_indicators(df)
            if df.empty:
                logs.append(f"{ticker}: Not enough data after indicator calc")
                continue

            signal = generate_signal(df)
            price = float(df["Close"].iloc[-1])

            logs.append(f"{ticker}: {signal}")

            # ──────────────────────
            # HANDLE BUY SIGNAL
            # ──────────────────────
            if signal == "BUY":

                if price <= 0:
                    logs.append(f"SKIP BUY {ticker} — invalid price {price}")
                    continue

                notional = allocation

                if notional < 1:
                    logs.append(f"SKIP BUY {ticker} — notional too small (${notional:.2f})")
                    continue

                logs.append(
                    f"BUY → {ticker} | price ${price:.2f} | notional ${notional:.2f}"
                )

                order = OrderRequest(
                    symbol=ticker,
                    notional=notional,       # fractional shares supported
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )

                try:
                    client.submit_order(order)
                    logs.append(f"ORDER PLACED: BUY {ticker}")
                except Exception as e:
                    logs.append(f"ORDER FAILED BUY {ticker}: {e}")

            # ──────────────────────
            # HANDLE SELL SIGNAL
            # ──────────────────────
            if signal == "SELL":

                try:
                    pos = client.get_open_position(ticker)
                    qty = float(pos.qty)

                    if qty <= 0:
                        logs.append(f"SKIP SELL {ticker} — no shares")
                        continue

                    order = OrderRequest(
                        symbol=ticker,
                        qty=qty,
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )

                    client.submit_order(order)
                    logs.append(f"ORDER PLACED: SELL {ticker} ({qty} shares)")

                except Exception:
                    logs.append(f"SKIP SELL {ticker} — no open position")

        except Exception as e:
            logs.append(f"ERROR processing {ticker}: {e}")
            logs.append(traceback.format_exc())

    # Print and optionally email logs
    log_output = "\n".join(logs)
    print("\n----- BOT LOGS -----\n" + log_output)

    send_email("Daily Trading Bot Report", log_output)


# ENTRYPOINT
if __name__ == "__main__":
    run_bot()

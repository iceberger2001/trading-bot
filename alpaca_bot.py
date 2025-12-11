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
# Config fallbacks / safety
# ----------------------------
# If any of these aren't defined in config.py, set safe defaults
EMAIL_ENABLED = globals().get("EMAIL_ENABLED", False)
ALPACA_PAPER = globals().get("ALPACA_PAPER", True)

# ----------------------------
# Bot universe (top 25)
# ----------------------------
UNIVERSE = [
    "NVDA", "AXON", "NFLX", "ENPH", "ANET",
    "AVGO", "TTWO", "PWR", "FSLR", "MU",
    "ALB", "LLY", "CMA", "DRI", "HWM",
    "CPRT", "FITB", "NTAP", "TSLA", "TYL",
    "LRCX", "DECK", "DVN", "RCL", "IDXX",
]

# ----------------------------
# Email function
# ----------------------------
def send_email(subject: str, message: str) -> None:
    if not EMAIL_ENABLED:
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())


# ----------------------------
# Indicator Calculation
# ----------------------------
def load_data(ticker: str) -> pd.DataFrame | None:
    """
    Load ~6 months of daily data from yfinance and compute indicators.
    """
    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty or len(df) < 60:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])

    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ----------------------------
# Strategy: breakout + RSI pullback
# ----------------------------
def generate_signal(df: pd.DataFrame) -> str | None:
    """
    Returns "BUY", "SELL", or None for no action.
    Uses last bar only, with simple floats to avoid pandas ambiguity.
    """
    if df is None or len(df) < 60:
        return None

    # If indicators aren't ready, skip
    if (
        pd.isna(df["MA20"].iloc[-1])
        or pd.isna(df["MA50"].iloc[-1])
        or pd.isna(df["RSI"].iloc[-1])
    ):
        return None

    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    rsi_val = df["RSI"].iloc[-1]
    close = df["Close"].iloc[-1]

    # 20-day high *before* today
    prev20_high = df["Close"].rolling(20).max().iloc[-2]

    uptrend = ma20 > ma50
    rsi_pullback = rsi_val < 55
    breakout = close > prev20_high

    if uptrend and (rsi_pullback or breakout):
        return "BUY"

    if close < ma50:
        return "SELL"

    return None


# ----------------------------
# Main Bot
# ----------------------------
def run_bot() -> None:
    logs: list[str] = []
    logs.append("Bot run starting...")

    # --- Alpaca client ---
    trading_client = TradingClient(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        paper=ALPACA_PAPER,
    )

    # --- Account + portfolio info ---
    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)

    # 10% of portfolio per position (notional, for fractional shares)
    target_allocation = portfolio_value * 0.10
    available_cash = cash

    logs.append(f"Portfolio value: {portfolio_value:.2f}")
    logs.append(f"Cash: {cash:.2f}")
    logs.append(f"Target notional per BUY: {target_allocation:.2f}")

    # --- Existing positions ---
    positions_raw = trading_client.get_all_positions()
    positions: dict[str, float] = {
        p.symbol: float(p.qty) for p in positions_raw
    }

    logs.append(f"Existing positions: {positions}")

    # --- Loop universe ---
    for ticker in UNIVERSE:
        df = load_data(ticker)
        if df is None:
            logs.append(f"{ticker}: skipped (not enough data)")
            continue

        signal = generate_signal(df)
        logs.append(f"{ticker}: {signal}")

        # ---------------- BUY logic (fractional via notional) ----------------
        if signal == "BUY":
            if available_cash <= 0:
                logs.append(f"{ticker}: SKIP BUY (no available cash)")
                continue

            if ticker in positions:
                logs.append(
                    f"{ticker}: SKIP BUY (already holding {positions[ticker]} shares)"
                )
                continue

            # Don't exceed available cash, keep a little buffer
            notional = min(target_allocation, available_cash * 0.95)

            # If too small, skip
            if notional < 5:  # don't bother with tiny orders
                logs.append(
                    f"{ticker}: SKIP BUY (notional {notional:.2f} too small)"
                )
                continue

            order = MarketOrderRequest(
                symbol=ticker,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                notional=round(notional, 2),
            )

            try:
                trading_client.submit_order(order)
                logs.append(f"BUY → {ticker} (${notional:.2f} notional)")
                available_cash -= notional
            except Exception as e:
                logs.append(f"ERROR submitting BUY for {ticker}: {e}")

        # ---------------- SELL logic ----------------
        elif signal == "SELL":
            if ticker not in positions:
                logs.append(f"{ticker}: SKIP SELL (no existing position)")
                continue

            qty = positions[ticker]
            if qty <= 0:
                logs.append(f"{ticker}: SKIP SELL (qty <= 0)")
                continue

            order = MarketOrderRequest(
                symbol=ticker,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                qty=qty,  # float qty → fractional sells ok
            )

            try:
                trading_client.submit_order(order)
                logs.append(f"SELL → {ticker} ({qty} shares)")
            except Exception as e:
                logs.append(f"ERROR submitting SELL for {ticker}: {e}")

        # If signal is None, do nothing

    # --- Logging + email ---
    log_text = "----- BOT LOGS -----\n" + "\n".join(logs)
    print(log_text)

    send_email("Daily Trading Bot Report", log_text)


if __name__ == "__main__":
    run_bot()

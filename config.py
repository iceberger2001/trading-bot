import os

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Paper or live
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# Bot settings
UNIVERSE = [
    "NVDA", "AXON", "NFLX", "ENPH", "ANET",
    "AVGO", "TTWO", "PWR", "FSLR", "MU",
    "ALB", "LLY", "CMA", "DRI", "HWM",
    "CPRT", "FITB", "NTAP", "TYL", "LRCX",
    "DECK", "RCL", "ZBRA", "MAA", "GNRC"
]

RISK_PER_TRADE = 0.10     # 10% of equity

# -----------------------------
# Email System
# -----------------------------
EMAIL_ENABLED = False      # set to True after configuring email

EMAIL_FROM = ""
EMAIL_TO = ""
SMTP_SERVER = ""
SMTP_PORT = 465
SMTP_USER = ""
SMTP_PASS = ""


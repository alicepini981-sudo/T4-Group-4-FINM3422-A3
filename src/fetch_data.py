# Installed and imported data source: Public API - Yahoo Finance
import yfinance as yf
import pandas as pd

ticker = yf.Ticker("BHP.AX")
df = ticker.history(period="2y")  # 2 years of daily data

# Keep only what you need
df = df[["Close", "Volume"]]
df.index = pd.to_datetime(df.index)

# -------------------------------------------------------
# Clean and Align Equity Price Data
# -------------------------------------------------------

# STEPS TAKEN TO CLEAN AND ALIGN THE DATA

# 1. Strip timezone info so dates are plain and consistent as yfinance returns AEST timezone-aware
df.index = df.index.tz_localize(None)

# 2. Remove any missing close prices (e.g. trading halts)
df = df.dropna(subset=["Close"])

# 3. Remove any duplicate dates as occasionally yfinance returns duplicate rows
df = df[~df.index.duplicated(keep="first")]

# 4. Sort in chronological order (oldest to newest)
df = df.sort_index()

# 5. Confirm the result
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total trading days: {len(df)}")
print(f"Missing values: {df.isnull().sum().to_dict()}")
print(df.tail())

# Save locally to 'data' folder so never need to call the API again
df.to_csv("data/Market_Data_BHP_prices.csv")  # equity price data (cached)

print(f"Downloaded {len(df)} rows")
print(df.tail())
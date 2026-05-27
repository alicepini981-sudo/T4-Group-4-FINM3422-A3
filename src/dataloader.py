from pathlib import Path
import os
import pandas as pd

TICKERS = {"BHP": "BHP.AX", "CBA": "CBA.AX", "CSL": "CSL.AX", "WOW": "WOW.AX"}

def load_asx_data(tickers: dict = TICKERS, data_dir: str = "../data") -> dict:
    all_data = {}
    for name, ticker_code in tickers.items():
        data_path = Path(data_dir) / f"Market_Data_{name}_prices.csv"
        if data_path.exists():
            df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        else:
            try:
                import yfinance as yf
            except ImportError as e:
                raise ImportError(f"yfinance not installed and no local data for {name}.") from e
            df = yf.Ticker(ticker_code).history(period="2y")[["Close", "Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["Close"]).pipe(lambda d: d[~d.index.duplicated(keep="first")]).sort_index()
            df = df.reset_index().rename(columns={"index": "Date"})
            os.makedirs(data_path.parent, exist_ok=True)
            df.to_csv(data_path, index=False)
        df.index = pd.to_datetime(df["Date"])
        all_data[name] = df
    return all_data
import yfinance as yf
import pandas as pd
import os

def get_data_with_cache(tickers, start, end, cache_file='sp500_stocks_only_data.pkl'):
    """Fetch data from Yahoo Finance or load from a local cache file."""
    if os.path.exists(cache_file):
        print("Loading data from local cache (fast)...")
        raw_data = pd.read_pickle(cache_file)
    else:
        print(f"Downloading data for {len(tickers)} assets...")
        raw_data = yf.download(
            tickers, start=start, end=end, 
            group_by='ticker', progress=True, threads=True
        )
        raw_data.to_pickle(cache_file)
    return raw_data

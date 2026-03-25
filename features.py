import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import config

def calculate_technical_indicators(df, benchmark_series):
    """Compute rolling technical indicators for a given dataframe."""
    df['Log_Ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std()

    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, np.nan).fillna(1e-6)
    df['RSI'] = (100 - (100 / (1 + gain / loss))) / 100.0

    exp12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = macd - signal

    sma20 = df['Adj Close'].rolling(window=20).mean()
    std20 = df['Adj Close'].rolling(window=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_Percent_B'] = (df['Adj Close'] - lower) / (upper - lower + 1e-6)

    df['Momentum_10D'] = df['Adj Close'] / df['Adj Close'].shift(10) - 1
    df['Dist_MA50'] = df['Adj Close'] / df['Adj Close'].rolling(window=50).mean() - 1

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'].shift())
    low_close = np.abs(df['Low'] - df['Adj Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['ATR_Rel'] = df['ATR'] / df['Adj Close']

    if benchmark_series is not None:
        aligned_bench = benchmark_series.reindex(df.index).fillna(method='ffill')
        df['SPY_Corr'] = df['Log_Ret'].rolling(window=30).corr(aligned_bench).fillna(0)
    else:
        df['SPY_Corr'] = 0.0

    return df

def process_features(raw_data, tickers):
    """Process all tickers to generate scaled feature arrays and raw returns."""
    features_dict = {}
    valid_tickers = []
    
    if config.BENCHMARK_TICKER in raw_data.columns.levels[0]:
        bench_df = raw_data[config.BENCHMARK_TICKER].copy()
        col = 'Adj Close' if 'Adj Close' in bench_df else 'Close'
        bench_ret = np.log(bench_df[col] / bench_df[col].shift(1))
    else:
        bench_ret = None

    print("Engineering features...")
    for ticker in tqdm(tickers):
        if ticker == config.BENCHMARK_TICKER or ticker in config.BLACKLIST:
            continue
            
        try:
            if ticker not in raw_data.columns.levels[0]: continue
            df = raw_data[ticker].copy()
            
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns: df['Adj Close'] = df['Close']
                else: continue
            
            if len(df) < 252: continue 

            df = calculate_technical_indicators(df, bench_ret)
            cols = ['Log_Ret', 'Volatility', 'RSI', 'MACD_Diff', 'BB_Percent_B', 
                    'Momentum_10D', 'Dist_MA50', 'ATR_Rel', 'SPY_Corr']
            
            df_clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(df_clean) < config.SEQUENCE_LENGTH: continue
            
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(df_clean.values)
            
            features_dict[ticker] = {
                'data': scaled_data,
                'index': df_clean.index,
                'raw_returns': df['Log_Ret'].reindex(df_clean.index)
            }
            valid_tickers.append(ticker)
        except Exception:
            continue
            
    return features_dict, valid_tickers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import random
import os
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

STOCKS = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'CRM', 'AMD', 'ADBE', 'CSCO', 'ACN', 'INTC', 'ORCL', 'QCOM', 'TXN', 'IBM', 'AMAT', 'NOW', 'INTU', 'UBER', 'MU', 'PANW', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'ROP', 'APH', 'NXPI', 'MCHP', 'FTNT', 'MSI', 'TEL', 'IT', 'HPQ', 'GLW', 'TRMB', 'STX', 'WDC', 'NTAP', 'PSTG', 'ANET', 'SMCI', 'PLTR', 'DELL', 'HPE', 'FFIV', 'JNPR', 'KEYS', 'TYL', 'ZBRA', 'AKAM', 'GEN',
    # Healthcare
    'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'AMGN', 'ISRG', 'PFE', 'DHR', 'ABT', 'BMY', 'VRTX', 'REGN', 'SYK', 'GILD', 'ELV', 'MDT', 'ZTS', 'BSX', 'BDX', 'CI', 'CVS', 'HCA', 'MCK', 'COR', 'HUM', 'EW', 'CNC', 'IQV', 'A', 'RMD', 'IDXX', 'DXCM', 'BIIB', 'MTD', 'STE', 'TFX', 'COO', 'WAT', 'ALGN', 'HOLX', 'DGX', 'LH', 'RVTY', 'PODD', 'TECH', 'CRL', 'BIO', 'WST',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SPGI', 'AXP', 'PGR', 'CB', 'MMC', 'SCHW', 'KKR', 'BX', 'ICE', 'CME', 'MCO', 'AON', 'USB', 'PNC', 'TRV', 'TFC', 'AFL', 'BK', 'ALL', 'COF', 'MET', 'AMP', 'HIG', 'DFS', 'FITB', 'STT', 'TROW', 'RJF', 'NDAQ', 'WTW', 'BRO', 'PFG', 'CINF', 'WRB', 'L', 'AJG', 'RE', 'AIZ', 'GL', 'BEN', 'IVZ',
    # Consumer Discretionary
    'HD', 'COST', 'MCD', 'DIS', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'MAR', 'LULU', 'CMG', 'HLT', 'YUM', 'LEN', 'DHI', 'ORLY', 'ROST', 'TSCO', 'AZO', 'ULTA', 'EXPE', 'RCL', 'CCL', 'NCLH', 'GPC', 'KMX', 'DRI', 'DPZ', 'MGM', 'WYNN', 'LVS', 'BBY', 'HAS', 'MAT', 'POOL', 'VFC', 'TPR', 'RL', 'PVH', 'HOG',
    # Consumer Staples
    'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'EL', 'CL', 'GIS', 'MDLZ', 'TGT', 'KMB', 'DG', 'DLTR', 'KR', 'ADM', 'STZ', 'TSN', 'HSY', 'K', 'MKC', 'CAG', 'CHD', 'CLX', 'HRL', 'CPB', 'SJM', 'TAP',
    # Industrials
    'CAT', 'GE', 'UNP', 'HON', 'UPS', 'BA', 'RTX', 'LMT', 'DE', 'ADP', 'ETN', 'ITW', 'WM', 'GD', 'FDX', 'NOC', 'CSX', 'NSC', 'EMR', 'PH', 'PCAR', 'GWW', 'TT', 'CARR', 'OTIS', 'ROK', 'CMI', 'AME', 'VRSK', 'FAST', 'EFX', 'URI', 'PWR', 'DOV', 'XYL', 'WAB', 'IR', 'HII', 'LDOS', 'AXON', 'EXPD', 'JBHT', 'CHRW', 'KNX', 'ODFL', 'SAIA', 'ARCB', 'LSTR', 'DAL', 'UAL', 'AAL', 'LUV',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI', 'WMB', 'BKR', 'HAL', 'DVN', 'TRGP', 'FANG', 'CTRA', 'EQT', 'APA', 'OVV', 'MRO', 'HES',
    # Materials
    'LIN', 'SHW', 'FCX', 'APD', 'ECL', 'NEM', 'DOW', 'DD', 'CTVA', 'PPG', 'MLM', 'VMC', 'ALB', 'FMC', 'LYB', 'CE', 'EMN', 'CF', 'MOS',
    # Utilities
    'NEE', 'SO', 'DUK', 'AEP', 'SRE', 'PEG', 'WEC', 'ES', 'XEL', 'ED', 'EIX', 'DTE', 'ETR', 'PPL', 'CMS', 'AEE', 'ATO', 'LNT', 'EVRG', 'CNP', 'NI', 'PNW', 'NRG',
    # Real Estate
    'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'VICI', 'AVB', 'EQR', 'CBRE', 'CSGP', 'SUI', 'INVH', 'MAA', 'ESS', 'UDR', 'KIM'
]

TICKERS = list(set(STOCKS))

BENCHMARK_TICKER = 'SPY'
if BENCHMARK_TICKER not in TICKERS:
    TICKERS.append(BENCHMARK_TICKER)

START_DATE = '2020-01-01'  
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

TRAIN_WINDOW = 1008     
REBALANCE_FREQ = 63     

# --- NEW PARAMETER: Transaction Cost ---
TRANSACTION_COST = 0.001 # 0.1% cost per trade (covers slippage and fees)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 9        
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2          
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 10             
TARGET_CLUSTERS = 30   

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Running on device: {DEVICE}")
print(f"Total Assets in Universe: {len(TICKERS)}")

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def calculate_technical_indicators(df, benchmark_series):
    df['Log_Ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std()

    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, np.nan).fillna(1e-6)
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'] / 100.0

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

    sma50 = df['Adj Close'].rolling(window=50).mean()
    df['Dist_MA50'] = df['Adj Close'] / sma50 - 1

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

def get_data_with_cache(tickers, start, end):
    cache_file = 'sp500_stocks_only_data.pkl'
    if os.path.exists(cache_file):
        print("Loading data from local cache (fast)...")
        raw_data = pd.read_pickle(cache_file)
    else:
        print(f"Downloading data for {len(tickers)} assets...")
        raw_data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=True, threads=True)
        raw_data.to_pickle(cache_file)
    return raw_data

def process_features(raw_data, tickers):
    features_dict = {}
    valid_tickers = []
    
    if BENCHMARK_TICKER in raw_data.columns.levels[0]:
        bench_df = raw_data[BENCHMARK_TICKER].copy()
        col = 'Adj Close' if 'Adj Close' in bench_df else 'Close'
        bench_ret = np.log(bench_df[col] / bench_df[col].shift(1))
    else:
        bench_ret = None

    BLACKLIST = ['BIL', 'SHV', 'MUB', 'AGG', 'BND', 'SPY', 'QQQ', 'DIA', 'IWM'] 

    print("Engineering features...")
    for ticker in tqdm(tickers):
        if ticker == BENCHMARK_TICKER or ticker in BLACKLIST:
            continue
            
        try:
            if ticker not in raw_data.columns.levels[0]: continue
            df = raw_data[ticker].copy()
            
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns: df['Adj Close'] = df['Close']
                else: continue
            
            if len(df) < 252: continue 

            df = calculate_technical_indicators(df, bench_ret)
            
            cols = [
                'Log_Ret', 'Volatility', 'RSI', 'MACD_Diff', 
                'BB_Percent_B', 'Momentum_10D', 'Dist_MA50', 'ATR_Rel', 'SPY_Corr'
            ]
            
            df_clean = df[cols].dropna()
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_clean) < SEQUENCE_LENGTH: continue
            
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

def create_model_components(feature_size, d_model, nhead, num_layers, dropout):
    input_net = nn.Sequential(
        nn.Linear(feature_size, d_model),
        nn.LayerNorm(d_model),
        nn.ReLU()
    ).to(DEVICE)
    pos_embedding = nn.Parameter(torch.randn(1, SEQUENCE_LENGTH, d_model, device=DEVICE))
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(DEVICE)
    decoder_net = nn.Sequential(
        nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, feature_size)
    ).to(DEVICE)
    return input_net, pos_embedding, transformer, decoder_net

def forward_pass(src, input_net, pos_embedding, transformer, decoder_net):
    x = input_net(src) + pos_embedding
    memory = transformer(x)
    embedding = torch.mean(memory, dim=1) 
    recon = decoder_net(memory)
    return recon, embedding

def train_model(features_dict, valid_tickers, start_date, end_date):
    slice_sequences = []
    for t in valid_tickers:
        pkg = features_dict[t]
        data = pkg['data']
        idx = pkg['index']
        mask = (idx >= start_date) & (idx < end_date)
        if mask.sum() < SEQUENCE_LENGTH + 5: continue
        subset = data[mask]
        n_samples = len(subset) - SEQUENCE_LENGTH
        if n_samples > 0:
            seqs = [subset[i : i + SEQUENCE_LENGTH] for i in range(n_samples)]
            slice_sequences.append(np.array(seqs))
            
    if not slice_sequences: return None
    flat_seqs = np.concatenate(slice_sequences, axis=0)
    tensor_data = torch.FloatTensor(flat_seqs)
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    components = create_model_components(FEATURE_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT)
    input_net, pos_embedding, transformer, decoder_net = components
    params = list(input_net.parameters()) + [pos_embedding] + list(transformer.parameters()) + list(decoder_net.parameters())
    optimizer = optim.AdamW(params, lr=1e-3)
    criterion = nn.MSELoss()
    
    input_net.train(); transformer.train(); decoder_net.train()
    
    for _ in range(EPOCHS):
        for bx, by in dataloader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = forward_pass(bx, input_net, pos_embedding, transformer, decoder_net)
            loss = criterion(recon, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
    return components

def extract_embeddings_slice(features_dict, tickers, components, end_date):
    input_net, pos_embedding, transformer, decoder_net = components
    input_net.eval(); transformer.eval(); decoder_net.eval()
    embeddings = {}
    with torch.no_grad():
        for t in tickers:
            pkg = features_dict[t]
            data = pkg['data']
            idx = pkg['index']
            locs = np.where(idx < end_date)[0]
            if len(locs) < SEQUENCE_LENGTH: continue
            last_idx = locs[-1]
            seq = data[last_idx - SEQUENCE_LENGTH + 1 : last_idx + 1]
            tensor_in = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            _, emb = forward_pass(tensor_in, input_net, pos_embedding, transformer, decoder_net)
            embeddings[t] = emb.cpu().numpy().flatten()
    return pd.DataFrame(embeddings).T

def select_portfolio_max_sharpe(embeddings_df, features_dict, start_date, end_date):
    metrics_list = []
    
    for t in embeddings_df.index:
        pkg = features_dict[t]
        raw_ret = pkg['raw_returns']
        idx = pkg['index']
        mask = (idx >= start_date) & (idx < end_date)
        period_ret = raw_ret[mask]
        if len(period_ret) < 20: continue
        ann_ret = period_ret.mean() * 252
        ann_vol = period_ret.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-6)
        metrics_list.append({'Ticker': t, 'Sharpe': sharpe, 'Vol': ann_vol})
        
    metrics_df = pd.DataFrame(metrics_list).set_index('Ticker')
    common = metrics_df.index.intersection(embeddings_df.index)
    metrics_df = metrics_df.loc[common]
    embeddings_df = embeddings_df.loc[common]
    
    if len(metrics_df) < TARGET_CLUSTERS: return pd.DataFrame()
    
    X_norm = normalize(embeddings_df.values)
    clustering = AgglomerativeClustering(n_clusters=TARGET_CLUSTERS, metric='euclidean', linkage='ward')
    clusters = clustering.fit_predict(X_norm)
    metrics_df['Cluster'] = clusters
    
    selected_assets = []
    for i in range(TARGET_CLUSTERS):
        group = metrics_df[metrics_df['Cluster'] == i]
        if group.empty: continue
        valid_group = group[group['Sharpe'] > 0]
        if valid_group.empty: continue
        best_ticker = valid_group['Sharpe'].idxmax()
        selected_assets.append(best_ticker)
        
    if not selected_assets: return pd.DataFrame()

    final_df = metrics_df.loc[selected_assets].copy()
    sharpe_vals = torch.tensor(final_df['Sharpe'].values / 0.5) 
    weights = F.softmax(sharpe_vals, dim=0).numpy()
    final_df['Weight'] = weights
    return final_df[['Weight', 'Sharpe']]

def run_rolling_backtest(raw_data):
    features_dict, valid_tickers = process_features(raw_data, TICKERS)
    
    all_dates = sorted(list(set(raw_data.index)))
    start_dt = pd.to_datetime(START_DATE)
    try:
        start_idx = next(i for i, d in enumerate(all_dates) if d >= start_dt)
    except: return
    
    current_idx = start_idx + TRAIN_WINDOW
    equity_curve = []
    dates_curve = []
    current_capital = 10000.0
    pbar = tqdm(total=len(all_dates) - current_idx)
    
    # --- State Tracker for Transaction Costs ---
    prev_weights = pd.Series(dtype=float) 
    
    print("\n====== Starting Rolling Backtest (Pure Stock) ======")
    print(f"Train Window: {TRAIN_WINDOW} days")
    print(f"Rebalance: Every {REBALANCE_FREQ} days")
    print(f"Transaction Cost Rate: {TRANSACTION_COST:.2%}")
    
    while current_idx < len(all_dates):
        train_end_date = all_dates[current_idx]
        train_start_date = all_dates[current_idx - TRAIN_WINDOW]
        test_end_idx = min(current_idx + REBALANCE_FREQ, len(all_dates))
        test_end_date = all_dates[test_end_idx - 1]
        
        components = train_model(features_dict, valid_tickers, train_start_date, train_end_date)
        if components is None:
            current_idx += REBALANCE_FREQ; pbar.update(REBALANCE_FREQ); continue
            
        embeddings = extract_embeddings_slice(features_dict, valid_tickers, components, train_end_date)
        portfolio_weights = select_portfolio_max_sharpe(embeddings, features_dict, train_start_date, train_end_date)
        
        if portfolio_weights.empty:
            current_idx += REBALANCE_FREQ; pbar.update(REBALANCE_FREQ); continue

        # --- Calculate Turnover and Apply Costs ---
        current_weights = portfolio_weights['Weight']
        all_assets = set(prev_weights.index).union(set(current_weights.index))
        turnover = sum(abs(current_weights.get(ticker, 0.0) - prev_weights.get(ticker, 0.0)) for ticker in all_assets)
        
        # Deduct transaction cost directly from capital at the start of rebalancing period
        trade_cost_capital = current_capital * (turnover * TRANSACTION_COST)
        current_capital -= trade_cost_capital
        
        # Save current weights for the next iteration
        prev_weights = current_weights.copy()
        
        # Display Logs
        top_picks = portfolio_weights.sort_values('Weight', ascending=False).head(5)
        top_str = ", ".join([f"{t}({w:.1%})" for t, w in zip(top_picks.index, top_picks['Weight'])])
        print(f"\n📅 Rebalance {str(train_end_date.date())} | Turnover: {turnover:.2f} | Cost: ${trade_cost_capital:.2f} | Top Picks: {top_str}")

        # Simulate Daily Returns for the Period
        test_data = raw_data.loc[train_end_date:test_end_date]
        period_daily_returns = pd.Series(0.0, index=test_data.index)
        
        for ticker, row in portfolio_weights.iterrows():
            if ticker not in test_data.columns.levels[0]: continue
            t_df = test_data[ticker]
            col = 'Adj Close' if 'Adj Close' in t_df else 'Close'
            rets = t_df[col].pct_change().fillna(0)
            period_daily_returns += rets * row['Weight']
            
        for date, ret in period_daily_returns.items():
            current_capital *= (1 + ret)
            equity_curve.append(current_capital)
            dates_curve.append(date)
            
        current_idx = test_end_idx
        pbar.update(test_end_idx - current_idx + REBALANCE_FREQ)
        
    pbar.close()
    
    equity_df = pd.DataFrame({'Strategy': equity_curve}, index=dates_curve)
    spy = raw_data[BENCHMARK_TICKER]
    col = 'Adj Close' if 'Adj Close' in spy else 'Close'
    spy_ret = spy[col].reindex(equity_df.index).pct_change().fillna(0)
    spy_equity = 10000.0 * (1 + spy_ret).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['Strategy'], label='AI Strategy (Net of Fees)', color='blue', linewidth=2)
    plt.plot(equity_df.index, spy_equity, label='S&P 500 (SPY)', color='gray', linestyle='--')
    plt.title('Rolling Backtest: 300+ Stocks Universe (Max Sharpe, with Tx Costs)')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Final Capital: ${equity_df['Strategy'].iloc[-1]:.2f}")

if __name__ == "__main__":
    set_deterministic(42)
    raw_data = get_data_with_cache(TICKERS, '2019-06-01', END_DATE)
    run_rolling_backtest(raw_data)

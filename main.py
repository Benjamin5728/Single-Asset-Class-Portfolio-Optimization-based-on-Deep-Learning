import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime


class Config:

    
    STOCKS = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', 'V', 'JPM', 'WMT', 'XOM', 'MA', 'UNH', 'PG', 'JNJ', 'HD', 'COST',
        'ABBV', 'MRK', 'CRM', 'AMD', 'PEP', 'CVX', 'NFLX', 'KO', 'ADBE', 'BAC', 'ACN', 'LIN', 'TMO', 'MCD', 'DIS', 'CSCO', 'ABT', 'INTC', 'WFC', 'CMCSA',
        'INTU', 'QCOM', 'VZ', 'IBM', 'AMGN', 'PFE', 'TXN', 'NOW', 'PM', 'SPGI', 'GE', 'UNP', 'CAT', 'ISRG', 'HON', 'RTX', 'LMT', 'GS', 'AMAT', 'UBER',
        'LOW', 'BKNG', 'ELV', 'BLK', 'SYK', 'TJX', 'MS', 'PGR', 'MDLZ', 'ADP', 'C', 'BMY', 'GILD', 'MMC', 'ADI', 'VRTX', 'LRCX', 'CB', 'REGN', 'SCHW',
        'MO', 'ETN', 'CI', 'BSX', 'SO', 'PANW', 'KLAC', 'TMUS', 'DE', 'MU', 'SNPS', 'SBUX', 'KKR', 'CDNS', 'EOG', 'SLB', 'COP', 'OXY', 'MPC',
        'PSX', 'VLO', 'KMI', 'WMB', 'BKR', 'HAL', 'DVN', 'TRGP', 'FANG', 'CTRA', 'EQT', 'APA', 'OVV', 'NEE', 'DUK', 'AEP', 'D',
        'SRE', 'PEG', 'WEC', 'ES', 'XEL', 'ED', 'EIX', 'DTE', 'ETR', 'PPL', 'CMS', 'AEE', 'ATO', 'LNT', 'EVRG', 'CNP', 'NI', 'PNW', 'NRG', 'BA',
        'MMM', 'ITW', 'EMR', 'PH', 'GD', 'TDG', 'NOC', 'LHX', 'HWM', 'TXT', 'CARR', 'OTIS', 'ROK', 'AME', 'FAST', 'VRSK', 'EFX',
        'URI', 'PWR', 'GWW', 'DOV', 'XYL', 'WAB', 'IR', 'HII', 'LDOS', 'AXON', 'EXPD', 'JBHT', 'CHRW', 'KNX', 'ODFL', 'SAIA', 'ARCB', 'LSTR'
    ]

    
    BONDS = [
        'TLT', 'IEF', 'SHY', 'IEI', 'AGG', 'BND', 'LQD', 'HYG', 'JNK', 'MBB', 'TIP', 'GOVT', 'SHV', 'BIL', 'VGIT', 'VGLT', 'VCIT', 'VCSH', 'BNDX', 'EMB',
        'VWOB', 'IGIB', 'IGSB', 'USIG', 'STIP', 'VTIP', 'SCHO', 'SCHR', 'SPTL', 'TLH', 'ZROZ', 'EDV', 'MUB'
    ]

    COMMODITIES = [
        'GLD', 'IAU', 'SLV', 'PPLT', 'PALL', 'USO', 'BNO', 'UNG', 'UGA', 'DBA', 'DBC', 'GSG', 'DJP', 'COW', 'MOO', 'CORN', 'SOYB', 'WEAT', 'JO',
        'SGG', 'BAL', 'CPER', 'JJN', 'LIT', 'URA', 'REMX', 'PICK', 'WOOD', 'CUT', 'GNR', 'GUNR'
    ]


    FOREX = [
        'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'EURCHF=X', 'EURCAD=X', 'CAD=X', 'CHF=X',
        'HKD=X', 'SGD=X', 'INR=X', 'MXN=X', 'PHP=X', 'IDR=X', 'THB=X', 'MYR=X', 'ZAR=X', 'RUB=X', 'SEK=X', 'NOK=X', 'DKK=X', 'PLN=X', 'HUF=X',
        'TRY=X', 'BRL=X', 'CNY=X'
    ]


    ETFS = [
        'SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'IVV', 'VTI', 'VEA', 'VWO', 'EFA', 'EEM', 'XLF', 'XLK', 'XLV', 'XLE', 'XLC', 'XLY', 'XLP', 'XLI', 'XLB', 'XLRE', 'XLU',
        'RSP', 'MTUM', 'VLUE', 'USMV', 'QUAL', 'IJR', 'IJH', 'IWB', 'IWR', 'SCHD', 'VYM', 'VIG'
    ]
    

    TICKERS = list(set(STOCKS + BONDS + COMMODITIES + FOREX + ETFS))
    
    START_DATE = '2023-01-01' 
    END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    
    SEQUENCE_LENGTH = 30  
    

    FEATURE_SIZE = 5      
    
    d_model = 64          
    nhead = 4
    num_layers = 1        
    dropout = 0.1
    
    BATCH_SIZE = 128      
    EPOCHS = 6            
    TARGET_CLUSTERS = 20  


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on device: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Running on device: CPU")


def calculate_technical_indicators(df):
    
    #RSI Calculation 
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    
    loss = loss.replace(0, np.nan).fillna(1e-6)
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['RSI'] = df['RSI'] / 100.0 

    #MACD Calculation 
    exp12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = macd - signal

    #Volatility Calculation 
    df['Log_Ret_Raw'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Volatility'] = df['Log_Ret_Raw'].rolling(window=20).std()
    
    return df

    
def get_feature_data(tickers, start, end):
    print(f"Downloading data for {len(tickers)} assets...")
    
    try:
        raw_data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=True, threads=True)
    except Exception as e:
        print(f"Download failed: {e}")
        return {}, []
    
    features_dict = {}
    valid_tickers = []
    
    print("Processing data & Engineering features...")
    
    for ticker in tqdm(tickers):
        try:
            if len(tickers) == 1:
                df = raw_data.copy()
            else:
                if ticker not in raw_data.columns.levels[0]: continue
                df = raw_data[ticker].copy()
            
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns: df['Adj Close'] = df['Close']
                else: continue
            
            if len(df) < 200: continue
            
            #Feature Engineering
            df['Log_Ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1.0)
                df['Log_Vol_Chg'] = np.log(df['Volume'] / df['Volume'].shift(1))
            else:
                df['Log_Vol_Chg'] = 0.0
            
            df['Log_Vol_Chg'] = df['Log_Vol_Chg'].replace([np.inf, -np.inf], 0.0)

            #Secondary indices calculation 
            df = calculate_technical_indicators(df)
            
            feature_cols = ['Log_Ret', 'Log_Vol_Chg', 'RSI', 'MACD_Diff', 'Volatility']
            df_clean = df[feature_cols].dropna()
            
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_clean) < Config.SEQUENCE_LENGTH: 
                continue
           
            #Normalization and Loading 
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(df_clean.values)
            
            features_dict[ticker] = scaled_data
            valid_tickers.append(ticker)
            
        except Exception as e:
            continue
            
    print(f"Successfully processed {len(valid_tickers)} assets.")
    if len(valid_tickers) == 0: raise ValueError("No assets processed.")
        
    return features_dict, valid_tickers


class MultiFactorDataset(Dataset):
    def __init__(self, features_dict, seq_len):
        self.sequences = []
        for ticker, data in features_dict.items():
            n_samples = len(data) - seq_len
            if n_samples > 0:
                for t in range(n_samples):
                    self.sequences.append(data[t : t + seq_len])
        self.sequences = torch.FloatTensor(np.array(self.sequences))

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.sequences[idx]

class AssetEmbeddingModel(nn.Module):
    def __init__(self, feature_size, d_model, nhead, num_layers, dropout=0.1):
        super(AssetEmbeddingModel, self).__init__()
        self.input_net = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, Config.SEQUENCE_LENGTH, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, feature_size) 
        )

    def forward(self, src):
        x = self.input_net(src) + self.pos_embedding
        memory = self.transformer(x)
        embedding = torch.mean(memory, dim=1) 
        recon = self.decoder_net(memory)
        return recon, embedding


def train_engine(model, dataloader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=False)
        total_loss = 0
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())

def generate_embeddings(model, features_dict, seq_len):
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for ticker, data in features_dict.items():
            last_sequence = data[-seq_len:]
            if len(last_sequence) == seq_len:
                tensor_in = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
                _, emb = model(tensor_in)
                embeddings[ticker] = emb.cpu().numpy().flatten()
    return pd.DataFrame(embeddings).T

def smart_portfolio_selection(embeddings_df, start_date, end_date):
    print("Fetching raw data for performance evaluation...")
    tickers = list(embeddings_df.index)
    
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False, threads=True)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    price_series_list = []
    
    for t in tickers:
        try:
            if len(tickers) == 1: df_t = raw_data
            else: 
                if t not in raw_data.columns.levels[0]: continue
                df_t = raw_data[t]
            
            s = None
            if 'Adj Close' in df_t.columns: s = df_t['Adj Close']
            elif 'Close' in df_t.columns: s = df_t['Close']
            
            if s is not None:
                s.name = t
                price_series_list.append(s)
        except: continue
            
    if not price_series_list: raise ValueError("Failed to retrieve price data.")
    
    raw_prices = pd.concat(price_series_list, axis=1)
    raw_prices = raw_prices.dropna(axis=1, how='all')


    returns = np.log(raw_prices / raw_prices.shift(1))
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-6)
    
    metrics = pd.DataFrame({'Sharpe': sharpe, 'Vol': ann_vol})
    
    common = metrics.index.intersection(embeddings_df.index)
    metrics, embeddings_df = metrics.loc[common], embeddings_df.loc[common]


    from sklearn.preprocessing import normalize
    X_norm = normalize(embeddings_df.values)
    clustering = AgglomerativeClustering(n_clusters=Config.TARGET_CLUSTERS, metric='euclidean', linkage='ward')
    results = metrics.copy()
    results['Cluster'] = clustering.fit_predict(X_norm)
    
    final_picks = []
    print("\n====== Multi-Asset Portfolio Selection (Sharpe Optimized) ======")
    for i in range(Config.TARGET_CLUSTERS):
        cluster_group = results[results['Cluster'] == i]
        if cluster_group.empty: continue
        best_ticker = cluster_group['Sharpe'].idxmax()
        stats = cluster_group.loc[best_ticker]
        final_picks.append({'Ticker': best_ticker, 'Cluster_ID': i, 'Sharpe': round(stats['Sharpe'], 2), 'Vol': round(stats['Vol'], 2)})

    return pd.DataFrame(final_picks), embeddings_df, results


if __name__ == "__main__":
    
    print("Phase 1: Training on Historical Data (2023-2024)...")
    train_features, _ = get_feature_data(Config.TICKERS, '2023-01-01', '2024-12-31')
    
    dataset = MultiFactorDataset(train_features, Config.SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = AssetEmbeddingModel(Config.FEATURE_SIZE, Config.d_model, Config.nhead, Config.num_layers, Config.dropout).to(device)
    train_engine(model, dataloader, Config.EPOCHS)

    
    print("\nPhase 2: Selecting Portfolio at 2024-12-31...")

    embeddings_df = generate_embeddings(model, train_features, Config.SEQUENCE_LENGTH)
    

    portfolio, _, _ = smart_portfolio_selection(embeddings_df, '2023-01-01', '2024-12-31')
    selected_tickers = portfolio['Ticker'].tolist()
    print(f"AI Selected Top Picks: {selected_tickers}")
    

    print("\nPhase 3: Validating Performance in 2025 (Out-of-Sample)...")
    
    def get_price_series(df_all, ticker):

        try:
            if ticker not in df_all.columns.levels[0]:
                return None
            
            df_t = df_all[ticker]
            
            if 'Adj Close' in df_t.columns:
                return df_t['Adj Close']
            elif 'Close' in df_t.columns:
                return df_t['Close']
            else:
                return None
        except Exception:
            return None

    try:

        validation_tickers = list(set(selected_tickers + ['SPY']))
        test_data = yf.download(validation_tickers, start='2025-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'), group_by='ticker', progress=False)
        
        strategy_ret = pd.Series(0.0, index=test_data.index)
        valid_count = 0
        
        for t in selected_tickers:
            close_price = get_price_series(test_data, t)
            
            if close_price is not None:

                ret = close_price.pct_change(fill_method=None).fillna(0)
                strategy_ret = strategy_ret + ret
                valid_count += 1
        
        if valid_count > 0:
            strategy_ret = strategy_ret / valid_count 
        else:
            raise ValueError("No valid price data found for selected tickers.")
        
        spy_close = get_price_series(test_data, 'SPY')
        if spy_close is not None:
            benchmark_ret = spy_close.pct_change(fill_method=None).fillna(0)
        else:
            print("Warning: SPY data missing, using flat line for benchmark.")
            benchmark_ret = pd.Series(0.0, index=test_data.index)

        
        cumulative_strategy = (1 + strategy_ret).cumprod()
        cumulative_benchmark = (1 + benchmark_ret).cumprod()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_strategy, label='AI Strategy (Out-of-Sample)', color='red', linewidth=2)
        plt.plot(cumulative_benchmark, label='S&P 500 Benchmark', color='gray', linestyle='--')
        plt.title('Performance Validation: 2025 YTD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Strategy Return: {(cumulative_strategy.iloc[-1]-1)*100:.2f}%")
        print(f"Benchmark Return: {(cumulative_benchmark.iloc[-1]-1)*100:.2f}%")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Validation failed details: {e}")

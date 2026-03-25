import torch
import datetime

# --- Universe & Base Config ---
STOCKS = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'CRM', 'AMD', 'ADBE', 'CSCO', 
    'ACN', 'INTC', 'ORCL', 'QCOM', 'TXN', 'IBM', 'AMAT', 'NOW', 'INTU', 'UBER', 'MU', 'PANW', 'ADI', 
    'LRCX', 'KLAC', 'SNPS', 'CDNS', 'ROP', 'APH', 'NXPI', 'MCHP', 'FTNT', 'MSI', 'TEL', 'IT', 'HPQ', 
    'GLW', 'TRMB', 'STX', 'WDC', 'NTAP', 'PSTG', 'ANET', 'SMCI', 'PLTR', 'DELL', 'HPE', 'FFIV', 'JNPR', 
    'KEYS', 'TYL', 'ZBRA', 'AKAM', 'GEN',
    # Healthcare
    'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'AMGN', 'ISRG', 'PFE', 'DHR', 'ABT', 'BMY', 'VRTX', 
    'REGN', 'SYK', 'GILD', 'ELV', 'MDT', 'ZTS', 'BSX', 'BDX', 'CI', 'CVS', 'HCA', 'MCK', 'COR', 'HUM', 
    'EW', 'CNC', 'IQV', 'A', 'RMD', 'IDXX', 'DXCM', 'BIIB', 'MTD', 'STE', 'TFX', 'COO', 'WAT', 'ALGN', 
    'HOLX', 'DGX', 'LH', 'RVTY', 'PODD', 'TECH', 'CRL', 'BIO', 'WST',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SPGI', 'AXP', 'PGR', 'CB', 'MMC', 'SCHW', 
    'KKR', 'BX', 'ICE', 'CME', 'MCO', 'AON', 'USB', 'PNC', 'TRV', 'TFC', 'AFL', 'BK', 'ALL', 'COF', 'MET', 
    'AMP', 'HIG', 'DFS', 'FITB', 'STT', 'TROW', 'RJF', 'NDAQ', 'WTW', 'BRO', 'PFG', 'CINF', 'WRB', 'L', 
    'AJG', 'RE', 'AIZ', 'GL', 'BEN', 'IVZ',
    # Consumer Discretionary
    'HD', 'COST', 'MCD', 'DIS', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'MAR', 'LULU', 'CMG', 'HLT', 'YUM', 
    'LEN', 'DHI', 'ORLY', 'ROST', 'TSCO', 'AZO', 'ULTA', 'EXPE', 'RCL', 'CCL', 'NCLH', 'GPC', 'KMX', 'DRI', 
    'DPZ', 'MGM', 'WYNN', 'LVS', 'BBY', 'HAS', 'MAT', 'POOL', 'VFC', 'TPR', 'RL', 'PVH', 'HOG',
    # Consumer Staples
    'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'EL', 'CL', 'GIS', 'MDLZ', 'TGT', 'KMB', 'DG', 'DLTR', 'KR', 
    'ADM', 'STZ', 'TSN', 'HSY', 'K', 'MKC', 'CAG', 'CHD', 'CLX', 'HRL', 'CPB', 'SJM', 'TAP',
    # Industrials
    'CAT', 'GE', 'UNP', 'HON', 'UPS', 'BA', 'RTX', 'LMT', 'DE', 'ADP', 'ETN', 'ITW', 'WM', 'GD', 'FDX', 
    'NOC', 'CSX', 'NSC', 'EMR', 'PH', 'PCAR', 'GWW', 'TT', 'CARR', 'OTIS', 'ROK', 'CMI', 'AME', 'VRSK', 
    'FAST', 'EFX', 'URI', 'PWR', 'DOV', 'XYL', 'WAB', 'IR', 'HII', 'LDOS', 'AXON', 'EXPD', 'JBHT', 'CHRW', 
    'KNX', 'ODFL', 'SAIA', 'ARCB', 'LSTR', 'DAL', 'UAL', 'AAL', 'LUV',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI', 'WMB', 'BKR', 'HAL', 'DVN', 
    'TRGP', 'FANG', 'CTRA', 'EQT', 'APA', 'OVV', 'MRO', 'HES',
    # Materials
    'LIN', 'SHW', 'FCX', 'APD', 'ECL', 'NEM', 'DOW', 'DD', 'CTVA', 'PPG', 'MLM', 'VMC', 'ALB', 'FMC', 
    'LYB', 'CE', 'EMN', 'CF', 'MOS',
    # Utilities
    'NEE', 'SO', 'DUK', 'AEP', 'SRE', 'PEG', 'WEC', 'ES', 'XEL', 'ED', 'EIX', 'DTE', 'ETR', 'PPL', 'CMS', 
    'AEE', 'ATO', 'LNT', 'EVRG', 'CNP', 'NI', 'PNW', 'NRG',
    # Real Estate
    'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'VICI', 'AVB', 'EQR', 'CBRE', 'CSGP', 
    'SUI', 'INVH', 'MAA', 'ESS', 'UDR', 'KIM'
]

TICKERS = list(set(STOCKS))

BENCHMARK_TICKER = 'SPY'
if BENCHMARK_TICKER not in TICKERS:
    TICKERS.append(BENCHMARK_TICKER)

BLACKLIST = ['BIL', 'SHV', 'MUB', 'AGG', 'BND', 'SPY', 'QQQ', 'DIA', 'IWM'] 

# --- Backtest Configuration ---
START_DATE = '2020-01-01'  
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
DATA_START_DATE = '2019-06-01' 

TRAIN_WINDOW = 1008      
REBALANCE_FREQ = 63      
TRANSACTION_COST = 0.001 

# --- Deep Learning Hyperparameters ---
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 9        
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2          
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 10             
TARGET_CLUSTERS = 30   

# --- Hardware Device ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

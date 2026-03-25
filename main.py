import config
from data import get_data_with_cache
from backtest import run_rolling_backtest
import random
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

def set_deterministic(seed=42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_deterministic(42)
    print(f"Running on device: {config.DEVICE}")
    print(f"Total Assets in Universe: {len(config.TICKERS)}")
    
    raw_data = get_data_with_cache(config.TICKERS, config.DATA_START_DATE, config.END_DATE)
    run_rolling_backtest(raw_data)

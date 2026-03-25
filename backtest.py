import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
from features import process_features
from model import train_model, extract_embeddings_slice
from portfolio import select_portfolio_max_sharpe

def run_rolling_backtest(raw_data):
    """Executes the rolling window backtest, handling training, trading, and logging."""
    features_dict, valid_tickers = process_features(raw_data, config.TICKERS)
    all_dates = sorted(list(set(raw_data.index)))
    start_dt = pd.to_datetime(config.START_DATE)
    
    try:
        start_idx = next(i for i, d in enumerate(all_dates) if d >= start_dt)
    except StopIteration:
        print("Error: Start date not found in data.")
        return
    
    current_idx = start_idx + config.TRAIN_WINDOW
    equity_curve, dates_curve = [], []
    current_capital = 10000.0
    prev_weights = pd.Series(dtype=float) 
    
    print("\n====== Starting Rolling Backtest (Net of Fees) ======")
    pbar = tqdm(total=len(all_dates) - current_idx)
    
    while current_idx < len(all_dates):
        train_end_date = all_dates[current_idx]
        train_start_date = all_dates[current_idx - config.TRAIN_WINDOW]
        test_end_idx = min(current_idx + config.REBALANCE_FREQ, len(all_dates))
        test_end_date = all_dates[test_end_idx - 1]
        
        # 1. Train & Select Portfolio
        components = train_model(features_dict, valid_tickers, train_start_date, train_end_date)
        if components is None:
            current_idx += config.REBALANCE_FREQ; pbar.update(config.REBALANCE_FREQ); continue
            
        embeddings = extract_embeddings_slice(features_dict, valid_tickers, components, train_end_date)
        portfolio_weights = select_portfolio_max_sharpe(embeddings, features_dict, train_start_date, train_end_date)
        
        if portfolio_weights.empty:
            current_idx += config.REBALANCE_FREQ; pbar.update(config.REBALANCE_FREQ); continue

        # 2. Calculate Transaction Costs
        current_weights = portfolio_weights['Weight']
        all_assets = set(prev_weights.index).union(set(current_weights.index))
        turnover = sum(abs(current_weights.get(ticker, 0.0) - prev_weights.get(ticker, 0.0)) for ticker in all_assets)
        trade_cost_capital = current_capital * (turnover * config.TRANSACTION_COST)
        current_capital -= trade_cost_capital
        prev_weights = current_weights.copy()
        
        # Display Period Logs
        top_picks = portfolio_weights.sort_values('Weight', ascending=False).head(5)
        top_str = ", ".join([f"{t}({w:.1%})" for t, w in zip(top_picks.index, top_picks['Weight'])])
        print(f"\n📅 Rebalance {train_end_date.date()} | Cost: ${trade_cost_capital:.2f} | Top: {top_str}")

        # 3. Calculate Period Returns
        test_data = raw_data.loc[train_end_date:test_end_date]
        period_daily_returns = pd.Series(0.0, index=test_data.index)
        
        for ticker, row in portfolio_weights.iterrows():
            if ticker not in test_data.columns.levels[0]: continue
            col = 'Adj Close' if 'Adj Close' in test_data[ticker] else 'Close'
            rets = test_data[ticker][col].pct_change().fillna(0)
            period_daily_returns += rets * row['Weight']
            
        for date, ret in period_daily_returns.items():
            current_capital *= (1 + ret)
            equity_curve.append(current_capital)
            dates_curve.append(date)
            
        current_idx = test_end_idx
        pbar.update(test_end_idx - current_idx + config.REBALANCE_FREQ)
        
    pbar.close()
    
    # 4. Benchmarking & Visualization
    equity_df = pd.DataFrame({'Strategy': equity_curve}, index=dates_curve)
    spy = raw_data[config.BENCHMARK_TICKER]
    col = 'Adj Close' if 'Adj Close' in spy else 'Close'
    spy_ret = spy[col].reindex(equity_df.index).pct_change().fillna(0)
    spy_equity = 10000.0 * (1 + spy_ret).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['Strategy'], label='AI Strategy (Net of Fees)', color='blue', linewidth=2)
    plt.plot(equity_df.index, spy_equity, label='S&P 500 (SPY)', color='gray', linestyle='--')
    plt.title('Rolling Backtest: Max Sharpe with Transaction Costs')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Final Capital: ${equity_df['Strategy'].iloc[-1]:.2f}")

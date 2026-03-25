# Single-Asset-Class Portfolio Optimization based on Deep Learning

## 📈 Project Overview
This project implements a sophisticated **quantitative equity strategy** focused on the S&P 500 universe. It leverages **Deep Learning (Transformer Autoencoders)** and **Unsupervised Clustering** to identify structural market regimes and capture Alpha through dynamic sector rotation. 

Unlike traditional multi-asset portfolios that often dilute returns by over-allocating to "safe havens" (Bonds/Cash), this strategy is designed for **pure equity exposure**, utilizing a **Rolling Walk-Forward** framework to aggressively target high risk-adjusted return opportunities in the stock market, strictly accounting for real-world transaction costs.

## 💡 Inspiration & Design Philosophy: The Quest for the "Holy Grail"

The genesis of this project lies in **Ray Dalio’s "Holy Grail" of Investing**: the idea that combining 10–15 uncorrelated return streams can dramatically reduce risk without sacrificing return. 

My initial goal was to build a true **Multi-Asset** deep learning model that dynamically allocates across Equities, Bonds, Commodities, and Forex, learning non-linear correlations that traditional linear models miss.

### 🛑 The "Multi-Asset" Dilemma & The Pivot
However, during the development of the multi-asset prototype, I encountered a persistent **Deep Learning "Safe Haven" Bias**:
* **The Observation:** When presented with a mixed universe (High-Vol Stocks vs. Low-Vol Treasuries) and optimized for Sharpe Ratio, the Transformer model consistently learned to "game" the objective function.
* **The Problem:** The model discovered that the easiest way to maximize risk-adjusted returns was not to predict stock alpha, but to **allocate 90%+ capital to cash equivalents (e.g., BIL/SHV)**. This resulted in a theoretically high-Sharpe but practically useless "dead fish" equity curve that missed all growth opportunities.
* **The Pivot:** Acknowledging this limitation in current single-model architectures, I pivoted to a **Pure Equity Strategy**. By restricting the universe to **300+ liquid S&P 500 stocks**, I force the AI to hunt for diversification and Alpha *within* the equity risk premium, effectively searching for uncorrelated drivers (Cluster Regimes) inside the stock market itself.

### 2. Deep Learning as a "Feature Extractor"
Inspired by **NLP (Natural Language Processing)**, this project treats daily price action not as random walks, but as "sequences" with latent grammar.
* Instead of feeding raw prices to a predictor, we use a **2-Layer Transformer Autoencoder** to compress 30-day noisy market data into dense **64-dimensional Latent Embeddings**.
* This allows the model to "see" market regimes (e.g., "Tech Momentum", "Defensive Rotation") that are invisible to linear correlation matrices.

### 3. "Winner-Takes-Most" Allocation
Moving away from conservative Risk Parity, the allocation logic is inspired by the **Power Law** distribution of stock returns.
* We use a **Softmax-weighted** approach (Temperature $T=0.5$).
* This mimics the behavior of top-performing active managers: aggressively tilting weights towards the top decile of high-conviction assets while maintaining a diversified tail, rather than equally weighting mediocrity.

## 🚀 Key Features

* **Universe:** 300+ S&P 500 Constituents (Filtered for liquidity and data integrity).
* **9-Factor Feature Engineering:** A robust multi-dimensional view of every asset, including:
    * *Momentum:* 10-Day ROC.
    * *Trend:* MACD, Distance-to-MA50.
    * *Volatility:* ATR (Average True Range), Rolling Volatility.
    * *Mean Reversion:* Bollinger Bands %B.
    * *Market Beta:* Rolling correlation with SPY.
* **Realistic Market Friction:** Hard-coded **0.1% transaction cost per trade** to simulate realistic slippage and broker commissions, penalizing excessive turnover.
* **Rolling Walk-Forward Backtest:**
    * **Training:** 4-Year Moving Window (1008 trading days) to capture long-term structural dependencies.
    * **Validation:** Out-of-Sample testing with a 63-day rebalancing frequency, simulating a real-world quarterly rebalancing fund.
* **Regime-Based Clustering:** Uses **Agglomerative Clustering** on latent embeddings to ensure the portfolio selects stocks that are *behaviorally distinct*, avoiding the trap of buying 30 correlated tech stocks.

## 📁 Project Structure

* **quant_trading_project/**

* **config.py**           # Global configurations, universe selection, and hyperparameters
* **data.py**             # Data downloading and local caching logic
* **features.py**         # Technical indicator computation and feature scaling
* **model.py**            # PyTorch Transformer architecture and training loop
* **portfolio.py**        # Asset clustering and Softmax weight allocation
* **backtest.py**         # Rolling backtest engine, cost deduction, and visualization
* **main.py**             # Main execution entry point
* **README.md**           # Project documentation

## ⚙️ Technical Architecture

### 1. Data Pipeline
* **Source:** Yahoo Finance (`yfinance`).
* **Preprocessing:** Automatic handling of delisted tickers and `RobustScaler` normalization to handle fat-tail distribution in stock returns.

### 2. The Model (Transformer AE)
* **Encoder:** 2-Layer Transformer with Multi-Head Attention ($d_{model}=64$, $n_{head}=4$).
* **Task:** Reconstruction of the 9-factor technical state sequence.
* **Output:** A static vector embedding representing the asset's current "market state."

### 3. Portfolio Construction
The "Brain" of the strategy follows a strict logic:
1.  **Filter:** Exclude assets with negative Sharpe Ratios over the lookback period.
2.  **Cluster:** Group remaining stocks into **30 Clusters** based on latent similarity.
3.  **Select:** Pick the #1 Stock from each cluster (Best-in-Class).
4.  **Weight:** Apply **Softmax Optimization** to allocate capital based on risk-adjusted momentum.

## 📊 Backtest Results (Visual Proof)

The following chart illustrates the strategy's cumulative performance during the **Out-of-Sample validation period (Jan 2024 – Mar 2026)**.

![Rolling Backtest Results](results.png)
> *Figure 1: The Blue Line represents the AI-Driven Pure Stock Strategy (Net of Fees), while the Grey Dashed Line represents the S&P 500 Benchmark (SPY).*

### Key Observations:
1.  **Resilient Alpha Generation (Net of Fees):** Even after strictly accounting for a 0.1% transaction cost per trade, the strategy achieved a final capital of **$15,669.57** (+56.7% Total Return), outperforming the SPY benchmark (~$15,000 / +50.0%).
2.  **Consistent Annualized Excess Return:** The strategy delivered a Compound Annual Growth Rate (CAGR) of **~22.1%**, generating an estimated **2.5% Annualized Alpha** over the S&P 500 during the validation window.
3.  **No "Cash Drag":** Unlike previous iterations that flat-lined by holding T-Bills, this curve shows active and profitable participation in major market rallies.
4.  **Rapid Drawdown Recovery:** During market pullbacks (e.g., April 2024 and mid-2025), the strategy demonstrated a capability to recover faster than the broad index, driven by its systematic rotation into high-momentum sectors.

| Metric | AI Strategy (Net of Fees) | S&P 500 (Benchmark) |
| :--- | :--- | :--- |
| **Total Return** | **+56.7%** | ~50.0% |
| **Est. CAGR** | **~22.1%** | ~19.6% |
| **Annualized Alpha** | **~2.5%** | Baseline |
| **Exposure** | **100% Equity** | 100% Equity |
| **Rebalancing** | Quarterly (63 Days) | N/A |
| **Transaction Costs** | 0.1% per trade | 0.0% (Index) |

## 🛠️ Future Roadmap (Towards True Multi-Asset)
To re-introduce Multi-Asset capabilities without the "Cash Trap," future versions will explore:
* **Hierarchical Risk Parity (HRP):** To allocate risk budgets across asset classes rather than maximizing Sharpe, preventing capital from flooding into low-volatility assets.
* **Regime-Switching Meta-Models:** Using separate sub-models for Equities, Bonds, and Commodities, aggregated by a top-level learner to enforce diversification constraints.

## ⚠️ Disclaimer
This project is a research prototype for **AI-driven quantitative finance**. It is not financial advice. Past performance in backtests does not guarantee future live results.

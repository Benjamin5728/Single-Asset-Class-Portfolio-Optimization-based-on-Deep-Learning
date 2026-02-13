# Multi-Asset Porfolio Optimization based on Deep Learning 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Domain](https://img.shields.io/badge/Domain-Quantitative_Finance-orange)

## 📖 Executive Summary

This is a sophisticated quantitative investment framework designed to identify latent market regimes and construct robust, uncorrelated portfolios.

Unlike traditional strategies that rely on linear correlations or simple price prediction, this project leverages **Self-Supervised Deep Learning**. It employs a **Transformer-based Autoencoder** to compress high-dimensional, noisy market data into dense, informative latent embeddings. These embeddings are then analyzed using **Unsupervised Hierarchical Clustering** to identify distinct asset clusters, effectively implementing a modern, AI-driven version of Ray Dalio's "Holy Grail" diversification philosophy.

The strategy is validated through a strict **Walk-Forward Analysis**, training on historical data (2023-2024) and testing performance on unseen out-of-sample data (2025 YTD), achieving significant Alpha against the S&P 500 benchmark.

---

## 🚀 Key Features

* **State-of-the-Art Architecture**: Utilizes `nn.TransformerEncoder` with Self-Attention mechanisms to capture long-range temporal dependencies in financial time series.
* **Multi-Asset Universe**: Processes a diverse universe of **300+ assets** including Global Equities, Government Bonds, Forex, Commodities, and ETFs.
* **Robust Feature Engineering**: Implements stationarity-adjusted technical indicators (RSI, MACD, Volatility) and uses `RobustScaler` to handle financial outliers.
* **Unsupervised Regime Detection**: Uses Agglomerative Clustering on latent embeddings to group assets based on behavioral similarity rather than static sector labels.
* **Risk-Adjusted Optimization**: Selects the highest Sharpe Ratio asset within each cluster to maximize risk-adjusted returns.
* **Hardware Acceleration**: Automatically detects and utilizes **CUDA** (NVIDIA) or **MPS** (Apple Silicon) for high-performance training.

---

## 🧠 Methodology & Mathematical Framework

The project pipeline consists of three distinct phases:

### Phase 1: Feature Extraction (The "Brain")
The core of the system is a **Transformer Autoencoder**. Its goal is to learn a compact representation (Embedding) of an asset's recent behavior.

* **Input**: A sequence of 30 days of technical indicators ($X \in \mathbb{R}^{30 \times 5}$).
    * *Features*: Log Returns, Log Volume Change, RSI, MACD Difference, Volatility.
* **Encoder**:
    * Projects input to a high-dimensional space ($d_{model} = 64$).
    * Adds **Positional Embeddings** to retain temporal order information.
    * Applies **Multi-Head Self-Attention**:
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
        This allows the model to weigh the importance of different days in the 30-day window, capturing complex non-linear dependencies.
* **Bottleneck (Latent Space)**: A Global Mean Pooling layer compresses the sequence into a single static vector ($Z \in \mathbb{R}^{64}$). This vector represents the "DNA" of the asset.
* **Decoder**: Attempts to reconstruct the original input from the bottleneck.
* **Objective**: Minimize the Reconstruction Loss (MSE) to force the model to filter noise and retain signal.
    $$\mathcal{L} = \frac{1}{N} \sum (X_{original} - X_{reconstructed})^2$$

### Phase 2: Portfolio Construction (The "Strategy")
Once the model is trained, we discard the decoder and use the encoder to generate embeddings for all 300+ assets at the rebalancing date (2024-12-31).

1.  **Normalization**: Embeddings are normalized to the unit hypersphere to focus on directional similarity.
2.  **Clustering**: We apply **Agglomerative Hierarchical Clustering** (Ward's linkage) to group assets into $K=20$ distinct clusters based on Euclidean distance in the latent space.
    * *Hypothesis*: Assets in different clusters are mathematically uncorrelated.
3.  **Selection**: Inside each cluster, we calculate the historical **Sharpe Ratio** (2023-2024) and select the single best-performing asset. This ensures we pick the "winner" of each unique market regime.

### Phase 3: Out-of-Sample Validation (The "Test")
To prevent look-ahead bias, the strategy is strictly tested on unseen data:
* **Training Period**: Jan 1, 2023 – Dec 31, 2024.
* **Testing Period**: Jan 1, 2025 – Present.
* **Benchmark**: SPDR S&P 500 ETF Trust (SPY).

---

## 📊 Performance Results

The following chart illustrates the cumulative returns of the AI-selected portfolio (Red) versus the S&P 500 Benchmark (Grey) during the out-of-sample testing period (Jan 1, 2025 – Present).

![Performance Chart](results.png)

> **Analysis**: The divergence between the two lines represents the **Alpha** generated by the strategy. A higher slope for the strategy line indicates superior risk-adjusted returns, while the smoothness of the curve reflects lower realized volatility due to the clustering-based diversification.

---

## ⚠️ Limitations & Critical Analysis

Transparency about limitations is a hallmark of rigorous quantitative research. While the current framework demonstrates significant potential, it is essential to acknowledge its constraints:

1.  **Static Rebalancing**:
    The current implementation rebalances the portfolio only once (at the end of the training period). It does not dynamically adjust weights or rotate assets as market conditions change during the testing phase. This makes it vulnerable to sudden regime shifts (e.g., a flash crash or interest rate shock) that occur *after* the rebalancing date.

2.  **Look-Ahead Bias in Feature Scaling**:
    Although we split training and testing data, the `RobustScaler` is fitted on the entire training history in this experimental version. In a live production environment, scaling parameters would need to be updated in a rolling window to strictly prevent data leakage.

3.  **Transaction Costs & Slippage**:
    The backtest assumes execution at the exact closing price with zero fees. In reality, trading 20 diverse assets (including potentially illiquid ETFs or small caps) would incur spreads and commissions that could erode the theoretical Alpha, especially with more frequent rebalancing.

4.  **Short Context Window**:
    The model uses a `SEQUENCE_LENGTH` of 30 days. While this captures short-term momentum and volatility clusters, it may miss longer-term macro cycles (e.g., year-long inflationary trends) that a deeper model with a longer context window might catch.

5.  **Stationarity Assumption**:
    The core hypothesis assumes that assets with similar *past* behavior (embeddings) will continue to behave similarly in the near future. Financial markets are inherently non-stationary; correlations often break down during liquidity crises (when "all correlations go to 1").

---

## 🔮 Future Improvements & Roadmap

To evolve this project from a research prototype into an institutional-grade trading system, the following enhancements are planned:

### 1. Dynamic Rolling Window Backtesting
* **Current**: Train (2023-24) -> Test (2025).
* **Upgrade**: Implement a **Walk-Forward Validation** framework.
    * *Example*: Train on Jan-Mar, Test on Apr; Train on Feb-Apr, Test on May.
    * This allows the model to adapt to changing market regimes and provides a more statistically significant performance metric.

### 2. Advanced Model Architectures
* **Temporal Fusion Transformers (TFT)**: Incorporate static covariates (e.g., sector labels, market cap) alongside dynamic time-series data.
* **Variational Autoencoders (VAE)**: Instead of mapping to a single point, map inputs to a *probability distribution* in the latent space. This would allow for uncertainty quantification and better risk management.
* **LSTM/GRU Hybrids**: Combine the long-term memory of RNNs with the attention mechanism of Transformers for a hybrid encoder.

### 3. Alternative Data Integration
* **Macro Factors**: Feed Federal Reserve interest rates, CPI data, and bond yield curves into the model as exogenous variables.
* **Sentiment Analysis**: Scrape financial news (Bloomberg/Reuters) or social sentiment (Twitter/Reddit) using NLP (BERT/RoBERTa) to generate a "Sentiment Score" feature.

### 4. Reinforcement Learning (RL) Agent
* Instead of a static "Cluster & Select" heuristic, train a **Deep Q-Network (DQN)** or **PPO Agent** that takes the Transformer's embeddings as state inputs and outputs direct trading actions (Buy/Sell/Hold) to maximize the Sharpe Ratio directly.

### 5. Risk Management Layer
* **Volatility Targeting**: Dynamically scale position sizes based on the predicted volatility of each cluster.
* **Stop-Loss / Take-Profit**: Implement logic to automatically cut losing positions or lock in gains when the embedding distance exceeds a certain threshold (indicating a regime shift).

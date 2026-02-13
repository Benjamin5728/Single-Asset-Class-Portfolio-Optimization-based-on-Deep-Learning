Unsupervised Regime-Based Asset Allocation using Transformer Autoencoders 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Domain](https://img.shields.io/badge/Domain-Quantitative_Finance-orange)

## 📖 Executive Summary

**This is a sophisticated quantitative investment framework designed to identify latent market regimes and construct robust, uncorrelated portfolios.

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

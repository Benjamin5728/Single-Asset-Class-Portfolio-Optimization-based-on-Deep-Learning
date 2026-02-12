# Multi-Asset-Portfolio-Optimization-based-on-Deep-Learning
A PyTorch-based quantitative investment framework. Utilizes Transformer Autoencoders for unsupervised market regime detection and hierarchical risk clustering to achieve robust multi-asset allocation.
# Deep-Quant: Unsupervised Regime-Based Asset Allocation 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Overview

**Deep-Quant** is a quantitative research project that applies state-of-the-art Deep Learning techniques to multi-asset portfolio management. 

Instead of trying to predict future price movements (which is prone to noise), this project focuses on **Representation Learning**. It uses a **Transformer-based Autoencoder** to extract robust, non-linear latent features from noisy market data, and applies **Unsupervised Clustering** to identify distinct market regimes.

The goal is to construct a "Holy Grail" portfolio—inspired by **Ray Dalio's All-Weather philosophy**—by systematically selecting uncorrelated assets that perform best within their specific clusters.

---

## 🚀 Key Features

* **Transformer Autoencoder Architecture**: Implements a self-attention mechanism to capture long-range temporal dependencies and filter out high-frequency market noise (Denoising).
* **Multi-Asset Universe**: Processes 300+ instruments including Global Equities, Govt Bonds, Forex, and Commodities to ensure true diversification.
* **Unsupervised Regime Detection**: Uses Agglomerative Clustering on latent embeddings to group assets by behavior rather than just sector labels.
* **Rigorous Walk-Forward Validation**: Strictly adheres to a temporal split (Train: 2023-2024, Test: 2025) to prevent look-ahead bias.

---

## 🛠️ Methodology Pipeline

The strategy follows a strictly isolated pipeline to ensure data integrity:

1.  **Data Ingestion**: Fetching 300+ tickers via Yahoo Finance.
2.  **Feature Engineering**: Calculating stationarity-adjusted metrics (RSI, MACD, Volatility, Log Returns).
3.  **Latent Encoding (The "Brain")**: 
    * Input: `(Batch, Sequence_Length=30, Features=5)`
    * Model: Transformer Encoder -> Bottleneck (64-dim) -> Decoder
    * Objective: Minimize Reconstruction Loss (MSE) to learn robust market structures.
4.  **Clustering & Selection**:
    * Group assets based on the similarity of their Latent Embeddings.
    * Select the asset with the highest Sharpe Ratio within each cluster.
5.  **Backtesting**: Out-of-Sample performance evaluation using 2025 data.

---

## 📊 Performance (Out-of-Sample)

*Validation Period: Jan 1, 2025 - Present*

> **Note:** The strategy demonstrated a significant **Alpha** against the S&P 500 (SPY) benchmark during the validation period, particularly outperforming during the Q1 2025 drawdown via dynamic allocation to defensive assets (BIL, SHV).

![Performance Chart](results.png)

| Metric | Strategy | Benchmark (SPY) |
| :--- | :--- | :--- |
| **Total Return** | **+28.12%** | +21.58% |
| **Alpha** | **+6.54%** | - |
| **Regime** | Multi-Asset | Equity Only |

*Disclaimer: Results do not account for transaction costs or slippage. Past performance is not indicative of future results.*

---

## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YourUsername/Deep-Quant-Allocation.git](https://github.com/YourUsername/Deep-Quant-Allocation.git)
   cd Deep-Quant-Allocation

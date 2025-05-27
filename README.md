# 📈 Predicting Stock Price Trend Reversal Points Using Deep Learning - A Case Study of the S&P 500  
by Cheng-Ping Lin | National Cheng Kung University | Advisor: Prof. Wei-Sheng Wu

## 🧠 Project Overview

This project aims to utilize deep learning models (GRU, CNN, LSTM, Transformer, etc.) to predict daily future trends in the S&P 500 index and identify trend reversal points. The research addresses class imbalance in reversal labels by predicting future trends and inferring reversal points based on changes in trend direction. The goal is to build a model that supports real trading decisions.

---

## 🔍 Research Motivation

- **Challenge:** Reversal points are rare events, creating a severe class imbalance.
- **Goal:** Predict daily future trends and derive reversal points from trend changes.
- **Application:** Construct trading strategies based on model signals and evaluate via historical backtesting.

---

## 🗂️ Dataset

- **Target:** S&P 500 Index (^GSPC)
- **Period:** January 2001 – December 2023
- **Features (32 total):**
  - Price data (Open, High, Low, Close, Volume)
  - Technical indicators (MACD, RSI, CCI, ADX, etc.)
  - US Treasury Yields (13W, 5Y, 10Y)
  - S&P 500 Volatility Index (VIX)

---

## ⚙️ Environment

```bash
python 3.10+
pandas
numpy
scikit-learn
matplotlib
yfinance
ta
torch
tqdm

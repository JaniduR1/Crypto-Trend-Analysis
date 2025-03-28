# Crypto Trend Analysis App

This project analyses the historical performance and volatility of Bitcoin (BTC-USD) using real-world market data. It combines time-series analysis, financial concepts, and Python visualisation to uncover key trends and behaviours in the crypto market. 

The goal is to not just visualise Bitcoin’s price movements, but also generate meaningful insights such as return patterns and risk indicators which can feed into simple machine learning models for prediction, which simulates how fintech platforms assess market behaviour to drive features like risk alerts, portfolio insights, or trading signals.

---

### Tools Used

- Python
- `pandas` – for data manipulation
- `matplotlib` – for plotting visualisations
- `seaborn` – optional, for styled plots
- `yfinance` – for fetching crypto price data

---

### Dataset Overview

**Source**: Yahoo Finance  
**Ticker**: `BTC-USD`  
**Date Range**: 2017-01-01 to 2025-03-01  
**Frequency**: Daily

#### Original Columns (from Yahoo Finance):

| Column | Description |
|--------|-------------|
| Open   | Price at the start of the day |
| High   | Highest price of the day |
| Low    | Lowest price of the day |
| Close  | Price at the end of the day |
| Volume | Total trading volume in USD |

#### New Columns We Created:

| Column                   | Description |
|--------------------------|-------------|
| Daily Return             | % change in closing price from the previous day |
| Rolling Volatility (30D) | 30-day rolling standard deviation of daily returns |

---

### Volatility Analysis

Calculated the 30-day rolling volatility of daily returns to measure how Bitcoin's risk levels change over time.  
This helps identify periods of extreme market movement (e.g. crashes or rallies).

- Volatility is defined as the **standard deviation of daily returns**
- A 30-day window gives a **monthly view of how volatile the market has been**
- Helps in understanding market cycles and risk levels visually


`.rolling(window=30)` This sets up a rolling window of 30 rows.
```
A rolling window means: “Take a fixed-size slice of data, calculate something, slide the window down by one row, repeat.”

So:
- On day 30 → looks at days 1–30

- On day 31 → looks at days 2–31

etc
```

---

### Top 5 Most Volatile Days (30-Day Rolling Volatility)

Sorted the dataset by 30-day rolling volatility to identify the periods with the greatest price instability.  
These were times where the market experienced **rapid fluctuations** — often driven by global news events or major crypto developments.

| Date       | 30D Rolling Volatility | BTC Closing Price (USD) |
|------------|------------------------|--------------------------|
| 2020-04-06 | 0.0913                 | $7,271.78                |
| 2020-04-03 | 0.0906                 | $6,733.39                |
| 2020-04-02 | 0.0906                 | $6,793.62                |
| 2020-04-10 | 0.0905                 | $6,865.49                |
| 2020-03-31 | 0.0905                 | $6,438.64                |

#### Why
- To identify **high-risk** time periods for Bitcoin
- These insights simulate how a fintech product might detect periods of increased market volatility and trigger alerts or adjust portfolio risk levels
- Also useful for training future ML models to recognise early signals of major swings

#### How
- Used `.rolling(window=30).std()` on daily returns to calculate 30-day rolling volatility
- Sorted the DataFrame by this column in descending order
- Displayed the top 5 dates with the highest volatility

---

### 📊 Daily Return Distribution

Plotted a histogram of daily returns to understand how often Bitcoin experiences different levels of price change.

#### Why
- Shows how returns are spread — are most days small gains/losses? How often do big moves happen?
- Helps you understand the **risk profile** of Bitcoin
- Lays the groundwork for risk metrics like **Value at Risk (VaR)** or **Sharpe Ratio**
- Provides clear **visual storytelling** around volatility

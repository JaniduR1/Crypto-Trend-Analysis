# Crypto Trend Analysis App

This project analyses the historical performance and volatility of Bitcoin (BTC-USD) using real-world market data. It combines time-series analysis, financial concepts, and Python visualisation to uncover key trends and behaviours in the crypto market. 

The goal is to not just visualise Bitcoin’s price movements, but also generate meaningful insights such as return patterns and risk indicators which can feed into simple machine learning models for prediction, which simulates how fintech platforms assess market behaviour to drive features like risk alerts, portfolio insights, or trading signals.

#### Local Development

To run locally with all packages:

```bash
pip install -r dev-requirements.txt
```

---

## Tools Used

- Python
- `pandas` – for data manipulation
- `matplotlib` – for plotting visualisations
- `seaborn` – optional, for styled plots
- `yfinance` – for fetching crypto price data

---

## Dataset Overview

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

## Volatility Analysis

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

## Top 5 Most Volatile Days (30-Day Rolling Volatility)

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

## Daily Return Distribution

Plotted a histogram of daily returns to understand how often Bitcoin experiences different levels of price change.

#### Why
- Shows how returns are spread — are most days small gains/losses? How often do big moves happen?
- Helps you understand the **risk profile** of Bitcoin
- Lays the groundwork for risk metrics like **Value at Risk (VaR)** or **Sharpe Ratio**
- Provides clear **visual storytelling** around volatility

---

## Addressing Class Imbalance with SMOTE

Initially, the logistic regression model always predicted class `1` (Bitcoin going up), because class `0` (Bitcoin not going up) had far fewer examples. 

#### Why:
| Class | Description | Count | Result |
|-------|-------------|-------|--------|
| `1` | BTC went up | High  | Model learns this bias |
| `0` | BTC did **not** go up | Low | Model ignores this class |

So the model plays it safe and always says "BTC will go up" — it appears accurate, but it's actually **blind to half the problem**.


To improve the initial model's performance, the dataset was  balanced using **SMOTE (Synthetic Minority Over-sampling Technique)**.

Example:
```if we had 400 yes's to something and 40 no's it would predict yes for all of them for safety based on the stastics. but using smote it would create an additonal 200 no's or 360 no's to balance out the yes's?```


__Used SMOTE to:__
- Analyse the minority class (`0`)
- Find “similar” examples
- Generate new **synthetic examples** that resemble real ones by blending real examples of class `0`'s, not random's or copies of ones
- Balance the dataset before retraining the model

#### Code:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=69)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

print("Class distribution after SMOTE:")
print(y_train_balanced.value_counts())
```
### Logistic Regression After Class Balancing (SMOTE)
#### Results (After SMOTE):

```text
Accuracy (Balanced): 0.5067

| Class               | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Did Not Increase (0)| 0.50      | 0.58   | 0.54     | 294     |
| Increased (1)       | 0.52      | 0.44   | 0.47     | 302     |

Overall Metrics:

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Accuracy      | –         | –      | 0.51     | 596     |
| Macro Avg     | 0.51      | 0.51   | 0.50     | 596     |
| Weighted Avg  | 0.51      | 0.51   | 0.50     | 596     |
```

- The model now **recognises both classes**, instead of just always predicting `1`
- Class `0` precision and recall improved significantly
- Overall accuracy stayed similar, but model behaviour is more realistic


#### Why
By fixing the class imbalance before training, the model is forced to pay attention to both types of outcomes (up and down). This improves fairness and makes future models (like Random Forest) more reliable.

#### Images:
- `images/confusion_matrix_balanced.png` — Improved prediction spread across both classes
- `images/classification_report_balanced.txt` — Saved performance breakdown

---

## Random Forest Model (After SMOTE Balancing)

After balancing the training data using SMOTE, a **Random Forest model** was trained and tested to predict whether Bitcoin will go up or not.

This model uses an number of decision trees to capture more complex, non-linear relationships in the data, which simpler models like logistic regression may miss.

Example:
```text
                     [Rolling Mean (5D) > 0.015?]
                            /           \
                          Yes           No
                 [Volatility < 0.025?]   [Return > 0?]
                     /       \              /    \
                 [Lag1 > 0]  [Lag1 < 0]   0      1
                    /   \       /  \
                   1     0     0    1
```
Each tree makes predictions by asking a series of questions. The **depth** of a tree refers to how many layers of questions it can ask before making a final prediction.
The deeper the tree, the more complex patterns it can learn, but this also increases the risk of overfitting.

#### Results (Random Forest, Balanced Dataset)

| Class               | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Did Not Increase (0)| 0.49      | 0.48   | 0.49     | 294     |
| Increased (1)       | 0.50      | 0.52   | 0.51     | 302     |

**Overall Metrics:**

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Accuracy      | –         | –      | 0.50     | 596     |
| Macro Avg     | 0.50      | 0.50   | 0.50     | 596     |
| Weighted Avg  | 0.50      | 0.50   | 0.50     | 596     |

#### Images:
- `images/confusion_matrix_rf.png` — Visualisation of Random Forest predictions
- `images/classification_report_rf.txt` — Full precision/recall breakdown

#### Insight:
Although the accuracy was similar to logistic regression, the Random Forest model was able to detect both classes equally well. This reflects a fairer and more stable decision boundary — especially important for volatile assets like Bitcoin.


### Improved V2 Model: Tuned Random Forest Classifier (Best Performance)

After testing multiple hyperparameter combinations, a **Random Forest model** with optimised settings was selected as the 'final' version.

This model showed the best trade-off between accuracy and balance across both target classes. It outperformed all previous configurations, including the default logistic regression and earlier forest versions.

#### Improved V2 Model Configuration

| Parameter            | Value   |
|----------------------|---------|
| `n_estimators`       | 100     |
| `max_depth`          | 8       |
| `min_samples_split`  | 50      |
| `min_samples_leaf`   | 2       |
| `max_features`       | 'sqrt'  |
| `random_state`       | 69      |


#### Final Evaluation Metrics

| Class               | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Did Not Increase (0)| 0.56      | 0.59   | 0.58     | 294     |
| Increased (1)       | 0.55      | 0.52   | 0.53     | 302     |

**Overall Metrics:**

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Accuracy      | –         | –      | **0.555**| 596     |
| Macro Avg     | 0.56      | 0.56   | 0.56     | 596     |
| Weighted Avg  | 0.56      | 0.56   | 0.56     | 596     |

#### Images:
- `images/confusion_matrix_rf_final.png` — Final model’s prediction breakdown
- `images/classification_report_rf_final.txt` — Detailed metrics for each class

#### Notes
This version of the model was chosen as the best-performing classifier based on accuracy, fairness across both labels, and resistance to overfitting. It represents the final model deployed in the Streamlit app and published to the public portfolio.




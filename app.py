import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Set page title and layout
st.set_page_config(page_title="Crypto Trend Analysis", layout="wide")

# Title and intro
st.title("Crypto Trend Analysis — BTC-USD")

st.markdown("""
Welcome to my crypto trend exploration tool built using real Bitcoin price data.  
This dashboard shows EDA, volatility patterns, and machine learning predictions  
trained to forecast whether Bitcoin will go up or not.
""")

# Sidebar for navigation
section = st.sidebar.radio("**Navigate**", ["Overview", "Price Trend", "Volatility", "ML Models", "Insights"])

if section == "Overview":
    st.header("Project Overview")

    st.markdown("""
    This app analyses the behaviour of Bitcoin over time, focusing on price trends, daily returns,  
    volatility patterns, and how machine learning can be used to make short-term predictions.

    The project started with raw price data pulled directly from Yahoo Finance using `yfinance`,  
    and built up to feature engineering and ML modelling.  
    The entire workflow reflects how data-driven decision making works in fintech platforms —  
    from understanding risk to predicting price movement.

    #### Key areas explored:
    - Historical price visualisation (bull and bear phases)
    - Daily return behaviour and volatility calculation
    - Handling imbalanced data using SMOTE
    - Using logistic regression and Random Forest to predict market direction
    - Evaluating model performance with confusion matrices and F1 scores

    **Data Source**: Yahoo Finance (`BTC-USD`)  
    **Date Range**: 2017-01-01 to 2025-03-01  
    **Frequency**: Daily  
    **Final Model Used**: Tuned Random Forest Classifier  
    """)



# === Section: Price Trend ===
elif section == "Price Trend":
    st.header("Bitcoin Closing Price Over Time")

    st.markdown("""
    This section shows how Bitcoin’s closing price has changed over time — from the early 2017  
    rally to the 2021 boom and beyond. It helps contextualise market phases like bull runs,  
    crashes, and periods of consolidation.

    Analysing long-term price movement is the foundation for trend-based investing and  
    risk assessment. Here, it gives a clear look at the volatility inherent in the crypto space,  
    and sets the stage for the return and volatility analysis that follows.
    """)

    st.image("images/btc_closing_price_over_time.png", use_container_width=True)



# === Section: Volatility ===
elif section == "Volatility":
    st.header("Bitcoin Volatility and Daily Return Behaviour")

    st.markdown("""
    Volatility measures how much the price of Bitcoin fluctuates over time — a key signal  
    for traders, risk managers, and fintech platforms.

    The first chart below shows the 30-day rolling standard deviation of daily returns.  
    This gives a smooth view of how volatile the market has been in different phases,  
    highlighting extreme periods like post-2017 and early COVID.

    The second chart is a histogram showing how daily returns are distributed.  
    Most daily moves are small, but there are occasional sharp gains or crashes.  
    Understanding this spread helps quantify risk and build features for ML models.
    """)

    st.image("images/btc_volatility_30d.png", caption="30-Day Rolling Volatility of Daily Returns", use_container_width=True)
    st.markdown("---")
    st.image("images/btc_daily_return_distribution.png", caption="Distribution of Daily Returns", use_container_width=True)



# === Section: ML Models ===
elif section == "ML Models":
    
    def load_classification_table(report_path):
        with open(report_path, "r") as f:
            lines = f.readlines()
        
        # Parse only the two class rows
        class_rows = lines[2:4]
        rows = []
        for line in class_rows:
            tokens = line.strip().split()
            label = " ".join(tokens[:-4])  # Handles class names like 'Did Not Increase'
            precision = float(tokens[-4])
            recall = float(tokens[-3])
            f1 = float(tokens[-2])
            support = int(tokens[-1])
            rows.append([label, precision, recall, f1, support])
        df = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-Score", "Support"])
        
        # Find the line with accuracy
        accuracy = None
        for line in lines:
            if "accuracy" in line.lower():
                parts = line.strip().split()
                try:
                    accuracy = float(parts[-2])
                except:
                    continue
        return df, accuracy

    # List of model blocks
    model_blocks = [
        {
            "title": "Initial Logistic Regression (Unbalanced)",
            "image": "images/confusion_matrix_initial.png",
            "report": "reports/classification_report_initial.txt",
            "caption": "Confusion Matrix — Initial Model",
            "note": """
            **What this shows:**  
            This model was trained on raw, imbalanced data. It learned to always predict class `1` (Bitcoin will go up)  
            and completely ignored the `0` class. Accuracy looked okay, but it was misleading.
            """
        },
        {
            "title": "Balanced Logistic Regression (with SMOTE)",
            "image": "images/confusion_matrix_balanced.png",
            "report": "reports/classification_report_balanced.txt",
            "caption": "Confusion Matrix — SMOTE Balanced",
            "note": """
            **What this shows:**  
            After applying SMOTE, the model was exposed to synthetic examples of class `0`.  
            This gave it the chance to learn from both classes and improve F1 scores for class `0`.
            """
        },
        {
            "title": "Initial Random Forest Classifier Model",
            "image": "images/confusion_matrix_rf.png",
            "report": "reports/classification_report_rf.txt",
            "caption": "Confusion Matrix — Untuned Random Forest",
            "note": """
            **What this shows:**   
            This model was trained using the Random Forest algorithm on SMOTE-balanced data  
            but without any hyperparameter tuning. It performed similar to the logistic regression model,  
            but still lacked optimisation for deeper patterns or overfitting control.
            """
        },
        {
            "title": "Tuned Random Forest Model",
            "image": "images/confusion_matrix_rf_improved_v2.png",
            "report": "reports/classification_report_rf_improved_v2.txt",
            "caption": "Confusion Matrix — Final Random Forest",
            "note": """
            **What this shows:**  
            This model used SMOTE and tuned hyperparameters. It had the best balance between class predictions,  
            and gave the highest combined F1 score. This is the model deployed on this app.
            """
        }
    ]

    st.header("Machine Learning Model Progression")
    st.markdown("""
    This section shows the evolution of models used to predict whether Bitcoin’s price  
    would increase the next day. Each model used the same features, but different training strategies.
    """)

    # Use columns to display image and report side by side
    for block in model_blocks:
        st.subheader(block["title"])
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(block["image"], caption=block["caption"], use_container_width=True)
        with col2:
            
            # Load the classification table and accuracy
            df, acc = load_classification_table(block["report"])
            st.markdown("##### Classification Report")
            st.dataframe(df.set_index("Class"))
            if acc is not None:
                st.markdown(f"**Accuracy:** `{acc:.3f}`")
        st.markdown(block["note"])
        st.markdown("---")



# === Section: Insights ===
elif section == "Insights":
    st.header("Project Insights and Reflections")

    st.markdown("""
    This project combined real-world financial data with data science techniques  
    to explore price behaviour and predict short-term Bitcoin movements.

    Key Takeaways:
    - Volatility patterns revealed Bitcoin’s most unstable periods, often following sharp rallies.
    - Daily return analysis showed that large price swings are rare but impactful.
    - A Random Forest classifier, tuned and trained on SMOTE-balanced data, gave the best predictive performance.
    - Model evaluation focused on class balance, using precision, recall, and F1-score, not just accuracy.

    What This Demonstrates:
    - How financial features like returns and volatility can feed into machine learning predictions.
    - The importance of fixing class imbalance to avoid biased models.
    - That a good model balances performance across both target classes.

    Next Steps:
    - Add more features like technical indicators (e.g., moving averages, RSI).
    - Try more advanced models like XGBoost or LSTMs.
    - Turn the model into an API that connects to live data.
    - Add probabilistic outputs or confidence scores.

    This project shows how a fintech platform might analyse crypto trends  
    to support features like risk alerts, portfolio recommendations, or trade insights.

    This dashboard was built using Python, Streamlit, and real Bitcoin market data  
    as a live portfolio project to demonstrate applied machine learning in finance.
    """)
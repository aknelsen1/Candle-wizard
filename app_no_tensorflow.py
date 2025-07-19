
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import smtplib
from email.message import EmailMessage
from sklearn.preprocessing import MinMaxScaler

# === APP CONFIG ===
st.set_page_config(page_title="Candle Wizard", layout="centered", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: white;'>Candle Wizard</h1>", unsafe_allow_html=True)

# === DATA TYPE SELECTION ===
data_type = st.selectbox("Select Data Type", ["Stocks", "Crypto", "Forex"])
ticker_input = st.text_input("Enter Ticker (e.g. AAPL, BTC-USD, EURUSD=X):", value="AAPL")

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2023-01-01", end="2025-01-01")

data = load_data(ticker_input)

# === Candlestick Pattern Detection ===
def detect_bullish_engulfing(df):
    prev = df.shift(1)
    pattern = (prev['Close'] < prev['Open']) &               (df['Close'] > df['Open']) &               (df['Open'] < prev['Close']) &               (df['Close'] > prev['Open'])
    return pattern.astype(int)

def detect_hammer(df):
    body = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']
    lower_shadow = df['Open'].where(df['Open'] < df['Close'], df['Close']) - df['Low']
    return ((lower_shadow > 2 * body) & (body / candle_range < 0.3)).astype(int)

data['bullish_engulfing'] = detect_bullish_engulfing(data)
data['hammer'] = detect_hammer(data)

# === Feature Engineering ===
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'bullish_engulfing', 'hammer']].copy()
features.dropna(inplace=True)
labels = (data['Close'].shift(-1) > data['Close']).astype(int).dropna()
features = features.iloc[:-1]

# === Train/Test Split & Model ===
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
latest_features = features.iloc[-1:]
prediction = model.predict(latest_features)[0]
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# === Display Results ===
st.subheader("📈 ML Model Prediction")
if prediction == 1:
    st.success("Prediction: Bullish 📈")
else:
    st.error("Prediction: Bearish 📉")
st.write(f"Random Forest Accuracy: {accuracy:.2%}")

# === Email Alert ===
st.subheader("📧 Email Alert (Optional)")
if st.checkbox("Send me an email if prediction is Bullish"):
    email = st.text_input("Enter your email address:")
    if prediction == 1 and email:
        msg = EmailMessage()
        msg.set_content(f"Candle Wizard: Bullish signal for {ticker_input} tomorrow!")
        msg["Subject"] = f"Candle Wizard Alert - {ticker_input}"
        msg["From"] = "your_email@gmail.com"
        msg["To"] = email
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login("your_email@gmail.com", "your_app_password")
                smtp.send_message(msg)
            st.success("Email alert sent.")
        except Exception as e:
            st.error(f"Email failed: {e}")

# === Candlestick Chart ===
st.subheader("📊 Last 30-Day Candlestick Chart")
fig, axlist = mpf.plot(data[-30:], type='candle', style='charles', mav=(3,6,9), volume=True, returnfig=True)
st.pyplot(fig)

# === Raw Data Toggle ===
if st.checkbox("Show raw data"):
    st.write(data.tail())

# === Backtesting ===
st.subheader("📉 Backtest: Bullish Engulfing")
def backtest(df, pattern_col, label_col):
    results = df[[pattern_col]].copy()
    results['next_day_movement'] = label_col
    results['strategy_signal'] = df[pattern_col]
    results['strategy_result'] = results['strategy_signal'] * results['next_day_movement']
    win_rate = results['strategy_result'].sum() / results['strategy_signal'].sum() if results['strategy_signal'].sum() > 0 else 0
    return win_rate

bt_result = backtest(data.iloc[:-1], 'bullish_engulfing', labels)
st.write(f"Win Rate: {bt_result:.2%}")

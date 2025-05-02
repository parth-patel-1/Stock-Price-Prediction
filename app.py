import streamlit as st
import numpy as np
import tensorflow as tf
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score


# NewsAPI Key (replace with your actual key)
newsapi = NewsApiClient(api_key='f0a8599b1a5e4f359c4f97aebf185a26')

model_paths = {
    'AAPL': 'models/AAPL.keras',
    'TSLA': 'models/TSLA.keras',
    'GOOG': 'models/GOOG.keras',
    'MSFT': 'models/MSFT.keras'
}

scaler_paths = {
    'AAPL': 'scalers/AAPL_scaler.save',
    'TSLA': 'scalers/TSLA_scaler.save',
    'GOOG': 'scalers/GOOG_scaler.save',
    'MSFT': 'scalers/MSFT_scaler.save'
}

def load_model(symbol):
    return tf.keras.models.load_model(model_paths[symbol])

def load_scaler(symbol):
    return joblib.load(scaler_paths[symbol])

def get_stock_prices(symbol, period="130d"):
    data = yf.download(symbol, period=period, interval="1d")
    return data.dropna()

def predict_next_day(model, scaler, data):
    scaled = scaler.transform(data.reshape(-1, 1)).reshape(1, 10, 1)
    pred_scaled = model.predict(scaled)
    return scaler.inverse_transform(pred_scaled).flatten()[0]

def evaluate_model_on_past_data(prices, model, scaler):
    preds, targets = [], []
    max_start = len(prices) - 10 - 1  # needs at least 11 points
    if max_start < 1:
        return None, None, None, None

    for i in range(max_start):
        input_seq = prices[i:i+10]
        actual = prices[i+10]
        pred = predict_next_day(model, scaler, input_seq)
        preds.append(pred)
        targets.append(actual)

    if not preds or not targets:
        return None, None, None, None

    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    mape = mean_absolute_percentage_error(targets, preds) * 100
    r2 = r2_score(targets, preds)
    return preds, targets, mape, r2

def get_news(symbol):
    news = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page_size=5)
    return [(a['title'], a['description'], a['url']) for a in news.get('articles', [])]

# Streamlit UI
st.title("📈 Stock Price Predictor by Neural Pioneers Group")

symbol = st.selectbox("Select Stock Symbol", ["AAPL", "TSLA", "GOOG", "MSFT"])

if st.button("Predict & Show News"):
    with st.spinner("Processing..."):
        data_df = get_stock_prices(symbol)
        close_prices = data_df['Close'].values

        if len(close_prices) < 110:
            st.warning(f"Only {len(close_prices)} data points found. Trying to fetch more...")
            data_df = get_stock_prices(symbol, period="180d")
            close_prices = data_df['Close'].values
            if len(close_prices) < 110:
                st.error("Still not enough data after retry. Try again later.")
                st.stop()
            else:
                st.success(f"Fetched {len(close_prices)} valid days.")

        # Recent 10 days for prediction
        recent_prices = close_prices[-10:]
        recent_dates = data_df.index[-10:]

        # Load model and scaler
        model = load_model(symbol)
        scaler = load_scaler(symbol)

        predicted_price = predict_next_day(model, scaler, recent_prices)
        current_price = recent_prices[-1]

        predicted_price = float(predicted_price)
        current_price = float(current_price)
        price_diff = predicted_price - current_price
        change_pct = (price_diff / current_price) * 100

        #Short recommendation with color
        if change_pct > 0.5:
            short_recommendation = '<span style="color:green">BUY 📈</span>'
        elif change_pct < -0.5:
            short_recommendation = '<span style="color:red">SELL 📉</span>'
        else:
            short_recommendation = '<span style="color:black">HOLD ➖</span>'

        # Prediction Display
        st.subheader("📍 Predicted Next Day Price")
        st.markdown(f"""
            **{symbol} (Next Trading Day)**  
            ### {predicted_price:.2f} &nbsp;&nbsp;&nbsp; {short_recommendation}
        """, unsafe_allow_html=True)

        # Matplotlib Price Plot
        fig, ax = plt.subplots()
        ax.plot(recent_dates, recent_prices, marker='o', label="Last 10 Days")
        ax.plot(recent_dates[-1] + pd.Timedelta(days=1), predicted_price, marker='X', color='red', markersize=10, label="Predicted")
        ax.set_title(f"{symbol} Stock Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 🚦 Large Recommendation
        st.subheader("🚦 Final Recommendation")
        if change_pct > 0.5:
            st.success(f"""
            ### ✅ **Recommendation: BUY**
            Predicted price is **{predicted_price:.2f}**, which is **{change_pct:.2f}% higher** than the current price **{current_price:.2f}**.
            Suggests upward movement. Consider buying or increasing position.
            """)
        elif change_pct < -0.5:
            st.error(f"""
            ### ❌ **Recommendation: SELL**
            Predicted price is **{predicted_price:.2f}**, which is **{abs(change_pct):.2f}% lower** than the current price **{current_price:.2f}**.
            Suggests decline. Consider selling or reducing position.
            """)
        else:
            st.warning(f"""
            ### ⚠️ **Recommendation: HOLD**
            Predicted price is **{predicted_price:.2f}**, only **{change_pct:.2f}% different** from current price **{current_price:.2f}**.
            No significant movement expected. Hold your position.
            """)

        # Model Evaluation
        st.subheader("📊 Model Performance on Past 100 Days")
        eval_prices = close_prices[-110:]
        eval_preds, eval_targets, mape, r2 = evaluate_model_on_past_data(eval_prices, model, scaler)

        if eval_preds is not None:
            fig2, ax2 = plt.subplots()
            ax2.plot(eval_targets, label='Actual', marker='o')
            ax2.plot(eval_preds, label='Predicted', marker='x')
            ax2.set_title("Model Prediction vs Actual (Last 100 Days)")
            ax2.set_xlabel("Days")
            ax2.set_ylabel("Price")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

            st.markdown(f"**MAPE**: `{mape:.2f}%`")
            st.markdown(f"**R² Score**: `{r2:.4f}`")

        else:
            st.warning("Not enough data for 100-day evaluation.")

        # News Headlines
        st.subheader("📰 Latest News Headlines")
        headlines = get_news(symbol)
        if headlines:
            for i, (title, desc, url) in enumerate(headlines, 1):
                st.markdown(f"**{i}. {title}**")
                if desc:
                    preview = desc.strip().split(".")[0][:160]
                    st.write(f"*{preview}...*")
                st.markdown(f"[🔗 Read more]({url})")
                st.write("---")
        else:
            st.info("No recent news found.")

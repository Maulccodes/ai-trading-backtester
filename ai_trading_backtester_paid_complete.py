import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import io
from fpdf import FPDF
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Page Setup
st.set_page_config(page_title="AI Trading Strategy Backtester (Paid)", layout="wide")
st.title("💼 AI Trading Strategy Backtester (Pro Version)")

# Sidebar Inputs


# Data Loading Functions
def load_data_from_file(file):
    try:
        df = pd.read_csv(file)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns]
        if not any('Close' in col for col in df.columns):
            st.error("❌ CSV must contain a 'Close' column.")
            return None, None
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        close_col = next((col for col in df.columns if 'Close' in col), None)
        return df, close_col
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

def load_data_from_yahoo(ticker):
    try:
        df = yf.download(ticker, period="5y")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns]
        close_col = [col for col in df.columns if 'Close' in col][0]
        return df, close_col
    except Exception as e:
        st.error(f"Yahoo error: {e}")
        return None, None

# Strategy Implementations
def apply_ma_strategy(df, close_col, short, long):
    df = df.copy()
    df['ShortMA'] = df[close_col].rolling(window=short).mean()
    df['LongMA'] = df[close_col].rolling(window=long).mean()
    df['Signal'] = 0
    df.loc[df['ShortMA'] > df['LongMA'], 'Signal'] = 1
    df['Position'] = df['Signal'].diff()
    return df.dropna()

def apply_rsi_strategy(df, close_col, period=14):
    df = df.copy()
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Signal'] = 0
    df.loc[df['RSI'] < 30, 'Signal'] = 1
    df.loc[df['RSI'] > 70, 'Signal'] = 0
    df['Position'] = df['Signal'].diff()
    return df.dropna()

def apply_macd_strategy(df, close_col):
    df = df.copy()
    exp1 = df[close_col].ewm(span=12, adjust=False).mean()
    exp2 = df[close_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Signal'] = (df['MACD'] > df['Signal Line']).astype(int)
    df['Position'] = df['Signal'].diff()
    return df.dropna()

# AI Prediction with Robust Error Handling
def ai_predict_next_close(df, close_col):
    try:
        if df is None or close_col not in df.columns or len(df) < 2:
            return None
            
        df = df.copy().dropna()
        if len(df) < 2:  # Need at least 2 data points
            return None
            
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[close_col].values
        
        if len(X) == 0 or len(y) == 0 or np.isnan(y).all():
            return None
            
        model = LinearRegression()
        model.fit(X, y)
        return float(model.predict([[len(df)]])[0])
    except Exception:
        return None

# Performance Calculation
def calculate_returns(df, close_col):
    df = df.copy()
    df['Returns'] = df[close_col].pct_change()
    df['Strategy'] = df['Returns'] * df['Signal'].shift(1)
    df['Cumulative Market Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative Strategy Returns'] = (1 + df['Strategy']).cumprod()
    return df

# Report Generation
def generate_pdf(strategy_name, market_return, strategy_return):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Trading Strategy Backtester Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Strategy: {strategy_name}", ln=True)
    pdf.cell(0, 10, f"Final Market Return: {market_return:.2f}x", ln=True)
    pdf.cell(0, 10, f"Final Strategy Return: {strategy_return:.2f}x", ln=True)
    pdf_output = pdf.output(dest='S').encode('latin-1')
    return pdf_output

# ChatGPT Tips
def get_chatgpt_tip(strategy_name):
    try:
        if not openai.api_key:
            return "⚠️ OpenAI API key not configured"
            
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise trading expert."},
                {"role": "user", "content": f"Give one short tip (1 sentence) for using {strategy_name} strategy"}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    
    except openai.AuthenticationError:
        return "🔑 Invalid OpenAI API key (check .env file)"
    except openai.RateLimitError:
        return "🚦 OpenAI API limit reached (try later or upgrade plan)"
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"


# Add near the top after load_dotenv()
if not openai.api_key:
    st.error("❌ OpenAI API key missing. Create a .env file with OPENAI_API_KEY=your_key")
    st.stop()  # Halt the app if no key

# Main App Execution
def main():
    try:
        # Sidebar Inputs
        strategy_choice = st.sidebar.selectbox(
            "Choose Strategy",
            ["Moving Average Crossover", "RSI", "MACD", "Ensemble (Compare All)"]
        )
        short_window = st.sidebar.slider("Short Moving Average Window", 5, 50, 20)
        long_window = st.sidebar.slider("Long Moving Average Window", 10, 200, 50)
        ticker_input = st.sidebar.text_input("Or enter a Ticker (e.g., AAPL)", value="")

        df, close_col = None, None
        uploaded_file = st.file_uploader("📂 Upload CSV with Date & Close", type=["csv"])

        # Load data from CSV or Yahoo
        if uploaded_file:
            df, close_col = load_data_from_file(uploaded_file)
        elif ticker_input:
            df, close_col = load_data_from_yahoo(ticker_input.strip().upper())

        if df is not None and close_col is not None:
            st.line_chart(df[close_col])

            # Strategy selection
            if strategy_choice == "Moving Average Crossover":
                df = apply_ma_strategy(df, close_col, short_window, long_window)
            elif strategy_choice == "RSI":
                df = apply_rsi_strategy(df, close_col)
            elif strategy_choice == "MACD":
                df = apply_macd_strategy(df, close_col)

            df = calculate_returns(df, close_col)

            # Safe AI prediction
            predicted_price = ai_predict_next_close(df, close_col)
            if predicted_price is not None:
                st.metric("📉 AI Predicted Next Close", f"${predicted_price:.2f}")
            else:
                st.warning("⚠️ Could not generate price prediction (insufficient data)")

            # Show backtest results
            st.line_chart(df[[close_col, "Cumulative Strategy Returns"]])
            market_return = df["Cumulative Market Returns"].iloc[-1]
            strategy_return = df["Cumulative Strategy Returns"].iloc[-1]
            st.write(f"📈 Market Return: {market_return:.2f}x")
            st.write(f"📈 Strategy Return: {strategy_return:.2f}x")

            # Export PDF report
            pdf_data = generate_pdf(strategy_choice, market_return, strategy_return)
            st.download_button("🧾 Download PDF Report", pdf_data, file_name="report.pdf")

            # Get ChatGPT tip
            st.subheader("💡 ChatGPT Strategy Tip")
            st.info(get_chatgpt_tip(strategy_choice))

        else:
            st.info("🟡 Please upload a file or enter a stock ticker to begin.")

    except Exception as e:
        if "WebSocketClosedError" in str(e):
            st.warning("🔌 Connection interrupted - please refresh the page.")
        else:
            st.error(f"❌ Unexpected error: {str(e)}")

# Run main function
if __name__ == "__main__":
    main()

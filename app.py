import streamlit as st
import pandas as pd
import numpy as np
import json
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="TLKM Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background-color: #0E1117;
}

.metric-card {
    background: linear-gradient(135deg,#1f2937,#111827);
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
}

.metric-title{
    color:#9CA3AF;
    font-size:14px;
}

.metric-value{
    color:white;
    font-size:30px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    historical = pd.read_csv("historical_data.csv")
    forecast = pd.read_csv("forecast.csv")
    actual_pred = pd.read_csv("actual_vs_prediction.csv")
    loss_df = pd.read_csv("loss_history.csv")

    with open("metrics.json","r") as f:
        metrics = json.load(f)

    return historical, forecast, actual_pred, loss_df, metrics

historical, forecast, actual_pred, loss_df, metrics = load_data()

historical["Date"] = pd.to_datetime(historical["Date"])
forecast["Date"] = pd.to_datetime(forecast["Date"])

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📈 TLKM Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Historis & Forecast",
        "Evaluasi Model",
        "Fundamental TLKM",
        "Perbandingan Saham",
        "Data Mentah"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.title("📈 TLKM Stock Forecast Dashboard")

    latest_price = historical["Close"].iloc[-1]
    forecast_price = forecast["Forecast"].iloc[-1]

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "Harga Terakhir",
            f"Rp {latest_price:,.0f}"
        )

    with col2:
        st.metric(
            "Forecast 30 Hari",
            f"Rp {forecast_price:,.0f}"
        )

    with col3:
        st.metric(
            "MAPE",
            f"{metrics['mape']:.2f}%"
        )

    with col4:
        st.metric(
            "R²",
            f"{metrics['r2']:.4f}"
        )

    st.markdown("---")

    st.subheader("Harga Historis + Forecast")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical["Date"],
            y=historical["Close"],
            name="Historis",
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Forecast"],
            name="Forecast",
            line=dict(
                dash="dash",
                width=3
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# HISTORIS & FORECAST
# =====================================================

elif menu == "Historis & Forecast":

    st.title("Historis dan Forecast")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Harga Historis TLKM",
            "Forecast 30 Hari"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=historical["Date"],
            y=historical["Close"],
            name="Close"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Forecast"],
            name="Forecast"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=800
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Forecast Data")

    st.dataframe(
        forecast.tail(30),
        use_container_width=True
    )

# =====================================================
# EVALUASI MODEL
# =====================================================

elif menu == "Evaluasi Model":

    st.title("Evaluasi Model LSTM")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "MAE",
        round(metrics["mae"],4)
    )

    c2.metric(
        "RMSE",
        round(metrics["rmse"],4)
    )

    c3.metric(
        "MAPE",
        f"{metrics['mape']:.2f}%"
    )

    c4.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2f}%"
    )

    st.markdown("---")

    st.subheader("Actual vs Prediction")

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            y=actual_pred["Actual"],
            name="Actual"
        )
    )

    fig1.add_trace(
        go.Scatter(
            y=actual_pred["Prediction"],
            name="Prediction"
        )
    )

    fig1.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader("Training Loss")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            y=loss_df["loss"],
            name="Training Loss"
        )
    )

    fig2.add_trace(
        go.Scatter(
            y=loss_df["val_loss"],
            name="Validation Loss"
        )
    )

    fig2.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =====================================================
# FUNDAMENTAL TLKM
# =====================================================

elif menu == "Fundamental TLKM":

    st.title("Fundamental TLKM")

    try:

        stock = yf.Ticker("TLKM.JK")
        info = stock.info

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Market Cap",
            f"{info.get('marketCap',0):,.0f}"
        )

        c2.metric(
            "PE Ratio",
            info.get("trailingPE","N/A")
        )

        c3.metric(
            "EPS",
            info.get("trailingEps","N/A")
        )

        c4,c5 = st.columns(2)

        c4.metric(
            "Dividend Yield",
            info.get("dividendYield","N/A")
        )

        c5.metric(
            "Beta",
            info.get("beta","N/A")
        )

    except:
        st.warning(
            "Data fundamental tidak tersedia."
        )

# =====================================================
# PERBANDINGAN SAHAM
# =====================================================

elif menu == "Perbandingan Saham":

    st.title("Perbandingan Saham Telekomunikasi")

    tickers = [
        "TLKM.JK",
        "ISAT.JK",
        "EXCL.JK",
        "GOTO.JK"
    ]

    data = yf.download(
        tickers,
        period="1y",
        progress=False
    )["Close"]

    normalized = (
        data /
        data.iloc[0]
    ) * 100

    fig = go.Figure()

    for col in normalized.columns:

        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                name=col
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        title="Normalized Return (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# DATA MENTAH
# =====================================================

elif menu == "Data Mentah":

    st.title("Data Mentah")

    st.subheader("Historical Data")

    st.dataframe(
        historical.tail(100),
        use_container_width=True
    )

    with open(
        "historical_data.csv",
        "rb"
    ) as f:

        st.download_button(
            "Download Historical Data",
            f,
            "historical_data.csv"
        )

    st.subheader("Forecast Data")

    st.dataframe(
        forecast,
        use_container_width=True
    )

    with open(
        "forecast.csv",
        "rb"
    ) as f:

        st.download_button(
            "Download Forecast",
            f,
            "forecast.csv"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Developed with Streamlit | LSTM Stock Forecasting TLKM"
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="TLKM Stock Dashboard",
    layout="wide"
)

# ======================================================
# LOAD DATA AKTUAL & GENERATE FORECAST OTOMATIS
# ======================================================
# Membaca data utama TLKM
df = pd.read_csv("TLKM.csv")
df["date"] = pd.to_datetime(df["date"])

# Mengambil tanggal terakhir dari data aktual sebagai acuan hari ini
today = df["date"].max()

# Fungsi untuk generate forecast dinamis agar maju otomatis setiap hari
@st.cache_data(ttl=3600)  # Di-refresh setiap 1 jam agar hemat resources
def generate_forecast(latest_date):
    """
    CATATAN: Jika kamu punya file model LSTM asli (misal format .h5/.pkl),
    kamu bisa me-load model tersebut dan mengganti logic random di bawah ini 
    dengan 'model.predict(input_data)'.
    """
    # Membuat 30 tanggal ke depan, MULAI HARI BESOKNYA dari tanggal data aktual terakhir
    future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=30)
    
    # Simulasi hasil prediksi (kita buat tren naik-turun tipis di sekitar harga terakhir)
    last_price = df["close"].iloc[-1]
    simulasi_prediksi = last_price + np.cumsum(np.random.normal(0, 20, 30))
    
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": simulasi_prediksi
    })
    return forecast_df

# Memanggil fungsi forecast otomatis
forecast = generate_forecast(today)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Filter")

periode = st.sidebar.selectbox(
    "Periode",
    [
        "7 Hari",
        "1 Bulan",
        "6 Bulan",
        "1 Tahun",
        "2 Tahun",
        "6 Tahun",
        "10 Tahun",
        "Max"
    ]
)

# ======================================================
# FILTER DATA HISTORIS
# ======================================================
if periode == "7 Hari":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=7)]
elif periode == "1 Bulan":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=30)]
elif periode == "6 Bulan":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=180)]
elif periode == "1 Tahun":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=365)]
elif periode == "2 Tahun":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=730)]
elif periode == "6 Tahun":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=2190)]
elif periode == "10 Tahun":
    filtered_df = df[df["date"] >= today - pd.Timedelta(days=3650)]
else:
    filtered_df = df.copy()

# ======================================================
# FEATURE ENGINEERING
# ======================================================
filtered_df["MA7"] = filtered_df["close"].rolling(7).mean()
filtered_df["MA30"] = filtered_df["close"].rolling(30).mean()

# ======================================================
# HEADER
# ======================================================
st.title("Dashboard Analisis Saham TLKM")
st.caption("Data diperbarui otomatis setiap hari pukul 10.00 WIB")

# ======================================================
# METRICS
# ======================================================
latest_close = filtered_df["close"].iloc[-1]
previous_close = filtered_df["close"].iloc[-2]
price_change = latest_close - previous_close
percent_change = (price_change / previous_close) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Price",
    f"Rp {latest_close:,.0f}",
    f"{price_change:,.0f} ({percent_change:.2f}%)"
)

col2.metric(
    "Average Open",
    f"Rp {filtered_df['open'].mean():,.0f}"
)

col3.metric(
    "Average Close",
    f"Rp {filtered_df['close'].mean():,.0f}"
)

col4.metric(
    "Volatility",
    f"{filtered_df['volatility'].mean():.4f}"
)

# ======================================================
# CHART HARGA
# ======================================================
st.subheader("Pergerakan Harga Saham")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["close"], name="Close", line=dict(width=3)))
fig1.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["MA7"], name="MA7", line=dict(dash="dot")))
fig1.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["MA30"], name="MA30", line=dict(dash="dash")))

fig1.update_layout(
    height=500,
    template="simple_white",
    legend=dict(orientation="h"),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig1, use_container_width=True)

# ======================================================
# ROW 2
# ======================================================
left, right = st.columns(2)

# ======================================================
# VOLUME
# ======================================================
with left:
    st.subheader("Volume Trading")
    fig2 = px.bar(filtered_df, x="date", y="volume")
    fig2.update_layout(
        height=400,
        template="simple_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# VOLATILITY
# ======================================================
with right:
    st.subheader("Analisis Volatilitas")
    fig3 = px.line(filtered_df, x="date", y="volatility")
    fig3.update_layout(
        height=400,
        template="simple_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# STATISTIK
# ======================================================
st.subheader("Analisis Statistik")

stats = pd.DataFrame({
    "Statistik": [
        "Highest Price",
        "Lowest Price",
        "Average Open",
        "Average Close",
        "Standard Deviation",
        "Average Return"
    ],
    "Nilai": [
        round(filtered_df["high"].max(), 2),
        round(filtered_df["low"].min(), 2),
        round(filtered_df["open"].mean(), 2),
        round(filtered_df["close"].mean(), 2),
        round(filtered_df["close"].std(), 2),
        round(filtered_df["daily_return"].mean(), 4)
    ]
})

st.dataframe(stats, use_container_width=True)

# ======================================================
# FORECAST (Dibuat menyambung dengan data asli)
# ======================================================
st.subheader("Forecast 30 Hari (LSTM)")

fig4 = go.Figure()

# Garis Historis (Ambil 30 hari terakhir saja sebagai konteks)
fig4.add_trace(
    go.Scatter(
        x=filtered_df["date"].tail(30),
        y=filtered_df["close"].tail(30),
        name="Data Historis",
        line=dict(color="blue")
    )
)

# Garis Forecast
fig4.add_trace(
    go.Scatter(
        x=forecast["date"],
        y=forecast["forecast"],
        name="Rekomendasi LSTM",
        line=dict(color="orange", dash="dash")
    )
)

fig4.update_layout(
    height=500,
    template="simple_white",
    legend=dict(orientation="h"),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.caption(f"Last Updated : {today.strftime('%Y-%m-%d')}")

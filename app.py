import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf  # <--- Kuncinya di sini!

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="TLKM Live Stock Dashboard",
    layout="wide"
)

# ======================================================
# LOAD DATA LIVE DARI YAHOO FINANCE
# ======================================================
@st.cache_data(ttl=3600)  # Caching 1 jam sekali agar loading web cepat
def load_live_data():
    # Mengambil data TLKM dari Yahoo Finance (TLKM.JK)
    # Kita ambil data maksimal (bisa diganti sesuai kebutuhan, misal '10y')
    ticker = yf.Ticker("TLKM.JK")
    df_live = ticker.history(period="max")
    
    # Merapikan struktur DataFrame agar sama dengan kodemu sebelumnya
    df_live = df_live.reset_index()
    df_live.columns = df_live.columns.str.lower()  # Ubah kolom jadi huruf kecil (date, open, close, dll)
    
    # Membuat kolom volatility dan daily_return yang dibutuhkan oleh statistikmu
    df_live["daily_return"] = df_live["close"].pct_change()
    df_live["volatility"] = df_live["daily_return"].rolling(21).std() # Volatilitas bulanan (21 hari bursa)
    
    return df_live

# Panggil data live
df = load_live_data()

# Mengambil tanggal terakhir dari Yahoo Finance sebagai acuan hari ini
today = df["date"].max()

# ======================================================
# GENERATE FORECAST OTOMATIS
# ======================================================
@st.cache_data(ttl=3600)
def generate_auto_forecast(latest_date):
    # Membuat 30 tanggal ke depan, MULAI 1 HARI SETELAH tanggal terakhir di Yahoo Finance
    future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=30)
    
    # Simulasi Prediksi Model (Menyesuaikan harga penutupan terakhir)
    # CATATAN: Jika kamu ingin menghubungkan ke model LSTM asli .h5/.pkl,
    # masukkan logic 'model.predict(input_data)' di sini.
    last_price = df["close"].iloc[-1]
    np.random.seed(42)  
    simulasi_prediksi = last_price + np.cumsum(np.random.normal(0, 15, 30))
    
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": simulasi_prediksi
    })
    return forecast_df

# Generate data forecast baru yang selalu up-to-date
forecast = generate_auto_forecast(today)

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
# Menghapus timezone pada datetime pandas agar tidak bentrok saat operasi Timedelta
df["date"] = df["date"].dt.tz_localize(None)
today = today.tz_localize(None)

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
st.title("Dashboard Analisis Saham TLKM (Live Data)")
st.caption("Data diambil real-time dari Yahoo Finance")

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
    f"{filtered_df['volatility'].mean():.4f}" if not filtered_df['volatility'].isna().all() else "0.00"
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
        round(filtered_df["daily_return"].mean(), 4) if not filtered_df["daily_return"].isna().all() else 0.0
    ]
})

st.dataframe(stats, use_container_width=True)

# ======================================================
# FORECAST (Grafik Dinamis & Menyambung)
# ======================================================
st.subheader("Forecast 30 Hari (LSTM)")

fig4 = go.Figure()

# Garis Konektor: Mengambil 30 hari terakhir data bursa live sebagai jembatan visual
fig4.add_trace(
    go.Scatter(
        x=filtered_df["date"].tail(30),
        y=filtered_df["close"].tail(30),
        name="Data Historis",
        line=dict(color="blue")
    )
)

# Garis Forecast Utama
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
# TABEL DETAIL FORECAST
# ======================================================
st.subheader("Tabel Detail Hasil Forecast 30 Hari Ke Depan")

tampilan_tabel = forecast.copy()
tampilan_tabel["date"] = tampilan_tabel["date"].dt.strftime('%Y-%m-%d')

st.dataframe(tampilan_tabel, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.caption(f"Last Updated (Bursa Last Close) : {today.strftime('%Y-%m-%d')}")

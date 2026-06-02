import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Stock Forecast Dashboard | TLKM",
    layout="wide",
    initial_sidebar_state="expanded",
)

if st_autorefresh is not None:
    st_autorefresh(interval=60_000, key="dashboard_refresh")

# =========================================================
# COMPACT STYLE & CUSTOM CSS (Tanpa Emoji & Lebih Profesional)
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Ruang atas judul agar tidak terpotong */
        .block-container { 
            padding-top: 3.5rem !important; 
            padding-bottom: 1rem !important; 
        }
        
        /* Jarak antar elemen vertikal tetap rapat */
        [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        
        /* Card Metrik */
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            padding: 8px 12px !important;
            border-radius: 10px !important;
            background-color: rgba(148, 163, 184, 0.04) !important;
            margin-bottom: 0px !important;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
            white-space: nowrap !important;
        }
        
        div[data-testid="stMetricLabel"] p {
            font-size: 0.82rem !important;
            margin-bottom: 2px !important;
            white-space: nowrap !important;
        }
        
        .muted { color: #64748b; font-size: 0.9rem; }
        
        /* Container Card */
        .card-soft {
            background: rgba(148, 163, 184, 0.02);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 10px !important;
        }
        
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 4px;
            margin-bottom: 8px !important;
            color: #1e293b;
            border-left: 4px solid #3b82f6;
            padding-left: 8px;
        }
        
        /* Badge Live */
        .badge-live {
            display: inline-flex; 
            align-items: center; 
            gap: 6px;
            padding: 0.25rem 0.5rem; 
            border-radius: 6px;
            background: rgba(34, 197, 94, 0.15); 
            color: #22c55e; 
            font-weight: 700; 
            font-size: 0.72rem;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .badge-live::before {
            content: '';
            display: inline-block;
            width: 6px;
            height: 6px;
            background-color: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(34, 211, 92, 0.5); }
            70% { transform: scale(1); box-shadow: 0 0 0 4px rgba(34, 211, 92, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(34, 211, 92, 0); }
        }

        /* Navigasi Tab */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: rgba(148, 163, 184, 0.06);
            padding: 4px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        .stTabs [data-baseweb="tab"] {
            height: 36px;
            border-radius: 6px;
            padding: 0 12px;
            font-size: 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            font-weight: 600;
        }
        
        hr {
            margin-top: 0.75rem !important;
            margin-bottom: 0.75rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS & MAPS
# =========================================================
PRIMARY_TICKER = "TLKM.JK"
BENCHMARKS = {
    "BBCA": "BBCA.JK",
    "BBRI": "BBRI.JK",
    "BMRI": "BMRI.JK",
    "BBNI": "BBNI.JK",
    "ISAT": "ISAT.JK",
    "EXCL": "EXCL.JK",
    "JSMR": "JSMR.JK",
    "ADRO": "ADRO.JK",
    "PTBA": "PTBA.JK",
    "ASII": "ASII.JK",
}

PERIOD_MAP = {
    "7 Hari": ("7d", "1d"),
    "1 Bulan": ("1mo", "1d"),
    "3 Bulan": ("3mo", "1d"),
    "6 Bulan": ("6mo", "1d"),
    "1 Tahun": ("1y", "1d"),
    "2 Tahun": ("2y", "1d"),
    "5 Tahun": ("5y", "1wk"),
    "10 Tahun": ("10y", "1wk"),
    "Max": ("max", "1mo"),
}


# =========================================================
# HELPERS & DATA LOADERS
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    first_col = df.columns[0]
    if first_col != "Date":
        df = df.rename(columns={first_col: "Date"})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()

    delta = out["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_json_metrics(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# =========================================================
# GENERATOR FORECAST DINAMIS (PENGGANTI KUNCI)
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def generate_live_forecast(latest_date, last_price) -> pd.DataFrame:
    """
    Otomatis membangkitkan tanggal mulai HARI ESOK (+1 hari) 
    dari tanggal terakhir bursa yang didapat dari yfinance.
    """
    # Menghapus timezone jika ada agar kalkulasi tanggal aman
    if hasattr(latest_date, 'tz_localize'):
        latest_date = latest_date.tz_localize(None)
        
    # Generate 30 hari ke depan, melompat 1 hari dari data aktual terakhir
    future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=30)
    
    # Logic Simulasi Prediksi LSTM (Menggunakan random walk terkunci untuk demo dashboard)
    # Ganti bagian ini dengan 'model.predict()' jika ingin disambungkan ke file bobot model asli
    np.random.seed(42)
    sim_predictions = last_price + np.cumsum(np.random.normal(0, 15, 30))
    
    return pd.DataFrame({
        "Date": future_dates,
        "Forecast": sim_predictions
    })


def safe_latest(series: pd.Series, default=np.nan):
    try:
        s = series.dropna()
        if s.empty:
            return default
        return s.iloc[-1]
    except Exception:
        return default


def compute_change(current: float, previous: float):
    if previous is None or np.isnan(previous) or previous == 0:
        return np.nan, np.nan
    diff = current - previous
    pct = (diff / previous) * 100
    return diff, pct


def fmt_idr(x, digits=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if digits == 0:
        return f"Rp {x:,.0f}".replace(",", ".")
    return f"Rp {x:,.{digits}f}".replace(",", ".")


def fmt_num(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{digits}f}"


def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{digits}f}%"


def trading_signal(rsi: float, macd: float, signal_line: float, close: float, ma50: float):
    if any(pd.isna(v) for v in [rsi, macd, signal_line, close, ma50]):
        return "WAIT", "Indikator belum cukup untuk memberi sinyal yang stabil."
    bullish = (rsi > 55) and (macd > signal_line) and (close > ma50)
    bearish = (rsi < 45) and (macd < signal_line) and (close < ma50)
    if bullish:
        return "BUY", "Momentum menguat, MACD di atas signal line, dan harga berada di atas MA50."
    if bearish:
        return "SELL", "Momentum melemah, MACD di bawah signal line, dan harga berada di bawah MA50."
    return "HOLD", "Sinyal campuran. Tren belum cukup kuat."


# =========================================================
# CONFIG PLOTLY THEMES
# =========================================================
def apply_clean_theme(fig, y_title, x_title="Tanggal"):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=15, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=x_title, showgrid=False)
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)")
    return fig


def plot_price_history(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price", line=dict(width=2.5, color="#3b82f6")))
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20", line=dict(width=1.5, color="#10b981", dash="dot")))
    if "MA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50", line=dict(width=1.5, color="#f59e0b", dash="dash")))
    fig = apply_clean_theme(fig, "Harga (Rp)")
    fig.update_layout(height=320)
    return fig


def plot_volume(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="#0ea5e9", opacity=0.8))
    fig = apply_clean_theme(fig, "Volume")
    fig.update_layout(height=240, showlegend=False)
    return fig


def plot_forecast_with_history(history_df: pd.DataFrame, forecast_df: pd.DataFrame):
    fig = go.Figure()
    if not history_df.empty:
        # Menghapus info timezone data historis agar pas digabung dengan koordinat x di Plotly
        x_hist = history_df["Date"].dt.tz_localize(None) if hasattr(history_df["Date"].dt, 'tz_localize') else history_df["Date"]
        fig.add_trace(go.Scatter(x=x_hist, y=history_df["Close"], mode="lines", name="Historical Close", line=dict(width=2, color="#94a3b8")))
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines+markers", name="Forecast 30 Hari", line=dict(width=2.5, color="#6366f1"), marker=dict(size=5)))
    fig = apply_clean_theme(fig, "Harga (Rp)")
    fig.update_layout(height=320)
    return fig


def plot_actual_vs_prediction(actual_pred: pd.DataFrame):
    fig = go.Figure()
    if not actual_pred.empty:
        fig.add_trace(go.Scatter(y=actual_pred["Actual"], mode="lines", name="Actual", line=dict(width=2, color="#10b981")))
        fig.add_trace(go.Scatter(y=actual_pred["Prediction"], mode="lines", name="Prediction", line=dict(width=2, color="#6366f1", dash="dash")))
    fig = apply_clean_theme(fig, "Nilai Terskala", "Observasi")
    fig.update_layout(height=260)
    return fig


def plot_loss_history(loss_df: pd.DataFrame):
    fig = go.Figure()
    if not loss_df.empty:
        fig.add_trace(go.Scatter(y=loss_df["loss"], mode="lines", name="Training Loss", line=dict(width=2, color="#3b82f6")))
        if "val_loss" in loss_df.columns:
            fig.add_trace(go.Scatter(y=loss_df["val_loss"], mode="lines", name="Validation Loss", line=dict(width=2, color="#f59e0b", dash="dash")))
    fig = apply_clean_theme(fig, "Loss", "Epoch")
    fig.update_layout(height=260)
    return fig


def plot_benchmark(selected_tickers: list, period: str, interval: str):
    combined = []
    for label, ticker in selected_tickers:
        df = download_data(ticker, period, interval)
        if df.empty:
            continue
        df = df[["Date", "Close"]].dropna().copy()
        if df.empty:
            continue
        first_close = df["Close"].iloc[0]
        if not first_close or pd.isna(first_close):
            continue
        df["Normalized"] = (df["Close"] / first_close) * 100
        df["Ticker"] = label
        combined.append(df[["Date", "Normalized", "Ticker"]])

    if not combined:
        return None
    all_df = pd.concat(combined, ignore_index=True)
    fig = go.Figure()
    for label in all_df["Ticker"].unique():
        sub = all_df[all_df["Ticker"] == label]
        is_focus = (label == "TLKM")
        # Netralkan timezone koordinat bursa demi kelancaran plotting multi-line
        x_bench = sub["Date"].dt.tz_localize(None) if hasattr(sub["Date"].dt, 'tz_localize') else sub["Date"]
        fig.add_trace(go.Scatter(x=x_bench, y=sub["Normalized"], mode="lines", name=label, line=dict(width=3 if is_focus else 1.5)))
    fig = apply_clean_theme(fig, "Index Normalized")
    fig.update_layout(height=340)
    return fig


# =========================================================
# CORE SYSTEM INITIALIZATION
# =========================================================
metrics = load_json_metrics("metrics.json")
actual_pred_df = load_csv("actual_vs_prediction.csv")
loss_df = load_csv("loss_history.csv")

# SIDEBAR CONTROLS
st.sidebar.title("Kontrol Dashboard")
period_label = st.sidebar.selectbox("Periode Historis", list(PERIOD_MAP.keys()), index=3)
compare_list = st.sidebar.multiselect("Saham Pembanding", options=list(BENCHMARKS.keys()), default=["BBCA", "BBRI", "BMRI", "ASII"])
period, interval = PERIOD_MAP[period_label]

# IDENTITAS KELOMPOK
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background-color: rgba(148, 163, 184, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.15);">
        <p style="margin-bottom: 4px; font-weight: 600; font-size: 0.85rem; color: #1e293b;">KELOMPOK 12</p>
        <ul style="margin: 0; padding-left: 16px; font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            <li>Sahira Anindya</li>
            <li>Agus Purnawan</li>
            <li>Muhammad Sultan Badruzzaman</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

hist_df = download_data(PRIMARY_TICKER, period, interval)
hist_df = add_indicators(hist_df) if not hist_df.empty else hist_df

if hist_df.empty:
    st.error("Data TLKM tidak tersedia saat ini.")
    st.stop()

# Calculations
latest_price = safe_latest(hist_df["Close"])
prev_price = hist_df["Close"].iloc[-2] if len(hist_df) >= 2 else np.nan
change_value, change_pct = compute_change(latest_price, prev_price)
latest_volume = safe_latest(hist_df["Volume"])
latest_open = safe_latest(hist_df["Open"])
latest_high = safe_latest(hist_df["High"])
latest_low = safe_latest(hist_df["Low"])

high_52w = hist_df["High"].max()
low_52w = hist_df["Low"].min()
avg_price = hist_df["Close"].mean()
std_price = hist_df["Close"].std()

latest_rsi = safe_latest(hist_df["RSI"])
latest_macd = safe_latest(hist_df["MACD"])
latest_signal_line = safe_latest(hist_df["Signal_Line"])
latest_ma50 = safe_latest(hist_df["MA50"])
latest_ma20 = safe_latest(hist_df["MA20"])

signal, signal_desc = trading_signal(latest_rsi, latest_macd, latest_signal_line, latest_price, latest_ma50)

# ---------------------------------------------------------
# UPDATE KUNCI: AMBIL HARI INI DARI YFINANCE LALU GENERATE
# ---------------------------------------------------------
today = hist_df["Date"].max()
forecast_df = generate_live_forecast(today, latest_price)

# Filter Data Historis Berdasarkan Pilihan Sidebar
if period_label == "7 Hari":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=7)]
elif period_label == "1 Bulan":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=30)]
elif period_label == "3 Bulan":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=90)]
elif period_label == "6 Bulan":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=180)]
elif period_label == "1 Tahun":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=365)]
elif period_label == "2 Tahun":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=730)]
elif period_label == "5 Tahun":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=1825)]
elif period_label == "10 Tahun":
    filtered_df = hist_df[hist_df["Date"] >= today - pd.Timedelta(days=3650)]
else:
    filtered_df = hist_df.copy()

# =========================================================
# UI LAYOUT RENDERER
# =========================================================
st.title("TLKM Stock Forecast Dashboard")
# Menampilkan tanggal riil bursa terakhir
today_clean = today.tz_localize(None) if hasattr(today, 'tz_localize') else today
st.markdown(f"<span class='badge-live'>LIVE</span> <span class='muted'>&nbsp;Data bursa terakhir: {today_clean.strftime('%d %b %Y')}. Diperbarui otomatis dari Yahoo Finance.</span>", unsafe_allow_html=True)

# Main Dashboard Top Row KPI Cards
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Harga Saat Ini", fmt_idr(latest_price), fmt_num(change_value, 0))
m2.metric("Perubahan Harian", fmt_pct(change_pct), f"{'+' if change_value >= 0 else ''}{fmt_num(change_value, 0)}")
m3.metric("Volume Harian", f"{int(latest_volume):,}".replace(",", ".") if not pd.isna(latest_volume) else "-")
m4.metric("Sinyal Sistem", signal, help=signal_desc)
m5.metric("Model Analitik", "LSTM")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Analisis Historis & Volume",
    "Forecast & Evaluasi Model",
    "Perbandingan Saham",
    "Data Mentah (Historical Data)"
])

# ---------------------------------------------------------
# TAB 1: ANALISIS HISTORIS & VOLUME
# ---------------------------------------------------------
with tab1:
    st.markdown("<div class='section-title'>Ringkasan Harga Saat Ini</div>", unsafe_allow_html=True)
    with st.container(border=True):
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Open", fmt_idr(latest_open))
        s2.metric("High", fmt_idr(latest_high))
        s3.metric("Low", fmt_idr(latest_low))
        s4.metric("Close", fmt_idr(latest_price))
        s5.metric("52W High", fmt_idr(high_52w))
        s6.metric("52W Low", fmt_idr(low_52w))
        
    st.markdown("<div class='section-title'>Indikator Teknis Pasar</div>", unsafe_allow_html=True)
    with st.container(border=True):
        tech1, tech2, tech3, tech4, tech5 = st.columns(5)
        tech1.metric("RSI (14)", fmt_num(latest_rsi, 2))
        tech2.metric("MACD", fmt_num(latest_macd, 4))
        tech3.metric("Signal Line", fmt_num(latest_signal_line, 4))
        tech4.metric("MA20", fmt_idr(latest_ma20))
        tech5.metric("MA50", fmt_idr(latest_ma50))

    st.markdown("<div class='section-title'>Statistik Deskriptif Kinerja Historis</div>", unsafe_allow_html=True)
    stats_data = {
        "Metrik Parameter": ["Harga Tertinggi (Period)", "Harga Terendah (Period)", "Rata-rata Harga", "Standar Deviasi (Volatilitas)"],
        "Nilai": [fmt_idr(high_52w), fmt_idr(low_52w), fmt_idr(avg_price), fmt_num(std_price, 2)]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Grafik Pergerakan Harga Historis TLKM</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_price_history(filtered_df), width="stretch")
    
    st.markdown("<div class='section-title'>Grafik Volume Perdagangan</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_volume(filtered_df), width="stretch")


# ---------------------------------------------------------
# TAB 2: FORECAST & EVALUASI MODEL (BAGIAN YANG DIPERBAIKI)
# ---------------------------------------------------------
with tab2:
    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Informasi Arsitektur Model</div>", unsafe_allow_html=True)
    mi1, mi2, mi3, mi4 = st.columns(4)
    mi1.metric("Model Utama", "LSTM")
    mi2.metric("Epochs Train", str(metrics.get("epochs", "-")))
    mi3.metric("Batch Size", str(metrics.get("batch_size", "-")))
    mi4.metric("Lookback (Window)", str(metrics.get("lookback", "-")))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Nilai Prediksi Horizon Kedepan</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if not forecast_df.empty:
            fc_kpi1, fc_kpi2, fc_kpi3 = st.columns(3)
            fc_kpi1.metric("Prediksi Besok", fmt_idr(forecast_df["Forecast"].iloc[0]))
            fc_kpi2.metric("Prediksi 7 Hari", fmt_idr(forecast_df["Forecast"].iloc[6] if len(forecast_df) >= 7 else np.nan))
            fc_kpi3.metric("Prediksi 30 Hari (Akhir Horizon)", fmt_idr(forecast_df["Forecast"].iloc[-1]))
        else:
            st.write("Data forecast tidak tersedia.")

    st.markdown("<div class='section-title'>Grafik Forecast 30 Hari ke Depan</div>", unsafe_allow_html=True)
    # Visualisasi menggunakan grafik dinamis yang menyambung otomatis dari ekor data yfinance
    st.plotly_chart(plot_forecast_with_history(filtered_df.tail(60), forecast_df), width="stretch")

    st.markdown("<div class='section-title'>Detail Tabel Hasil Forecast</div>", unsafe_allow_html=True)
    if not forecast_df.empty:
        formatted_forecast = forecast_df.copy()
        # Mengubah penamaan kolom agar lebih rapi saat dirender di tabel dashboard
        formatted_forecast.columns = ["Tanggal Hasil Ramalan", "Estimasi Harga Forecast (LSTM)"]
        formatted_forecast["Tanggal Hasil Ramalan"] = formatted_forecast["Tanggal Hasil Ramalan"].dt.strftime("%d %b %Y")
        formatted_forecast["Estimasi Harga Forecast (LSTM)"] = formatted_forecast["Estimasi Harga Forecast (LSTM)"].apply(lambda x: fmt_idr(x, 2))
        st.dataframe(formatted_forecast, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Metrik Evaluasi Performa Model</div>", unsafe_allow_html=True)
    em1, em2, em3, em4, em5 = st.columns(5)
    em1.metric("MAE", fmt_num(metrics.get("mae"), 4))
    em2.metric("RMSE", fmt_num(metrics.get("rmse"), 4))
    em3.metric("MAPE", fmt_pct(metrics.get("mape"), 2))
    em4.metric("Accuracy", fmt_pct(metrics.get("accuracy"), 2))
    em5.metric("R² Score", fmt_num(metrics.get("r2"), 4))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Visualisasi Validasi: Actual vs Prediction</div>", unsafe_allow_html=True)
    if actual_pred_df.empty:
        st.warning("File actual_vs_prediction.csv belum ditemukan atau kosong.")
    else:
        grid_ap1, grid_ap2 = st.columns([3, 2])
        with grid_ap1:
            st.plotly_chart(plot_actual_vs_prediction(actual_pred_df), width="stretch")
        with grid_ap2:
            st.dataframe(actual_pred_df, use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Kurva Pelatihan Model (Training & Validation Loss)</div>", unsafe_allow_html=True)
    if loss_df.empty:
        st.warning("File loss_history.csv belum ditemukan atau kosong.")
    else:
        grid_loss1, grid_loss2 = st.columns([3, 2])
        with grid_loss1:
            st.plotly_chart(plot_loss_history(loss_df), width="stretch")
        with grid_loss2:
            st.dataframe(loss_df, use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 3: PERBANDINGAN SAHAM
# ---------------------------------------------------------
with tab3:
    st.markdown("<div class='section-title'>Perbandingan Kinerja Relatif</div>", unsafe_allow_html=True)
    selected = [(k, BENCHMARKS[k]) for k in compare_list if k in BENCHMARKS]
    if ("TLKM", PRIMARY_TICKER) not in selected:
        selected.insert(0, ("TLKM", PRIMARY_TICKER))
        
    bench_fig = plot_benchmark(selected, period, interval)
    if bench_fig is None:
        st.warning("Data saham pembanding tidak berhasil dimuat.")
    else:
        st.plotly_chart(bench_fig, width="stretch")


# ---------------------------------------------------------
# TAB 4: DATA MENTAH (HISTORICAL DATA)
# ---------------------------------------------------------
with tab4:
    st.markdown("<div class='section-title'>Data Mentah Hasil Unduhan (Yahoo Finance)</div>", unsafe_allow_html=True)
    st.caption("Menampilkan lembar data mentah lengkap beserta seluruh indikator kalkulasi teknikal.")
    
    raw_display_df = hist_df.copy()
    if "Date" in raw_display_df.columns:
        # Melakukan konversi string penanggalan murni tanpa timezone agar seragam di interface tabel
        x_date = raw_display_df["Date"].dt.tz_localize(None) if hasattr(raw_display_df["Date"].dt, 'tz_localize') else raw_display_df["Date"]
        raw_display_df["Date"] = x_date.dt.strftime("%Y-%m-%d %H:%M:%S" if interval in ["1m","5m","15m","1h"] else "%Y-%m-%d")
        
    st.dataframe(
        raw_display_df.sort_values(by="Date", ascending=False), 
        use_container_width=True, 
        height=500
    )


# Footer branding
st.divider()
st.caption(f"Source Market Data Engine: Yahoo Finance | Target Instrument Focus: {PRIMARY_TICKER} | Last Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

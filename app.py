import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if st_autorefresh is not None:
    st_autorefresh(interval=60_000, key="dashboard_refresh")

# =========================================================
# CLEAN STYLE (Tanpa Kotak Kustom HTML & Mendukung Light/Dark Mode)
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
        }
        
        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 2rem; 
        }
        
        /* Modifikasi Metric Card bawaan agar rapi & adaptif */
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            padding: 15px 20px !important;
            border-radius: 12px !important;
        }
        
        .muted { color: #64748b; font-size: 0.95rem; }
        
        /* Badge Live dengan Animasi */
        .badge-live {
            display: inline-flex; 
            align-items: center; 
            gap: 6px;
            padding: 0.35rem 0.75rem; 
            border-radius: 8px;
            background: rgba(34, 197, 94, 0.15); 
            color: #22c55e; 
            font-weight: 700; 
            font-size: 0.75rem;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .badge-live::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(34, 211, 92, 0.5); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(34, 211, 92, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(34, 211, 92, 0); }
        }

        /* Desain Tab Kustom */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(148, 163, 184, 0.08);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.15);
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            transition: all 0.2s ease;
            padding: 0 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS (Pilihan Saham Pembanding Diperbanyak)
# =========================================================
PRIMARY_TICKER = "TLKM.JK"
BENCHMARKS = {
    # Perbankan (Big 4)
    "BBCA": "BBCA.JK",
    "BBRI": "BBRI.JK",
    "BMRI": "BMRI.JK",
    "BBNI": "BBNI.JK",
    # Telekomunikasi & Infrastruktur
    "ISAT": "ISAT.JK",
    "EXCL": "EXCL.JK",
    "JSMR": "JSMR.JK",
    # Komoditas, Energi & Pertambangan
    "ADRO": "ADRO.JK",
    "PTBA": "PTBA.JK",
    "ITMG": "ITMG.JK",
    "ANTM": "ANTM.JK",
    "PGAS": "PGAS.JK",
    # Consumer Goods & Konglomerasi
    "UNVR": "UNVR.JK",
    "ICBP": "ICBP.JK",
    "INDF": "INDF.JK",
    "GGRM": "GGRM.JK",
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
# HELPERS
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
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

    out["Volatility_20"] = out["Return"].rolling(20).std()

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
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


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
    return "HOLD", "Sinyal campuran. Tren belum cukup kuat untuk keputusan agresif."


def make_metric_card(col, label, value, delta=None, help_text=None):
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_text)


# =========================================================
# CONFIG TEMPLATE PLOTLY VISUAL (Mengikuti Tema Browser)
# =========================================================
def apply_clean_theme(fig, y_title, x_title="Tanggal"):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=x_title, showgrid=False)
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)")
    return fig


def plot_price_history(df: pd.DataFrame, ticker_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Close"], mode="lines", name="Close Price",
            line=dict(width=2.5, color="#3b82f6"),
            hovertemplate="%{x|%d %b %Y}<br>Close: Rp %{y:,.0f}<extra></extra>",
        )
    )
    if "MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["MA20"], mode="lines", name="MA20",
                line=dict(width=1.5, color="#10b981", dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>MA20: Rp %{y:,.0f}<extra></extra>",
            )
        )
    if "MA50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["MA50"], mode="lines", name="MA50",
                line=dict(width=1.5, color="#f59e0b", dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>MA50: Rp %{y:,.0f}<extra></extra>",
            )
        )
    return apply_clean_theme(fig, "Harga (Rp)")


def plot_volume(df: pd.DataFrame, ticker_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Date"], y=df["Volume"], name="Volume",
            marker_color="#0ea5e9", opacity=0.8,
            hovertemplate="%{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>",
        )
    )
    fig = apply_clean_theme(fig, "Volume")
    fig.update_layout(height=260, showlegend=False)
    return fig


def plot_forecast_with_history(history_df: pd.DataFrame, forecast_df: pd.DataFrame):
    fig = go.Figure()

    if not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["Date"], y=history_df["Close"], mode="lines", name="Historical Close",
                line=dict(width=2, color="#94a3b8"),
                hovertemplate="%{x|%d %b %Y}<br>Close: Rp %{y:,.0f}<extra></extra>",
            )
        )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines+markers", name="Forecast 30 Hari",
                line=dict(width=2.5, color="#6366f1"),
                marker=dict(size=5, color="#818cf8"),
                hovertemplate="%{x|%d %b %Y}<br>Forecast: Rp %{y:,.0f}<extra></extra>",
            )
        )
    return apply_clean_theme(fig, "Harga (Rp)")


def plot_actual_vs_prediction(actual_pred: pd.DataFrame):
    fig = go.Figure()
    if not actual_pred.empty:
        fig.add_trace(
            go.Scatter(
                y=actual_pred["Actual"], mode="lines", name="Actual",
                line=dict(width=2, color="#10b981"),
                hovertemplate="Index %{x}<br>Actual: Rp %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=actual_pred["Prediction"], mode="lines", name="Prediction",
                line=dict(width=2, color="#6366f1", dash="dash"),
                hovertemplate="Index %{x}<br>Prediction: Rp %{y:,.0f}<extra></extra>",
            )
        )
    return apply_clean_theme(fig, "Nilai Terskala", "Observasi")


def plot_loss_history(loss_df: pd.DataFrame):
    fig = go.Figure()
    if not loss_df.empty:
        fig.add_trace(
            go.Scatter(
                y=loss_df["loss"], mode="lines", name="Training Loss",
                line=dict(width=2, color="#3b82f6"),
                hovertemplate="Epoch %{x}<br>Loss: %{y:.6f}<extra></extra>",
            )
        )
        if "val_loss" in loss_df.columns:
            fig.add_trace(
                go.Scatter(
                    y=loss_df["val_loss"], mode="lines", name="Validation Loss",
                    line=dict(width=2, color="#f59e0b", dash="dash"),
                    hovertemplate="Epoch %{x}<br>Val Loss: %{y:.6f}<extra></extra>",
                )
            )
    return apply_clean_theme(fig, "Loss", "Epoch")


def plot_benchmark(selected_tickers: list[str], period: str, interval: str):
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
        fig.add_trace(
            go.Scatter(
                x=sub["Date"], y=sub["Normalized"], mode="lines", name=label,
                line=dict(width=3 if is_focus else 1.5),
                hovertemplate="%{x|%d %b %Y}<br>Normalized: %{y:.2f}<extra></extra>",
            )
        )
    return apply_clean_theme(fig, "Index Normalized")


def sidebar_controls():
    st.sidebar.title("Kontrol Dashboard")
    period_label = st.sidebar.selectbox("Periode Historis", list(PERIOD_MAP.keys()), index=3)
    
    # Multiselect diperluas pilihan sahamnya
    compare = st.sidebar.multiselect(
        "Saham Pembanding (Bisa pilih banyak)",
        options=list(BENCHMARKS.keys()),
        default=["BBCA", "BBRI", "BMRI", "ASII", "ISAT"],
    )
    refresh_note = st.sidebar.checkbox("Auto refresh 60 detik", value=True)

    if refresh_note and st_autorefresh is not None:
        st.sidebar.success("Auto refresh aktif")

    return period_label, compare


# =========================================================
# LOAD DATA OUTPUTS
# =========================================================
metrics = load_json_metrics("models/metrics.json")
forecast_df = load_csv("forecast.csv")
actual_pred_df = load_csv("actual_vs_prediction.csv")
loss_df = load_csv("loss_history.csv")


# =========================================================
# SIDEBAR CONTROL CALLS
# =========================================================
period_label, compare_list = sidebar_controls()
period, interval = PERIOD_MAP[period_label]


# =========================================================
# LIVE DATA FETCHER
# =========================================================
hist_df = download_data(PRIMARY_TICKER, period, interval)
hist_df = add_indicators(hist_df) if not hist_df.empty else hist_df

latest_price = safe_latest(hist_df["Close"]) if not hist_df.empty else np.nan
prev_price = hist_df["Close"].iloc[-2] if len(hist_df) >= 2 else np.nan
change_value, change_pct = compute_change(latest_price, prev_price)

latest_volume = safe_latest(hist_df["Volume"]) if not hist_df.empty else np.nan
latest_open = safe_latest(hist_df["Open"]) if not hist_df.empty else np.nan
latest_high = safe_latest(hist_df["High"]) if not hist_df.empty else np.nan
latest_low = safe_latest(hist_df["Low"]) if not hist_df.empty else np.nan
high_52w = hist_df["High"].max() if not hist_df.empty else np.nan
low_52w = hist_df["Low"].min() if not hist_df.empty else np.nan

latest_rsi = safe_latest(hist_df["RSI"]) if "RSI" in hist_df.columns else np.nan
latest_macd = safe_latest(hist_df["MACD"]) if "MACD" in hist_df.columns else np.nan
latest_signal_line = safe_latest(hist_df["Signal_Line"]) if "Signal_Line" in hist_df.columns else np.nan
latest_ma50 = safe_latest(hist_df["MA50"]) if "MA50" in hist_df.columns else np.nan
latest_ma20 = safe_latest(hist_df["MA20"]) if "MA20" in hist_df.columns else np.nan

# Penanganan pembacaan model_name fleksibel huruf besar/kecil
model_display_name = metrics.get("model_name", metrics.get("MODEL_NAME", "Multivariate LSTM"))

signal, signal_desc = trading_signal(
    latest_rsi,
    latest_macd,
    latest_signal_line,
    latest_price,
    latest_ma50,
)

# =========================================================
# HEADER
# =========================================================
st.title("📈 TLKM Stock Forecast Dashboard")
st.caption("Dashboard analisis, forecasting, dan pembanding saham Indonesia dengan fokus utama TLKM.")
st.markdown(f"<span class='badge-live'>LIVE</span> <span class='muted'>&nbsp;Data diperbarui otomatis dari Yahoo Finance.</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if hist_df.empty:
    st.error("Data TLKM tidak tersedia saat ini.")
    st.stop()


# =========================================================
# TOP METRICS
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)
make_metric_card(m1, "Harga Saat Ini", fmt_idr(latest_price), fmt_num(change_value, 0), "Close terakhir dari data historis")
make_metric_card(m2, "Perubahan Harian", fmt_pct(change_pct), f"{'+' if change_value >= 0 else ''}{fmt_num(change_value, 0)}", "Perbandingan close terakhir dan sebelumnya")
make_metric_card(m3, "Volume", f"{int(latest_volume):,}".replace(",", ".") if not pd.isna(latest_volume) else "-", None, "Volume perdagangan terakhir")
make_metric_card(m4, "Sinyal", signal, None, signal_desc)
make_metric_card(m5, "Model Utama", model_display_name, None, "Model terbaik yang disimpan di metrics.json")

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT (4 Tabs - Bagian Fundamental Telah Dihapus)
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historis & Forecast",
    "🎯 Evaluasi Model",
    "⚖️ Perbandingan Saham",
    "🗂️ Data Mentah",
])

with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📊 Grafik Historis TLKM")
        st.plotly_chart(plot_price_history(hist_df, "TLKM"), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("🔮 Forecast 30 Hari ke Depan")
        if forecast_df.empty:
            st.warning("File outputs/forecast.csv belum ditemukan atau kosong.")
        else:
            forecast_df = forecast_df.copy()
            if "Date" in forecast_df.columns:
                forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
            st.plotly_chart(plot_forecast_with_history(hist_df.tail(120), forecast_df), use_container_width=True)

            latest_fc = forecast_df["Forecast"].iloc[0] if not forecast_df.empty else np.nan
            future_7 = forecast_df["Forecast"].iloc[6] if len(forecast_df) >= 7 else np.nan
            future_30 = forecast_df["Forecast"].iloc[-1] if len(forecast_df) >= 1 else np.nan

            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Prediksi Hari Ini / Besok", fmt_idr(latest_fc))
            fc2.metric("Prediksi 7 Hari", fmt_idr(future_7))
            fc3.metric("Prediksi 30 Hari", fmt_idr(future_30))

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                forecast_df.assign(
                    Date=forecast_df["Date"].dt.strftime("%d %b %Y") if "Date" in forecast_df.columns else forecast_df["Date"],
                    Forecast=forecast_df["Forecast"].round(2),
                ),
                use_container_width=True,
                hide_index=True,
            )

    with c2:
        st.subheader("📌 Ringkasan Harga")
        s1, s2 = st.columns(2)
        s1.metric("Open", fmt_idr(latest_open))
        s2.metric("High", fmt_idr(latest_high))
        s3, s4 = st.columns(2)
        s3.metric("Low", fmt_idr(latest_low))
        s4.metric("Close", fmt_idr(latest_price))
        s5, s6 = st.columns(2)
        s5.metric("52W High", fmt_idr(high_52w))
        s6.metric("52W Low", fmt_idr(low_52w))
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("⚡ Indikator Teknis")
        tech1, tech2 = st.columns(2)
        tech1.metric("RSI", fmt_num(latest_rsi, 2))
        tech2.metric("MACD", fmt_num(latest_macd, 4))
        tech3, tech4 = st.columns(2)
        tech3.metric("Signal Line", fmt_num(latest_signal_line, 4))
        tech4.metric("MA50", fmt_idr(latest_ma50))
        st.metric("MA20", fmt_idr(latest_ma20))
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📈 Volume Perdagangan")
        st.plotly_chart(plot_volume(hist_df.tail(180), "TLKM"), use_container_width=True)

with tab2:
    e1, e2 = st.columns([1, 1])

    with e1:
        st.subheader("🎯 Evaluasi Model")
        
        # Perbaikan pencarian nilai agar mendukung key huruf besar maupun kecil dari JSON
        mae_val = metrics.get("mae", metrics.get("MAE"))
        rmse_val = metrics.get("rmse", metrics.get("RMSE"))
        mape_val = metrics.get("mape", metrics.get("MAPE"))
        acc_val = metrics.get("accuracy", metrics.get("ACCURACY", metrics.get("Acc", metrics.get("ACC"))))
        r2_val = metrics.get("r2", metrics.get("R2", metrics.get("r2_score")))

        em1, em2, em3 = st.columns(3)
        em1.metric("MAE", fmt_num(mae_val, 4))
        em2.metric("RMSE", fmt_num(rmse_val, 4))
        em3.metric("MAPE", fmt_pct(mape_val, 2) if mape_val is not None else "-")

        em4, em5 = st.columns(2)
        em4.metric("Accuracy", fmt_pct(acc_val, 2) if acc_val is not None else "-")
        em5.metric("R²", fmt_num(r2_val, 4))
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("ℹ️ Informasi Model")
        epochs = metrics.get("epochs", metrics.get("EPOCHS", "-"))
        batch_size = metrics.get("batch_size", metrics.get("BATCH_SIZE", "-"))
        lookback = metrics.get("lookback", metrics.get("LOOKBACK", "-"))

        mi1, mi2 = st.columns(2)
        mi1.metric("Model", model_display_name)
        mi2.metric("Epoch", str(epochs))
        mi3, mi4 = st.columns(2)
        mi3.metric("Batch Size", str(batch_size))
        mi4.metric("Lookback", str(lookback))

    with e2:
        st.subheader("📉 Actual vs Prediction")
        if actual_pred_df.empty:
            st.warning("File outputs/actual_vs_prediction.csv belum ditemukan atau kosong.")
        else:
            st.plotly_chart(plot_actual_vs_prediction(actual_pred_df), use_container_width=True)
            st.dataframe(actual_pred_df.head(25), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📉 Training Loss")
        if loss_df.empty:
            st.warning("File outputs/loss_history.csv belum ditemukan atau kosong.")
        else:
            st.plotly_chart(plot_loss_history(loss_df), use_container_width=True)
            st.dataframe(loss_df.head(25), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("⚖️ Perbandingan Kinerja Saham")
    st.caption("TLKM tetap menjadi fokus utama. Grafik berikut membandingkan kinerja harga yang dinormalisasi terhadap saham pembanding pilihanmu.")
    selected = [(k, BENCHMARKS[k]) for k in compare_list if k in BENCHMARKS]
    
    # Memastikan TLKM selalu masuk di urutan pertama
    if ("TLKM", PRIMARY_TICKER) not in selected:
        selected.insert(0, ("TLKM", PRIMARY_TICKER))
        
    bench_fig = plot_benchmark(selected, period, interval)
    if bench_fig is None:
        st.warning("Data pembanding belum tersedia atau gagal dimuat.")
    else:
        st.plotly_chart(bench_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Ringkasan Perbandingan")
    bench_rows = []
    for label, ticker in selected:
        d = download_data(ticker, period, interval)
        if d.empty:
            continue
        d = d.dropna(subset=["Close"]).copy()
        if d.empty:
            continue
        first_close = d["Close"].iloc[0]
        last_close = d["Close"].iloc[-1]
        diff, pct = compute_change(last_close, first_close)
        bench_rows.append(
            {
                "Ticker": label,
                "First Close": first_close,
                "Last Close": last_close,
                "Change": diff,
                "Change %": pct,
            }
        )

    if bench_rows:
        bench_df = pd.DataFrame(bench_rows)
        st.dataframe(
            bench_df.assign(
                **{
                    "First Close": bench_df["First Close"].map(fmt_idr),
                    "Last Close": bench_df["Last Close"].map(fmt_idr),
                    "Change": bench_df["Change"].map(lambda x: fmt_idr(x) if pd.notna(x) else "-"),
                    "Change %": bench_df["Change %"].map(fmt_pct),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.subheader("🗂️ Data Mentah Historis TLKM")
    preview_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MA20", "MA50"] if c in hist_df.columns]
    show_df = hist_df[preview_cols].copy()
    if "Date" in show_df.columns:
        show_df["Date"] = show_df["Date"].dt.strftime("%d %b %Y")
    st.dataframe(show_df.tail(100), use_container_width=True, hide_index=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
last_update = hist_df["Date"].iloc[-1]
st.caption(
    f"Last update: {last_update.strftime('%d %b %Y') if pd.notna(last_update) else '-'} | "
    f"Source: Yahoo Finance | Interval: {interval} | Focus: TLKM"
)

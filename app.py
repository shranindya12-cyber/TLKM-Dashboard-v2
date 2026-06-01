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
# STYLE
# =========================================================
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
        .stMetric { background: rgba(15, 23, 42, 0.35); border: 1px solid rgba(148, 163, 184, 0.15); padding: 14px; border-radius: 16px; }
        .card-soft {
            background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(2,6,23,0.92));
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        }
        .section-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.4rem; }
        .muted { color: #94a3b8; font-size: 0.92rem; }
        .small { font-size: 0.85rem; }
        .badge-live {
            display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
            background: rgba(34,197,94,0.15); color: #22c55e; font-weight: 700; font-size: 0.75rem;
            border: 1px solid rgba(34,197,94,0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================
PRIMARY_TICKER = "TLKM.JK"
BENCHMARKS = {
    "BBCA": "BBCA.JK",
    "BBRI": "BBRI.JK",
    "BMRI": "BMRI.JK",
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


def plot_price_history(df: pd.DataFrame, ticker_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2),
            hovertemplate="%{x|%d %b %Y}<br>Close: Rp %{y:,.0f}<extra></extra>",
        )
    )
    if "MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA20"],
                mode="lines",
                name="MA20",
                line=dict(width=2, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>MA20: Rp %{y:,.0f}<extra></extra>",
            )
        )
    if "MA50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA50"],
                mode="lines",
                name="MA50",
                line=dict(width=2, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>MA50: Rp %{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Pergerakan Harga {ticker_label}",
        height=520,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Tanggal", showgrid=False)
    fig.update_yaxes(title_text="Harga (Rp)", tickformat=",.0f")
    return fig


def plot_volume(df: pd.DataFrame, ticker_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            hovertemplate="%{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Volume Perdagangan {ticker_label}",
        height=380,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Tanggal", showgrid=False)
    fig.update_yaxes(title_text="Volume", tickformat=",.0f")
    return fig


def plot_forecast_with_history(history_df: pd.DataFrame, forecast_df: pd.DataFrame):
    fig = go.Figure()

    if not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["Date"],
                y=history_df["Close"],
                mode="lines",
                name="Historical Close",
                line=dict(width=2),
                hovertemplate="%{x|%d %b %Y}<br>Close: Rp %{y:,.0f}<extra></extra>",
            )
        )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines+markers",
                name="Forecast 30 Hari",
                line=dict(width=3, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>Forecast: Rp %{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Forecast 30 Hari TLKM",
        height=520,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Tanggal")
    fig.update_yaxes(title_text="Harga (Rp)", tickformat=",.0f")
    return fig


def plot_actual_vs_prediction(actual_pred: pd.DataFrame):
    fig = go.Figure()
    if not actual_pred.empty:
        fig.add_trace(
            go.Scatter(
                y=actual_pred["Actual"],
                mode="lines",
                name="Actual",
                line=dict(width=2),
                hovertemplate="Index %{x}<br>Actual: Rp %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=actual_pred["Prediction"],
                mode="lines",
                name="Prediction",
                line=dict(width=2, dash="dash"),
                hovertemplate="Index %{x}<br>Prediction: Rp %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Actual vs Prediction pada Data Test",
        height=500,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Observasi")
    fig.update_yaxes(title_text="Nilai Terskala / Inverse sesuai output model")
    return fig


def plot_loss_history(loss_df: pd.DataFrame):
    fig = go.Figure()
    if not loss_df.empty:
        fig.add_trace(
            go.Scatter(
                y=loss_df["loss"],
                mode="lines",
                name="Training Loss",
                line=dict(width=2),
                hovertemplate="Epoch %{x}<br>Loss: %{y:.6f}<extra></extra>",
            )
        )
        if "val_loss" in loss_df.columns:
            fig.add_trace(
                go.Scatter(
                    y=loss_df["val_loss"],
                    mode="lines",
                    name="Validation Loss",
                    line=dict(width=2, dash="dash"),
                    hovertemplate="Epoch %{x}<br>Val Loss: %{y:.6f}<extra></extra>",
                )
            )
    fig.update_layout(
        title="Kurva Training dan Validation Loss",
        height=500,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    return fig


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
        fig.add_trace(
            go.Scatter(
                x=sub["Date"],
                y=sub["Normalized"],
                mode="lines",
                name=label,
                hovertemplate="%{x|%d %b %Y}<br>Normalized: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Perbandingan Kinerja Terkumpul (Basis 100)",
        height=500,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Tanggal")
    fig.update_yaxes(title_text="Index Normalized")
    return fig


def show_fundamentals(ticker: str):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    fields = {
        "Long Name": info.get("longName"),
        "Market Cap": info.get("marketCap"),
        "PE Ratio": info.get("trailingPE"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
        "52W High": info.get("fiftyTwoWeekHigh"),
        "52W Low": info.get("fiftyTwoWeekLow"),
        "Currency": info.get("currency"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
    }

    c1, c2 = st.columns(2)
    items = list(fields.items())
    for idx, (label, value) in enumerate(items):
        col = c1 if idx % 2 == 0 else c2
        with col:
            if label == "Market Cap" and value is not None:
                display_value = f"{value:,.0f}".replace(",", ".")
            elif label == "Dividend Yield" and value is not None:
                display_value = f"{value*100:.2f}%"
            elif isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = "-" if value is None else str(value)
            st.metric(label, display_value)


def sidebar_controls():
    st.sidebar.title("Kontrol Dashboard")
    period_label = st.sidebar.selectbox("Periode Historis", list(PERIOD_MAP.keys()), index=3)
    compare = st.sidebar.multiselect(
        "Saham pembanding",
        options=list(BENCHMARKS.keys()),
        default=["BBCA", "BBRI", "BMRI", "ASII"],
    )
    refresh_note = st.sidebar.checkbox("Auto refresh 60 detik", value=True)

    if refresh_note and st_autorefresh is not None:
        st.sidebar.success("Auto refresh aktif")

    return period_label, compare


# =========================================================
# LOAD OUTPUTS
# =========================================================
metrics = load_json_metrics("models/metrics.json")
forecast_df = load_csv("forecast.csv")
actual_pred_df = load_csv("actual_vs_prediction.csv")
loss_df = load_csv("loss_history.csv")


# =========================================================
# SIDEBAR
# =========================================================
period_label, compare_list = sidebar_controls()
period, interval = PERIOD_MAP[period_label]


# =========================================================
# DATA
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
st.title("TLKM Stock Forecast Dashboard")
st.caption("Dashboard analisis, forecasting, dan pembanding saham Indonesia dengan fokus utama TLKM.")
st.markdown(f"<span class='badge-live'>LIVE</span> <span class='muted'>&nbsp;Data diperbarui otomatis dan ditarik dari Yahoo Finance.</span>", unsafe_allow_html=True)

if hist_df.empty:
    st.error("Data TLKM tidak tersedia saat ini.")
    st.stop()


# =========================================================
# TOP METRICS
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)
make_metric_card(m1, "Harga Saat Ini", fmt_idr(latest_price), fmt_num(change_value, 0), "Close terakhir dari data historis")
make_metric_card(m2, "Perubahan Harian", fmt_pct(change_pct), None, "Perbandingan close terakhir dan sebelumnya")
make_metric_card(m3, "Volume", f"{int(latest_volume):,}".replace(",", ".") if not pd.isna(latest_volume) else "-", None, "Volume perdagangan terakhir")
make_metric_card(m4, "Sinyal", signal, None, signal_desc)
make_metric_card(m5, "Model", metrics.get("model_name", "Multivariate LSTM"), None, "Model terbaik yang disimpan di metrics.json")

st.divider()

# =========================================================
# MAIN LAYOUT
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Historis & Forecast",
    "Evaluasi Model",
    "Fundamental TLKM",
    "Perbandingan Saham",
    "Data Mentah",
])

with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Grafik Historis TLKM</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_price_history(hist_df, "TLKM"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card-soft' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Forecast 30 Hari ke Depan</div>", unsafe_allow_html=True)
        if forecast_df.empty:
            st.warning("File forecast.csv belum ditemukan atau kosong.")
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

            st.dataframe(
                forecast_df.assign(
                    Date=forecast_df["Date"].dt.strftime("%d %b %Y") if "Date" in forecast_df.columns else forecast_df["Date"],
                    Forecast=forecast_df["Forecast"].round(2),
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ringkasan Harga</div>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.metric("Open", fmt_idr(latest_open))
        s2.metric("High", fmt_idr(latest_high))
        s3, s4 = st.columns(2)
        s3.metric("Low", fmt_idr(latest_low))
        s4.metric("Close", fmt_idr(latest_price))
        s5, s6 = st.columns(2)
        s5.metric("52W High", fmt_idr(high_52w))
        s6.metric("52W Low", fmt_idr(low_52w))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card-soft' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Indikator Teknis</div>", unsafe_allow_html=True)
        tech1, tech2 = st.columns(2)
        tech1.metric("RSI", fmt_num(latest_rsi, 2))
        tech2.metric("MACD", fmt_num(latest_macd, 4))
        tech3, tech4 = st.columns(2)
        tech3.metric("Signal Line", fmt_num(latest_signal_line, 4))
        tech4.metric("MA50", fmt_idr(latest_ma50))
        st.metric("MA20", fmt_idr(latest_ma20))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card-soft' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Volume Perdagangan</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_volume(hist_df.tail(180), "TLKM"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    e1, e2 = st.columns([1, 1])

    with e1:
        st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Evaluasi Model</div>", unsafe_allow_html=True)

        em1, em2, em3 = st.columns(3)
        em1.metric("MAE", fmt_num(metrics.get("mae"), 4))
        em2.metric("RMSE", fmt_num(metrics.get("rmse"), 4))
        em3.metric("MAPE", fmt_pct(metrics.get("mape"), 2))

        em4, em5 = st.columns(2)
        em4.metric("Accuracy", fmt_pct(metrics.get("accuracy"), 2))
        em5.metric("R²", fmt_num(metrics.get("r2"), 4))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card-soft' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Informasi Model</div>", unsafe_allow_html=True)
        model_name = metrics.get("model_name", "Multivariate LSTM")
        epochs = metrics.get("epochs", "-")
        batch_size = metrics.get("batch_size", "-")
        lookback = metrics.get("lookback", "-")

        mi1, mi2 = st.columns(2)
        mi1.metric("Model", model_name)
        mi2.metric("Epoch", str(epochs))
        mi3, mi4 = st.columns(2)
        mi3.metric("Batch Size", str(batch_size))
        mi4.metric("Lookback", str(lookback))
        st.markdown("</div>", unsafe_allow_html=True)

    with e2:
        st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Actual vs Prediction</div>", unsafe_allow_html=True)
        if actual_pred_df.empty:
            st.warning("File actual_vs_prediction.csv belum ditemukan atau kosong.")
        else:
            st.plotly_chart(plot_actual_vs_prediction(actual_pred_df), use_container_width=True)
            st.dataframe(actual_pred_df.head(25), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card-soft' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Training Loss</div>", unsafe_allow_html=True)
        if loss_df.empty:
            st.warning("File loss_history.csv belum ditemukan atau kosong.")
        else:
            st.plotly_chart(plot_loss_history(loss_df), use_container_width=True)
            st.dataframe(loss_df.head(25), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Fundamental TLKM</div>", unsafe_allow_html=True)
    st.caption("Data ini diambil dari Yahoo Finance melalui yfinance. Nilai bisa berubah sesuai ketersediaan data.")
    show_fundamentals(PRIMARY_TICKER)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Perbandingan Kinerja Saham</div>", unsafe_allow_html=True)
    st.caption("TLKM tetap menjadi fokus utama. Grafik berikut membandingkan kinerja harga yang dinormalisasi terhadap saham lain.")
    selected = [(k, BENCHMARKS[k]) for k in compare_list]
    selected.insert(0, ("TLKM", PRIMARY_TICKER))
    bench_fig = plot_benchmark(selected, period, interval)
    if bench_fig is None:
        st.warning("Data pembanding belum tersedia.")
    else:
        st.plotly_chart(bench_fig, use_container_width=True)

    st.markdown("### Ringkasan perbandingan")
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
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Data Mentah Historis TLKM</div>", unsafe_allow_html=True)
    preview_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MA20", "MA50"] if c in hist_df.columns]
    show_df = hist_df[preview_cols].copy()
    if "Date" in show_df.columns:
        show_df["Date"] = show_df["Date"].dt.strftime("%d %b %Y")
    st.dataframe(show_df.tail(100), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.divider()
last_update = hist_df["Date"].iloc[-1]
st.caption(
    f"Last update: {last_update.strftime('%d %b %Y') if pd.notna(last_update) else '-'} | "
    f"Source: Yahoo Finance | Interval: {interval} | Focus: TLKM"
)

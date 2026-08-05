import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import metrics

# ==========================================
# 1. СТРАНИЦА И ТЕМАТИЧЕСКИЕ СТИЛИ (CSS)
# ==========================================
st.set_page_config(
    page_title="QuantMetrics Lab — Quant Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для создания тёмного финансового интерфейса
st.markdown("""
<style>
    /* Главный фон и шрифты */
    .stApp {
        background-color: #0E1117;
        color: #E0E3EA;
    }
    
    /* Карточки метрик */
    div[data-testid="stMetric"] {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: #8A91A0 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2962FF !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    /* Настройка боковой панели */
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2A2E39;
    }

    /* Стилизация вкладок */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        background-color: #1E222D;
        border-radius: 8px 8px 0px 0px;
        color: #8A91A0;
        border: 1px solid #2A2E39;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2962FF !important;
        color: #FFFFFF !important;
        border-color: #2962FF !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. СЛОВАРЬ ПЕРЕВОДОВ И ЛОКАЛИЗАЦИЯ
# ==========================================
TRANSLATIONS = {
    "RU": {
        "sidebar_title": "📊 QuantMetrics Lab",
        "sidebar_caption": "Quantitative Risk & Strategy Backtesting Engine",
        "sec1": "1. Активы и Пресеты",
        "preset_label": "Выбор готового пресета:",
        "tickers_label": "Тикеры через запятую (Yahoo Finance):",
        "sec2": "2. Параметры Таймфрейма",
        "start_date": "Дата начала:",
        "end_date": "Дата конца:",
        "timeframe_label": "Интервал сэмплирования:",
        "sec3": "3. Бенчмарк и Безрисковая Ставка",
        "benchmark_label": "Бенчмарк для сравнения:",
        "rf_label": "Безрисковая ставка (Rf, %):",
        "sec4": "4. Распределение весов портфеля",
        "weight_mode_label": "Режим распределения весов:",
        "weight_mode_opts": ["Равные доли (1/N)", "Пользовательские веса"],
        "weight_each": "Вес каждого актива:",
        "specify_weights": "Укажите относительные веса активов:",
        "norm_weights": "Нормированные веса:",
        "err_no_tickers": "Пожалуйста, введите хотя бы один корректный тикер актива.",
        "spinner_load": "Загрузка и обработка финансовых данных...",
        "err_load_fail": "❌ Не удалось загрузить данные по указанным тикерам. Проверьте правильность написания (например, AAPL, MSFT, BTC-USD).",
        "warn_missing": "⚠️ Не удалось загрузить данные для следующих тикеров: ",
        "err_no_valid": "❌ Ни один из введенных тикеров не содержит валидных данных за указанный период.",
        "main_title": "📊 QuantMetrics Lab — Quantitative Analytics Dashboard",
        "summary_metrics_header": "🎯 Сводные Метрики Риска и Доходности Портфеля",
        "m_cagr": "Годовая Доходность (CAGR)",
        "m_vol": "Годовая Волатильность",
        "m_sharpe": "Коэффициент Шарпа",
        "m_sortino": "Коэффициент Сортино",
        "m_mdd": "Макс. Просадка (Max DD)",
        "m_calmar": "Коэффициент Калмара",
        "m_var": "Daily VaR (95% MC)",
        "m_cvar": "Conditional VaR (CVaR 95%)",
        "tab1": "📊 Анализ Портфеля и Рисков",
        "tab2": "📈 Бэктестинг Стратегий",
        "tab3": "🥊 Сравнение с Бенчмарком",
        "tab4": "📋 Сводная Матрица Активов",
        "t1_header": "📈 Кривая Динамики Капитала и Просадки Портфеля (Underwater Plot)",
        "t1_sub1": "Рост Капитала Портфеля (Base = 1.0)",
        "t1_sub2": "Подводный График Просадок (Underwater Drawdown Plot)",
        "t1_trace_port": "Сводный Портфель",
        "t1_y1": "Капитал",
        "t1_y2": "Просадка (%)",
        "t1_corr_header": "🔥 Корреляционная Матрица Активов",
        "t1_corr_title": "Матрица Корреляций (Pearson)",
        "t1_comp_header": "⚖️ Сравнение Риск/Доходность по Активам",
        "t1_comp_title": "CAGR vs Волатильность vs Макс. Просадка",
        "t2_header": "📈 Бэктестинг Торговых Стратегий",
        "t2_settings": "Настройки Стратегии",
        "t2_target": "Целевой Актив для Бэктеста:",
        "t2_comb_port": "Сводный Портфель",
        "t2_strat_label": "Торговая Стратегия:",
        "t2_strat_opts": [
            "SMA Crossover (Скользящие средние)",
            "RSI Strategy (Перекупленность/Перепроданность)",
            "Buy & Hold (Базовая)"
        ],
        "t2_fast_sma": "Быстрая SMA (Дней):",
        "t2_slow_sma": "Медленная SMA (Дней):",
        "t2_rsi_period": "Период RSI:",
        "t2_rsi_over": "Уровень Перепроданности (Buy):",
        "t2_rsi_under": "Уровень Перекупленности (Sell):",
        "t2_results_header": "Результаты Эффективности Стратегии",
        "t2_m_return": "Итоговая Доходность",
        "t2_vs_market": "vs Рынок",
        "t2_m_sharpe": "Коэффициент Шарпа",
        "t2_m_mdd": "Макс. Просадка",
        "t2_m_trades": "Сделок / Win Rate",
        "t2_fig_title1": "Кривая Капитала: Стратегия vs Buy & Hold",
        "t2_fig_title2": "Индикатор / Сигналы Позиции",
        "t2_trace_strat": "Стратегия",
        "t2_trace_bh": "Buy & Hold (Рынок)",
        "t2_trace_pos": "Позиция (Long/Cash)",
        "t3_beta": "Коэффициент Бета (Beta)",
        "t3_alpha": "Годовая Альфа (Alpha)",
        "t3_warn": "⚠️ Данные по бенчмарку {ticker} недоступны. Проверьте правильность тикера в боковой панели.",
        "t4_col_asset": "Актив",
        "t4_col_weight": "Вес в портфеле",
        "t4_col_cagr": "CAGR (%)",
        "t4_col_vol": "Волатильность (%)",
        "t4_col_sharpe": "Коэф. Шарпа",
        "t4_col_sortino": "Коэф. Сортино",
        "t4_col_mdd": "Макс. Просадка (%)",
        "t4_col_calmar": "Коэф. Калмара",
        "t4_col_var": "Daily VaR 95% (%)",
        "t4_col_cvar": "CVaR 95% (%)",
    },
    "EN": {
        "sidebar_title": "📊 QuantMetrics Lab",
        "sidebar_caption": "Quantitative Risk & Strategy Backtesting Engine",
        "sec1": "1. Assets & Presets",
        "preset_label": "Choose preset:",
        "tickers_label": "Tickers separated by comma (Yahoo Finance):",
        "sec2": "2. Timeframe Parameters",
        "start_date": "Start Date:",
        "end_date": "End Date:",
        "timeframe_label": "Sampling Interval:",
        "sec3": "3. Benchmark & Risk-Free Rate",
        "benchmark_label": "Benchmark Ticker:",
        "rf_label": "Risk-Free Rate (Rf, %):",
        "sec4": "4. Portfolio Weight Allocation",
        "weight_mode_label": "Weight Allocation Mode:",
        "weight_mode_opts": ["Equal Weights (1/N)", "Custom Weights"],
        "weight_each": "Weight of each asset:",
        "specify_weights": "Specify relative asset weights:",
        "norm_weights": "Normalized weights:",
        "err_no_tickers": "Please enter at least one valid asset ticker.",
        "spinner_load": "Loading and processing financial data...",
        "err_load_fail": "❌ Failed to load data for specified tickers. Check spelling (e.g. AAPL, MSFT, BTC-USD).",
        "warn_missing": "⚠️ Failed to load data for the following tickers: ",
        "err_no_valid": "❌ None of the entered tickers contain valid data for the specified period.",
        "main_title": "📊 QuantMetrics Lab — Quantitative Analytics Dashboard",
        "summary_metrics_header": "🎯 Portfolio Risk & Return Summary Metrics",
        "m_cagr": "Annual Return (CAGR)",
        "m_vol": "Annualized Volatility",
        "m_sharpe": "Sharpe Ratio",
        "m_sortino": "Sortino Ratio",
        "m_mdd": "Max Drawdown (Max DD)",
        "m_calmar": "Calmar Ratio",
        "m_var": "Daily VaR (95% MC)",
        "m_cvar": "Conditional VaR (CVaR 95%)",
        "tab1": "📊 Portfolio & Risk Analysis",
        "tab2": "📈 Strategy Backtesting",
        "tab3": "🥊 Benchmark Comparison",
        "tab4": "📋 Asset Summary Matrix",
        "t1_header": "📈 Portfolio Equity Curve & Underwater Drawdown Plot",
        "t1_sub1": "Portfolio Equity Growth (Base = 1.0)",
        "t1_sub2": "Underwater Drawdown Plot",
        "t1_trace_port": "Combined Portfolio",
        "t1_y1": "Equity",
        "t1_y2": "Drawdown (%)",
        "t1_corr_header": "🔥 Asset Correlation Matrix",
        "t1_corr_title": "Correlation Matrix (Pearson)",
        "t1_comp_header": "⚖️ Risk/Return Comparison by Asset",
        "t1_comp_title": "CAGR vs Volatility vs Max Drawdown",
        "t2_header": "📈 Trading Strategy Backtesting",
        "t2_settings": "Strategy Settings",
        "t2_target": "Target Asset for Backtest:",
        "t2_comb_port": "Combined Portfolio",
        "t2_strat_label": "Trading Strategy:",
        "t2_strat_opts": [
            "SMA Crossover",
            "RSI Strategy",
            "Buy & Hold"
        ],
        "t2_fast_sma": "Fast SMA (Days):",
        "t2_slow_sma": "Slow SMA (Days):",
        "t2_rsi_period": "RSI Period:",
        "t2_rsi_over": "Oversold Level (Buy):",
        "t2_rsi_under": "Overbought Level (Sell):",
        "t2_results_header": "Strategy Performance Results",
        "t2_m_return": "Total Return",
        "t2_vs_market": "vs Market",
        "t2_m_sharpe": "Sharpe Ratio",
        "t2_m_mdd": "Max Drawdown",
        "t2_m_trades": "Trades / Win Rate",
        "t2_fig_title1": "Equity Curve: Strategy vs Buy & Hold",
        "t2_fig_title2": "Indicator / Position Signals",
        "t2_trace_strat": "Strategy",
        "t2_trace_bh": "Buy & Hold (Market)",
        "t2_trace_pos": "Position (Long/Cash)",
        "t3_beta": "Beta Coefficient",
        "t3_alpha": "Annualized Alpha",
        "t3_warn": "⚠️ Benchmark data for {ticker} is unavailable. Check ticker symbol in the sidebar.",
        "t4_col_asset": "Asset",
        "t4_col_weight": "Portfolio Weight",
        "t4_col_cagr": "CAGR (%)",
        "t4_col_vol": "Volatility (%)",
        "t4_col_sharpe": "Sharpe Ratio",
        "t4_col_sortino": "Sortino Ratio",
        "t4_col_mdd": "Max Drawdown (%)",
        "t4_col_calmar": "Calmar Ratio",
        "t4_col_var": "Daily VaR 95% (%)",
        "t4_col_cvar": "CVaR 95% (%)",
    }
}


# ==========================================
# 3. КЭШИРОВАНИЕ ЗАГРУЗКИ ДАННЫХ
# ==========================================
PRESETS = {
    "Custom Input": "",
    "🚀 Tech Giants": "AAPL, MSFT, NVDA, GOOGL, AMZN",
    "🏛️ S&P 500 Top 5": "AAPL, MSFT, NVDA, AMZN, META",
    "⚖️ Balanced 60/40": "SPY, TLT",
    "🛡️ All Weather": "SPY, TLT, GLD, DBC",
    "🪙 Crypto & Stocks": "BTC-USD, ETH-USD, AAPL, NVDA"
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker_data(tickers_tuple: tuple, start_date, end_date) -> pd.DataFrame:
    """
    Безопасно скачивает рыночные данные через yfinance и извлекает цены закрытия.
    """
    if not tickers_tuple:
        return pd.DataFrame()

    tickers_str = " ".join(tickers_tuple)
    try:
        raw_df = yf.download(
            tickers=tickers_str,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        close_df = metrics.extract_close_prices(raw_df)
        return close_df
    except Exception as e:
        st.error(f"Ошибка загрузки данных yfinance: {e}")
        return pd.DataFrame()


# ==========================================
# 4. БОКОВАЯ ПАНЕЛЬ И НАСТРОЙКИ
# ==========================================
# Выпадающий выбор языка в самом верху боковой панели
lang = st.sidebar.selectbox("Language / Язык", ["RU", "EN"])

def t(key):
    return TRANSLATIONS.get(lang, TRANSLATIONS["RU"]).get(key, key)

st.sidebar.title(t("sidebar_title"))
st.sidebar.caption(t("sidebar_caption"))

st.sidebar.subheader(t("sec1"))
preset_choice = st.sidebar.selectbox(t("preset_label"), list(PRESETS.keys()))

default_tickers = PRESETS[preset_choice] if preset_choice != "Custom Input" else "AAPL, MSFT, NVDA, SPY"
tickers_input = st.sidebar.text_input(t("tickers_label"), value=default_tickers)

parsed_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.subheader(t("sec2"))
col_date1, col_date2 = st.sidebar.columns(2)
start_date = col_date1.date_input(t("start_date"), value=pd.to_datetime("2022-01-01"))
end_date = col_date2.date_input(t("end_date"), value=pd.to_datetime("today"))

timeframe_choice = st.sidebar.selectbox(
    t("timeframe_label"),
    ["Daily", "Weekly", "Monthly"],
    index=0
)

st.sidebar.subheader(t("sec3"))
benchmark_ticker = st.sidebar.text_input(t("benchmark_label"), value="SPY").strip().upper()
risk_free_rate = st.sidebar.number_input(t("rf_label"), value=2.0, step=0.25) / 100.0

# Настройка весов активов
st.sidebar.subheader(t("sec4"))
weight_mode = st.sidebar.radio(t("weight_mode_label"), t("weight_mode_opts"))

weights = []
if len(parsed_tickers) > 0:
    if weight_mode == t("weight_mode_opts")[0]:
        eq_w = 1.0 / len(parsed_tickers)
        weights = [eq_w] * len(parsed_tickers)
        st.sidebar.info(f"{t('weight_each')} {eq_w * 100:.2f}%")
    else:
        raw_weights = []
        st.sidebar.caption(t("specify_weights"))
        for t_symbol in parsed_tickers:
            w_val = st.sidebar.slider(f"Weight {t_symbol} (%)" if lang == "EN" else f"Вес {t_symbol} (%)", min_value=0, max_value=100, value=int(100 / len(parsed_tickers)))
            raw_weights.append(w_val)
        sum_w = sum(raw_weights)
        if sum_w == 0:
            weights = [1.0 / len(parsed_tickers)] * len(parsed_tickers)
        else:
            weights = [w / sum_w for w in raw_weights]

        st.sidebar.caption(t("norm_weights"))
        for t_symbol, w in zip(parsed_tickers, weights):
            st.sidebar.text(f"• {t_symbol}: {w * 100:.1f}%")


# ==========================================
# 5. ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ
# ==========================================
if not parsed_tickers:
    st.error(t("err_no_tickers"))
    st.stop()

all_tickers_to_fetch = list(set(parsed_tickers + ([benchmark_ticker] if benchmark_ticker else [])))

with st.spinner(t("spinner_load")):
    df_close_all = load_ticker_data(tuple(all_tickers_to_fetch), start_date, end_date)

if df_close_all.empty:
    st.error(t("err_load_fail"))
    st.stop()

# Проверяем, какие активы загрузились успешно
valid_tickers = [t_symbol for t_symbol in parsed_tickers if t_symbol in df_close_all.columns and not df_close_all[t_symbol].isna().all()]
missing_tickers = list(set(parsed_tickers) - set(valid_tickers))

if missing_tickers:
    st.warning(f"{t('warn_missing')}{', '.join(missing_tickers)}")

if not valid_tickers:
    st.error(t("err_no_valid"))
    st.stop()

# Нормируем веса только для валидных тикеров
if weight_mode == t("weight_mode_opts")[0]:
    valid_weights = [1.0 / len(valid_tickers)] * len(valid_tickers)
else:
    valid_raw_weights = [weights[parsed_tickers.index(t_symbol)] for t_symbol in valid_tickers]
    w_sum = sum(valid_raw_weights)
    valid_weights = [w / w_sum for w in valid_raw_weights] if w_sum > 0 else [1.0 / len(valid_tickers)] * len(valid_tickers)

# Ресемплинг цен под выбранный таймфрейм
df_close_resampled = metrics.resample_prices(df_close_all[valid_tickers], timeframe=timeframe_choice)
periods_per_year = 252 if timeframe_choice == 'Daily' else (52 if timeframe_choice == 'Weekly' else 12)

# Расчет доходностей активов и портфеля
returns_df = metrics.calculate_daily_returns(df_close_resampled)
portfolio_returns = metrics.calculate_portfolio_returns(returns_df, valid_weights)
returns_df_with_port = returns_df.copy()
returns_df_with_port['Portfolio'] = portfolio_returns

# Кумулятивный капитал портфеля
portfolio_equity = (1.0 + portfolio_returns).cumprod()
portfolio_equity.name = "Portfolio Equity"

# ==========================================
# 6. ГЛАВНАЯ ПАНЕЛЬ И РАСШИРЕННЫЕ МЕТРИКИ
# ==========================================
st.title(t("main_title"))
period_text = t("period_info").format(start_date=start_date, end_date=end_date, timeframe=timeframe_choice, num_assets=len(valid_tickers))
st.caption(period_text)

# Расчет ключевых метрик риска для портфеля
port_cagr = metrics.calculate_cagr(portfolio_equity, periods_per_year=periods_per_year)
port_vol = metrics.calculate_annualized_volatility(portfolio_returns, periods_per_year=periods_per_year)
port_sharpe = metrics.calculate_sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
port_sortino = metrics.calculate_sortino_ratio(portfolio_returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
port_mdd = metrics.calculate_max_drawdown(portfolio_equity)
port_calmar = metrics.calculate_calmar_ratio(port_cagr, port_mdd)
hist_var, mc_var, cvar = metrics.calculate_var_cvar(portfolio_returns, confidence_level=0.95)

# Карточки показателей (2 строки по 4 метрики)
st.subheader(t("summary_metrics_header"))

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric(t("m_cagr"), f"{port_cagr * 100:.2f}%")
m_col2.metric(t("m_vol"), f"{port_vol * 100:.2f}%")
m_col3.metric(t("m_sharpe"), f"{port_sharpe:.2f}")
m_col4.metric(t("m_sortino"), f"{port_sortino:.2f}")

m_col5, m_col6, m_col7, m_col8 = st.columns(4)
m_col5.metric(t("m_mdd"), f"{port_mdd * 100:.2f}%")
m_col6.metric(t("m_calmar"), f"{port_calmar:.2f}")
m_col7.metric(t("m_var"), f"{mc_var * 100:.2f}%")
m_col8.metric(t("m_cvar"), f"{cvar * 100:.2f}%")

st.markdown("---")

# ==========================================
# 7. ВКЛАДКИ ФУНКЦИОНАЛА (ST.TABS)
# ==========================================
tab_risk, tab_backtest, tab_benchmark, tab_matrix = st.tabs([
    t("tab1"),
    t("tab2"),
    t("tab3"),
    t("tab4")
])


# ------------------------------------------
# ВКЛАДКА 1: АНАЛИЗ ПОРТФЕЛЯ И РИСКОВ
# ------------------------------------------
with tab_risk:
    st.subheader(t("t1_header"))

    # Рассчитываем серии просадок
    drawdown_df = metrics.calculate_drawdowns(portfolio_equity)

    fig_port = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(t("t1_sub1"), t("t1_sub2")),
        row_heights=[0.68, 0.32]
    )

    # График 1: Equity
    fig_port.add_trace(
        go.Scatter(
            x=portfolio_equity.index,
            y=portfolio_equity.values,
            mode='lines',
            name=t("t1_trace_port"),
            line=dict(color='#2962FF', width=2.5)
        ),
        row=1, col=1
    )

    # Добавляем индивидуальные активы
    for t_symbol in valid_tickers:
        asset_eq = (1.0 + returns_df[t_symbol]).cumprod()
        fig_port.add_trace(
            go.Scatter(
                x=asset_eq.index,
                y=asset_eq.values,
                mode='lines',
                name=t_symbol,
                opacity=0.45,
                line=dict(width=1.2)
            ),
            row=1, col=1
        )

    # График 2: Drawdown Area Chart
    fig_port.add_trace(
        go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['Drawdown'] * 100,
            mode='lines',
            name='Drawdown (%)',
            fill='tozeroy',
            fillcolor='rgba(255, 77, 77, 0.3)',
            line=dict(color='#FF4D4D', width=1.5)
        ),
        row=2, col=1
    )

    fig_port.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_port.update_yaxes(title_text=t("t1_y1"), row=1, col=1)
    fig_port.update_yaxes(title_text=t("t1_y2"), row=2, col=1)

    st.plotly_chart(fig_port, width="stretch")

    # Корреляционная матрица Heatmap и гистограммы
    col_heat, col_bar = st.columns([1, 1])

    with col_heat:
        st.subheader(t("t1_corr_header"))
        corr_matrix = metrics.calculate_correlation_matrix(returns_df[valid_tickers])

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            aspect="auto",
            template="plotly_dark",
            title=t("t1_corr_title")
        )
        fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, width="stretch")

    with col_bar:
        st.subheader(t("t1_comp_header"))
        metrics_comp = []
        for col_name in valid_tickers + ['Portfolio']:
            ret_col = returns_df_with_port[col_name]
            eq_col = (1.0 + ret_col).cumprod()
            metrics_comp.append({
                'Ticker': col_name,
                'CAGR (%)': metrics.calculate_cagr(eq_col, periods_per_year) * 100,
                'Volatility (%)': metrics.calculate_annualized_volatility(ret_col, periods_per_year) * 100,
                'Sharpe': metrics.calculate_sharpe_ratio(ret_col, risk_free_rate, periods_per_year),
                'Max DD (%)': abs(metrics.calculate_max_drawdown(eq_col)) * 100
            })

        df_metrics_comp = pd.DataFrame(metrics_comp)

        fig_comp = px.bar(
            df_metrics_comp,
            x='Ticker',
            y=['CAGR (%)', 'Volatility (%)', 'Max DD (%)'],
            barmode='group',
            template='plotly_dark',
            title=t("t1_comp_title")
        )
        fig_comp.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_comp, width="stretch")


# ------------------------------------------
# ВКЛАДКА 2: БЭКТЕСТИНГ СТРАТЕГИЙ
# ------------------------------------------
with tab_backtest:
    st.subheader(t("t2_header"))

    bt_col1, bt_col2 = st.columns([1, 2])

    with bt_col1:
        st.markdown(f"#### {t('t2_settings')}")
        strat_opts_translated = t("t2_strat_opts")
        backtest_target = st.selectbox(t("t2_target"), [t("t2_comb_port")] + valid_tickers)

        if backtest_target == t("t2_comb_port"):
            target_series = portfolio_equity
        else:
            target_series = df_close_resampled[backtest_target]

        strategy_type_translated = st.selectbox(t("t2_strat_label"), strat_opts_translated)

        strat_idx = strat_opts_translated.index(strategy_type_translated)
        
        if strat_idx == 0:
            fast_sma = st.slider(t("t2_fast_sma"), min_value=5, max_value=50, value=20)
            slow_sma = st.slider(t("t2_slow_sma"), min_value=20, max_value=200, value=50)
            backtest_res = metrics.run_sma_crossover_strategy(target_series, fast_period=fast_sma, slow_period=slow_sma)
        elif strat_idx == 1:
            rsi_period = st.slider(t("t2_rsi_period"), min_value=5, max_value=30, value=14)
            rsi_oversold = st.slider(t("t2_rsi_over"), min_value=10, max_value=45, value=30)
            rsi_overbought = st.slider(t("t2_rsi_under"), min_value=55, max_value=90, value=70)
            backtest_res = metrics.run_rsi_strategy(target_series, rsi_period=rsi_period, oversold=rsi_oversold, overbought=rsi_overbought)
        else:
            backtest_res = metrics.run_buy_and_hold_strategy(target_series)

        perf_stats = metrics.analyze_backtest_performance(backtest_res, periods_per_year=periods_per_year)

    with bt_col2:
        st.markdown(f"#### {t('t2_results_header')}")
        b_m1, b_m2, b_m3, b_m4 = st.columns(4)

        b_m1.metric(t("t2_m_return"), f"{perf_stats['total_strat_return'] * 100:.2f}%", delta=f"{perf_stats['excess_return'] * 100:.2f}% {t('t2_vs_market')}")
        b_m2.metric(t("t2_m_sharpe"), f"{perf_stats['sharpe_strat']:.2f}")
        b_m3.metric(t("t2_m_mdd"), f"{perf_stats['mdd_strat'] * 100:.2f}%")
        b_m4.metric(t("t2_m_trades"), f"{perf_stats['num_trades']} / {perf_stats['win_rate'] * 100:.1f}%")

        fig_bt = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(t("t2_fig_title1"), t("t2_fig_title2")),
            row_heights=[0.68, 0.32]
        )

        fig_bt.add_trace(
            go.Scatter(x=backtest_res.index, y=backtest_res['Strategy_Equity'], name=t("t2_trace_strat"), line=dict(color='#00E676', width=2.5)),
            row=1, col=1
        )
        fig_bt.add_trace(
            go.Scatter(x=backtest_res.index, y=backtest_res['Market_Equity'], name=t("t2_trace_bh"), line=dict(color='#8A91A0', width=1.5, dash='dash')),
            row=1, col=1
        )

        if strat_idx == 0 and 'Fast_SMA' in backtest_res.columns:
            fig_bt.add_trace(go.Scatter(x=backtest_res.index, y=backtest_res['Fast_SMA'], name='Fast SMA', line=dict(color='#FFEA00', width=1)), row=2, col=1)
            fig_bt.add_trace(go.Scatter(x=backtest_res.index, y=backtest_res['Slow_SMA'], name='Slow SMA', line=dict(color='#FF2929', width=1)), row=2, col=1)
        elif strat_idx == 1 and 'RSI' in backtest_res.columns:
            fig_bt.add_trace(go.Scatter(x=backtest_res.index, y=backtest_res['RSI'], name='RSI', line=dict(color='#00B0FF', width=1.5)), row=2, col=1)
            fig_bt.add_hline(y=rsi_overbought, line_dash="dot", line_color="red", row=2, col=1)
            fig_bt.add_hline(y=rsi_oversold, line_dash="dot", line_color="green", row=2, col=1)
        else:
            fig_bt.add_trace(go.Scatter(x=backtest_res.index, y=backtest_res['Position'], name=t("t2_trace_pos"), line=dict(color='#2962FF', width=1.5)), row=2, col=1)

        fig_bt.update_layout(template="plotly_dark", height=500, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bt, width="stretch")


# ------------------------------------------
# ВКЛАДКА 3: СРАВНЕНИЕ С БЕНЧМАРКОМ
# ------------------------------------------
with tab_benchmark:
    st.subheader(f"🥊 {t('t3_header').format(ticker=benchmark_ticker)}")

    if benchmark_ticker not in df_close_all.columns:
        st.warning(t("t3_warn").format(ticker=benchmark_ticker))
    else:
        bench_close = metrics.resample_prices(df_close_all[[benchmark_ticker]], timeframe=timeframe_choice)[benchmark_ticker]
        bench_returns = bench_close.pct_change().dropna()
        bench_equity = (1.0 + bench_returns).cumprod()

        beta, alpha_annualized = metrics.calculate_beta_alpha(portfolio_returns, bench_returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

        aligned_df = pd.DataFrame({'Portfolio': portfolio_equity, 'Benchmark': bench_equity}).dropna()
        aligned_returns = pd.DataFrame({'Portfolio': portfolio_returns, 'Benchmark': bench_returns}).dropna()

        bench_cagr = metrics.calculate_cagr(aligned_df['Benchmark'], periods_per_year)
        bench_vol = metrics.calculate_annualized_volatility(aligned_returns['Benchmark'], periods_per_year)
        bench_sharpe = metrics.calculate_sharpe_ratio(aligned_returns['Benchmark'], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric(t("t3_beta"), f"{beta:.2f}")
        bc2.metric(t("t3_alpha"), f"{alpha_annualized * 100:.2f}%")
        bc3.metric(f"Sharpe ({benchmark_ticker})", f"{bench_sharpe:.2f}")
        bc4.metric(f"CAGR ({benchmark_ticker})", f"{bench_cagr * 100:.2f}%")

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=aligned_df.index, y=aligned_df['Portfolio'], mode='lines', name=t("t1_trace_port"), line=dict(color='#2962FF', width=2.5)))
        fig_bench.add_trace(go.Scatter(x=aligned_df.index, y=aligned_df['Benchmark'], mode='lines', name=f"Benchmark ({benchmark_ticker})" if lang == "EN" else f"Бенчмарк ({benchmark_ticker})", line=dict(color='#FF6D00', width=2, dash='dash')))

        fig_bench.update_layout(
            template="plotly_dark",
            title=f"Portfolio vs {benchmark_ticker}" if lang == "EN" else f"Сравнительная Динамика: Портфель vs {benchmark_ticker}",
            height=450,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bench, width="stretch")


# ------------------------------------------
# ВКЛАДКА 4: СВОДНАЯ МАТРИЦА АКТИВОВ
# ------------------------------------------
with tab_matrix:
    st.subheader(t("t4_header"))

    summary_rows = []

    summary_rows.append({
        t("t4_col_asset"): "💼 " + t("t1_trace_port"),
        t("t4_col_weight"): "100.0%",
        t("t4_col_cagr"): f"{port_cagr * 100:.2f}%",
        t("t4_col_vol"): f"{port_vol * 100:.2f}%",
        t("t4_col_sharpe"): f"{port_sharpe:.2f}",
        t("t4_col_sortino"): f"{port_sortino:.2f}",
        t("t4_col_mdd"): f"{port_mdd * 100:.2f}%",
        t("t4_col_calmar"): f"{port_calmar:.2f}",
        t("t4_col_var"): f"{mc_var * 100:.2f}%",
        t("t4_col_cvar"): f"{cvar * 100:.2f}%"
    })

    for t_symbol, w in zip(valid_tickers, valid_weights):
        r_col = returns_df[t_symbol]
        eq_col = (1.0 + r_col).cumprod()

        a_cagr = metrics.calculate_cagr(eq_col, periods_per_year)
        a_vol = metrics.calculate_annualized_volatility(r_col, periods_per_year)
        a_sharpe = metrics.calculate_sharpe_ratio(r_col, risk_free_rate, periods_per_year)
        a_sortino = metrics.calculate_sortino_ratio(r_col, risk_free_rate, periods_per_year)
        a_mdd = metrics.calculate_max_drawdown(eq_col)
        a_calmar = metrics.calculate_calmar_ratio(a_cagr, a_mdd)
        a_hvar, a_mcvar, a_cvar = metrics.calculate_var_cvar(r_col)

        summary_rows.append({
            t("t4_col_asset"): t_symbol,
            t("t4_col_weight"): f"{w * 100:.1f}%",
            t("t4_col_cagr"): f"{a_cagr * 100:.2f}%",
            t("t4_col_vol"): f"{a_vol * 100:.2f}%",
            t("t4_col_sharpe"): f"{a_sharpe:.2f}",
            t("t4_col_sortino"): f"{a_sortino:.2f}",
            t("t4_col_mdd"): f"{a_mdd * 100:.2f}%",
            t("t4_col_calmar"): f"{a_calmar:.2f}",
            t("t4_col_var"): f"{a_mcvar * 100:.2f}%",
            t("t4_col_cvar"): f"{a_cvar * 100:.2f}%"
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, width="stretch", hide_index=True)

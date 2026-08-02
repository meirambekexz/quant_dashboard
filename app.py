import streamlit as st
import yfinance as yf
import pandas as pd
from metrics import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown, monte_carlo_var, extract_series

st.set_page_config(page_title="QuantMetrics Lab", layout="wide")

st.title("📊 QuantMetrics Lab — Анализ Рисков и Бэктестинг")

st.sidebar.header("Параметры ввода")
ticker = st.sidebar.text_input("Тикер актива (Yahoo Finance)", value="AAPL")
start_date = st.sidebar.date_input("Начальная дата", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Конечная дата", value=pd.to_datetime("today"))

if st.sidebar.button("Загрузить и рассчитать"):
    with st.spinner("Скачиваем данные с биржи..."):
        data = yf.download(ticker, start=start_date, end=end_date)

        if not data.empty:
            returns = calculate_returns(data)
            close_prices = extract_series(data, 'Close')

            # Визуализация метрик
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Коэффициент Шарпа", f"{calculate_sharpe_ratio(returns):.2f}")
            col2.metric("Max Drawdown", f"{calculate_max_drawdown(data) * 100:.2f}%")
            col3.metric("Daily VaR (95%)", f"{abs(monte_carlo_var(returns)) * 100:.2f}%")
            col4.metric("Ср. доходность (в день)", f"{returns.mean() * 100:.3f}%")

            # График
            st.subheader(f"График цены закрытия {ticker}")
            st.line_chart(close_prices)
        else:
            st.error("Ошибка: Не удалось загрузить данные. Проверьте правильность тикера.")
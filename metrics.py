import numpy as np
import pandas as pd


def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает цены закрытия в чистый DataFrame (даты x тикеры),
    обрабатывая MultiIndex или обычные колонки от yfinance.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Проверяем наличие 'Close' или 'Adj Close'
    col_type = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)

    if col_type is None:
        close_df = df.copy()
    elif isinstance(df.columns, pd.MultiIndex):
        close_df = df[col_type].copy()
    else:
        close_df = df[[col_type]].copy()

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()

    # Заполняем пропуски и приводим к float
    close_df = close_df.dropna(how='all').ffill().bfill().astype(float)
    return close_df


def resample_prices(df_close: pd.DataFrame, timeframe: str = 'Daily') -> pd.DataFrame:
    """
    Ресемплит цены закрытия на Daily ('D'), Weekly ('W-FRI') или Monthly ('ME'/'M').
    """
    if df_close.empty or timeframe == 'Daily':
        return df_close

    if timeframe == 'Weekly':
        resampled = df_close.resample('W-FRI').last()
    elif timeframe == 'Monthly':
        try:
            resampled = df_close.resample('ME').last()
        except Exception:
            resampled = df_close.resample('M').last()
    else:
        resampled = df_close

    return resampled.dropna(how='all')


def calculate_daily_returns(df_close: pd.DataFrame) -> pd.DataFrame:
    """
    Расчет процентных дневных доходностей.
    """
    return df_close.pct_change().dropna(how='all')


def calculate_portfolio_returns(returns_df: pd.DataFrame, weights: list | np.ndarray) -> pd.Series:
    """
    Расчет взвешенной доходности портфеля.
    Веса автоматически нормируются к сумме 1.0.
    """
    w = np.array(weights, dtype=float)
    if w.sum() == 0 or np.isnan(w.sum()):
        w = np.ones(len(w)) / len(w)
    else:
        w = w / w.sum()

    portfolio_returns = returns_df.dot(w)
    portfolio_returns.name = "Portfolio"
    return portfolio_returns


def calculate_cagr(series_or_df: pd.Series | pd.DataFrame, periods_per_year: int = 252) -> float | pd.Series:
    """
    Расчет совокупного среднегодового темпа роста (CAGR).
    """
    if isinstance(series_or_df, pd.Series):
        s = series_or_df.dropna()
        if len(s) < 2:
            return 0.0
        total_return = (s.iloc[-1] / s.iloc[0]) - 1.0
        num_periods = len(s)
        if num_periods <= 1 or total_return <= -1.0:
            return -1.0
        cagr = (1.0 + total_return) ** (periods_per_year / num_periods) - 1.0
        return float(cagr)
    else:
        return series_or_df.apply(lambda col: calculate_cagr(col, periods_per_year))


def calculate_annualized_volatility(returns: pd.Series | pd.DataFrame, periods_per_year: int = 252) -> float | pd.Series:
    """
    Годовая волатильность (Annualized Volatility).
    """
    vol = returns.std() * np.sqrt(periods_per_year)
    if isinstance(vol, pd.Series):
        return vol.astype(float)
    return float(vol)


def calculate_sharpe_ratio(returns: pd.Series | pd.DataFrame, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float | pd.Series:
    """
    Расчет коэффициента Шарпа (Sharpe Ratio).
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    mean_excess = excess_returns.mean()
    std_returns = returns.std()

    if isinstance(std_returns, pd.Series):
        sharpe = np.sqrt(periods_per_year) * (mean_excess / std_returns.replace(0, np.nan))
        return sharpe.fillna(0.0).astype(float)
    else:
        if std_returns == 0 or np.isnan(std_returns):
            return 0.0
        return float(np.sqrt(periods_per_year) * (mean_excess / std_returns))


def calculate_sortino_ratio(returns: pd.Series | pd.DataFrame, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float | pd.Series:
    """
    Расчет коэффициента Сортино (Sortino Ratio), учитывающего только отрицательную волатильность.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    def _sortino_single(r_col: pd.Series) -> float:
        downside = r_col[r_col < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = np.sqrt(np.mean(downside**2))
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float(np.sqrt(periods_per_year) * (r_col.mean() / downside_std))

    if isinstance(excess_returns, pd.Series):
        return _sortino_single(excess_returns)
    else:
        return excess_returns.apply(_sortino_single)


def calculate_drawdowns(equity_or_returns: pd.Series) -> pd.DataFrame:
    """
    Расчет кумулятивного капитала, пика и серии просадок (Drawdown).
    Возвращает DataFrame с колонками ['Equity', 'Peak', 'Drawdown'].
    """
    s = equity_or_returns.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=['Equity', 'Peak', 'Drawdown'])

    # Если передан ряд доходностей, переводим в кумулятивный капитал от 1.0
    if abs(s.mean()) < 0.2 and s.min() > -1.0 and s.max() < 1.0:
        equity = (1.0 + s).cumprod()
    else:
        equity = s / s.iloc[0]

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return pd.DataFrame({
        'Equity': equity,
        'Peak': peak,
        'Drawdown': drawdown
    })


def calculate_max_drawdown(equity_or_returns: pd.Series) -> float:
    """
    Максимальная просадка (Max Drawdown) как отрицательное число (например, -0.25 для -25%).
    """
    dd_df = calculate_drawdowns(equity_or_returns)
    if dd_df.empty:
        return 0.0
    mdd = dd_df['Drawdown'].min()
    return float(mdd) if not np.isnan(mdd) else 0.0


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Расчет коэффициента Калмара (Calmar Ratio = CAGR / |Max Drawdown|).
    """
    mdd_abs = abs(max_drawdown)
    if mdd_abs == 0 or np.isnan(mdd_abs):
        return 0.0
    return float(cagr / mdd_abs)


def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95, simulations: int = 10000) -> tuple[float, float, float]:
    """
    Оценка Value at Risk (VaR) и Conditional VaR (CVaR / Expected Shortfall 95%).
    Возвращает (Historical_VaR, MonteCarlo_VaR, CVaR) в виде положительных % убытка.
    """
    r = returns.dropna()
    if len(r) < 5:
        return 0.0, 0.0, 0.0

    # 1. Historical VaR
    hist_var = -float(np.percentile(r, (1.0 - confidence_level) * 100))

    # 2. Monte Carlo VaR
    mu = float(r.mean())
    sigma = float(r.std())
    if sigma == 0 or np.isnan(sigma):
        mc_var = hist_var
    else:
        simulated = np.random.normal(mu, sigma, simulations)
        mc_var = -float(np.percentile(simulated, (1.0 - confidence_level) * 100))

    # 3. Conditional VaR (CVaR) - средний убыток в худших (1 - confidence_level)% дней
    cutoff = np.percentile(r, (1.0 - confidence_level) * 100)
    worst_returns = r[r <= cutoff]
    if len(worst_returns) > 0:
        cvar = -float(worst_returns.mean())
    else:
        cvar = hist_var

    return max(0.0, hist_var), max(0.0, mc_var), max(0.0, cvar)


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляционная матрица Пирсона между доходностями активов.
    """
    return returns_df.corr(method='pearson')


def calculate_beta_alpha(asset_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> tuple[float, float]:
    """
    Расчет Бета (Beta) и Альфа (Alpha - годовая) относительно бенчмарка.
    """
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 0.0

    y = aligned.iloc[:, 0]  # Актив/Портфель
    x = aligned.iloc[:, 1]  # Бенчмарк

    cov_matrix = np.cov(y, x)
    cov = cov_matrix[0, 1]
    var_x = cov_matrix[1, 1]

    if var_x == 0 or np.isnan(var_x):
        return 0.0, 0.0

    beta = float(cov / var_x)

    rf_daily = risk_free_rate / periods_per_year
    alpha_daily = (y.mean() - rf_daily) - beta * (x.mean() - rf_daily)
    alpha_annualized = float(alpha_daily * periods_per_year)

    return beta, alpha_annualized


# ==========================================
# БЭКТЕСТИНГ СТРАТЕГИЙ
# ==========================================

def run_sma_crossover_strategy(close_prices: pd.Series, fast_period: int = 20, slow_period: int = 50) -> pd.DataFrame:
    """
    Стратегия пересечения скользящих средних (SMA Crossover).
    Позиция = 1 (Long) при Fast SMA > Slow SMA, иначе 0 (Cash).
    """
    df = pd.DataFrame({'Close': close_prices})
    df['Fast_SMA'] = df['Close'].rolling(window=fast_period).mean()
    df['Slow_SMA'] = df['Close'].rolling(window=slow_period).mean()

    df['Signal'] = np.where(df['Fast_SMA'] > df['Slow_SMA'], 1, 0)
    # Сдвиг на 1 день исключает Lookahead Bias
    df['Position'] = df['Signal'].shift(1).fillna(0)

    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Position'] * df['Market_Return']

    df['Market_Equity'] = (1.0 + df['Market_Return']).cumprod()
    df['Strategy_Equity'] = (1.0 + df['Strategy_Return']).cumprod()

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Расчет индекса относительной силы (RSI).
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def run_rsi_strategy(close_prices: pd.Series, rsi_period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.DataFrame:
    """
    Стратегия RSI.
    Покупка при RSI < oversold (Signal = 1), продажа при RSI > overbought (Signal = 0).
    """
    df = pd.DataFrame({'Close': close_prices})
    df['RSI'] = calculate_rsi(close_prices, period=rsi_period)

    signals = np.zeros(len(df))
    current_pos = 0

    for i in range(len(df)):
        rsi_val = df['RSI'].iloc[i]
        if rsi_val < oversold:
            current_pos = 1
        elif rsi_val > overbought:
            current_pos = 0
        signals[i] = current_pos

    df['Signal'] = signals
    df['Position'] = df['Signal'].shift(1).fillna(0)

    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Position'] * df['Market_Return']

    df['Market_Equity'] = (1.0 + df['Market_Return']).cumprod()
    df['Strategy_Equity'] = (1.0 + df['Strategy_Return']).cumprod()

    return df


def run_buy_and_hold_strategy(close_prices: pd.Series) -> pd.DataFrame:
    """
    Базовая стратегия Buy & Hold.
    """
    df = pd.DataFrame({'Close': close_prices})
    df['Signal'] = 1
    df['Position'] = 1

    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Market_Return']

    df['Market_Equity'] = (1.0 + df['Market_Return']).cumprod()
    df['Strategy_Equity'] = df['Market_Equity']

    return df


def analyze_backtest_performance(backtest_df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Вычисляет итоговые метрики эффективности бэктеста.
    """
    strat_returns = backtest_df['Strategy_Return']
    market_returns = backtest_df['Market_Return']

    strat_equity = backtest_df['Strategy_Equity']
    market_equity = backtest_df['Market_Equity']

    total_strat_return = float((strat_equity.iloc[-1] / strat_equity.iloc[0]) - 1.0)
    total_market_return = float((market_equity.iloc[-1] / market_equity.iloc[0]) - 1.0)
    excess_return = total_strat_return - total_market_return

    cagr_strat = calculate_cagr(strat_equity, periods_per_year)
    cagr_market = calculate_cagr(market_equity, periods_per_year)

    vol_strat = calculate_annualized_volatility(strat_returns, periods_per_year)
    sharpe_strat = calculate_sharpe_ratio(strat_returns, periods_per_year=periods_per_year)
    mdd_strat = calculate_max_drawdown(strat_equity)

    # Количество сделок (смена состояния позиции)
    position_changes = backtest_df['Position'].diff().abs()
    num_trades = int((position_changes > 0).sum())

    # Процент прибыльных дней в позиции (Win Rate)
    active_days = backtest_df[backtest_df['Position'] > 0]
    if len(active_days) > 0:
        winning_days = (active_days['Strategy_Return'] > 0).sum()
        win_rate = float(winning_days / len(active_days))
    else:
        win_rate = 0.0

    return {
        'total_strat_return': total_strat_return,
        'total_market_return': total_market_return,
        'excess_return': excess_return,
        'cagr_strat': cagr_strat,
        'cagr_market': cagr_market,
        'vol_strat': vol_strat,
        'sharpe_strat': sharpe_strat,
        'mdd_strat': mdd_strat,
        'num_trades': num_trades,
        'win_rate': win_rate
    }

import numpy as np
import pandas as pd


def extract_series(df: pd.DataFrame, col_name: str = 'Close') -> pd.Series:
    """Извлекает одномерный Series из DataFrame даже при наличии MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        s = df[col_name].iloc[:, 0]
    else:
        s = df[col_name]
    return s.dropna().astype(float)


def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Расчет ежедневной логарифмической доходности."""
    close_prices = extract_series(df, 'Close')
    return np.log(close_prices / close_prices.shift(1)).dropna()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Расчет коэффициента Шарпа."""
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    std_val = float(returns.std())
    if std_val == 0 or np.isnan(std_val):
        return 0.0
    return float(np.sqrt(252) * (excess_returns.mean() / std_val))


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """Расчет максимальной просадки (Max Drawdown)."""
    close_prices = extract_series(df, 'Close')
    cumulative = (1 + close_prices.pct_change()).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def monte_carlo_var(returns: pd.Series, confidence_level: float = 0.95, simulations: int = 10000) -> float:
    """Оценка Value at Risk (VaR) методом Монте-Карло."""
    mu = float(returns.mean())
    sigma = float(returns.std())
    simulated_returns = np.random.normal(mu, sigma, simulations)
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return float(var)
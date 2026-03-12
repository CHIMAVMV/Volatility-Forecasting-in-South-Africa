"""
Preprocessing and exploratory data analysis (EDA) utilities.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def descriptive_statistics(returns):
    """
    Compute descriptive statistics for return series.

    Parameters
    ----------
    returns : pd.DataFrame or pd.Series
        Log-return series.

    Returns
    -------
    pd.DataFrame
        Table with mean, std, skewness, kurtosis, min, max, JB stat, JB p-value.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        jb_stat, jb_pval = stats.jarque_bera(s)
        rows.append(
            {
                "Ticker": col,
                "Observations": len(s),
                "Mean (%)": s.mean() * 100,
                "Std Dev (%)": s.std() * 100,
                "Skewness": s.skew(),
                "Excess Kurtosis": s.kurtosis(),
                "Min (%)": s.min() * 100,
                "Max (%)": s.max() * 100,
                "JB Stat": jb_stat,
                "JB p-value": jb_pval,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


def check_stationarity(returns, significance=0.05):
    """
    Perform Augmented Dickey-Fuller test on each return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series (columns = tickers).
    significance : float
        Significance level for the ADF test.

    Returns
    -------
    pd.DataFrame
        ADF test results for each ticker.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        result = adfuller(s, autolag="AIC")
        rows.append(
            {
                "Ticker": col,
                "ADF Statistic": result[0],
                "p-value": result[1],
                "Critical Value (1%)": result[4]["1%"],
                "Critical Value (5%)": result[4]["5%"],
                "Critical Value (10%)": result[4]["10%"],
                "Stationary": result[1] < significance,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


def check_arch_effects(returns, lags=10):
    """
    Test for ARCH effects (heteroskedasticity) using the Engle LM test.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series.
    lags : int
        Number of lags for the ARCH-LM test.

    Returns
    -------
    pd.DataFrame
        ARCH-LM test results per ticker.
    """
    from statsmodels.stats.diagnostic import het_arch

    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        lm_stat, lm_pval, f_stat, f_pval = het_arch(s, nlags=lags)
        rows.append(
            {
                "Ticker": col,
                "LM Stat": lm_stat,
                "LM p-value": lm_pval,
                "F Stat": f_stat,
                "F p-value": f_pval,
                "ARCH Effects": lm_pval < 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


def rolling_statistics(returns, window=21):
    """
    Compute rolling mean and standard deviation (realised volatility).

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series.
    window : int
        Rolling window in trading days (default 21 ≈ 1 month).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (rolling_mean, rolling_std)
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return rolling_mean, rolling_std


def compute_realised_volatility(returns, window=21, annualise=True):
    """
    Compute realised (historical) volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series.
    window : int
        Rolling window size.
    annualise : bool
        If True, annualise by multiplying by sqrt(252).

    Returns
    -------
    pd.DataFrame
        Realised volatility series.
    """
    rv = returns.rolling(window).std()
    if annualise:
        rv = rv * np.sqrt(252)
    return rv

"""
Data collection module for JSE (Johannesburg Stock Exchange) stock data.
Fetches historical price data using yfinance.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Major JSE-listed stocks and indices
JSE_TICKERS = {
    "JSE Top 40 Index": "^JTOPI",
    "Naspers": "NPN.JO",
    "Anglo American": "AGL.JO",
    "BHP Group": "BHG.JO",
    "Standard Bank": "SBK.JO",
    "FirstRand": "FSR.JO",
    "Absa Group": "ABG.JO",
    "MTN Group": "MTN.JO",
    "Sasol": "SOL.JO",
    "Richemont": "CFR.JO",
    "Impala Platinum": "IMP.JO",
}

DEFAULT_TICKERS = ["^JTOPI", "NPN.JO", "SBK.JO", "MTN.JO"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = datetime.today().strftime("%Y-%m-%d")


def fetch_stock_data(
    tickers,
    start=DEFAULT_START,
    end=DEFAULT_END,
    auto_adjust=True,
):
    """
    Fetch historical closing prices for the given JSE ticker(s).

    Parameters
    ----------
    tickers : list or str
        One or more Yahoo Finance ticker symbols (e.g. "NPN.JO").
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.
    auto_adjust : bool
        Whether to auto-adjust prices for splits and dividends.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted closing prices for the requested tickers.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    logger.info("Fetching data for %s from %s to %s", tickers, start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        multi_level_index=len(tickers) > 1,
    )

    if raw.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    if len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close = raw["Close"].copy()

    close.dropna(how="all", inplace=True)
    logger.info("Downloaded %d rows for %d ticker(s).", len(close), len(tickers))
    return close


def compute_log_returns(prices):
    """
    Compute daily log returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted closing prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns (percentage, first row dropped).
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def fetch_and_prepare(
    tickers=None,
    start=DEFAULT_START,
    end=DEFAULT_END,
):
    """
    Convenience wrapper: fetch prices and return (prices, log_returns).

    Parameters
    ----------
    tickers : list or str, optional
        Ticker symbols to fetch.  Defaults to ``DEFAULT_TICKERS``.
    start : str
        Start date.
    end : str
        End date.

    Returns
    -------
    tuple
        (prices DataFrame, log_returns DataFrame)
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    prices = fetch_stock_data(tickers, start=start, end=end)
    log_returns = compute_log_returns(prices)
    return prices, log_returns

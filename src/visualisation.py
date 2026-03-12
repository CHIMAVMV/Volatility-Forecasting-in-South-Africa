"""
Visualisation utilities for the volatility forecasting project.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE = (12, 5)


def _save_or_show(fig, output_path=None):
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info("Figure saved to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


def plot_prices(prices, title="JSE Stock Prices", output_path=None):
    """Plot adjusted closing prices for all tickers."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for col in prices.columns:
            ax.plot(prices.index, prices[col], label=col, linewidth=0.9)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (ZAR)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_returns(returns, title="Daily Log Returns (%)", output_path=None):
    """Plot log return series for all tickers."""
    pct = returns * 100
    n = len(pct.columns)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, pct.columns):
            ax.plot(pct.index, pct[col], linewidth=0.6, color="steelblue")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel(col, fontsize=10)
        axes[0].set_title(title, fontsize=14)
        axes[-1].set_xlabel("Date")
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_return_distribution(returns, output_path=None):
    """Plot histogram + KDE for each ticker's return distribution."""
    n = len(returns.columns)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, returns.columns):
            data = returns[col].dropna() * 100
            sns.histplot(data, kde=True, ax=ax, bins=60, color="steelblue",
                         stat="density", alpha=0.6)
            ax.set_title(col, fontsize=11)
            ax.set_xlabel("Return (%)")
        fig.suptitle("Return Distributions", fontsize=14)
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_realised_volatility(rv, title="Realised Volatility (Annualised, %)",
                             output_path=None):
    """Plot rolling realised volatility."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for col in rv.columns:
            ax.plot(rv.index, rv[col], label=col, linewidth=0.9)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualised Volatility (%)")
        ax.legend()
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_conditional_volatility(result, ticker, model_type="GARCH",
                                output_path=None):
    """
    Plot the conditional volatility (annualised) from a fitted GARCH result.
    """
    cond_vol = result.conditional_volatility
    annualised = cond_vol / 100 * np.sqrt(252) * 100  # already in pct scale

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(cond_vol.index, annualised, color="darkorange", linewidth=0.9)
        ax.set_title(
            f"{ticker} — {model_type} Conditional Volatility (Annualised %)",
            fontsize=13,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_forecast_vs_realised(forecast_df, ticker, output_path=None):
    """
    Plot out-of-sample forecast variance vs. realised variance (squared returns).
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(
            forecast_df.index,
            forecast_df["actual_variance"] * 1e4,
            label="Realised Variance (×10⁴)",
            color="steelblue",
            linewidth=0.7,
            alpha=0.8,
        )
        ax.plot(
            forecast_df.index,
            forecast_df["forecast_variance"] * 1e4,
            label="GARCH Forecast (×10⁴)",
            color="darkorange",
            linewidth=0.9,
        )
        ax.set_title(f"{ticker} — Forecast vs Realised Variance", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Variance (×10⁴)")
        ax.legend()
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_acf_pacf(returns, ticker, lags=40, output_path=None):
    """Plot ACF and PACF of squared returns (to check for ARCH effects)."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    sq = (returns[ticker].dropna() * 100) ** 2
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(sq, lags=lags, ax=axes[0], title=f"{ticker} — ACF of Squared Returns")
        plot_pacf(sq, lags=lags, ax=axes[1],
                  title=f"{ticker} — PACF of Squared Returns")
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_model_comparison(summary_df, metric="RMSE", output_path=None):
    """Bar chart comparing model performance across tickers."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        summary_df[metric].sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Model Comparison — {metric}", fontsize=13)
        ax.set_xlabel(metric)
        fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_forecast_horizon(vol_forecast, ticker, output_path=None):
    """Bar chart of annualised volatility forecast over a multi-step horizon."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = [f"h={i+1}" for i in range(len(vol_forecast))]
        ax.bar(steps, vol_forecast.values.flatten(), color="steelblue", alpha=0.8)
        ax.set_title(
            f"{ticker} — Volatility Forecast Horizon (Annualised %)",
            fontsize=13,
        )
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Annualised Volatility (%)")
        fig.tight_layout()
    _save_or_show(fig, output_path)

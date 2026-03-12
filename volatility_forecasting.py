"""
Main script — Volatility Forecasting for the South African (JSE) Stock Market.

Usage
-----
    python volatility_forecasting.py

The script:
1. Downloads JSE price data via Yahoo Finance.
2. Computes log returns and descriptive statistics.
3. Tests for stationarity and ARCH effects.
4. Fits GARCH, EGARCH and GJR-GARCH models.
5. Selects the best model by BIC.
6. Produces out-of-sample volatility forecasts.
7. Evaluates and reports forecast accuracy.
8. Saves all plots to ./outputs/plots/ and results to ./outputs/results/.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection import fetch_and_prepare, DEFAULT_TICKERS
from src.preprocessing import (
    descriptive_statistics,
    check_stationarity,
    check_arch_effects,
    compute_realised_volatility,
)
from src.volatility_models import fit_garch, select_best_model, forecast_volatility
from src.evaluation import rolling_window_forecast, summarise_model_performance
from src.visualisation import (
    plot_prices,
    plot_returns,
    plot_return_distribution,
    plot_realised_volatility,
    plot_conditional_volatility,
    plot_forecast_vs_realised,
    plot_acf_pacf,
    plot_model_comparison,
    plot_forecast_horizon,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TICKERS = DEFAULT_TICKERS      # ["^JTOPI", "NPN.JO", "SBK.JO", "MTN.JO"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
PLOTS_DIR = Path("outputs/plots")
RESULTS_DIR = Path("outputs/results")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = "GJR-GARCH"       # Best-performing variant by default
DIST = "StudentsT"
FORECAST_HORIZON = 10          # 10-day ahead forecast
ROLLING_TRAIN_SIZE = 0.8


def main():
    logger.info("=" * 60)
    logger.info("Volatility Forecasting — JSE Stock Market")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data collection
    # ------------------------------------------------------------------
    logger.info("Step 1: Downloading price data …")
    prices, log_returns = fetch_and_prepare(
        tickers=TICKERS, start=START_DATE, end=END_DATE
    )
    logger.info("Price data shape: %s", prices.shape)

    plot_prices(
        prices,
        title="JSE Stock Prices (Adjusted Close)",
        output_path=PLOTS_DIR / "01_prices.png",
    )
    plot_returns(
        log_returns,
        title="JSE Daily Log Returns",
        output_path=PLOTS_DIR / "02_log_returns.png",
    )

    # ------------------------------------------------------------------
    # 2. Descriptive statistics
    # ------------------------------------------------------------------
    logger.info("Step 2: Descriptive statistics …")
    desc_stats = descriptive_statistics(log_returns)
    print("\n--- Descriptive Statistics ---")
    print(desc_stats.to_string())
    desc_stats.to_csv(RESULTS_DIR / "descriptive_statistics.csv")

    plot_return_distribution(
        log_returns,
        output_path=PLOTS_DIR / "03_return_distributions.png",
    )

    # ------------------------------------------------------------------
    # 3. Stationarity & ARCH-effects tests
    # ------------------------------------------------------------------
    logger.info("Step 3: Stationarity tests (ADF) …")
    adf_results = check_stationarity(log_returns)
    print("\n--- ADF Stationarity Test ---")
    print(adf_results.to_string())
    adf_results.to_csv(RESULTS_DIR / "adf_test.csv")

    logger.info("Step 3b: ARCH-LM test …")
    arch_results = check_arch_effects(log_returns)
    print("\n--- ARCH-LM Test ---")
    print(arch_results.to_string())
    arch_results.to_csv(RESULTS_DIR / "arch_lm_test.csv")

    # ------------------------------------------------------------------
    # 4. Realised volatility
    # ------------------------------------------------------------------
    logger.info("Step 4: Realised volatility …")
    rv = compute_realised_volatility(log_returns, window=21, annualise=True)
    plot_realised_volatility(
        rv,
        title="21-Day Realised Volatility (Annualised %)",
        output_path=PLOTS_DIR / "04_realised_volatility.png",
    )

    # ------------------------------------------------------------------
    # 5. GARCH model fitting & selection
    # ------------------------------------------------------------------
    logger.info("Step 5: Model selection …")
    best_results = {}
    selection_tables = {}

    for ticker in log_returns.columns:
        logger.info("  Selecting best model for %s …", ticker)
        best_res, sel_df = select_best_model(
            log_returns[ticker],
            ticker=ticker,
            p_range=(1, 2),
            q_range=(1, 2),
            model_types=("GARCH", "EGARCH", "GJR-GARCH"),
            dist=DIST,
            criterion="bic",
        )
        best_results[ticker] = best_res
        selection_tables[ticker] = sel_df
        sel_df.to_csv(RESULTS_DIR / f"model_selection_{ticker.replace('^', '')}.csv")
        print(f"\n--- Model Selection: {ticker} ---")
        print(sel_df.to_string(index=False))

        plot_conditional_volatility(
            best_res,
            ticker=ticker,
            model_type="Best",
            output_path=PLOTS_DIR / f"05_cond_vol_{ticker.replace('^', '')}.png",
        )
        plot_acf_pacf(
            log_returns,
            ticker=ticker,
            lags=40,
            output_path=PLOTS_DIR / f"06_acf_pacf_{ticker.replace('^', '')}.png",
        )

    # ------------------------------------------------------------------
    # 6. Multi-step ahead forecast
    # ------------------------------------------------------------------
    logger.info("Step 6: Multi-step ahead forecasts …")
    for ticker, res in best_results.items():
        vol_fc = forecast_volatility(res, horizon=FORECAST_HORIZON)
        print(f"\n--- {FORECAST_HORIZON}-Day Volatility Forecast: {ticker} ---")
        print(vol_fc.to_string())
        vol_fc.to_csv(
            RESULTS_DIR / f"vol_forecast_{ticker.replace('^', '')}.csv"
        )
        plot_forecast_horizon(
            vol_fc,
            ticker=ticker,
            output_path=PLOTS_DIR / f"07_forecast_horizon_{ticker.replace('^', '')}.png",
        )

    # ------------------------------------------------------------------
    # 7. Out-of-sample rolling-window evaluation
    # ------------------------------------------------------------------
    logger.info("Step 7: Rolling-window out-of-sample evaluation …")
    oos_results = {}
    for ticker in log_returns.columns:
        logger.info("  Rolling forecast for %s …", ticker)
        oos_df = rolling_window_forecast(
            log_returns[ticker],
            ticker=ticker,
            model_type=MODEL_TYPE,
            dist=DIST,
            train_size=ROLLING_TRAIN_SIZE,
            horizon=1,
        )
        oos_results[ticker] = oos_df
        oos_df.to_csv(
            RESULTS_DIR / f"oos_forecast_{ticker.replace('^', '')}.csv"
        )
        plot_forecast_vs_realised(
            oos_df,
            ticker=ticker,
            output_path=PLOTS_DIR / f"08_oos_forecast_{ticker.replace('^', '')}.png",
        )

    summary = summarise_model_performance(oos_results)
    print("\n--- Out-of-Sample Performance Summary ---")
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "oos_performance_summary.csv")

    plot_model_comparison(
        summary,
        metric="RMSE",
        output_path=PLOTS_DIR / "09_model_comparison_rmse.png",
    )

    logger.info("Done. Outputs saved to %s and %s", PLOTS_DIR, RESULTS_DIR)


if __name__ == "__main__":
    main()

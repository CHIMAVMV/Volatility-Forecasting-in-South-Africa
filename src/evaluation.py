"""
Evaluation metrics and out-of-sample forecasting utilities.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def realised_variance(returns, window=1):
    """
    Compute squared returns as a proxy for realised variance.

    Parameters
    ----------
    returns : pd.Series
        Log-return series.
    window : int
        Aggregation window (default 1 = daily).

    Returns
    -------
    pd.Series
        Daily realised variance series.
    """
    return returns.rolling(window).apply(lambda x: np.sum(x ** 2))


def mse(actual, predicted):
    """Mean squared error."""
    return np.mean((actual - predicted) ** 2)


def rmse(actual, predicted):
    """Root mean squared error."""
    return np.sqrt(mse(actual, predicted))


def mae(actual, predicted):
    """Mean absolute error."""
    return np.mean(np.abs(actual - predicted))


def qlike(actual, predicted):
    """
    QLIKE loss function — a standard volatility loss function that is robust
    to noise in the variance proxy.

    QLIKE = mean( log(predicted) + actual/predicted )
    """
    return np.mean(np.log(predicted) + actual / predicted)


def evaluate_forecast(actual_variance, predicted_variance):
    """
    Compare one-step-ahead variance forecasts against a realised variance proxy.

    Parameters
    ----------
    actual_variance : array-like
        Realised variance (e.g. squared returns).
    predicted_variance : array-like
        Model variance forecasts (same scale as actual_variance).

    Returns
    -------
    dict
        Dictionary with MSE, RMSE, MAE and QLIKE.
    """
    actual = np.asarray(actual_variance)
    predicted = np.asarray(predicted_variance)
    mask = np.isfinite(actual) & np.isfinite(predicted) & (predicted > 0)
    actual = actual[mask]
    predicted = predicted[mask]

    return {
        "MSE": mse(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "QLIKE": qlike(actual, predicted),
    }


def rolling_window_forecast(
    returns,
    ticker,
    model_type="GARCH",
    p=1,
    q=1,
    dist="StudentsT",
    train_size=0.8,
    horizon=1,
    scale=100,
):
    """
    Perform a rolling-window out-of-sample volatility forecast.

    Parameters
    ----------
    returns : pd.Series
        Full log-return series.
    ticker : str
        Asset name.
    model_type : str
        GARCH variant.
    p, q : int
        Lag orders.
    dist : str
        Error distribution.
    train_size : float
        Proportion of data used for the initial training window.
    horizon : int
        Steps-ahead forecast (currently only 1-step-ahead used for evaluation).
    scale : float
        Return scaling factor.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['actual_return', 'actual_variance',
        'forecast_variance', 'forecast_vol_annualised'].
    """
    from src.volatility_models import fit_garch

    series = returns.dropna()
    n = len(series)
    train_end = int(n * train_size)

    forecasts = []
    dates = []

    logger.info(
        "Rolling forecast for %s — train=%d, test=%d",
        ticker, train_end, n - train_end,
    )

    for i in range(train_end, n):
        train = series.iloc[:i]
        try:
            res = fit_garch(
                train, ticker, p=p, q=q,
                model_type=model_type, dist=dist, scale=scale,
            )
            fc = res.forecast(horizon=horizon, reindex=False)
            var_fc = fc.variance.iloc[-1, 0] / (scale ** 2)
        except Exception as exc:
            logger.warning("Forecast failed at step %d: %s", i, exc)
            var_fc = np.nan

        forecasts.append(var_fc)
        dates.append(series.index[i])

    test_returns = series.iloc[train_end:]
    df = pd.DataFrame(
        {
            "actual_return": test_returns.values,
            "actual_variance": test_returns.values ** 2,
            "forecast_variance": forecasts,
        },
        index=dates,
    )
    df["forecast_vol_annualised"] = np.sqrt(df["forecast_variance"]) * np.sqrt(252) * 100
    return df


def summarise_model_performance(results_dict):
    """
    Build a summary table of out-of-sample evaluation metrics.

    Parameters
    ----------
    results_dict : dict[str, pd.DataFrame]
        Mapping of ticker → rolling forecast DataFrame (from
        ``rolling_window_forecast``).

    Returns
    -------
    pd.DataFrame
        Summary table indexed by ticker.
    """
    rows = []
    for ticker, df in results_dict.items():
        metrics = evaluate_forecast(df["actual_variance"], df["forecast_variance"])
        metrics["Ticker"] = ticker
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("Ticker")

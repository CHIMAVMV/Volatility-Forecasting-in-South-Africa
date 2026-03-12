"""
Volatility modelling using GARCH-family models (GARCH, EGARCH, GJR-GARCH)
via the `arch` library.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

# Supported model types
SUPPORTED_MODELS = ("GARCH", "EGARCH", "GJR-GARCH")


def fit_garch(
    returns,
    ticker,
    p=1,
    q=1,
    model_type="GARCH",
    dist="Normal",
    scale=100,
):
    """
    Fit a GARCH-family model to a single return series.

    Parameters
    ----------
    returns : pd.Series
        Log-return series.
    ticker : str
        Name of the asset (used for logging).
    p : int
        ARCH lag order.
    q : int
        GARCH lag order.
    model_type : str
        One of "GARCH", "EGARCH", "GJR-GARCH".
    dist : str
        Error distribution: "Normal", "StudentsT", "SkewStudent", "GED".
    scale : float
        Multiplier applied to returns before fitting (improves numerical
        stability; 100 converts to percentage returns).

    Returns
    -------
    arch.univariate.base.ARCHModelResult
        Fitted model result object.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'."
        )

    series = returns.dropna() * scale

    vol_map = {"GARCH": "Garch", "EGARCH": "EGARCH", "GJR-GARCH": "GARCH"}
    power_map = {"GARCH": 2.0, "EGARCH": 2.0, "GJR-GARCH": 2.0}
    o_map = {"GARCH": 0, "EGARCH": 0, "GJR-GARCH": 1}

    vol = vol_map[model_type]
    o = o_map[model_type]

    am = arch_model(
        series,
        vol=vol,
        p=p,
        o=o,
        q=q,
        dist=dist,
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = am.fit(disp="off", show_warning=False)

    logger.info(
        "%s [%s(%d,%d)] AIC=%.4f BIC=%.4f",
        ticker,
        model_type,
        p,
        q,
        result.aic,
        result.bic,
    )
    return result


def select_best_model(
    returns,
    ticker,
    p_range=(1, 2),
    q_range=(1, 2),
    model_types=("GARCH", "EGARCH", "GJR-GARCH"),
    dist="StudentsT",
    criterion="bic",
    scale=100,
):
    """
    Grid-search over model type, p, and q and select the best model by AIC/BIC.

    Parameters
    ----------
    returns : pd.Series
        Log-return series.
    ticker : str
        Asset name.
    p_range : tuple[int, int]
        Min and max ARCH lag to try (inclusive).
    q_range : tuple[int, int]
        Min and max GARCH lag to try (inclusive).
    model_types : tuple[str, ...]
        Model variants to evaluate.
    dist : str
        Error distribution.
    criterion : str
        "aic" or "bic".
    scale : float
        Return scaling factor.

    Returns
    -------
    tuple
        (best_result, results_df) where results_df summarises every candidate.
    """
    records = []
    best_result = None
    best_score = np.inf

    for mt in model_types:
        for p in range(p_range[0], p_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                try:
                    res = fit_garch(
                        returns, ticker, p=p, q=q,
                        model_type=mt, dist=dist, scale=scale,
                    )
                    score = res.aic if criterion == "aic" else res.bic
                    records.append(
                        {
                            "model": mt,
                            "p": p,
                            "q": q,
                            "aic": res.aic,
                            "bic": res.bic,
                            "log_likelihood": res.loglikelihood,
                        }
                    )
                    if score < best_score:
                        best_score = score
                        best_result = res
                except Exception as exc:
                    logger.warning(
                        "Failed to fit %s(%d,%d) for %s: %s",
                        mt, p, q, ticker, exc,
                    )

    results_df = pd.DataFrame(records).sort_values(criterion)
    return best_result, results_df


def forecast_volatility(result, horizon=10, scale=100):
    """
    Produce a volatility forecast from a fitted GARCH model.

    Parameters
    ----------
    result : arch.univariate.base.ARCHModelResult
        Fitted model result.
    horizon : int
        Number of steps ahead to forecast.
    scale : float
        The same scaling factor used when fitting; used to convert back.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['h.1', 'h.2', ...] containing annualised
        volatility forecasts (%).
    """
    fc = result.forecast(horizon=horizon, reindex=False)
    variance_forecast = fc.variance.iloc[-1]
    vol_forecast = np.sqrt(variance_forecast) / scale * np.sqrt(252) * 100
    return vol_forecast.to_frame(name="Annualised Vol Forecast (%)")


def fit_all_tickers(
    returns,
    model_type="GARCH",
    p=1,
    q=1,
    dist="StudentsT",
    scale=100,
):
    """
    Fit a GARCH model to every column in a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series (columns = tickers).
    model_type : str
        GARCH variant.
    p, q : int
        Lag orders.
    dist : str
        Error distribution.
    scale : float
        Return scaling factor.

    Returns
    -------
    dict[str, arch.univariate.base.ARCHModelResult]
        Mapping of ticker → fitted result.
    """
    results = {}
    for col in returns.columns:
        logger.info("Fitting %s for %s …", model_type, col)
        try:
            results[col] = fit_garch(
                returns[col], col, p=p, q=q,
                model_type=model_type, dist=dist, scale=scale,
            )
        except Exception as exc:
            logger.error("Could not fit model for %s: %s", col, exc)
    return results

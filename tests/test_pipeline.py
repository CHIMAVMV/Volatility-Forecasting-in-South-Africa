"""
Tests for the volatility forecasting pipeline.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_garch_returns():
    """Simulate a GARCH(1,1) return series (500 obs, 2 assets)."""
    np.random.seed(0)
    n = 500
    dates = pd.bdate_range("2018-01-01", periods=n)

    omega, alpha, beta = 1e-5, 0.08, 0.88
    h = np.zeros(n)
    r = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * np.random.standard_t(6)

    r2 = r * 1.1 + np.random.normal(0, 5e-4, n)
    return pd.DataFrame({"^JTOPI": r, "NPN.JO": r2}, index=dates)


@pytest.fixture(scope="module")
def synthetic_prices(synthetic_garch_returns):
    """Derive a price series from synthetic returns."""
    prices = (1 + synthetic_garch_returns).cumprod() * 100
    return prices


# ---------------------------------------------------------------------------
# data_collection
# ---------------------------------------------------------------------------

class TestDataCollection:
    def test_compute_log_returns_shape(self, synthetic_prices):
        from src.data_collection import compute_log_returns

        lr = compute_log_returns(synthetic_prices)
        # log returns drop the first row
        assert len(lr) == len(synthetic_prices) - 1
        assert list(lr.columns) == list(synthetic_prices.columns)

    def test_compute_log_returns_no_nan(self, synthetic_prices):
        from src.data_collection import compute_log_returns

        lr = compute_log_returns(synthetic_prices)
        assert not lr.isnull().any().any()

    def test_compute_log_returns_values(self, synthetic_prices):
        from src.data_collection import compute_log_returns

        lr = compute_log_returns(synthetic_prices)
        # First log return should equal log(p1/p0)
        expected = np.log(
            synthetic_prices.iloc[1] / synthetic_prices.iloc[0]
        )
        np.testing.assert_allclose(lr.iloc[0].values, expected.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_descriptive_statistics_columns(self, synthetic_garch_returns):
        from src.preprocessing import descriptive_statistics

        stats = descriptive_statistics(synthetic_garch_returns)
        expected_cols = [
            "Observations", "Mean (%)", "Std Dev (%)", "Skewness",
            "Excess Kurtosis", "Min (%)", "Max (%)", "JB Stat", "JB p-value",
        ]
        for col in expected_cols:
            assert col in stats.columns

    def test_descriptive_statistics_rows(self, synthetic_garch_returns):
        from src.preprocessing import descriptive_statistics

        stats = descriptive_statistics(synthetic_garch_returns)
        assert len(stats) == len(synthetic_garch_returns.columns)

    def test_check_stationarity_stationary(self, synthetic_garch_returns):
        from src.preprocessing import check_stationarity

        result = check_stationarity(synthetic_garch_returns)
        # GARCH returns should be stationary
        assert result["Stationary"].all()

    def test_check_arch_effects_returns_df(self, synthetic_garch_returns):
        from src.preprocessing import check_arch_effects

        result = check_arch_effects(synthetic_garch_returns)
        assert "ARCH Effects" in result.columns
        assert len(result) == len(synthetic_garch_returns.columns)

    def test_compute_realised_volatility_shape(self, synthetic_garch_returns):
        from src.preprocessing import compute_realised_volatility

        rv = compute_realised_volatility(synthetic_garch_returns, window=21)
        assert rv.shape == synthetic_garch_returns.shape

    def test_compute_realised_volatility_annualised(self, synthetic_garch_returns):
        from src.preprocessing import compute_realised_volatility

        rv_ann = compute_realised_volatility(
            synthetic_garch_returns, window=21, annualise=True
        )
        rv_raw = compute_realised_volatility(
            synthetic_garch_returns, window=21, annualise=False
        )
        # Annualised should be larger
        assert (rv_ann.dropna() > rv_raw.dropna()).all().all()


# ---------------------------------------------------------------------------
# volatility_models
# ---------------------------------------------------------------------------

class TestVolatilityModels:
    def test_fit_garch_runs(self, synthetic_garch_returns):
        from src.volatility_models import fit_garch

        res = fit_garch(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            p=1, q=1,
            model_type="GARCH",
            dist="Normal",
        )
        assert hasattr(res, "aic")
        assert hasattr(res, "bic")
        assert np.isfinite(res.aic)

    def test_fit_egarch_runs(self, synthetic_garch_returns):
        from src.volatility_models import fit_garch

        res = fit_garch(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            p=1, q=1,
            model_type="EGARCH",
            dist="Normal",
        )
        assert np.isfinite(res.aic)

    def test_fit_gjr_garch_runs(self, synthetic_garch_returns):
        from src.volatility_models import fit_garch

        res = fit_garch(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            p=1, q=1,
            model_type="GJR-GARCH",
            dist="Normal",
        )
        assert np.isfinite(res.aic)

    def test_invalid_model_type_raises(self, synthetic_garch_returns):
        from src.volatility_models import fit_garch

        with pytest.raises(ValueError, match="model_type"):
            fit_garch(
                synthetic_garch_returns["^JTOPI"],
                ticker="^JTOPI",
                model_type="UNKNOWN",
            )

    def test_forecast_volatility_shape(self, synthetic_garch_returns):
        from src.volatility_models import fit_garch, forecast_volatility

        res = fit_garch(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            p=1, q=1,
            model_type="GARCH",
            dist="Normal",
        )
        fc = forecast_volatility(res, horizon=5)
        assert len(fc) == 5

    def test_forecast_volatility_positive(self, synthetic_garch_returns):
        from src.volatility_models import fit_garch, forecast_volatility

        res = fit_garch(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            p=1, q=1,
            model_type="GARCH",
            dist="Normal",
        )
        fc = forecast_volatility(res, horizon=5)
        assert (fc.values > 0).all()

    def test_select_best_model(self, synthetic_garch_returns):
        from src.volatility_models import select_best_model

        best, df = select_best_model(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            p_range=(1, 1),
            q_range=(1, 1),
            model_types=("GARCH", "GJR-GARCH"),
            dist="Normal",
            criterion="bic",
        )
        assert best is not None
        assert "bic" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_rolling_window_forecast_shape(self, synthetic_garch_returns):
        from src.evaluation import rolling_window_forecast

        oos = rolling_window_forecast(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            model_type="GARCH",
            dist="Normal",
            train_size=0.9,
        )
        expected_rows = int(len(synthetic_garch_returns) * 0.1)
        # Allow ±2 due to rounding
        assert abs(len(oos) - expected_rows) <= 2

    def test_rolling_window_forecast_columns(self, synthetic_garch_returns):
        from src.evaluation import rolling_window_forecast

        oos = rolling_window_forecast(
            synthetic_garch_returns["^JTOPI"],
            ticker="^JTOPI",
            model_type="GARCH",
            dist="Normal",
            train_size=0.9,
        )
        for col in ("actual_return", "actual_variance", "forecast_variance",
                    "forecast_vol_annualised"):
            assert col in oos.columns

    def test_evaluate_forecast_metrics(self):
        from src.evaluation import evaluate_forecast

        actual = np.array([0.01, 0.02, 0.015, 0.025])
        predicted = np.array([0.011, 0.019, 0.016, 0.024])
        metrics = evaluate_forecast(actual, predicted)
        for key in ("MSE", "RMSE", "MAE", "QLIKE"):
            assert key in metrics
            assert np.isfinite(metrics[key])

    def test_rmse_non_negative(self):
        from src.evaluation import evaluate_forecast

        actual = np.array([0.01, 0.02, 0.015])
        predicted = np.array([0.012, 0.018, 0.014])
        metrics = evaluate_forecast(actual, predicted)
        assert metrics["RMSE"] >= 0
        assert metrics["MAE"] >= 0
        assert metrics["MSE"] >= 0

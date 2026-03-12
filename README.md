# Volatility Forecasting in South Africa Stock Market

This project implements a complete **volatility-forecasting pipeline** for equities listed on the
**Johannesburg Stock Exchange (JSE)** using GARCH-family models.

---

## Project Structure

```
├── src/
│   ├── data_collection.py   # Fetch JSE price data via yfinance
│   ├── preprocessing.py     # Descriptive stats, ADF test, ARCH-LM test
│   ├── volatility_models.py # GARCH, EGARCH, GJR-GARCH fitting & selection
│   ├── evaluation.py        # Rolling-window OOS forecast & loss metrics
│   └── visualisation.py     # All plotting utilities
├── notebooks/
│   └── Volatility_Forecasting_JSE.ipynb  # Interactive analysis notebook
├── outputs/
│   ├── plots/               # Generated figures
│   └── results/             # CSV outputs (stats, forecasts, metrics)
├── volatility_forecasting.py  # Main end-to-end analysis script
└── requirements.txt
```

---

## Covered Assets

| Ticker    | Name                  |
|-----------|-----------------------|
| `^JTOPI`  | JSE Top 40 Index      |
| `NPN.JO`  | Naspers               |
| `SBK.JO`  | Standard Bank         |
| `MTN.JO`  | MTN Group             |
| `AGL.JO`  | Anglo American        |
| `BHG.JO`  | BHP Group             |
| `FSR.JO`  | FirstRand             |
| `ABG.JO`  | Absa Group            |
| `SOL.JO`  | Sasol                 |
| `CFR.JO`  | Richemont             |
| `IMP.JO`  | Impala Platinum       |

---

## Methodology

1. **Data** – Daily adjusted closing prices downloaded from Yahoo Finance (2015 – present).
2. **Returns** – Daily log returns: $r_t = \ln(P_t / P_{t-1})$.
3. **EDA** – Descriptive statistics, normality tests (Jarque-Bera), ACF/PACF of squared returns.
4. **Stationarity** – Augmented Dickey-Fuller (ADF) test.
5. **ARCH effects** – Engle's ARCH-LM test.
6. **Models** – GARCH(p,q), EGARCH(p,q), GJR-GARCH(p,q) with Student-t innovations.
7. **Model selection** – Grid search over p,q ∈ {1,2}; best model chosen by BIC.
8. **Forecasting** – Multi-step ahead volatility forecasts (annualised %).
9. **Evaluation** – Rolling-window out-of-sample forecasts; MSE, RMSE, MAE, QLIKE metrics.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full analysis script

```bash
python volatility_forecasting.py
```

Plots are saved to `outputs/plots/` and CSV results to `outputs/results/`.

### 3. Interactive notebook

```bash
jupyter lab notebooks/Volatility_Forecasting_JSE.ipynb
```

---

## Key Findings

* JSE return series exhibit **fat tails** (excess kurtosis > 3) and **negative skewness**,
  rejecting the normality assumption.
* The **ARCH-LM test** confirms significant time-varying volatility (heteroskedasticity) in all series.
* **GJR-GARCH** with Student-t innovations captures the leverage effect and generally achieves
  the lowest BIC among the candidate models.
* Conditional volatility spiked significantly during the **COVID-19 shock (2020)** and during
  subsequent global macro stress episodes.

---

## Dependencies

See `requirements.txt`.  Key packages:

| Package       | Purpose                        |
|---------------|--------------------------------|
| `arch`        | GARCH model estimation         |
| `yfinance`    | Market data download           |
| `statsmodels` | ADF test, ARCH-LM test, ACF    |
| `pandas`      | Data manipulation              |
| `numpy`       | Numerical computing            |
| `matplotlib`  | Plotting                       |
| `seaborn`     | Statistical visualisations     |
| `scipy`       | Jarque-Bera test               |


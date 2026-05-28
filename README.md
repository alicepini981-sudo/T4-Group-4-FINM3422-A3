# Overview
A prototype risk and derivatives modelling platform that constructs yield curves, prices equity options using Black-Scholes, Binomial Trees, and Monte Carlo simulation, and computes portfolio-level risk metrics including VaR and scenario analysis.

# Installation

1. Clone the repo:
git clone https://github.com/alicepini981-sudo/T4-Group-4-FINM3422-A3

2. Install dependencies:
pip install -r requirements.txt

# Repo Structure

T4-Group-4-FINM3422-A3/
├── data/
│   ├── Yield_Data.csv
│   ├── Market_Data_BHP_prices.csv
│   ├── Market_Data_CBA_prices.csv
│   ├── Market_Data_CSL_prices.csv
│   └── Market_Data_WOW_prices.csv
├── notebooks/
│   └── trading_desk_analysis.ipynb
├── src/
│   ├── yieldcurve.py
│   ├── derivatives.py
│   ├── portfolio.py
│   └── dataloader.py
├── README.md
└── AI_USAGE.md

# Modules

## yieldcurve.py
Constructs a yield curve from RBA government bond yield data (Table F2)
- get_zero_rate(T) — returns the interpolated zero rate for a given maturity
- get_discount_factor(T) — returns the discount factor for a given maturity using continuous compounding
- plot() — generates a plot of the resulting yield curve
- load_yield_data() — loads and cleans the RBA yield data from CSV

## derivatives.py
Derivative pricing engine using Object-Oriented Programming
- Derivative — base class with common attributes (spot price, strike, maturity, volatility, yield curve reference)
- EuropeanCall / EuropeanPut — prices European options via Black-Scholes closed-form
- EuropeanCallGreeks / EuropeanPutGreeks — extends European classes with analytical Greeks (delta, gamma, theta, vega, rho)
- AmericanPut — prices American puts via a 100-step CRR binomial tree with early exercise
- AsianCall / AsianPut — prices Asian options via Monte Carlo simulation (10,000 paths)
- All subclasses integrate with yieldcurve.py for discount rates

## portfolio.py
Portfolio construction and risk analysis
- EquityPosition — simple equity position with price and delta
 - Portfolio — aggregates equity and option positions, computes:
  - Portfolio value and portfolio delta
  - Historical VaR (non-parametric, from observed returns)
  - Parametric VaR (normal distribution assumption)
 - Scenario analysis and portfolio Greeks are computed via helper functions

## dataloader.py
### Equity data pipeline
 - Loads 2-year daily price history for BHP, CBA, CSL, and WOW
 - Caches data locally as CSV for reproducibility
 - Downloads from Yahoo Finance (yfinance) if local cache is missing

# Dependencies
 Install all dependencies with:
 pip install -r requirements.txt

## Key Packages
numpy, scipy, pandas, matplotlib, jupyter, yfinance

# Data
Data, Source and Details 
Risk-free rates | RBA Table F2 — Capital Market Yields, Government Bonds | April 2026, 40 data points, quarterly 0.25–10yr 
Equity prices | Yahoo Finance (via yfinance API) | 2-year daily history, ~508 trading days per stock |
Tickers | BHP.AX, CBA.AX, CSL.AX, WOW.AX | Materials, Financials, Healthcare, Consumer Staples |

All data is cached locally in the data/ directory so the notebook runs without an internet connection.

# AI Usage
See AI_USAGE.md for a full log of AI tools used, what they were used for, and how all AI-generated code was validated and tested.
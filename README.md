# Overview
### A prototype risk and derivatives modelling platform that constructs yield curves, prices equity options using Black-Scholes, and computes portfolio-level risk metrics.

# Installation

### 1. Clone the repo:git clone https://github.com/alicepini981-sudo/T4-Group-4-FINM3422-A3

### 2. Install dependencies: pip install - r requirements.txt

# Repo Structure

### - data/
### - notebooks/
###   main_report.ipynb
### - src/
###   yieldcurve.py
###   derivatives.py
###   portfolio.py
### - README.md
### - AI_USAGE.md

# Modules

## yieldcurve.py
### Constructs a yield curve form interest rate/government bond yield data
### get_zero_rate - returns the discount factor for a given maturity
### get_discount_factor methods - returns the zero rate for a given maturity 
### Generates a plot of the resulting yield curve

## derivatives.py
### Derivative - base class with common attributes (underlying variable price, strike, maturity, volatility and risk-free rate)
### EuropeanCall - prices a European call option via Black-Scholes
### EuropeanPut - prices a European put option via Black-Scholes
### Integrates with yieldcurve.py for discount rates

## portfolio.py
### Defines a sample portfolio of equity and option positions and computes risk metrics:
### Portfolio value under user-defined scenarios
### 1-day or 10-day Value-at-Risk (VaR)

# Dependencies
### Install all dependencies with pip install -r requirements.txt

## Key Packages
### numpy, scipy, pandas, matplotlib, jupyter

# Data
# TODO: add data sourcing

# AI Usage
### See AI_Usage.md for a full log of AI tools used, what they were used for, and how all AI-generated code was validated and tested.
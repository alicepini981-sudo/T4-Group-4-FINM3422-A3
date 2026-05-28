import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EquityPosition:
    """Simple equity object for the portfolio layer."""

    def __init__(self, ticker, spot):
        self.ticker = ticker
        self.spot = spot

    def price(self):
        return self.spot

    def delta(self):
        return 1.0


class Portfolio:

    def __init__(self):
        self.positions = []

    def add_position(self, instrument, quantity, label=None):
        self.positions.append({"instrument": instrument, "quantity": quantity, "label": label})

    def value(self):
        return sum(p["quantity"] * p["instrument"].price() for p in self.positions)

    def delta(self):
        return sum(p["quantity"] * p["instrument"].delta() for p in self.positions)

    def position_table(self):
        rows = []
        for p in self.positions:
            inst = p["instrument"]
            qty  = p["quantity"]
            name = p["label"] or (inst.ticker if hasattr(inst, "ticker") else inst.__class__.__name__)
            rows.append({
                "Position":       name,
                "Quantity":       qty,
                "Unit Value":     inst.price(),
                "Position Value": qty * inst.price(),
                "Unit Delta":     inst.delta(),
                "Position Delta": qty * inst.delta(),
            })
        df = pd.DataFrame(rows)
        if len(df) > 0:
            total = pd.DataFrame([{
                "Position": "TOTAL", "Quantity": np.nan, "Unit Value": np.nan,
                "Position Value": df["Position Value"].sum(),
                "Unit Delta": np.nan, "Position Delta": df["Position Delta"].sum()
            }])
            df = pd.concat([df, total], ignore_index=True)
        return df

    def historical_var(self, returns, alpha=0.95, horizon_days=1):
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            raise ValueError("Return series is empty.")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")
        q = (returns * np.sqrt(horizon_days)).quantile(1 - alpha)
        return max(-q * self.value(), 0)

    def parametric_var(self, sigma_portfolio, alpha=0.95, horizon_days=1):
        from scipy.stats import norm
        return norm.ppf(alpha) * sigma_portfolio * np.sqrt(horizon_days) * self.value()


def build_portfolio(equity_params, instruments):
    """Equal dollar weighting: $100,000 total, 25% per stock."""
    quantities = {
        name: int((100000 * 0.25) / equity_params[name]["S0"])
        for name in equity_params
    }
    portfolio = Portfolio()
    for name, inst in instruments.items():
        portfolio.add_position(inst["equity"], quantity=quantities[name], label=f"Long {name} Equity")
        portfolio.add_position(inst["call"],   quantity=10,               label=f"Long {name} Call")
        portfolio.add_position(inst["put"],    quantity=-5,               label=f"Short {name} Put")
    return portfolio, quantities


def compute_var(equity_params, instruments, quantities, all_data, portfolio, portfolio_value):
    """Compute historical and parametric VaR at 95% and 90% confidence."""
    base_index = all_data["BHP"]["Close"].pct_change().dropna().index
    portfolio_returns = sum(
        quantities[name] * equity_params[name]["S0"] *
        all_data[name]["Close"].pct_change().dropna().reindex(base_index, fill_value=0)
        for name in instruments
    ) / portfolio_value
    sigma = portfolio_returns.std()
    return {
        "returns":  portfolio_returns,
        "hist_95":  portfolio.historical_var(portfolio_returns, alpha=0.95, horizon_days=1),
        "hist_90":  portfolio.historical_var(portfolio_returns, alpha=0.90, horizon_days=1),
        "param_95": portfolio.parametric_var(sigma, alpha=0.95, horizon_days=1),
        "param_90": portfolio.parametric_var(sigma, alpha=0.90, horizon_days=1),
    }


def compute_portfolio_greeks(instruments, portfolio):
    """Compute aggregated portfolio Greeks across all positions."""
    def agg(greek):
        return sum(
            10 * instruments[name]["call_greeks"].all_greeks()[greek]
            + (-5) * getattr(instruments[name]["put"], greek)()
            for name in instruments
        )
    return {
        "delta": portfolio.delta(),
        "gamma": agg("gamma"),
        "vega":  agg("vega"),
        "theta": agg("theta"),
        "rho":   agg("rho"),
    }


def shift_curve(base_curve, rate_shift):
    """Parallel shift a yield curve by rate_shift."""
    from yieldcurve import YieldCurve
    return YieldCurve(maturities=base_curve.maturities,
                      zero_rates=[r + rate_shift for r in base_curve.zero_rates])


def build_scenario_portfolio(instruments, quantities, base_curve, price_shock=0.0, rate_shift=0.0):
    """Rebuild the full portfolio under shocked price and rate inputs."""
    from derivatives import EuropeanCall, EuropeanPut
    shocked_curve = shift_curve(base_curve, rate_shift)
    shocked_portfolio = Portfolio()
    for name, inst in instruments.items():
        shocked_equity = EquityPosition(ticker=name, spot=inst["equity"].spot * (1 + price_shock))
        shocked_call = EuropeanCall(S0=inst["call"].S0 * (1 + price_shock), K=inst["call"].K,
                                    T=inst["call"].T, sigma=inst["call"].sigma, yield_curve=shocked_curve)
        shocked_put  = EuropeanPut(S0=inst["put"].S0  * (1 + price_shock), K=inst["put"].K,
                                    T=inst["put"].T,  sigma=inst["put"].sigma,  yield_curve=shocked_curve)
        shocked_portfolio.add_position(shocked_equity, quantity=quantities[name], label=f"Long {name} Equity")
        shocked_portfolio.add_position(shocked_call,   quantity=10,               label=f"Long {name} Call")
        shocked_portfolio.add_position(shocked_put,    quantity=-5,               label=f"Short {name} Put")
    return shocked_portfolio


def run_scenarios(instruments, quantities, base_curve, base_value):
    """Run standard scenario analysis and return a list of result dicts."""
    scenarios = [
        {"Scenario": "Underlying +5%", "price_shock":  0.05, "rate_shift":  0.0},
        {"Scenario": "Underlying -5%", "price_shock": -0.05, "rate_shift":  0.0},
        {"Scenario": "Rates +50bps",   "price_shock":  0.0,  "rate_shift":  0.005},
        {"Scenario": "Rates -50bps",   "price_shock":  0.0,  "rate_shift": -0.005},
    ]
    results = []
    for s in scenarios:
        sp = build_scenario_portfolio(instruments, quantities, base_curve,
                                      price_shock=s["price_shock"], rate_shift=s["rate_shift"])
        shocked_value = sp.value()
        pnl = shocked_value - base_value
        results.append({"Scenario": s["Scenario"], "Base Value": base_value,
                        "Shocked Value": shocked_value, "P&L": pnl,
                        "% Change": pnl / base_value if base_value != 0 else np.nan})
    return results


def plot_sensitivity(sens, vols, maturities_range):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name, d in sens.items():
        axes[0, 0].plot(vols,             d["call_v"], label=name)
        axes[0, 1].plot(vols,             d["put_v"],  label=name)
        axes[1, 0].plot(maturities_range, d["call_t"], label=name)
        axes[1, 1].plot(maturities_range, d["put_t"],  label=name)
    for ax, title, xlabel, ylabel in [
        (axes[0, 0], "Call Price vs. Volatility",  "Volatility",       "Call Price"),
        (axes[0, 1], "Put Price vs. Volatility",   "Volatility",       "Put Price"),
        (axes[1, 0], "Call Price vs. Maturity",    "Maturity (Years)", "Call Price"),
        (axes[1, 1], "Put Price vs. Maturity",     "Maturity (Years)", "Put Price"),
    ]:
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_var_distribution(portfolio_returns, portfolio_value, hist_var_95, hist_var_90):
    """
    Plot histogram of portfolio returns with VaR thresholds 
    and normal distribution overlay.
    """
    from scipy.stats import norm
 
    fig, ax = plt.subplots(figsize=(12, 6))
 
    returns = portfolio_returns.dropna()
 
    ax.hist(returns, bins=50, color='steelblue', alpha=0.7,
            edgecolor='white', label='Daily Portfolio Returns')
 
    ax.axvline(-hist_var_95 / portfolio_value, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'95% Historical VaR (${hist_var_95:,.0f})')
    ax.axvline(-hist_var_90 / portfolio_value, color='#e67e22', linestyle='--', linewidth=2,
               label=f'90% Historical VaR (${hist_var_90:,.0f})')
 
    x_range = np.linspace(returns.min(), returns.max(), 200)
    mu = returns.mean()
    sigma_port = returns.std()
    normal_curve = norm.pdf(x_range, mu, sigma_port)
    bin_width = (returns.max() - returns.min()) / 50
    scale = len(returns) * bin_width
    ax.plot(x_range, normal_curve * scale, color='black', linewidth=1.5,
            linestyle='-', label='Normal Distribution Fit')
 
    ax.set_xlabel('Daily Portfolio Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Portfolio Return Distribution with VaR Thresholds', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
 
def plot_scenarios(scenario_df):
    """
    Plot horizontal bar chart of scenario P&L results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
 
    scenarios = scenario_df['Scenario'].values
    pnl_values = scenario_df['P&L'].values
    colors = ['#27ae60' if x >= 0 else '#e74c3c' for x in pnl_values]
 
    bars = ax.barh(scenarios, pnl_values, color=colors, edgecolor='white', height=0.5)
 
    for bar, val in zip(bars, pnl_values):
        label = f'+${val:,.0f}' if val >= 0 else f'-${abs(val):,.0f}'
        x_pos = val + (200 if val >= 0 else -200)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                va='center', ha=ha, fontsize=11, fontweight='bold')
 
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('P&L Impact ($)', fontsize=12)
    ax.set_title('Scenario Analysis — Portfolio P&L Under Hypothetical Shocks', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
 
 
def plot_portfolio_composition(quantities, equity_params, instruments, portfolio_value):
    """
    Plot portfolio allocation pie chart and position type breakdown.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 
    stock_names = list(quantities.keys())
    equity_values = [quantities[n] * equity_params[n]["S0"] for n in stock_names]
    colors_pie = ['#2c3e50', '#e67e22', '#27ae60', '#e74c3c']
 
    wedges, texts, autotexts = ax1.pie(
        equity_values, labels=stock_names, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    ax1.set_title('Equity Allocation by Stock', fontsize=13)
 
    equity_total = sum(equity_values)
    call_total = sum(10 * instruments[n]["call"].price() for n in instruments)
    put_total = sum(5 * instruments[n]["put"].price() for n in instruments)
 
    categories = ['Equity\nPositions', 'Long Calls\n(10 per stock)', 'Short Puts\n(-5 per stock)']
    values = [equity_total, call_total, put_total]
    bar_colors = ['#2c3e50', '#27ae60', '#e74c3c']
 
    bars = ax2.bar(categories, values, color=bar_colors, edgecolor='white', width=0.5)
    display_values = [equity_total, call_total, -put_total]
    for bar, val in zip(bars, display_values):
        label = f'${val:,.0f}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, label,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
 
    ax2.set_ylabel('Value ($)', fontsize=12)
    ax2.set_title('Portfolio Value by Position Type', fontsize=13)
    ax2.grid(True, axis='y', alpha=0.3)
 
    plt.suptitle(f'Portfolio Composition — ${portfolio_value:,.0f} Total', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
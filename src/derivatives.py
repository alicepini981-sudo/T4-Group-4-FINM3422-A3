import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class Derivative:
    """Base for all derivative instruments."""

    def __init__(self, S0, K, T, sigma, yield_curve):
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.yield_curve = yield_curve

    def price(self):
        raise NotImplementedError("Pricing logic must be implemented in the subclass.")

    def delta(self):
        raise NotImplementedError("Delta calculation must be implemented in the subclass.")


class EuropeanCall(Derivative):
    """European Call Option priced using the Black-Scholes formula."""

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (np.log(self.S0 / self.K) + (r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S0 * norm.cdf(d1) - self.K * np.exp(-r * self.T) * norm.cdf(d2)

    def delta(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (np.log(self.S0 / self.K) + (r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.cdf(d1)


class EuropeanPut(Derivative):
    """European Put Option priced using the Black-Scholes formula."""

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (np.log(self.S0 / self.K) + (r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.K * np.exp(-r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

    def delta(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (np.log(self.S0 / self.K) + (r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.cdf(d1) - 1


# Aliases for backwards compatibility
EuropeanCallOption = EuropeanCall
EuropeanPutOption  = EuropeanPut

def build_options(equity_params, T, curve):
    """Build EuropeanCall and EuropeanPut objects for each ticker."""
    return {
        name: {
            "call": EuropeanCall(S0=p["S0"], K=p["K"], T=T, sigma=p["sigma"], yield_curve=curve),
            "put":  EuropeanPut(S0=p["S0"],  K=p["K"], T=T, sigma=p["sigma"], yield_curve=curve),
            **p
        }
        for name, p in equity_params.items()
    }


def check_put_call_parity(equity_params, options, T, curve):
    """Returns parity results as a list of dicts."""
    r = curve.get_zero_rate(T)
    results = []
    for name, params in equity_params.items():
        lhs = options[name]["call"].price() - options[name]["put"].price()
        rhs = params["S0"] - params["K"] * np.exp(-r * T)
        results.append({"name": name, "lhs": lhs, "rhs": rhs, "residual": abs(lhs - rhs)})
    return results


def asian_call_confidence_intervals(equity_params, T, curve, n=10000, steps=252):
    """Returns Monte Carlo price, std error, and 95% CI for each ticker."""
    r = curve.get_zero_rate(T)
    dt = T / steps
    results = []
    for name, params in equity_params.items():
        np.random.seed(42)
        Z = np.random.standard_normal((n, steps))
        S = np.zeros((n, steps + 1))
        S[:, 0] = params["S0"]
        for t in range(1, steps + 1):
            S[:, t] = S[:, t-1] * np.exp(
                (r - 0.5 * params["sigma"]**2) * dt
                + params["sigma"] * np.sqrt(dt) * Z[:, t-1]
            )
        discounted = np.exp(-r * T) * np.maximum(S[:, 1:].mean(axis=1) - params["K"], 0)
        price, se = discounted.mean(), discounted.std() / np.sqrt(n)
        results.append({"name": name, "price": price, "se": se,
                        "ci_lower": price - 1.96 * se, "ci_upper": price + 1.96 * se})
    return results


def compute_sensitivity_data(equity_params, T, curve, vols=None, maturities=None):
    """Compute call/put prices across vol and maturity ranges for all tickers."""
    if vols is None:
        vols = np.linspace(0.1, 0.4, 10)
    if maturities is None:
        maturities = np.linspace(0.25, 3.0, 10)
    results = {}
    for name, params in equity_params.items():
        results[name] = {
            "call_v": [EuropeanCall(S0=params["S0"], K=params["K"], T=T, sigma=v, yield_curve=curve).price() for v in vols],
            "put_v":  [EuropeanPut(S0=params["S0"],  K=params["K"], T=T, sigma=v, yield_curve=curve).price() for v in vols],
            "call_t": [EuropeanCall(S0=params["S0"], K=params["K"], T=t, sigma=params["sigma"], yield_curve=curve).price() for t in maturities],
            "put_t":  [EuropeanPut(S0=params["S0"],  K=params["K"], T=t, sigma=params["sigma"], yield_curve=curve).price() for t in maturities],
        }
    return results, vols, maturities


def print_european_call_greeks(option):
    """Print a formatted Greeks table for a EuropeanCallGreeks instance."""
    greeks = option.all_greeks()
    print("=" * 45)
    print("       EUROPEAN CALL OPTION - GREEKS")
    print("=" * 45)
    for name, value, note in [
        ("Delta", greeks["delta"], "price change per $1 move in S0"),
        ("Gamma", greeks["gamma"], "rate of change of delta"),
        ("Theta", greeks["theta"], "price decay per calendar day"),
        ("Vega",  greeks["vega"],  "price change per 1% vol move"),
        ("Rho",   greeks["rho"],   "price change per 1% rate move"),
    ]:
        print(f"  {name:<10} {value:>10.4f}   ({note})")
    print("=" * 45)


def print_american_put(option):
    """Print a formatted price + Greeks table for an AmericanPut instance."""
    print("=" * 45)
    print("       AMERICAN PUT OPTION - PRICE")
    print("=" * 45)
    print(f"  {'Binomial Steps':<20} {option.steps:>10}")
    print(f"  {'Price':<20} ${option.price():>9.4f}")
    print(f"  {'Delta':<20} {option.delta():>10.4f}")
    print(f"  {'Gamma':<20} {option.gamma():>10.4f}")
    print(f"  {'Theta':<20} {option.theta():>10.4f}")
    print(f"  {'Vega':<20} {option.vega():>10.4f}")
    print("=" * 45)


def print_asian_call(option):
    """Print a formatted price + Greeks table for an AsianCall instance."""
    print("=" * 45)
    print("       ASIAN CALL OPTION - MONTE CARLO")
    print("=" * 45)
    print(f"  {'Simulations':<20} {option.simulations:>10,}")
    print(f"  {'Steps':<20} {option.steps:>10}")
    print(f"  {'Price':<20} ${option.price():>9.4f}")
    print(f"  {'Delta':<20} {option.delta():>10.4f}")
    print(f"  {'Gamma':<20} {option.gamma():>10.4f}")
    print(f"  {'Theta':<20} {option.theta():>10.4f}")
    print(f"  {'Vega':<20} {option.vega():>10.4f}")
    print("=" * 45)


# -------------------------------------------------------
# AMERICAN OPTIONS - BINOMIAL TREE
# -------------------------------------------------------

class AmericanCall(Derivative):
    """American Call Option priced using the Binomial Tree model."""

    def __init__(self, S0, K, T, sigma, yield_curve, steps=100):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.steps = steps

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        asset_prices = np.array([self.S0 * (u ** (self.steps - 2 * j)) for j in range(self.steps + 1)])
        option_values = np.maximum(asset_prices - self.K, 0)
        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([self.S0 * (u ** (i - 2 * j)) for j in range(i + 1)])
            option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            option_values = np.maximum(option_values, asset_prices - self.K)
        return float(option_values[0])

    def delta(self):
        h = self.S0 * 0.01
        up   = AmericanCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up   = AmericanCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        mid  = AmericanCall(self.S0,     self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - 2 * mid.price() + down.price()) / (h ** 2)

    def theta(self):
        h = 1 / 365
        if self.T <= h:
            return 0.0
        fwd = AmericanCall(self.S0, self.K, self.T - h, self.sigma, self.yield_curve, self.steps)
        return fwd.price() - self.price()

    def vega(self):
        h = 0.01
        up   = AmericanCall(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.steps)
        down = AmericanCall(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-(r + h) * dt)
        p = (np.exp((r + h) * dt) - d) / (u - d)
        asset_prices = np.array([self.S0 * (u ** (self.steps - 2 * j)) for j in range(self.steps + 1)])
        option_values = np.maximum(asset_prices - self.K, 0)
        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([self.S0 * (u ** (i - 2 * j)) for j in range(i + 1)])
            option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            option_values = np.maximum(option_values, asset_prices - self.K)
        return (float(option_values[0]) - self.price()) / h


class AmericanPut(Derivative):
    """American Put Option priced using the Binomial Tree model."""

    def __init__(self, S0, K, T, sigma, yield_curve, steps=100):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.steps = steps

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        asset_prices = np.array([self.S0 * (u ** (self.steps - 2 * j)) for j in range(self.steps + 1)])
        option_values = np.maximum(self.K - asset_prices, 0)
        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([self.S0 * (u ** (i - 2 * j)) for j in range(i + 1)])
            option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            option_values = np.maximum(option_values, self.K - asset_prices)
        return float(option_values[0])

    def delta(self):
        h = self.S0 * 0.01
        up   = AmericanPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanPut(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up   = AmericanPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        mid  = AmericanPut(self.S0,     self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanPut(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - 2 * mid.price() + down.price()) / (h ** 2)

    def theta(self):
        h = 1 / 365
        if self.T <= h:
            return 0.0
        fwd = AmericanPut(self.S0, self.K, self.T - h, self.sigma, self.yield_curve, self.steps)
        return fwd.price() - self.price()

    def vega(self):
        h = 0.01
        up   = AmericanPut(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.steps)
        down = AmericanPut(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-(r + h) * dt)
        p = (np.exp((r + h) * dt) - d) / (u - d)
        asset_prices = np.array([self.S0 * (u ** (self.steps - 2 * j)) for j in range(self.steps + 1)])
        option_values = np.maximum(self.K - asset_prices, 0)
        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([self.S0 * (u ** (i - 2 * j)) for j in range(i + 1)])
            option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            option_values = np.maximum(option_values, self.K - asset_prices)
        return (float(option_values[0]) - self.price()) / h


# -------------------------------------------------------
# ASIAN OPTIONS - MONTE CARLO
# -------------------------------------------------------

class AsianCall(Derivative):
    """Asian Call Option priced using Monte Carlo simulation."""

    def __init__(self, S0, K, T, sigma, yield_curve, simulations=10000, steps=252):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.simulations = simulations
        self.steps = steps

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(np.cumsum(
            (r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z, axis=1))
        return float(np.exp(-r * self.T) * np.mean(np.maximum(np.mean(price_paths, axis=1) - self.K, 0)))

    def delta(self):
        h = self.S0 * 0.01
        up   = AsianCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        down = AsianCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up   = AsianCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        mid  = AsianCall(self.S0,     self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        down = AsianCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        return (up.price() - 2 * mid.price() + down.price()) / (h ** 2)

    def theta(self):
        h = 1 / 365
        if self.T <= h:
            return 0.0
        fwd = AsianCall(self.S0, self.K, self.T - h, self.sigma, self.yield_curve, self.simulations, self.steps)
        return fwd.price() - self.price()

    def vega(self):
        h = 0.01
        up   = AsianCall(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.simulations, self.steps)
        down = AsianCall(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(np.cumsum(
            ((r + h) - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z, axis=1))
        price_up = float(np.exp(-(r + h) * self.T) * np.mean(np.maximum(np.mean(price_paths, axis=1) - self.K, 0)))
        return (price_up - self.price()) / h


class AsianPut(Derivative):
    """Asian Put Option priced using Monte Carlo simulation."""

    def __init__(self, S0, K, T, sigma, yield_curve, simulations=10000, steps=252):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.simulations = simulations
        self.steps = steps

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(np.cumsum(
            (r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z, axis=1))
        return float(np.exp(-r * self.T) * np.mean(np.maximum(self.K - np.mean(price_paths, axis=1), 0)))

    def delta(self):
        h = self.S0 * 0.01
        up   = AsianPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        down = AsianPut(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up   = AsianPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        mid  = AsianPut(self.S0,     self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        down = AsianPut(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        return (up.price() - 2 * mid.price() + down.price()) / (h ** 2)

    def theta(self):
        h = 1 / 365
        if self.T <= h:
            return 0.0
        fwd = AsianPut(self.S0, self.K, self.T - h, self.sigma, self.yield_curve, self.simulations, self.steps)
        return fwd.price() - self.price()

    def vega(self):
        h = 0.01
        up   = AsianPut(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.simulations, self.steps)
        down = AsianPut(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(np.cumsum(
            ((r + h) - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z, axis=1))
        price_up = float(np.exp(-(r + h) * self.T) * np.mean(np.maximum(self.K - np.mean(price_paths, axis=1), 0)))
        return (price_up - self.price()) / h


# -------------------------------------------------------
# GREEKS FOR EUROPEAN OPTIONS (Closed-Form)
# -------------------------------------------------------

class EuropeanCallGreeks(EuropeanCall):
    """Extends EuropeanCall with full closed-form Greeks."""

    def _d1_d2(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (np.log(self.S0 / self.K) + (r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1, d1 - self.sigma * np.sqrt(self.T)

    def delta(self):
        d1, _ = self._d1_d2()
        return norm.cdf(d1)

    def gamma(self):
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def theta(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1, d2 = self._d1_d2()
        return (-(self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                - r * self.K * np.exp(-r * self.T) * norm.cdf(d2)) / 365

    def vega(self):
        d1, _ = self._d1_d2()
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def rho(self):
        r = self.yield_curve.get_zero_rate(self.T)
        _, d2 = self._d1_d2()
        return self.K * self.T * np.exp(-r * self.T) * norm.cdf(d2) * 0.01

    def all_greeks(self):
        return {"delta": self.delta(), "gamma": self.gamma(),
                "theta": self.theta(), "vega": self.vega(), "rho": self.rho()}


class EuropeanPutGreeks(EuropeanPut):
    """Extends EuropeanPut with full closed-form Greeks."""

    def _d1_d2(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (np.log(self.S0 / self.K) + (r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1, d1 - self.sigma * np.sqrt(self.T)

    def delta(self):
        d1, _ = self._d1_d2()
        return norm.cdf(d1) - 1

    def gamma(self):
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def theta(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1, d2 = self._d1_d2()
        return (-(self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                + r * self.K * np.exp(-r * self.T) * norm.cdf(-d2)) / 365

    def vega(self):
        d1, _ = self._d1_d2()
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def rho(self):
        r = self.yield_curve.get_zero_rate(self.T)
        _, d2 = self._d1_d2()
        return -self.K * self.T * np.exp(-r * self.T) * norm.cdf(-d2) * 0.01

    def all_greeks(self):
        return {"delta": self.delta(), "gamma": self.gamma(),
                "theta": self.theta(), "vega": self.vega(), "rho": self.rho()}

def plot_greeks_comparison(instruments):
    """
    Plot grouped bar chart comparing European Call Greeks across all stocks.
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
 
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    stock_names = list(instruments.keys())
    bar_colors = ['#2c3e50', '#e67e22', '#27ae60', '#e74c3c']
 
    greek_data = {}
    for name in stock_names:
        g = instruments[name]["call_greeks"].all_greeks()
        greek_data[name] = [g['delta'], g['gamma'], g['theta'], g['vega'], g['rho']]
 
    x = np.arange(len(stock_names))
    for i, (ax, greek) in enumerate(zip(axes, greek_names)):
        values = [greek_data[n][i] for n in stock_names]
        bars = ax.bar(x, values, color=bar_colors, width=0.6, edgecolor='white')
        ax.set_title(greek, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stock_names, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
 
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
 
    plt.suptitle('European Call Greeks — Cross-Stock Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
 
 
def plot_price_history(all_data):
    """
    Plot 2-year price history for all equities.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    stock_colors = {'BHP': '#2c3e50', 'CBA': '#e67e22', 'CSL': '#27ae60', 'WOW': '#e74c3c'}
 
    for ax, (name, df) in zip(axes.flatten(), all_data.items()):
        ax.plot(df.index, df['Close'], color=stock_colors[name], linewidth=1.2)
        ax.set_title(f'{name} — Daily Close Price (2yr)', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
 
        last_price = df['Close'].iloc[-1]
        ax.axhline(last_price, color=stock_colors[name], linestyle='--', alpha=0.4)
        ax.text(df.index[-1], last_price, f'  ${last_price:.2f}',
                va='center', fontsize=9, color=stock_colors[name], fontweight='bold')
 
    plt.suptitle('Historical Equity Prices — 2-Year Window', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
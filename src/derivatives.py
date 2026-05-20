import numpy as np
from scipy.stats import norm


class Derivative:
    """Base for all derivative instruments."""

    def __init__(self, S0, K, T, sigma, yield_curve):
        """
        Parameters:
            S0: current underlying price
            K: strike price
            T: time to maturity (in years)
            sigma: float
                Volatility of the underlying asset (annualised)
            yield_curve: YieldCurve
                Instance of the YieldCurve class used for discounting
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.yield_curve = yield_curve

    def price(self):
        """
        Computes the price of the derivative.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Pricing logic must be implemented in the subclass."
        )

    def delta(self):
        raise NotImplementedError(
            "Delta calculation must be implemented in the subclass."
        )


class EuropeanCall(Derivative):
    """
    European Call Option priced using the Black-Scholes formula.
    """

    def price(self):
        """
        Returns the Black-Scholes price of a European call option.
        """
        # Obtain the appropriate zero rate from the yield curve
        r = self.yield_curve.get_zero_rate(self.T)

        # Black-Scholes d1 and d2
        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

        d2 = d1 - self.sigma * np.sqrt(self.T)

        # Black-Scholes pricing formula
        call_price = (
            self.S0 * norm.cdf(d1)
            - self.K
            * np.exp(-r * self.T)
            * norm.cdf(d2)
        )

        return call_price

    def delta(self):
        """
        Returns delta
        """
        r = self.yield_curve.get_zero_rate(self.T)

        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

        return norm.cdf(d1)


class EuropeanPut(Derivative):

    def price(self):
        r = self.yield_curve.get_zero_rate(self.T)

        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

        d2 = d1 - self.sigma * np.sqrt(self.T)

        put_price = (
            self.K
            * np.exp(-r * self.T)
            * norm.cdf(-d2)
            - self.S0 * norm.cdf(-d1)
        )

        return put_price

    def delta(self):
        """
        Returns delta
        """
        r = self.yield_curve.get_zero_rate(self.T)

        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

        return norm.cdf(d1) - 1


# Aliases for backwards compatibility
EuropeanCallOption = EuropeanCall
EuropeanPutOption  = EuropeanPut

# -------------------------------------------------------
# AMERICAN OPTIONS - BINOMIAL TREE
# -------------------------------------------------------

class AmericanCall(Derivative):
    """
    American Call Option priced using the Binomial Tree model.
    Allows for early exercise at each node.
    """

    def __init__(self, S0, K, T, sigma, yield_curve, steps=100):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.steps = steps

    def price(self):
        """
        Returns the Binomial Tree price of an American call option.
        """
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        # Initialise asset prices at maturity
        asset_prices = np.array([
            self.S0 * (u ** (self.steps - 2 * j))
            for j in range(self.steps + 1)
        ])

        # Option values at maturity
        option_values = np.maximum(asset_prices - self.K, 0)

        # Step backwards through the tree
        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([
                self.S0 * (u ** (i - 2 * j))
                for j in range(i + 1)
            ])
            option_values = discount * (
                p * option_values[:-1] + (1 - p) * option_values[1:]
            )
            # Early exercise check
            option_values = np.maximum(option_values, asset_prices - self.K)

        return float(option_values[0])

    def delta(self):
        """
        Returns delta using finite difference approximation.
        """
        h = self.S0 * 0.01
        up = AmericanCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        """
        Returns gamma using finite difference approximation.
        """
        h = self.S0 * 0.01
        up = AmericanCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        mid = AmericanCall(self.S0, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - 2 * mid.price() + down.price()) / (h ** 2)

    def theta(self):
        """
        Returns theta (per calendar day).
        """
        h = 1 / 365
        if self.T <= h:
            return 0.0
        fwd = AmericanCall(self.S0, self.K, self.T - h, self.sigma, self.yield_curve, self.steps)
        return fwd.price() - self.price()

    def vega(self):
        """
        Returns vega (sensitivity to 1% change in volatility).
        """
        h = 0.01
        up = AmericanCall(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.steps)
        down = AmericanCall(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        """
        Returns rho (sensitivity to interest rate).
        """
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-(r + h) * dt)
        p = (np.exp((r + h) * dt) - d) / (u - d)

        asset_prices = np.array([
            self.S0 * (u ** (self.steps - 2 * j))
            for j in range(self.steps + 1)
        ])
        option_values = np.maximum(asset_prices - self.K, 0)

        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([
                self.S0 * (u ** (i - 2 * j))
                for j in range(i + 1)
            ])
            option_values = discount * (
                p * option_values[:-1] + (1 - p) * option_values[1:]
            )
            option_values = np.maximum(option_values, asset_prices - self.K)

        price_up = float(option_values[0])
        return (price_up - self.price()) / h


class AmericanPut(Derivative):
    """
    American Put Option priced using the Binomial Tree model.
    Allows for early exercise at each node.
    """

    def __init__(self, S0, K, T, sigma, yield_curve, steps=100):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.steps = steps

    def price(self):
        """
        Returns the Binomial Tree price of an American put option.
        """
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        # Initialise asset prices at maturity
        asset_prices = np.array([
            self.S0 * (u ** (self.steps - 2 * j))
            for j in range(self.steps + 1)
        ])

        # Option values at maturity
        option_values = np.maximum(self.K - asset_prices, 0)

        # Step backwards through the tree
        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([
                self.S0 * (u ** (i - 2 * j))
                for j in range(i + 1)
            ])
            option_values = discount * (
                p * option_values[:-1] + (1 - p) * option_values[1:]
            )
            # Early exercise check
            option_values = np.maximum(option_values, self.K - asset_prices)

        return float(option_values[0])

    def delta(self):
        h = self.S0 * 0.01
        up = AmericanPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        down = AmericanPut(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up = AmericanPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.steps)
        mid = AmericanPut(self.S0, self.K, self.T, self.sigma, self.yield_curve, self.steps)
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
        up = AmericanPut(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.steps)
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

        asset_prices = np.array([
            self.S0 * (u ** (self.steps - 2 * j))
            for j in range(self.steps + 1)
        ])
        option_values = np.maximum(self.K - asset_prices, 0)

        for i in range(self.steps - 1, -1, -1):
            asset_prices = np.array([
                self.S0 * (u ** (i - 2 * j))
                for j in range(i + 1)
            ])
            option_values = discount * (
                p * option_values[:-1] + (1 - p) * option_values[1:]
            )
            option_values = np.maximum(option_values, self.K - asset_prices)

        price_up = float(option_values[0])
        return (price_up - self.price()) / h


# -------------------------------------------------------
# ASIAN OPTION - MONTE CARLO
# -------------------------------------------------------

class AsianCall(Derivative):
    """
    Asian Call Option priced using Monte Carlo simulation.
    Payoff is based on the average price over the life of the option.
    """

    def __init__(self, S0, K, T, sigma, yield_curve, simulations=10000, steps=252):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.simulations = simulations
        self.steps = steps

    def price(self):
        """
        Returns the Monte Carlo price of an Asian call option.
        """
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)

        # Simulate price paths
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(
            np.cumsum(
                (r - 0.5 * self.sigma ** 2) * dt
                + self.sigma * np.sqrt(dt) * Z,
                axis=1
            )
        )

        # Average price along each path
        avg_prices = np.mean(price_paths, axis=1)

        # Payoff
        payoffs = np.maximum(avg_prices - self.K, 0)

        # Discount back to present
        return float(np.exp(-r * self.T) * np.mean(payoffs))

    def delta(self):
        h = self.S0 * 0.01
        up = AsianCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        down = AsianCall(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up = AsianCall(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        mid = AsianCall(self.S0, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
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
        up = AsianCall(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.simulations, self.steps)
        down = AsianCall(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(
            np.cumsum(
                ((r + h) - 0.5 * self.sigma ** 2) * dt
                + self.sigma * np.sqrt(dt) * Z,
                axis=1
            )
        )
        avg_prices = np.mean(price_paths, axis=1)
        payoffs = np.maximum(avg_prices - self.K, 0)
        price_up = float(np.exp(-(r + h) * self.T) * np.mean(payoffs))
        return (price_up - self.price()) / h


class AsianPut(Derivative):
    """
    Asian Put Option priced using Monte Carlo simulation.
    Payoff is based on the average price over the life of the option.
    """

    def __init__(self, S0, K, T, sigma, yield_curve, simulations=10000, steps=252):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.simulations = simulations
        self.steps = steps

    def price(self):
        """
        Returns the Monte Carlo price of an Asian put option.
        """
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)

        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(
            np.cumsum(
                (r - 0.5 * self.sigma ** 2) * dt
                + self.sigma * np.sqrt(dt) * Z,
                axis=1
            )
        )

        avg_prices = np.mean(price_paths, axis=1)
        payoffs = np.maximum(self.K - avg_prices, 0)
        return float(np.exp(-r * self.T) * np.mean(payoffs))

    def delta(self):
        h = self.S0 * 0.01
        up = AsianPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        down = AsianPut(self.S0 - h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h)

    def gamma(self):
        h = self.S0 * 0.01
        up = AsianPut(self.S0 + h, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
        mid = AsianPut(self.S0, self.K, self.T, self.sigma, self.yield_curve, self.simulations, self.steps)
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
        up = AsianPut(self.S0, self.K, self.T, self.sigma + h, self.yield_curve, self.simulations, self.steps)
        down = AsianPut(self.S0, self.K, self.T, self.sigma - h, self.yield_curve, self.simulations, self.steps)
        return (up.price() - down.price()) / (2 * h) * 0.01

    def rho(self):
        h = 0.0001
        r = self.yield_curve.get_zero_rate(self.T)
        dt = self.T / self.steps
        np.random.seed(42)
        Z = np.random.standard_normal((self.simulations, self.steps))
        price_paths = self.S0 * np.exp(
            np.cumsum(
                ((r + h) - 0.5 * self.sigma ** 2) * dt
                + self.sigma * np.sqrt(dt) * Z,
                axis=1
            )
        )
        avg_prices = np.mean(price_paths, axis=1)
        payoffs = np.maximum(self.K - avg_prices, 0)
        price_up = float(np.exp(-(r + h) * self.T) * np.mean(payoffs))
        return (price_up - self.price()) / h


# -------------------------------------------------------
# GREEKS FOR EUROPEAN OPTIONS (Closed-Form)
# -------------------------------------------------------

class EuropeanCallGreeks(EuropeanCall):
    """
    Extends EuropeanCall with full closed-form Greeks.
    """

    def _d1_d2(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def delta(self):
        """Rate of change of price with respect to S0."""
        d1, _ = self._d1_d2()
        return norm.cdf(d1)

    def gamma(self):
        """Rate of change of delta with respect to S0."""
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def theta(self):
        """Time decay (per calendar day)."""
        r = self.yield_curve.get_zero_rate(self.T)
        d1, d2 = self._d1_d2()
        theta = (
            - (self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
            - r * self.K * np.exp(-r * self.T) * norm.cdf(d2)
        )
        return theta / 365

    def vega(self):
        """Sensitivity to 1% change in volatility."""
        d1, _ = self._d1_d2()
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def rho(self):
        """Sensitivity to 1% change in interest rate."""
        r = self.yield_curve.get_zero_rate(self.T)
        _, d2 = self._d1_d2()
        return self.K * self.T * np.exp(-r * self.T) * norm.cdf(d2) * 0.01

    def all_greeks(self):
        """Returns all Greeks as a dictionary."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega":  self.vega(),
            "rho":   self.rho(),
        }


class EuropeanPutGreeks(EuropeanPut):
    """
    Extends EuropeanPut with full closed-form Greeks.
    """

    def _d1_d2(self):
        r = self.yield_curve.get_zero_rate(self.T)
        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def delta(self):
        """Rate of change of price with respect to S0."""
        d1, _ = self._d1_d2()
        return norm.cdf(d1) - 1

    def gamma(self):
        """Rate of change of delta with respect to S0."""
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def theta(self):
        """Time decay (per calendar day)."""
        r = self.yield_curve.get_zero_rate(self.T)
        d1, d2 = self._d1_d2()
        theta = (
            - (self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
            + r * self.K * np.exp(-r * self.T) * norm.cdf(-d2)
        )
        return theta / 365

    def vega(self):
        """Sensitivity to 1% change in volatility."""
        d1, _ = self._d1_d2()
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def rho(self):
        """Sensitivity to 1% change in interest rate."""
        r = self.yield_curve.get_zero_rate(self.T)
        _, d2 = self._d1_d2()
        return -self.K * self.T * np.exp(-r * self.T) * norm.cdf(-d2) * 0.01

    def all_greeks(self):
        """Returns all Greeks as a dictionary."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega":  self.vega(),
            "rho":   self.rho(),
        }

"""
display.py  –  pretty-print helpers for derivatives objects.
 
Usage in notebook:
    from src.display import print_european_call_greeks, print_american_put, print_asian_call
"""
 
 
def print_european_call_greeks(option):
    """Print a formatted Greeks table for a EuropeanCallGreeks instance."""
    greeks = option.all_greeks()
    print("=" * 45)
    print("       EUROPEAN CALL OPTION - GREEKS")
    print("=" * 45)
    rows = [
        ("Delta", greeks["delta"], "price change per $1 move in S0"),
        ("Gamma", greeks["gamma"], "rate of change of delta"),
        ("Theta", greeks["theta"], "price decay per calendar day"),
        ("Vega",  greeks["vega"],  "price change per 1% vol move"),
        ("Rho",   greeks["rho"],   "price change per 1% rate move"),
    ]
    for name, value, note in rows:
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
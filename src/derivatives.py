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
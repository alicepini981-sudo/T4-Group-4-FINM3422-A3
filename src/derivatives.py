import numpy as np

class Derivative:
    # Base for all derivative instruments

    def __init__(self, S, K, T, sigma, yield_curve, q=0.0):
        # Parameters:

        # S: current underlying price
        # K: strike price
        # T: time to maturity
        # sigma: annualised volatility
        # yield_curve: used for discounting and zero rates
        # q: continuous dividend yield (defaults to 0)

        if S <= 0:
            raise ValueError(f"S must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.sigma = float(sigma)
        self.yield_curve = yield_curve
        self.q = float(q)

    def price(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement price()")

    def get_discount_factor(self):
        return self.yield_curve.get_discount_factor(self.T)

    def get_zero_rate(self):
        return self.yield_curve.get_zero_rate(self.T)
    
class EuropeanCallOption(Derivative): 
    def payoff(self, spot_at_maturity):
            spot_at_maturity = float(spot_at_maturity)
            return max(0, spot_at_maturity - self.K)
    
class EuropeanPutOption(Derivative):
    def payoff(self, spot_at_maturity):
            spot_at_maturity = float(spot_at_maturity)
            return max(0, self.K - spot_at_maturity)

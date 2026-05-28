import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_yield_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'Yield_Data.csv')
    df = pd.read_csv(data_path)
    df.columns = ["maturity", "zero_rate"]
    df["maturity"] = pd.to_numeric(df["maturity"], errors='coerce')
    df["zero_rate"] = pd.to_numeric(df["zero_rate"], errors='coerce') / 100
    return df.dropna()


class YieldCurve:
    """
    Represents a term structure of zero rates and provides
    discount factors for valuation. All interest-rate logic
    lives here and is reused by other models.
    """

    def __init__(self, maturities, zero_rates, compounding="continuous"):
        self.maturities = np.array(maturities, dtype=float)
        self.zero_rates = np.array(zero_rates, dtype=float)
        self.compounding = compounding
        if len(self.maturities) != len(self.zero_rates):
            raise ValueError("Maturities and zero rates must have the same length.")
        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.zero_rates = self.zero_rates[order]

    @classmethod
    def from_dataframe(cls, df, compounding='continuous'):
        """Build a YieldCurve from the output of load_yield_data()."""
        return cls(df['maturity'].values, df['zero_rate'].values, compounding=compounding)

    def get_zero_rate(self, T):
        """Return the interpolated zero rate for maturity T (in years)."""
        return float(np.interp(float(T), self.maturities, self.zero_rates))

    def get_discount_factor(self, T):
        """Return the discount factor D(T) using the yield curve."""
        z = self.get_zero_rate(T)
        if self.compounding == "continuous":
            return np.exp(-z * T)
        elif self.compounding == "annual":
            return 1.0 / (1.0 + z) ** T
        else:
            raise ValueError("Unsupported compounding type.")

    def plot(self, max_maturity=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.maturities, self.zero_rates * 100,
                 color='steelblue', linewidth=1.5, label='Linear interpolation')
        plt.scatter(self.maturities, self.zero_rates * 100,
                    marker='s', color='black', s=40, zorder=5, label='Observed Data Points')
        plt.xlabel("Maturity (Years)", fontsize=12)
        plt.ylabel("Zero Rate (%)", fontsize=12)
        plt.title("Australian Government Bond Yield Curve (April 2026 Data)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
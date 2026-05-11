import numpy as np
import matplotlib.pyplot as plt

class YieldCurve:
    """
    YieldCureve represents a term structure of zero rates and
    providees discount factors fo valuation
    This class is infrastructre: all interest-rate logice
    should live here and be reused by other models
    """
    def __init__(self, maturities, zero_rates, compounding="continuous"):
        """
        Parameters
        ----------
        maturities : array-like
            Maturities in years (e.g. [0.25, 0.5, 1, 2, 5, 10])
        zero_rates : array-like
            Annualised zero rates as decimals (e.g. 0.045)
        compounding : str
            'continuous' or 'annual'
        """
        self.maturities = np.array(maturities, dtype=float)
        self.zero_rates = np.array(zero_rates, dtype=float)
        self.compounding = compounding
        if len(self.maturities) != len(self.zero_rates):
            raise ValueError("Maturities and zero rates must have the same length")
        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.zero_rates = self.zero_rates[order]

    def get_zero_rate(self, T):
        T = float(T)
        return float(np.interp(T, self.maturities, self.zero_rates))

    def get_discount_factor(self, T):
        z = self.get_zero_rate(T)
        if self.compounding == "continuous":
            return np.exp(-z * T)
        elif self.compounding == "annual":
            return 1.0 / (1.0 + z) ** T
        else:
            raise ValueError("Unsupported compounding type")

    def plot(self, max_maturity=None):
        """
        Plot the zero-rate yield curve
        """
        if max_maturity is None:
            T_grid = self.maturities
        else:
            T_grid = np.linspace(
                self.maturities.min(),
                max_maturity,
                100
            )
        z_grid = [self.get_zero_rate(T) * 100 for T in T_grid]
        plt.figure()
        plt.plot(T_grid, z_grid)
        plt.xlabel("Maturity (years)")
        plt.ylabel("Zero Rate (% p.a.)")
        plt.title("Yield Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ── Everything below this line is NEW ──────────────────────────────────────

import pandas as pd

def load_rba_yield_curve(data_path: str, as_of_date: str = None) -> YieldCurve:
    df = pd.read_csv(
        data_path,
        skiprows=10,
        header=None,
        names=[
            'date', 'cash_rate_target', 'interbank_rate', 'interbank_high',
            'interbank_low', 'interbank_vol', 'interbank_num',
            'bab_1m', 'bab_3m', 'bab_6m',
            'ois_1m', 'ois_3m', 'ois_6m',
            'tn_1m', 'tn_3m', 'tn_6m'
        ]
    )
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    rate_cols = ['bab_1m', 'bab_3m', 'bab_6m']
    df[rate_cols] = df[rate_cols].apply(pd.to_numeric, errors='coerce')
    if as_of_date:
        target = pd.to_datetime(as_of_date, dayfirst=True)
        row = df[df['date'] == target].iloc[0]
    else:
        row = df.dropna(subset=rate_cols).iloc[-1]
    maturities = [1/12, 3/12, 6/12]
    zero_rates  = [row['bab_1m'] / 100, row['bab_3m'] / 100, row['bab_6m'] / 100]
    return YieldCurve(maturities, zero_rates)


if __name__ == "__main__":
    yc = load_rba_yield_curve("data/f1.1-data.csv")

    print(f"1-month zero rate:    {yc.get_zero_rate(1/12) * 100:.4f}%")
    print(f"3-month zero rate:    {yc.get_zero_rate(3/12) * 100:.4f}%")
    print(f"6-month zero rate:    {yc.get_zero_rate(6/12) * 100:.4f}%")
    print(f"3-month disc factor:  {yc.get_discount_factor(3/12):.6f}")
    print(f"6-month disc factor:  {yc.get_discount_factor(6/12):.6f}")

    yc.plot()
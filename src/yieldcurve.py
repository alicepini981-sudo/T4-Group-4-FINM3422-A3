import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline


class YieldCurve:
    """
    YieldCurve represents a term structure of zero rates and
    provides discount factors for valuation.
    This class is infrastructure: all interest-rate logic
    should live here and be reused by other models.
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
        Plot the zero-rate yield curve.
        """
        if max_maturity is None:
            T_grid = np.linspace(self.maturities.min(), self.maturities.max(), 200)
        else:
            T_grid = np.linspace(self.maturities.min(), max_maturity, 200)

        z_grid = [self.get_zero_rate(T) * 100 for T in T_grid]

        plt.figure(figsize=(9, 5))
        plt.plot(T_grid, z_grid, color='steelblue', linewidth=2, label='Interpolated curve')
        plt.scatter(self.maturities, self.zero_rates * 100,
                    color='crimson', zorder=5, label='Input data points')
        plt.xlabel("Maturity (years)")
        plt.ylabel("Zero Rate (% p.a.)")
        plt.title("RBA Yield Curve")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# ── Data loading ────────────────────────────────────────────────────────────

def load_rba_yield_curve(f1_path: str, f2_path: str = None, as_of_date: str = None) -> YieldCurve:
    """
    Build a YieldCurve from RBA F1.1 (short end) and optionally F2 (long end).

    Parameters
    ----------
    f1_path    : path to f1.1-data.csv
    f2_path    : path to f2-data.csv (optional, extends curve to 10 years)
    as_of_date : 'DD/MM/YYYY' — defaults to most recent available row
    """

    # ── Load F1.1 short-end data (1m, 3m, 6m) ───────────────────────────
    df1 = pd.read_csv(
        f1_path,
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
    df1['date'] = pd.to_datetime(df1['date'], dayfirst=True, errors='coerce')
    df1 = df1.dropna(subset=['date'])
    short_cols = ['bab_1m', 'bab_3m', 'bab_6m']
    df1[short_cols] = df1[short_cols].apply(pd.to_numeric, errors='coerce')

    # ── Load F2 long-end data (2yr, 3yr, 5yr, 10yr) ─────────────────────
    if f2_path:
        df2 = pd.read_csv(
            f2_path,
            skiprows=10,
            header=None,
            names=['date', 'gov_2y', 'gov_3y', 'gov_5y', 'gov_10y', 'indexed_10y']
        )
        df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')
        df2 = df2.dropna(subset=['date'])
        long_cols = ['gov_2y', 'gov_3y', 'gov_5y', 'gov_10y']
        df2[long_cols] = df2[long_cols].apply(pd.to_numeric, errors='coerce')

    # ── Select the right date row ────────────────────────────────────────
    if as_of_date:
        target = pd.to_datetime(as_of_date, dayfirst=True)
        row1 = df1[df1['date'] == target].iloc[0]
        row2 = df2[df2['date'] == target].iloc[0] if f2_path else None
    else:
        row1 = df1.dropna(subset=short_cols).iloc[-1]
        row2 = df2.dropna(subset=long_cols).iloc[-1] if f2_path else None

    # ── Build combined maturity/rate arrays ──────────────────────────────
    maturities = [1/12, 3/12, 6/12]
    zero_rates = [row1['bab_1m'] / 100, row1['bab_3m'] / 100, row1['bab_6m'] / 100]

    if f2_path and row2 is not None:
        maturities += [2.0, 3.0, 5.0, 10.0]
        zero_rates += [
            row2['gov_2y']  / 100,
            row2['gov_3y']  / 100,
            row2['gov_5y']  / 100,
            row2['gov_10y'] / 100,
        ]

    return YieldCurve(maturities, zero_rates)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    yc = load_rba_yield_curve(
        "data/f1.1-data.csv",
        f2_path="data/f2-data.csv",
        as_of_date="31/03/2026"
    )

    print(f"1-month zero rate:    {yc.get_zero_rate(1/12)  * 100:.4f}%")
    print(f"3-month zero rate:    {yc.get_zero_rate(3/12)  * 100:.4f}%")
    print(f"6-month zero rate:    {yc.get_zero_rate(6/12)  * 100:.4f}%")
    print(f"2-year zero rate:     {yc.get_zero_rate(2.0)   * 100:.4f}%")
    print(f"5-year zero rate:     {yc.get_zero_rate(5.0)   * 100:.4f}%")
    print(f"10-year zero rate:    {yc.get_zero_rate(10.0)  * 100:.4f}%")
    print(f"3-month disc factor:  {yc.get_discount_factor(3/12):.6f}")
    print(f"10-year disc factor:  {yc.get_discount_factor(10.0):.6f}")

    yc.plot(max_maturity=10)
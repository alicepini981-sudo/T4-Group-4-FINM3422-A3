import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        if max_maturity is None:
            T_grid = np.linspace(self.maturities.min(), self.maturities.max(), 200)
        else:
            T_grid = np.linspace(self.maturities.min(), max_maturity, 200)

        z_grid = [self.get_zero_rate(T) * 100 for T in T_grid]

        plt.figure(figsize=(9, 5))
        plt.plot(T_grid, z_grid, color='steelblue', linewidth=2, label='Interpolated curve')
        plt.scatter(self.maturities, self.zero_rates * 100,
                    color='black', zorder=5, s=40, label='Observed yields')
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title("Australian Government Bond Yield Curve as of Mar-26")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(bottom=4.1)
        plt.tight_layout()
        plt.show()


def plot_historical(curves: dict):
    """Plot multiple yield curves on one chart."""
    plt.figure(figsize=(9, 5))

    n = len(curves)
    blues = cm.Blues(np.linspace(0.3, 0.9, n))

    for (label, yc), color in zip(curves.items(), blues):
        T_grid = np.linspace(yc.maturities.min(), yc.maturities.max(), 200)
        z_grid = [yc.get_zero_rate(T) * 100 for T in T_grid]
        plt.plot(T_grid, z_grid, linewidth=2, label=label, color=color)
        plt.scatter(yc.maturities, yc.zero_rates * 100, zorder=5, s=30, color=color)

    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.title("Australian Government Bond Yield Curves — 2022 to 2026")
    plt.legend(title="Year")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def load_rba_yield_curve(f1_path: str, f2_path: str = None, as_of_date: str = None) -> YieldCurve:
    df1 = pd.read_csv(
        f1_path,
        skiprows=11,
        header=None,
        names=[
            'date', 'cash_rate_target', 'interbank_rate', 'interbank_high',
            'interbank_low', 'interbank_vol', 'interbank_num',
            'bab_1m', 'bab_3m', 'bab_6m',
            'ois_1m', 'ois_3m', 'ois_6m',
            'tn_1m', 'tn_3m', 'tn_6m'
        ]
    )
    df1['date'] = pd.to_datetime(df1['date'], format='%d/%m/%Y', errors='coerce')
    df1 = df1.dropna(subset=['date'])
    short_cols = ['bab_1m', 'bab_3m', 'bab_6m']
    df1[short_cols] = df1[short_cols].apply(pd.to_numeric, errors='coerce')

    if f2_path:
        df2 = pd.read_csv(
            f2_path,
            skiprows=11,
            header=None,
            names=['date', 'gov_2y', 'gov_3y', 'gov_5y', 'gov_10y', 'indexed_10y']
        )
        df2['date'] = pd.to_datetime(df2['date'], format='%d-%b-%Y', errors='coerce')
        df2 = df2.dropna(subset=['date'])
        long_cols = ['gov_2y', 'gov_3y', 'gov_5y', 'gov_10y']
        df2[long_cols] = df2[long_cols].apply(pd.to_numeric, errors='coerce')

    if as_of_date:
        target = pd.to_datetime(as_of_date, format='%d/%m/%Y')
        row1 = df1.iloc[(df1['date'] - target).abs().argsort().iloc[0]]
        row2 = df2.iloc[(df2['date'] - target).abs().argsort().iloc[0]] if f2_path else None
    else:
        row1 = df1.dropna(subset=short_cols).iloc[-1]
        row2 = df2.dropna(subset=long_cols).iloc[-1] if f2_path else None

    maturities = [1/12, 3/12, 6/12, 2.0, 3.0, 5.0, 10.0]
    zero_rates = [
        row1['bab_1m'] / 100,
        row1['bab_3m'] / 100,
        row1['bab_6m'] / 100,
        row2['gov_2y']  / 100,
        row2['gov_3y']  / 100,
        row2['gov_5y']  / 100,
        row2['gov_10y'] / 100,
    ]

    return YieldCurve(maturities, zero_rates)
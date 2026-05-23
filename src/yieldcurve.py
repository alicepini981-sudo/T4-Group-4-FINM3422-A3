import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class YieldCurve:
    def __init__(self, maturities, zero_rates, compounding="continuous"):
        self.maturities = np.array(maturities, dtype=float)
        self.zero_rates = np.array(zero_rates, dtype=float)
        self.compounding = compounding

        if len(self.maturities) != len(self.zero_rates):
            raise ValueError("Maturities and Zero Rates should have the same number of elements")

        if self.compounding not in ("continuous", "annual"):
            raise ValueError("compounding must be either 'continuous' or 'annual'")

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
        else:
            return 1.0 / (1 + z) ** T

    def plot(self, max_maturity=None):
        if max_maturity is None:
            T_grid = self.maturities
        else:
            T_grid = np.linspace(self.maturities.min(), max_maturity, 100)

        z_grid = [self.get_zero_rate(T) * 100 for T in T_grid]

        plt.figure()
        plt.plot(T_grid, z_grid)
        plt.xlabel("Time to Maturity (Years)")
        plt.ylabel("Zero Rate (%)")
        plt.title("Australian Government Bond Yield Curve — Mar 2026")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def load_rba_yield_curve(filepath, f2_path=None, as_of_date=None, compounding="continuous"):
    df1 = pd.read_csv(filepath, skiprows=11, header=None, names=[
        'date','cash_rate_target','interbank_rate','interbank_high','interbank_low',
        'interbank_vol','interbank_num','bab_1m','bab_3m','bab_6m',
        'ois_1m','ois_3m','ois_6m','tn_1m','tn_3m','tn_6m'
    ])
    df1['date'] = pd.to_datetime(df1['date'], format='%d/%m/%Y', errors='coerce')
    df1 = df1.dropna(subset=['date'])
    short_cols = ['bab_1m','bab_3m','bab_6m']
    df1[short_cols] = df1[short_cols].apply(pd.to_numeric, errors='coerce')

    df2 = pd.read_csv(f2_path, skiprows=11, header=None, names=[
        'date','gov_2y','gov_3y','gov_5y','gov_10y','indexed_10y'
    ])
    df2['date'] = pd.to_datetime(df2['date'], format='%d-%b-%Y', errors='coerce')
    df2 = df2.dropna(subset=['date'])
    long_cols = ['gov_2y','gov_3y','gov_5y','gov_10y']
    df2[long_cols] = df2[long_cols].apply(pd.to_numeric, errors='coerce')

    if as_of_date:
        target = pd.to_datetime(as_of_date, format='%d/%m/%Y')
        row1 = df1.iloc[(df1['date'] - target).abs().argsort().iloc[0]]
        row2 = df2.iloc[(df2['date'] - target).abs().argsort().iloc[0]]
    else:
        row1 = df1.dropna(subset=short_cols).iloc[-1]
        row2 = df2.dropna(subset=long_cols).iloc[-1]

    print(f"Using yield curve data from: {row2['date'].strftime('%d-%b-%Y')}")

    maturities = [1/12, 3/12, 6/12, 2.0, 3.0, 5.0, 10.0]
    zero_rates = [
        row1['bab_1m']/100, row1['bab_3m']/100, row1['bab_6m']/100,
        row2['gov_2y']/100, row2['gov_3y']/100, row2['gov_5y']/100, row2['gov_10y']/100,
    ]
    return YieldCurve(maturities, zero_rates, compounding=compounding)
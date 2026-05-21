import numpy as np
import pandas as pd


class EquityPosition:
    """
    Simple equity object for the portfolio layer.
    """

    def __init__(self, ticker, spot):
        self.ticker = ticker
        self.spot = spot

    def price(self):
        return self.spot

    def delta(self):
        # Equity moves one-for-one with itself
        return 1.0


class Portfolio:
    """
    Simple portfolio class for A3.

    This class:
    - stores positions
    - computes total portfolio value
    - computes total portfolio delta
    - computes basic historical VaR
    - creates a simple position table

    Scenario analysis should be done in the notebook, not here.
    """

    def __init__(self):
        self.positions = []

    def add_position(self, instrument, quantity, label=None):
        """
        Add an instrument position to the portfolio.
        """
        self.positions.append({
            "instrument": instrument,
            "quantity": quantity,
            "label": label
        })

    def value(self):
        """
        Compute total portfolio value.
        """
        total_value = 0.0

        for position in self.positions:
            instrument = position["instrument"]
            quantity = position["quantity"]

            total_value += quantity * instrument.price()

        return total_value

    def delta(self):
        """
        Compute total portfolio delta.
        """
        total_delta = 0.0

        for position in self.positions:
            instrument = position["instrument"]
            quantity = position["quantity"]

            total_delta += quantity * instrument.delta()

        return total_delta

    def position_table(self):
        """
        Return a table showing each position's contribution
        to value and delta.
        """
        rows = []

        for position in self.positions:
            instrument = position["instrument"]
            quantity = position["quantity"]

            if position["label"] is not None:
                name = position["label"]
            elif hasattr(instrument, "ticker"):
                name = instrument.ticker
            else:
                name = instrument.__class__.__name__

            unit_value = instrument.price()
            unit_delta = instrument.delta()

            rows.append({
                "Position": name,
                "Quantity": quantity,
                "Unit Value": unit_value,
                "Position Value": quantity * unit_value,
                "Unit Delta": unit_delta,
                "Position Delta": quantity * unit_delta
            })

        df = pd.DataFrame(rows)

        if len(df) > 0:
            total_row = pd.DataFrame([{
                "Position": "TOTAL",
                "Quantity": np.nan,
                "Unit Value": np.nan,
                "Position Value": df["Position Value"].sum(),
                "Unit Delta": np.nan,
                "Position Delta": df["Position Delta"].sum()
            }])

            df = pd.concat([df, total_row], ignore_index=True)

        return df

    def historical_var(self, returns, alpha=0.95, horizon_days=1):
        """
        Compute basic historical VaR.

        Parameters
        ----------
        returns : array-like
            Historical return series
        alpha : float
            Confidence level (e.g. 0.95 or 0.99)
        horizon_days : int
            1-day or 10-day horizon

        Returns
        -------
        float
            Historical VaR in dollars
        """
        returns = pd.Series(returns).dropna()

        if len(returns) == 0:
            raise ValueError("Return series is empty.")

        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")

        # Simple educational scaling for horizon
        scaled_returns = returns * np.sqrt(horizon_days)

        q = scaled_returns.quantile(1 - alpha)
        var_dollar = -q * self.value()

        return max(var_dollar, 0)
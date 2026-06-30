"""
Volatility and Drift Calculations from Random Price Data:

This module demonstrates how to:
1. Generate sample stock price data with random daily fluctuations.
2. Calculate daily logarithmic returns.
3. Compute daily and annualized volatility (sigma).
4. Calculate daily and annualized drift (mu).

The calculations follow standard financial mathematics principles where:
- Volatility scales with the square root of time.
- Drift scales linearly with time.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------------
# RANDOOM PRICE GENERATION:

# A: Create 252 days of "sample" closing prices for a stock. 252 trading days in a year.
# We start at $100 and add some random daily noise (fluctuations to simulate price movements).


# A1: First we need to set a random seed for reproducibility.
# We are setting the seed to 42, a common choice.
np.random.seed(42)

# A2: Now we generate the daily price changes/shocks using a normal distribution.
# Small mean (0.0005) and standard deviation (0.015) to mimic real market conditions.
daily_shocks = np.random.normal(0.0005, 0.015, 252)


# A3: Creating closing prices by starting at 100 and applying the cumulative sum of daily shocks.
closing_prices = 100 * np.exp(np.cumsum(daily_shocks))

# A4: Printing the first and last 5 closing prices in a tabular format.
# Convert to DataFrame for better presentation.
df_prices = pd.DataFrame({
    "Day": range(len(closing_prices)),
    "Closing Price": closing_prices
})

# Select first 5 and last 5 rows for display.
df_display = pd.concat([df_prices.head(5), df_prices.tail(5)])

# Displaying rows with the index, formatted to 6 decimal places for better readability.
print(df_display.to_string(index=False, float_format="%.6f"))

# --------------------------------------------------------------------------------------------------
# VOLATILITY AND DRIFT CALCULATIONS:

# B: Calculating volatility.

# B1: Calculate daily logarithmic returns.
log_returns = np.log(closing_prices[1:] / closing_prices[:-1])

# B2: Calculate daily volatility (standard deviation of log returns).
daily_volatility = np.std(log_returns)

# B3: Annualize the daily volatility (sigma).
# Concept: Scale the 'daily volatility' to a 'yearly volatility' using sqrt(252).
# Reason for sqrt(252): Volatility scales with the sqrt of time, so we multiply it by sqrt(252).
annual_volatility = daily_volatility * np.sqrt(252)

# C: Calculating drift.

# C1: Calculate daily drift (mean of log returns).
# The drift (mu) represents the expected return of the stock, excluding volatility.
# It is calculated as the mean of the daily log returns.
daily_drift = np.mean(log_returns)

# C2: Annualize the drift (mu).
# Similar to volatility, we scale the daily drift to an annual drift by multiplying by 252.
# However, drift scales linearly with time, so we multiply by 252 (not sqrt(252)).
annual_drift = daily_drift * 252

# --------------------------------------------------------------------------------------------------
# BETTER PRESENTATION:

# Creating a DataFrame to display the results in a tabular format:
df_results = pd.DataFrame({
    "Metric": ["Annualized Volatility (σ)", "Annualized Drift (μ)"],
    "Value": [f"{annual_volatility}", f"{annual_drift}"]
})

print(df_results.to_string(index=False, float_format="%.4f"))
# This will print the DataFrame without the index, with float values formatted to 4 decimal places.

# --------------------------------------------------------------------------------------------------

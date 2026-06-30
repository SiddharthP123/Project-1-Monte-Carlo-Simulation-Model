"""
Returns and Volatility Analysis:

This module demonstrates the calculation of arithmetic and logarithmic returns
for stock prices, along with their corresponding volatility measures. It shows
the differences between arithmetic and log returns, particularly how log returns
are additive over time and provide more accurate total return calculations.

The code uses NumPy to perform vectorized operations on price data and calculates:
- Daily arithmetic returns using the formula (P_t / P_t-1) - 1.
- Daily logarithmic returns using the formula ln(P_t / P_t-1).
- Standard deviation (volatility) for both return types.

Note: We are using a sample dataset, not real-time market data.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------
# SAMPLE DATA:

# A: Example data for a stock's closing prices over 10 days (in $):
prices = np.array([100, 102, 101, 105, 104, 103, 102, 101, 102, 100])

print(prices[1:])
# This will give us the prices from the second day to the last day. Index begins at 0.

print(prices[:-1])
# This will give us the prices from the first day to the second last day.

# --------------------------------------------------------------------------------------------------
# RETURNS CALCULATIONS:

# B: Calculating daily arithmetic returns:

# B1: Return formula is generally (Price_t - Price_t-1) / Price_t-1.
# This can be rearranged to (Price_t / Price_t-1) - 1.
arithmetic_returns = (prices[1:] / prices[:-1]) - 1

# B2: Calculating total return using arithmetic returns:
# The problem with arithmetic returns is that they are misleading.
# The total return over the period is not simply the sum of daily returns due to compounding.


#Step C: Calculating daily logarithmic returns:

# C1: Log returns are calculated using the ln of price ratio: ln(Price_t / Price_t-1).
logarithmic_returns = np.log(prices[1:] / prices[:-1])


# C2: Calculating total return using logarithmic returns:
# Log returns are additive over time & the total log return is independent of compounding.

# --------------------------------------------------------------------------------------------------
# VOLATILITY CALCULATIONS:

# Step D: Calculating arithmetic volatility (standard deviation of arithmetic returns):

# D1: Standard deviation meausures the variation from  mean return.
# Volatility is a measure of the dispersion of returns.
# Higher volatility indicates a riskier asset.
arithmetic_volatility = np.std(arithmetic_returns)

# D2: Calculating logarithmic volatility (standard deviation of logarithmic returns):
logarithmic_volatility = np.std(logarithmic_returns)

# --------------------------------------------------------------------------------------------------
# PRESENTATION OF RESULTS:

# Creating a DataFrame to display the results in a tabular format:
# Range of days is from 1 to the len (returns + 1), as returns are calculated from day 2 to day 10.
df = pd.DataFrame({
    "Day": range(1, len(arithmetic_returns) + 1),
    "Arithmetic Return": arithmetic_returns,
    "Logarithmic Return": logarithmic_returns,
    "Sum of Arithmetic": np.sum(arithmetic_returns),
    "Sum of Logarithmic": np.sum(logarithmic_returns),
    "Arithmetic Volatility": arithmetic_volatility,
    "Logarithmic Volatility": logarithmic_volatility
})


print(df.to_string(index=False, float_format="%.6f"))
# This will print the DataFrame without the index, with float values formatted to 6 decimal places.

# --------------------------------------------------------------------------------------------------

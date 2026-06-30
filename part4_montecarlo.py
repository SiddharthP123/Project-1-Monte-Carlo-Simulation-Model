"""
Monte Carlo Simulation for Stock Price Prediction using Geometric Brownian Motion (GBM):

This module implements a Monte Carlo simulation to predict future stock prices based on 
historical data. It uses the Geometric Brownian Motion model to simulate multiple possible 
future price paths for a given stock.

Key Features:
- Generates historical stock price data with specified mean and volatility.
- Calculates historical volatility and drift from the generated data.
- Runs Monte Carlo simulations using GBM to predict future prices.
- Visualizes historical data alongside simulated future paths.
- Provides statistical analysis of simulation results.

Dependencies:
- numpy: For numerical computations and random number generation.
- matplotlib: For plotting historical data and simulation results.

Usage:
The script runs a complete simulation with predefined parameters and displays the results
including average predicted price, highest potential price, and lowest potential price.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------
# SETUP & REPRODUCIBILITY:
np.random.seed(42)
ORIGINAL_PRICE = 100

# --------------------------------------------------------------------------------------------------
# GENERATING HISTORICAL DATA:
AVG_MEAN = 0.0005
AVG_STD = 0.015
TRADING_DAYS = 252

hist_shocks = np.random.normal(AVG_MEAN, AVG_STD, TRADING_DAYS)
history_prices = ORIGINAL_PRICE * np.exp(np.cumsum(hist_shocks))

# --------------------------------------------------------------------------------------------------
# CALCULATING HISTORICAL VOLATILITY AND DRIFT:
log_returns = np.log(history_prices[1:] / history_prices[:-1])
daily_volatility = np.std(log_returns)
sigma = daily_volatility * np.sqrt(252)
daily_drift = np.mean(log_returns)
mu = daily_drift * 252

# --------------------------------------------------------------------------------------------------
# THE MONTE CARLO GBM SIMULATION::

# A: Define Monte Carlo Parameters:
# We will simulate 100 possible futures for the year 2026.
NUM_SIMULATIONS = 100
T = 1.0
STEPS = 252
DT = T / STEPS
St = history_prices[-1]

# B: Create a 2D Array (Matrix) to store all paths.
# Rows = Number of Simulations, Columns = Trading Days.
all_simulations = np.zeros((NUM_SIMULATIONS, STEPS + 1))
# Set the starting price for EVERY row to the end of 2025.
# This is the "initial condition" for our GBM simulation. All paths start from the same point.
all_simulations[:, 0] = St

# --------------------------------------------------------------------------------------------------
# C: Nested GBM Simulation Loop:

## Reset the random seed to ensure that the randomness is restricted to the GBM.
np.random.seed(None)

for n in range(NUM_SIMULATIONS):
    for i in range(STEPS):
        Z = np.random.standard_normal()
        # Unlike earlier, now we have 100 different paths, so we need to loop in the 2D array.
        # We have to loop with respect to both the number of simulations (n) and steps (i).
        current_S = all_simulations[n, i]
        drift_part = mu * current_S * DT
        diffusion_part = sigma * current_S * np.sqrt(DT) * Z
        dS = drift_part + diffusion_part
        # Similar to last time, just updated for a 2D array.
        all_simulations[n, i+1] = current_S + dS

# --------------------------------------------------------------------------------------------------
# PRESENTING THE RESULTS:

plt.figure(figsize=(12, 6))

# 1. Historical data plot:
plt.plot(np.arange(0, 252), history_prices, label="Historical Data (2025)", color='blue', alpha=0.6)

# 2. Plotting all the 100 simulated paths:
# We use a very low alpha (0.1) so we can see the "density" of the cloud.
# This creates the x-axis values for the future paths.
# We start from 252 because the future starts right after the historical data ends.
t_future = np.arange(252, 252 + STEPS + 1)

for n in range(NUM_SIMULATIONS):
    # This loop within a plot allows us to plot each of the 100 simulated paths row by row.
    # This creates a Monte Carlo "cloud" of possible futures.
    plt.plot(t_future, all_simulations[n, :], color='red', alpha=0.4)

# 3. Plotting the "Average" path (yellow):
# This shows the "Expected value" of all 100 futures.
avg_path = np.mean(all_simulations, axis=0)
plt.plot(t_future, avg_path, color='yellow', linewidth=2, label="Expected Future (Mean)")

# Formatting the Plot:
# We also add a vertical dashed line to indicate the split between past and future.
plt.axvline(x=252, color='black', linestyle='--', label="Today (The Split)")
plt.title(f"Monte Carlo Simulation: {NUM_SIMULATIONS} Possible Futures for 2026")
plt.xlabel("Trading Days")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --------------------------------------------------------------------------------------------------
# FINAL OUTPUT CALCULATION:
final_prices = all_simulations[:, -1]
avg_final_price = np.mean(final_prices)

print(f"Simulation of {NUM_SIMULATIONS} paths complete.")
print(f"Average Predicted Price for end of 2026: ${avg_final_price:.2f}")
# We can calculate the max and min of the final prices.
print(f"Highest Potential Price: ${np.max(final_prices):.2f}")
print(f"Lowest Potential Price: ${np.min(final_prices):.2f}")

# --------------------------------------------------------------------------------------------------

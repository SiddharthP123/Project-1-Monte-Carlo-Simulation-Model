"""
Geometric Brownian Motion (GBM) Stock Price Simulation:

This module demonstrates the application of Geometric Brownian Motion (GBM) 
for stock price simulation. It generates historical stock price data, 
calculates volatility and drift parameters, and then simulates future 
stock prices using the GBM model.

The simulation creates 252 days of historical data for 2025, calculates 
the annualized volatility (sigma) and drift (mu) from this data, and then 
uses these parameters to simulate one year of future stock prices for 2026.

Key components:
- Historical data generation using normal distribution.
- Volatility and drift calculation from log returns.
- GBM simulation using the calculated parameters.
- Visualization of historical vs simulated price paths.
- Final predicted price at end of the simulation period (2026).
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------
# SETUP & REPRODUCIBILITY:

np.random.seed(42)

# Say this is the first price our fake stock ever had (on Day 0 of our historical data).
# Day 0 is the first trading day of 2025.
ORIGINAL_PRICE = 100

# --------------------------------------------------------------------------------------------------
# GENERATING HISTORICAL DATA:
# We create 252 days of fake history with the respective paramters for mean & standard deviation.
# We use these values because they are reasonable estimates for a typical stock.
AVG_MEAN = 0.0005
AVG_STD = 0.015
TRADING_DAYS = 252

# Generating fake historical log returns and converting to 252 fake prices for our stock.
hist_shocks = np.random.normal(AVG_MEAN, AVG_STD, TRADING_DAYS)
history_prices = ORIGINAL_PRICE * np.exp(np.cumsum(hist_shocks))

# Essentially we have created fake stock prices for the whole year of 2025.

# --------------------------------------------------------------------------------------------------
# CALCULATING HISTORICAL VOLATILITY AND DRIFT:

# Calculate the log returns from the historical prices to get the volatility (sigma) and drift (mu).
log_returns = np.log(history_prices[1:] / history_prices[:-1])

# Calculate daily volatility (sigma) and annualize it.
daily_volatility = np.std(log_returns)

#Anualized volatility (sigma):
sigma = daily_volatility * np.sqrt(252)

# Calculate daily drift (mu) and annualize it.
daily_drift = np.mean(log_returns)

# Annualized drift (mu):
mu = daily_drift * 252

# At the end of 2025, we have mu and sigma, which we need to run our GBM simulation for 2026.

# --------------------------------------------------------------------------------------------------
# THE GBM SIMULATION (begins at the start of 2026):

# A: Lets define the parameters for our GBM simulation:
# Simulate 1 year into the future (the end of 2026).
T = 1.0

# Steps is the number of trading days in a year.
STEPS = 252        # Daily steps.
DT = T / STEPS     # Time step (1/252).

# The "future" starts where the "past" ended.
# This is the last trading price of 2025, our starting point for the GBM simulation in 2026.
St = history_prices[-1]

# --------------------------------------------------------------------------------------------------
# B: Creating empty arrays to store the simulated price path.
# We do this to avoid appending to a list in a loop, which is inefficient.
# Here, we save memory and speed up the simulation by pre-allocating the array with zeros.
simulated_path = np.zeros(STEPS + 1)

# We set the first price in our simulated path to be the last price from our historical data.
simulated_path[0] = St

# --------------------------------------------------------------------------------------------------
# C: Actual GBM Simulation Loop:
# Reset the random seed to ensure that the randomness is restricted to the GBM.
np.random.seed(None)

for i in range(STEPS):
    # Generate a random shock from a standard normal distribution for this time step.
    # This is one part of our Weiner Process, which adds randomness to our price path.
    # Z has a mean of 0 and a standard deviation of 1, which is what we need for the GBM formula.
    # Z is the unit version of shock, which is scaled by the root of time step.
    Z = np.random.standard_normal()
    # This ensures that the current stock prices updates upon every iteration.
    # It is initially set to the last price of 2025, but will evolve as we simulate forward.
    current_S = simulated_path[i]
    # The drift part is calculated:
    drift_part = mu * current_S * DT
    # The diffusion part is calculated:
    diffusion_part = sigma * current_S * np.sqrt(DT) * Z
    # The change in stock price (dS) is the sum of the drift and diffusion parts.
    dS = drift_part + diffusion_part
    # This step ensures iterative price updates, where the new price is = old price + change (dS).
    simulated_path[i+1] = current_S + dS

# --------------------------------------------------------------------------------------------------
# PRESENTING THE RESULTS:
# We will plot the historical data  and the simulated future path on the same graph.

# Here we are setting up the plot with a specific size for better visibility.
plt.figure(figsize=(12, 6))

# Plotting historical data (the past) in blue with some transparency (alpha=0.6).
plt.plot(np.arange(0, 252), history_prices, label="Historical Data (2025)", color='blue', alpha=0.6)

# Plot the simulated future path in red with a thicker line for emphasis.
plt.plot(np.arange(252, 252 + len(simulated_path)), simulated_path,
         label="GBM Simulation (Future)", color='red', linewidth=2)

# Formatting the Plot:
# We also add a vertical dashed line to indicate the split between past and future.
plt.axvline(x=252, color='black', linestyle='--', label="Today (The Split)")
plt.title(f"Stock Price Simulation: History vs. GBM Future\n(Sigma: {sigma:.2f}, Mu: {mu:.2f})")
plt.xlabel("Trading Days")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Simulation Complete.")
print(f"Final Price after 1 year: ${simulated_path[-1]:.2f}")

# --------------------------------------------------------------------------------------------------

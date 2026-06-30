"""
Monte Carlo Simulation for Financial Forecasting

This script performs a Monte Carlo simulation using Geometric Brownian Motion (GBM)
to forecast future stock prices. It generates historical data, calculates volatility
and drift parameters, and then simulates multiple future price paths to provide
statistical insights about potential outcomes for 2026.

The simulation includes:
- Historical data generation with normal distribution
- Volatility and drift calculation from historical returns
- Monte Carlo GBM simulation with 100 scenarios
- Dark theme visualization showing mean, max, and min paths
- Summary statistics of the forecast results
"""
import numpy as np
import matplotlib.pyplot as plt

# We will use Seaborn for enhanced color palettes and styling in our plots.
import seaborn as sns

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
# THE MONTE CARLO GBM SIMULATION:
NUM_SIMULATIONS = 100
T = 1.0
STEPS = 252
DT = T / STEPS
St = history_prices[-1]

all_simulations = np.zeros((NUM_SIMULATIONS, STEPS + 1))
all_simulations[:, 0] = St

np.random.seed(None)
for n in range(NUM_SIMULATIONS):
    for i in range(STEPS):
        Z = np.random.standard_normal()
        current_S = all_simulations[n, i]
        dS = (mu * current_S * DT) + (sigma * current_S * np.sqrt(DT) * Z)
        all_simulations[n, i+1] = current_S + dS

# --------------------------------------------------------------------------------------------------
# DATA PRESENTATION AND PLOTTING:

# Set the dark theme using Seaborn and Matplotlib.
plt.style.use('dark_background')
# rc means "runtime configuration" - we can set specific parameters for the plot aesthetics.
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117"})

# Initializing 2 variables to the plot.
fig, ax = plt.subplots(figsize=(14, 8))

# --------------------------------------------------------------------------------------------------
# Working on the finer grid-lines, known as ticks.

# Enabling the "in-between" ticks.
ax.minorticks_on()

# Style the Major Grid (the main lines).
ax.grid(which='major', color='#ffffff', linestyle='-', alpha=0.1)
# Style the Minor Grid (the new extra light lines)
ax.grid(which='minor', color='#ffffff', linestyle=':', alpha=0.05)

# --------------------------------------------------------------------------------------------------
# Define the X-axis starting from Day 0 of the simulation (Day 252 of the total timeline).
t_future = np.arange(0, STEPS + 1)

# Generate 100 random colors using a Seaborn palette.
# We use 'husl' to get a wide variety of distinct, vibrant colors.
colors = sns.color_palette("husl", NUM_SIMULATIONS)

# 1. Plot the 100 individual simulations.
for n in range(NUM_SIMULATIONS):
    plt.plot(t_future, all_simulations[n, :], color=colors[n], alpha=0.3, linewidth=0.8)

# 2. Identify and plot statistical paths (Mean, Max, Min).

avg_path = np.mean(all_simulations, axis=0)

# Find the indices of the paths that ended at the absolute max and min.
# We use argmax instead of max to get the index of the path, similarly for argmin.
max_idx = np.argmax(all_simulations[:, -1])
min_idx = np.argmin(all_simulations[:, -1])

# Plotting the mean line:
plt.plot(t_future, avg_path, color='white', linewidth=2, label="Expected Future (Mean).")

# Plotting the min and max lines:
plt.plot(t_future, all_simulations[max_idx, :], color='green', linewidth=2.5,
         linestyle=':', label="Best Case (Max).")
plt.plot(t_future, all_simulations[min_idx, :], color='red', linewidth=2.5,
         linestyle=':', label="Worst Case (Min).")

# --------------------------------------------------------------------------------------------------
# MARKERS AND ANNOTATIONS:

# zorder tells us which layer the marker is at.
# Using textcoords = 'offset points", we can offset our label x,y distance to the data point.
# alpha controls the opacity.

# 1. Historical End Point (The Start of 2026):
plt.scatter(0, St, color='white', marker='o', s=50,
            linewidths=1, zorder=5, label="2025 Final Price.")
plt.annotate(f'Start: ${St:.2f}', (0, St),
             xytext=(-30, 30), textcoords='offset points',
             color='white', alpha=0.8)

# 2. Mean End Point:
plt.scatter(STEPS, avg_path[-1],
            color='white', marker='x', s=75, linewidths=1, zorder=5,
            label="Est. Mean End Price.")
plt.annotate(f'Mean: ${avg_path[-1]:.2f}', (STEPS, avg_path[-1]),
             xytext=(-80, 15), textcoords='offset points',
             color='white', alpha = 0.8)

# 3. Max End Point:
plt.scatter(STEPS, all_simulations[max_idx, -1],
            color='green', marker='x', s=75, linewidths=1, zorder=5,
            label="Est. Max End Price.")
plt.annotate(f'Max: ${all_simulations[max_idx, -1]:.2f}', (STEPS, all_simulations[max_idx, -1]),
             xytext=(-80, 15), textcoords='offset points',
             color='green', alpha = 0.8)

# 4. Min End Point:
plt.scatter(STEPS, all_simulations[min_idx, -1],
            color='red', marker='x', s=75, linewidths=1, zorder=5,
            label="Est. Min End Price.")
plt.annotate(f'Min: ${all_simulations[min_idx, -1]:.2f}', (STEPS, all_simulations[min_idx, -1]),
             xytext=(-80, -15), textcoords='offset points',
             color='red', alpha = 0.8)

# --------------------------------------------------------------------------------------------------
# SCREEN LAYOUT FORMATTING:

plt.title("Monte Carlo Forecast - 2026 Projections:", fontsize=16, fontweight='bold', color='white')
plt.xlabel("Trading Days (into 2026):", fontsize=12, color='white')
plt.ylabel("Projected Price ($):", fontsize=12, color='white')

# Customizing the legend.
plt.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=10, labelcolor='white')

# The tight layout automatically ensures all the data and subplots fit perfectly.
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------
# FINAL OUTPUT FOR USER:
final_prices = all_simulations[:, -1]
avg_final_price = np.mean(final_prices)

print(f"Simulation of {NUM_SIMULATIONS} paths complete.")
print(f"Average Predicted Price for end of 2026: ${avg_final_price:.2f}")
print(f"Highest Potential Price: ${np.max(final_prices):.2f}")
print(f"Lowest Potential Price: ${np.min(final_prices):.2f}")

# --------------------------------------------------------------------------------------------------

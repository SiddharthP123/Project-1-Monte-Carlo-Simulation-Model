"""
Monte Carlo Simulation for Financial Forecasting:

This script performs a Monte Carlo simulation using Geometric Brownian Motion (GBM)
to forecast future stock prices for the year 2026. The simulation generates multiple
price paths based on historical volatility and drift parameters derived from 2025 data.

Key Features:
- Generates historical price data using normal distribution parameters.
- Calculates historical volatility and drift from the generated data.
- Performs Monte Carlo GBM simulation with 100 different paths.
- Creates visualizations showing expected future prices, best/worst case scenarios.
- Exports detailed analysis report to a text file with statistical metrics.

The script produces both visual output (matplotlib chart) and textual analysis
including expected prices, probability of profit, and price ranges for 2026.
"""

import datetime
# This is to ensure report generation occurs on accurate date.

import numpy as np
import matplotlib.pyplot as plt
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
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117"})
fig, ax = plt.subplots(figsize=(14, 8))

# Additional gridlines:
ax.minorticks_on()
ax.grid(which='major', color='#ffffff', linestyle='-', alpha=0.1)
ax.grid(which='minor', color='#ffffff', linestyle=':', alpha=0.05)

t_future = np.arange(0, STEPS + 1)
colors = sns.color_palette("husl", NUM_SIMULATIONS)

# 100 individual simulations:
for n in range(NUM_SIMULATIONS):
    plt.plot(t_future, all_simulations[n, :], color=colors[n], alpha=0.3, linewidth=0.8)

# Mean, Max, Min:

avg_path = np.mean(all_simulations, axis=0)

max_idx = np.argmax(all_simulations[:, -1])
min_idx = np.argmin(all_simulations[:, -1])

plt.plot(t_future, avg_path, color='white', linewidth=2, label="Expected Future (Mean).")
plt.plot(t_future, all_simulations[max_idx, :], color='green', linewidth=2.5,
         linestyle=':', label="Best Case (Max).")
plt.plot(t_future, all_simulations[min_idx, :], color='red', linewidth=2.5,
         linestyle=':', label="Worst Case (Min).")

# --------------------------------------------------------------------------------------------------
# MARKERS AND ANNOTATIONS:

plt.scatter(0, St, color='white', marker='o', s=50,
            linewidths=1, zorder=5, label="2025 Final Price.")
plt.annotate(f'Start: ${St:.2f}', (0, St),
             xytext=(-30, 30), textcoords='offset points',
             color='white', alpha=0.8)

plt.scatter(STEPS, avg_path[-1],
            color='white', marker='x', s=75, linewidths=1, zorder=5,
            label="Est. Mean End Price.")
plt.annotate(f'Mean: ${avg_path[-1]:.2f}', (STEPS, avg_path[-1]),
             xytext=(-80, 15), textcoords='offset points',
             color='white', alpha = 0.8)

plt.scatter(STEPS, all_simulations[max_idx, -1],
            color='green', marker='x', s=75, linewidths=1, zorder=5,
            label="Est. Max End Price.")
plt.annotate(f'Max: ${all_simulations[max_idx, -1]:.2f}', (STEPS, all_simulations[max_idx, -1]),
             xytext=(-80, 15), textcoords='offset points',
             color='green', alpha = 0.8)

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

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------
# EXPORTING ANALYSIS TO TEXT FILE:

# We separate our visual chart from our analysis report for visual clarity.

# Calculating the final statistics:

final_prices = all_simulations[:, -1]
avg_final_price = np.mean(final_prices)
profitable_paths = np.sum(final_prices > St)
prob_of_profit = (profitable_paths / NUM_SIMULATIONS) * 100

# Getting the current date and time for our report:

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Creating a multi-line f-string.
report_content = f"""
==========================================================
MONTE CARLO SIMULATION REPORT: 2026 FORECAST
==========================================================
Generated on: {current_time}

Historical Baseline (2025):
- Final Price (St): ${St:.2f}
- Annualized Drift (mu): {mu:.2%}
- Annualized Volatility (sigma): {sigma:.2%}

Simulation Parameters (2026):
- Number of Paths: {NUM_SIMULATIONS}
- Trading Horizon: 252 Days (1 Year)

Quantitative Results:
- Expected Price: ${avg_final_price:.2f}
- Probability of Profit: {prob_of_profit:.1f}%
- Max Potential Price: ${np.max(final_prices):.2f}
- Min Potential Price: ${np.min(final_prices):.2f}
- Total Forecast Range: ${np.max(final_prices) - np.min(final_prices):.2f}
==========================================================
"""

# Writing to and creating a text file:
# File management in Python (notes):

# - The 'with open' block opens a file with the specified file name.
# - The 'with' ensures that as soon as the code is executed, it is immediately saved & closed.
# - The 'w' parameter overwrites any existing content.
# - The 'write' command actually writes the content into the file.
# - This is done from our variable stored as a multi-line f-string.

# - The encoding parameter is not necessary, just to avoid linter warnings. Good practice.
with open("part6_simulation_report.txt", "w", encoding="utf-8") as f:
    f.write(report_content)

# Output for us to see regarding final report generation:
print(f"\nAnalysis report exported to 'part6_simulation_report.txt' at {current_time}")

# --------------------------------------------------------------------------------------------------

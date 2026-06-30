"""
Monte Carlo Simulation for Stock Price Forecasting:

This module implements a Geometric Brownian Motion (GBM) model to forecast
stock prices using real historical market data. It downloads historical
price data for a user-specified ticker, calculates real-world parameters
(drift and volatility), and runs Monte Carlo simulations to project future
price movements for the next year.

The simulation generates multiple price paths and provides statistical
analysis including expected returns, probability of profit, and risk metrics.
Results are visualized using matplotlib and exported to a text report stored
in a determined sub-folder.

Usage:
    Run the script and enter a stock ticker symbol when prompted.
    The script will fetch 2 years of historical data and forecast 2026 prices.
"""
# We are using this library for file management purposes on our operating system.
import os

import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# We are now using yahoo finance to download real historical data for a selected ticker.
import yfinance as yf

# --------------------------------------------------------------------------------------------------
# SETUP & DATA INGESTION:

# Allowing the user to select any ticker from the library:
TICKER = input("Enter the Stock Ticker (e.g., AAPL, TSLA, NVDA, ^GSPC): ").upper()


# We will be using the last 2 years of data for the selected ticker:
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

print(f"--- Fetching Market Data for {TICKER} ---")

# Downloading the data for the last 2 years:
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

# We extract the 'Close' prices and flatten them into a 1D NumPy array.
history_prices = data['Close'].values.flatten()

# If the download fails or the data is empty, then we must handle the error.
if len(history_prices) == 0:
    print(f"Error: Could not find data for {TICKER}. Check ticker or connection.")
else:
    # ----------------------------------------------------------------------------------------------
    # CALCULATING REAL-WORLD PARAMETERS:

    # Calculate log returns from the actual historical closing prices:
    log_returns = np.log(history_prices[1:] / history_prices[:-1])

    # Calculate Real Annualized Volatility (Sigma):
    sigma = np.std(log_returns) * np.sqrt(252)

    # Calculate Real Annualized Drift (Mu):
    mu = np.mean(log_returns) * 252

    # The simulation starts at the very last recorded price of 2025:
    St = history_prices[-1]

    print(f"Successfully derived parameters for {TICKER}:")
    print(f"Annualized Volatility (Sigma): {sigma:.2%}")
    print(f"Annualized Drift (Mu): {mu:.2%}")
    print(f"Last Market Price: ${St:.2f}\n")

    # ----------------------------------------------------------------------------------------------
    # THE MONTE CARLO GBM SIMULATION:
    NUM_SIMULATIONS = 100
    T = 1.0
    STEPS = 252
    DT = T / STEPS

    all_simulations = np.zeros((NUM_SIMULATIONS, STEPS + 1))
    all_simulations[:, 0] = St

    # Resetting seed to None for the future projections to stay stochastic:
    np.random.seed(None)
    for n in range(NUM_SIMULATIONS):
        for i in range(STEPS):
            Z = np.random.standard_normal()
            current_S = all_simulations[n, i]
            # GBM Formula: dS = (mu * S * dt) + (sigma * S * sqrt(dt) * Z):
            dS = (mu * current_S * DT) + (sigma * current_S * np.sqrt(DT) * Z)
            all_simulations[n, i+1] = current_S + dS

    # ----------------------------------------------------------------------------------------------
    # DATA PRESENTATION AND PLOTTING:
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117"})
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.minorticks_on()
    ax.grid(which='major', color='#ffffff', linestyle='-', alpha=0.1)
    ax.grid(which='minor', color='#ffffff', linestyle=':', alpha=0.05)

    t_future = np.arange(0, STEPS + 1)
    colors = sns.color_palette("husl", NUM_SIMULATIONS)

    # 1. Plot the 100 individual simulations:
    for n in range(NUM_SIMULATIONS):
        plt.plot(t_future, all_simulations[n, :], color=colors[n], alpha=0.3, linewidth=0.8)

    # 2. Statistical paths:
    avg_path = np.mean(all_simulations, axis=0)
    max_idx = np.argmax(all_simulations[:, -1])
    min_idx = np.argmin(all_simulations[:, -1])

    plt.plot(t_future, avg_path,
             color='white', linewidth=2,
             label="Expected Future (Mean).")
    plt.plot(t_future, all_simulations[max_idx, :],
             color='green', linewidth=2.5, linestyle=':',
             label="Best Case (Max).")
    plt.plot(t_future, all_simulations[min_idx, :],
             color='red', linewidth=2.5, linestyle=':',
             label="Worst Case (Min).")

    # 3. Markers and annotations:
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

    # 4. Formatting:
    plt.title(f"2026 Forecast: {TICKER} Monte Carlo Analysis",
              fontsize=16, fontweight='bold', color='white', pad=20)
    plt.xlabel("Trading Days (into 2026)",
               fontsize=12, color='gray')
    plt.ylabel("Projected Price ($)",
               fontsize=12, color='gray')
    plt.legend(facecolor='#161b22', edgecolor='#30363d',
               fontsize=10, labelcolor='white', loc='upper left')

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------------------------------
    # EXPORTING ANALYSIS TO TEXT FILE:
    final_prices = all_simulations[:, -1]
    avg_final_price = np.mean(final_prices)
    profitable_paths = np.sum(final_prices > St)
    prob_of_profit = (profitable_paths/ NUM_SIMULATIONS) * 100
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # FILE MANAGEMENT FOR ORGANISATION:

    # Defining the sub-folder name:
    SUB_FOLDER = "part7_tested_tickers"

    # First, we need to ensure this folder exists. If not, create one.
    if not os.path.exists(SUB_FOLDER):
        os.makedirs(SUB_FOLDER)

    # Defining the final path:
    file_path = os.path.join(SUB_FOLDER, f"{TICKER}_2026_forecast.txt")

    report_content = f"""
==========================================================
MONTE CARLO SIMULATION REPORT: {TICKER} 2026 FORECAST
==========================================================
Generated on: {current_time}

Historical Market Context ({START_DATE} to {END_DATE}):
- Terminal Price (St): ${St:.2f}
- Realized Annual Drift (mu): {mu:.2%}
- Realized Annual Volatility (sigma): {sigma:.2%}

Simulation Parameters (2026):
- Number of Scenarios: {NUM_SIMULATIONS}
- Time Horizon: 1.0 Year (252 Trading Days)

Quantitative Outcomes:
- Expected End Price: ${avg_final_price:.2f}
- Probability of Gain: {prob_of_profit:.1f}%
- Highest Observed Outcome: ${np.max(final_prices):.2f}
- Lowest Observed Outcome: ${np.min(final_prices):.2f}
- Absolute Risk Range: ${np.max(final_prices) - np.min(final_prices):.2f}
==========================================================
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Analysis report for {TICKER} exported to '{file_path}' at {current_time}")

# --------------------------------------------------------------------------------------------------

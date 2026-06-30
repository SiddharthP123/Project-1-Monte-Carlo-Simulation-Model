"""
Monte Carlo Simulation for Stock Price Forecasting with Value at Risk (VaR) Analysis:

This script performs a Monte Carlo simulation using Geometric Brownian Motion (GBM) 
to forecast stock prices and calculate Value at Risk (VaR) metrics. It fetches 
real market data using yfinance, calculates volatility and drift parameters, 
runs multiple simulation scenarios, and provides risk analysis including 
95% confidence VaR calculations.

Key Features:
- Real-time market data fetching via yfinance.
- Monte Carlo simulation with customizable parameters.
- Value at Risk (VaR) calculation at 95% confidence level.
- Comprehensive visualization of simulation results.
- Export of analysis results to text files.

Usage: Run the script and input a stock ticker symbol and investment capital
to generate a 2026 forecast with risk metrics.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# --------------------------------------------------------------------------------------------------
# SETUP & DATA INGESTION:

TICKER = input("Enter the Stock Ticker (e.g., AAPL, TSLA, NVDA, ^GSPC): ").upper()
INVESTMENT_CAPITAL = float(input(f"Enter your total investment capital in {TICKER} (e.g. $1000): "))

# AUTOMATED DATE RANGE: Exactly 2 years before today, for dynamic data analysis.
end_dt = datetime.date.today()
start_dt = end_dt - datetime.timedelta(days=2*365)

START_DATE = start_dt.strftime("%Y-%m-%d")
END_DATE = end_dt.strftime("%Y-%m-%d")

print(f"--- Fetching Market Data for {TICKER} ---")
print(f"Lookback Period: {START_DATE} to {END_DATE}")

data = yf.download(TICKER, start=START_DATE, end=END_DATE)
history_prices = data['Close'].values.flatten()

if len(history_prices) == 0:
    print(f"Error: Could not find data for {TICKER}. Check ticker or connection.")
else:
    # ----------------------------------------------------------------------------------------------
    # CALCULATING REAL-WORLD PARAMETERS:
    log_returns = np.log(history_prices[1:] / history_prices[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)
    mu = np.mean(log_returns) * 252
    St = history_prices[-1]

    print(f"Successfully derived parameters for {TICKER}:")
    print(f"Annualized Volatility (Sigma): {sigma:.2%}")
    print(f"Annualized Drift (Mu): {mu:.2%}")
    print(f"Last Market Price: ${St:.2f}\n")

    # ----------------------------------------------------------------------------------------------
    # THE MONTE CARLO GBM SIMULATION:
    NUM_SIMULATIONS = 100
    T, STEPS = 1.0, 252
    DT = T / STEPS

    all_simulations = np.zeros((NUM_SIMULATIONS, STEPS + 1))
    all_simulations[:, 0] = St

    np.random.seed(None)
    for n in range(NUM_SIMULATIONS):
        for i in range(STEPS):
            Z = np.random.standard_normal()
            current_S = all_simulations[n, i]
            dS = (mu * current_S * DT) + (sigma * current_S * np.sqrt(DT) * Z)
            all_simulations[n, i+1] = current_S + dS

    # ----------------------------------------------------------------------------------------------
    # RISK METRICS (VaR):
    final_prices = all_simulations[:, -1]

    # Calculate 95% Value at Risk (the bottom 5th percentile):
    var_95_price = np.percentile(final_prices, 5)
    percent_loss_at_var = (var_95_price - St) / St
    capital_at_risk = INVESTMENT_CAPITAL * abs(percent_loss_at_var)

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

    for n in range(NUM_SIMULATIONS):
        plt.plot(t_future, all_simulations[n, :], color=colors[n], alpha=0.3, linewidth=0.8)

    avg_path = np.mean(all_simulations, axis=0)
    max_idx, min_idx = np.argmax(final_prices), np.argmin(final_prices)

    plt.plot(t_future, avg_path,
             color='white', linewidth=2,
             label="Expected Future (Mean).")
    plt.plot(t_future, all_simulations[max_idx, :],
             color='green', linewidth=2.5, linestyle='--',
             label="Best Case (Max).")
    plt.plot(t_future, all_simulations[min_idx, :],
             color='red', linewidth=2.5, linestyle='--',
             label="Worst Case (Min).")

    # ADDING THE VaR LINE TO THE CHART:
    plt.axhline(y=var_95_price,
                color='white', linestyle='--', linewidth=2,
                label="95% Value at Risk (VaR).")

    # Markers and annotations:
    plt.scatter(0, St,
                color='white', marker='o', s=50, zorder=5,
                label="Current Price.")
    plt.annotate(f'Start: ${St:.2f}', (0, St),
                 xytext=(-30, 30), textcoords='offset points',
                 color='white', alpha=0.8)

    plt.scatter(STEPS, avg_path[-1],
                color='white', marker='x', s=75, zorder=5,
                label="Est. Mean End Price.")
    plt.annotate(f'Mean: ${avg_path[-1]:.2f}', (STEPS, avg_path[-1]),
                 xytext=(-80, 15), textcoords='offset points',
                 color='white', alpha=0.8)

    plt.scatter(STEPS, max_price := np.max(final_prices),
                color='green', marker='x', s=75, zorder=5,
                label="Est. Max End Price.")
    plt.annotate(f'Max: ${max_price:.2f}', (STEPS, max_price),
                 xytext=(-80, 15), textcoords='offset points',
                 color='green', alpha=0.8)

    plt.scatter(STEPS, min_price := np.min(final_prices),
                color='red', marker='x', s=75, zorder=5,
                label="Est. Min End Price.")
    plt.annotate(f'Min: ${min_price:.2f}', (STEPS, min_price),
                 xytext=(-80, -15), textcoords='offset points',
                 color='red', alpha=0.8)

    # VaR Text Label:
    plt.text(5, var_95_price + (var_95_price*0.05),
             f'95% VaR Threshold: ${var_95_price:.2f}',
             color='white')
    # Axis Labels:
    plt.title(f"2026 Forecast - {TICKER} Monte Carlo Analysis:",
              fontsize=16, fontweight='bold', color='white', pad=20)
    plt.xlabel(f"Trading Days (from {END_DATE}):",
               fontsize=12, color='white')
    plt.ylabel("Projected Share Price ($):",
               fontsize=12, color='white')
    plt.legend(facecolor='#161b22', edgecolor='#30363d',
               fontsize=10, labelcolor='white', loc='upper left')

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------------------------------
    # EXPORTING ANALYSIS TO TEXT FILE:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SUB_FOLDER = "part8_tested_tickers"

    if not os.path.exists(SUB_FOLDER):
        os.makedirs(SUB_FOLDER)

    file_path = os.path.join(SUB_FOLDER, f"{TICKER}_2026_forecast.txt")

    report_content = f"""
==========================================================
MONTE CARLO SIMULATION REPORT - {TICKER} 2026 FORECAST:
==========================================================
Generated on: {current_time}

Historical Market Context ({START_DATE} to {END_DATE}):
- Current Share Price (St): ${St:.2f}
- Realized Annual Drift (mu): {mu:.2%}
- Realized Annual Volatility (sigma): {sigma:.2%}

Simulation Parameters (2026):
- Number of Scenarios: {NUM_SIMULATIONS}
- Time Horizon: 1.0 Year (252 Trading Days)

Quantitative Outcomes:
- Expected End Share Price: ${np.mean(final_prices):.2f}
- Probability of Gain: {(np.sum(final_prices > St)/NUM_SIMULATIONS)*100:.1f}%
- Max. End Share Price: ${np.max(final_prices):.2f}
- Min. End Share Price: ${np.min(final_prices):.2f}

Risk Management (95% Confidence):
- Initial Capital Invested: ${INVESTMENT_CAPITAL:,.2f}
- 95% Annnual VaR Price Level: ${var_95_price:.2f}
- Projected Cash at Risk: ${capital_at_risk:,.2f}
- Min. Expected Investment Value: ${INVESTMENT_CAPITAL - capital_at_risk:,.2f}
==========================================================
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Analysis report for {TICKER} exported to '{file_path}' at {current_time}")

# --------------------------------------------------------------------------------------------------

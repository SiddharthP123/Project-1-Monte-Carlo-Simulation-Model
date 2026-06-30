"""
Monte Carlo Simulation with Merton Jump Diffusion Model for Black Swan Analysis:

This script performs a comprehensive financial risk analysis using a Monte carlo simulation
combined with the Merton jump diffusion model to account for rare but significant market
events (black swans). The analysis includes:
- Historical data retrieval and parameter estimation.
- Merton jump diffusion calibration for black swan events.
- Monte carlo simulation with jump components.
- Value at risk (VaR) calculation and risk metrics.
- Data visualization and report generation.

The script allows users to input a stock ticker symbol, investment capital,
and custom jump parameters to simulate various market crash scenarios.
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# --------------------------------------------------------------------------------------------------
# 1. SETUP & DATA INGESTION:

TICKER = input("Enter the stock ticker (e.g. AAPL, TSLA, NVDA, ^GSPC): ").upper()
INVESTMENT_CAPITAL = float(input(f"Enter your total investment capital in {TICKER}: "))

# Automated 2-year lookback period (for dynamic data analysis).
end_dt = datetime.date.today()
start_dt = end_dt - datetime.timedelta(days=2*365)
START_DATE = start_dt.strftime("%Y-%m-%d")
END_DATE = end_dt.strftime("%Y-%m-%d")

print(f"--- Fetching market data for {TICKER} ---")
print(f"Lookback period: {START_DATE} to {END_DATE}")

data = yf.download(TICKER, start=START_DATE, end=END_DATE)
history_prices = data['Close'].values.flatten()

if len(history_prices) == 0:
    print(f"Error: Could not find data for {TICKER}. Check ticker or connection.")
else:
    # ----------------------------------------------------------------------------------------------
    # 2. CALCULATING REAL-WORLD PARAMETERS:
    log_returns = np.log(history_prices[1:] / history_prices[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)
    mu = np.mean(log_returns) * 252
    St = history_prices[-1]

    print(f"Successfully derived parameters for {TICKER}:")
    print(f"Realised annual volatility (sigma): {sigma:.2%}")
    print(f"Realised annual drift (mu): {mu:.2%}")
    print(f"Current market share price: ${St:.2f}\n")
    # ----------------------------------------------------------------------------------------------
    # 3. MERTON JUMP DIFFUSION CALIBRATION (BLACK SWAN):
    print("\n--- Merton jump diffusion (MJD) calibration ---")
    # Allowing the user to input the required parameters:

    # A - Jump frequency (lambda): Number of expected jumps per year.
    lam = float(input("Enter jump frequency (lambda) - e.g. 1 crash in 2 years = 0.5: "))

    # B - Mean jump size (mu_j): Center of the crash distribution & its severity.
    mu_j = float(input("Enter mean jump size (mu_j) - e.g. a crash of 20% = -0.20: "))

    # C - Jump uncertainty (sigma_j): Std. dev/variation in crash size.
    sigma_j = float(input("Enter jump uncertainty (sigma_j) - e.g. a variation of 5% = 0.05: "))

    # Calculating the drift correction factor (k).
    # Remember that drift (mu) is the mean annualised logarithmic return.
    # This ensures that mu is preserved despite the manual jumps.
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    print(f"Input locked: {lam} events/year | Avg. crash: {mu_j:.1%} | Std. dev: {sigma_j:.1%}\n")

    # ----------------------------------------------------------------------------------------------
    # 4. UPDATED MONTE CARLO SIMULATION ENGINE:
    NUM_SIMULATIONS = 100
    T, STEPS = 1.0, 252
    DT = T / STEPS

    all_simulations = np.zeros((NUM_SIMULATIONS, STEPS + 1))
    all_simulations[:, 0] = St

    np.random.seed(None)
    for n in range(NUM_SIMULATIONS):
        for i in range(STEPS):
            # A - Normal/continuous diffusion (Wiener process):
            Z = np.random.standard_normal()
            diffusion = sigma * np.sqrt(DT) * Z
            # B - Discontinuous jump component (Poisson process):
            JUMP = 0
            # Check if a jump occurs in this time step/is triggered today.
            if np.random.poisson(lam * DT) > 0:
                # If triggered, the jump magnitude is pulled from normal(mu_j, sigma_j^2).
                # In maths, we denote the normal using variance as the last parameter, so sq.
                # In Python, we denote the normal using std dev as the last parameter, so no sq.
                JUMP = np.random.normal(mu_j, sigma_j)
            # C - Compensated drift:
            drift = (mu - lam * k) * DT
            current_S = all_simulations[n, i]
            # D - Update the price:
            # We use the exponential form for better stability with jumps.
            all_simulations[n, i+1] = current_S * np.exp(drift + diffusion + JUMP)

    # ----------------------------------------------------------------------------------------------
    # 5. RISK METRICS/VaR:
    final_prices = all_simulations[:, -1]

    # Calculating 95% annual value at risk:

    # Price level which represents the (bottom) 5th percentile of the final simulated prices.
    var_95_price = np.percentile(final_prices, 5)
    # Percentage loss at the 95% VaR price level.
    percent_loss_at_var = (var_95_price - St) / St
    # Capital at risk = product of the capital invested and the abs(percentage loss at 95% VaR).
    capital_at_risk = INVESTMENT_CAPITAL * abs(percent_loss_at_var)

    # ----------------------------------------------------------------------------------------------
    # 6. DATA PRESENTATION AND PLOTTING:

    # Background styles and setting the plot sizes.
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117"})
    fig, ax = plt.subplots(figsize=(14, 8))

    # Setting up the gridlines (ticks).
    ax.minorticks_on()
    ax.grid(which='major', color='#ffffff', linestyle='-', alpha=0.1)
    ax.grid(which='minor', color='#ffffff', linestyle=':', alpha=0.05)
    ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='white')

    # Setting the axis up and allowing random color generation for our simulations.
    t_future = np.arange(0, STEPS + 1)
    colors = sns.color_palette("husl", NUM_SIMULATIONS)

    for n in range(NUM_SIMULATIONS):
        plt.plot(t_future, all_simulations[n, :], color=colors[n], alpha=0.3, linewidth=0.8)

    # Average, min, and max paths.
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
    # VaR line addition to the chart.
    plt.axhline(y=var_95_price,
                color='white', linestyle='--', linewidth=2,
                label="95% Value at Risk (VaR).")
     # Markers and annotations.
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

    # VaR text label.
    plt.text(5, var_95_price + (var_95_price*0.05),
             f'95% VaR Threshold: ${var_95_price:.2f}',
             color='white')

    # Axis labels.
    plt.title(f"Monte Carlo Simulation & Black Swan Analysis (MJD Model):{TICKER}",
              fontsize=16, fontweight='bold', color='white')
    plt.xlabel(f"Trading Days (from {END_DATE}):",
               fontsize=12, color='white')
    plt.ylabel("Projected Share Price ($):",
               fontsize=12, color='white')
    plt.legend(facecolor='#161b22', edgecolor='#30363d',
               fontsize=10, labelcolor='white', loc='upper left')

    plt.tight_layout()
    plt.show()
    # ----------------------------------------------------------------------------------------------
    # 7. EXPORTING ANALYSIS TO TEXT FILE:

    # Integration of actual date and file management.
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SUB_FOLDER = "part9_tested_tickers"
    if not os.path.exists(SUB_FOLDER):
        os.makedirs(SUB_FOLDER)
    file_path = os.path.join(SUB_FOLDER, f"{TICKER}_black_swan_forecast.txt")

    report_content = f"""
==========================================================
MONTE CARLO + MERTON JUMP DIFFUSION REPORT: {TICKER}
==========================================================
Generated on: {current_time}

Historical market context ({START_DATE} to {END_DATE}):
- Current market share price (St): ${St:.2f}
- Realised annual drift (mu): {mu:.2%}
- Realized annual volatility (sigma): {sigma:.2%}

Simulation parameters (2026):
- Number of scenarios: {NUM_SIMULATIONS}
- Time horizon: 1.0 Year (252 trading days)

MJD calibration (Black swan stress test):
- Jump frequency (lambda): {lam} events/year
- Mean jump size (mu_j) - Avg. crash size/severity: {mu_j:.1%}
- Jump uncertainty (sigma_j) - Crash std. dev./variation: {sigma_j:.1%}
- Drift correction (k): {k:.4f}

Quantitative outcomes:
- Expected end share price: ${np.mean(final_prices):.2f}
- Probability of gain: {(np.sum(final_prices > St)/NUM_SIMULATIONS)*100:.1f}%
- Max. end share price: ${np.max(final_prices):.2f}
- Min. end share price: ${np.min(final_prices):.2f}

Risk management (95% confidence):
- Initial capital invested: ${INVESTMENT_CAPITAL:,.2f}
- 95% annnual VaR price level: ${var_95_price:.2f}
- Projected cash at risk: ${capital_at_risk:,.2f}
- Min. expected investment value: ${INVESTMENT_CAPITAL - capital_at_risk:,.2f}
==========================================================
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Analysis report for {TICKER} exported to '{file_path}' at {current_time}")

# --------------------------------------------------------------------------------------------------

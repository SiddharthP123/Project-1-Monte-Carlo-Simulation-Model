# Learning Python with Finance
### A 10-Part Project: From Stock Returns to a Live Monte Carlo Web App

This repository documents a progressive, self-taught journey applying Python to quantitative finance. Each part builds on the last — starting from manually computing stock returns in a spreadsheet-style script, and ending with a fully deployed, interactive Monte Carlo simulation dashboard accessible to anyone on the web.

**Live App (Part 10):** [project-1-monte-carlo-simulation-model-siddharth-premanand.streamlit.app](https://project-1-monte-carlo-simulation-model-siddharth-premanand.streamlit.app)

---

## Repository Structure

```
├── part1_returns_and_volatility.py       # Arithmetic vs log returns
├── part2_volatility_and_drift.py         # Annualizing sigma and mu
├── part3_gbm_application.py              # Single GBM price path
├── part4_montecarlo.py                   # 100-path Monte Carlo (first version)
├── part5_montecarlo_new.py               # Improved visualization
├── part6_added_analysis.py               # Probability of profit + report export
├── part7_realdata.py                     # Real market data via yfinance
├── part8_using_var.py                    # Value at Risk (VaR) integration
├── part9_black_swan_merton_jump.py       # Merton Jump Diffusion model
├── part10_app.py                         # Streamlit web app (live dashboard)
├── part5_montecarlo_snapshot.png
├── part8_adv_montecarlo_snapshot.png
├── part9_black_swan_monte_carlo.png
├── part6_simulation_report.txt
├── part7_tested_tickers/
├── part8_tested_tickers/
├── part9_tested_tickers/
├── part9_understanding_mjd_effects/
├── requirements.txt
└── runtime.txt
```

---

## The Journey: Part by Part

### Part 1 — Returns and Volatility
**File:** `part1_returns_and_volatility.py`

The starting point. Given a fixed list of 10 daily closing prices, this script manually computes both types of daily return and then measures how much they vary (volatility).

**What it does:**
- Calculates arithmetic returns and log returns side by side
- Computes the standard deviation of each
- Displays everything in a clean pandas DataFrame

**Key insight:** Log returns add up correctly across time (they are *time-additive*), while arithmetic returns can mislead you when compounding is involved.

---

### Part 2 — Volatility and Drift from Random Prices
**File:** `part2_volatility_and_drift.py`

Instead of a fixed dataset, we now generate 252 days of random price data (mimicking one trading year) and extract two fundamental parameters from it: how much the price moves (volatility) and in which direction on average (drift).

**What it does:**
- Generates synthetic daily prices using a normal distribution
- Calculates daily and annualized volatility (sigma)
- Calculates daily and annualized drift (mu)

**Key insight:** Volatility scales with the *square root* of time, while drift scales *linearly*. This is why you multiply daily volatility by √252 to annualize it, but multiply daily drift by 252.

---

### Part 3 — Geometric Brownian Motion (First Simulation)
**File:** `part3_gbm_application.py`

The first real simulation. We use the parameters derived from fake "2025" history to simulate a single possible price path for "2026" using the GBM formula step by step in a loop.

**What it does:**
- Generates one year of fake historical prices
- Extracts sigma and mu from that history
- Runs a single GBM simulation for the next year
- Plots historical vs simulated price on one chart

**Key insight:** Each step of the simulation is: `dS = (mu × S × dt) + (sigma × S × √dt × Z)`, where Z is a random number drawn from a standard normal distribution every day.

---

### Part 4 — Monte Carlo Simulation (100 Paths)
**File:** `part4_montecarlo.py`

Instead of simulating one future, we simulate 100 — each one a different random outcome. Together, these paths form a probability "cloud" of where the stock price might go.

**What it does:**
- Stores all 100 simulated paths in a 2D NumPy array (rows = simulations, columns = days)
- Plots all paths plus a mean (expected value) overlay
- Reports the average, maximum, and minimum final prices

**Key insight:** No single path is the "prediction." The whole cloud tells you the range of plausible outcomes and where they cluster. The wider the cloud, the riskier the stock.

---

### Part 5 — Better Visualization
**File:** `part5_montecarlo_new.py`

Same simulation logic as Part 4, but the presentation is significantly improved using Seaborn and a dark theme. The chart now clearly highlights the mean, best-case, and worst-case paths.

**What it does:**
- Applies a dark GitHub-style background using `matplotlib` and `seaborn`
- Assigns each of the 100 paths a unique colour from the `husl` palette
- Adds scatter point markers and annotation labels at start and end points

---

### Part 6 — Analysis Report Export
**File:** `part6_added_analysis.py`

Adds a probability calculation and the ability to export a written analysis to a `.txt` file — making the output useful beyond just looking at a chart.

**What it does:**
- Calculates the probability of profit (percentage of paths that end above the starting price)
- Exports a formatted simulation report to `part6_simulation_report.txt`
- Uses Python's `datetime` module for timestamped reports

**Key insight:** The probability of profit is simply: `(number of paths ending above start price) / (total paths) × 100`.

---

### Part 7 — Real Market Data via yfinance
**File:** `part7_realdata.py`

The first part that uses real data. The user types any stock ticker (e.g. AAPL, TSLA, ^GSPC) and the script automatically downloads two years of historical closing prices from Yahoo Finance to derive mu and sigma from actual market behaviour.

**What it does:**
- Accepts a ticker input from the user
- Downloads real closing prices using `yfinance`
- Runs the full GBM Monte Carlo simulation on real parameters
- Saves the output report to a `part7_tested_tickers/` subfolder

**Key insight:** When you run this on real stocks, the drift and volatility values reflect what that stock has actually done — NVIDIA will look very different from a government bond ETF.

---

### Part 8 — Value at Risk (VaR)
**File:** `part8_using_var.py`

Adds proper risk management. The user now inputs their investment capital and the simulation calculates how much money is at risk in a bad scenario (the 95% VaR).

**What it does:**
- Takes investment capital as an additional user input
- Computes the 95% VaR price level (the 5th percentile of all final simulated prices)
- Calculates the monetary amount at risk given the user's capital
- Adds a VaR threshold line to the chart
- Exports a report with full risk metrics to `part8_tested_tickers/`

**Key insight:** If the 95% VaR price is $85 and you started at $100, it means there is a 5% chance you lose at least $15 per share. Multiply by shares held to get your total capital at risk.

---

### Part 9 — Merton Jump Diffusion (Black Swan Events)
**File:** `part9_black_swan_merton_jump.py`

The most advanced simulation. Standard GBM assumes price changes are smooth and continuous. Real markets are not — crashes happen suddenly. The Merton Jump Diffusion model adds a second random process (Poisson jumps) to simulate these sudden, severe drops.

**What it does:**
- Keeps all of Part 8's features
- Adds three user-defined jump parameters: frequency (lambda), average crash size (mu_j), crash variability (sigma_j)
- Applies a drift correction factor (k) so that the expected return is preserved despite the crashes
- Uses the exponential price update formula for numerical stability
- Saves reports to `part9_tested_tickers/`

**Key insight:** Without jumps, GBM underestimates tail risk. With jumps, some simulation paths suddenly drop 20-30% in a single day — which is exactly what happened to many stocks in March 2020 or during the 2008 financial crisis.

---

### Part 10 — Live Interactive Web App (Streamlit)
**File:** `part10_app.py` | **Live:** [streamlit app link](https://project-1-monte-carlo-simulation-model-siddharth-premanand.streamlit.app)

The final product. Everything from Parts 1–9 is packaged into a publicly accessible, interactive web dashboard. Users can adjust any parameter from the sidebar and see the simulation update in real time, with no coding required.

**What it does:**
- Full Streamlit sidebar with controls for ticker, simulation count, jump parameters, and chart styling
- Fetches real market data on demand using `yfinance`
- Runs the full MJD Monte Carlo simulation
- Displays an interactive Plotly chart with hover tooltips, VaR line, and annotated start/end points
- Uses `st.cache_data` to avoid re-downloading data on every interaction
- Uses `st.session_state` to persist simulation results between UI interactions
- Deployed permanently on Streamlit Community Cloud

---

## Finance Concepts & Definitions

### Returns

**Arithmetic Return**
The simple percentage change in price from one day to the next.
`Return = (Price today / Price yesterday) − 1`
Intuitive to read but misleading when chained together over time because of compounding.

**Logarithmic (Log) Return**
`Return = ln(Price today / Price yesterday)`
Preferred in quantitative finance because log returns are *additive* — you can simply sum daily log returns to get the total log return over any period, with no compounding error.

---

### Volatility and Drift

**Volatility (σ — sigma)**
How much a stock's price fluctuates day to day. Measured as the standard deviation of log returns. A stock with σ = 30% per year moves around far more than one with σ = 10%. Higher volatility = higher risk, but also higher potential reward.

**Drift (μ — mu)**
The average direction the stock tends to move, stripped of random noise. It is the mean of the log returns. Positive drift means the stock trends upward on average; negative drift means it trends downward.

**Annualization**
Daily figures are converted to yearly ones using these rules:
- Volatility: `σ_annual = σ_daily × √252` (volatility scales with the square root of time)
- Drift: `μ_annual = μ_daily × 252` (drift scales linearly with time)

252 is used because there are approximately 252 trading days in a year (excluding weekends and public holidays).

---

### Stochastic Processes

**Stochastic Process**
Any process that involves randomness evolving over time. Stock prices are stochastic — you cannot know the next price with certainty, only describe the probability distribution of where it might go.

**Wiener Process (Brownian Motion)**
The mathematical foundation of GBM. It describes a random walk where each step is drawn from a standard normal distribution (mean 0, standard deviation 1). The "Z" in the GBM formula is one draw from this process at each time step.

**Geometric Brownian Motion (GBM)**
The industry-standard model for stock price behaviour. It combines a steady upward drift (mu) with random daily noise (sigma × Brownian motion). The full formula for each time step is:

`ΔS = μ × S × Δt + σ × S × √Δt × Z`

Where S is the current price, Δt is the time step (1/252 for daily), and Z is a random standard normal draw.

**Poisson Process**
A random process that counts how many rare events occur in a given time window. Used in the Merton model to decide whether a market crash happens on any given day. Lambda (λ) controls the average rate: λ = 0.5 means roughly one crash every two years.

---

### Monte Carlo Simulation

**Monte Carlo Simulation**
A technique that runs hundreds or thousands of randomised scenarios to understand the range of possible outcomes. In finance, each scenario is one possible future price path. The collection of all paths gives you a probability distribution of outcomes rather than a single point estimate.

**Expected Value (Mean Path)**
The average of all simulated final prices. This is what you would expect to happen on average across many parallel universes. It does not mean the stock *will* hit that price — it is the centre of the probability distribution.

**Probability of Profit**
The percentage of simulated paths that end above the starting price. If 73 out of 100 paths end in profit, the probability of profit is 73%. This is a useful heuristic, but remember it is based on the model's assumptions — not a guarantee.

---

### Risk Management

**Value at Risk (VaR)**
A risk metric that answers: *"In the worst X% of scenarios, how much can I expect to lose?"*

At 95% confidence, the VaR is the 5th percentile of simulated final prices. If your starting price is $100 and the 95% VaR is $78, it means there is a 5% chance of losing at least $22 per share over the forecast period. Multiply by the number of shares to get total capital at risk.

**Confidence Level**
The probability threshold used in VaR. A 95% confidence level means you are looking at the worst 5% of outcomes. A 99% level would be even more conservative (the worst 1%).

**Tail Risk**
The risk of extreme, low-probability outcomes — the events in the "tails" of a probability distribution. Standard GBM underestimates tail risk because it assumes returns are normally distributed. Real markets have "fat tails" — crashes happen more often than pure GBM predicts.

---

### The Merton Jump Diffusion Model

**Black Swan Event**
A term coined by Nassim Taleb for a rare, unpredictable event with massive consequences — the 2008 financial crisis, COVID-19 market crash, or a single company's accounting scandal. Standard GBM cannot produce these.

**Merton Jump Diffusion (MJD)**
An extension of GBM that adds sudden, discontinuous price jumps via a Poisson process. Each day, the model asks: *"Did a crash happen today?"* If yes (triggered by the Poisson process), the price drops by an amount drawn from a separate normal distribution centred on the average crash size.

**Jump Parameters:**
- **Lambda (λ):** How often crashes occur. λ = 1.0 means one crash per year on average. λ = 0.5 means one every two years.
- **Mu_j (μⱼ):** The average size of a crash. μⱼ = -0.20 means the average crash drops the price by 20%.
- **Sigma_j (σⱼ):** The variability in crash size. σⱼ = 0.05 means crashes range roughly 20% ± 5%.

**Drift Correction Factor (k)**
When you add jumps, the expected return of the stock changes. The correction factor k adjusts the drift term (mu) so that the average long-run return is preserved as intended, even with jumps baked in. Without this correction, adding negative jumps would artificially drag down expected returns.

---

## Python Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Vectorized numerical computation, random number generation |
| `pandas` | Tabular data handling and display |
| `matplotlib` | Static chart plotting (Parts 1–9) |
| `seaborn` | Enhanced styling and colour palettes |
| `yfinance` | Downloading real historical stock price data from Yahoo Finance |
| `plotly` | Interactive chart rendering in the Streamlit app |
| `streamlit` | Web app framework for the live dashboard |
| `datetime` | Timestamping reports and computing dynamic date ranges |
| `os` | File and folder management for report exports |

---

## How to Run Locally

**Parts 1–6** (no internet required):
```bash
python part1_returns_and_volatility.py
```

**Parts 7–9** (requires internet for market data):
```bash
python part7_realdata.py
# Enter a ticker when prompted, e.g. AAPL
```

**Part 10** (Streamlit app):
```bash
pip install -r requirements.txt
streamlit run part10_app.py
```

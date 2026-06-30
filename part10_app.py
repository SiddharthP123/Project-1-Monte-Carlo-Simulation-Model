"""
Monte Carlo Simulation & Stock Analysis Tool:

A Streamlit web application that performs Monte Carlo simulations on stock prices
using the Merton Jump Diffusion model to extend the Black-Scholes framework.
The tool allows users to simulate potential price trajectories for selected assets,
incorporating both continuous Geometric Brownian motion and discontinuous jumps
to model Black Swan events.

Key Features:
- Interactive Monte Carlo simulation with customizable parameters.
- Merton Jump Diffusion model implementation.
- Real-time stock data fetching from Yahoo Finance.
- Visual analysis with customizable path colors and display options.
- Value at Risk (VaR) calculations and probability metrics.
- Capital exposure analysis and risk assessment.

Usage:
Run with: streamlit run part10_app.py
"""
import datetime
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# --------------------------------------------------------------------------------------------------
# 1. INITIAL SYSTEM CONFIGURATION:
# Setting up the page title.
st.set_page_config(page_title="Monte Carlo Simulation & Stock Analysis Tool", layout="wide")

# --------------------------------------------------------------------------------------------------
# 2. CUSTOM CSS: THE "PYTHON-TERMINAL" THEME:
st.markdown("""
    <style>
    /* 1. GLOBAL TYPOGRAPHY: */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;700&display=swap');
    
    * {
        font-family: 'Fira Code', monospace !important;
    }

    /* 2. THEME & CONTAINER OPTIMIZATION: */
    .main { background-color: #0d1117; }
    
    [data-testid="stHeader"], .stAppHeader { display: none !important; }

    .block-container {
        padding: 0.5rem 1rem 0rem 1rem !important;
        max-width: 98% !important;
    }

    /* 3. SIDEBAR STYLING: */
    [data-testid="stSidebar"] { 
        background-color: #0d1117; 
        border-right: 1px solid #30363d; 
    }

    /* Remove Sidebar Glitches */
    [data-testid="stSidebarCollapseIcon"], [data-testid="collapsedControl"], button[kind="headerNoContext"] {
        display: none !important;
    }

    /* 4. WIDGET & LABEL HARMONIZATION: */
    div[data-testid="stWidgetLabel"] p, label p {
        font-size: 14px !important;
        color: #ffffff !important;
        font-weight: 400 !important;
    }

    /* 5. INPUT BOXES: */
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
        color: #e6edf3 !important;
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
    }

    /* 6. ANALYSIS REPORT METRIC CARDS: */
    [data-testid="stMetric"] {
        background-color: #161b22; 
        border: 1px solid #30363d; 
        padding: 15px; 
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase;
    }

    /* 7. UTILITY: */
    hr { margin: 12px 0px !important; border: 0.5px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS:
def compact_line():
    """This function ensures we can control the margin widths of our line separators."""
    st.markdown("<hr style='margin: 12px 0px; border: 0.5px solid #30363d;'>",
                unsafe_allow_html=True)
# 3.1 Creating our actual sidebar.
with st.sidebar:
    # Set margin-top to 0 or a negative value to pull it up even further.
    # Primary heading for our sidebar - "User Configurations:":
    st.markdown("<h1 style='text-align: center; font-weight: bold;" \
    " margin-top: -15px; margin-bottom: 0px;" \
    "'>User Configurations:</h1>",
                unsafe_allow_html=True)
    # Separation line 1:
    compact_line()
    # Section A: "Basic User Inputs:" - contains the ticker and capital entry.
    # Heading for Section A:
    st.markdown("<h3 style='font-weight: bold; margin-bottom: -10px;'"
    ">1. Basic User Inputs:</h3>",
                unsafe_allow_html=True)
    # Separation line 2:
    compact_line()
    # Asset ticker and capital exposure selection menu.
    # The default ticker is set to AAPL (Apple).
    # We can use captions for the descriptions of our functions within the sidebar.
    # Caption for Section A.1:
    st.caption("Select which ticker you wish to invest in.")
    ticker = st.text_input("ASSET TICKER:", value="AAPL").upper()
     # Caption for Section A.2:
    st.caption("How much capital do you wish to invest in this ticker?")
    capital = st.number_input("CAPITAL EXPOSURE ($):", value=10000, step=100)
    # Separation line 3:
    compact_line()
    # Section B: "Black Swan Event Calibration:":
    # Heading for Section B:
    st.markdown("<h3 style='font-weight: bold; margin-bottom: -10px;'"
    ">2. Black Swan Event Calibration:</h3>",
                unsafe_allow_html=True)
    # Separation line 4:
    compact_line()
    # Caption for Section B:
    st.caption("Using the Merton Jump Diffusion (MJD) framework to allow for sudden," \
    " discontinuous price shocks.")
    # Our 3 MJD Parameters.
    # Jump Frequency:
    lam = st.slider("JUMP FREQUENCY (λ):", 0.0, 5.0, 1.0, 0.05)
    st.caption("This is the expected number of random crashes/jumps per year.")
    mu_j = st.slider("MEAN JUMP SIZE (μj):", -0.90, 0.0, -0.20, 0.01)
    st.caption("This is the average crash distribution & its severity/size.")
    sigma_j = st.slider("JUMP UNCERTAINTY (σj):", 0.01, 0.90, 0.05, 0.01)
    st.caption("This is the variation/standard deviation in crash size.")
    # Separation line 5:
    compact_line()
    # Section C: "Visual Preferences:":
    # Heading for Section C:
    st.markdown("<h3 style='font-weight: bold; margin-bottom: -10px;'"
    ">3. Visual Preferences:</h3>",
                unsafe_allow_html=True)
    # Separation line 6:
    compact_line()
    # Section C.1: "Path Color Generation:":
    # Heading for C.1:
    st.markdown("<h4 style='font-weight: bold;" \
    "margin-bottom: -10px;'>A) Path Color Generation:</h4>",
    unsafe_allow_html=True)
    # Caption for C.1:
    st.caption("If random color assignment doesn't suit you, you can choose from our color picker.")
    # Setting the 2 color modes available for path color generation.
    color_mode = st.radio(
        "Select Color Mode:", 
        ["UNIFORM (SINGLE-COLOR):", "RANDOM (MULTI-COLOR):"],
        index=1,  # <--- This sets the random color as the default (index 1).
        label_visibility="collapsed"
    )
    # Initialize a default color so the simulation doesn't error out.
    PATH_COLOR_INPUT = "#4A90E2"
    # Allowing the user to select the color they wish if the default doesn't suit them.
    # The default uniform color is a shade of blue.
    if color_mode == "UNIFORM (SINGLE-COLOR):":
        PATH_COLOR_INPUT = st.color_picker("Pick Path Color", "#4A90E2")
    # Separation line 7:
    compact_line()
    # Section C.2: "Simulation Preferences:":
    # Heading for C.2:
    st.markdown("<h4 style='font-weight: bold;" \
    "margin-bottom: -10px;'>B) Simulation Preferences:</h4>",
    unsafe_allow_html=True)
    # Caption for C.3:
    st.caption("Toggle path visbility and highlight important paths as per your liking.")
    # Options for the user to select from, all and mean are defaults.
    show_all = st.checkbox("SHOW ALL:", value=True)
    show_mean = st.checkbox("SHOW MEAN:", value=True)
    show_max = st.checkbox("SHOW MAX:", value=False)
    show_min = st.checkbox("SHOW MIN:", value=False)
    show_var = st.checkbox("SHOW VaR (95%):", value=True)
    show_spread = st.checkbox("SHOW RANGE SHADE:", value=False)
    # Separation line 8:
    compact_line()
    # Section C.3: "Density Preferences:":
    # Heading for C.3:
    st.markdown("<h4 style='font-weight: bold;" \
    "margin-bottom: -10px;'>C) Density Preferences:</h4>",
    unsafe_allow_html=True)
    # Caption for C.3:
    st.caption("Select how many Monte Carlo simulations you wish to run.")
    # Setting up our slider values for path density.
    # The default value is 100 simulations.
    sim_count = st.select_slider(
        "SET PATH DENSITY:",
        options=[1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250], value=100
    )

# --------------------------------------------------------------------------------------------------
# 4. DATA ENGINE (MONTE CARLO + MJD):
# This is the Streamlit performance optimiser, it remembers the data for 1 hour.
@st.cache_data(ttl=3600)
def fetch_data(symbol: str) -> pd.DataFrame:
    """Downloads data from Yahoo Finance from today to exactly 2 years ago."""
    try:
        df = yf.download(symbol, start=datetime.date.today() - datetime.timedelta(days=730))
        return df if not df.empty else pd.DataFrame()
    except Exception: # pylint: disable=broad-except
        return pd.DataFrame()
# Fetching data from the API for our selected ticker.
data_df = fetch_data(ticker)
end_dt = datetime.date.today()
# ACTUAL MATHEMATICAL ENGINE:
if not data_df.empty:
    prices = data_df['Close'].values.flatten()
    # Obtaining our logarithmic return.
    log_returns = np.log(prices[1:] / prices[:-1])
    # Obtaining annual realised volatility, drift, and the share price.
    sigma, mu, St = np.std(log_returns) * np.sqrt(252), np.mean(log_returns) * 252, prices[-1]
    # Calculating the drift correction factor for black swan events (k).
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    STEPS, DT = 252, 1/252
    # LOCKING ESSENTIAL PARAMETERS FOR KEY-BASED REGENERATION:
    # We create a list of all inputs that should trigger a new simulation.
    current_params = [ticker, lam, mu_j, sigma_j, sim_count]
    # If the parameters change OR it's the first run, we simulate.
    if "last_params" not in st.session_state or st.session_state.last_params != current_params:
        # Running the actual stochastic equations.
        sims = np.zeros((sim_count, STEPS + 1))
        sims[:, 0] = St
        for n in range(sim_count):
            for i in range(STEPS):
                jump = np.random.normal(mu_j, sigma_j) if np.random.poisson(lam * DT) > 0 else 0
                c_drift = mu - lam * k
                nm = np.random.standard_normal()
                sims[n, i+1] = sims[n, i] * np.exp((c_drift) * DT + sigma * np.sqrt(DT) * nm + jump)
        # Storing the results AND the params used to generate them.
        st.session_state.sim_results = sims
        st.session_state.last_params = current_params
    # Using the stored results for our relevant/running output.
    sims = st.session_state.sim_results
    final_prices = sims[:, -1]
    # Assigning a value for the 95% VaR component.
    var_95 = np.percentile(final_prices, 5)

# --------------------------------------------------------------------------------------------------
    # 5. UI LAYOUT OF SITE:
    # 5.1 Centered title above the entire dashboard.
    st.markdown(f"""
        <h1 style='text-align: center; font-weight: bold; margin-top: -25px; margin-bottom: 0px;'>
            Monte Carlo Simulation: {ticker}
        </h1>
        """, unsafe_allow_html=True)
    # 5.2 Description of the tool.
    st.markdown(f"""
        <p style='text-align: center; color: #8b949e; font-family: "Fira Code", monospace; font-size: 14px; margin-top: 5px;'>
            This tool operates as a predictive analytics engine, allowing the user to simulate
            {sim_count} potential price trajectories for a selected asset. It leverages
            <b>Merton Jump Diffusion</b> to extend the <b>Black-Scholes Model</b>,
            combining continuous GBM diffusion with a Poisson-driven jump.
        </p>
        """, unsafe_allow_html=True)
    # 5.3 Independent horizontal line.
    # This line stays above the columns and won't intersect with the vertical line below.
    st.markdown("<hr style='margin-top: 2px; margin-bottom: 10px;border: 0.5px solid #30363d;'>",
            unsafe_allow_html=True)
    # 5.4 Layout/dashboard columns.
    col_graph, col_spacer, col_stats = st.columns([4, 0.1, 1])
    # 5.4 Creating a column spacer.
    # The spacer column creates the vertical line via CSS in the main block or inline here.
    with col_spacer:
        # This creates a thin vertical line that spans the height of the dashboard.
        st.markdown("""
            <div style="border-left: 1px solid #30363d; height: 750px; margin-left: auto; margin-top: 0px;"></div>
        """, unsafe_allow_html=True)
    # 5.5 Generating our main graph/interactive screen.
    with col_graph:
        # 5.5.1 Defining 'Data Elements' (actual paths or ranges).
        data_visible = any([show_all, show_mean, show_max, show_min, show_spread])
        # 5.5.2 Defining if the graph should actually render.
        # We only render if there is at least one data element; show_var is treated as an overlay.
        if not data_visible:
            # This is our "No data to display" message.
            st.markdown("""
                <div style="height: 780px; display: flex; flex-direction: column; 
                            justify-content: center; align-items: center; 
                            border: 1px dashed #30363d; border-radius: 10px;">
                    <h2 style='color: #8b949e;'>NO DATA TO DISPLAY:</h2>
                    <p style='color: #555;'>Please enable at least one Price Path or Range Shade to view the analysis.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            fig = go.Figure()
            t_space = np.arange(STEPS + 1)
            # 5.5.2.1 Generating multi-colored paths.
            def get_path_color(idx):
                """Function - ensure this matches the sidebar radio string exactly."""
                if color_mode == "RANDOM (MULTI-COLOR):":
                    return f"hsla({(idx * 137.5) % 360}, 70%, 60%, 0.3)"
                return PATH_COLOR_INPUT
            # Setting the graph limits and important indices.
            y_min, y_max = np.min(sims) * 0.95, np.max(sims) * 1.05
            mean_path = np.mean(sims, axis=0)
            max_idx, min_idx = np.argmax(final_prices), np.argmin(final_prices)
            # 5.5.3 Layer shade.
            if show_spread:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([t_space, t_space[::-1]]),
                    y=np.concatenate([sims[max_idx, :], sims[min_idx, ::-1]]),
                    fill='toself', fillcolor='rgba(255, 255, 255, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'
                ))
            # 5.5.4 All paths.
            if show_all:
                for n in range(sim_count):
                    fig.add_trace(go.Scatter(
                        x=t_space, y=sims[n, :], mode='lines',
                        line=dict(width=0.8, color=get_path_color(n)), hoverinfo='skip'
                    ))
            # 5.5.5 Starting annotation (always on if graph is on).
            fig.add_annotation(
                x=0, y=St, text=f"START: ${St:.2f}", showarrow=True, arrowhead=1,
                ax=50, ay=-30, font=dict(color="white", size=11), bgcolor="rgba(13, 17, 23, 0.8)"
            )
            # 5.5.6 Highlighted paths (mean, max, min).
            if show_mean:
                fig.add_trace(go.Scatter(x=t_space, y=mean_path, mode='lines',
                                         line=dict(color='white', width=3, dash='dash')))
                fig.add_annotation(x=STEPS, y=mean_path[-1],
                                   text=f"MEAN: ${mean_path[-1]:.2f}",
                                   showarrow=True, ax=-60, ay=-20,
                                   font=dict(color="white", size=11),
                                   bgcolor="rgba(13, 17, 23, 0.8)")
            if show_max:
                fig.add_trace(go.Scatter(x=t_space, y=sims[max_idx, :], mode='lines',
                                         line=dict(color='#00ff88', width=3, dash='dash')))
                fig.add_annotation(x=STEPS, y=sims[max_idx, -1],
                                   text=f"MAX: ${sims[max_idx, -1]:.2f}",
                                   showarrow=True, ax=-60, ay=-30,
                                   font=dict(color="#00ff88", size=11),
                                   bgcolor="rgba(13, 17, 23, 0.8)")
            if show_min:
                fig.add_trace(go.Scatter(x=t_space, y=sims[min_idx, :], mode='lines',
                                         line=dict(color='#ff4b4b', width=3, dash='dash')))
                fig.add_annotation(x=STEPS, y=sims[min_idx, -1],
                                   text=f"MIN: ${sims[min_idx, -1]:.2f}",
                                   showarrow=True, ax=-60, ay=40,
                                   font=dict(color="#ff4b4b", size=11),
                                   bgcolor="rgba(13, 17, 23, 0.8)")
            # 5.5.7 VaR overlay (only if show_var is on and data is visible).
            if show_var:
                fig.add_shape(type="line", x0=0, x1=STEPS, y0=var_95, y1=var_95,
                              line=dict(color="#FFFFFF", width=2, dash="dot"))
                fig.add_annotation(x=STEPS/2, y=var_95,
                                   text=f"VaR (95%): ${var_95:.2f}",
                                   showarrow=True, arrowhead=2, ay=-30,
                                   font=dict(color="#FFFFFF", size=11),
                                   bgcolor="rgba(13, 17, 23, 0.8)")
            # 5.5.8 Historical context legend (Internal overlay).
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=(
                    f"<b>HISTORICAL CONTEXT ({ticker}):</b><br>"
                    f"Realised annual volatility (σ): {sigma:.1%}<br>"
                    f"Realised annual drift (μ): {mu:.1%}<br>"
                ),
                showarrow=False,
                align="left",
                font=dict(family="Fira Code, monospace", size=12, color="#ffffff"),
                bgcolor="rgba(13, 17, 23, 0.8)",
                bordercolor="#ffffff",
                borderwidth=1,
                borderpad=10
            )
            # 5.6 Layout styling.
            fig.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(title=f"Days from {end_dt}", showgrid=True,
                           gridcolor='#30363d', showline=True, linecolor='white', linewidth=2),
                yaxis=dict(title="Price ($)", range=[y_min, y_max], showgrid=True,
                           gridcolor='#30363d', showline=True, linecolor='white', linewidth=2),
                showlegend=False, hovermode="x", height=780
            )
            st.plotly_chart(fig, use_container_width=True)
    # 5.7 Side column configuration.
    # Heading for the side column - "Analysis Report:"
    with col_stats:
        # Heading 1: Top part - Analysis.
        st.markdown("<h2 style='text-align: center; font-weight: bold;" \
        "margin-bottom: 0px;'>Analysis</h2>", 
                    unsafe_allow_html=True)
        # Heading 2: Bottom part, pulled UP via negative margin - Report:.
        st.markdown("<h2 style='text-align: center; font-weight: bold;" \
        "margin-top: -20px;'>Report:</h2>", 
                    unsafe_allow_html=True)
        # 5.7.1 Expected price metric.
        exp_price = np.mean(final_prices)
        raw_change = exp_price - St
        pct_change = (raw_change / St) * 100
        # 5.7.2 Standard logic: -/+ determines red/green bubble.
        if raw_change < 0:
            price_delta = f"-${abs(raw_change):.2f} ({pct_change:.1f}%)"
        else:
            price_delta = f"+${raw_change:.2f} ({pct_change:+.1f}%)"
        # 5.7.3 Actual expected price metric.
        st.metric(label="EXPECTED PRICE:", value=f"${exp_price:.2f}", delta=price_delta)
        # 5.7.4 Probability of gain metric.
        prob_gain_val = (np.sum(final_prices > St) / sim_count) * 100
        # We show the delta relative to a 50/50 coin flip.
        # If prob is 60%, delta is +10.0%. If 40%, delta is -10.0%.
        prob_delta_val = prob_gain_val - 50
        prob_delta_str = f"{prob_delta_val:+.1f}% vs. 50% Base"
        # 5.7.5 Actual probability of gain metric.
        st.metric(label="PROBABILITY OF GAIN:", value=f"{prob_gain_val:.1f}%", delta=prob_delta_str)
        # Split between earlier segment and "Capital Exposure Segment".
        st.markdown("<hr style='border: 0.5px solid #30363d;'>", unsafe_allow_html=True)
        # 5.7.6 Capital exposure section.
        st.markdown("<p style='text-align: center; font-size: 14px; color: #ffffff; " \
                    "font-weight: normal; margin-bottom: 10px; margin-top: 15px;'>" \
                    "CAPITAL EXPOSURE ($):</p>", unsafe_allow_html=True)
        st.metric(label="RISKED CAPITAL:", value=f"${capital * abs((var_95 - St) / St):,.2f}")
        st.metric(label="FLOOR (LOWEST) VALUE:",
                  value=f"${capital - (capital * abs((var_95 - St) / St)):,.2f}")
else:
    st.error("Error: Ticker not found. Check spelling or terminal connection.")

# --------------------------------------------------------------------------------------------------

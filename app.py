import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import project modules
from src.models.market_model import MarketModel
from src.simulations.simulation import Simulation
from src.utils.financial_utils import calibrate_merton_model
from src.utils.portfolio import calculate_portfolio_metrics, calculate_esg_metrics

# Set page configuration
st.set_page_config(
    page_title="CAC40 Financial Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CAC40 Financial Analysis and Portfolio Optimization"
    }
)

# Apply dark theme
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f1f1f;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #31333F;
    }
    .stMarkdown a {
        color: #4da6ff;
    }
    .stDataFrame {
        background-color: #1f1f1f;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1f1f1f;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data
def load_data():
    """Load historical price data and ESG data"""
    # Create dictionary to store dataFrames for the CAC40 companies
    data_10y_dic = {}
    data_1min_dic = {}
    
    # Get list of files containing 'Data' in the "Data/CAC40 daily 10y"
    directory_10y = "Data/CAC40 daily 10y"
    files_10y = [f for f in os.listdir(directory_10y) if 'Data' in f]
    
    # Process each file for 10y data
    for file in files_10y:
        # Get company name (first word before '_')
        company = file.split('_')[0]
        
        # Read the file with tab separator
        df = pd.read_csv(os.path.join(directory_10y, file), sep='\t')
        
        # Convert first column to datetime and set as index
        df.index = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
        df = df.drop('date', axis=1)  # Remove the original date column
        df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
        
        # Store in dictionary
        data_10y_dic[company] = df
    
    # Get list of files containing 'Data' in the "Data/CAC40 min 5d"
    directory_1min = "Data/CAC40 min 5d"
    files_1min = [f for f in os.listdir(directory_1min) if 'Data' in f]
    
    # Process each file for 1min data
    for file in files_1min:
        # Get company name (first word before '_')
        company = file.split('_')[0]
        
        # Read the file with tab separator
        df = pd.read_csv(os.path.join(directory_1min, file), sep='\t')
        
        # Convert first column to datetime and set as index
        df.index = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
        df = df.drop('date', axis=1)  # Remove the original date column
        df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
        
        # Store in dictionary
        data_1min_dic[company] = df
    
    # Create DataFrames with all the closing prices
    dfs_10y = [df['clot'] for df in data_10y_dic.values()]
    histo_CAC40_10y = pd.concat(dfs_10y, axis=1, keys=data_10y_dic.keys())
    histo_CAC40_10y = histo_CAC40_10y.sort_index()
    
    # Load ESG data
    data_esg = pd.read_csv('Data/InfoCAC40_restructured.csv', sep=';').set_index('Company')
    # Convert string values to float where needed, handling French number format (comma as decimal separator)
    for col in data_esg.columns:
        # Replace commas with periods for decimal values
        data_esg[col] = data_esg[col].astype(str).str.replace(',', '.', regex=False)
        data_esg[col] = pd.to_numeric(data_esg[col], errors='coerce')
    
    # Extract the weights data of CAC40
    cac40_weights = data_esg['Weight in the CAC40']
    
    return histo_CAC40_10y.dropna(), data_10y_dic, data_1min_dic, data_esg, cac40_weights

# Main function
def main():
    # Load data
    with st.spinner("Loading data..."):
        data, data_10y_dic, data_1min_dic, data_esg, cac40_weights = load_data()
    
    # Sidebar
    st.sidebar.title("CAC40 Financial Analysis")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Data Explorer", "Market Model", "Portfolio Simulation", "ESG Portfolio Optimization"]
    )
    
    # Display selected page
    if page == "Home":
        home_page()
    elif page == "Data Explorer":
        data_explorer_page(data, data_10y_dic, data_1min_dic)
    elif page == "Market Model":
        market_model_page(data)
    elif page == "Portfolio Simulation":
        portfolio_simulation_page(data, data_esg, cac40_weights)
    elif page == "ESG Portfolio Optimization":
        esg_optimization_page(data, data_esg, cac40_weights)

def home_page():
    """Home page with project overview"""
    st.title("CAC40 Financial Analysis and Portfolio Optimization")
    
    st.markdown("""
    This application provides tools for financial modeling, portfolio simulation, and ESG-constrained portfolio optimization. 
    It focuses on the CAC40 index and includes implementations of various financial models and portfolio strategies.
    
    ## Features
    
    - **Data Explorer**: Visualize historical price data for CAC40 stocks
    - **Market Models**: Calibrate and visualize Black-Scholes and Merton jump-diffusion models
    - **Portfolio Simulation**: Simulate portfolio performance with different investment strategies
    - **ESG Portfolio Optimization**: Optimize portfolios with Environmental, Social, and Governance (ESG) constraints
    - **Risk Metrics**: Calculate and visualize various risk metrics including Value-at-Risk (VaR) and Expected Shortfall (ES)
    
    ## Getting Started
    
    Use the sidebar to navigate between different pages of the application.
    """)
    
    # Display CAC40 companies
    st.subheader("CAC40 Companies")
    
    # Load data
    data, _, _, data_esg, cac40_weights = load_data()
    
    # Create a DataFrame with company information
    company_info = pd.DataFrame({
        'Weight in CAC40': cac40_weights,
        'Sustainability Risk': data_esg['Sustainability risk'],
        'Carbon Risk': data_esg['Carbon risk'],
        'Carbon Intensity': data_esg['Carbon intensity (Tons of CO2)']
    })
    
    # Sort by weight
    company_info = company_info.sort_values('Weight in CAC40', ascending=False)
    
    # Display as a table
    st.dataframe(company_info.style.background_gradient(cmap='viridis', subset=['Weight in CAC40'])
                                  .background_gradient(cmap='RdYlGn_r', subset=['Sustainability Risk', 'Carbon Risk', 'Carbon Intensity']))
    
    # Display a treemap of CAC40 companies by weight
    fig = px.treemap(
        company_info.reset_index(), 
        path=['Company'], 
        values='Weight in CAC40',
        color='Sustainability Risk',
        color_continuous_scale='RdYlGn_r',
        title='CAC40 Companies by Weight and Sustainability Risk'
    )
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=30),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def data_explorer_page(data, data_10y_dic, data_1min_dic):
    """Data Explorer page for visualizing historical price data"""
    st.title("Data Explorer")
    
    # Select data type
    data_type = st.radio("Select Data Type", ["Daily (10 years)", "Minute (5 days)"], horizontal=True)
    
    # Select companies
    if data_type == "Daily (10 years)":
        companies = list(data_10y_dic.keys())
        data_dic = data_10y_dic
    else:
        companies = list(data_1min_dic.keys())
        data_dic = data_1min_dic
    
    selected_companies = st.multiselect("Select Companies", companies, default=companies[:3])
    
    if not selected_companies:
        st.warning("Please select at least one company.")
        return
    
    # Date range selection
    if data_type == "Daily (10 years)":
        min_date = data.index.min().date()
        max_date = data.index.max().date()
        default_start = max_date - timedelta(days=365)  # Default to last year
    else:
        min_date = min([data_dic[company].index.min().date() for company in selected_companies])
        max_date = max([data_dic[company].index.max().date() for company in selected_companies])
        default_start = max_date - timedelta(days=5)  # Default to last 5 days
    
    # Make sure default_start is not before min_date
    if default_start < min_date:
        default_start = min_date
    
    date_range = st.date_input(
        "Select Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) != 2:
        st.warning("Please select a start and end date.")
        return
    
    start_date, end_date = date_range
    
    # Chart type selection
    chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick", "OHLC"])
    
    # Normalize prices option
    normalize = st.checkbox("Normalize Prices", value=False)
    
    # Display charts
    st.subheader(f"{chart_type} Chart")
    
    # Create figure
    fig = go.Figure()
    
    for company in selected_companies:
        df = data_dic[company]
        
        # Filter by date range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        df_filtered = df[mask]
        
        if df_filtered.empty:
            st.warning(f"No data available for {company} in the selected date range.")
            continue
        
        if chart_type == "Line":
            # For line chart, only use closing prices
            prices = df_filtered['clot']
            
            # Normalize if selected
            if normalize:
                prices = prices / prices.iloc[0] * 100
            
            fig.add_trace(go.Scatter(
                x=df_filtered.index,
                y=prices,
                mode='lines',
                name=company
            ))
        elif chart_type == "Candlestick":
            # For candlestick chart, use OHLC data
            fig.add_trace(go.Candlestick(
                x=df_filtered.index,
                open=df_filtered['ouv'],
                high=df_filtered['haut'],
                low=df_filtered['bas'],
                close=df_filtered['clot'],
                name=company
            ))
        elif chart_type == "OHLC":
            # For OHLC chart, use OHLC data
            fig.add_trace(go.Ohlc(
                x=df_filtered.index,
                open=df_filtered['ouv'],
                high=df_filtered['haut'],
                low=df_filtered['bas'],
                close=df_filtered['clot'],
                name=company
            ))
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        title=f"{'Normalized ' if normalize else ''}Prices ({start_date} to {end_date})",
        xaxis_title="Date",
        yaxis_title="Price" if not normalize else "Normalized Price (Base 100)",
        legend_title="Companies",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display returns analysis
    st.subheader("Returns Analysis")
    
    # Calculate returns
    returns_data = {}
    for company in selected_companies:
        df = data_dic[company]
        
        # Filter by date range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        df_filtered = df[mask]
        
        if df_filtered.empty:
            continue
        
        # Calculate log returns
        prices = df_filtered['clot']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        returns_data[company] = returns
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Display returns statistics
    if not returns_df.empty:
        # Calculate statistics
        stats = pd.DataFrame({
            'Mean': returns_df.mean(),
            'Std Dev': returns_df.std(),
            'Min': returns_df.min(),
            'Max': returns_df.max(),
            'Skewness': returns_df.skew(),
            'Kurtosis': returns_df.kurtosis()
        })
        
        st.dataframe(stats.style.background_gradient(cmap='viridis'))
        
        # Plot returns distribution
        fig = go.Figure()
        
        for company in returns_df.columns:
            fig.add_trace(go.Histogram(
                x=returns_df[company],
                name=company,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            template='plotly_dark',
            title="Returns Distribution",
            xaxis_title="Log Returns",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot returns correlation matrix
        corr_matrix = returns_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            title="Returns Correlation Matrix"
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def market_model_page(data):
    """Market Model page for calibrating and visualizing market models"""
    st.title("Market Model")
    
    # Select model
    model_type = st.selectbox("Select Model", ["Black-Scholes", "Merton Jump-Diffusion"])
    
    if model_type == "Black-Scholes":
        black_scholes_model(data)
    else:
        merton_model(data)

def black_scholes_model(data):
    """Black-Scholes model calibration and visualization"""
    st.subheader("Black-Scholes Model")
    
    # Select companies for calibration
    companies = data.columns.tolist()
    selected_companies = st.multiselect("Select Companies for Calibration", companies, default=companies[:5])
    
    if not selected_companies:
        st.warning("Please select at least one company.")
        return
    
    # Filter data
    filtered_data = data[selected_companies]
    
    # Calibrate model
    with st.spinner("Calibrating Black-Scholes model..."):
        market_model_BS = MarketModel(model_name="BS")
        market_model_BS.fit(filtered_data)
    
    # Display model parameters
    st.subheader("Model Parameters")
    
    # Create DataFrame with parameters
    params_df = pd.DataFrame({
        'Annual Return': market_model_BS.parameters['Returns'],
        'Annual Volatility': market_model_BS.parameters['Volatilities']
    })
    
    st.dataframe(params_df.style.background_gradient(cmap='viridis'))
    
    # Plot correlation matrix
    corr_matrix = market_model_BS.parameters['Correlation matrix']
    
    fig = px.imshow(
        corr_matrix,
        x=selected_companies,
        y=selected_companies,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix"
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate scenarios
    st.subheader("Generate Scenarios")
    
    # Scenario parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        begin_date = st.date_input("Begin Date", value=datetime.now().date())
    
    with col2:
        end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=365)).date())
    
    with col3:
        num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=100, value=5)
    
    if begin_date >= end_date:
        st.warning("Begin date must be before end date.")
        return
    
    # Generate scenarios
    if st.button("Generate Scenarios"):
        with st.spinner("Generating scenarios..."):
            scenarios_BS = market_model_BS.generate_logreturns(
                begin_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                int(num_scenarios)
            )
        
        # Select company to visualize
        company = st.selectbox("Select Company to Visualize", selected_companies)
        
        # Plot cumulative returns
        fig = go.Figure()
        
        for i in range(1, int(num_scenarios) + 1):
            scenario = scenarios_BS[f'Scenario {i}']
            cumulative_returns = np.exp(np.cumsum(scenario[company]))
            
            fig.add_trace(go.Scatter(
                x=scenario.index,
                y=cumulative_returns,
                mode='lines',
                name=f'Scenario {i}'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            title=f"Cumulative Returns for {company}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def merton_model(data):
    """Merton jump-diffusion model calibration and visualization"""
    st.subheader("Merton Jump-Diffusion Model")
    
    # Select company for calibration
    company = st.selectbox("Select Company for Calibration", data.columns.tolist())
    
    # Calculate log returns
    prices = data[company]
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calibrate model
    with st.spinner("Calibrating Merton model..."):
        dt = 1/252  # Daily time step (assuming 252 trading days per year)
        merton_params = calibrate_merton_model(returns.values, dt)
    
    # Extract parameters
    mu, sigma, lambda_, mu_J, sigma_J = merton_params
    
    # Display calibrated parameters
    st.subheader("Calibrated Parameters")
    
    params_df = pd.DataFrame({
        'Parameter': ['Drift (μ)', 'Volatility (σ)', 'Jump Intensity (λ)', 'Jump Mean (μ_J)', 'Jump Volatility (σ_J)'],
        'Value': [mu, sigma, lambda_, mu_J, sigma_J]
    })
    
    st.dataframe(params_df)
    
    # Visualize returns distribution
    st.subheader("Returns Distribution")
    
    # Generate x values for plotting
    x = np.linspace(min(returns), max(returns), 1000)
    
    # Calculate PDF for Merton model
    pdf_merton = (1 - lambda_ * dt) * np.exp(-(x - mu * dt)**2 / (2 * sigma**2 * dt)) / (sigma * np.sqrt(2 * np.pi * dt)) + \
                 (lambda_ * dt) * np.exp(-(x - mu * dt - mu_J)**2 / (2 * (sigma**2 * dt + sigma_J**2))) / (np.sqrt(2 * np.pi * (sigma**2 * dt + sigma_J**2)))
    
    # Calculate PDF for normal distribution
    pdf_normal = np.exp(-(x - np.mean(returns))**2 / (2 * np.var(returns))) / (np.sqrt(2 * np.pi * np.var(returns)))
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram of returns
    fig.add_trace(go.Histogram(
        x=returns,
        histnorm='probability density',
        name='Historical Returns',
        opacity=0.5,
        nbinsx=30
    ))
    
    # Add Merton model PDF
    fig.add_trace(go.Scatter(
        x=x,
        y=pdf_merton,
        mode='lines',
        name='Merton Model',
        line=dict(color='red', width=2)
    ))
    
    # Add normal distribution PDF
    fig.add_trace(go.Scatter(
        x=x,
        y=pdf_normal,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=f"Returns Distribution for {company}",
        xaxis_title="Log Returns",
        yaxis_title="Density",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simulate price paths
    st.subheader("Simulate Price Paths")
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.number_input("Time Horizon (years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    with col2:
        num_steps = st.number_input("Number of Time Steps", min_value=10, max_value=1000, value=252)
    
    with col3:
        num_paths = st.number_input("Number of Paths", min_value=1, max_value=50, value=10)
    
    if st.button("Simulate Price Paths"):
        with st.spinner("Simulating price paths..."):
            # Simulation parameters
            S0 = prices.iloc[-1]      # Initial price (last observed price)
            T = horizon               # Time horizon
            N = int(num_steps)        # Number of time steps
            M = int(num_paths)        # Number of simulated paths
            
            dt = T / N                # Time step
            t = np.linspace(0, T, N+1)
            
            # Simulation of price paths
            S = np.zeros((M, N+1))
            S[:, 0] = S0
            
            for m in range(M):
                W = np.random.normal(0, np.sqrt(dt), N)  # Brownian motion increments
                N_t = np.random.poisson(lambda_ * dt, N) # Poisson process (number of jumps)
                J_t = np.random.normal(mu_J, sigma_J, N) * N_t  # Jump sizes
                
                for i in range(1, N+1):
                    dS = mu * S[m, i-1] * dt + sigma * S[m, i-1] * W[i-1] + S[m, i-1] * J_t[i-1]
                    S[m, i] = S[m, i-1] + dS
        
        # Plot simulated price paths
        fig = go.Figure()
        
        # Add historical prices for context (last year)
        historical_days = min(252, len(prices))
        historical_prices = prices.iloc[-historical_days:]
        historical_t = np.linspace(-1, 0, historical_days)
        
        fig.add_trace(go.Scatter(
            x=historical_t,
            y=historical_prices.values,
            mode='lines',
            name='Historical',
            line=dict(color='white', width=2)
        ))
        
        # Add simulated paths
        for m in range(M):
            fig.add_trace(go.Scatter(
                x=t,
                y=S[m, :],
                mode='lines',
                name=f'Path {m+1}',
                opacity=0.7
            ))
        
        fig.update_layout(
            template='plotly_dark',
            title=f"Merton Model: Simulated Price Paths for {company}",
            xaxis_title="Time (years)",
            yaxis_title="Price",
            height=500,
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=0,
                    y0=0,
                    x1=0,
                    y1=1,
                    line=dict(
                        color="white",
                        width=2,
                        dash="dash",
                    )
                )
            ],
            annotations=[
                dict(
                    x=0.01,
                    y=S0,
                    xref="x",
                    yref="y",
                    text="Today",
                    showarrow=False,
                    bgcolor="rgba(50, 50, 50, 0.8)",
                    bordercolor="white",
                    borderwidth=1
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate risk metrics
        st.subheader("Risk Metrics")
        
        # Increase number of simulations for better risk metrics
        M_risk = 1000
        S_final = np.zeros(M_risk)
        
        with st.spinner("Calculating risk metrics..."):
            # Simulate terminal values only
            for m in range(M_risk):
                S_path = S0
                for i in range(N):
                    W = np.random.normal(0, np.sqrt(dt))
                    N_t = np.random.poisson(lambda_ * dt)
                    J_t = np.random.normal(mu_J, sigma_J) * N_t if N_t > 0 else 0
                    dS = mu * S_path * dt + sigma * S_path * W + S_path * J_t
                    S_path += dS
                S_final[m] = S_path
            
            # Calculate returns
            returns_final = (S_final - S0) / S0
            
            # Calculate risk metrics
            alpha = 0.95  # Confidence level
            VaR = -np.percentile(returns_final, 100 * (1 - alpha))
            ES = -np.mean(returns_final[returns_final <= -VaR])
        
        # Display risk metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Value-at-Risk (VaR)', 'Expected Shortfall (ES)'],
            'Value': [f"{VaR:.4f} ({VaR*100:.2f}%)", f"{ES:.4f} ({ES*100:.2f}%)"],
            'Confidence Level': [f"{alpha*100}%", f"{alpha*100}%"]
        })
        
        st.dataframe(metrics_df)
        
        # Plot histogram of simulated returns
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns_final,
            name='Simulated Returns',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.add_vline(
            x=-VaR,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({alpha*100}%): {VaR:.4f}",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=-ES,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"ES ({alpha*100}%): {ES:.4f}",
            annotation_position="top left"
        )
        
        fig.update_layout(
            template='plotly_dark',
            title="Distribution of Simulated Returns with Risk Metrics",
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def portfolio_simulation_page(data, data_esg, cac40_weights):
    """Portfolio Simulation page for creating and comparing portfolio simulations"""
    st.title("Portfolio Simulation")
    
    # Select companies for portfolio
    companies = data.columns.tolist()
    selected_companies = st.multiselect("Select Companies for Portfolio", companies, default=companies[:10])
    
    if not selected_companies:
        st.warning("Please select at least one company.")
        return
    
    # Filter data
    filtered_data = data[selected_companies]
    
    # Calibrate model
    with st.spinner("Calibrating Black-Scholes model..."):
        market_model_BS = MarketModel(model_name="BS")
        market_model_BS.fit(filtered_data)
    
    # Portfolio allocation
    st.subheader("Portfolio Allocation")
    
    allocation_type = st.radio("Allocation Type", ["Equal Weight", "CAC40 Weight", "Custom"], horizontal=True)
    
    if allocation_type == "Equal Weight":
        # Equal weight allocation
        allocation = np.ones(len(selected_companies)) / len(selected_companies)
    elif allocation_type == "CAC40 Weight":
        # Use CAC40 weights
        allocation = cac40_weights[selected_companies].values
        # Normalize to sum to 1
        allocation = allocation / allocation.sum()
    else:
        # Custom allocation
        st.write("Enter custom allocation (values will be normalized to sum to 1):")
        
        # Create columns for input
        cols = st.columns(5)
        custom_allocation = {}
        
        for i, company in enumerate(selected_companies):
            col_idx = i % 5
            with cols[col_idx]:
                custom_allocation[company] = st.number_input(
                    company,
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(selected_companies),
                    step=0.01,
                    format="%.2f"
                )
        
        # Convert to array and normalize
        allocation = np.array([custom_allocation[company] for company in selected_companies])
        allocation = allocation / allocation.sum()
    
    # Display allocation as a pie chart
    fig = px.pie(
        values=allocation,
        names=selected_companies,
        title="Portfolio Allocation",
        template="plotly_dark",
        hole=0.3
    )
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        begin_date = st.date_input("Begin Date", value=datetime.now().date())
    
    with col2:
        end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=365)).date())
    
    with col3:
        num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=1000, value=50)
    
    if begin_date >= end_date:
        st.warning("Begin date must be before end date.")
        return
    
    # Strategy selection
    strategy = st.radio("Investment Strategy", ["Buy and hold", "Rebalancing"], horizontal=True)
    
    if strategy == "Rebalancing":
        rebalancing_period = st.slider("Rebalancing Period (days)", min_value=1, max_value=90, value=20)
    else:
        rebalancing_period = -1
    
    # Create simulation parameters
    if strategy == "Buy and hold":
        parameters = {
            "Begin date": begin_date.strftime('%Y-%m-%d'),
            "End date": end_date.strftime('%Y-%m-%d'),
            "Allocation": allocation
        }
    else:
        parameters = {
            "Begin date": begin_date.strftime('%Y-%m-%d'),
            "End date": end_date.strftime('%Y-%m-%d'),
            "Allocation": allocation,
            "Rebalancing period": rebalancing_period
        }
    
    # Run simulation
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Create simulation
            simulation = Simulation(
                nb_scenarios=int(num_scenarios),
                model=market_model_BS,
                strategy=strategy,
                parameters=parameters
            )
            
            # Generate scenarios and evolutions
            simulation.generate_scenarios()
            simulation.generate_evolutions()
            
            # Compute risk metrics
            simulation.compute_metrics()
        
        # Display results
        st.subheader("Simulation Results")
        
        # Display risk metrics
        metrics = simulation.metrics
        
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        st.dataframe(metrics_df)
        
        # Plot portfolio evolution
        st.subheader("Portfolio Evolution")
        
        # Create figure
        fig = go.Figure()
        
        # Plot individual scenarios in light blue
        first = True
        for evolution_name, evolution_data in simulation.evolutions.items():
            if first:
                fig.add_trace(go.Scatter(
                    x=evolution_data.index,
                    y=evolution_data.sum(axis=1),
                    mode='lines',
                    name='Scenarios',
                    line=dict(color='lightblue', width=1),
                    opacity=0.2
                ))
                first = False
            else:
                fig.add_trace(go.Scatter(
                    x=evolution_data.index,
                    y=evolution_data.sum(axis=1),
                    mode='lines',
                    name='Scenarios',
                    line=dict(color='lightblue', width=1),
                    opacity=0.2,
                    showlegend=False
                ))
        
        # Calculate and plot mean trajectory
        mean_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).mean(axis=1)
        
        fig.add_trace(go.Scatter(
            x=mean_evolution.index,
            y=mean_evolution,
            mode='lines',
            name='Mean Trajectory',
            line=dict(color='red', width=2)
        ))
        
        # Calculate and plot median trajectory
        median_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).median(axis=1)
        
        fig.add_trace(go.Scatter(
            x=median_evolution.index,
            y=median_evolution,
            mode='lines',
            name='Median Trajectory',
            line=dict(color='blue', width=2)
        ))
        
        # Calculate and plot VaR
        alpha = 0.95
        var_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).quantile(1-alpha, axis=1)
        
        fig.add_trace(go.Scatter(
            x=var_evolution.index,
            y=var_evolution,
            mode='lines',
            name=f'VaR ({alpha*100}%)',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            title=f"Portfolio Evolution ({strategy})",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display terminal value distribution
        st.subheader("Terminal Value Distribution")
        
        # Extract terminal values
        terminal_values = [evolution.iloc[-1].sum() for evolution in simulation.evolutions.values()]
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=terminal_values,
            name='Terminal Values',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Add mean line
        fig.add_vline(
            x=np.mean(terminal_values),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(terminal_values):.4f}",
            annotation_position="top right"
        )
        
        # Add median line
        fig.add_vline(
            x=np.median(terminal_values),
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Median: {np.median(terminal_values):.4f}",
            annotation_position="top left"
        )
        
        # Add VaR line
        fig.add_vline(
            x=np.percentile(terminal_values, 100 * (1-alpha)),
            line_dash="dash",
            line_color="black",
            annotation_text=f"VaR ({alpha*100}%): {np.percentile(terminal_values, 100 * (1-alpha)):.4f}",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            title="Terminal Value Distribution",
            xaxis_title="Terminal Value",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def esg_optimization_page(data, data_esg, cac40_weights):
    """ESG Portfolio Optimization page for optimizing portfolios with ESG constraints"""
    st.title("ESG Portfolio Optimization")
    
    # Select companies for portfolio
    companies = data.columns.tolist()
    selected_companies = st.multiselect("Select Companies for Portfolio", companies, default=companies[:20])
    
    if not selected_companies:
        st.warning("Please select at least one company.")
        return
    
    # Filter ESG data for selected companies
    # Make sure company names match between data sources
    valid_companies = [company for company in selected_companies if company in data_esg.index]
    company_mapping = {}
    
    if not valid_companies:
        # Try to match companies with slight name differences
        for company in selected_companies:
            # Check for companies that might have slightly different names
            for esg_company in data_esg.index:
                if company in esg_company or esg_company in company:
                    company_mapping[company] = esg_company
                    break
        
        if company_mapping:
            valid_mapped_companies = [company_mapping.get(company) for company in selected_companies 
                                     if company_mapping.get(company) in data_esg.index]
            filtered_esg = data_esg.loc[valid_mapped_companies]
            # Create a mapping from original companies to ESG companies
            valid_companies = [company for company in selected_companies if company_mapping.get(company) in data_esg.index]
        else:
            st.error("None of the selected companies match the ESG data. Please select different companies.")
            return
    else:
        filtered_esg = data_esg.loc[valid_companies]
        # Create a mapping for exact matches
        for company in valid_companies:
            company_mapping[company] = company
    
    # Filter price data to only include companies with ESG data
    filtered_data = data[valid_companies]
    
    # Show which companies were included
    st.info(f"Using {len(valid_companies)} out of {len(selected_companies)} selected companies that have ESG data.")
    
    # Calibrate model
    with st.spinner("Calibrating Black-Scholes model..."):
        market_model_BS = MarketModel(model_name="BS")
        market_model_BS.fit(filtered_data)
    
    # Display ESG data
    st.subheader("ESG Data")
    
    # Display as a table
    st.dataframe(filtered_esg.style.background_gradient(cmap='RdYlGn_r', subset=['Sustainability risk', 'Carbon risk', 'Carbon intensity (Tons of CO2)'])
                                  .background_gradient(cmap='RdYlGn', subset=['Score management']))
    
    # ESG constraints
    st.subheader("ESG Constraints")
    
    # Create columns for constraints
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Financial Constraints")
        max_allocation = st.slider("Maximum Allocation per Stock", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
        max_volatility = st.slider("Maximum Portfolio Volatility", min_value=0.05, max_value=0.5, value=0.2, step=0.01)
    
    with col2:
        st.write("ESG Constraints")
        max_carbon_risk = st.slider("Maximum Carbon Risk", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        min_score_management = st.slider("Minimum Score Management", min_value=50.0, max_value=80.0, value=65.0, step=1.0)
    
    # Create constraints dictionary
    constraints = {
        "List": ["Maximal allocation", "Maximal volatility", "Maximal Carbon risk", "Minimal Score management"],
        "Value": [max_allocation, max_volatility, max_carbon_risk, min_score_management]
    }
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        begin_date = st.date_input("Begin Date", value=datetime.now().date())
    
    with col2:
        end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=365)).date())
    
    with col3:
        num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=1000, value=50)
    
    if begin_date >= end_date:
        st.warning("Begin date must be before end date.")
        return
    
    # Strategy selection
    strategy = st.radio("Investment Strategy", ["Buy and hold", "Rebalancing"], horizontal=True)
    
    if strategy == "Rebalancing":
        rebalancing_period = st.slider("Rebalancing Period (days)", min_value=1, max_value=90, value=20)
    else:
        rebalancing_period = -1
    
    # Create simulation parameters
    if strategy == "Buy and hold":
        parameters = {
            "Begin date": begin_date.strftime('%Y-%m-%d'),
            "End date": end_date.strftime('%Y-%m-%d')
        }
    else:
        parameters = {
            "Begin date": begin_date.strftime('%Y-%m-%d'),
            "End date": end_date.strftime('%Y-%m-%d'),
            "Rebalancing period": rebalancing_period
        }
    
    # Run optimization
    if st.button("Run Optimization"):
        with st.spinner("Running optimization..."):
            # Create simulation
            simulation = Simulation(
                nb_scenarios=int(num_scenarios),
                model=market_model_BS,
                strategy=strategy,
                parameters=parameters
            )
            
            # Set ESG data and constraints
            simulation.set_dataESG(filtered_esg)
            simulation.set_constraints(constraints)
            
            # Compute optimal allocation
            try:
                optimal_allocation = simulation.compute_allocation()
                
                # Generate scenarios and evolutions
                simulation.generate_scenarios()
                simulation.generate_evolutions()
                
                # Compute risk metrics
                simulation.compute_metrics()
                
                optimization_success = True
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.info("Try relaxing the constraints or selecting different companies.")
                optimization_success = False
        
        if optimization_success:
            # Display results
            st.subheader("Optimization Results")
            
            # Display optimal allocation
            optimal_allocation = simulation.parameters["Allocation"]
            
            # Create DataFrame with allocation
            allocation_df = pd.DataFrame({
                'Company': valid_companies,
                'Allocation': optimal_allocation
            })
            
            # Sort by allocation
            allocation_df = allocation_df.sort_values('Allocation', ascending=False)
            
            # Display as a table
            st.dataframe(allocation_df.style.background_gradient(cmap='viridis', subset=['Allocation']))
            
            # Display allocation as a pie chart
            fig = px.pie(
                values=optimal_allocation,
                names=valid_companies,
                title="Optimal Portfolio Allocation",
                template="plotly_dark",
                hole=0.3
            )
            
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate portfolio metrics
            returns = market_model_BS.parameters["Returns"]
            volatilities = market_model_BS.parameters["Volatilities"]
            correlation_matrix = market_model_BS.parameters["Correlation matrix"]
            cov_matrix = np.diag(volatilities) @ correlation_matrix @ np.diag(volatilities)
            
            portfolio_return = optimal_allocation @ returns
            portfolio_volatility = np.sqrt(optimal_allocation @ cov_matrix @ optimal_allocation)
            portfolio_sharpe = (portfolio_return - 0.02) / portfolio_volatility
            
            # Calculate ESG metrics
            portfolio_carbon_risk = optimal_allocation @ filtered_esg['Carbon risk']
            portfolio_carbon_intensity = optimal_allocation @ filtered_esg['Carbon intensity (Tons of CO2)']
            portfolio_score_management = optimal_allocation @ filtered_esg['Score management']
            
            # Display portfolio metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Carbon Risk', 'Carbon Intensity', 'Score Management'],
                'Value': [
                    f"{portfolio_return:.4f} ({portfolio_return*100:.2f}%)",
                    f"{portfolio_volatility:.4f} ({portfolio_volatility*100:.2f}%)",
                    f"{portfolio_sharpe:.4f}",
                    f"{portfolio_carbon_risk:.4f}",
                    f"{portfolio_carbon_intensity:.4f}",
                    f"{portfolio_score_management:.4f}"
                ]
            })
            
            st.dataframe(metrics_df)
            
            # Display risk metrics
            metrics = simulation.metrics
            
            risk_metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            st.dataframe(risk_metrics_df)
            
            # Plot portfolio evolution
            st.subheader("Portfolio Evolution")
            
            # Create figure
            fig = go.Figure()
            
            # Plot individual scenarios in light blue
            first = True
            for evolution_name, evolution_data in simulation.evolutions.items():
                if first:
                    fig.add_trace(go.Scatter(
                        x=evolution_data.index,
                        y=evolution_data.sum(axis=1),
                        mode='lines',
                        name='Scenarios',
                        line=dict(color='lightblue', width=1),
                        opacity=0.2
                    ))
                    first = False
                else:
                    fig.add_trace(go.Scatter(
                        x=evolution_data.index,
                        y=evolution_data.sum(axis=1),
                        mode='lines',
                        name='Scenarios',
                        line=dict(color='lightblue', width=1),
                        opacity=0.2,
                        showlegend=False
                    ))
            
            # Calculate and plot mean trajectory
            mean_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).mean(axis=1)
            
            fig.add_trace(go.Scatter(
                x=mean_evolution.index,
                y=mean_evolution,
                mode='lines',
                name='Mean Trajectory',
                line=dict(color='red', width=2)
            ))
            
            # Calculate and plot median trajectory
            median_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).median(axis=1)
            
            fig.add_trace(go.Scatter(
                x=median_evolution.index,
                y=median_evolution,
                mode='lines',
                name='Median Trajectory',
                line=dict(color='blue', width=2)
            ))
            
            # Calculate and plot VaR
            alpha = 0.95
            var_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).quantile(1-alpha, axis=1)
            
            fig.add_trace(go.Scatter(
                x=var_evolution.index,
                y=var_evolution,
                mode='lines',
                name=f'VaR ({alpha*100}%)',
                line=dict(color='black', width=2, dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                title=f"ESG-Optimized Portfolio Evolution ({strategy})",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display terminal value distribution
            st.subheader("Terminal Value Distribution")
            
            # Extract terminal values
            terminal_values = [evolution.iloc[-1].sum() for evolution in simulation.evolutions.values()]
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=terminal_values,
                name='Terminal Values',
                opacity=0.7,
                nbinsx=30
            ))
            
            # Add mean line
            fig.add_vline(
                x=np.mean(terminal_values),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {np.mean(terminal_values):.4f}",
                annotation_position="top right"
            )
            
            # Add median line
            fig.add_vline(
                x=np.median(terminal_values),
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Median: {np.median(terminal_values):.4f}",
                annotation_position="top left"
            )
            
            # Add VaR line
            fig.add_vline(
                x=np.percentile(terminal_values, 100 * (1-alpha)),
                line_dash="dash",
                line_color="black",
                annotation_text=f"VaR ({alpha*100}%): {np.percentile(terminal_values, 100 * (1-alpha)):.4f}",
                annotation_position="bottom right"
            )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                title="Terminal Value Distribution",
                xaxis_title="Terminal Value",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()

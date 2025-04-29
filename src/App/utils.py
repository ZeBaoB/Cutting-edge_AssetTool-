import numpy as np
import pandas as pd
import os
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from src.models.market_model import MarketModel
from src.utils.financial_utils import calibrate_merton_model, calibrate_heston_model

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

def black_scholes_model(data):
    """Black-Scholes model calibration and visualization"""
    st.subheader("Presentation Black-Scholes Model")
    # Display the dynamic in the model
    st.write("The Black-Scholes model is a mathematical model used to price options. It assumes that each stock price S^i follows a geometric Brownian motion with constant volatility and drift. The model is defined by the following stochastic differential equation:")
    st.latex(r"""
    dS_t^{(i)} = \mu^{(i)} S_t^{(i)}\, dt + \sigma^{(i)} S_t^{(i)}\, dB_t^{(i)}
    """)
    st.write("So, the parameters of the model are:")
    st.latex(r"""
    \begin{cases}
    \mu^{(i)} : \text{drift} \\
    \sigma^{(i)} : \text{volatility} \\
    \end{cases}
    """)
    st.write("Finally, we need to estimate the correlation matrix of the assets in order to generate correlated scenarios.")


    st.subheader("Calibration of parameters on historical data")
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
        title="Correlation Matrix",
        zmin=-0.7,
        zmax=1.0,
        text_auto=".3f"  # Display values in each cell, 3 decimal places
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate scenarios
    st.subheader("Generate Scenario")
    
    # Scenario parameters
    col1, col2 = st.columns(2)
    
    with col1:
        begin_date = st.date_input("Begin Date", value=datetime.now().date())
    
    with col2:
        end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=365)).date())
    
    num_scenarios = 1
    
    if begin_date >= end_date:
        st.warning("Begin date must be before end date.")
        return
    
    # Generate scenario
    if st.button("Generate Scenario"):
        with st.spinner("Generating scenario..."):
            scenarios_BS, _ = market_model_BS.generate_logreturns(
                begin_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                num_scenarios
            )
            
        # Plot cumulative returns
        fig = go.Figure()
        scenario = scenarios_BS[f'Scenario 1']
        for columns in scenario.columns:
            cumulative_returns = np.exp(np.cumsum(scenario[columns]))
            fig.add_trace(go.Scatter(
                x=scenario.index,
                y=cumulative_returns,
                mode='lines',
                name=columns
            ))
        
        fig.update_layout(
            template='plotly_dark',
            title=f"Cumulative Returns for the stocks",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def heston_model(data):
    """Heston model calibration and visualization"""
    st.subheader("Presentation Heston Model")
    # Display the dynamic in the model
    st.write("The Heston model is a stochastic model with non-constant volatility used to model the evolution of the price of an asset. For each asset i, it is defined by the following system of stochastic differential equations:")
    st.latex(r"""
    \begin{cases}
    dS_t^{(i)} = \mu^{(i)} S_t^{(i)}\, dt + \sqrt{v_t^{(i)}}\, S_t^{(i)}\, dB_t^{(i)} \\
    dv_t^{(i)} = \kappa^{(i)}(\theta^{(i)} - v_t^{(i)})\, dt + \sigma^{(i)} \sqrt{v_t^{(i)}}\, dW_t^{(i)}
    \end{cases}
    """)
    st.write("Alternatively, the logarithmic form of the price dynamics is given by:")
    st.latex(r"""
    d\ln(S_t^{(i)}) = \left(\mu^{(i)} - 0.5 \cdot v_t^{(i)}\right) dt + \sqrt{v_t^{(i)}}\, dB_t^{(i)}
    """)
    st.write("So, the parameters of the model are:")
    st.latex(r"""
    \begin{cases}
    \mu^{(i)} : \text{drift} \\
    \kappa^{(i)} : \text{volatility mean reversion speed} \\
    \theta^{(i)} : \text{long-term volatility} \\
    \sigma^{(i)} : \text{volatility of volatility} \\
    \end{cases}
    """)
    st.write("Finally, we need to estimate the correlation matrix of the assets and variances in order to generate correlated scenarios.")
    
    st.subheader("Calibration of parameters on historical data")
    # Select companies for calibration
    companies = data.columns.tolist()
    selected_companies = st.multiselect("Select Companies for Calibration", companies, default=companies[:5])
    
    if not selected_companies:
        st.warning("Please select at least one company.")
        return
    
    # Filter data
    filtered_data = data[selected_companies]
    
    # Calibrate model
    with st.spinner("Calibrating Heston model..."):
        market_model_Heston = MarketModel(model_name="Heston")
        market_model_Heston.fit(filtered_data)
    
    df_heston_params, dB_dW_corr = market_model_Heston.parameters['Parameters Heston'], market_model_Heston.parameters['dB_dW correlation']
    
    # Display model parameters
    params_df = pd.DataFrame({
        'Annual Return': df_heston_params.loc['mu'],
        'Volatility mean reversion speed (kappa)': df_heston_params.loc['kappa'],
        'Long-term volatility (theta)': df_heston_params.loc['theta'],
        'Volatility of volatility (sigma)': df_heston_params.loc['sigma'],
    })
    st.dataframe(params_df.style.background_gradient(cmap='viridis'))

    # Plot correlation matrix
    fig = px.imshow(
        dB_dW_corr,
        x=dB_dW_corr.columns.tolist(),
        y=dB_dW_corr.columns.tolist(),
        color_continuous_scale='RdBu_r',
        zmin=-0.7,
        zmax=1.0,
        title="Correlation Matrix",
        text_auto=".3f"  # Display values in each cell, 3 decimal places
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate scenario
    st.subheader("Generate Scenario")

    # Scenario parameters
    col1, col2 = st.columns(2)
    
    with col1:
        begin_date = st.date_input("Begin Date", value=datetime.now().date())
    
    with col2:
        end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=365)).date())
    
    num_scenarios = 1
    
    if begin_date >= end_date:
        st.warning("Begin date must be before end date.")
        return
    
    # Generate scenario
    if st.button("Generate Scenario"):
        with st.spinner("Generating scenario..."):
            scenarios_Heston, Var_scenarios = market_model_Heston.generate_logreturns(
                begin_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                num_scenarios
            )
        
        # Plot cumulative returns
        fig = go.Figure()
        scenario = scenarios_Heston[f'Scenario 1']
        for columns in scenario.columns:
            cumulative_returns = np.exp(np.cumsum(scenario[columns]))
            fig.add_trace(go.Scatter(
                x=scenario.index,
                y=cumulative_returns,
                mode='lines',
                name=columns
            ))
        fig.update_layout(
            template='plotly_dark',
            title=f"Cumulative Returns for the stocks",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot variance scenarios
        fig2 = go.Figure()
        scenario_var = Var_scenarios[f'Scenario 1']
        for columns in scenario_var.columns:
            fig2.add_trace(go.Scatter(
            x=scenario_var.index,
            y=scenario_var[columns],
            mode='lines',
            name=columns
            ))
        fig2.update_layout(
            template='plotly_dark',
            title=f"Variance for the stocks",
            xaxis_title="Date",
            yaxis_title="Variance",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)




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

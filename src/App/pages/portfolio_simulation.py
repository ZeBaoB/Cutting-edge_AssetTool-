import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import project modules
from src.models.market_model import MarketModel
from src.simulations.simulation import Simulation

def portfolio_simulation_page(data, data_esg, cac40_weights):
    """Portfolio Simulation page for creating and comparing portfolio simulations"""
    st.title("Portfolio Simulation")
    
    # Select companies for portfolio
    companies = data.columns.tolist()
    selected_companies = st.multiselect("Select Companies for Portfolio", companies, default=companies[:5])
    
    if not selected_companies:
        st.warning("Please select at least one company.")
        return
    
    # Filter data
    filtered_data = data[selected_companies]
    
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
    st.subheader("Model")
    # Select model
    model_type = st.selectbox("Select Model", ["Black-Scholes", "Heston"])
    model_type = "BS" if model_type == "Black-Scholes" else model_type
    # Calibrate model
    with st.spinner(f"Calibrating {model_type} model..."):
        market_model_simu = MarketModel(model_name=model_type)
        market_model_simu.fit(filtered_data)

    if model_type == "BS":
        params_df = pd.DataFrame({
        'Annual Return': market_model_simu.parameters['Returns'],
        'Annual Volatility': market_model_simu.parameters['Volatilities']
        })
        st.write("Annual returns and volatilities:")
        # Crée un DataFrame avec les valeurs par défaut issues de params_df
        default_params_df = params_df.copy()
        param_input = st.data_editor(
            default_params_df,
            num_rows="fixed",
            use_container_width=True,
            key="manual_param_editor")
        
        params_df2 = pd.DataFrame(param_input, columns=["Annual Return", "Annual Volatility"])
        default_corr_df = market_model_simu.parameters['Correlation matrix']
        default_corr_df = default_corr_df.round(3)
        corr_input = st.data_editor(
            default_corr_df,
            num_rows="fixed",
            use_container_width=True,
            key="manual_corr_editor"
        )
        corr_df = pd.DataFrame(corr_input, index=selected_companies, columns=selected_companies)
        # Enforce symmetry and identity diagonal
        corr_df = (corr_df + corr_df.T) / 2
        np.fill_diagonal(corr_df.values, 1.0)

        Parameters_simu = {'Returns': params_df2["Annual Return"],
                            'Volatilities': params_df2["Annual Volatility"],
                            'Correlation matrix': corr_df}
        market_model_simu.set_parameters(Parameters_simu)
        
    elif model_type == "Heston":
        df_heston_params, dB_dW_corr = market_model_simu.parameters['Parameters Heston'], market_model_simu.parameters['dB_dW correlation']
        # Display model parameters
        params_df = pd.DataFrame({
            'Annual Return': df_heston_params.loc['mu'],
            'Volatility mean reversion speed (kappa)': df_heston_params.loc['kappa'],
            'Long-term volatility (theta)': df_heston_params.loc['theta'],
            'Volatility of volatility (sigma)': df_heston_params.loc['sigma'],
        })
        default_params_df = params_df.copy()
        param_input = st.data_editor(
            default_params_df,
            num_rows="fixed",
            use_container_width=True,
            key="manual_param_editor"
        )
        default_corr_df = dB_dW_corr.copy()
        # Round for readability
        default_corr_df = default_corr_df.round(3)
        corr_input = st.data_editor(
            default_corr_df,
            num_rows="fixed",
            use_container_width=True,
            key="manual_corr_editor"
        )
        # Convert user inputs
        params_df2 = pd.DataFrame(param_input, columns=["Annual Return", "Volatility mean reversion speed (kappa)", "Long-term volatility (theta)", "Volatility of volatility (sigma)"])
        corr_df = pd.DataFrame(corr_input)
        # Enforce symmetry and identity diagonal
        corr_df = (corr_df + corr_df.T) / 2
        np.fill_diagonal(corr_df.values, 1.0)
        # Validation simple (optionnelle)
        if not ((corr_df.values >= -1).all() and (corr_df.values <= 1).all()):
            st.warning("Some correlation values are out of bounds [-1, 1].")

        # Update model parameters
        params_heston = df_heston_params.copy()
        corr_cal = dB_dW_corr.copy()
        params_heston.loc['mu'] = params_df2["Annual Return"]
        params_heston.loc['kappa'] = params_df2["Volatility mean reversion speed (kappa)"]
        params_heston.loc['theta'] = params_df2["Long-term volatility (theta)"]
        params_heston.loc['sigma'] = params_df2["Volatility of volatility (sigma)"]
        Parameters_simu = {'Parameters Heston': params_heston,
                            'dB_dW correlation': corr_df}
        market_model_simu.set_parameters(Parameters_simu)
    # Simulation parameters
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        begin_date = st.date_input("Begin Date", value=datetime.now().date())
        alpha_var = st.slider("Alpha for VaR", min_value=0.950, max_value=0.995, value=0.99, step=0.005, format="%.3f")
        alpha_ES = st.slider("Alpha for ES", min_value=0.950, max_value=0.995, value=0.975, step=0.005, format="%.3f")
    
    with col2:
        end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=365)).date())
        nbinsx = st.slider("Number of bins for histogram", min_value=1, max_value=100, value=30, step=1)
    
    with col3:
        num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=1000, value=50)
        T_recompute = st.slider("Recomputing period", min_value=20, max_value=300, value=300, step=1)
        st.write("Note: The recomputing period is the period in days for recomputing the allocation target. If it equals 300, the allocation target won't be recomputed.")
        T_recompute = 0 if T_recompute == 300 else T_recompute
    
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
                model=market_model_simu,
                strategy=strategy,
                parameters=parameters
            )
            
            # Generate scenarios and evolutions
            simulation.generate_scenarios()
            simulation.generate_evolutions(T_allocation = T_recompute)
            
            # Compute risk metrics
            simulation.compute_metrics(alpha_var=alpha_var, alpha_ES=alpha_ES)
        
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
        var_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in simulation.evolutions.items()}).quantile(1-alpha_var, axis=1)
        
        fig.add_trace(go.Scatter(
            x=var_evolution.index,
            y=var_evolution,
            mode='lines',
            name=f'VaR ({alpha_var*100}%)',
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
            nbinsx=nbinsx
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
            x=np.percentile(terminal_values, 100 * (1-alpha_var)),
            line_dash="dash",
            line_color="black",
            annotation_text=f"VaR ({alpha_var*100}%): {np.percentile(terminal_values, 100 * (1-alpha_var)):.4f}",
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

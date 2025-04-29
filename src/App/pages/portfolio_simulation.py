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

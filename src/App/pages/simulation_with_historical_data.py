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

def simulation_with_historical_data_page(data, cac40_weights):
    """Portfolio simulation with historical data"""
    st.title("Portfolio simulation on historical data")
    st.write("This page allows you to simulate a portfolio using historical data.")
     # Select companies for portfolio
    companies = data.columns.tolist()
    IsSelect_all = st.checkbox("Take all companies", value=False)
    selected_companies = companies if IsSelect_all else st.multiselect("Select Companies for Portfolio", companies, default=companies[:10])
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
    # Display allocation as a pie chart if there are 20 or fewer companies
    if len(selected_companies) <= 20:
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
    
    # Simulation
    st.subheader("Simulation")
    logreturn_data = np.log(filtered_data / filtered_data.shift() )
    logreturn_data.iloc[0,:] = 0.0

    begin_date_data = str(data.index[0])[:10]
    end_date_data = str(data.index[-1])[:10]
    # Create a MarketModel object
    market_model_BS_CAC40 = MarketModel(model_name="BS")
    market_model_BS_CAC40.fit(filtered_data)
    scenario_histo = {
        'Scenario 1' : logreturn_data
    }

    
    col1, col2 = st.columns(2)
    with col1:
        alpha_var = st.slider("Alpha for VaR", min_value=0.950, max_value=0.995, value=0.99, step=0.005, format="%.3f")
    with col2:
        alpha_ES = st.slider("Alpha for ES", min_value=0.950, max_value=0.995, value=0.975, step=0.005, format="%.3f")

    big_col1, big_col2 = st.columns(2)
    with big_col1:
        strategy1  = st.selectbox("Investment Strategy 1", ["Buy and hold", "Rebalancing"], index=0, key="strategy1")
        rebalancing_period1 = st.slider("Rebalancing Period 1 (days)", min_value=1, max_value=90, value=20, key="rebalancing_period1") if strategy1 == "Rebalancing" else -1
    with big_col2:
        strategy2  = st.selectbox("Investment Strategy 2", ["Buy and hold", "Rebalancing"], index=1, key="strategy2")
        rebalancing_period2 = st.slider("Rebalancing Period 2 (days)", min_value=1, max_value=90, value=20, key="rebalancing_period2") if strategy2 == "Rebalancing" else -1
    if strategy1 == strategy2 and rebalancing_period1 == rebalancing_period2:
        st.warning("Please select two different strategies.")
        return
    # Create simulation parameters for strategy 1
    parameters1 = {
        "Begin date": begin_date_data,
        "End date": end_date_data,
        "Allocation": allocation,
        "Rebalancing period": rebalancing_period1
    }
    # Create simulation parameters for strategy 2
    parameters2 = {
        "Begin date": begin_date_data,
        "End date": end_date_data,
        "Allocation": allocation,
        "Rebalancing period": rebalancing_period2
    }
    
    # Run simulation
    nbinsx = st.slider("Number of bins", min_value=100, max_value=400, value=250, step=1, key="nbinsx" )
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            simulation1 = Simulation(
            nb_scenarios=1,
            model=market_model_BS_CAC40,
            strategy=strategy1,
            parameters=parameters1
        )
            simulation2 = Simulation(
                nb_scenarios=1,
                model=market_model_BS_CAC40,
                strategy=strategy2,
                parameters=parameters2
            )
            simulation1.set_scenarios(scenario_histo)
            simulation2.set_scenarios(scenario_histo)
            simulation1.generate_evolutions()
            simulation2.generate_evolutions()
        # Display results
        st.subheader("Results")
        # Display portfolio evolution for the two strategies
        fig1 = go.Figure()
        evolution1 = simulation1.evolutions["Evolution 1"]
        name1 = f"{strategy1} - {rebalancing_period1} days" if strategy1 == "Rebalancing" else strategy1
        fig1.add_trace(go.Scatter(x=evolution1.index, 
                                y=evolution1.sum(axis=1), 
                                mode='lines', 
                                name=name1,
                                line=dict(color='blue')))
        evolution2 = simulation2.evolutions["Evolution 1"]
        name2 = f"{strategy2} - {rebalancing_period2} days" if strategy2 == "Rebalancing" else strategy2
        fig1.add_trace(go.Scatter(x=evolution2.index, 
                                y=evolution2.sum(axis=1), 
                                mode='lines', 
                                name=name2,
                                line=dict(color='red')))
        st.plotly_chart(fig1, use_container_width=True)
        # Display logreturn distribution for the two strategies
        fig2 = go.Figure()
        sum_evol1 = evolution1.sum(axis=1)
        sum_evol2 = evolution2.sum(axis=1)
        logreturns1 = np.log(sum_evol1 / sum_evol1.shift())
        logreturns2 = np.log(sum_evol2 / sum_evol2.shift())
        #drop na values
        logreturns1 = logreturns1.dropna()
        logreturns2 = logreturns2.dropna()
        # Create histogram for logreturns1
        fig2.add_trace(go.Histogram(
            x=logreturns1,
            name=name1,
            histnorm='probability density',
            opacity=0.5,
            marker=dict(color='blue'),
            nbinsx=nbinsx
        ))
        # Create histogram for logreturns2
        fig2.add_trace(go.Histogram(
            x=logreturns2,
            name=name2,
            histnorm='probability density',
            opacity=0.5,
            marker=dict(color='red'),
            nbinsx=nbinsx
        ))
        # Update layout
        fig2.update_layout(
            title="Logreturn Distribution",
            xaxis_title="Logreturn",
            yaxis_title="Density",
            barmode= 'overlay',
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)
        # Display statistics for the two strategies
        stats1 = pd.DataFrame({
            'Mean': logreturns1.mean(),
            'Std Dev': logreturns1.std(),
            'Min': logreturns1.min(),
            'Max': logreturns1.max(),
            f'VaR ({alpha_var*100}%)': logreturns1.quantile(1-alpha_var),
            f'EShortfall ({alpha_ES*100}%)': logreturns1[logreturns1 <= logreturns1.quantile(1-alpha_ES)].mean(),
            'Skewness': logreturns1.skew(),
            'Kurtosis': logreturns1.kurtosis()
        }, index=[name1])
        stats2 = pd.DataFrame({
            'Mean': logreturns2.mean(),
            'Std Dev': logreturns2.std(),
            'Min': logreturns2.min(),
            'Max': logreturns2.max(),
            f'VaR ({alpha_var*100}%)': logreturns2.quantile(1-alpha_var),
            f'EShortfall ({alpha_ES*100}%)': logreturns2[logreturns2 <= logreturns2.quantile(1-alpha_ES)].mean(),
            'Skewness': logreturns2.skew(),
            'Kurtosis': logreturns2.kurtosis()
        }, index=[name2])
        stats = pd.concat([stats1, stats2])
        st.write("**Statistics**")
        st.dataframe(stats.style.background_gradient(cmap='viridis'))
        

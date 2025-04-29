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

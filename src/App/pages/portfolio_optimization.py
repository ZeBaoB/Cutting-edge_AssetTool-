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

def portfolio_optimization_page(data, data_esg):
    """Portfolio Optimization page for optimizing portfolios with ESG constraints"""
    st.title("Portfolio Optimization")
    
    # Select companies for portfolio
    companies = data.columns.tolist()
    selected_companies = st.multiselect("Select Companies for Portfolio", companies, default=companies[:5])
    
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
    
    # Simulation parameters
    st.subheader("Model")
    # Select model
    model_type = st.selectbox("Select Model", ["Black-Scholes", "Heston"])
    model_type = "BS" if model_type == "Black-Scholes" else model_type
    # Calibrate BS model
    with st.spinner("Calibrating Black-Scholes model..."):
        market_model_BS = MarketModel(model_name="BS")
        market_model_BS.fit(filtered_data)
        volatilities_BS = market_model_BS.parameters["Volatilities"]
        parameters_test = {
            "Begin date": "2023-01-01",
            "End date": "2024-01-01",
            "Rebalancing period": -1
        }
        simu_BS = Simulation(
                nb_scenarios=1,
                model=market_model_BS,
                strategy="Buy and hold",
                parameters=parameters_test
            )
    
    # Calibrate model for simulation
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
        st.write("Correlation matrix:")
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
        st.write("Heston parameters for the stocks:")
        param_input = st.data_editor(
            default_params_df,
            num_rows="fixed",
            use_container_width=True,
            key="manual_param_editor"
        )
        default_corr_df = dB_dW_corr.copy()
        # Round for readability
        default_corr_df = default_corr_df.round(3)
        st.write("Correlation matrix:")
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
    
    # Display ESG data
    st.subheader("ESG Data")
    
    # Display as a table
    st.dataframe(filtered_esg.style.background_gradient(cmap='RdYlGn_r', subset=['Sustainability risk', 'Exposure risk', 'Carbon risk', 'Carbon intensity (Tons of CO2)'])
                                  .background_gradient(cmap='RdYlGn', subset=['Score management']))
    
    # ESG constraints
    st.subheader("Constraints")
    
    # Create columns for constraints
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Financial Constraints")
        max_allocation = st.slider("Maximum Allocation per Stock", min_value= 1/len(selected_companies), max_value=1.0, value=max(1/len(selected_companies), 0.3), step=0.005)
        max_volatility = st.slider("Maximum Portfolio Volatility", min_value=0.05, max_value=volatilities_BS.max(), value=0.22, step=0.005)
    
    with col2:
        st.write("ESG Constraints")
        sustainability_risk_max = filtered_esg['Sustainability risk'].max()
        sustainability_risk_min = filtered_esg['Sustainability risk'].min()
        max_sustainability_risk = st.slider("Maximum Sustainability Risk", min_value=sustainability_risk_min, max_value=sustainability_risk_max, value=sustainability_risk_max, step=0.1)
        exposure_risk_max = filtered_esg['Exposure risk'].max()
        exposure_risk_min = filtered_esg['Exposure risk'].min()
        max_exposure_risk = st.slider("Maximum Exposure Risk", min_value=exposure_risk_min, max_value=exposure_risk_max, value=exposure_risk_max, step=0.1) 
        score_management_min = filtered_esg['Score management'].min()
        score_management_max = filtered_esg['Score management'].max()
        min_score_management = st.slider("Minimum Score Management", min_value=score_management_min, max_value=score_management_max, value=score_management_min, step=0.1)
        carbon_risk_max = filtered_esg['Carbon risk'].max()
        carbon_risk_min = filtered_esg['Carbon risk'].min()
        max_carbon_risk = st.slider("Maximum Carbon Risk", min_value=carbon_risk_min, max_value=carbon_risk_max, value=carbon_risk_max, step=0.1)
        carbon_intensity_min = filtered_esg['Carbon intensity (Tons of CO2)'].min()
        carbon_intensity_max = filtered_esg['Carbon intensity (Tons of CO2)'].max()
        max_carbon_intensity = st.slider("Maximum Carbon Intensity", min_value=carbon_intensity_min, max_value=carbon_intensity_max, value=carbon_intensity_max, step=0.1)
    # Create constraints dictionary
    constraints = {
        "List": ["Maximal allocation", "Maximal volatility", "Maximal Carbon risk", "Minimal Score management", "Maximal Exposure risk", "Maximal Carbon intensity", "Maximal Sustainability risk", "Maximal Carbon intensity (Tons of CO2)"],
        "Value": [max_allocation, max_volatility, max_carbon_risk, min_score_management, max_exposure_risk, max_carbon_intensity, max_sustainability_risk, max_carbon_intensity]
    }
    
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
    rebalancing_period = st.slider("Rebalancing Period (days)", min_value=1, max_value=90, value=20) if strategy == "Rebalancing" else -1

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
                model=market_model_simu,
                strategy=strategy,
                parameters=parameters
            )
            # Set ESG data and constraints
            simulation.set_dataESG(filtered_esg)
            simulation.set_constraints(constraints)
            simu_BS.set_dataESG(filtered_esg)
            simu_BS.set_constraints(constraints)
            
            # Compute optimal allocation
            try:
                allocation_init = simu_BS.compute_allocation()
                simulation.set_allocation(allocation_init)
                if allocation_init is None:
                    st.error("No optimal allocation found. Please adjust your constraints.")
                    return
                # Generate scenarios and evolutions
                simulation.generate_scenarios()
                simulation.generate_evolutions(T_allocation=T_recompute)
                # Compute risk metrics
                simulation.compute_metrics(alpha_var=alpha_var, alpha_ES=alpha_ES)
                optimization_success = True
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.info("Try relaxing the constraints or selecting different companies. there might be no solution.")
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
            # Sort by allocation and filter out zero allocations
            allocation_df = allocation_df.sort_values('Allocation', ascending=False)
            allocation_df = allocation_df[allocation_df['Allocation'] > 1e-5]
            # Display as a table
            st.dataframe(allocation_df.style.background_gradient(cmap='viridis', subset=['Allocation']))
            
            # Display allocation as a pie chart
            values_allocation = allocation_df['Allocation'].values
            names_allocation = allocation_df['Company'].values
            fig = px.pie(
                values=values_allocation,
                names=names_allocation,
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
            portfolio_sustainability_risk = optimal_allocation @ filtered_esg['Sustainability risk']
            portfolio_exposure_risk = optimal_allocation @ filtered_esg['Exposure risk']
            portfolio_score_management = optimal_allocation @ filtered_esg['Score management']
            portfolio_carbon_risk = optimal_allocation @ filtered_esg['Carbon risk']
            portfolio_carbon_intensity = optimal_allocation @ filtered_esg['Carbon intensity (Tons of CO2)']
            
            # Display portfolio metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Sustainability Risk', 'Exposure Risk', 'Score Management', 'Carbon Risk', 'Carbon Intensity'],
                'Value': [
                    f"{portfolio_return:.4f} ({portfolio_return*100:.2f}%)",
                    f"{portfolio_volatility:.4f} ({portfolio_volatility*100:.2f}%)",
                    f"{portfolio_sharpe:.4f}",
                    f"{portfolio_sustainability_risk:.4f}",
                    f"{portfolio_exposure_risk:.4f}",
                    f"{portfolio_score_management:.4f}",
                    f"{portfolio_carbon_risk:.4f}",
                    f"{portfolio_carbon_intensity:.4f}"
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

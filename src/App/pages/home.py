import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# import function for app
from src.App.utils import load_data

def home_page():
    """Home page with project overview"""
    st.title("Stocks price analysis and Portfolio Optimization")
    
    st.markdown("""
    This application provides tools for financial modeling, portfolio simulation, and ESG-constrained portfolio optimization. 
    It includes implementations of various financial models and portfolio strategies.
    
    ## Features
    
    - **Data Explorer**: Visualize historical price data
    - **Market Models**: Calibrate and visualize models such as Black-Scholes, Heston, and Merton Jump-Diffusion
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

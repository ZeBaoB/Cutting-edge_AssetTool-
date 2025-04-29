import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# import function for app
from src.App.utils import load_data
# import pages
from src.App.pages import (
    home_page,
    data_explorer_page,
    market_model_page,
    portfolio_simulation_page,
    esg_optimization_page,
)

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

# Main function
def main():
    # Load data
    with st.spinner("Loading data..."):
        data, data_10y_dic, data_1min_dic, data_esg, cac40_weights = load_data()
    
    # Sidebar
    st.sidebar.title("Asset management. Analysis and optimization")
    
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

# Run the app
if __name__ == "__main__":
    main()

import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# import function for app
from src.App.utils import black_scholes_model, merton_model, heston_model

def market_model_page(data):
    """Market Model page for calibrating and visualizing market models"""
    st.title("Market Model")
    
    # Select model
    model_type = st.selectbox("Select Model", ["Black-Scholes", "Heston", "Merton Jump-Diffusion"])
    
    if model_type == "Black-Scholes":
        black_scholes_model(data)
    elif model_type == "Heston":
        heston_model(data)
    else:
        merton_model(data)

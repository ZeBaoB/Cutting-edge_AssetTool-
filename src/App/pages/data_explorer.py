import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
    normalize = False
    if chart_type == "Line":
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
            'VaR (99%)': returns_df.quantile(0.01),
            'EShortfall (97.5%)': returns_df[returns_df <= returns_df.quantile(0.025)].mean(),
            'Skewness': returns_df.skew(),
            'Kurtosis': returns_df.kurtosis()
        })
        st.write("**Statistics**")
        st.dataframe(stats.style.background_gradient(cmap='viridis'))
        
        # Plot returns distribution
        fig = go.Figure()
        st.write("**Distribution**")
        nbinsx = st.slider("Number of bins", min_value=10, max_value=100, value=30, step=1)
        for company in returns_df.columns:
            fig.add_trace(go.Histogram(
                x=returns_df[company],
                name=company,
                opacity=0.6,
                nbinsx=nbinsx,
                histnorm='probability density',
            ))
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_title="Log Returns",
            yaxis_title="Density",
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot returns correlation matrix
        corr_matrix = returns_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            title="Returns Correlation Matrix",
            text_auto=".3f",  # Display values in each cell, 3 decimal places
            zmin=-0.7,
            zmax=1.0
        )
        fig.update_traces(textfont_size=20)  # Update font size for text in cells
        
        fig.update_layout(
            template='plotly_dark',
            height=800
        )
        
        st.plotly_chart(fig, use_container_width=True)

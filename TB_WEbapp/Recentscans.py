import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import skew, kurtosis 
from google.cloud import firestore
from google.cloud.firestore import FieldFilter
from datetime import datetime, timedelta
import time
import zipfile
import os
import random
from google.api_core.exceptions import ResourceExhausted, RetryError
from collections import defaultdict
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go

def exponential_backoff(retries):
    base_delay = 1
    max_delay = 60
    delay = base_delay * (2 ** retries) + random.uniform(0, 1)
    return min(delay, max_delay)

def get_firestore_data(query):
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            results = query.stream()
            return list(results)
        except ResourceExhausted as e:
            st.warning(f"Quota exceeded, retrying... (attempt {retries + 1})")
            time.sleep(exponential_backoff(retries))
            retries += 1
        except RetryError as e:
            st.warning(f"Retry error: {e}, retrying... (attempt {retries + 1})")
            time.sleep(exponential_backoff(retries))
            retries += 1
        except Exception as e:
            st.error(f"An error occurred: {e}")
            break
    raise Exception("Max retries exceeded")


# Set page configuration
st.set_page_config(layout="wide")
st.title("Farm Analytics")

db = firestore.Client.from_service_account_json("TB_WEbapp/testdata1-20ec5-firebase-adminsdk-an9r6-d15c118c96.json")

# Fetch the most recent scan data from the "demo_db" collection
def get_recent_scans(db, num_scans=2):
    docs = (
        db.collection('demo_db')
        .order_by('timestamp', direction=firestore.Query.DESCENDING)
        .limit(num_scans)
        .stream()
    )
    radar_data_list = []
    timestamps = []
    for doc in docs:
        data_dict = doc.to_dict()
        radar_raw = data_dict.get('RadarRaw', [])
        timestamp = data_dict.get('timestamp')
        radar_data_list.append(radar_raw)
        timestamps.append(timestamp)
    return radar_data_list, timestamps

# Preprocess data for each scan
def preprocess_multiple_scans(radar_data_list):
    processed_data_list = []
    for radar_raw in radar_data_list:
        df_radar = pd.DataFrame(radar_raw, columns=['Radar'])
        df_radar.dropna(inplace=True)
        df_radar.fillna(df_radar.mean(), inplace=True)
        processed_data_list.append(df_radar)
    return processed_data_list

# Function to calculate statistics
def calculate_statistics(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(), inplace=True)
    stats = {
        'Column': df.columns,
        'Mean': df.mean(),
        'Median': df.median(),
        #'Std Deviation': df.std(),
        'PTP': df.apply(lambda x: np.ptp(x)),
        #'Skewness': skew(df),
        #'Kurtosis': kurtosis(df),
        'Min': df.min(),
        'Max': df.max()
    }
    stats_df = pd.DataFrame(stats)
    return stats_df

# Plot multiple scans in time domain
def plot_multiple_time_domain(data_list, timestamps):
    st.write("## Time Domain")
    # Initialize the Plotly figure
    fig = go.Figure()
    
    # Define colors for the different scans
    colors = ['#E24E42', '#59C3C3']
    
    # Add traces (lines) for each scan
    for i, data in enumerate(data_list):
        fig.add_trace(go.Scatter(
            y=data,  # Plot the raw index data on the y-axis
            mode='lines',
            name=f'Scan {i+1} - {timestamps[i].strftime("%Y-%m-%d %H:%M:%S")}',
            line=dict(color=colors[i])
        ))
    
    # Update layout for transparent background
    fig.update_layout(
        template='plotly_white',  # Use a template with no dark background
        xaxis_title="Index",  # Raw index numbers
        yaxis_title="Signal",
        legend_title="Scans",
        font=dict(color="black"),  # Adjust text color if needed
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    # Render the plot using Streamlit
    st.plotly_chart(fig)
    return fig
    
# Plot multiple scans in frequency domain using Plotly
def plot_multiple_frequency_domain(data_list, timestamps):
    st.write("## Frequency Domain")
    fig = go.Figure()

    colors = ['green', 'blue']

    for i, data in enumerate(data_list):
        # Perform FFT
        frequencies = np.fft.fftfreq(len(data), d=1/100)
        fft_values = np.fft.fft(data)
        powers = np.abs(fft_values) / len(data)
        powers_db = 20 * np.log10(powers)

        # Add trace to the Plotly figure
        fig.add_trace(go.Scatter(
            x=frequencies[:len(frequencies)//2], 
            y=powers_db[:len(powers_db)//2], 
            mode='lines',
            name=f'Scan {i+1} - {timestamps[i].strftime("%Y-%m-%d %H:%M:%S")}',
            line=dict(color=colors[i])
        ))

    # Update layout for transparent background
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectrum (dB)",
        legend_title="Scans",
        font=dict(color="white"),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    st.plotly_chart(fig)
    return fig

# Plot statistics for multiple scans using Plotly
def plot_multiple_statistics(stats_dfs, timestamps):
    st.write("## Radar Column Statistics")
    
    fig = go.Figure()

    stats_measures = ['Mean', 'Median', 'PTP', 'Min', 'Max']
    colors = ['green', 'blue']

    for i, stats_df in enumerate(stats_dfs):
        for measure in stats_measures:
            fig.add_trace(go.Bar(
                x=stats_measures,
                y=[stats_df[measure].values[0] for measure in stats_measures],  # Assuming one radar column
                name=f'Scan {i+1} - {timestamps[i].strftime("%Y-%m-%d %H:%M:%S")}',
                marker_color=colors[i],
            ))

    # Update layout for transparent background
    fig.update_layout(
        barmode='group',
        template='plotly_white',
        xaxis_title="Statistics",
        yaxis_title="Values",
        font=dict(color="white"),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    st.plotly_chart(fig)
    return fig

def main():
    radar_data_list, timestamps = get_recent_scans(db, num_scans=2)

    if radar_data_list:
        # Preprocess data for each scan
        processed_data_list = preprocess_multiple_scans(radar_data_list)
        
        # Display timestamps of scans
        #st.markdown(f"**Timestamps of Recent Scans:** {', '.join([ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps])}")
        st.markdown(f"**DATA ANALAYSIS OF 2 RECENT SCANS**")

        # Create three columns for plots
        col1, col2, col3 = st.columns(3)

        # Time domain plot for multiple scans
        with col1:
            time_fig = plot_multiple_time_domain([df['Radar'].values for df in processed_data_list], timestamps)
            

        # Frequency domain plot for multiple scans
        with col2:
            freq_fig = plot_multiple_frequency_domain([df['Radar'].values for df in processed_data_list], timestamps)
            

        # Statistics plot for multiple scans
        with col3:
            stats_dfs = [calculate_statistics(df) for df in processed_data_list]
            stats_fig = plot_multiple_statistics(stats_dfs, timestamps)
    else:
        st.error("No data available in the 'Dananjay Yadav' collection.")

if __name__ == "__main__":
    main()

st.write(f"**Farmer Name:** Dananjay Yadav", color='white')
st.write(f"**Farm Location:** Rahuri Nashik", color='white')
st.write(f"**Farm Age:** 7 Years", color='white')
st.write(f"**Plot Size:** 2.5 Acre", color='white')

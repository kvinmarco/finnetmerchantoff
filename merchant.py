import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px 
import requests
import numpy as np
from io import BytesIO

# Title of the Streamlit app
st.set_page_config(page_title='FM OFF MERCHANT FINNET', page_icon=':bar_chart:')
st.title('Customer Segmentation using FM Analysis for FINNET')

# Function to load image from URL and convert to Matplotlib compatible format
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Load and preprocess the data
    df = pd.read_excel(uploaded_file)
    df = df.rename(columns={'jumlah': 'Frequency', 'merchant_name': 'Merchant Name', 'amount': 'Monetary', 'merchant': 'Merchant Type'})
    st.write("Preview of the dataset:")
    st.write(df.head())

    # Display total monetary spent
    total_monetary_spent = df['Monetary'].sum()
    st.write(f'Total Monetary spent: Rp{total_monetary_spent:,.2f}')

    # Grouping for Frequency and Monetary
    customer_spending = df.groupby('Merchant Name')['Monetary'].sum().reset_index()
    customer_frequency = df.groupby('Merchant Name')['Frequency'].sum().reset_index()
    df_fm = pd.merge(customer_frequency, customer_spending, on='Merchant Name')
    df_fm = df_fm[['Merchant Name', 'Frequency', 'Monetary']]
    df_fm[['Frequency', 'Monetary']] = df_fm[['Frequency', 'Monetary']].astype(int)
    
    # Copy for returning values
    original_fm = df_fm[['Merchant Name', 'Frequency', 'Monetary']].copy()

    # StandardScaler Scaling Function
    sc = StandardScaler()
    df_fm[['Frequency', 'Monetary']] = sc.fit_transform(df_fm[['Frequency', 'Monetary']])

    # Function to plot 2D scatter
    def plot_2d_scatter(df_fm, n_clusters):
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_fm['Frequency'], df_fm['Monetary'], c=df_fm['Cluster'], cmap='viridis', s=50)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Monetary')
        ax.set_title(f'Clustering of Customers based on Frequency and Monetary (Clusters: {n_clusters})')
        legend = ax.legend(*scatter.legend_elements(), title='Clusters')
        ax.add_artist(legend)
        st.pyplot(fig)

    # Elbow Method for K-Means Clustering
    st.subheader('Elbow Method for Optimal Number of Clusters')
    data = df_fm[['Frequency', 'Monetary']]
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    st.pyplot(plt)
    
    # Slider to select number of clusters
    n_clusters = st.slider('Select number of clusters', 1, 10, 4)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_fm['Cluster'] = kmeans.fit_predict(data)
    
    # Scale RFM values for visualization
    scaler = MinMaxScaler(feature_range=(0, 5))
    df_fm[['Frequency', 'Monetary']] = scaler.fit_transform(df_fm[['Frequency', 'Monetary']])
    
    # Plot scaled clusters with background image
    st.subheader('Clusters with Background Image')
    background_image_url = "https://github.com/kvinmarco/rfmapp.py/raw/main/rfmtable.png"
    background_image = load_image_from_url(background_image_url)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(background_image, extent=[0, 5, 0, 5], aspect='auto', alpha=0.4)
    scatter = ax.scatter(df_fm['Frequency'], df_fm['Monetary'], c=df_fm['Cluster'], cmap='viridis', s=50)
    ax.set_xlabel('Frequency (Scaled)')
    ax.set_ylabel('Monetary (Scaled)')
    ax.set_title('Scaled Clusters based on Frequency and Monetary')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # Segment definition
    segments = {
        'Lost': ((-1, 2), (-1, 2)),
        'Hibernating': ((-1, 2), (2, 4)),
        'Can’t Lose Them': ((-1, 2), (4, 6)),
        'About to Sleep': ((2, 3), (-1, 2)),
        'Needs attention': ((2, 3), (2, 3)),
        'Loyal Customers': ((2, 4), (3, 6)),
        'Promising': ((3, 4), (-1, 1)),
        'Potential Loyalist': [((3, 4), (1, 3)), ((4, 6), (2, 3))],
        'Price Sensitive': ((4, 6), (-1, 1)),
        'Recent users': ((4, 6), (1, 2)),
        'Champions': ((4, 6), (3, 6))
    }
    
    def assign_segment(row):
        for segment, bounds in segments.items():
            if isinstance(bounds, list):
                for (x_range, y_range) in bounds:
                    if x_range[0] <= row['Frequency'] <= x_range[1] and y_range[0] <= row['Monetary'] <= y_range[1]:
                        return segment
            else:
                x_range, y_range = bounds
                if x_range[0] <= row['Frequency'] <= x_range[1] and y_range[0] <= row['Monetary'] <= y_range[1]:
                    return segment
        return 'Other'

    df_fm['Segment'] = df_fm.apply(assign_segment, axis=1)

    # Button to show customer segments
    if st.button('Show Customer Segments'):
        # Calculate segment stats
        df_fm['Original Frequency'] = original_fm['Frequency']
        df_fm['Original Monetary'] = original_fm['Monetary']
        segment_stats = df_fm.groupby('Segment').agg({
            'Merchant Name': 'count',
        }).reset_index()

        original_segment_stats = df_fm.groupby('Segment').agg({
            'Original Frequency': 'mean',
            'Original Monetary': 'mean'
        }).reset_index()

        segment_stats = segment_stats.merge(original_segment_stats, on='Segment')
        segment_stats.columns = ['Segment', 'Count', 'Average Frequency', 'Average Monetary']

        total_customers = df_fm.shape[0]
        segment_stats['Percentage'] = (segment_stats['Count'] / total_customers) * 100
        segment_stats['Percentage'] = segment_stats['Percentage'].apply(lambda x: f'{x:.2f}%')

        # Prepare the data for displaying in a table
        segment_stats['Average Monetary(Spending)'] = segment_stats['Average Monetary'].apply(lambda x: f'Rp{x:,.2f}')
        segment_stats = segment_stats[['Segment', 'Count', 'Percentage', 'Average Frequency', 'Average Monetary(Spending)']]

        # Reset index and start from 1
        segment_stats.reset_index(drop=True, inplace=True)
        segment_stats.index += 1

        # Display the segment statistics in a table without the index
        st.table(segment_stats)
        
        # Plot pie chart for customer segments distribution
        fig = px.pie(segment_stats, names='Segment', values='Count',
                     hover_data=['Percentage', 'Average Frequency', 'Average Monetary(Spending)'],
                     title='Customer Segments Distribution')
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate=(
                '<b>%{label}</b><br>'
                'Count: %{value}<br>'
                'Percentage: %{customdata[0]}<br>'
            ),
            customdata=segment_stats[['Percentage']].values
        )
        st.plotly_chart(fig)
        
    # Multiselect to show Merchant Names based on segments
    selected_segments = st.multiselect('Select Segments to Display Merchant Names', df_fm['Segment'].unique())
    if selected_segments:
        filtered_df = df_fm[df_fm['Segment'].isin(selected_segments)]
        filtered_df = filtered_df.merge(original_fm, on='Merchant Name', suffixes=('', '_Original'))
        st.write(f"Merchants in selected segments: {', '.join(selected_segments)}")
        st.write(filtered_df[['Merchant Name', 'Frequency_Original', 'Monetary_Original', 'Segment']].rename(
            columns={'Frequency_Original': 'Frequency', 'Monetary_Original': 'Monetary'}).sort_values(by='Segment'))

    # Multiselect to select clusters for re-segmentation
    clusters_to_resegment = st.multiselect('Select Clusters to Re-Segment', df_fm['Cluster'].unique())
    resegment_clusters = st.button('Re-Segment Selected Clusters')

    if resegment_clusters:
        # Filter the DataFrame for the selected clusters
        filtered_df = df_fm[df_fm['Cluster'].isin(clusters_to_resegment)]
        
        # Perform re-segmentation on the filtered DataFrame
        data = filtered_df[['Frequency', 'Monetary']]
        inertias = []
        
        # Elbow method to determine the optimal number of clusters
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Plot the Elbow Graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), inertias, marker='o')
        plt.title('Elbow Graph')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        st.pyplot(plt)
        
        # Perform KMeans clustering with the optimal number of clusters
        optimal_clusters = 4  # Set the optimal number of clusters as determined from the Elbow Graph
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        filtered_df['Cluster'] = kmeans.fit_predict(data)
        
        # Plot the clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_df['Frequency'], filtered_df['Monetary'], c=filtered_df['Cluster'], cmap='viridis', s=50)
        plt.xlabel('Frequency')
        plt.ylabel('Monetary')
        plt.title('Clusters based on Frequency and Monetary')
        plt.colorbar(label='Cluster')
        st.pyplot(plt)
        
        # Scale the RFM values for visualization
        scaler = MinMaxScaler(feature_range=(0, 5))
        filtered_df[['Frequency', 'Monetary']] = scaler.fit_transform(filtered_df[['Frequency', 'Monetary']])
        
        # Plot the scaled clusters with a background image
        background_image_url = "https://github.com/kvinmarco/rfmapp.py/raw/main/rfmtable.png"
        background_image = load_image_from_url(background_image_url)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(background_image, extent=[0, 5, 0, 5], aspect='auto', alpha=0.4)
        scatter = ax.scatter(filtered_df['Frequency'], filtered_df['Monetary'], c=filtered_df['Cluster'], cmap='viridis', s=50)
        ax.set_xlabel('Frequency (Scaled)')
        ax.set_ylabel('Monetary (Scaled)')
        ax.set_title('Scaled Clusters based on Frequency and Monetary')
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)
        
        # Assign segments to the filtered and re-segmented DataFrame
        filtered_df['Segment'] = filtered_df.apply(assign_segment, axis=1)
        
        # Button to show customer segments for the re-segmented data
        if st.button('Show Customer Segments for Re-Segmented Data'):
            segment_stats = filtered_df.groupby('Segment').agg({
                'Merchant Name': 'count',
            }).reset_index()

            original_segment_stats = filtered_df.groupby('Segment').agg({
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).reset_index()

            segment_stats = segment_stats.merge(original_segment_stats, on='Segment')
            segment_stats.columns = ['Segment', 'Count', 'Average Frequency', 'Average Monetary']

            total_customers = filtered_df.shape[0]
            segment_stats['Percentage'] = (segment_stats['Count'] / total_customers) * 100
            segment_stats['Percentage'] = segment_stats['Percentage'].apply(lambda x: f'{x:.2f}%')

            # Prepare the data for displaying in a table
            segment_stats['Average Monetary(Spending)'] = segment_stats['Average Monetary'].apply(lambda x: f'Rp{x:,.2f}')
            segment_stats = segment_stats[['Segment', 'Count', 'Percentage', 'Average Frequency', 'Average Monetary(Spending)']]

            # Reset index and start from 1
            segment_stats.reset_index(drop=True, inplace=True)
            segment_stats.index += 1

            # Display the segment statistics in a table without the index
            st.table(segment_stats)
            
            fig = px.pie(segment_stats, names='Segment', values='Count',
                        hover_data=['Percentage', 'Average Frequency', 'Average Monetary(Spending)'],
                        title='Customer Segments Distribution for Re-Segmented Data')
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Count: %{value}<br>'
                    'Percentage: %{customdata[0]}<br>'
                ),
                customdata=segment_stats[['Percentage']].values
            )
            st.plotly_chart(fig)
            
        # Multiselect to select segments to display merchant names
        segments_to_display = st.multiselect('Select Segments to Display Merchant Names', filtered_df['Segment'].unique())
        show_merchants = st.button('Show Merchants in Selected Segments')

        if show_merchants:
            for segment in segments_to_display:
                st.write(f"{segment}:")
                merchants = filtered_df[filtered_df['Segment'] == segment]['Merchant Name'].tolist()
                for merchant in merchants:
                    st.write(f"- {merchant}")

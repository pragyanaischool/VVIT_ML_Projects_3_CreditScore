import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Page Config
st.set_page_config(page_title=" PragyanAI - Customer Segmentation Tool", layout="wide")
st.title(" PragyanAI Customer Segmentation Pro")
st.markdown("Upload a dataset or use our sample to cluster customers and test new inputs.")

# 2. Data Loading
@st.cache_data
def load_sample_data():
    # Standard Mall Customer Dataset
    url = "Mall_Customers.csv"
    df = pd.read_csv(url)
    df.columns = ['ID', 'Gender', 'Age', 'Income', 'SpendingScore']
    return df

df = load_sample_data()

# 3. Sidebar - Parameters
st.sidebar.header("Cluster Settings")
features = ['Age', 'Income', 'SpendingScore']
k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 8, 5)

# 4. Processing Pipeline
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# KMeans Model
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# PCA for Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PC1'] = pca_data[:, 0]
df['PC2'] = pca_data[:, 1]

# 5. UI Layout: Visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Cluster Visualization (PCA)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title(f"Customer Segments (K={k_clusters})")
    
    # 6. New Input Prediction
    st.divider()
    st.subheader("📍 Predict New Customer Segment")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        new_age = c1.number_input("Age", 18, 100, 30)
        new_income = c2.number_input("Annual Income (k$)", 15, 150, 50)
        new_score = c3.number_input("Spending Score (1-100)", 1, 100, 50)
        
        # Process New Input
        new_data = np.array([[new_age, new_income, new_score]])
        new_scaled = scaler.transform(new_data)
        new_cluster = kmeans.predict(new_scaled)[0]
        new_pca = pca.transform(new_scaled)
        
        # Plot new point on graph
        ax.scatter(new_pca[0,0], new_pca[0,1], c='red', marker='X', s=300, label='New Input')
        ax.legend()
        st.pyplot(fig)
        
        st.success(f"This customer belongs to **Cluster {new_cluster}**")

with col2:
    st.subheader("Data Preview")
    st.dataframe(df[['Age', 'Income', 'SpendingScore', 'Cluster']].head(10))
    
    st.subheader("Cluster Statistics")
    stats = df.groupby('Cluster')[features].mean()
    st.table(stats)

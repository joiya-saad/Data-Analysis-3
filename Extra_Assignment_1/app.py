import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (replace with your public link or file path)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/joiya-saad/Data-Analysis-3/refs/heads/main/Extra_Assignment_1/Bias-Variance-Results.csv"  # Replace with your CSV link
    return pd.read_csv(url)

# Load data
results_df = load_data()

# Streamlit app layout
st.title("Bias-Variance Tradeoff Visualization")

# Slicer for Polynomial Degree
min_degree = int(results_df['Degree'].min())
max_degree = int(results_df['Degree'].max())
degree_range = st.slider("Select Polynomial Degree Range", min_degree, max_degree, (min_degree, max_degree))

# Filter data based on slider
filtered_df = results_df[(results_df['Degree'] >= degree_range[0]) & (results_df['Degree'] <= degree_range[1])]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bias² plot
color = 'tab:blue'
ax1.set_xlabel('Degree of Polynomial')
ax1.set_ylabel('Bias²', color=color)
ax1.plot(filtered_df['Degree'], filtered_df['Bias^2'], marker='o', color=color, label='Bias²')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(degree_range[0], degree_range[1] + 1))

# Variance plot
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Variance', color=color)
ax2.plot(filtered_df['Degree'], filtered_df['Variance'], marker='s', linestyle='--', color=color, label='Variance')
ax2.tick_params(axis='y', labelcolor=color)

# Title and grid
plt.title('Bias-Variance Tradeoff Across Polynomial Degrees')
fig.tight_layout()
plt.grid(True)

# Show plot
st.pyplot(fig)

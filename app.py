import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model


# DL model + data

model = load_model('Mod16-32-16.keras')  # adjust path if needed
xData = np.loadtxt("data.csv", delimiter=",")
yData = xData[:,0]
xData = xData[:, 1:]
scaler = MinMaxScaler()
scaler = scaler.fit(xData)


# page config

st.set_page_config(
    page_title="FireSight",
    layout="wide"
)


# Sidebar w/ files

st.sidebar.header("Scenario Simulation")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with features: Precipitation, Max Temp, Min Temp, Avg Wind Speed, Year, Temp Range, Wind/Temp Ratio, Lagged Precipitation, Lagged Avg Wind Speed, Day of Year",
    type="csv"
)


# CSS styling

st.markdown("""
<style>
body {background: #fff5f0; font-family: 'Arial', sans-serif; color: #333;}
.header {background: linear-gradient(90deg, firebrick, orange); color: white; padding: 35px; text-align: center; border-radius: 0 0 20px 20px; box-shadow: 0 5px 25px rgba(0,0,0,0.3); animation: slideDown 1s ease-out;}
.header h1 {font-size: 56px; font-weight: bold; text-shadow: 2px 2px 12px rgba(0,0,0,0.5); margin: 0; transition: color 0.3s;}
.header h1:hover {color: yellow; transform: scale(1.03) rotateZ(1deg);}
.section-title {color: firebrick; font-size: 32px; margin-top: 50px; margin-bottom: 20px; text-align:center; animation: fadeIn 1s ease-in-out;}
@keyframes fadeIn {from { opacity: 0; transform: translateY(20px);} to { opacity: 1; transform: translateY(0);}}
@keyframes slideDown {from { opacity: 0; transform: translateY(-50px);} to { opacity: 1; transform: translateY(0);}}
</style>
""", unsafe_allow_html=True)


# header

st.markdown("""
<div class="header">
    <h1> FireSight</h1>
</div>
""", unsafe_allow_html=True)


# about us

st.write("""
FireSight is a machine learning-based tool designed to predict wildfire risk based on environmental data.
You can upload a CSV file with the relevant features and visualize predicted fire probabilities,
categorized risks, and relationships between variables.
""")
st.markdown("**Project Features:**")
st.write("""
- Predict wildfire probability using a trained neural network  
- Visualize risk distributions with interactive histograms  
- Explore relationships between variables with a scatter plot  
- Friendly column naming for easier analysis
""")
st.markdown("[View on GitHub](https://github.com/DeUniversalRoot/CodeThePast-Hackathon-Submission)")
st.markdown("[Powered by Zenodo](https://zenodo.org/records/14712845)")


#  predictions from model

if uploaded_file is not None:
    # Load CSV (assume no headers)
    user_data = np.loadtxt(uploaded_file, delimiter=",")
    
    # Scale features
    user_scaled = scaler.transform(user_data)
    
    # Predict probabilities
    probs = model.predict(user_scaled).flatten()
    
    # Categorize risk
    def categorize(p):
        if p > 0.6: return "High"
        elif p > 0.3: return "Moderate"
        else: return "Low"
    
    bye = pd.Series([categorize(p) for p in probs])

    
    # risk overview

    st.markdown("## Risk Overview")
    risk_counts = bye.value_counts().reindex(['High','Moderate','Low']).fillna(0)
    total = len(bye)
    risk_percent = (risk_counts / total * 100).round(1)

    col1, col2, col3 = st.columns(3)
    def card_html(title, color, count, percent):
        return f"""
        <div style='background:{color}; padding:20px; border-radius:12px; text-align:center; color:white'>
            <h3>{title}</h3>
            <p>{count} locations ({percent}%)</p>
        </div>
        """
    col1.markdown(card_html("🔴 High Risk","firebrick",risk_counts['High'],risk_percent['High']), unsafe_allow_html=True)
    col2.markdown(card_html("🟠 Moderate Risk","orange",risk_counts['Moderate'],risk_percent['Moderate']), unsafe_allow_html=True)
    col3.markdown(card_html("🟢 Low Risk","green",risk_counts['Low'],risk_percent['Low']), unsafe_allow_html=True)
    

    # bar graph for probability of fire

    st.markdown("<h2 class='section-title'>Fire Probability Histogram</h2>", unsafe_allow_html=True)
    bye_df = pd.DataFrame({'fire_probability': probs, 'risk': bye})
    fig_hist = px.histogram(
        bye_df,
        x='fire_probability',
        nbins=10,
        color='risk',
        color_discrete_map={'High':'firebrick','Moderate':'orange','Low':'green'},
        title="Probability Distribution"
    )
    fig_hist.update_traces(marker_line_width=0.5, marker_line_color="black")
    fig_hist.update_layout(transition={'duration':1000})
    st.plotly_chart(fig_hist, use_container_width=True)
    

    # scatterplot w/ customizable axes

    st.markdown("<h2 class='section-title'> Custom Scatter Plot</h2>", unsafe_allow_html=True)
    
    # Assign friendly column names to the uploaded CSV
    feature_names = [
        "Precipitation", "Max Temp", "Min Temp", "Avg Wind Speed", "Year",
        "Temp Range", "Wind/Temp Ratio", "Lagged Precipitation", "Lagged Avg Wind Speed", "Day of Year"
    ]
    
    scatter_df = pd.DataFrame(user_data, columns=feature_names)
    scatter_df['Fire Probability'] = probs
    scatter_df['Risk'] = bye
    
    # Select which numeric columns to plot
    numeric_cols = scatter_df.select_dtypes(include=np.number).columns.tolist()
    x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
    y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)
    
    fig_scatter = px.scatter(
        scatter_df,
        x=x_axis,
        y=y_axis,
        color='Risk',
        color_discrete_map={'High':'firebrick','Moderate':'orange','Low':'green'},
        size='Fire Probability',
        size_max=25,
        hover_data=scatter_df.columns
    )
    
    fig_scatter.update_layout(transition={'duration':1000})
    st.plotly_chart(fig_scatter, use_container_width=True)


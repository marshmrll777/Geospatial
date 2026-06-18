# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# 1. Broad Fluid Canvas Config
st.set_page_config(
    page_title="GeoGuard // Landslide AI",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Re-engineered Cyberpunk Neon UI Stylesheet
st.markdown("""
    <style>
    /* Dark Deep-Space Canvas Background */
    .stApp {
        background: linear-gradient(135deg, #0b0914 0%, #030307 100%) !important;
        color: #d1d1f0 !important;
    }
    
    /* Clean Sidebar Separation Matrix */
    section[data-testid="stSidebar"] {
        background-color: #07050e !important;
        border-right: 2px solid #00f2fe;
    }
    
    /* Elegant Clean Titles (No raw text clipping) */
    .dashboard-title {
        color: #ffffff;
        font-family: 'Courier New', monospace;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 5px;
        text-shadow: 0 0 8px #00f2fe;
    }
    
    /* Glassmorphic Unified Grid Blocks */
    .glass-card {
        background: rgba(15, 12, 28, 0.7);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Critical Alert Active Glow Panel */
    .critical-glow {
        background: rgba(30, 5, 18, 0.7);
        border: 1px solid #ff007f;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.3);
    }
    
    .neon-magenta { color: #ff007f; font-weight: bold; text-shadow: 0 0 5px #ff007f; }
    .neon-cyan { color: #00f2fe; font-weight: bold; text-shadow: 0 0 5px #00f2fe; }
    </style>
    """, unsafe_allow_html=True)

# 3. Secure Asset Extraction
@st.cache_resource
def load_assets():
    with open('landslide_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('uttarakhand_geospatial_data.csv')
    return model, df

try:
    model, df = load_assets()
except FileNotFoundError:
    st.error("🚨 Configuration matrix or model file missing! Execute 'train_model.py' first.")
    st.stop()

# 4. Standard Unified Title Banner
st.markdown("<h1 class='dashboard-title'>🛰️ GEOGUARD: InSAR HAZARD PLATFORM</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#767693; margin-top:-10px;'>Advanced Landslide Susceptibility Engine // Uttarakhand Grid Vector</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(0,242,254,0.1); margin-top:0px; margin-bottom:25px;'>", unsafe_allow_html=True)

# 5. Control Center Sidebar Layout
st.sidebar.markdown("<h3 class='neon-cyan'>⚡ VECTOR CONTROL</h3>", unsafe_allow_html=True)
st.sidebar.markdown("Adjust sensory variables to force diagnostic calculations:")

input_slope = st.sidebar.slider("Slope Inclination (Degrees)", 5.0, 50.0, 25.0)
input_moisture = st.sidebar.slider("Soil Moisture Saturation (%)", 10.0, 80.0, 45.0)
input_insar = st.sidebar.slider("InSAR Ground Velocity (mm/yr)", -30.0, 5.0, -5.0)

# 6. Balanced Two-Column Layout Grid
col1, col2 = st.columns([5, 7], gap="large")

with col1:
    st.markdown("<h3 class='neon-cyan'>🔮 Diagnostic Vector</h3>", unsafe_allow_html=True)
    
    # Run ML Inference Core
    features = np.array([[input_slope, input_moisture, input_insar]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] * 100
    
    # Conditional Styling Layout
    if prediction == 1:
        st.markdown(f"""
        <div class='critical-glow'>
            <h4 style='color: #ff007f; margin:0;'>⚠️ CRITICAL LANDSLIDE RISK ALERT</h4>
            <p style='font-size: 32px; margin: 12px 0; font-weight: bold; color: #ffffff;'>Probability: {prob:.1f}%</p>
            <p style='color: #ffb3d1; font-size: 14px; line-height:1.5;'>Active ground subsidence detected. Slope parameters suggest systemic shear failure constraints under hydro-saturation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='glass-card' style='border-color: #00f2fe;'>
            <h4 style='color: #00f2fe; margin:0;'>🛡️ STABLE GEOMORPHIC PROFILE</h4>
            <p style='font-size: 32px; margin: 12px 0; font-weight: bold; color: #ffffff;'>Probability: {prob:.1f}%</p>
            <p style='color: #b3f7ff; font-size: 14px; line-height:1.5;'>Sensory metrics normal. Radar satellite backscatter tracking displays negligible displacement vector variations.</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Standard Native Layout Metric Presentation (No misalignments)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Operational Array Input Vector:")
    st.write(f"📐 Slope Angle Matrix: **{input_slope}°**")
    st.write(f"💧 Moisture Saturation Index: **{input_moisture}%**")
    st.write(f"📡 Radar Line-of-Sight Shift: **{input_insar} mm/year**")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h3 class='neon-cyan'>🗺️ Spatial Hazard Projection</h3>", unsafe_allow_html=True)
    
    # Map Initialization
    fig = px.scatter_mapbox(
        df, 
        lat="Latitude", 
        lon="Longitude", 
        color="Landslide_Susceptibility",
        color_continuous_scale=[[0, '#00f2fe'], [1, '#ff007f']], 
        size="Slope_Angle", 
        hover_data=["InSAR_Displacement", "Soil_Moisture"],
        zoom=8.5,
        height=480
    )
    
    # FIXED: Replaced bad 'font' property with compliant Plotly title_font/tickfont structures
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title="Risk Profile",
            tickvals=[0, 1],
            ticktext=["Stable Grid", "Critical Risk Zone"],
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#ffffff")
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 7. Base Matrix Streaming Data Stream
st.markdown("<h3 class='neon-cyan'>📊 Real-Time Telemetry Matrix</h3>", unsafe_allow_html=True)
st.dataframe(
    df.head(10).style.background_gradient(cmap="coolwarm", subset=["InSAR_Displacement"]),
    use_container_width=True
)
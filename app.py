import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wind Farm Financial Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOAD THE MODELS ---
@st.cache_resource
def load_models():
    models = {}
    
    # Define the expected filenames
    model_files = {
        "Baseline Model (Vanilla)": "model_1_baseline.joblib",
        "Optimized Model (GridSearch)": "model_2_optimized.joblib",
        "Physics-Informed Model (Phase 7)": "model_3_physics.joblib"
    }
    
    for name, filename in model_files.items():
        if os.path.exists(filename):
            models[name] = joblib.load(filename)
        else:
            models[name] = None # Will handle missing files gracefully in UI
            
    return models

models_dict = load_models()

# --- 3. UI DASHBOARD HEADER ---
st.title("⚡ Gabal El Zeit: Wind Turbine Analytics")
st.markdown("""
Compare the progression of our Machine Learning models—from the baseline algorithm to the Physics-Informed Architecture ($R^2 = 0.9989$). 
Adjust the meteorological conditions to project real-time financial revenue.
""")
st.divider()

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("🧠 Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Engine:", 
    ["Physics-Informed Model (Phase 7)", "Optimized Model (GridSearch)", "Baseline Model (Vanilla)"]
)

st.sidebar.header("⚙️ Meteorological Inputs")
wind_speed = st.sidebar.slider("Current Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=12.0, step=0.1)
theoretical_power = st.sidebar.slider("Theoretical Power (kW)", min_value=0.0, max_value=3600.0, value=2500.0, step=10.0)
wind_dir = st.sidebar.slider("Wind Direction (Degrees)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)

st.sidebar.header("🕒 Temporal Context")
month = st.sidebar.selectbox("Month of Year", options=list(range(1, 13)), index=5)
hour = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=12, step=1)

st.sidebar.header("💰 Financial Market Parameters")
price_per_mwh = st.sidebar.number_input("Current Energy Price ($/MWh)", min_value=0.0, value=50.0, step=5.0)

# --- 5. DYNAMIC FEATURE ENGINEERING & PREDICTION ---
active_model = models_dict.get(selected_model_name)

if active_model is not None:
    # Build the correct DataFrame based on what the selected model expects
    if selected_model_name == "Physics-Informed Model (Phase 7)":
        input_df = pd.DataFrame([{
            'Wind_Speed': float(wind_speed),
            'Theoretical_Power': float(theoretical_power),
            'Wind_Speed_Cubed': float(wind_speed) ** 3,
            'Dir_Sin': np.sin(float(wind_dir) * (2. * np.pi / 360)),
            'Dir_Cos': np.cos(float(wind_dir) * (2. * np.pi / 360)),
            'Hour_Sin': np.sin(float(hour) * (2. * np.pi / 24)),
            'Hour_Cos': np.cos(float(hour) * (2. * np.pi / 24)),
            'Month_Sin': np.sin(float(month) * (2. * np.pi / 12)),
            'Month_Cos': np.cos(float(month) * (2. * np.pi / 12))
        }])
        accuracy_label = "99.89% R² Confidence"
    else:
        # Standard features for Baseline and Optimized models
        input_df = pd.DataFrame([{
            'Wind_Speed': float(wind_speed),
            'Theoretical_Power': float(theoretical_power),
            'Wind_Direction': float(wind_dir),
            'Month': int(month),
            'Hour': int(hour)
        }])
        accuracy_label = "93.75% R² Confidence" if "Optimized" in selected_model_name else "93.40% R² Confidence"

    # Generate Prediction
    prediction_kw = active_model.predict(input_df)[0]
    prediction_kw = max(0.0, float(prediction_kw))  # Prevent negative power
    
    # Efficiency calculations
    efficiency = (prediction_kw / theoretical_power) * 100 if theoretical_power > 0 else 0.0
    efficiency = min(100.0, efficiency)

    # Financials (Converting kW to MW)
    megawatts = prediction_kw / 1000.0
    hourly_revenue = megawatts * price_per_mwh
    daily_revenue = hourly_revenue * 24        
    annual_revenue = daily_revenue * 365       

    # --- 6. DASHBOARD DISPLAY ---
    st.subheader(f"📊 Real-Time Turbine Performance: {selected_model_name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Predicted Power Output", value=f"{prediction_kw:,.2f} kW", delta=accuracy_label)
    with col2:
        st.metric(label="Operating Efficiency", value=f"{efficiency:.1f} %", delta=f"{theoretical_power} kW Expected", delta_color="off")
    with col3:
        st.metric(label="Active Wind Speed", value=f"{wind_speed} m/s", delta="Live Meteorological Data", delta_color="normal")

    st.divider()

    st.subheader("💵 Estimated Revenue Projections")
    st.markdown("*(Assuming current meteorological conditions and market pricing remain stable)*")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.metric(label="Hourly Revenue generation", value=f"${hourly_revenue:,.2f}")
    with f2:
        st.metric(label="Projected Daily Revenue", value=f"${daily_revenue:,.2f}")
    with f3:
        st.metric(label="Projected Annual Revenue", value=f"${annual_revenue:,.2f}")

    # Add Price Sensitivity Graph
    st.divider()
    st.subheader("📈 Revenue Scaling (Price Sensitivity)")
    st.markdown("How revenue shifts if the Energy Price fluctuates today:")
    
    prices = np.linspace(30, 100, 15)
    revenues = megawatts * prices
    chart_data = pd.DataFrame({"Energy Price ($/MWh)": prices, "Hourly Revenue ($)": revenues})
    chart_data.set_index("Energy Price ($/MWh)", inplace=True)
    st.line_chart(chart_data)

else:
    st.error(f"⚠️ Model file for '{selected_model_name}' not found!")
    st.info("Please ensure 'model_1_baseline.joblib', 'model_2_optimized.joblib', and 'model_3_physics.joblib' are uploaded to your GitHub repository alongside app.py.")
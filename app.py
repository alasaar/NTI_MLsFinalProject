import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Gabal El Zeit | AI Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    h1, h2, h3 { color: #1E293B; font-family: 'Inter', sans-serif; }
    .stMetric label { font-size: 1.1rem !important; font-weight: 600 !important; color: #475569 !important; }
    .stMetric value { font-size: 2.5rem !important; font-weight: 800 !important; color: #0F172A !important; }
    
    .css-1d391kg { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "1. Baseline (Vanilla)": "model_1_vanilla_baseline.joblib",
        "2. Advanced (Vanilla)": "model_2_vanilla_advanced.joblib",
        "3. Ablation (Vanilla)": "model_3_vanilla_ablation.joblib",
        "4. Baseline (Optimized)": "model_4_opt_baseline.joblib",
        "5. Advanced (Optimized)": "model_5_opt_advanced.joblib",
        "6. Ablation (Optimized)": "model_6_opt_ablation.joblib"
    }
    
    for name, filename in model_files.items():
        if os.path.exists(filename):
            models[name] = joblib.load(filename)
        else:
            models[name] = None
    return models

models_dict = load_models()

col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>⚡</h1>", unsafe_allow_html=True)
with col_title:
    st.title("Gabal El Zeit: Predictive AI Telemetry")
    st.markdown("Real-time Wind Farm SCADA monitoring powered by Physics-Informed XGBoost Architecture.")
st.divider()

with st.sidebar:
    st.header("🧠 Engine Configuration")
    selected_model_name = st.selectbox(
        "Select Active ML Architecture:", 
        list(models_dict.keys()),
        index=4
    )
    
    active_model = models_dict.get(selected_model_name)
    
    is_advanced = "Advanced" in selected_model_name
    is_ablation = "Ablation" in selected_model_name
    needs_lags = is_advanced or is_ablation
    needs_current = not is_ablation

    st.markdown("---")
    st.header("⚙️ Core Parameters")
    theoretical_power = st.number_input("Theoretical Power Limit (kW)", value=2500.0, step=50.0)

    if needs_current:
        st.header("🌬️ Live Meteorology")
        wind_speed = st.slider("Current Wind Speed (m/s)", 0.0, 25.0, 12.5, 0.1)
        wind_dir = st.slider("Wind Direction (°)", 0.0, 360.0, 180.0, 1.0)
    else:
        wind_speed = 12.5
        wind_dir = 180.0
        st.info("👁️ **Ablation Mode Active:** Current anemometer readings are intentionally blinded from the AI.")

    if needs_lags:
        st.header("⏪ Historical Memory (Lags)")
        st.caption("AI requires past 60 mins of momentum data.")
        ws_10 = st.slider("Wind Speed - 10 min (m/s)", 0.0, 25.0, float(wind_speed), 0.1)
        ws_30 = st.slider("Wind Speed - 30 min (m/s)", 0.0, 25.0, max(0.0, float(wind_speed)-1.0), 0.1)
        ws_60 = st.slider("Wind Speed - 60 min (m/s)", 0.0, 25.0, max(0.0, float(wind_speed)-2.0), 0.1)
        
        ws_rolling = (wind_speed + ws_10 + ws_30 + ws_60) / 4.0
        dir_rolling = wind_dir 

    st.markdown("---")
    st.header("🕒 Environment & Market")
    col_m, col_h = st.columns(2)
    with col_m:
        month = st.selectbox("Month", range(1, 13), index=5)
    with col_h:
        hour = st.selectbox("Hour", range(0, 24), index=12)
        
    price_per_mwh = st.number_input("Energy Market Price ($/MWh)", value=55.0, step=1.0)

if active_model is not None:
    feature_dict = {
        'Theoretical_Power': float(theoretical_power),
        'Month': int(month),
        'Hour': int(hour)
    }
    
    if needs_current:
        feature_dict['Wind_Speed'] = float(wind_speed)
        feature_dict['Wind_Direction'] = float(wind_dir)
        
    if needs_lags:
        feature_dict['WS_10min_ago'] = float(ws_10)
        feature_dict['WS_30min_ago'] = float(ws_30)
        feature_dict['WS_60min_ago'] = float(ws_60)
        feature_dict['WS_1hr_Rolling_Avg'] = float(ws_rolling)
        feature_dict['Dir_1hr_Rolling_Avg'] = float(dir_rolling)

    input_df = pd.DataFrame([feature_dict])
    expected_cols = active_model.get_booster().feature_names
    
    try:
        input_df = input_df[expected_cols]
    except KeyError:
        st.error(f"🚨 Critical Data Mismatch. Expected: {expected_cols}")
        st.stop()

    prediction_kw = float(active_model.predict(input_df)[0])
    prediction_kw = max(0.0, prediction_kw)
    prediction_kw = min(theoretical_power, prediction_kw)
    
    efficiency = (prediction_kw / theoretical_power) * 100 if theoretical_power > 0 else 0.0

    megawatts = prediction_kw / 1000.0
    hourly_rev = megawatts * price_per_mwh
    daily_rev = hourly_rev * 24        
    annual_rev = daily_rev * 365       

    tab1, tab2, tab3 = st.tabs(["📊 Live Telemetry", "💰 Financial Projections", "🧠 Architecture Specs"])

    with tab1:
        st.markdown("### Real-Time AI Output")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Output (kW)", f"{prediction_kw:,.1f}", f"{selected_model_name.split(' ')[0]} Engine")
        m2.metric("Operating Efficiency", f"{efficiency:.1f}%", f"{theoretical_power} kW Limit", delta_color="off")
        
        if needs_current:
            m3.metric("Live Anemometer", f"{wind_speed} m/s", "Sensors Active")
        else:
            m3.metric("Live Anemometer", "BLINDED", "Ablation Mode", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = efficiency,
            title = {'text': "Turbine Efficiency Yield", 'font': {'size': 24}},
            number = {'suffix': "%", 'font': {'size': 50}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#3B82F6"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#FEE2E2"},
                    {'range': [40, 80], 'color': "#FEF3C7"},
                    {'range': [80, 100], 'color': "#D1FAE5"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95}
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.markdown("### Revenue Engine")
        st.caption(f"Calculated at current market rate of **${price_per_mwh}/MWh** assuming sustained weather conditions.")
        
        f1, f2, f3 = st.columns(3)
        f1.metric("Hourly Run-Rate", f"${hourly_rev:,.2f}")
        f2.metric("Projected Daily", f"${daily_rev:,.0f}")
        f3.metric("Projected Annual", f"${annual_rev:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        prices = np.linspace(20, 120, 20)
        revenues = megawatts * prices
        df_chart = pd.DataFrame({'Price per MWh ($)': prices, 'Hourly Revenue ($)': revenues})
        
        fig_area = px.area(df_chart, x='Price per MWh ($)', y='Hourly Revenue ($)', 
                           color_discrete_sequence=['#10B981'],
                           title="Revenue Scaling Curve")
        
        fig_area.add_vline(x=price_per_mwh, line_dash="dash", line_color="red", 
                           annotation_text=f"Current Price (${price_per_mwh})")
        
        st.plotly_chart(fig_area, use_container_width=True)

    with tab3:
        st.markdown("### Decision Engine Analysis")
        st.info(f"**Currently Active:** {selected_model_name}")
        
        with st.expander("🔍 Inspect Raw Input Matrix"):
            st.markdown("This is the exact feature vector passed to the XGBoost C++ engine:")
            st.dataframe(input_df, use_container_width=True)
        
        st.markdown("#### Pipeline Notes:")
        if is_ablation:
            st.warning("⚠️ **Ablation Notice:** This model is artificially blinded to current weather conditions to prove the mathematical strength of historical momentum features.")
        elif is_advanced:
            st.success("✅ **Advanced Memory Active:** This model utilizes 1-hour rolling averages and sequential 10/30/60 minute lags to calculate physical inertia.")
        else:
            st.info("ℹ️ **Baseline Mode:** This model evaluates weather purely instantaneously, without understanding momentum.")

else:
    st.error("🚨 Model files not found! Please upload the 6 `.joblib` files to the same directory as this script.")
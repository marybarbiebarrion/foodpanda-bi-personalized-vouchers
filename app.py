import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Foodpanda Incentive Optimization Engine", layout="centered")

# --- LOAD MODELS ---
# The @st.cache_resource decorator ensures models load only once for performance
@st.cache_resource
def load_models():
    kmeans = joblib.load('foodpanda_kmeans_model.pkl')
    scaler = joblib.load('foodpanda_scaler.pkl')
    return kmeans, scaler

loaded_kmeans, loaded_scaler = load_models()

# --- REALISTIC DEMO OVERRIDE MAPPING ---
realistic_rules = {
    "Burger": "McDonald's",
    "Chicken": "Jollibee",
    "Fast Food": "Burger King",
    "Italian": "Italianni's",
    "Pasta": "Mama Lou's",
    "Sandwich": "Subway",
    "Chinese": "Chowking",
    "Coffee": "Starbucks"
}
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --- UI FRONTEND ---
st.title("Foodpanda Incentive Optimization Engine")
st.markdown("Enter the customer's behavioral metrics below to prescribe the optimal retention strategy.")
st.divider()

# Input UI elements
st.subheader("Customer Profile Configuration")
col1, col2 = st.columns(2)

with col1:
    craving_choice = st.selectbox("Category Affinity:", options=list(realistic_rules.keys()))
    day_choice = st.selectbox("Preferred Delivery Day:", options=days_of_week)

with col2:
    spend_input = st.number_input("Average Spend per Order (PHP):", min_value=100.0, max_value=10000.0, value=800.0, step=50.0)
    order_freq = st.number_input("Monthly Order Frequency:", min_value=1, max_value=50, value=5, step=1)

st.divider()

# --- EXECUTE ENGINE BUTTON ---
# Using type="primary" gives the button a solid, corporate accent color
if st.button("Execute Prescriptive Engine", use_container_width=True, type="primary"):
    with st.spinner("Processing architectural logic..."):
        
        # Calculate derived metrics
        loyalty_estimate = order_freq * 15
        user_data = pd.DataFrame({'avg_order_value': [spend_input], 'loyalty_points': [loyalty_estimate]})
        scaled_user_data = loaded_scaler.transform(user_data)
        
        # Predict Tier
        predicted_cluster = loaded_kmeans.predict(scaled_user_data)[0]

        # Business Logic Mapping
        if predicted_cluster == 0:
            tier_profile = "Tier 0: Loyal Low-Spender"
            authorized_action = "Standard 10% Discount"
            alert_type = "info"
        elif predicted_cluster == 1:
            tier_profile = "Tier 1: At-Risk Low-Spender"
            authorized_action = "Aggressive 20% Discount (Maximum Budget)"
            alert_type = "error"
        else:
            tier_profile = "Tier 2: VIP High-Spender"
            authorized_action = "Free Delivery Only (Margin Protection)"
            alert_type = "success"

        realistic_voucher = realistic_rules[craving_choice]

        # --- DISPLAY RESULTS ---
        st.subheader("Prescriptive Output")
        
        # Display the classification tier
        if alert_type == "error":
            st.error(f"**Customer Classification:** {tier_profile}")
        elif alert_type == "success":
            st.success(f"**Customer Classification:** {tier_profile}")
        else:
            st.info(f"**Customer Classification:** {tier_profile}")

        st.markdown("### Recommended Business Action")
        
        # Using columns and st.code to create a clean, boxed dashboard look for the outputs
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("**Step 1: Margin Control**")
            st.code(authorized_action, language=None)
            
            st.markdown("**Step 2: Cross-Selling Assignment**")
            st.code(f"{realistic_voucher} Voucher", language=None)
            
        with res_col2:
            st.markdown("**Step 3: Deployment Schedule**")
            st.code(day_choice, language=None)
            
            st.markdown("**Estimated Loyalty Metric**")
            st.code(f"{loyalty_estimate} Points", language=None)
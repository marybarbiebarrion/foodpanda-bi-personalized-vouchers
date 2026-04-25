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

# =========================================================================
# NEW FEATURE: BATCH PROCESSING (CSV UPLOAD)
# =========================================================================

st.divider()
st.subheader("📂 Batch Processing Engine (CSV Upload)")
st.markdown("Upload a customer dataset to automatically generate retention tiers and cross-selling vouchers for thousands of users at once.")

# Drag and drop file uploader
uploaded_file = st.file_uploader("Upload your Customer Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df_batch = pd.read_csv(uploaded_file)
        
        # Check if the necessary mathematical columns exist
        if 'avg_order_value' in df_batch.columns and 'loyalty_points' in df_batch.columns:
            with st.spinner("Processing thousands of rows through the Incentive Optimization Engine..."):
                
                # 1. Extract and Scale inputs
                X_batch = df_batch[['avg_order_value', 'loyalty_points']]
                X_batch_scaled = loaded_scaler.transform(X_batch)
                
                # 2. Predict K-Means Tiers
                df_batch['Predicted_Cluster'] = loaded_kmeans.predict(X_batch_scaled)
                
                # 3. Map Tiers to Business Logic Profiles
                def get_tier_profile(cluster):
                    if cluster == 0: return "Tier 0: Loyal Low-Spender"
                    elif cluster == 1: return "Tier 1: At-Risk Low-Spender"
                    else: return "Tier 2: VIP High-Spender"

                def get_discount(cluster):
                    if cluster == 0: return "Standard 10% Discount"
                    elif cluster == 1: return "Aggressive 20% Discount"
                    else: return "Free Delivery Only"

                df_batch['Assigned_Tier'] = df_batch['Predicted_Cluster'].apply(get_tier_profile)
                df_batch['Prescribed_Discount'] = df_batch['Predicted_Cluster'].apply(get_discount)
                
                # 4. Map the Restaurant/Voucher based on their Dish or Category preference
                def get_voucher(row):
                    # Combine their dish and category preferences into a searchable text string
                    search_text = f"{row.get('dish_name', '')} {row.get('category', '')}".lower()
                    
                    # Search for our realistic rules in their historical preference
                    for rule_key, vendor in realistic_rules.items():
                        if rule_key.lower() in search_text:
                            return f"{vendor} Voucher"
                    
                    # If no specific match, default to a generic fallback based on their cluster
                    return "Foodpanda Fast-Food Voucher"
                
                df_batch['Prescribed_Cross_Sell_Voucher'] = df_batch.apply(get_voucher, axis=1)
                
                # Drop the raw cluster number to keep the business output clean
                df_batch = df_batch.drop(columns=['Predicted_Cluster'])
                
                st.success("✅ Batch processing completed successfully!")
                
                # Display a preview of the new table
                st.markdown("**Preview of the Optimized Dataset:**")
                # Show key columns alongside the newly generated predictions
                preview_columns = ['customer_id', 'avg_order_value', 'loyalty_points', 'Assigned_Tier', 'Prescribed_Discount', 'Prescribed_Cross_Sell_Voucher']
                # Try to show these columns, if customer_id doesn't exist, show all columns
                cols_to_show = [c for c in preview_columns if c in df_batch.columns]
                st.dataframe(df_batch[cols_to_show].head(10), use_container_width=True)
                
                # 5. Create a Download Button for the processed dataset
                csv_output = df_batch.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Processed Strategy Data (CSV)",
                    data=csv_output,
                    file_name="Foodpanda_Batch_Results.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.error("⚠️ Error: The uploaded CSV must contain the columns 'avg_order_value' and 'loyalty_points'.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
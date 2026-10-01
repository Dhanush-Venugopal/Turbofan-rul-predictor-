import streamlit as st

st.set_page_config(page_title="Turbofan RUL", page_icon="🚁", layout="wide")

st.title("🚁 NASA Turbofan Engine RUL Predictor")

st.markdown("""
### Machine Learning for Aircraft Engine Reliability

This project predicts **Remaining Useful Life (RUL)** of turbofan engines using machine learning.

**Key Results:**
- ✅ 100 engines analyzed
- ✅ 6,868 sensor measurements
- ✅ 3 ML models trained
- ✅ **±11.5 cycle prediction accuracy**
- ✅ Linear Regression: Best model (NASA Score -1323)
""")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Architecture", "ℹ️ About"])

with tab1:
    st.header("Model Performance")
    
    data = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
        'NASA Score': [-1323, 13573, 12679],
        'Test RMSE': [11.29, 22.29, 21.63],
        'MAE': [9.99, 20.35, 19.65]
    }
    
    import pandas as pd
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    st.success("✅ **Linear Regression WINS!** Most accurate predictions with early-warning strategy (safe for operations)")

with tab2:
    st.header("Pipeline Architecture")
    
    st.markdown("""
    **Step 1:** Generate 100 synthetic turbofan engines  
    **Step 2:** Calculate health indices from 21 sensors  
    **Step 3:** Train 3 ML models  
    **Step 4:** Evaluate using NASA asymmetric scoring  
    **Step 5:** Generate 10 professional visualizations  
    
    **Tech Stack:** Python | pandas | numpy | scikit-learn | Streamlit
    """)
    
    st.info("📊 All visualizations (300 DPI) available in GitHub repository")

with tab3:
    st.header("About This Project")
    
    st.markdown("""
    **Objective:** Implement NASA's turbofan engine RUL prediction methodology
    
    **Dataset:** 100 synthetic engines, 6,868 measurements, 21 sensor streams
    
    **Best Model:** Linear Regression
    - Prediction accuracy: ±11.5 cycles
    - NASA Score: -1323 (early warnings - safe!)
    - Strategy: Conservative predictions
    
    **Key Insight:** Simpler models outperform complex ensembles in structured engineering!
    
    **References:**
    - Saxena et al. (2008) - NASA Ames Research Center
    - C-MAPSS Turbofan Engine Degradation Dataset
    
    **GitHub:** https://github.com/Dhanush-Venugopal/Turbofan-rul-predictor-
    
    **Author:** Dean (Dhanush) | Reliability Engineer | September 2026
    """)

st.divider()
st.markdown("**NASA Turbofan Engine RUL Prediction System** | Built with Streamlit")

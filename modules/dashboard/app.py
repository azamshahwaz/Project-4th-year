import streamlit as st
import pandas as pd
import os

st.title(
    "Responsible AI Dashboard"
)

df = pd.read_csv(
    "outputs/final_processed_dataset.csv"
)

st.subheader(
    "Processed Dataset"
)

st.dataframe(df.head())

# =========================
# METRICS
# =========================

st.metric(
    "EDQS",
    "98.15"
)

st.metric(
    "ERI",
    "12.46"
)

st.metric(
    "RAI",
    "84.28"
)

# =========================
# SHAP
# =========================

image_path = "outputs/graphs/shap_summary.png"

if os.path.exists(image_path):

    st.image(image_path)

else:

    st.warning(
        "SHAP summary graph not found. Run main.py first."
    )
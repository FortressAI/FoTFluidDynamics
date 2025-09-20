import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("🏆 FoT Millennium Prize Solver - Test")
st.write("Testing basic Streamlit functionality...")

# Test basic functionality
if st.button("Test Button"):
    st.success("✅ Basic Streamlit working!")
    
# Test data loading
st.write("📊 Testing data capabilities...")
df = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [10, 11, 12, 13]
})
st.dataframe(df)

# Test plotting
fig = px.line(df, x='x', y='y', title='Test Plot')
st.plotly_chart(fig)

st.write("🎉 If you see this, basic functionality is working!")
st.write("The issue might be with complex imports in the main app.")

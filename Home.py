import streamlit as st

st.set_page_config(
    page_name="BondSuite",
    page_icon="📊",
    layout="wide"
)

st.title("BondSuite Analytics")
st.markdown("""
## Welcome to BondSuite Analytics

This application provides comprehensive tools for bond portfolio management and analysis.

### Available Tools:

#### 1. Portfolio Optimization
- Load and analyze bond universe
- Apply sophisticated filtering
- Optimize portfolio based on multiple constraints
- Visualize optimization results

#### 2. Portfolio Analysis
- Upload and analyze portfolio data
- Risk metrics and analysis
- Performance attribution
- Portfolio composition visualization

Please select a tool from the sidebar to get started.
""")

"""Main application entry point"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import logging
from dotenv import load_dotenv
import os
import sys
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent
sys.path.append(str(app_dir))

from app.data.models import Bond, PortfolioConstraints, CreditRating
from app.optimization.engine import PortfolioOptimizer
from app.ui.components import (
    render_constraints_form,
    display_optimization_results,
    render_main_constraints_form,
    render_optional_constraints
)
from app.ui.filter_components import render_filter_controls
from app.filters import FilterManager
from app.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging()

# Load environment variables
load_dotenv()

def load_bond_universe(uploaded_file: UploadedFile) -> list[Bond]:
    """Load bond universe from Excel/CSV file"""
    try:
        logger.info(f"Loading bond universe from file: {uploaded_file.name}")
        # Get file extension from the name
        file_extension = Path(uploaded_file.name).suffix.lower()

        # Read the file based on its extension
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            error_msg = f"Unsupported file format: {file_extension}"
            logger.error(error_msg)
            st.error(error_msg)
            return []

        logger.info(f"Successfully loaded {len(df)} rows from file")

        bonds = []
        for _, row in df.iterrows():
            try:
                # Parse credit rating using the new from_string method
                credit_rating = CreditRating.from_string(str(row['CreditRating']))

                # Create base bond object
                bond = Bond(
                    isin=str(row['ISIN']),  # Ensure ISIN is string
                    clean_price=float(row['CleanPrice']),
                    ytm=float(row['YTM']),
                    modified_duration=float(row['ModifiedDuration']),
                    maturity_date=pd.to_datetime(row['MaturityDate']).to_pydatetime(),
                    coupon_rate=float(row['CouponRate']),
                    coupon_frequency=int(row['CouponFrequency']),
                    credit_rating=credit_rating,
                    min_piece=float(row['MinPiece']),
                    increment_size=float(row['IncrementSize']),
                    currency=str(row['Currency']),
                    day_count_convention=str(row['DayCountConvention']),
                    issuer=str(row['Issuer'])
                )

                # Add new attributes if they exist in the file
                if 'Country' in df.columns:
                    setattr(bond, 'country', str(row['Country']))
                if 'Sector' in df.columns:
                    setattr(bond, 'sector', str(row['Sector']))
                if 'PaymentRank' in df.columns:
                    setattr(bond, 'payment_rank', str(row['PaymentRank']))

                bonds.append(bond)
            except Exception as e:
                error_msg = f"Error loading bond {row.get('ISIN', 'Unknown')}: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)

        logger.info(f"Successfully created {len(bonds)} bond objects")
        return bonds
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        logger.exception(error_msg)
        st.error(error_msg)
        return []

def main():
    """Main application entry point"""
    st.set_page_config(layout="wide")  # Set wide mode
    st.title("Bond Portfolio Optimizer")

    # Initialize session state
    if 'constraints' not in st.session_state:
        st.session_state.constraints = None
    if 'universe' not in st.session_state:
        st.session_state.universe = None
    if 'filtered_universe' not in st.session_state:
        st.session_state.filtered_universe = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None

    # Initialize filter manager
    filter_manager = FilterManager()

    # File uploader for bond universe
    st.header("Bond Universe")
    uploaded_file = st.file_uploader("Upload Bond Universe (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

    # Load sample universe if no file uploaded
    if uploaded_file is None:
        sample_universe_path = Path(__file__).parent.parent / "data" / "sample_universe_expanded.csv"
        if sample_universe_path.exists():
            logger.info("Loading sample universe")
            uploaded_file = sample_universe_path.open('rb')
            st.info("Using sample bond universe")

    # Load bond universe
    if uploaded_file:
        universe = load_bond_universe(uploaded_file)
        if universe:
            st.session_state.universe = universe
            st.success(f"Loaded {len(universe)} bonds")

            # Display universe summary with additional columns in expander
            with st.expander("View Bond Universe", expanded=False):
                df = pd.DataFrame([{
                    'ISIN': bond.isin,
                    'Price': f"{bond.clean_price:.2f}",
                    'YTM': f"{bond.ytm:.2%}",
                    'Duration': f"{bond.modified_duration:.2f}",
                    'Maturity': bond.maturity_date.strftime('%Y-%m-%d'),
                    'Rating': bond.credit_rating.display(),
                    'Issuer': bond.issuer,
                    'Country': getattr(bond, 'country', 'Unknown'),
                    'Sector': getattr(bond, 'sector', 'Unknown'),
                    'Payment Rank': getattr(bond, 'payment_rank', 'Unknown'),
                    'Min Piece': f"{bond.min_piece:,.2f}",
                    'Increment': f"{bond.increment_size:,.2f}"
                } for bond in universe])

                # Add summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    # Rating distribution
                    rating_dist = df['Rating'].value_counts()
                    st.subheader("Rating Distribution")
                    fig = go.Figure(data=[go.Bar(x=rating_dist.index, y=rating_dist.values)])
                    fig.update_layout(
                        xaxis_title="Rating",
                        yaxis_title="Count",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Sector distribution
                    sector_dist = df['Sector'].value_counts()
                    st.subheader("Sector Distribution")
                    fig = go.Figure(data=[go.Bar(
                        x=sector_dist.values,
                        y=sector_dist.index,
                        orientation='h'
                    )])
                    fig.update_layout(
                        xaxis_title="Count",
                        yaxis_title="Sector",
                        showlegend=False,
                        height=400,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Payment Rank distribution
                    rank_dist = df['Payment Rank'].value_counts()
                    st.subheader("Payment Rank Distribution")
                    fig = px.pie(
                        values=rank_dist.values,
                        names=rank_dist.index,
                        title='Payment Rank Breakdown'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Country distribution
                    country_dist = df['Country'].value_counts()
                    st.subheader("Country Distribution")
                    fig = px.pie(
                        values=country_dist.values,
                        names=country_dist.index,
                        title='Country Breakdown'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Sort by YTM descending and display table
                df = df.sort_values('YTM', ascending=False)
                st.dataframe(df, hide_index=True)


            # Apply filters
            filtered_universe = render_filter_controls(universe, filter_manager)
            st.session_state.filtered_universe = filtered_universe

    # Get constraints and check if optimization should run
    constraints, run_optimization = render_main_constraints_form(st.session_state.universe or [])

    # Always render optional constraints
    render_optional_constraints(st.session_state.universe or [])

    # Display optimization results below the constraints form
    results_container = st.container()

    with results_container:
        if constraints and run_optimization:
            st.markdown("---")
            st.header("Optimization Results")

            try:
                # Use filtered universe if available, otherwise use full universe
                optimization_universe = st.session_state.filtered_universe or st.session_state.universe

                if optimization_universe:
                    optimizer = PortfolioOptimizer(optimization_universe, constraints)
                    result = optimizer.optimize()
                    logger.info(f"Optimization completed with status: {result.status}")

                    if result.success:
                        if result.constraints_satisfied:
                            st.success(f"Optimization completed successfully in {result.solve_time:.2f} seconds")
                        display_optimization_results(result, optimization_universe, constraints.total_size)

                        # Add download buttons for results
                        portfolio_df = pd.DataFrame([{
                            'ISIN': isin,
                            'Weight': weight,
                            'Notional': weight * constraints.total_size,
                            'Bond': next(b for b in optimization_universe if b.isin == isin)
                        } for isin, weight in result.portfolio.items()])

                        # Add bond details
                        portfolio_df['Issuer'] = portfolio_df['Bond'].apply(lambda x: x.issuer)
                        portfolio_df['Rating'] = portfolio_df['Bond'].apply(lambda x: x.credit_rating.display())
                        portfolio_df['YTM'] = portfolio_df['Bond'].apply(lambda x: x.ytm)
                        portfolio_df['Duration'] = portfolio_df['Bond'].apply(lambda x: x.modified_duration)
                        portfolio_df['Min Piece'] = portfolio_df['Bond'].apply(lambda x: x.min_piece)
                        portfolio_df['Increment'] = portfolio_df['Bond'].apply(lambda x: x.increment_size)

                        # Drop Bond column and sort by Weight
                        portfolio_df = portfolio_df.drop('Bond', axis=1).sort_values('Weight', ascending=False)

                        # Format columns
                        portfolio_df['Weight'] = portfolio_df['Weight'].apply(lambda x: f"{x:.2%}")
                        portfolio_df['Notional'] = portfolio_df['Notional'].apply(lambda x: f"{x:,.2f}")
                        portfolio_df['YTM'] = portfolio_df['YTM'].apply(lambda x: f"{x:.2%}")
                        portfolio_df['Duration'] = portfolio_df['Duration'].apply(lambda x: f"{x:.2f}")
                        portfolio_df['Min Piece'] = portfolio_df['Min Piece'].apply(lambda x: f"{x:,.2f}")
                        portfolio_df['Increment'] = portfolio_df['Increment'].apply(lambda x: f"{x:,.2f}")

                    else:
                        st.error(f"Optimization failed: {result.status}")
                        if result.constraint_violations:
                            st.write("Constraint violations:")
                            for violation in result.constraint_violations:
                                st.write(f"- {violation}")
                else:
                    st.error("Please load a bond universe before optimizing")

            except Exception as e:
                error_msg = f"Optimization error: {str(e)}"
                logger.exception(error_msg)
                st.error(error_msg)

if __name__ == "__main__":
    main()

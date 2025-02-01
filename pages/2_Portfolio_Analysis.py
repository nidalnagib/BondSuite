"""Portfolio Analysis Application"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from app.data.models import Bond, CreditRating, RatingGrade
from app.ui.components import display_optimization_results
from app.ui.presentation import PresentationTheme, generate_portfolio_presentation

# Set up page config
st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_bond_objects(df: pd.DataFrame) -> List[Bond]:
    """Convert DataFrame rows to Bond objects"""
    bonds = []
    for _, row in df.iterrows():
        try:
            credit_rating = CreditRating.from_string(str(row['Credit Rating']))
            
            bond = Bond(
                isin=str(row['ISIN']),
                clean_price=100.0,  # Placeholder as we don't have this in analysis
                ytm=float(row['YTM']),
                modified_duration=float(row['Modified Duration']),
                maturity_date=pd.to_datetime(row['Maturity Date']).to_pydatetime(),
                coupon_rate=float(row['Coupon Rate']),
                coupon_frequency=2,  # Assuming semi-annual
                credit_rating=credit_rating,
                min_piece=10000,  # Default minimum piece
                increment_size=1000,  # Default increment
                currency=str(row['Currency']),
                day_count_convention='30/360',  # Default convention
                issuer=str(row['Issuer']),
                sector=str(row['Sector']) if 'Sector' in row else None,
                country=str(row['Country']) if 'Country' in row else None,
                payment_rank=str(row['Payment Rank']) if 'Payment Rank' in row else None
            )
            bonds.append(bond)
        except Exception as e:
            logger.error(f"Error converting bond {row.get('ISIN', 'Unknown')}: {str(e)}")
    return bonds

def process_portfolio_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Bond]]:
    """Process portfolio data and calculate weights"""
    # Calculate weights from market values
    df['Weight'] = df['Market Value'] / df['Market Value'].sum()
    
    # Convert data types
    if 'Maturity Date' in df.columns:
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
    
    # Convert bonds
    bonds = convert_to_bond_objects(df)
    
    return df, bonds

def load_and_process_portfolio(uploaded_file) -> Tuple[pd.DataFrame, List[Bond]]:
    """Load and process portfolio data from uploaded file"""
    try:
        # Read the file based on its type
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        logger.info(f"Successfully loaded portfolio from {uploaded_file.name}")
        return process_portfolio_data(df)
        
    except Exception as e:
        logger.error(f"Error loading portfolio: {str(e)}")
        raise

def calculate_portfolio_metrics(df: pd.DataFrame, bonds: List[Bond]) -> Dict[str, float]:
    """Calculate key portfolio metrics"""
    metrics = {}
    
    if 'Market Value' in df.columns:
        total_value = df['Market Value'].sum()
        metrics['Total Value'] = total_value
        metrics['Number of Holdings'] = len(df)
        
        # Calculate weighted average metrics
        metrics['Modified Duration'] = np.average([b.modified_duration for b in bonds], weights=df['Weight'].values)
        metrics['YTM'] = np.average([b.ytm for b in bonds], weights=df['Weight'].values)
        
        # Investment Grade calculation
        ig_mask = [bond.credit_rating.is_investment_grade() for bond in bonds]
        metrics['Investment Grade %'] = np.sum(df['Weight'].values[ig_mask]) * 100
        
        # Rating calculation (numeric value)
        rating_values = [bond.credit_rating.value for bond in bonds]
        metrics['Average Rating Value'] = np.average(rating_values, weights=df['Weight'].values)
        
        # Convert average rating value back to nearest rating
        metrics['Average Rating'] = CreditRating.from_score(metrics['Average Rating Value']).display()
        
        # Risk metrics
        if 'Modified Duration' in df.columns:
            metrics['Portfolio Duration'] = (df['Modified Duration'] * df['Weight']).sum()
            metrics['Duration Risk (1% yield change)'] = metrics['Portfolio Duration'] * total_value * 0.01
        
        if 'YTM' in df.columns:
            metrics['Portfolio YTM'] = (df['YTM'] * df['Weight']).sum()
            
        # Credit metrics
        if bonds:
            rating_scores = [bond.credit_rating.value * weight 
                           for bond, weight in zip(bonds, df['Weight'])]
            avg_rating_score = sum(rating_scores)
            metrics['Average Rating'] = CreditRating.from_score(avg_rating_score).display()
            
            # Calculate rating distribution
            ig_weight = sum(df['Weight'][i] for i, bond in enumerate(bonds) 
                          if bond.credit_rating.is_investment_grade())
            metrics['Investment Grade %'] = ig_weight * 100
            
        # Concentration metrics
        metrics['Top 5 Holdings %'] = df.nlargest(5, 'Weight')['Weight'].sum() * 100
        
        # Issuer concentration
        issuer_conc = df.groupby('Issuer')['Weight'].sum()
        metrics['Largest Issuer %'] = issuer_conc.max() * 100
        
        # Sector metrics if available
        if 'Sector' in df.columns:
            sector_conc = df.groupby('Sector')['Weight'].sum()
            metrics['Largest Sector %'] = sector_conc.max() * 100
            
    return metrics

def plot_risk_metrics(df: pd.DataFrame, bonds: List[Bond]):
    """Enhanced risk metrics visualization"""
    if 'Modified Duration' in df.columns and 'YTM' in df.columns:
                
        # Create a copy of the dataframe with YTM as percentage
        plot_df = df.copy()
        plot_df['YTM'] = plot_df['YTM'] * 100  # Convert to percentage
        
        # Create scatter plot with size based on market value
        fig = px.scatter(
            plot_df,
            x='Modified Duration',
            y='YTM',
            size='Market Value',
            color='Credit Rating',
            hover_data=['ISIN', 'Issuer', 'Sector'],
            title="Risk-Return Profile",
            labels={
                'Modified Duration': 'Modified Duration (years)',
                'YTM': 'Yield to Maturity (%)',
                'Market Value': 'Position Size'
            }
        )

        # Add portfolio average point
        avg_dur = (plot_df['Modified Duration'] * plot_df['Weight']).sum()
        avg_ytm = (plot_df['YTM'] * plot_df['Weight']).sum()
        fig.add_trace(
            go.Scatter(
                x=[avg_dur],
                y=[avg_ytm],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='yellow',
                    line=dict(color='black', width=2)
                ),
                name='Portfolio Average'
            )
        )
        
        fig.update_layout(
            showlegend=True,
            height=600,
            yaxis=dict(
                tickformat='.2f',  # Show 2 decimal places
                ticksuffix='%'     # Add % symbol to ticks
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Add duration distribution
        st.subheader("Duration Distribution")
        
        # Add bin customization controls
        col1, _ = st.columns(2)
        with col1:
            bin_size = st.slider(
                "Bin Size (years)",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                help="Size of each duration bucket in years"
            )
        
        # Calculate bins from 0 to ceiling of max duration
        max_duration = df['Modified Duration'].max()
        ceiling_duration = math.ceil(max_duration)
        bins = np.arange(0, ceiling_duration + bin_size, bin_size)  # Add bin_size to include last value
        
        # Create bin labels
        labels = [f"[{bins[i]:.1f}-{bins[i+1]:.1f}[" for i in range(len(bins)-1)]
        
        # Create duration bins and calculate weights
        duration_bins = pd.cut(
            df['Modified Duration'],
            bins=bins,
            labels=labels,
            right=False  # Use [a,b[ intervals
        )
        weights_by_bin = df.groupby(duration_bins)['Weight'].sum() * 100
        
        # Create bar chart
        duration_dist = go.Figure()
        duration_dist.add_trace(go.Bar(
            x=labels,
            y=weights_by_bin.values,
            text=[f'{w:.1f}%' for w in weights_by_bin.values],
            textposition='auto',
            hovertemplate='Duration: %{x}<br>Weight: %{y:.1f}%<extra></extra>'
        ))
        
        duration_dist.update_layout(
            title="Duration Distribution",
            xaxis_title="Duration Bucket (years)",
            yaxis_title="Weight (%)",
            showlegend=False,
            height=400,
            bargap=0.1,
            uniformtext=dict(
                mode='hide',  # Hide labels if they don't fit
                minsize=8    # Minimum text size
            ),
            yaxis=dict(
                tickformat='.1f',  # Format y-axis as percentage
                ticksuffix='%'
            )
        )
        
        st.plotly_chart(duration_dist, use_container_width=True)
        
        # Duration buckets table (existing functionality)
        duration_buckets = pd.cut(
            df['Modified Duration'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['0-2y', '2-5y', '5-10y', '10y+']
        )
        bucket_weights = df.groupby(duration_buckets)['Weight'].sum() * 100
        
        st.markdown("#### Duration Breakdown")
        bucket_df = pd.DataFrame({
            'Bucket': bucket_weights.index,
            'Weight (%)': bucket_weights.values
        })
        st.dataframe(
            bucket_df.style.format({'Weight (%)': '{:.1f}%'}),
            hide_index=True
        )

def plot_credit_analysis(df: pd.DataFrame, bonds: List[Bond]):
    """Enhanced credit analysis visualization"""
    cols = st.columns(2)
    
    with cols[0]:
        # Get ordered ratings and create distribution DataFrame
        ordered_ratings = CreditRating.get_ordered_ratings()
        
        # Ensure weights are properly normalized
        weights = df['Weight'] / df['Weight'].sum()  # Normalize weights to sum to 1
        
        rating_dist = pd.DataFrame([
            {'Rating': bond.credit_rating.display(), 
             'Weight': weight * 100,  # Convert to percentage
             'Grade': bond.rating_grade.value,
             'Order': bond.credit_rating.value}  # Add order for sorting
            for bond, weight in zip(bonds, weights)
        ])
        
        # Group by Rating and Grade, maintaining the order
        rating_dist = rating_dist.groupby(['Rating', 'Grade', 'Order'], as_index=False)['Weight'].sum()
        rating_dist = rating_dist.sort_values('Order')
        
        # Create figure with both traces
        fig = go.Figure()
        
        # Rating view
        fig.add_trace(go.Bar(
            x=rating_dist['Rating'],
            y=rating_dist['Weight'],
            name='By Rating',
            visible=True,
            text=rating_dist['Weight'].round(1).astype(str) + '%',
            textposition='auto',
        ))
        
        # Grade view
        grade_dist = rating_dist.groupby('Grade')['Weight'].sum().reset_index()
        
        fig.add_trace(go.Bar(
            x=grade_dist['Grade'],
            y=grade_dist['Weight'],
            name='By Grade',
            visible=False,
            text=grade_dist['Weight'].round(1).astype(str) + '%',
            textposition='auto',
        ))
        
        # Update layout with buttons
        fig.update_layout(
            height=400,
            title='Rating Breakdown',
            xaxis_title='Rating',
            yaxis_title='Weight (%)',
            showlegend=False,
            updatemenus=[{
                'buttons': [
                    {'label': 'By Rating', 'method': 'update', 'args': [{'visible': [True, False]}]},
                    {'label': 'By Grade', 'method': 'update', 'args': [{'visible': [False, True]}]}
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
            }]
        )
        
        # Ensure correct rating order for the rating view
        fig.update_xaxes(categoryorder='array', 
                        categoryarray=[r.display() for r in ordered_ratings])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        # Issuer concentration
        issuer_conc = df.groupby('Issuer')['Weight'].sum().sort_values(ascending=True) * 100
        fig = go.Figure(data=[
            go.Bar(
                x=issuer_conc.values,
                y=issuer_conc.index,
                orientation='h',
                text=[f'{x:.1f}%' for x in issuer_conc.values],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Issuer Concentration",
            xaxis_title="% of Portfolio",
            yaxis_title="Issuer",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_composition(df: pd.DataFrame):
    """Enhanced portfolio composition analysis"""
    cols = st.columns(3)  # Changed to 3 columns for Payment Rank
    
    with cols[0]:
        # Sector composition as bar chart
        if 'Sector' in df.columns:
            sector_comp = df.groupby('Sector')['Weight'].sum() * 100
            sector_comp = sector_comp.sort_values(ascending=True)  # Sort for better visualization
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sector_comp.values,
                    y=sector_comp.index,
                    orientation='h',
                    text=[f'{x:.1f}%' for x in sector_comp.values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sector Composition",
                xaxis_title="Weight (%)",
                yaxis_title="Sector",
                height=400,
                xaxis=dict(ticksuffix='%'),
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        # Geographic composition as bar chart
        if 'Country' in df.columns:
            country_comp = df.groupby('Country')['Weight'].sum() * 100
            country_comp = country_comp.sort_values(ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=country_comp.values,
                    y=country_comp.index,
                    orientation='h',
                    text=[f'{x:.1f}%' for x in country_comp.values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Geographic Composition",
                xaxis_title="Weight (%)",
                yaxis_title="Country",
                height=400,
                xaxis=dict(ticksuffix='%'),
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with cols[2]:
        # Payment Rank composition as bar chart
        if 'Payment Rank' in df.columns:
            rank_comp = df.groupby('Payment Rank')['Weight'].sum() * 100
            rank_comp = rank_comp.sort_values(ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=rank_comp.values,
                    y=rank_comp.index,
                    orientation='h',
                    text=[f'{x:.1f}%' for x in rank_comp.values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Payment Rank Composition",
                xaxis_title="Weight (%)",
                yaxis_title="Payment Rank",
                height=400,
                xaxis=dict(ticksuffix='%'),
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function for the Portfolio Analysis application"""
    st.title("Portfolio Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your portfolio file (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing your portfolio data"
    )
    
    # Sample portfolio option
    use_sample = st.checkbox("Use sample portfolio", value=not bool(uploaded_file))
    
    try:
        if use_sample:
            # Load sample portfolio
            sample_path = Path(__file__).parent.parent / 'data' / 'sample_portfolio.csv'
            df = pd.read_csv(sample_path)
            df, bonds = process_portfolio_data(df)
        elif uploaded_file is not None:
            # Load and process uploaded portfolio
            df, bonds = load_and_process_portfolio(uploaded_file)
        else:
            st.info("Please upload a portfolio file or use the sample portfolio")
            return

        # Create tabs for different analyses
        tabs = st.tabs(["Overview", "Risk Analysis", "Credit Analysis", "Composition", "Spare Sheet"])

        with tabs[0]:
            st.header("Portfolio Overview")
            metrics = calculate_portfolio_metrics(df, bonds)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Value", f"${metrics['Total Value']:,.2f}")
                st.metric("Number of Holdings", metrics['Number of Holdings'])
                st.metric("Portfolio YTM", f"{metrics['YTM']:.2%}")
            
            with col2:
                st.metric("Portfolio Duration", f"{metrics['Modified Duration']:.2f}")
                st.metric("Duration Risk", f"${metrics['Duration Risk (1% yield change)']:,.2f}")
                st.metric("Average Rating", metrics['Average Rating'])
            
            with col3:
                st.metric("Investment Grade %", f"{metrics['Investment Grade %']:.1f}%")
                st.metric("Largest Issuer %", f"{metrics['Largest Issuer %']:.1f}%")
                if 'Largest Sector %' in metrics:
                    st.metric("Largest Sector %", f"{metrics['Largest Sector %']:.1f}%")

            st.header("Portfolio Holdings")
            # Enhanced holdings view with sorting and filtering
            holdings_df = pd.DataFrame([{
                'ISIN': bond.isin,
                'Issuer': bond.issuer,
                'Sector': bond.sector,
                'Country': bond.country,
                'Rating': bond.credit_rating.display(),
                'Market Value': df.loc[i, 'Market Value'],
                'Weight': df.loc[i, 'Weight'],
                'YTM': bond.ytm,
                'Modified Duration': bond.modified_duration,
                'Maturity Date': bond.maturity_date.strftime('%Y-%m-%d')
            } for i, bond in enumerate(bonds)])
            
            # Format columns
            holdings_df['Market Value'] = holdings_df['Market Value'].map('${:,.2f}'.format)
            holdings_df['Weight'] = holdings_df['Weight'].map('{:.2%}'.format)
            holdings_df['YTM'] = holdings_df['YTM'].map('{:.2%}'.format)
            holdings_df['Modified Duration'] = holdings_df['Modified Duration'].map('{:.2f}'.format)
            
            st.dataframe(
                holdings_df,
                hide_index=True,
                column_config={
                    'ISIN': st.column_config.TextColumn('ISIN', width='medium'),
                    'Market Value': st.column_config.TextColumn('Market Value', width='medium'),
                    'Weight': st.column_config.TextColumn('Weight', width='small'),
                    'Rating': st.column_config.TextColumn('Rating', width='small'),
                }
            )

        with tabs[1]:
            st.header("Risk Analysis")
            plot_risk_metrics(df, bonds)

        with tabs[2]:
            st.header("Credit Analysis")
            plot_credit_analysis(df, bonds)

        with tabs[3]:
            st.header("Portfolio Composition")
            plot_composition(df)

        with tabs[4]:
            st.write("TBD")
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

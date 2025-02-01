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

    # Calculate required metrics
    total_mv = df['Market Value'].sum()
    df['Weight'] = df['Market Value'] / total_mv
    df['Contribution to Duration'] = df['Weight'] * df['Modified Duration']
    df['Contribution to Yield'] = df['Weight'] * df['YTM']

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
            hover_data=['ISIN', 'Security Name', 'Sector'],
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
                ticksuffix='%'  # Add % symbol to ticks
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

        # Use the new plot_duration_distribution function
        duration_dist = plot_duration_distribution(df, bin_size)
        st.plotly_chart(duration_dist, use_container_width=True)

        # Duration buckets table
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


def plot_duration_distribution(df: pd.DataFrame, bin_size: float = 1.0):
    """Plot duration distribution with toggles for weight and duration contribution"""

    # Create duration bins
    max_duration = math.ceil(df['Modified Duration'].max())
    bins = list(np.arange(0, max_duration + bin_size, bin_size))
    bin_labels = [f"{b:.1f}-{b + bin_size:.1f}" for b in bins[:-1]]

    # Calculate distributions
    weight_dist = pd.cut(df['Modified Duration'], bins=bins, right=False).value_counts(
        normalize=True).sort_index() * 100
    contrib_dist = pd.DataFrame({
        'Modified Duration': df['Modified Duration'],
        'Contribution': df['Contribution to Duration']
    }).groupby(pd.cut(df['Modified Duration'], bins=bins, right=False))['Contribution'].sum()
    # contrib_dist = (contrib_dist / contrib_dist.sum() * 100).sort_index()

    # Create figure with both traces
    fig = go.Figure()

    # Add weight-based distribution
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=weight_dist.values,
        name="By Weight",
        text=[f"{v:.1f}%" for v in weight_dist.values],
        textposition='auto',
        visible=True,
        showlegend=True
    ))

    # Add contribution-based distribution
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=contrib_dist.values,
        name="By Duration Contribution",
        text=[f"{v:.1f}" for v in contrib_dist.values],
        textposition='auto',
        visible=False,
        showlegend=True
    ))

    # Update layout with toggle buttons
    fig.update_layout(
        height=400,
        title='Duration Distribution',
        xaxis_title='Modified Duration (years)',
        yaxis_title='Distribution (%)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        updatemenus=[{
            'buttons': [
                {
                    'label': 'By Weight',
                    'method': 'update',
                    'args': [
                        {'visible': [True, False]},
                        {
                            'yaxis.title.text': 'Weight Distribution (%)',
                            'showlegend': True
                        }
                    ]
                },
                {
                    'label': 'By Duration Contribution',
                    'method': 'update',
                    'args': [
                        {'visible': [False, True]},
                        {
                            'yaxis.title.text': 'Duration Contribution (Years)',
                            'showlegend': True
                        }
                    ]
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15,
        }],
        yaxis=dict(ticksuffix='%'),
        margin=dict(t=80, b=20)
    )

    return fig


def plot_credit_analysis(df: pd.DataFrame, bonds: List[Bond]):
    """Enhanced credit analysis visualization"""

    # Add Rating Grade based on Credit Rating
    df['Rating Grade'] = [bond.rating_grade.value for bond in bonds]

    cols = st.columns(2)

    with cols[0]:
        # Rating distribution with three traces (weight, duration contrib, yield contrib)
        rating_weight = df.groupby('Credit Rating')['Weight'].sum() * 100
        rating_duration = df.groupby('Credit Rating')['Contribution to Duration'].sum()
        rating_yield = df.groupby('Credit Rating')['Contribution to Yield'].sum() * 100

        fig = go.Figure()

        # Add rating traces
        for values, name in [(rating_weight, 'Weight'),
                             (rating_duration, 'Duration Contribution'),
                             (rating_yield, 'Yield Contribution')]:
            fig.add_trace(go.Bar(
                x=values.index,
                y=values.values,
                name=name,
                text=[f'{v:.1f}' for v in values.values],
                textposition='auto',
                visible=(name == 'Weight')
            ))

        # Add grade traces
        grade_weight = df.groupby('Rating Grade')['Weight'].sum() * 100
        grade_duration = df.groupby('Rating Grade')['Contribution to Duration'].sum()
        grade_yield = df.groupby('Rating Grade')['Contribution to Yield'].sum()

        for values, name in [(grade_weight, 'Weight'),
                             (grade_duration, 'Duration Contribution'),
                             (grade_yield, 'Yield Contribution')]:
            fig.add_trace(go.Bar(
                x=values.index,
                y=values.values,
                name=name,
                text=[f'{v:.1f}' for v in values.values],
                textposition='auto',
                visible=False
            ))

        # Update layout with buttons for both rating/grade and weight/contribution toggles
        fig.update_layout(
            height=400,
            title='Rating Breakdown',
            xaxis_title='Rating',
            yaxis_title=r'Distribution (% or Yrs)',
            showlegend=False,
            updatemenus=[
                # Rating/Grade Toggle
                {
                    'buttons': [
                        {
                            'label': 'By Rating',
                            'method': 'update',
                            'args': [
                                {'visible': [True, False, False, False, False, False]},
                                {
                                    'xaxis.title.text': 'Rating',
                                    'updatemenus[1].buttons[0].args[0].visible': [True, False, False, False, False,
                                                                                  False],
                                    'updatemenus[1].buttons[1].args[0].visible': [False, True, False, False, False,
                                                                                  False],
                                    'updatemenus[1].buttons[2].args[0].visible': [False, False, True, False, False,
                                                                                  False]
                                }
                            ]
                        },
                        {
                            'label': 'By Grade',
                            'method': 'update',
                            'args': [
                                {'visible': [False, False, False, True, False, False]},
                                {
                                    'xaxis.title.text': 'Rating Grade',
                                    'updatemenus[1].buttons[0].args[0].visible': [False, False, False, True, False,
                                                                                  False],
                                    'updatemenus[1].buttons[1].args[0].visible': [False, False, False, False, True,
                                                                                  False],
                                    'updatemenus[1].buttons[2].args[0].visible': [False, False, False, False, False,
                                                                                  True]
                                }
                            ]
                        }
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'y': 1.15,
                },
                # Metric Toggle
                {
                    'buttons': [
                        {
                            'label': 'Weight',
                            'method': 'update',
                            'args': [
                                {'visible': [True, False, False, False, False, False]},
                                {'yaxis.title.text': 'Weight (%)'}
                            ]
                        },
                        {
                            'label': 'Duration Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, True, False, False, False, False]},
                                {'yaxis.title.text': 'Duration Contribution (Years)'}
                            ]
                        },
                        {
                            'label': 'Yield Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, False, True, False, False, False]},
                                {'yaxis.title.text': 'Yield Contribution (%)'}
                            ]
                        }
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.3,
                    'y': 1.15,
                }
            ],
            yaxis=dict(ticksuffix='%'),
            margin=dict(t=80)
        )
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

    st.subheader("Portfolio Breakdowns")
    cols = st.columns(3)

    with cols[0]:
        # Sector composition as bar chart with toggles
        if 'Sector' in df.columns:
            sector_weight = df.groupby('Sector')['Weight'].sum() * 100
            sector_duration = df.groupby('Sector')['Contribution to Duration'].sum()
            sector_yield = df.groupby('Sector')['Contribution to Yield'].sum() * 100

            # Sort all series the same way
            sector_weight = sector_weight.sort_values(ascending=True)
            sector_duration = sector_duration.reindex(sector_weight.index)
            sector_yield = sector_yield.reindex(sector_weight.index)

            fig = go.Figure()

            # Add all three traces
            fig.add_trace(go.Bar(
                x=sector_weight.values,
                y=sector_weight.index,
                orientation='h',
                name='Weight',
                text=[f'{x:.1f}%' for x in sector_weight.values],
                textposition='auto',
                visible=True
            ))

            fig.add_trace(go.Bar(
                x=sector_duration.values,
                y=sector_duration.index,
                orientation='h',
                name='Duration Contribution',
                text=[f'{x:.2f}' for x in sector_duration.values],
                textposition='auto',
                visible=False
            ))

            fig.add_trace(go.Bar(
                x=sector_yield.values,
                y=sector_yield.index,
                orientation='h',
                name='Yield Contribution',
                text=[f'{x:.2f}' for x in sector_yield.values],
                textposition='auto',
                visible=False
            ))

            fig.update_layout(
                title="Sector Composition",
                xaxis_title="Distribution",
                yaxis_title="Sector",
                height=400,
                margin=dict(l=20, r=20, t=80, b=20),
                updatemenus=[{
                    'buttons': [
                        {
                            'label': 'Weight',
                            'method': 'update',
                            'args': [
                                {'visible': [True, False, False]},
                                {
                                    'xaxis.title.text': 'Weight (%)',
                                    'xaxis.ticksuffix': '%'
                                }
                            ]
                        },
                        {
                            'label': 'Duration Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, True, False]},
                                {
                                    'xaxis.title.text': 'Duration Contribution (Years)',
                                    'xaxis.ticksuffix': ''
                                }
                            ]
                        },
                        {
                            'label': 'Yield Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, False, True]},
                                {
                                    'xaxis.title.text': 'Yield Contribution (%)',
                                    'xaxis.ticksuffix': '%'
                                }
                            ]
                        }
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'y': 1.15,
                }]
            )
            st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        # Geographic composition as bar chart with toggles
        if 'Country' in df.columns:
            country_weight = df.groupby('Country')['Weight'].sum() * 100
            country_duration = df.groupby('Country')['Contribution to Duration'].sum()
            country_yield = df.groupby('Country')['Contribution to Yield'].sum() * 100

            # Sort all series the same way
            country_weight = country_weight.sort_values(ascending=True)
            country_duration = country_duration.reindex(country_weight.index)
            country_yield = country_yield.reindex(country_weight.index)

            fig = go.Figure()

            # Add all three traces
            fig.add_trace(go.Bar(
                x=country_weight.values,
                y=country_weight.index,
                orientation='h',
                name='Weight',
                text=[f'{x:.1f}%' for x in country_weight.values],
                textposition='auto',
                visible=True
            ))

            fig.add_trace(go.Bar(
                x=country_duration.values,
                y=country_duration.index,
                orientation='h',
                name='Duration Contribution',
                text=[f'{x:.2f}' for x in country_duration.values],
                textposition='auto',
                visible=False
            ))

            fig.add_trace(go.Bar(
                x=country_yield.values,
                y=country_yield.index,
                orientation='h',
                name='Yield Contribution',
                text=[f'{x:.2f}' for x in country_yield.values],
                textposition='auto',
                visible=False
            ))

            fig.update_layout(
                title="Geographic Composition",
                xaxis_title="Distribution",
                yaxis_title="Country",
                height=400,
                margin=dict(l=20, r=20, t=80, b=20),
                updatemenus=[{
                    'buttons': [
                        {
                            'label': 'Weight',
                            'method': 'update',
                            'args': [
                                {'visible': [True, False, False]},
                                {
                                    'xaxis.title.text': 'Weight (%)',
                                    'xaxis.ticksuffix': '%'
                                }
                            ]
                        },
                        {
                            'label': 'Duration Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, True, False]},
                                {
                                    'xaxis.title.text': 'Duration Contribution (Years)',
                                    'xaxis.ticksuffix': ''
                                }
                            ]
                        },
                        {
                            'label': 'Yield Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, False, True]},
                                {
                                    'xaxis.title.text': 'Yield Contribution (%)',
                                    'xaxis.ticksuffix': '%'
                                }
                            ]
                        }
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'y': 1.15,
                }]
            )
            st.plotly_chart(fig, use_container_width=True)

    with cols[2]:
        # Payment Rank composition as bar chart with toggles
        if 'Payment Rank' in df.columns:
            rank_weight = df.groupby('Payment Rank')['Weight'].sum() * 100
            rank_duration = df.groupby('Payment Rank')['Contribution to Duration'].sum()
            rank_yield = df.groupby('Payment Rank')['Contribution to Yield'].sum() * 100

            # Sort all series the same way
            rank_weight = rank_weight.sort_values(ascending=True)
            rank_duration = rank_duration.reindex(rank_weight.index)
            rank_yield = rank_yield.reindex(rank_weight.index)

            fig = go.Figure()

            # Add all three traces
            fig.add_trace(go.Bar(
                x=rank_weight.values,
                y=rank_weight.index,
                orientation='h',
                name='Weight',
                text=[f'{x:.1f}%' for x in rank_weight.values],
                textposition='auto',
                visible=True
            ))

            fig.add_trace(go.Bar(
                x=rank_duration.values,
                y=rank_duration.index,
                orientation='h',
                name='Duration Contribution',
                text=[f'{x:.2f}' for x in rank_duration.values],
                textposition='auto',
                visible=False
            ))

            fig.add_trace(go.Bar(
                x=rank_yield.values,
                y=rank_yield.index,
                orientation='h',
                name='Yield Contribution',
                text=[f'{x:.2f}' for x in rank_yield.values],
                textposition='auto',
                visible=False
            ))

            fig.update_layout(
                title="Payment Rank Composition",
                xaxis_title="Distribution",
                yaxis_title="Payment Rank",
                height=400,
                margin=dict(l=20, r=20, t=80, b=20),
                updatemenus=[{
                    'buttons': [
                        {
                            'label': 'Weight',
                            'method': 'update',
                            'args': [
                                {'visible': [True, False, False]},
                                {
                                    'xaxis.title.text': 'Weight (%)',
                                    'xaxis.ticksuffix': '%'
                                }
                            ]
                        },
                        {
                            'label': 'Duration Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, True, False]},
                                {
                                    'xaxis.title.text': 'Duration Contribution (Years)',
                                    'xaxis.ticksuffix': ''
                                }
                            ]
                        },
                        {
                            'label': 'Yield Contrib.',
                            'method': 'update',
                            'args': [
                                {'visible': [False, False, True]},
                                {
                                    'xaxis.title.text': 'Yield Contribution (%)',
                                    'xaxis.ticksuffix': '%'
                                }
                            ]
                        }
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'y': 1.15,
                }]
            )
            st.plotly_chart(fig, use_container_width=True)

    # Add Top 10 Issuers and Bonds
    st.subheader("Top Holdings")
    cols2 = st.columns(2)

    with cols2[0]:
        # Top 10 issuers with toggles
        issuer_weight = df.groupby('Issuer')['Weight'].sum() * 100
        issuer_duration = df.groupby('Issuer')['Contribution to Duration'].sum()
        issuer_yield = df.groupby('Issuer')['Contribution to Yield'].sum() * 100

        # Get top 10 by weight and use this order for all metrics
        top_issuers = issuer_weight.nlargest(10).sort_values(ascending=True)
        top_issuer_duration = issuer_duration.reindex(top_issuers.index)
        top_issuer_yield = issuer_yield.reindex(top_issuers.index)

        fig = go.Figure()

        # Add all three traces
        fig.add_trace(go.Bar(
            x=top_issuers.values,
            y=top_issuers.index,
            orientation='h',
            name='Weight',
            text=[f'{x:.1f}%' for x in top_issuers.values],
            textposition='auto',
            visible=True
        ))

        fig.add_trace(go.Bar(
            x=top_issuer_duration.values,
            y=top_issuer_duration.index,
            orientation='h',
            name='Duration Contribution',
            text=[f'{x:.2f}' for x in top_issuer_duration.values],
            textposition='auto',
            visible=False
        ))

        fig.add_trace(go.Bar(
            x=top_issuer_yield.values,
            y=top_issuer_yield.index,
            orientation='h',
            name='Yield Contribution',
            text=[f'{x:.2f}' for x in top_issuer_yield.values],
            textposition='auto',
            visible=False
        ))

        fig.update_layout(
            title="Top 10 Issuers",
            xaxis_title="Distribution",
            yaxis_title="Issuer",
            height=400,
            margin=dict(l=20, r=20, t=80, b=20),
            updatemenus=[{
                'buttons': [
                    {
                        'label': 'Weight',
                        'method': 'update',
                        'args': [
                            {'visible': [True, False, False]},
                            {
                                'xaxis.title.text': 'Weight (%)',
                                'xaxis.ticksuffix': '%'
                            }
                        ]
                    },
                    {
                        'label': 'Duration Contrib.',
                        'method': 'update',
                        'args': [
                            {'visible': [False, True, False]},
                            {
                                'xaxis.title.text': 'Duration Contribution (Years)',
                                'xaxis.ticksuffix': ''
                            }
                        ]
                    },
                    {
                        'label': 'Yield Contrib.',
                        'method': 'update',
                        'args': [
                            {'visible': [False, False, True]},
                            {
                                'xaxis.title.text': 'Yield Contribution (%)',
                                'xaxis.ticksuffix': '%'
                            }
                        ]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
            }]
        )
        st.plotly_chart(fig, use_container_width=True)

    with cols2[1]:
        # Top 10 bonds with toggles
        # Get top 10 by weight and use this order for all metrics
        top_bonds = df.nlargest(10, 'Weight').sort_values('Weight', ascending=True)
        bond_labels = [f"{row['Security Name']}" for _, row in top_bonds.iterrows()]

        fig = go.Figure()

        # Add all three traces
        fig.add_trace(go.Bar(
            x=top_bonds['Weight'] * 100,
            y=bond_labels,
            orientation='h',
            name='Weight',
            text=[f'{x:.1f}%' for x in (top_bonds['Weight'] * 100)],
            textposition='auto',
            hovertemplate='%{y}<br>Weight: %{x:.1f}%<br>ISIN: %{customdata}<extra></extra>',
            customdata=top_bonds['ISIN'],
            visible=True
        ))

        fig.add_trace(go.Bar(
            x=top_bonds['Contribution to Duration'],
            y=bond_labels,
            orientation='h',
            name='Duration Contribution',
            text=[f'{x:.2f}' for x in top_bonds['Contribution to Duration']],
            textposition='auto',
            hovertemplate='%{y}<br>Duration Contrib: %{x:.2f}<br>ISIN: %{customdata}<extra></extra>',
            customdata=top_bonds['ISIN'],
            visible=False
        ))

        fig.add_trace(go.Bar(
            x=top_bonds['Contribution to Yield'] * 100,
            y=bond_labels,
            orientation='h',
            name='Yield Contribution',
            text=[f'{x:.2f}' for x in top_bonds['Contribution to Yield']],
            textposition='auto',
            hovertemplate='%{y}<br>Yield Contrib: %{x:.2f}%<br>ISIN: %{customdata}<extra></extra>',
            customdata=top_bonds['ISIN'],
            visible=False
        ))

        fig.update_layout(
            title="Top 10 Positions",
            xaxis_title="Distribution",
            yaxis_title="Security",
            height=400,
            margin=dict(l=20, r=20, t=80, b=20),
            updatemenus=[{
                'buttons': [
                    {
                        'label': 'Weight',
                        'method': 'update',
                        'args': [
                            {'visible': [True, False, False]},
                            {
                                'xaxis.title.text': 'Weight (%)',
                                'xaxis.ticksuffix': '%'
                            }
                        ]
                    },
                    {
                        'label': 'Duration Contrib.',
                        'method': 'update',
                        'args': [
                            {'visible': [False, True, False]},
                            {
                                'xaxis.title.text': 'Duration Contribution (Years)',
                                'xaxis.ticksuffix': ''
                            }
                        ]
                    },
                    {
                        'label': 'Yield Contrib.',
                        'method': 'update',
                        'args': [
                            {'visible': [False, False, True]},
                            {
                                'xaxis.title.text': 'Yield Contribution (%)',
                                'xaxis.ticksuffix': '%'
                            }
                        ]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
            }]
        )
        st.plotly_chart(fig, use_container_width=True)


def display_portfolio_holdings(df: pd.DataFrame):
    """Display portfolio holdings with enhanced metrics"""
    # Calculate portfolio total market value
    total_market_value = df['Market Value'].sum()

    # Calculate contribution metrics
    df['Weight'] = df['Market Value'] / total_market_value
    df['Contribution to Duration'] = df['Weight'] * df['Modified Duration']
    df['Contribution to Yield'] = df['Weight'] * df['YTM']

    # Format YTM and Coupon Rate as percentages
    df['YTM'] = df['YTM'] * 100
    df['Coupon Rate'] = df['Coupon Rate'] * 100

    # Select and reorder columns for display
    display_df = df[[
        'Security Name',
        'ISIN',
        'Market Value',
        'Weight',
        'Modified Duration',
        'Contribution to Duration',
        'YTM',
        'Contribution to Yield',
        'Coupon Rate',
        'Credit Rating',
        'Sector',
        'Country',
        'Payment Rank',
        'Maturity Date'
    ]].copy()

    # Format columns
    display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
    display_df['YTM'] = display_df['YTM'].map('{:.2f}%'.format)
    display_df['Coupon Rate'] = display_df['Coupon Rate'].map('{:.2f}%'.format)
    display_df['Modified Duration'] = display_df['Modified Duration'].map('{:.2f}'.format)
    display_df['Contribution to Duration'] = display_df['Contribution to Duration'].map('{:.3f}'.format)
    display_df['Contribution to Yield'] = display_df['Contribution to Yield'].map('{:.3%}'.format)
    display_df['Market Value'] = display_df['Market Value'].map('{:,.0f}'.format)
    display_df['Maturity Date'] = pd.to_datetime(display_df['Maturity Date']).dt.strftime('%Y-%m-%d')

    # Rename columns for display
    display_df.columns = [
        'Security Name',
        'ISIN',
        'Market Value',
        'Weight',
        'Mod. Duration',
        'Contrib. Duration',
        'YTM',
        'Contrib. Yield',
        'Coupon',
        'Rating',
        'Sector',
        'Country',
        'Payment Rank',
        'Maturity'
    ]

    st.dataframe(
        display_df,
        column_config={
            "Security Name": st.column_config.TextColumn(
                "Security Name",
                width="large",
            ),
            "ISIN": st.column_config.TextColumn(
                "ISIN",
                width="medium",
            ),
            "Market Value": st.column_config.TextColumn(
                "Market Value",
                width="medium",
            ),
        },
        hide_index=True,
    )


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
            display_portfolio_holdings(df)

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

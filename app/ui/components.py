import streamlit as st
import pandas as pd
from typing import List, Optional, Tuple
from app.data.models import Bond, PortfolioConstraints, CreditRating, OptimizationResult, RatingGrade
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np


def initialize_constraint_state():
    """Initialize session state for constraints if not exists"""
    if 'min_securities' not in st.session_state:
        st.session_state.min_securities = 5
    if 'max_securities' not in st.session_state:
        st.session_state.max_securities = 15
    if 'min_hy' not in st.session_state:
        st.session_state.min_hy = 0
    if 'max_hy' not in st.session_state:
        st.session_state.max_hy = 100
    if 'sector_constraints' not in st.session_state:
        st.session_state.sector_constraints = []  # List of (sector, max_exposure) tuples
    if 'payment_rank_constraints' not in st.session_state:
        st.session_state.payment_rank_constraints = []  # List of (rank, max_exposure) tuples
    if 'maturity_bucket_constraints' not in st.session_state:
        st.session_state.maturity_bucket_constraints = []  # List of (start_year, end_year, max_exposure) tuples


def add_constraint_row(constraint_type: str, value: tuple):
    """Add a new constraint row of the specified type"""
    if constraint_type == 'sector':
        st.session_state.sector_constraints.append(value)
    elif constraint_type == 'payment_rank':
        st.session_state.payment_rank_constraints.append(value)
    elif constraint_type == 'maturity_bucket':
        st.session_state.maturity_bucket_constraints.append(value)


def remove_constraint_row(constraint_type: str, index: int):
    """Remove a constraint row of the specified type"""
    if constraint_type == 'sector':
        st.session_state.sector_constraints.pop(index)
    elif constraint_type == 'payment_rank':
        st.session_state.payment_rank_constraints.pop(index)
    elif constraint_type == 'maturity_bucket':
        st.session_state.maturity_bucket_constraints.pop(index)


def validate_min_max(min_val: float, max_val: float, field_name: str) -> bool:
    """Validate min/max values and show error if invalid"""
    if min_val > max_val:
        st.error(f"{field_name}: Minimum value ({min_val}) cannot be greater than maximum value ({max_val})")
        return False
    return True


def render_main_constraints_form(universe: List[Bond]):
    """Render the main constraints form"""
    
    # Initialize session state for constraints if not exists
    if 'sector_constraints' not in st.session_state:
        st.session_state.sector_constraints = []
    if 'payment_rank_constraints' not in st.session_state:
        st.session_state.payment_rank_constraints = []
    if 'maturity_bucket_constraints' not in st.session_state:
        st.session_state.maturity_bucket_constraints = []

    # Main Constraints Form
    with st.form("constraints_form"):
        st.subheader("Portfolio Constraints")
        
        # Portfolio size
        total_size = st.number_input(
            "Portfolio Size",
            min_value=1_000_000,
            max_value=1_000_000_000,
            value=10_000_000,
            step=1_000_000,
            format="%d"
        )

        # Minimum yield constraint
        min_yield = st.number_input(
            "Minimum Portfolio Yield (%)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.1,
            format="%.1f"
        ) / 100.0

        # Duration constraints
        col1, col2 = st.columns(2)
        with col1:
            target_duration = st.number_input(
                "Target Duration",
                min_value=0.0,
                max_value=30.0,
                value=3.0,
                step=0.5
            )
        with col2:
            duration_tolerance = st.number_input(
                "Duration Tolerance",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1
            )

        # Rating constraints
        col1, col2 = st.columns(2)
        with col1:
            min_rating = st.selectbox(
                "Minimum Rating",
                options=list(CreditRating),
                format_func=lambda x: x.display(),
                index=len(CreditRating) - 10  # Default to BBB-
            )
        with col2:
            rating_tolerance = st.number_input(
                "Rating Tolerance (notches)",
                min_value=0,
                max_value=5,
                value=2,
                step=1
            )

        # Position size constraints
        col1, col2 = st.columns(2)
        with col1:
            min_position_size = st.number_input(
                "Minimum Position Size (%)",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                format="%.1f"
            ) / 100.0
        with col2:
            max_position_size = st.number_input(
                "Maximum Position Size (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.1,
                format="%.1f"
            ) / 100.0

        # Number of securities constraints
        col1, col2 = st.columns(2)
        with col1:
            min_securities = st.number_input(
                "Minimum Securities",
                min_value=1,
                max_value=100,
                value=1,
                step=1
            )
        with col2:
            max_securities = st.number_input(
                "Maximum Securities",
                min_value=min_securities,
                max_value=100,
                value=max(20, min_securities),
                step=1
            )

        # Issuer exposure constraint
        max_issuer_exposure = st.number_input(
            "Maximum Issuer Exposure (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=0.1,
            format="%.1f"
        ) / 100.0

        # High yield constraints
        st.subheader("High Yield Constraints")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_hy = st.number_input(
                "Minimum HY (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.0,
                format="%.1f",
                key="min_hy"
            ) / 100.0
        with col2:
            max_hy = st.number_input(
                "Maximum HY (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                format="%.1f",
                key="max_hy"
            ) / 100.0
        with col3:
            max_hy_position = st.number_input(
                "Maximum HY Position Size (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.1,
                format="%.1f",
                key="max_hy_position"
            ) / 100.0

        # Submit button
        submitted = st.form_submit_button("Run Optimization")

        if submitted:
            # Create PortfolioConstraints object
            constraints = PortfolioConstraints(
                total_size=total_size,
                target_duration=target_duration,
                duration_tolerance=duration_tolerance,
                min_rating=min_rating,
                rating_tolerance=rating_tolerance,
                min_position_size=min_position_size,
                max_position_size=max_position_size,
                min_securities=min_securities,
                max_securities=max_securities,
                max_issuer_exposure=max_issuer_exposure,
                min_yield=min_yield,
                grade_constraints={
                    RatingGrade.HIGH_YIELD: (min_hy, max_hy)
                } if max_hy > 0 else None,
                max_hy_position_size=max_hy_position,
                sector_constraints=dict(st.session_state.sector_constraints),
                payment_rank_constraints=dict(st.session_state.payment_rank_constraints),
                maturity_bucket_constraints={
                    f"{start_year}-{end_year}": max_exposure 
                    for start_year, end_year, max_exposure in st.session_state.maturity_bucket_constraints
                }
            )
            return constraints, True
    
    return None, False


def render_optional_constraints(universe: List[Bond]):
    """Render the optional constraints section"""
    
    # Get unique sectors and payment ranks from universe
    available_sectors = sorted(list(set(bond.sector for bond in universe if bond.sector)))
    available_payment_ranks = sorted(list(set(bond.payment_rank for bond in universe if bond.payment_rank)))
    
    st.subheader("Optional Constraints")
    
    # Sector constraints
    with st.expander("Sector Constraints", expanded=False):
        st.info("Add maximum exposure constraints for specific sectors")
        
        # Display existing sector constraints
        for i, (sector, max_exposure) in enumerate(st.session_state.sector_constraints):
            col1, col2, col3 = st.columns([2, 1, 0.2])
            with col1:
                sector = st.selectbox(
                    f"Sector {i+1}",
                    options=available_sectors,
                    index=available_sectors.index(sector) if sector in available_sectors else 0,
                    key=f"sector_{i}"
                )
            with col2:
                max_exposure = st.number_input(
                    f"Max Exposure {i+1} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=max_exposure * 100,
                    step=0.1,
                    format="%.1f",
                    key=f"sector_exposure_{i}"
                ) / 100.0
            with col3:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"remove_sector_{i}"):
                    remove_constraint_row('sector', i)
                    st.rerun()
            st.session_state.sector_constraints[i] = (sector, max_exposure)
        
        # Add new sector constraint
        if st.button("Add Sector"):
            add_constraint_row('sector', ("", 1.0))
            st.rerun()

    # Payment rank constraints
    with st.expander("Payment Rank Constraints", expanded=False):
        st.info("Add maximum exposure constraints for specific payment ranks")
        
        # Display existing payment rank constraints
        for i, (rank, max_exposure) in enumerate(st.session_state.payment_rank_constraints):
            col1, col2, col3 = st.columns([2, 1, 0.2])
            with col1:
                rank = st.selectbox(
                    f"Payment Rank {i+1}",
                    options=available_payment_ranks,
                    index=available_payment_ranks.index(rank) if rank in available_payment_ranks else 0,
                    key=f"rank_{i}"
                )
            with col2:
                max_exposure = st.number_input(
                    f"Max Exposure {i+1} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=max_exposure * 100,
                    step=0.1,
                    format="%.1f",
                    key=f"rank_exposure_{i}"
                ) / 100.0
            with col3:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"remove_rank_{i}"):
                    remove_constraint_row('payment_rank', i)
                    st.rerun()
            st.session_state.payment_rank_constraints[i] = (rank, max_exposure)
        
        # Add new payment rank constraint
        if st.button("Add Payment Rank"):
            add_constraint_row('payment_rank', ("", 1.0))
            st.rerun()
            
    # Maturity bucket constraints
    with st.expander("Maturity Bucket Constraints", expanded=False):
        st.info("Add maximum exposure constraints for specific maturity buckets")
        
        # Display existing maturity bucket constraints
        for i, (start_year, end_year, max_exposure) in enumerate(st.session_state.maturity_bucket_constraints):
            col1, col2, col3, col4 = st.columns([1, 1, 1, 0.2])
            with col1:
                start_year = st.number_input(
                    f"Start Year {i+1}",
                    min_value=2020,
                    max_value=2050,
                    value=start_year,
                    step=1,
                    key=f"start_year_{i}"
                )
            with col2:
                end_year = st.number_input(
                    f"End Year {i+1}",
                    min_value=start_year,
                    max_value=2050,
                    value=max(end_year, start_year),
                    step=1,
                    key=f"end_year_{i}"
                )
            with col3:
                max_exposure = st.number_input(
                    f"Max Exposure {i+1} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=max_exposure * 100,
                    step=0.1,
                    format="%.1f",
                    key=f"maturity_exposure_{i}"
                ) / 100.0
            with col4:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"remove_maturity_{i}"):
                    remove_constraint_row('maturity_bucket', i)
                    st.rerun()
            st.session_state.maturity_bucket_constraints[i] = (start_year, end_year, max_exposure)
        
        # Add new maturity bucket constraint
        if st.button("Add Maturity Bucket"):
            add_constraint_row('maturity_bucket', (datetime.now().year, datetime.now().year+1, 1.0))
            st.rerun()

def render_constraints_form(universe: List[Bond]):
    """Render both main and optional constraints forms"""
    constraints, run_optimization = render_main_constraints_form(universe)
    render_optional_constraints(universe)
    st.markdown("---")
    return constraints, run_optimization


def display_optimization_results(result: OptimizationResult, universe: List[Bond], total_size: float):
    """Display optimization results"""
    
    # Create portfolio dataframe with non-zero weights
    portfolio_df = pd.DataFrame([
        {
            'ISIN': bond.isin,
            'Issuer': bond.issuer,
            'Country': bond.country,
            'Sector': bond.sector,
            'Rating': bond.credit_rating.display(),
            'Payment Rank': bond.payment_rank,
            'Maturity': bond.maturity_date.strftime('%Y-%m-%d'),
            'Yield': bond.ytm,
            'Duration': bond.modified_duration,
            'Weight': result.portfolio.get(bond.isin, 0),
            'Market Value': result.portfolio.get(bond.isin, 0) * total_size
        }
        for bond in universe if result.portfolio.get(bond.isin, 0) > 0
    ])

    if not result.success:
        st.error("Optimization failed to find a solution")
        if result.constraint_violations:
            st.write("Constraint violations:")
            for violation in result.constraint_violations:
                st.write(f"- {violation}")
        return

    if not result.constraints_satisfied:
        st.warning("Found a solution but some constraints are slightly violated")
        if result.constraint_violations:
            st.write("Constraint violations (may be due to numerical precision):")
            for violation in result.constraint_violations:
                st.write(f"- {violation}")

    # Portfolio metrics
    st.subheader("Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Yield", f"{result.metrics['yield']:.2%}")
    with col2:
        st.metric("Duration", f"{result.metrics['duration']:.2f}")
    with col3:
        rating_score = result.metrics['rating']
        rating = CreditRating.from_score(float(rating_score))
        st.metric(
            "Rating",
            rating.display(),
            help="Portfolio average rating calculated on a logarithmic scale where AAA=1, AA+=2, AA=3, etc. Lower score means better rating."
        )
    with col4:
        st.metric("Number of Securities", f"{int(result.metrics['num_securities'])}")
        st.metric("Number of Issuers", f"{int(result.metrics['num_issuers'])}")

    # Grade exposures
    st.subheader("Rating Grade Exposures")
    grade_cols = st.columns(len(RatingGrade))
    for i, grade in enumerate(RatingGrade):
        with grade_cols[i]:
            exposure = result.metrics.get(f'grade_{grade.value}', 0)
            st.metric(grade.value, f"{exposure:.1%}")

    # Portfolio breakdown
    st.subheader("Portfolio Breakdown")
    col1, col2 = st.columns(2)

    # Create portfolio dataframe
    portfolio_data = []

    for isin, weight in result.portfolio.items():
        bond = next(b for b in universe if b.isin == isin)
        notional = weight * total_size  # Use passed total_size parameter
        min_notional = bond.min_piece
        increment = bond.increment_size

        # Calculate rounded notional
        if notional < min_notional:
            warning = f"Position too small (min: {min_notional:,.0f})"
            rounded_notional = 0
        else:
            rounded_notional = (notional // increment) * increment
            if rounded_notional < min_notional:
                rounded_notional = min_notional
                warning = f"Rounded up to minimum piece size ({min_notional:,.0f})"
            else:
                warning = ""

        portfolio_data.append({
            'isin': isin,
            'weight': weight,
            'country': bond.country,
            'issuer': bond.issuer,
            'coupon': bond.coupon_rate,
            'maturity': bond.maturity_date,
            'currency': bond.currency,
            'ytm': bond.ytm,
            'duration': bond.modified_duration,
            'rating': bond.credit_rating.display(),
            'grade': bond.rating_grade.value,
            'payment_rank': bond.payment_rank,
            'target_notional': notional,
            'rounded_notional': rounded_notional,
            'min_piece': min_notional,
            'increment': increment,
            'warning': warning,
            'sector': bond.sector
        })
    df_portfolio = pd.DataFrame(portfolio_data)

    # Add rounded weight column
    df_portfolio['rounded_weight'] = df_portfolio['rounded_notional'] / total_size

    # Country breakdown pie chart
    with col1:
        country_weights = df_portfolio.groupby('country')['weight'].sum()
        fig = px.pie(
            values=country_weights.values,
            names=country_weights.index,
            title='Country Breakdown'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sector breakdown bar chart
        sector_weights = df_portfolio.groupby('sector')['weight'].sum().sort_values(ascending=True)
        fig = go.Figure(data=[go.Bar(
            x=sector_weights.values * 100,  # Convert to percentage
            y=sector_weights.index,
            orientation='h'
        )])
        fig.update_layout(
            title='Sector Breakdown',
            xaxis_title='Weight (%)',
            yaxis_title='Sector',
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rating breakdown and Payment Rank
    with col2:
        # Create two bar charts
        fig = go.Figure()

        # Detailed rating breakdown
        rating_weights = df_portfolio.groupby('rating')['weight'].sum()
        fig.add_trace(go.Bar(
            x=rating_weights.index,
            y=rating_weights.values * 100,
            name='By Rating',
            visible=True
        ))

        # IG/HY breakdown
        grade_weights = df_portfolio.groupby('grade')['weight'].sum()
        fig.add_trace(go.Bar(
            x=grade_weights.index,
            y=grade_weights.values * 100,
            name='By Grade',
            visible=False
        ))

        # Add buttons to switch between views
        fig.update_layout(
            title='Rating Breakdown',
            yaxis_title='Weight (%)',
            updatemenus=[{
                'buttons': [
                    {'label': 'By Rating', 'method': 'update', 'args': [{'visible': [True, False]}]},
                    {'label': 'By Grade', 'method': 'update', 'args': [{'visible': [False, True]}]}
                ],
                'direction': 'down',
                'showactive': True,
            }]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Payment Rank breakdown
        rank_weights = df_portfolio.groupby('payment_rank')['weight'].sum()
        fig = px.pie(
            values=rank_weights.values * 100,  # Convert to percentage
            names=rank_weights.index,
            title='Payment Rank Breakdown'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cash flow distribution
    st.subheader("Cash Flow Distribution")
    years = range(datetime.now().year, max(df_portfolio['maturity']).year + 1)
    coupons = []
    redemptions = []

    for year in years:
        # Calculate coupon payments using rounded notionals
        year_coupons = sum(
            row['rounded_notional'] * row['coupon']
            for _, row in df_portfolio.iterrows()
            if row['maturity'].year >= year
        )
        coupons.append(year_coupons)

        # Calculate redemptions using rounded notionals
        year_redemptions = sum(
            row['rounded_notional']
            for _, row in df_portfolio.iterrows()
            if row['maturity'].year == year
        )
        redemptions.append(year_redemptions)

    cash_flows = pd.DataFrame({
        'year': years,
        'coupons': coupons,
        'redemptions': redemptions
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(years),
        y=coupons,
        name='Coupons'
    ))
    fig.add_trace(go.Bar(
        x=list(years),
        y=redemptions,
        name='Redemptions'
    ))

    fig.update_layout(
        title='Cash Flow Distribution',
        xaxis_title='Year',
        yaxis_title='Amount',
        barmode='stack'
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Cash Flows Table"):
        st.dataframe(cash_flows,
                     column_config={
                         'year': st.column_config.NumberColumn(
                             'Year',
                             format="%.0f"),
                         'redemptions': st.column_config.NumberColumn(
                             'Redemptions',
                             format="%.0f"),  # No decimal place,
                         'coupons': st.column_config.NumberColumn(
                             'Coupons',
                             format="%.2f"),  # Two decimal place,
                     },
                     hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        # Top 10 issuers
        st.subheader("Top 10 Issuers")
        issuer_weights = df_portfolio.groupby('issuer')['weight'].sum().sort_values(ascending=False).head(10)
        issuer_df = pd.DataFrame({
            'Issuer': issuer_weights.index,
            'Weight': issuer_weights.values * 100  # Convert to percentage
        })
        st.dataframe(
            issuer_df,
            column_config={
                'Weight': st.column_config.NumberColumn(
                    'Weight',
                    format="%.1f%%"  # Use % format with one decimal place
                )
            },
            hide_index=True
        )

    with col2:
        # Top 10 bonds
        st.subheader("Top 10 Bonds")
        top_bonds = df_portfolio.nlargest(10, 'weight')[
            ['isin', 'issuer', 'coupon', 'maturity', 'currency', 'ytm', 'weight', 'payment_rank']
        ].copy()
        top_bonds['coupon'] = top_bonds['coupon'].map('{:.2%}'.format)
        top_bonds['ytm'] = top_bonds['ytm'].map('{:.2%}'.format)
        top_bonds['weight'] = top_bonds['weight'].map('{:.2%}'.format)
        top_bonds['maturity'] = top_bonds['maturity'].dt.strftime('%Y-%m-%d')

        st.dataframe(top_bonds, 
        column_config={
            'isin': 'ISIN',
            'issuer': 'Issuer',
            'coupon': 'Coupon',
            'ytm': 'YTM',
            'weight': 'Weight',
            'maturity': 'Maturity',
            'currency':'Currency',
            'payment_rank': 'Payment Rank'
        },
        hide_index=True)

    # Complete portfolio
    st.subheader("Complete Portfolio")
    df_display = df_portfolio.copy()
    df_display['ytm'] = df_display['ytm'].map('{:.2%}'.format)
    df_display['weight'] = df_display['weight'].map('{:.2%}'.format)
    df_display['rounded_weight'] = df_display['rounded_weight'].map('{:.2%}'.format)
    df_display['coupon'] = df_display['coupon'].map('{:.2%}'.format)
    df_display['maturity'] = df_display['maturity'].dt.strftime('%Y-%m-%d')
    df_display['target_notional'] = df_display['target_notional'].map('{:,.2f}'.format)
    df_display['rounded_notional'] = df_display['rounded_notional'].map('{:,.2f}'.format)
    df_display['min_piece'] = df_display['min_piece'].map('{:,.2f}'.format)
    df_display['increment'] = df_display['increment'].map('{:,.2f}'.format)

    st.dataframe(
        df_display,
        column_config={
            'isin': 'ISIN',
            'issuer': 'Issuer',
            'rating': 'Rating',
            'payment_rank': 'Payment Rank',
            'ytm': 'YTM',
            'currency': 'Currency',
            'duration': 'Duration',
            'weight': 'Target Weight',
            'rounded_weight': 'Rounded Weight',
            'target_notional': 'Target Notional',
            'rounded_notional': 'Rounded Notional',
            'min_piece': 'Min Piece',
            'increment': 'Increment',
            'country': 'Country',
            'coupon': 'Coupon',
            'maturity': 'Maturity',
            'grade': 'Grade',
            'warning': 'Warning',
            'sector': 'Sector'
        },
        hide_index=True
    )

    # Add CSV download button
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📝 Download Portfolio as CSV",
        csv,
        "portfolio.csv",
        "text/csv",
        key='download-csv'
    )

    # Display any warnings about position sizes
    warnings = df_portfolio[df_portfolio['warning'] != '']
    if not warnings.empty:
        st.warning("Position Size Adjustments Required:")
        for _, row in warnings.iterrows():
            st.write(f"- {row['isin']} ({row['issuer']}): {row['warning']}")

    # Show total portfolio metrics after rounding
    if len(df_portfolio) > 0:
        st.subheader("Portfolio Summary")
        total_target = df_portfolio['target_notional'].sum()
        total_rounded = df_portfolio['rounded_notional'].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Target Size", f"{total_target:,.0f}")
        with col2:
            st.metric("Total Rounded Size", f"{total_rounded:,.0f}")
        with col3:
            diff_pct = (total_rounded - total_target) / total_target * 100
            st.metric("Size Difference", f"{diff_pct:+.2f}%")

    # Add PowerPoint download button
    try:
        with st.spinner('Generating PowerPoint presentation...'):
            from .presentation import generate_portfolio_presentation
            #st.info('Starting PowerPoint generation...')
            pptx_stream = generate_portfolio_presentation(result, universe, total_size)
            #st.success('Presentation ready for download!')
            st.download_button(
                label="📺 Download as PowerPoint",
                data=pptx_stream,
                file_name="portfolio_analysis.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
    except Exception as e:
        st.error(f"Error generating PowerPoint: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
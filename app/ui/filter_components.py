"""Filter UI components"""
import streamlit as st
from typing import List, Dict, Any, Optional
from app.data.models import Bond
from app.filters import FilterManager
import uuid
from datetime import datetime

def delete_condition(group_id: str, condition_index: int):
    """Callback to delete a condition from a group"""
    for group in st.session_state.active_filters['exclusion_groups']:
        if group['id'] == group_id:
            group['conditions'].pop(condition_index)
            break

def delete_group(group_id: str):
    """Callback to delete a group"""
    st.session_state.active_filters['exclusion_groups'] = [
        group for group in st.session_state.active_filters['exclusion_groups'] 
        if group['id'] != group_id
    ]

def on_ytm_change():
    """Callback for YTM slider changes"""
    if 'ytm_range' in st.session_state:
        st.session_state.active_filters['range_filters']['ytm'] = {
            'min': st.session_state.ytm_range[0] / 100,
            'max': st.session_state.ytm_range[1] / 100
        }

def on_duration_change():
    """Callback for duration slider changes"""
    if 'duration_range' in st.session_state:
        st.session_state.active_filters['range_filters']['modified_duration'] = {
            'min': st.session_state.duration_range[0],
            'max': st.session_state.duration_range[1]
        }

def on_maturity_change():
    """Callback for maturity year slider changes"""
    if 'maturity_range' in st.session_state:
        st.session_state.active_filters['range_filters']['maturity_year'] = {
            'min': st.session_state.maturity_range[0],
            'max': st.session_state.maturity_range[1]
        }

def render_filter_controls(universe: List[Bond], filter_manager: FilterManager) -> Optional[List[Bond]]:
    """Render filter controls and return filtered universe"""
    if not universe:
        return None
        
    st.subheader("Universe Filters")
    
    # Initialize session state for filters if not exists
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {
            'exclusion_groups': [],
            'range_filters': {
                'ytm': {'min': min(bond.ytm for bond in universe), 'max': max(bond.ytm for bond in universe)},
                'modified_duration': {'min': min(bond.modified_duration for bond in universe), 'max': max(bond.modified_duration for bond in universe)},
                'maturity_year': {'min': min(bond.maturity_date.year for bond in universe), 'max': max(bond.maturity_date.year for bond in universe)}
            }
        }
    elif not isinstance(st.session_state.active_filters, dict):
        st.session_state.active_filters = {
            'exclusion_groups': [],
            'range_filters': {
                'ytm': {'min': min(bond.ytm for bond in universe), 'max': max(bond.ytm for bond in universe)},
                'modified_duration': {'min': min(bond.modified_duration for bond in universe), 'max': max(bond.modified_duration for bond in universe)},
                'maturity_year': {'min': min(bond.maturity_date.year for bond in universe), 'max': max(bond.maturity_date.year for bond in universe)}
            }
        }
    else:
        # Ensure all required keys exist
        if 'exclusion_groups' not in st.session_state.active_filters:
            st.session_state.active_filters['exclusion_groups'] = []
        if 'range_filters' not in st.session_state.active_filters:
            st.session_state.active_filters['range_filters'] = {}
        if 'ytm' not in st.session_state.active_filters.get('range_filters', {}):
            st.session_state.active_filters['range_filters']['ytm'] = {
                'min': min(bond.ytm for bond in universe),
                'max': max(bond.ytm for bond in universe)
            }
        if 'modified_duration' not in st.session_state.active_filters.get('range_filters', {}):
            st.session_state.active_filters['range_filters']['modified_duration'] = {
                'min': min(bond.modified_duration for bond in universe),
                'max': max(bond.modified_duration for bond in universe)
            }
        if 'maturity_year' not in st.session_state.active_filters.get('range_filters', {}):
            st.session_state.active_filters['range_filters']['maturity_year'] = {
                'min': min(bond.maturity_date.year for bond in universe),
                'max': max(bond.maturity_date.year for bond in universe)
            }
    
    # Initialize session state
    if 'selected_predefined_filter' not in st.session_state:
        st.session_state.selected_predefined_filter = "None"
    if 'show_success_message' not in st.session_state:
        st.session_state.show_success_message = None
    if 'filter_loaded' not in st.session_state:
        st.session_state.filter_loaded = False
    
    # Predefined filters section
    st.write("Predefined Filters")
    predefined = filter_manager.get_predefined_filters()
    predefined_options = {"None": "No filter"} | predefined
    
    # Handle save/delete/update operations before creating the selectbox
    if 'save_filter_clicked' in st.session_state and st.session_state.save_filter_clicked:
        filter_name = st.session_state.get('filter_name', '')
        filter_desc = st.session_state.get('filter_desc', '')
        if filter_manager.save_filter(filter_name, filter_desc, st.session_state.active_filters):
            st.session_state.show_success_message = f"Filter '{filter_name}' saved successfully"
            st.session_state.selected_predefined_filter = filter_name
        st.session_state.save_filter_clicked = False
        st.rerun()
    
    if 'delete_filter_clicked' in st.session_state and st.session_state.delete_filter_clicked:
        filter_to_delete = st.session_state.selected_predefined_filter
        if filter_manager.delete_filter(filter_to_delete):
            st.session_state.show_success_message = f"Filter '{filter_to_delete}' deleted"
            st.session_state.selected_predefined_filter = "None"
            st.session_state.filter_loaded = False
        st.session_state.delete_filter_clicked = False
        st.rerun()
    
    if 'update_filter_clicked' in st.session_state and st.session_state.update_filter_clicked:
        filter_to_update = st.session_state.selected_predefined_filter
        if filter_manager.update_filter(filter_to_update, st.session_state.active_filters):
            st.session_state.show_success_message = f"Filter '{filter_to_update}' updated successfully"
        st.session_state.update_filter_clicked = False
        st.rerun()
    
    # Show success message if exists
    if st.session_state.show_success_message:
        st.success(st.session_state.show_success_message)
        st.session_state.show_success_message = None
    
    # Create the filter selection dropdown
    selected_filter = st.selectbox(
        "Select Filter",
        options=list(predefined_options.keys()),
        format_func=lambda x: f"{x}: {predefined_options[x]}" if x in predefined else x,
        key="selected_predefined_filter"
    )
    
    # Update session state and get filter details if changed
    if selected_filter != "None":
        # Only load filter if it's newly selected
        if not st.session_state.filter_loaded or st.session_state.selected_predefined_filter != selected_filter:
            filter_config = filter_manager._predefined_filters[selected_filter]['filters']
            
            # Add IDs to groups and conditions if they don't exist
            for group in filter_config['exclusion_groups']:
                if 'id' not in group:
                    group['id'] = str(uuid.uuid4())
                for condition in group['conditions']:
                    if 'id' not in condition:
                        condition['id'] = str(uuid.uuid4())
            
            st.session_state.active_filters = filter_config
            st.session_state.filter_loaded = True
    
    # Custom filters section
    with st.expander("Custom Filters", expanded=True):
        # Save/Delete/Update filter controls
        if selected_filter != "None":
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Delete Filter"):
                    st.session_state.delete_filter_clicked = True
                    st.rerun()
            with col2:
                if st.button("📝 Update Filter"):
                    st.session_state.update_filter_clicked = True
                    st.rerun()
        else:
            # Save filter form
            st.write("Save Current Filter")
            col1, col2 = st.columns(2)
            with col1:
                filter_name = st.text_input("Filter Name", placeholder="e.g., High Grade Tech", key="filter_name")
            with col2:
                filter_desc = st.text_input("Description", placeholder="e.g., High grade technology sector bonds", key="filter_desc")
            
            if st.button("💾 Save Filter"):
                if not filter_name or not filter_desc:
                    st.error("Please provide both name and description")
                elif filter_name in filter_manager.get_predefined_filters():
                    st.error(f"Filter name '{filter_name}' already exists")
                else:
                    st.session_state.save_filter_clicked = True
                    st.rerun()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Range Filters")
            
            # YTM filter
            st.write("Yield")
            ytm_min = min(bond.ytm for bond in universe)
            ytm_max = max(bond.ytm for bond in universe)
            current_ytm = st.session_state.active_filters.get('range_filters', {}).get('ytm', {})
            ytm_range = st.slider(
                "",
                min_value=float(ytm_min * 100),
                max_value=float(ytm_max * 100),
                value=(
                    float(current_ytm.get('min', ytm_min) * 100),
                    float(current_ytm.get('max', ytm_max) * 100)
                ),
                step=0.1,
                format="%.1f%%",
                key="ytm_range",
                on_change=on_ytm_change
            )
            
            # Duration filter
            st.write("Duration")
            dur_min = min(bond.modified_duration for bond in universe)
            dur_max = max(bond.modified_duration for bond in universe)
            current_dur = st.session_state.active_filters.get('range_filters', {}).get('modified_duration', {})
            dur_range = st.slider(
                "",
                min_value=float(dur_min),
                max_value=float(dur_max),
                value=(
                    float(current_dur.get('min', dur_min)),
                    float(current_dur.get('max', dur_max))
                ),
                step=0.1,
                format="%.1f",
                key="duration_range",
                on_change=on_duration_change
            )

            # Maturity filter
            st.write("Maturity Year")
            mat_min = min(bond.maturity_date.year for bond in universe)
            mat_max = max(bond.maturity_date.year for bond in universe)
            current_mat = st.session_state.active_filters.get('range_filters', {}).get('maturity_year', {})
            mat_range = st.slider(
                "",
                min_value=int(mat_min),
                max_value=int(mat_max),
                value=(
                    int(current_mat.get('min', mat_min)),
                    int(current_mat.get('max', mat_max))
                ),
                step=1,
                format="%d",
                key="maturity_range",
                on_change=on_maturity_change
            )
            
        with col2:
            st.write("Exclusion Rules")
            
            # Add new group button
            if st.button("+ Add Exclusion Group"):
                st.session_state.active_filters['exclusion_groups'].append({
                    'id': str(uuid.uuid4()),
                    'conditions': []
                })
            
            # Render each exclusion group
            for i, group in enumerate(st.session_state.active_filters['exclusion_groups']):
                with st.container():
                    st.write(f"Group {i + 1}")
                    
                    # Add condition button
                    if st.button(f"+ Add Condition", key=f"add_cond_{group['id']}"):
                        # Get category from first condition in group if it exists
                        default_category = 'sector'
                        if group['conditions']:
                            default_category = group['conditions'][0].get('category', 'sector')
                        
                        group['conditions'].append({
                            'id': str(uuid.uuid4()),
                            'category': default_category,
                            'value': None
                        })
                    
                    # Render conditions
                    for j, condition in enumerate(group['conditions']):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            current_category = condition.get('category', 'sector')
                            new_category = st.selectbox(
                                "Category",
                                options=['sector', 'payment_rank', 'rating', 'issuer', 'country'],
                                key=f"cat_{condition['id']}",
                                index=(['sector', 'payment_rank', 'rating', 'issuer', 'country'].index(current_category))
                            )
                            if new_category != current_category:
                                condition['category'] = new_category
                                condition['value'] = None  # Reset value when category changes
                        
                        with col2:
                            # Get available values for the selected category
                            if new_category == 'rating':
                                values = sorted(list(set(bond.credit_rating.display() for bond in universe)))
                            else:
                                values = sorted(list(set(getattr(bond, new_category, 'Unknown') for bond in universe)))
                            
                            current_value = condition.get('value')
                            try:
                                value_index = values.index(current_value) if current_value in values else 0
                            except ValueError:
                                value_index = 0
                                
                            new_value = st.selectbox(
                                "Value",
                                options=values,
                                key=f"val_{condition['id']}",
                                index=value_index
                            )
                            if new_value != current_value:
                                condition['value'] = new_value
                        
                        with col3:
                            st.write("")
                            st.write("")
                            st.button(
                                "🗑️", 
                                key=f"del_cond_{condition['id']}", 
                                on_click=delete_condition,
                                args=(group['id'], j)
                            )
                    
                    # Group delete button
                    st.button(
                        "Delete Group", 
                        key=f"del_group_{group['id']}", 
                        on_click=delete_group,
                        args=(group['id'],)
                    )
                    
                    st.markdown("---")
    
    # Apply filters
    filtered_universe = filter_manager.apply_filter(universe, st.session_state.active_filters)
    
    # Show filter stats
    st.info(f"Remaining {len(filtered_universe)} of {len(universe)} bonds", icon="ℹ️")
    
    return filtered_universe

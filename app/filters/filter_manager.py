"""Filter management module"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from app.data.models import Bond

class FilterManager:
    """Manages universe filters"""
    def __init__(self):
        self.filters_path = Path(__file__).parent.parent.parent / "data" / "filters"
        self.filters_file = self.filters_path / "filters.json"
        self.last_used_file = self.filters_path / "last_used.json"
        self._predefined_filters = self._load_predefined_filters()
        
    def _load_predefined_filters(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined filters from JSON file"""
        if not self.filters_file.exists():
            return {}
        with open(self.filters_file, 'r') as f:
            return json.load(f)
    
    def get_predefined_filters(self) -> Dict[str, str]:
        """Get list of predefined filters with descriptions"""
        return {k: v['description'] for k, v in self._predefined_filters.items()}
    
    def apply_filter(self, universe: List[Bond], filter_config: Dict[str, Any]) -> List[Bond]:
        """Apply filter configuration to universe"""
        if not filter_config:
            return universe

        df = pd.DataFrame([bond.__dict__ for bond in universe])
        
        # Apply range filters
        range_filters = filter_config.get('range_filters', {})
        for field, range_values in range_filters.items():
            if field == 'maturity_year':
                # Special handling for maturity year to include full years
                min_val = range_values.get('min')
                max_val = range_values.get('max')
                if min_val is not None:
                    df = df[df['maturity_date'].dt.year >= min_val]
                if max_val is not None:
                    # Include bonds maturing up to the end of the max year
                    df = df[df['maturity_date'].dt.year <= max_val]
            elif field in df.columns:
                min_val = range_values.get('min')
                max_val = range_values.get('max')
                if min_val is not None:
                    df = df[df[field] >= min_val]
                if max_val is not None:
                    df = df[df[field] <= max_val]
        
        # Apply exclusion groups if present
        if 'exclusion_groups' in filter_config:
            group_masks = []
            for group in filter_config['exclusion_groups']:
                # Start with all True for this group
                group_mask = pd.Series([True] * len(df))
                # Apply all conditions in the group (AND logic)
                for condition in group['conditions']:
                    field = condition['category']
                    value = condition['value']
                    if field in df.columns:
                        group_mask &= (df[field] == value)
                group_masks.append(group_mask)
            
            # Combine all group masks with OR logic and invert (we want to exclude matches)
            if group_masks:
                exclude_mask = pd.concat(group_masks, axis=1).any(axis=1)
                df = df[~exclude_mask]
        
        # Convert back to list of Bond objects
        filtered_isins = df['isin'].tolist()
        return [bond for bond in universe if bond.isin in filtered_isins]
    
    def apply_predefined_filter(self, universe: List[Bond], filter_name: str) -> List[Bond]:
        """Apply a predefined filter to the universe"""
        if filter_name not in self._predefined_filters:
            return universe
        
        filter_config = self._predefined_filters[filter_name]['filters']
        return self.apply_filter(universe, filter_config)
    
    def save_last_used(self, filter_config: Dict[str, Any]) -> None:
        """Save last used filter configuration"""
        with open(self.last_used_file, 'w') as f:
            json.dump(filter_config, f, indent=2)
    
    def load_last_used(self) -> Optional[Dict[str, Any]]:
        """Load last used filter configuration"""
        if not self.last_used_file.exists():
            return None
        try:
            with open(self.last_used_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def save_predefined_filter(self, name: str, description: str, filter_config: Dict[str, Any]) -> bool:
        """Save a new predefined filter"""
        if not name or not description:
            return False
            
        # Format the filter entry
        filter_entry = {
            "description": description,
            "filters": filter_config
        }
        
        # Load existing filters
        try:
            with open(self.filters_file, 'r') as f:
                filters = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            filters = {}
            
        # Add new filter
        filters[name] = filter_entry
        
        # Save back to file
        try:
            with open(self.filters_file, 'w') as f:
                json.dump(filters, f, indent=4)
            return True
        except Exception:
            return False
    
    def delete_predefined_filter(self, filter_name: str) -> bool:
        """Delete a predefined filter"""
        try:
            with open(self.filters_file, 'r') as f:
                filters = json.load(f)
                
            if filter_name in filters:
                del filters[filter_name]
                
                with open(self.filters_file, 'w') as f:
                    json.dump(filters, f, indent=4)
                return True
            return False
        except Exception:
            return False

    def save_filter(self, name: str, description: str, filters: Dict[str, Any]) -> bool:
        """Save a new filter or update existing one"""
        try:
            self._predefined_filters[name] = {
                "description": description,
                "filters": filters
            }
            self.save_predefined_filters()
            return True
        except Exception as e:
            print(f"Error saving filter: {e}")
            return False

    def delete_filter(self, name: str) -> bool:
        """Delete a filter by name"""
        try:
            if name in self._predefined_filters:
                del self._predefined_filters[name]
                self.save_predefined_filters()
                return True
            return False
        except Exception as e:
            print(f"Error deleting filter: {e}")
            return False

    def save_predefined_filters(self):
        """Save predefined filters to JSON file"""
        # Ensure directory exists
        self.filters_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filters_file, "w") as f:
            json.dump(self._predefined_filters, f, indent=4)

    def update_filter(self, name: str, filters: dict) -> bool:
        """Update an existing filter while preserving its description"""
        try:
            if name in self._predefined_filters:
                # Preserve the original description
                description = self._predefined_filters[name]["description"]
                self._predefined_filters[name] = {
                    "description": description,
                    "filters": filters
                }
                self.save_predefined_filters()
                return True
            return False
        except Exception as e:
            print(f"Error updating filter: {e}")
            return False

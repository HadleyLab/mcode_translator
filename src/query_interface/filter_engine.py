"""
Filter Engine module for applying structured filters to medical code search results.
Provides advanced filtering capabilities based on multiple criteria.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from .search_index import IndexedCode


@dataclass
class FilterCriteria:
    """Represents filter criteria for medical codes"""
    types: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    status: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None
    custom_filters: Optional[Dict[str, Any]] = None


class FilterEngine:
    """Handles filtering of medical codes based on structured criteria"""

    def __init__(self):
        self._filter_functions: Dict[str, Callable] = {}
        self._initialize_filters()

    def _initialize_filters(self) -> None:
        """Initialize standard filter functions"""
        self._filter_functions.update({
            'type': self._filter_by_type,
            'category': self._filter_by_category,
            'status': self._filter_by_status,
            'date_range': self._filter_by_date_range
        })

    def add_custom_filter(self, name: str, filter_func: Callable[[IndexedCode, Any], bool]) -> None:
        """
        Add a custom filter function
        
        Args:
            name: Name of the filter
            filter_func: Function that takes (IndexedCode, filter_value) and returns bool
        """
        self._filter_functions[name] = filter_func

    def apply_filters(self, codes: List[IndexedCode], criteria: FilterCriteria) -> List[IndexedCode]:
        """
        Apply filter criteria to a list of medical codes
        
        Args:
            codes: List of IndexedCode objects to filter
            criteria: FilterCriteria specifying the filters to apply
            
        Returns:
            Filtered list of IndexedCode objects
        """
        filtered_codes = codes

        # Apply type filters
        if criteria.types:
            filtered_codes = [
                code for code in filtered_codes
                if self._filter_by_type(code, criteria.types)
            ]

        # Apply category filters
        if criteria.categories:
            filtered_codes = [
                code for code in filtered_codes
                if self._filter_by_category(code, criteria.categories)
            ]

        # Apply status filters
        if criteria.status:
            filtered_codes = [
                code for code in filtered_codes
                if self._filter_by_status(code, criteria.status)
            ]

        # Apply date range filter
        if criteria.date_range:
            filtered_codes = [
                code for code in filtered_codes
                if self._filter_by_date_range(code, criteria.date_range)
            ]

        # Apply custom filters
        if criteria.custom_filters:
            for filter_name, filter_value in criteria.custom_filters.items():
                if filter_name in self._filter_functions:
                    filtered_codes = [
                        code for code in filtered_codes
                        if self._filter_functions[filter_name](code, filter_value)
                    ]

        return filtered_codes

    def _filter_by_type(self, code: IndexedCode, allowed_types: List[str]) -> bool:
        """Filter codes by their type (ICD-10, SNOMED, etc.)"""
        return code.type in allowed_types

    def _filter_by_category(self, code: IndexedCode, allowed_categories: List[str]) -> bool:
        """Filter codes by their category"""
        return code.category in allowed_categories

    def _filter_by_status(self, code: IndexedCode, allowed_status: List[str]) -> bool:
        """Filter codes by their status"""
        return code.status in allowed_status

    def _filter_by_date_range(self, code: IndexedCode, date_range: Dict[str, datetime]) -> bool:
        """
        Filter codes by date range using metadata
        
        Args:
            code: IndexedCode object
            date_range: Dictionary with 'start' and/or 'end' datetime objects
        """
        if 'effective_date' not in code.metadata:
            return True  # Include codes without dates by default

        effective_date = code.metadata['effective_date']
        
        if not isinstance(effective_date, datetime):
            try:
                effective_date = datetime.fromisoformat(effective_date)
            except (ValueError, TypeError):
                return True  # Include codes with invalid dates by default

        if 'start' in date_range and effective_date < date_range['start']:
            return False
        if 'end' in date_range and effective_date > date_range['end']:
            return False
            
        return True

    def create_filter_criteria(self, filter_dict: Dict[str, Any]) -> FilterCriteria:
        """
        Create FilterCriteria from a dictionary
        
        Args:
            filter_dict: Dictionary containing filter parameters
            
        Returns:
            FilterCriteria object
            
        Example:
            {
                'types': ['ICD-10', 'SNOMED'],
                'categories': ['respiratory'],
                'status': ['active'],
                'date_range': {
                    'start': '2023-01-01',
                    'end': '2023-12-31'
                }
            }
        """
        # Process date range if present
        date_range = None
        if 'date_range' in filter_dict:
            date_range = {}
            for key in ['start', 'end']:
                if key in filter_dict['date_range']:
                    date_str = filter_dict['date_range'][key]
                    try:
                        date_range[key] = datetime.fromisoformat(date_str)
                    except ValueError:
                        raise ValueError(f"Invalid date format for {key}: {date_str}")

        return FilterCriteria(
            types=filter_dict.get('types'),
            categories=filter_dict.get('categories'),
            status=filter_dict.get('status'),
            date_range=date_range,
            custom_filters=filter_dict.get('custom_filters')
        )
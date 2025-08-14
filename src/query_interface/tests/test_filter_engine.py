"""
Unit tests for FilterEngine component.
Tests filtering functionality and criteria handling.
"""
import pytest
from datetime import datetime
from ..filter_engine import FilterEngine, FilterCriteria
from ..search_index import IndexedCode


@pytest.fixture
def filter_engine():
    """Create FilterEngine instance for tests"""
    return FilterEngine()


@pytest.fixture
def sample_codes():
    """Sample medical codes for testing"""
    return [
        IndexedCode(
            code='J45.50',
            type='ICD-10',
            category='respiratory',
            description='Severe persistent asthma',
            status='active',
            metadata={'effective_date': '2023-01-01'}
        ),
        IndexedCode(
            code='E11.9',
            type='ICD-10',
            category='endocrine',
            description='Type 2 diabetes without complications',
            status='active',
            metadata={'effective_date': '2023-02-01'}
        ),
        IndexedCode(
            code='99213',
            type='CPT',
            category='evaluation',
            description='Office visit, established patient',
            status='active',
            metadata={'effective_date': '2023-03-01'}
        ),
        IndexedCode(
            code='R06.02',
            type='ICD-10',
            category='respiratory',
            description='Shortness of breath',
            status='inactive',
            metadata={'effective_date': '2022-01-01'}
        )
    ]


class TestFilterEngine:
    """Test cases for FilterEngine"""

    def test_filter_by_type(self, filter_engine, sample_codes):
        """Test filtering by code type"""
        criteria = FilterCriteria(types=['ICD-10'])
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 3
        assert all(code.type == 'ICD-10' for code in results)

    def test_filter_by_category(self, filter_engine, sample_codes):
        """Test filtering by category"""
        criteria = FilterCriteria(categories=['respiratory'])
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 2
        assert all(code.category == 'respiratory' for code in results)

    def test_filter_by_status(self, filter_engine, sample_codes):
        """Test filtering by status"""
        criteria = FilterCriteria(status=['active'])
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 3
        assert all(code.status == 'active' for code in results)

    def test_filter_by_date_range(self, filter_engine, sample_codes):
        """Test filtering by date range"""
        date_range = {
            'start': datetime(2023, 1, 1),
            'end': datetime(2023, 12, 31)
        }
        criteria = FilterCriteria(date_range=date_range)
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 3
        for code in results:
            date = datetime.fromisoformat(code.metadata['effective_date'])
            assert date_range['start'] <= date <= date_range['end']

    def test_multiple_filters(self, filter_engine, sample_codes):
        """Test applying multiple filters"""
        criteria = FilterCriteria(
            types=['ICD-10'],
            categories=['respiratory'],
            status=['active']
        )
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 1
        assert results[0].code == 'J45.50'
        assert results[0].type == 'ICD-10'
        assert results[0].category == 'respiratory'
        assert results[0].status == 'active'

    def test_no_matching_results(self, filter_engine, sample_codes):
        """Test filtering with no matching results"""
        criteria = FilterCriteria(types=['SNOMED'])  # No SNOMED codes in sample
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 0

    def test_empty_criteria(self, filter_engine, sample_codes):
        """Test filtering with empty criteria"""
        criteria = FilterCriteria()
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == len(sample_codes)
        assert set(results) == set(sample_codes)

    def test_invalid_date_format(self, filter_engine, sample_codes):
        """Test handling of invalid date format"""
        sample_codes[0].metadata['effective_date'] = 'invalid-date'
        
        date_range = {
            'start': datetime(2023, 1, 1),
            'end': datetime(2023, 12, 31)
        }
        criteria = FilterCriteria(date_range=date_range)
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        # Should include code with invalid date
        assert len(results) > 0
        assert any(code.code == 'J45.50' for code in results)

    def test_custom_filter(self, filter_engine, sample_codes):
        """Test adding and using custom filter"""
        def custom_description_filter(code: IndexedCode, keywords: list) -> bool:
            return any(keyword in code.description.lower() for keyword in keywords)
            
        filter_engine.add_custom_filter('description_keywords', custom_description_filter)
        
        criteria = FilterCriteria(
            custom_filters={'description_keywords': ['asthma', 'diabetes']}
        )
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 2
        assert any('asthma' in code.description.lower() for code in results)
        assert any('diabetes' in code.description.lower() for code in results)

    def test_filter_with_multiple_values(self, filter_engine, sample_codes):
        """Test filtering with multiple allowed values"""
        criteria = FilterCriteria(
            types=['ICD-10', 'CPT'],
            categories=['respiratory', 'evaluation']
        )
        results = filter_engine.apply_filters(sample_codes, criteria)
        
        assert len(results) == 3
        assert all(code.type in ['ICD-10', 'CPT'] for code in results)
        assert all(code.category in ['respiratory', 'evaluation'] for code in results)

    def test_create_filter_criteria(self, filter_engine):
        """Test creating FilterCriteria from dictionary"""
        filter_dict = {
            'types': ['ICD-10'],
            'categories': ['respiratory'],
            'status': ['active'],
            'date_range': {
                'start': '2023-01-01',
                'end': '2023-12-31'
            }
        }
        
        criteria = filter_engine.create_filter_criteria(filter_dict)
        
        assert isinstance(criteria, FilterCriteria)
        assert criteria.types == ['ICD-10']
        assert criteria.categories == ['respiratory']
        assert criteria.status == ['active']
        assert isinstance(criteria.date_range, dict)
        assert isinstance(criteria.date_range['start'], datetime)
        assert isinstance(criteria.date_range['end'], datetime)

    def test_invalid_filter_criteria(self, filter_engine):
        """Test handling of invalid filter criteria"""
        invalid_date_dict = {
            'date_range': {
                'start': 'invalid-date',
                'end': '2023-12-31'
            }
        }
        
        with pytest.raises(ValueError):
            filter_engine.create_filter_criteria(invalid_date_dict)
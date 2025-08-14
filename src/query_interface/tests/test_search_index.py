"""
Unit tests for SearchIndex component.
Tests indexing, searching, and retrieval functionality.
"""
import pytest
from ..search_index import SearchIndex, IndexedCode
from ..query_parser import QueryParser, QueryExpression, QueryOperator


@pytest.fixture
def search_index():
    """Create SearchIndex instance for tests"""
    return SearchIndex()


@pytest.fixture
def sample_codes():
    """Sample medical codes for testing"""
    return [
        {
            'code': 'J45.50',
            'type': 'ICD-10',
            'category': 'respiratory',
            'description': 'Severe persistent asthma',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        },
        {
            'code': 'E11.9',
            'type': 'ICD-10',
            'category': 'endocrine',
            'description': 'Type 2 diabetes without complications',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        },
        {
            'code': '99213',
            'type': 'CPT',
            'category': 'evaluation',
            'description': 'Office visit, established patient',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        }
    ]


@pytest.fixture
def populated_index(search_index, sample_codes):
    """SearchIndex populated with sample data"""
    for code_data in sample_codes:
        search_index.add_code(code_data)
    return search_index


@pytest.fixture
def parser():
    """Create QueryParser instance for tests"""
    return QueryParser()


class TestSearchIndex:
    """Test cases for SearchIndex"""

    def test_add_code(self, search_index):
        """Test adding a single code to the index"""
        code_data = {
            'code': 'J45.50',
            'type': 'ICD-10',
            'category': 'respiratory',
            'description': 'Severe persistent asthma',
            'status': 'active',
            'metadata': {}
        }
        
        search_index.add_code(code_data)
        
        # Verify code was indexed
        assert 'J45.50' in search_index._codes
        assert isinstance(search_index._codes['J45.50'], IndexedCode)
        
        # Check inverted index
        assert 'icd-10' in search_index._inverted_index['type_tokens']
        assert 'respiratory' in search_index._inverted_index['category_tokens']

    def test_exact_code_search(self, populated_index, parser):
        """Test searching for exact code match"""
        query = parser.parse('code:J45.50')
        results = populated_index.search(query)
        
        assert len(results) == 1
        assert results[0].code == 'J45.50'
        assert results[0].type == 'ICD-10'

    def test_category_search(self, populated_index, parser):
        """Test searching by category"""
        query = parser.parse('category:respiratory')
        results = populated_index.search(query)
        
        assert len(results) == 1
        assert results[0].category == 'respiratory'

    def test_type_search(self, populated_index, parser):
        """Test searching by code type"""
        query = parser.parse('type:ICD-10')
        results = populated_index.search(query)
        
        assert len(results) == 2  # Should find both ICD-10 codes
        assert all(code.type == 'ICD-10' for code in results)

    def test_description_search(self, populated_index, parser):
        """Test searching in description text"""
        query = parser.parse('description:asthma')
        results = populated_index.search(query)
        
        assert len(results) == 1
        assert 'asthma' in results[0].description.lower()

    def test_fuzzy_search(self, populated_index, parser):
        """Test fuzzy search functionality"""
        query = parser.parse('description:diabet~')
        results = populated_index.search(query)
        
        assert len(results) == 1
        assert 'diabetes' in results[0].description.lower()

    def test_boolean_search(self, populated_index, parser):
        """Test boolean search operations"""
        query = parser.parse('type:ICD-10 AND category:respiratory')
        results = populated_index.search(query)
        
        assert len(results) == 1
        assert results[0].type == 'ICD-10'
        assert results[0].category == 'respiratory'

    def test_partial_match(self, populated_index, parser):
        """Test partial text matching"""
        query = parser.parse('description:patient')
        results = populated_index.search(query)
        
        assert len(results) == 1
        assert 'patient' in results[0].description.lower()

    def test_case_insensitive_search(self, populated_index, parser):
        """Test case-insensitive searching"""
        query1 = parser.parse('type:ICD-10')
        query2 = parser.parse('type:icd-10')
        
        results1 = populated_index.search(query1)
        results2 = populated_index.search(query2)
        
        assert len(results1) == len(results2)
        assert {code.code for code in results1} == {code.code for code in results2}

    def test_multiple_results(self, populated_index, parser):
        """Test search returning multiple results"""
        query = parser.parse('status:active')
        results = populated_index.search(query)
        
        assert len(results) == 3  # All sample codes are active
        assert all(code.status == 'active' for code in results)

    def test_no_results(self, populated_index, parser):
        """Test search with no matching results"""
        query = parser.parse('category:nonexistent')
        results = populated_index.search(query)
        
        assert len(results) == 0

    def test_invalid_field(self, populated_index, parser):
        """Test search with invalid field name"""
        query = parser.parse('nonexistent_field:value')
        results = populated_index.search(query)
        
        assert len(results) == 0

    def test_empty_index_search(self, search_index, parser):
        """Test searching on empty index"""
        query = parser.parse('code:J45.50')
        results = search_index.search(query)
        
        assert len(results) == 0

    def test_tokenization(self, populated_index):
        """Test internal tokenization functionality"""
        tokens = populated_index._tokenize("Type 2 diabetes")
        
        assert isinstance(tokens, list)
        assert 'type' in tokens
        assert '2' in tokens
        assert 'diabetes' in tokens

    def test_update_existing_code(self, populated_index):
        """Test updating an existing code in the index"""
        updated_code = {
            'code': 'J45.50',
            'type': 'ICD-10',
            'category': 'respiratory',
            'description': 'Updated description',
            'status': 'inactive',
            'metadata': {}
        }
        
        populated_index.add_code(updated_code)
        
        # Verify update
        stored_code = populated_index._codes['J45.50']
        assert stored_code.description == 'Updated description'
        assert stored_code.status == 'inactive'
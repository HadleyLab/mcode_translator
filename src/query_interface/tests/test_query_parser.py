"""
Unit tests for QueryParser component.
Tests query parsing functionality and error handling.
"""
import pytest
from ..query_parser import QueryParser, QueryExpression, QueryTerm, QueryOperator


@pytest.fixture
def parser():
    """Create QueryParser instance for tests"""
    return QueryParser()


class TestQueryParser:
    """Test cases for QueryParser"""

    def test_simple_query(self, parser):
        """Test parsing of simple search query"""
        query = 'asthma'
        result = parser.parse(query)
        
        assert isinstance(result, QueryExpression)
        assert result.operator == QueryOperator.AND
        assert len(result.terms) == 1
        assert isinstance(result.terms[0], QueryTerm)
        assert result.terms[0].field is None
        assert result.terms[0].value == 'asthma'

    def test_field_query(self, parser):
        """Test parsing of field-specific query"""
        query = 'code:J45.50'
        result = parser.parse(query)
        
        assert isinstance(result, QueryExpression)
        assert len(result.terms) == 1
        assert result.terms[0].field == 'code'
        assert result.terms[0].value == 'J45.50'

    def test_exact_match_query(self, parser):
        """Test parsing of exact match query with quotes"""
        query = 'description:"severe persistent asthma"'
        result = parser.parse(query)
        
        assert isinstance(result, QueryExpression)
        assert len(result.terms) == 1
        assert result.terms[0].field == 'description'
        assert result.terms[0].value == 'severe persistent asthma'
        assert result.terms[0].is_exact == True

    def test_boolean_operators(self, parser):
        """Test parsing of queries with boolean operators"""
        query = 'code:J45.50 AND category:respiratory'
        result = parser.parse(query)
        
        assert result.operator == QueryOperator.AND
        assert len(result.terms) == 2
        assert result.terms[0].field == 'code'
        assert result.terms[1].field == 'category'

    def test_fuzzy_search(self, parser):
        """Test parsing of fuzzy search queries"""
        query = 'description:asthma~'
        result = parser.parse(query)
        
        assert result.terms[0].is_fuzzy == True
        assert result.terms[0].value == 'asthma'

    def test_multiple_terms(self, parser):
        """Test parsing of multiple search terms"""
        query = 'type:ICD-10 status:active category:respiratory'
        result = parser.parse(query)
        
        assert len(result.terms) == 3
        fields = [term.field for term in result.terms]
        assert 'type' in fields
        assert 'status' in fields
        assert 'category' in fields

    def test_invalid_query(self, parser):
        """Test handling of invalid query syntax"""
        invalid_queries = [
            'field:',  # Missing value
            ':value',  # Missing field
            'AND',     # Operator without terms
            '"unclosed quote'  # Unclosed quotation
        ]
        
        for query in invalid_queries:
            with pytest.raises(Exception):
                parser.parse(query)

    def test_complex_query(self, parser):
        """Test parsing of complex query with multiple operators"""
        query = 'type:ICD-10 AND (code:J45.50 OR code:J45.51) NOT status:inactive'
        result = parser.parse(query)
        
        assert result.operator == QueryOperator.AND
        assert len(result.terms) > 0

    def test_special_characters(self, parser):
        """Test handling of special characters in queries"""
        query = 'code:"J45.50+" description:"patient (severe)"'
        result = parser.parse(query)
        
        assert len(result.terms) == 2
        assert result.terms[0].value == 'J45.50+'
        assert result.terms[1].value == 'patient (severe)'

    def test_case_sensitivity(self, parser):
        """Test case sensitivity handling in queries"""
        query1 = 'TYPE:icd-10'
        query2 = 'type:ICD-10'
        
        result1 = parser.parse(query1)
        result2 = parser.parse(query2)
        
        assert result1.terms[0].field.lower() == result2.terms[0].field.lower()

    def test_whitespace_handling(self, parser):
        """Test handling of various whitespace patterns"""
        queries = [
            'type:ICD-10    category:respiratory',  # Multiple spaces
            '  type:ICD-10  ',  # Leading/trailing spaces
            'type:ICD-10\tcategory:respiratory'  # Tab character
        ]
        
        for query in queries:
            result = parser.parse(query)
            assert isinstance(result, QueryExpression)
            assert len(result.terms) > 0

    def test_empty_query(self, parser):
        """Test handling of empty or whitespace-only queries"""
        empty_queries = ['', '   ', '\t\n']
        
        for query in empty_queries:
            with pytest.raises(Exception):
                parser.parse(query)

    def test_to_dict_conversion(self, parser):
        """Test conversion of parsed query to dictionary format"""
        query = 'code:J45.50 AND category:respiratory'
        result = parser.parse(query)
        dict_result = parser.to_dict(result)
        
        assert isinstance(dict_result, dict)
        assert 'operator' in dict_result
        assert 'terms' in dict_result
        assert isinstance(dict_result['terms'], list)
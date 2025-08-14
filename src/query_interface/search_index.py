"""
Search Index module for efficient medical code searching.
Implements an inverted index structure for fast lookup and retrieval.
"""
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import re
from dataclasses import dataclass
from .query_parser import QueryExpression, QueryTerm, QueryOperator


@dataclass
class IndexedCode:
    """Represents an indexed medical code with its metadata"""
    code: str
    type: str
    category: str
    description: str
    status: str
    metadata: Dict[str, Any]
    
    def __hash__(self) -> int:
        """Make IndexedCode hashable for set operations"""
        return hash(self.code)
    
    def __eq__(self, other: object) -> bool:
        """Define equality for IndexedCode"""
        if not isinstance(other, IndexedCode):
            return NotImplemented
        return self.code == other.code


class SearchIndex:
    """Manages indexing and searching of medical codes"""
    
    def __init__(self):
        self._codes: Dict[str, IndexedCode] = {}
        self._inverted_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._initialize_index_structure()
        
    def _initialize_index_structure(self):
        """Initialize the index structure with supported fields"""
        self._indexed_fields = {
            'code', 'type', 'category', 'description', 'status'
        }
        
    def add_code(self, code_data: Dict) -> None:
        """
        Add a medical code to the search index
        
        Args:
            code_data: Dictionary containing code information
        """
        indexed_code = IndexedCode(
            code=code_data['code'],
            type=code_data['type'],
            category=code_data['category'],
            description=code_data['description'],
            status=code_data['status'],
            metadata=code_data.get('metadata', {})
        )
        
        self._codes[indexed_code.code] = indexed_code
        
        # Index each field
        for field in self._indexed_fields:
            value = getattr(indexed_code, field)
            if isinstance(value, str):
                # Index full value
                self._add_to_index(field, value.lower(), indexed_code.code)
                
                # Index tokens
                tokens = self._tokenize(value)
                for token in tokens:
                    self._add_to_index(f"{field}_tokens", token.lower(), indexed_code.code)
    
    def _add_to_index(self, field: str, value: str, code: str) -> None:
        """Add a value to the inverted index for a field"""
        self._inverted_index[field][value].add(code)
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens for indexing"""
        return [token.lower() for token in re.findall(r'\w+', text)]
    
    def search(self, query: QueryExpression) -> List[IndexedCode]:
        """
        Search the index using a parsed query expression
        
        Args:
            query: QueryExpression from QueryParser
            
        Returns:
            List of matching IndexedCode objects
        """
        results = self._evaluate_expression(query)
        return [self._codes[code] for code in results]
    
    def _evaluate_expression(self, expr: QueryExpression) -> Set[str]:
        """Recursively evaluate a query expression"""
        results = []
        
        for term in expr.terms:
            if isinstance(term, QueryExpression):
                term_results = self._evaluate_expression(term)
            else:
                term_results = self._evaluate_term(term)
            results.append(term_results)
            
        if not results:
            return set()
            
        # Combine results based on operator
        if expr.operator == QueryOperator.AND:
            final_results = results[0]
            for r in results[1:]:
                final_results &= r
        elif expr.operator == QueryOperator.OR:
            final_results = results[0]
            for r in results[1:]:
                final_results |= r
        else:  # NOT
            all_codes = set(self._codes.keys())
            final_results = all_codes - results[0]
            
        return final_results
    
    def _evaluate_term(self, term: QueryTerm) -> Set[str]:
        """Evaluate a single query term"""
        if not term.field:
            # Search all fields
            results = set()
            for field in self._indexed_fields:
                results |= self._search_field(field, term.value, term.is_exact, term.is_fuzzy)
            return results
        else:
            return self._search_field(term.field, term.value, term.is_exact, term.is_fuzzy)
    
    def _search_field(self, field: str, value: str, is_exact: bool, is_fuzzy: bool) -> Set[str]:
        """Search within a specific field"""
        value = value.lower()
        
        # Try exact field match first
        if value in self._inverted_index[field]:
            return self._inverted_index[field][value]
        
        # For non-exact matches, search tokens
        token_field = f"{field}_tokens"
        results = set()
        
        if is_fuzzy:
            # Fuzzy matching using token similarity
            value_tokens = self._tokenize(value)
            for indexed_token in self._inverted_index[token_field]:
                if any(self._is_similar(vt, indexed_token) for vt in value_tokens):
                    results |= self._inverted_index[token_field][indexed_token]
        else:
            # Partial matching
            value_tokens = self._tokenize(value)
            for token in value_tokens:
                for indexed_token in self._inverted_index[token_field]:
                    if token in indexed_token:
                        results |= self._inverted_index[token_field][indexed_token]
        
        return results
    
    def _is_similar(self, token1: str, token2: str, threshold: float = 0.8) -> bool:
        """
        Check if two tokens are similar using Levenshtein distance
        
        Args:
            token1: First token
            token2: Second token
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if tokens are similar enough
        """
        if not token1 or not token2:
            return False
            
        # Simple implementation - could be improved with proper Levenshtein distance
        shorter = token1 if len(token1) <= len(token2) else token2
        longer = token2 if len(token1) <= len(token2) else token1
        
        if len(longer) == 0:
            return True
            
        # Calculate similarity based on common characters
        common_chars = sum(1 for c in shorter if c in longer)
        similarity = common_chars / len(longer)
        
        return similarity >= threshold
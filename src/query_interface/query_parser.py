"""
Query Parser module for processing search queries into structured formats.
Handles boolean operations, field-specific searches, and advanced query features.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from enum import Enum


class QueryOperator(Enum):
    """Supported query operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class QueryTerm:
    """Represents a single search term with field and value"""
    field: Optional[str]
    value: str
    is_exact: bool = False
    is_fuzzy: bool = False
    proximity: Optional[int] = None


@dataclass
class QueryExpression:
    """Represents a complex query expression with operators"""
    operator: QueryOperator
    terms: List[Union['QueryExpression', QueryTerm]]


class QueryParser:
    """Parses search queries into structured format for processing"""

    def __init__(self):
        self.special_chars = {':', '"', '~', '*', '[', ']', '(', ')', ' '}
        self.operators = {op.value for op in QueryOperator}

    def parse(self, query_string: str) -> QueryExpression:
        """
        Parse a query string into a structured QueryExpression.
        
        Args:
            query_string: Raw query string to parse
            
        Returns:
            QueryExpression representing the parsed query
        
        Raises:
            ValueError: If query syntax is invalid
        """
        if not query_string or query_string.isspace():
            raise ValueError("Empty or whitespace-only query")

        tokens = self._tokenize(query_string)
        return self._parse_tokens(tokens)

    def _tokenize(self, query_string: str) -> List[str]:
        """Break query string into tokens"""
        tokens = []
        current_token = []
        in_quotes = False
        
        i = 0
        while i < len(query_string):
            char = query_string[i]
            
            # Handle quoted strings
            if char == '"':
                if in_quotes:
                    # End quote - add entire quoted string as one token
                    current_token.append(char)
                    tokens.append(''.join(current_token))
                    current_token = []
                    in_quotes = False
                else:
                    # Start quote
                    if current_token:
                        tokens.append(''.join(current_token))
                        current_token = []
                    current_token.append(char)
                    in_quotes = True
                i += 1
                continue

            if in_quotes:
                current_token.append(char)
                i += 1
                continue

            # Handle operators
            if char == ' ':
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                i += 1
                continue

            # Handle field:value pairs
            if char == ':':
                if current_token:
                    field = ''.join(current_token)
                    current_token = []
                    # Look ahead for value
                    i += 1
                    while i < len(query_string) and query_string[i] not in {' ', ':', '"'}:
                        if query_string[i] == '~':  # Fuzzy indicator
                            break
                        current_token.append(query_string[i])
                        i += 1
                    value = ''.join(current_token)
                    # Check for fuzzy search
                    is_fuzzy = i < len(query_string) and query_string[i] == '~'
                    if is_fuzzy:
                        i += 1
                    tokens.append(f"{field}:{value}{'~' if is_fuzzy else ''}")
                    current_token = []
                continue

            current_token.append(char)
            i += 1

        if current_token:
            tokens.append(''.join(current_token))

        if in_quotes:
            raise ValueError("Unclosed quotation mark")

        return tokens

    def _parse_tokens(self, tokens: List[str]) -> QueryExpression:
        """Convert tokens into QueryExpression structure"""
        terms: List[Union[QueryExpression, QueryTerm]] = []
        current_operator = QueryOperator.AND  # Default operator
        
        i = 0
        while i < len(tokens):
            token = tokens[i].strip()
            
            # Skip empty tokens
            if not token:
                i += 1
                continue
                
            # Handle operators
            if token in self.operators:
                current_operator = QueryOperator(token)
                i += 1
                continue
                
            # Handle field:value pairs
            if ':' in token:
                field, value = token.rsplit(':', 1)
                is_fuzzy = value.endswith('~')
                if is_fuzzy:
                    value = value[:-1]
                
                # Handle quoted values
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                    is_exact = True
                else:
                    is_exact = False
                    
                terms.append(QueryTerm(
                    field=field,
                    value=value,
                    is_exact=is_exact,
                    is_fuzzy=is_fuzzy
                ))
            else:
                # Handle quoted terms
                if token.startswith('"') and token.endswith('"'):
                    value = token[1:-1]
                    is_exact = True
                else:
                    value = token
                    is_exact = False
                
                terms.append(QueryTerm(
                    field=None,
                    value=value,
                    is_exact=is_exact
                ))
            
            i += 1
            
        if not terms:
            raise ValueError("No valid search terms found")
            
        return QueryExpression(operator=current_operator, terms=terms)

    def to_dict(self, expression: QueryExpression) -> Dict:
        """Convert QueryExpression to dictionary format"""
        return {
            "operator": expression.operator.value,
            "terms": [
                self.to_dict(term) if isinstance(term, QueryExpression)
                else {
                    "field": term.field,
                    "value": term.value,
                    "is_exact": term.is_exact,
                    "is_fuzzy": term.is_fuzzy,
                    "proximity": term.proximity
                }
                for term in expression.terms
            ]
        }
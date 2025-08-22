#!/usr/bin/env python3
"""
Analyze NCT03879174 eligibility criteria for biomarkers and genomic variants
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data_fetcher.fetcher import get_full_study
import re

def analyze_criteria():
    """Analyze the eligibility criteria for NCT03879174"""
    study = get_full_study('NCT03879174')
    criteria = study['protocolSection']['eligibilityModule']['eligibilityCriteria']
    
    print("=== NCT03879174 Eligibility Criteria Analysis ===")
    print(f"Total length: {len(criteria)} characters")
    print()
    
    # Search for key terms
    terms_to_search = [
        'hormone receptor', 'HR', 'ER', 'PR', 'HER2', 'ESR1', 'estrogen receptor',
        'progesterone receptor', 'biomarker', 'mutation', 'genomic', 'variant'
    ]
    
    for term in terms_to_search:
        matches = re.findall(rf'\b{term}\b', criteria, re.IGNORECASE)
        if matches:
            print(f"{term.upper()}: {len(matches)} mentions")
    
    print()
    print("=== Sample context around key terms ===")
    
    # Show context around key terms
    lines = criteria.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(term in line_lower for term in ['hormone receptor', 'esr1', 'er+', 'pr+', 'her2']):
            print(f"Line {i}: {line.strip()}")
            # Show surrounding context
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if j != i:
                    print(f"  {j}: {lines[j].strip()}")
            print()
    
    # Search for specific inclusion criteria numbers that might contain biomarker info
    print("=== Searching for specific inclusion criteria ===")
    for i, line in enumerate(lines):
        if line.strip().startswith(('7.', '8.', '9.', '10.', '11.', '12.')):
            line_lower = line.lower()
            if any(term in line_lower for term in ['cancer', 'tumor', 'metastatic', 'advanced', 'stage']):
                print(f"Inclusion {line.strip()}")
                # Show next few lines for context
                for j in range(i+1, min(len(lines), i+6)):
                    if lines[j].strip() and not lines[j].strip().startswith(tuple(str(x) + '.' for x in range(1, 20))):
                        print(f"  {lines[j].strip()}")
                    else:
                        break
                print()

if __name__ == "__main__":
    analyze_criteria()
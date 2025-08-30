"""Configuration of regex patterns for clinical text processing.

Contains all regular expression patterns used for pattern matching,
organized by category for better maintainability.
"""

import re

# Biomarker patterns
BIOMARKER_PATTERNS = {
    'ER': re.compile(r'ER\s*(?:status)?\s*[:=]?\s*(positive|negative|\d+\s*%|\+|\-)', re.IGNORECASE),
    'PR': re.compile(r'PR\s*(?:status)?\s*[:=]?\s*(positive|negative|\d+\s*%|\+|\-)', re.IGNORECASE),
    'HER2': re.compile(r'HER2\s*(?:status)?\s*[:=]?\s*(positive|negative|\d+\+?)', re.IGNORECASE),
    'PD-L1': re.compile(r'PD-?L1\s*(?:status)?\s*[:=]?\s*(positive|negative|\d+\s*%)', re.IGNORECASE),
    'Ki-67': re.compile(r'Ki-?67\s*(?:status)?\s*[:=]?\s*(positive|negative|\d+\s*%)', re.IGNORECASE),
    'MSI': re.compile(r'MSI\s*(?:status)?\s*[:=]?\s*(high|low|stable)', re.IGNORECASE),
    'TMB': re.compile(r'TMB\s*(?:status)?\s*[:=]?\s*(high|low|\d+\s*mut/Mb)', re.IGNORECASE)
}

# Genomic variant patterns
GENE_PATTERN = re.compile(
    r'\b(BRCA[12]|TP53|PIK3CA|PTEN|AKT1|ERBB2|HER2|EGFR|ALK|ROS1|KRAS|NRAS|BRAF|MEK[12]?|NTRK[123])\b',
    re.IGNORECASE
)

VARIANT_PATTERN = re.compile(
    r'\b([A-Z0-9]+)\s*(?:mutation|variant|alteration|amplification|fusion|rearrangement|deletion|insertion)\b',
    re.IGNORECASE
)

COMPLEX_VARIANT_PATTERN = re.compile(
    r'\b([A-Z0-9]+)\s*(?:p\.)?([A-Z][a-z]{2}[0-9]+(?:[A-Za-z]|\*)?)\b',
    re.IGNORECASE
)

# Cancer condition patterns
STAGE_PATTERN = re.compile(r'stage\s+(I{1,3}V?|IV)', re.IGNORECASE)
CANCER_TYPE_PATTERN = re.compile(r'\b(breast|lung|colorectal)\s+cancer\b', re.IGNORECASE)

# General condition patterns
CONDITION_PATTERN = re.compile(r'\b(diabetes|hypertension|heart disease)\b', re.IGNORECASE)

# Performance status patterns
ECOG_PATTERN = re.compile(r'ECOG\s+status?\s*[:=]?\s*([0-4])', re.IGNORECASE)

# Demographic patterns
GENDER_PATTERN = re.compile(r'\b(male|female)\b', re.IGNORECASE)
AGE_PATTERN = re.compile(r'age\s+([0-9]+)\s+to\s+([0-9]+)', re.IGNORECASE)
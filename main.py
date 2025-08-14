#!/usr/bin/env python3
"""
mCODE Translator - Clinical Trial Data Fetcher

This script provides a command-line interface to fetch clinical trial data
from clinicaltrials.gov API and process it for mCODE translation.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.fetcher import main

if __name__ == '__main__':
    main()
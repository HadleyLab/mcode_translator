#!/usr/bin/env python3
"""
Unit tests for the OutputFormatter
"""

import sys
import os
import unittest

# Add src directory to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.output_formatter import OutputFormatter


class TestOutputFormatter(unittest.TestCase):
    """
    Unit tests for the OutputFormatter
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.formatter = OutputFormatter()
    
    def test_to_json(self):
        """
        Test converting data to JSON format
        """
        data = {
            "resourceType": "Patient",
            "gender": "female"
        }
        
        result = self.formatter.to_json(data)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that it's valid JSON
        import json
        parsed = json.loads(result)
        self.assertEqual(parsed["resourceType"], "Patient")
        self.assertEqual(parsed["gender"], "female")
    
    def test_to_xml(self):
        """
        Test converting data to XML format
        """
        data = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-1",
                        "gender": "female"
                    }
                }
            ]
        }
        
        result = self.formatter.to_xml(data)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that it contains expected XML elements
        self.assertIn("<Bundle", result)
        self.assertIn("xmlns", result)
        self.assertIn("<gender value=\"female\" />", result)
    
    def test_format_validation_report(self):
        """
        Test formatting validation results into a report
        """
        validation_results = {
            "valid": True,
            "compliance_score": 0.95,
            "quality_metrics": {
                "completeness": 0.9,
                "accuracy": 0.95,
                "consistency": 0.98
            },
            "errors": [],
            "warnings": ["Minor issue with data formatting"]
        }
        
        result = self.formatter.format_validation_report(validation_results)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that it contains expected content
        self.assertIn("PASSED", result)
        self.assertIn("Compliance Score: 0.95", result)
        self.assertIn("Minor issue with data formatting", result)
    
    def test_format_resource_summary(self):
        """
        Test formatting a resource summary
        """
        resources = [
            {"resourceType": "Patient"},
            {"resourceType": "Condition"},
            {"resourceType": "Condition"},
            {"resourceType": "Procedure"}
        ]
        
        result = self.formatter.format_resource_summary(resources)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that it contains expected content
        self.assertIn("Patient: 1", result)
        self.assertIn("Condition: 2", result)
        self.assertIn("Procedure: 1", result)


if __name__ == '__main__':
    unittest.main()
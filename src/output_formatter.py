#!/usr/bin/env python3
"""
Output Formatter for mCODE Translator
Formats generated mCODE data into various output formats (JSON, XML, etc.)
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Output Formatter for mCODE Translator
    Handles formatting of generated mCODE data into various output formats
    """
    
    def __init__(self):
        """
        Initialize the Output Formatter
        """
        logger.info("Output Formatter initialized")
    
    def to_json(self, data: Dict[str, Any], indent: int = 2) -> str:
        """
        Convert data to JSON format
        
        Args:
            data: Data to convert to JSON
            indent: Number of spaces for indentation (default: 2)
            
        Returns:
            JSON string representation of the data
        """
        try:
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error converting data to JSON: {str(e)}")
            raise
    
    def to_xml(self, data: Dict[str, Any]) -> str:
        """
        Convert FHIR resource data to XML format
        
        Args:
            data: FHIR resource data to convert
            
        Returns:
            XML string representation
        """
        try:
            if data.get("resourceType") != "Bundle":
                # If not a bundle, wrap in bundle
                data = self._create_bundle([data])
            
            # Create root element
            root = ET.Element("Bundle")
            root.set("xmlns", "http://hl7.org/fhir")
            
            # Add id
            if "id" in data:
                id_elem = ET.SubElement(root, "id")
                id_elem.set("value", data["id"])
            
            # Add type
            if "type" in data:
                type_elem = ET.SubElement(root, "type")
                type_elem.set("value", data["type"])
            
            # Add entries
            if "entry" in data:
                for entry in data["entry"]:
                    if "resource" in entry:
                        entry_elem = ET.SubElement(root, "entry")
                        self._add_resource_xml(entry_elem, entry["resource"])
            
            # Convert to string
            return ET.tostring(root, encoding="unicode")
        except Exception as e:
            logger.error(f"Error converting data to XML: {str(e)}")
            raise
    
    def _add_resource_xml(self, parent: ET.Element, resource: Dict[str, Any]) -> None:
        """
        Add a resource to XML parent element
        
        Args:
            parent: Parent XML element
            resource: Resource data to add
        """
        resource_elem = ET.SubElement(parent, "resource")
        
        # Add resourceType
        if "resourceType" in resource:
            resource_type_elem = ET.SubElement(resource_elem, "resourceType")
            resource_type_elem.set("value", resource["resourceType"])
        
        # Add id
        if "id" in resource:
            id_elem = ET.SubElement(resource_elem, "id")
            id_elem.set("value", resource["id"])
        
        # Add other elements based on resource type
        if resource.get("resourceType") == "Patient":
            self._add_patient_xml_elements(resource_elem, resource)
        elif resource.get("resourceType") == "Condition":
            self._add_condition_xml_elements(resource_elem, resource)
        elif resource.get("resourceType") == "Procedure":
            self._add_procedure_xml_elements(resource_elem, resource)
        elif resource.get("resourceType") == "MedicationStatement":
            self._add_medication_statement_xml_elements(resource_elem, resource)
    
    def _add_patient_xml_elements(self, parent: ET.Element, patient: Dict[str, Any]) -> None:
        """
        Add Patient-specific XML elements to parent
        
        Args:
            parent: Parent XML element
            patient: Patient resource data
        """
        # Add gender
        if "gender" in patient:
            gender_elem = ET.SubElement(parent, "gender")
            gender_elem.set("value", patient["gender"])
        
        # Add birthDate
        if "birthDate" in patient:
            birth_date_elem = ET.SubElement(parent, "birthDate")
            birth_date_elem.set("value", patient["birthDate"])
    
    def _add_condition_xml_elements(self, parent: ET.Element, condition: Dict[str, Any]) -> None:
        """
        Add Condition-specific XML elements to parent
        
        Args:
            parent: Parent XML element
            condition: Condition resource data
        """
        # Add code
        if "code" in condition:
            code_elem = ET.SubElement(parent, "code")
            self._add_codeable_concept_xml_elements(code_elem, condition["code"])
    
    def _add_procedure_xml_elements(self, parent: ET.Element, procedure: Dict[str, Any]) -> None:
        """
        Add Procedure-specific XML elements to parent
        
        Args:
            parent: Parent XML element
            procedure: Procedure resource data
        """
        # Add code
        if "code" in procedure:
            code_elem = ET.SubElement(parent, "code")
            self._add_codeable_concept_xml_elements(code_elem, procedure["code"])
    
    def _add_medication_statement_xml_elements(self, parent: ET.Element, medication: Dict[str, Any]) -> None:
        """
        Add MedicationStatement-specific XML elements to parent
        
        Args:
            parent: Parent XML element
            medication: MedicationStatement resource data
        """
        # Add medicationCodeableConcept
        if "medicationCodeableConcept" in medication:
            med_elem = ET.SubElement(parent, "medicationCodeableConcept")
            self._add_codeable_concept_xml_elements(med_elem, medication["medicationCodeableConcept"])
    
    def _add_codeable_concept_xml_elements(self, parent: ET.Element, concept: Dict[str, Any]) -> None:
        """
        Add CodeableConcept XML elements to parent
        
        Args:
            parent: Parent XML element
            concept: CodeableConcept data
        """
        if "coding" in concept:
            coding_elem = ET.SubElement(parent, "coding")
            for coding in concept["coding"]:
                coding_item_elem = ET.SubElement(coding_elem, "coding")
                
                # Add system
                if "system" in coding:
                    system_elem = ET.SubElement(coding_item_elem, "system")
                    system_elem.set("value", coding["system"])
                
                # Add code
                if "code" in coding:
                    code_elem = ET.SubElement(coding_item_elem, "code")
                    code_elem.set("value", coding["code"])
                
                # Add display
                if "display" in coding:
                    display_elem = ET.SubElement(coding_item_elem, "display")
                    display_elem.set("value", coding["display"])
    
    def _create_bundle(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a FHIR Bundle containing the provided resources
        
        Args:
            resources: List of FHIR resources
            
        Returns:
            Bundle resource dictionary
        """
        import uuid
        
        bundle = {
            "resourceType": "Bundle",
            "id": f"bundle-{uuid.uuid4()}",
            "type": "collection",
            "entry": []
        }
        
        for resource in resources:
            bundle["entry"].append({
                "resource": resource
            })
        
        return bundle
    
    def format_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Format validation results into a human-readable report
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("mCODE Validation Report")
        report.append("=" * 30)
        
        # Overall status
        status = "PASSED" if validation_results.get("valid", False) else "FAILED"
        report.append(f"Overall Status: {status}")
        
        # Compliance score
        compliance_score = validation_results.get("compliance_score", 0)
        report.append(f"Compliance Score: {compliance_score:.2f}")
        
        # Quality metrics
        if "quality_metrics" in validation_results:
            metrics = validation_results["quality_metrics"]
            report.append("\nQuality Metrics:")
            report.append(f"  Completeness: {metrics.get('completeness', 0):.2f}")
            report.append(f"  Accuracy: {metrics.get('accuracy', 0):.2f}")
            report.append(f"  Consistency: {metrics.get('consistency', 0):.2f}")
        
        # Errors
        if validation_results.get("errors"):
            report.append("\nErrors:")
            for error in validation_results["errors"]:
                report.append(f"  - {error}")
        
        # Warnings
        if validation_results.get("warnings"):
            report.append("\nWarnings:")
            for warning in validation_results["warnings"]:
                report.append(f"  - {warning}")
        
        return "\n".join(report)
    
    def format_resource_summary(self, resources: List[Dict[str, Any]]) -> str:
        """
        Format a summary of generated resources
        
        Args:
            resources: List of FHIR resources
            
        Returns:
            Formatted resource summary string
        """
        summary = []
        summary.append("Generated Resources Summary")
        summary.append("=" * 30)
        
        # Count resources by type
        resource_counts = {}
        for resource in resources:
            resource_type = resource.get("resourceType", "Unknown")
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        # Display counts
        for resource_type, count in resource_counts.items():
            summary.append(f"{resource_type}: {count}")
        
        return "\n".join(summary)


    def format_test_summary(self, test_results: Dict[str, Any]) -> str:
        """
        Format test results into a comprehensive summary with engine performance metrics
        
        Args:
            test_results: Dictionary containing test results and engine metrics
            
        Returns:
            Formatted test summary string
        """
        summary = []
        summary.append("============================================================")
        summary.append("TEST SUITE SUMMARY")
        summary.append("============================================================")
        
        # Engine Performance Section
        summary.append("\nEngine Performance Metrics:")
        summary.append("---------------------------")
        
        if 'llm_engine' in test_results:
            llm = test_results['llm_engine']
            summary.append(f"LLM NLP Engine:")
            summary.append(f"- Processing time: {llm.get('processing_time', 0):.4f} seconds")
            summary.append(f"- API response time: {llm.get('api_response_time', 0):.4f} seconds")
            summary.append(f"- Entities extracted: {len(llm.get('entities', []))}")
            summary.append(f"- Average confidence: {llm.get('avg_confidence', 0):.2f}")
        
        if 'regex_engine' in test_results:
            regex = test_results['regex_engine']
            summary.append(f"\nRegex NLP Engine:")
            summary.append(f"- Processing time: {regex.get('processing_time', 0):.4f} seconds")
            summary.append(f"- Conditions found: {regex.get('conditions_found', 0)}")
            summary.append(f"- Procedures found: {regex.get('procedures_found', 0)}")
        
        if 'spacy_engine' in test_results:
            spacy = test_results['spacy_engine']
            summary.append(f"\nSpaCy NLP Engine:")
            summary.append(f"- Processing time: {spacy.get('processing_time', 0):.4f} seconds")
            summary.append(f"- Entities found: {spacy.get('entities_found', 0)}")
            summary.append(f"- Average confidence: {spacy.get('avg_confidence', 0):.2f}")

        # Test Summary Section
        summary.append("\n============================================================")
        summary.append(f"Total tests run: {test_results.get('total', 0)}")
        summary.append(f"Failures: {test_results.get('failures', 0)}")
        summary.append(f"Errors: {test_results.get('errors', 0)}")
        summary.append("\nAll tests passed!" if test_results.get('failures', 1) == 0 else "\nTests failed!")
        return "\n".join(summary)


# Example usage
if __name__ == "__main__":
    # This is just for testing purposes
    formatter = OutputFormatter()
    
    # Sample data
    sample_data = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "gender": "female",
                    "birthDate": "1950-01-01"
                }
            }
        ]
    }
    
    # Test JSON output
    json_output = formatter.to_json(sample_data)
    print("JSON Output:")
    print(json_output)
    
    # Test XML output
    xml_output = formatter.to_xml(sample_data)
    print("\nXML Output:")
    print(xml_output)
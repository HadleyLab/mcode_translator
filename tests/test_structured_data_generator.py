import unittest
import sys
import os

# Add src directory to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.structured_data_generator import StructuredDataGenerator

class TestStructuredDataGenerator(unittest.TestCase):
    """
    Unit tests for the StructuredDataGenerator
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.generator = StructuredDataGenerator()
    
    def test_generate_patient_resource(self):
        """
        Test generating Patient resource
        """
        demographics = {
            "gender": "female",
            "age": "55",
            "ethnicity": "hispanic-or-latino"
        }
        
        result = self.generator.generate_patient_resource(demographics)
        
        # Check basic structure
        self.assertEqual(result["resourceType"], "Patient")
        self.assertIn("id", result)
        self.assertIn("meta", result)
        self.assertIn("profile", result["meta"])
        self.assertIn(self.generator.mcode_profiles["Patient"], result["meta"]["profile"])
        
        # Check gender
        self.assertEqual(result["gender"], "female")
        
        # Check extension
        self.assertIn("extension", result)
        self.assertEqual(len(result["extension"]), 1)
        self.assertEqual(result["extension"][0]["url"], 
                         "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity")
    
    def test_generate_condition_resource(self):
        """
        Test generating Condition resource
        """
        condition_data = {
            "mcode_element": "Condition",
            "primary_code": {"system": "ICD10CM", "code": "C50.911"},
            "mapped_codes": {"SNOMEDCT": "254837009"}
        }
        
        result = self.generator.generate_condition_resource(condition_data)
        
        # Check basic structure
        self.assertEqual(result["resourceType"], "Condition")
        self.assertIn("id", result)
        self.assertIn("meta", result)
        self.assertIn("profile", result["meta"])
        self.assertIn(self.generator.mcode_profiles["Condition"], result["meta"]["profile"])
        
        # Check code
        self.assertIn("code", result)
        self.assertIn("coding", result["code"])
        self.assertEqual(len(result["code"]["coding"]), 1)
        self.assertEqual(result["code"]["coding"][0]["system"], 
                         self.generator.system_uris["ICD10CM"])
        self.assertEqual(result["code"]["coding"][0]["code"], "C50.911")
        
        # Check bodySite
        self.assertIn("bodySite", result)
        self.assertIn("coding", result["bodySite"])
        self.assertEqual(len(result["bodySite"]["coding"]), 1)
        self.assertEqual(result["bodySite"]["coding"][0]["system"], 
                         self.generator.system_uris["SNOMEDCT"])
        self.assertEqual(result["bodySite"]["coding"][0]["code"], "254837009")
    
    def test_generate_procedure_resource(self):
        """
        Test generating Procedure resource
        """
        procedure_data = {
            "mcode_element": "Procedure",
            "primary_code": {"system": "CPT", "code": "12345"},
            "mapped_codes": {}
        }
        
        result = self.generator.generate_procedure_resource(procedure_data)
        
        # Check basic structure
        self.assertEqual(result["resourceType"], "Procedure")
        self.assertIn("id", result)
        self.assertIn("meta", result)
        self.assertIn("profile", result["meta"])
        self.assertIn(self.generator.mcode_profiles["Procedure"], result["meta"]["profile"])
        
        # Check code
        self.assertIn("code", result)
        self.assertIn("coding", result["code"])
        self.assertEqual(len(result["code"]["coding"]), 1)
        self.assertEqual(result["code"]["coding"][0]["system"], 
                         self.generator.system_uris["CPT"])
        self.assertEqual(result["code"]["coding"][0]["code"], "12345")
    
    def test_generate_medication_statement_resource(self):
        """
        Test generating MedicationStatement resource
        """
        medication_data = {
            "mcode_element": "MedicationStatement",
            "primary_code": {"system": "RxNorm", "code": "123456"},
            "mapped_codes": {}
        }
        
        result = self.generator.generate_medication_statement_resource(medication_data)
        
        # Check basic structure
        self.assertEqual(result["resourceType"], "MedicationStatement")
        self.assertIn("id", result)
        self.assertIn("meta", result)
        self.assertIn("profile", result["meta"])
        self.assertIn(self.generator.mcode_profiles["MedicationStatement"], result["meta"]["profile"])
        
        # Check medicationCodeableConcept
        self.assertIn("medicationCodeableConcept", result)
        self.assertIn("coding", result["medicationCodeableConcept"])
        self.assertEqual(len(result["medicationCodeableConcept"]["coding"]), 1)
        self.assertEqual(result["medicationCodeableConcept"]["coding"][0]["system"], 
                         self.generator.system_uris["RxNorm"])
        self.assertEqual(result["medicationCodeableConcept"]["coding"][0]["code"], "123456")
    
    def test_generate_observation_resource(self):
        """
        Test generating Observation resource
        """
        observation_data = {
            "mcode_element": "Observation",
            "primary_code": {"system": "LOINC", "code": "12345-6"},
            "mapped_codes": {},
            "value": 42.0
        }
        
        result = self.generator.generate_observation_resource(observation_data)
        
        # Check basic structure
        self.assertEqual(result["resourceType"], "Observation")
        self.assertIn("id", result)
        self.assertIn("meta", result)
        self.assertIn("profile", result["meta"])
        self.assertIn(self.generator.mcode_profiles["Observation"], result["meta"]["profile"])
        
        # Check code
        self.assertIn("code", result)
        self.assertIn("coding", result["code"])
        self.assertEqual(len(result["code"]["coding"]), 1)
        self.assertEqual(result["code"]["coding"][0]["system"], 
                         self.generator.system_uris["LOINC"])
        self.assertEqual(result["code"]["coding"][0]["code"], "12345-6")
        
        # Check value
        self.assertIn("valueQuantity", result)
        self.assertEqual(result["valueQuantity"]["value"], 42.0)
    
    def test_create_bundle(self):
        """
        Test creating FHIR Bundle
        """
        resources = [
            {"resourceType": "Patient", "id": "patient-1"},
            {"resourceType": "Condition", "id": "condition-1"}
        ]
        
        result = self.generator.create_bundle(resources)
        
        # Check basic structure
        self.assertEqual(result["resourceType"], "Bundle")
        self.assertIn("id", result)
        self.assertEqual(result["type"], "collection")
        self.assertIn("entry", result)
        self.assertEqual(len(result["entry"]), 2)
        
        # Check entries
        for i, entry in enumerate(result["entry"]):
            self.assertIn("resource", entry)
            self.assertEqual(entry["resource"]["id"], resources[i]["id"])
    
    def test_validate_patient_resource(self):
        """
        Test validating Patient resource
        """
        # Valid patient resource
        patient = {
            "resourceType": "Patient",
            "gender": "female"
        }
        
        result = self.generator.validate_resource(patient)
        
        self.assertTrue(result["valid"])
        self.assertEqual(result["resource_type"], "Patient")
        self.assertEqual(len(result["errors"]), 0)
        
        # Invalid patient resource (missing gender)
        patient_invalid = {
            "resourceType": "Patient"
        }
        
        result = self.generator.validate_resource(patient_invalid)
        
        self.assertFalse(result["valid"])
        self.assertIn("Missing required element: gender", result["errors"])
    
    def test_validate_condition_resource(self):
        """
        Test validating Condition resource
        """
        # Valid condition resource
        condition = {
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "C50.911"}
                ]
            },
            "bodySite": {
                "coding": [
                    {"system": "http://snomed.info/sct", "code": "254837009"}
                ]
            }
        }
        
        result = self.generator.validate_resource(condition)
        
        self.assertTrue(result["valid"])
        self.assertEqual(result["resource_type"], "Condition")
        self.assertEqual(len(result["errors"]), 0)
        
        # Invalid condition resource (missing code)
        condition_invalid = {
            "resourceType": "Condition",
            "bodySite": {
                "coding": [
                    {"system": "http://snomed.info/sct", "code": "254837009"}
                ]
            }
        }
        
        result = self.generator.validate_resource(condition_invalid)
        
        self.assertFalse(result["valid"])
        self.assertIn("Missing required element: code", result["errors"])
    
    def test_validate_bundle(self):
        """
        Test validating FHIR Bundle
        """
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "gender": "female"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [
                                {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "C50.911"}
                            ]
                        },
                        "bodySite": {
                            "coding": [
                                {"system": "http://snomed.info/sct", "code": "254837009"}
                            ]
                        }
                    }
                }
            ]
        }
        
        result = self.generator.validate_bundle(bundle)
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["resource_validations"]), 2)
        self.assertEqual(result["compliance_score"], 1.0)
    
    def test_generate_mcode_resources(self):
        """
        Test generating mCODE FHIR resources
        """
        mapped_elements = [
            {
                "mcode_element": "Condition",
                "primary_code": {"system": "ICD10CM", "code": "C50.911"},
                "mapped_codes": {"SNOMEDCT": "254837009"}
            },
            {
                "mcode_element": "MedicationStatement",
                "primary_code": {"system": "RxNorm", "code": "123456"},
                "mapped_codes": {}
            }
        ]
        
        demographics = {
            "gender": "female",
            "age": "55"
        }
        
        result = self.generator.generate_mcode_resources(mapped_elements, demographics)
        
        # Check result structure
        self.assertIn("bundle", result)
        self.assertIn("resources", result)
        self.assertIn("validation", result)
        
        # Check resources
        self.assertEqual(len(result["resources"]), 3)  # Patient + Condition + MedicationStatement
        self.assertEqual(result["resources"][0]["resourceType"], "Patient")
        self.assertEqual(result["resources"][1]["resourceType"], "Condition")
        self.assertEqual(result["resources"][2]["resourceType"], "MedicationStatement")
        
        # Check bundle
        self.assertEqual(result["bundle"]["resourceType"], "Bundle")
        self.assertEqual(len(result["bundle"]["entry"]), 3)
        
        # Check validation
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(len(result["validation"]["resource_validations"]), 3)
        self.assertEqual(result["validation"]["compliance_score"], 1.0)
    
    def test_to_json(self):
        """
        Test converting data to JSON format
        """
        data = {
            "resourceType": "Patient",
            "gender": "female"
        }
        
        result = self.generator.to_json(data)
        
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
        
        result = self.generator.to_xml(data)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that it contains expected XML elements
        self.assertIn("<Bundle", result)
        self.assertIn("xmlns", result)
        self.assertIn("<gender value=\"female\" />", result)


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    tests_dir = os.path.dirname(__file__)
    if tests_dir and not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    
    unittest.main()
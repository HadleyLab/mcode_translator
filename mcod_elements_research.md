# mCODE (Minimal Common Oncology Data Elements) Research

## Overview
mCODE is an initiative by the American Society of Clinical Oncology (ASCO) to develop a standard set of common data elements for oncology electronic health records and clinical trials. It is built on HL7 FHIR standards and includes a superset of ICD, CPT, and other medical codes.

## Key mCODE Data Elements Relevant to Clinical Trial Criteria

### Patient Demographics
- **Patient Birth Date** - Patient's date of birth
- **Patient Gender** - Patient's gender identity
- **Patient Race** - Patient's race
- **Patient Ethnicity** - Patient's ethnicity

### Cancer Condition Elements
- **Primary Cancer Condition** - The primary cancer diagnosis
- **Secondary Cancer Condition** - Metastatic or secondary cancer sites
- **Cancer Stage** - TNM staging or other staging systems
- **Tumor Marker** - Biomarker test results
- **ECOG Performance Status** - Patient's functional status

### Cancer Treatment Elements
- **Cancer Treatment Plan** - Overall treatment plan
- **Cancer Treatment Course** - Specific treatment courses
- **Medication Statement** - Medications used in treatment
- **Radiotherapy Course** - Radiation therapy details
- **Procedure** - Surgical procedures performed

### Clinical Trial Specific Elements
- **Eligibility Criteria** - Inclusion/exclusion criteria mapping
- **Study Participation** - Patient's participation in clinical trials
- **Adverse Events** - Side effects experienced during treatment

## mCODE Code Systems
- **ICD-10-CM** - International Classification of Diseases
- **ICD-O-3** - International Classification of Diseases for Oncology
- **SNOMED CT** - Systematized Nomenclature of Medicine
- **LOINC** - Logical Observation Identifiers Names and Codes
- **RxNorm** - Standardized names for clinical drugs
- **CPT** - Current Procedural Terminology
- **HCPCS** - Healthcare Common Procedure Coding System

## Data Structure Example
```json
{
  "resourceType": "Bundle",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "gender": "male",
        "birthDate": "1951-07-20"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "code": {
          "coding": [
            {
              "system": "http://hl7.org/fhir/sid/icd-10-cm",
              "code": "C50.911",
              "display": "Malignant neoplasm of upper-outer quadrant of right female breast"
            }
          ]
        }
      }
    }
  ]
}
```

## Mapping Clinical Trial Criteria to mCODE
Clinical trial eligibility criteria often reference:
1. **Diagnosis codes** - Map to ICD-10-CM/ICD-O-3
2. **Procedure codes** - Map to CPT/HCPCS
3. **Medication requirements** - Map to RxNorm
4. **Lab test values** - Map to LOINC
5. **Performance status** - Map to ECOG scales
6. **Demographic restrictions** - Map to Patient elements
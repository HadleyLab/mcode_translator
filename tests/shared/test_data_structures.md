# ClinicalTrials.gov API Data Structures

This document describes the actual data structures returned by the pytrials client when querying the ClinicalTrials.gov API. These structures should be used when creating mocks for unit tests.

## Search Trials Response Structure

When calling `ct.get_study_fields()`:

```json
{
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT06485076",
          "briefTitle": "Study Title"
        },
        "statusModule": {
          "overallStatus": "RECRUITING"
        },
        "conditionsModule": {
          "conditions": ["Condition 1", "Condition 2"]
        }
      }
    }
  ],
  "nextPageToken": "token123",
  "totalCount": 1500
}
```

## Full Studies Response Structure

When calling `ct.get_full_studies()`:

```json
{
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT06485076",
          "orgStudyIdInfo": {
            "id": "StudyID"
          },
          "organization": {
            "fullName": "Organization Name",
            "class": "OTHER"
          },
          "briefTitle": "Brief Study Title",
          "officialTitle": "Official Study Title"
        },
        "statusModule": {
          "statusVerifiedDate": "2024-08",
          "overallStatus": "RECRUITING",
          "expandedAccessInfo": {
            "hasExpandedAccess": false
          },
          "startDateStruct": {
            "date": "2024-07-18",
            "type": "ACTUAL"
          }
        },
        "eligibilityModule": {
          "eligibilityCriteria": "Inclusion Criteria:\n- Criterion 1\n- Criterion 2\n\nExclusion Criteria:\n- Criterion A\n- Criterion B"
        }
      },
      "derivedSection": {
        "miscInfoModule": {
          "versionHolder": "2024-08-21"
        }
      },
      "hasResults": false
    }
  ],
  "nextPageToken": "token123"
}
```

## Count Total Response Structure

When calling `ct.get_study_fields()` with `&countTotal=true` in the search expression:

```json
{
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT06485076",
          "briefTitle": "Study Title"
        }
      }
    }
  ],
  "totalCount": 1500
}
```

Note: When using `&countTotal=true`, the `totalCount` field contains the total number of studies matching the search criteria, regardless of the number of studies returned in the `studies` array.

## Key Differences from Previous Versions

1. **Root key**: Use `"studies"` instead of `"StudyFields"` or `"FullStudies"`
2. **Case sensitivity**: Use lowercase key names (e.g., `"nctId"` instead of `"NCTId"`)
3. **Nested structure**: Follow the nested structure: `studies` → `protocolSection` → `identificationModule` → `nctId`
4. **Full studies**: Include additional sections like `derivedSection` and `hasResults`

## Mock Structure Recommendations

When creating mocks for unit tests, ensure they match these structures:

### For search_trials function:
```python
mock_clinical_trials_api.return_value.get_study_fields.return_value = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study 1"
                }
            }
        }
    ],
    "nextPageToken": "token123"
}
```

### For calculate_total_studies function:
```python
mock_clinical_trials_api.return_value.get_study_fields.return_value = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study 1"
                }
            }
        }
    ],
    "totalCount": 1000
}
```

### For get_full_study function:
```python
mock_clinical_trials_api.return_value.get_full_studies.return_value = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study 1"
                }
            },
            "derivedSection": {},
            "hasResults": False
        }
    ]
}
```

## Field Mappings

| Our Field Name | API Field Name |
|----------------|----------------|
| NCTId          | nctId          |
| BriefTitle     | briefTitle     |
| Condition      | conditions     |
| OverallStatus  | overallStatus  |

Note: Field names in the API response are lowercase, while our internal field mapping uses mixed case.

## Additional Fields

When using `&countTotal=true` in the search expression, an additional `totalCount` field is included in the response, which contains the total number of studies matching the search criteria.
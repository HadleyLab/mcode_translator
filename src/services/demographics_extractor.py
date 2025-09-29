"""
Demographics Extractor - Extract patient demographics from FHIR resources.

This module provides specialized extraction of patient demographics
from FHIR Patient resources for mCODE compliance.
"""

from typing import Any, Dict

from src.utils.logging_config import get_logger


class DemographicsExtractor:
    """
    Extractor for patient demographics from FHIR Patient resources.

    Provides comprehensive extraction of demographics information
    required for mCODE compliance and clinical documentation.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def extract_demographics(self, patient_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive demographics for mCODE compliance."""
        demographics = {}

        try:
            # Extract name
            names = patient_resource.get("name", [])
            self.logger.debug(f"Names type: {type(names)}, value: {names}")
            if names and isinstance(names, list) and len(names) > 0:
                name_obj = names[0]
                self.logger.debug(f"Name obj type: {type(name_obj)}, value: {name_obj}")
                if isinstance(name_obj, dict):
                    given_names = name_obj.get("given", [])
                    family_name = name_obj.get("family", "")
                    self.logger.debug(
                        f"Given names type: {type(given_names)}, value: {given_names}"
                    )
                    self.logger.debug(
                        f"Family name type: {type(family_name)}, value: {family_name}"
                    )

                    if given_names and family_name:
                        demographics["name"] = f"{given_names[0]} {family_name}"
                    elif given_names:
                        demographics["name"] = given_names[0]
                    elif family_name:
                        demographics["name"] = family_name

            # Extract birth date (DOB)
            birth_date = patient_resource.get("birthDate")
            if birth_date:
                demographics["birthDate"] = birth_date
                try:
                    from datetime import datetime

                    birth_year = datetime.fromisoformat(birth_date[:10]).year
                    current_year = datetime.now().year
                    age = current_year - birth_year
                    demographics["age"] = f"{age}"
                except Exception as e:
                    self.logger.debug(f"Error parsing birth date: {e}")
                    demographics["age"] = "Unknown"
            else:
                demographics["birthDate"] = "Unknown"
                demographics["age"] = "Unknown"

            # Extract gender (administrative gender)
            gender = patient_resource.get("gender")
            if gender:
                demographics["gender"] = gender.capitalize()
            else:
                demographics["gender"] = "Unknown"

            # Extract birth sex (if available via extension)
            birth_sex = "Unknown"
            extensions = patient_resource.get("extension", [])
            for ext in extensions:
                if isinstance(ext, dict):
                    url = ext.get("url", "")
                    if "birthsex" in url.lower():
                        value_code = ext.get("valueCode", "")
                        if value_code:
                            birth_sex = value_code.capitalize()
                            break
            demographics["birthSex"] = birth_sex

            # Extract marital status
            marital_status = patient_resource.get("maritalStatus", {})
            if isinstance(marital_status, dict):
                coding = marital_status.get("coding", [{}])[0]
                if isinstance(coding, dict):
                    display = coding.get("display", "")
                    if display:
                        demographics["maritalStatus"] = display
                    else:
                        demographics["maritalStatus"] = "Unknown"
                else:
                    demographics["maritalStatus"] = "Unknown"
            else:
                demographics["maritalStatus"] = "Unknown"

            # Extract communication/language preferences
            communication = patient_resource.get("communication", [])
            if communication and isinstance(communication, list):
                comm_obj = communication[0]
                if isinstance(comm_obj, dict):
                    language = comm_obj.get("language", {})
                    if isinstance(language, dict):
                        coding = language.get("coding", [{}])[0]
                        if isinstance(coding, dict):
                            display = coding.get("display", "")
                            if display:
                                demographics["language"] = display
                            else:
                                demographics["language"] = "Unknown"
                        else:
                            demographics["language"] = "Unknown"
                    else:
                        demographics["language"] = "Unknown"
                else:
                    demographics["language"] = "Unknown"
            else:
                demographics["language"] = "Unknown"

            # Extract address information for geographic context
            addresses = patient_resource.get("address", [])
            if addresses and isinstance(addresses, list):
                address = addresses[0]
                if isinstance(address, dict):
                    city = address.get("city", "")
                    state = address.get("state", "")
                    country = address.get("country", "")
                    if city or state or country:
                        demographics["address"] = f"{city}, {state}, {country}".strip(
                            ", "
                        )
                    else:
                        demographics["address"] = "Unknown"
                else:
                    demographics["address"] = "Unknown"
            else:
                demographics["address"] = "Unknown"

        except Exception as e:
            self.logger.error(f"Error extracting demographics: {e}")
            import traceback

            traceback.print_exc()

        return demographics

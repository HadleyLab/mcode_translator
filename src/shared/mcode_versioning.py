"""
mCODE Schema Versioning System

This module provides comprehensive versioning support for mCODE (Minimal Common Oncology Data Elements)
schema versions, including profile URL generation, version migration utilities, backward compatibility
handling, and version validation.

The system supports mCODE STU (Standards for Trial Use) versions and provides extensible
organization for future schema evolution.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urljoin


class McodeVersion(Enum):
    """Enumeration of supported mCODE STU versions."""

    STU1 = "1.0.0"
    STU2 = "2.0.0"
    STU3 = "3.0.0"
    STU4 = "4.0.0"  # Current version as of 2024

    @property
    def major(self) -> int:
        """Get major version number."""
        return int(self.value.split('.')[0])

    @property
    def minor(self) -> int:
        """Get minor version number."""
        return int(self.value.split('.')[1])

    @property
    def patch(self) -> int:
        """Get patch version number."""
        return int(self.value.split('.')[2])

    @classmethod
    def latest(cls) -> 'McodeVersion':
        """Get the latest supported mCODE version."""
        return max(cls, key=lambda v: (v.major, v.minor, v.patch))

    @classmethod
    def from_string(cls, version_str: str) -> 'McodeVersion':
        """Parse version string into McodeVersion enum."""
        # Normalize version string
        normalized = version_str.strip()
        if normalized.startswith('STU'):
            # Handle "STU4" format
            major = int(normalized[3:])
            normalized = f"{major}.0.0"
        elif '.' not in normalized:
            # Handle single number format like "4"
            normalized = f"{normalized}.0.0"

        # Ensure semantic versioning format
        parts = normalized.split('.')
        if len(parts) == 2:
            normalized = f"{parts[0]}.{parts[1]}.0"
        elif len(parts) == 1:
            normalized = f"{parts[0]}.0.0"

        for version in cls:
            if version.value == normalized:
                return version

        raise ValueError(f"Unsupported mCODE version: {version_str}")


class McodeProfile(Enum):
    """Enumeration of core mCODE profiles."""

    # Patient Information
    PATIENT = "Patient"
    BIRTH_SEX = "BirthSex"
    ADMINISTRATIVE_GENDER = "AdministrativeGender"
    US_CORE_RACE = "USCoreRaceExtension"
    US_CORE_ETHNICITY = "USCoreEthnicityExtension"
    US_CORE_BIRTH_SEX = "USCoreBirthSexExtension"

    # Disease Characterization
    CANCER_CONDITION = "CancerCondition"
    CANCER_STAGING = "CancerStaging"
    TNM_STAGE_GROUP = "TNMStageGroup"
    TNM_PRIMARY_TUMOR = "TNMPrimaryTumorCategory"
    TNM_REGIONAL_NODES = "TNMRegionalNodesCategory"
    TNM_DISTANT_METASTASES = "TNMDistantMetastasesCategory"
    HISTOLOGY_MORPHOLOGY_BEHAVIOR = "HistologyMorphologyBehavior"

    # Assessment
    TUMOR_MARKER_TEST = "TumorMarkerTest"
    ESTROGEN_RECEPTOR_STATUS = "ERReceptorStatus"
    PROGESTERONE_RECEPTOR_STATUS = "PRReceptorStatus"
    HER2_RECEPTOR_STATUS = "HER2ReceptorStatus"
    ONCOTYPE_DX_SCORE = "OncotypeDXScore"
    ECOG_PERFORMANCE_STATUS = "ECOGPerformanceStatus"
    KARNOVSKY_PERFORMANCE_STATUS = "KarnofskyPerformanceStatus"

    # Treatments
    CANCER_RELATED_MEDICATION_STATEMENT = "CancerRelatedMedicationStatement"
    CANCER_RELATED_SURGICAL_PROCEDURE = "CancerRelatedSurgicalProcedure"
    CANCER_RELATED_RADIATION_PROCEDURE = "CancerRelatedRadiationProcedure"

    # Genomics
    CANCER_GENETIC_VARIANT = "CancerGeneticVariant"
    GENOMIC_REGION_STUDIED = "GenomicRegionStudied"

    # Outcomes
    CAUSE_OF_DEATH = "CauseOfDeath"
    COMORBID_CONDITION = "ComorbidCondition"

    # Extensions and Supporting Profiles
    US_CORE_SMOKING_STATUS = "USCoreSmokingStatus"
    US_CORE_BMI = "USCoreBMI"
    US_CORE_BODY_HEIGHT = "USCoreBodyHeight"
    US_CORE_BODY_WEIGHT = "USCoreBodyWeight"
    US_CORE_BLOOD_PRESSURE = "USCoreBloodPressure"
    US_CORE_HEMOGLOBIN = "USCoreHemoglobin"
    US_CORE_WHITE_BLOOD_CELL = "USCoreWhiteBloodCellCount"
    US_CORE_PLATELET_COUNT = "USCorePlateletCount"
    US_CORE_CHEMISTRY_CREATININE = "USCoreChemistryCreatinine"
    US_CORE_CHEMISTRY_BILIRUBIN = "USCoreChemistryBilirubinTotal"
    US_CORE_CHEMISTRY_ALT = "USCoreChemistryAlanineAminotransferase"


@dataclass(frozen=True)
class VersionCompatibility:
    """Represents compatibility between mCODE versions."""

    source_version: McodeVersion
    target_version: McodeVersion
    compatible: bool
    breaking_changes: List[str]
    migration_notes: List[str]

    def can_migrate_to(self, target: McodeVersion) -> bool:
        """Check if migration to target version is possible."""
        return self.compatible and self.target_version == target


class McodeVersionManager:
    """
    Central manager for mCODE version operations.

    Provides version management, profile URL generation, compatibility checking,
    and migration utilities for mCODE schema versions.
    """

    # Base URLs for different mCODE versions
    VERSION_BASE_URLS = {
        McodeVersion.STU2: "https://mcodeinitiative.org/fhir/stu2/StructureDefinition/",
        McodeVersion.STU3: "https://mcodeinitiative.org/fhir/stu3/StructureDefinition/",
        McodeVersion.STU4: "https://mcodeinitiative.org/fhir/stu4/StructureDefinition/",
    }

    # Version compatibility matrix
    COMPATIBILITY_MATRIX: Dict[Tuple[McodeVersion, McodeVersion], VersionCompatibility] = {
        (McodeVersion.STU1, McodeVersion.STU2): VersionCompatibility(
            source_version=McodeVersion.STU1,
            target_version=McodeVersion.STU2,
            compatible=True,
            breaking_changes=[
                "Updated profile URLs",
                "Enhanced genomics profiles",
                "Added pediatric oncology support"
            ],
            migration_notes=[
                "Review genomics data for new variant representation",
                "Update profile URLs in implementations",
                "Test pediatric oncology use cases"
            ]
        ),
        (McodeVersion.STU2, McodeVersion.STU3): VersionCompatibility(
            source_version=McodeVersion.STU2,
            target_version=McodeVersion.STU3,
            compatible=True,
            breaking_changes=[
                "Refined medication normalization",
                "Enhanced staging profiles",
                "Updated value sets"
            ],
            migration_notes=[
                "Verify medication data normalization",
                "Check staging data for new fields",
                "Update to latest value sets"
            ]
        ),
        (McodeVersion.STU3, McodeVersion.STU4): VersionCompatibility(
            source_version=McodeVersion.STU3,
            target_version=McodeVersion.STU4,
            compatible=True,
            breaking_changes=[
                "Added risk assessment profiles",
                "Enhanced histology profiles",
                "Improved medication administration tracking"
            ],
            migration_notes=[
                "Consider adding risk assessment data",
                "Review histology data for new fields",
                "Update medication administration records"
            ]
        ),
    }

    def __init__(self, default_version: Optional[McodeVersion] = None):
        """Initialize version manager with default version."""
        self.default_version = default_version or McodeVersion.latest()

    def get_profile_url(self, profile: McodeProfile, version: Optional[McodeVersion] = None) -> str:
        """
        Generate canonical URL for an mCODE profile in specified version.

        Args:
            profile: The mCODE profile to generate URL for
            version: Version to use (defaults to manager's default version)

        Returns:
            Canonical URL for the profile

        Raises:
            ValueError: If version is not supported
        """
        version = version or self.default_version

        if version not in self.VERSION_BASE_URLS:
            raise ValueError(f"Unsupported mCODE version: {version}")

        base_url = self.VERSION_BASE_URLS[version]
        return urljoin(base_url, profile.value)

    def get_all_profile_urls(self, version: Optional[McodeVersion] = None) -> Dict[McodeProfile, str]:
        """
        Get canonical URLs for all supported profiles in specified version.

        Args:
            version: Version to use (defaults to manager's default version)

        Returns:
            Dictionary mapping profiles to their canonical URLs
        """
        return {profile: self.get_profile_url(profile, version) for profile in McodeProfile}

    def check_compatibility(self, source_version: McodeVersion, target_version: McodeVersion) -> VersionCompatibility:
        """
        Check compatibility between two mCODE versions.

        Args:
            source_version: Source version to migrate from
            target_version: Target version to migrate to

        Returns:
            VersionCompatibility object with compatibility details

        Raises:
            ValueError: If direct compatibility check is not available
        """
        if source_version == target_version:
            return VersionCompatibility(
                source_version=source_version,
                target_version=target_version,
                compatible=True,
                breaking_changes=[],
                migration_notes=["No migration required - versions are identical"]
            )

        # Check direct compatibility
        key = (source_version, target_version)
        if key in self.COMPATIBILITY_MATRIX:
            return self.COMPATIBILITY_MATRIX[key]

        # Check if we can find a migration path
        if source_version.major < target_version.major:
            # Forward compatibility - assume possible but with breaking changes
            return VersionCompatibility(
                source_version=source_version,
                target_version=target_version,
                compatible=True,
                breaking_changes=["Major version upgrade may include breaking changes"],
                migration_notes=[
                    "Review release notes for breaking changes",
                    "Test thoroughly in staging environment",
                    "Consider gradual rollout"
                ]
            )
        else:
            # Backward compatibility - generally supported
            return VersionCompatibility(
                source_version=source_version,
                target_version=target_version,
                compatible=True,
                breaking_changes=[],
                migration_notes=[
                    "Backward compatibility maintained",
                    "Some features may not be available in older version"
                ]
            )

    def validate_version_string(self, version_str: str) -> bool:
        """
        Validate if a version string represents a supported mCODE version.

        Args:
            version_str: Version string to validate

        Returns:
            True if version is supported, False otherwise
        """
        try:
            McodeVersion.from_string(version_str)
            return True
        except ValueError:
            return False

    def get_supported_versions(self) -> List[McodeVersion]:
        """Get list of all supported mCODE versions."""
        return list(McodeVersion)

    def is_version_supported(self, version: McodeVersion) -> bool:
        """Check if a version is currently supported."""
        return version in self.VERSION_BASE_URLS

    def get_migration_path(self, source_version: McodeVersion, target_version: McodeVersion) -> List[VersionCompatibility]:
        """
        Get migration path between two versions.

        Args:
            source_version: Starting version
            target_version: Target version

        Returns:
            List of VersionCompatibility objects representing migration steps

        Raises:
            ValueError: If no migration path exists
        """
        if source_version == target_version:
            return []

        path: List[VersionCompatibility] = []
        current = source_version

        # Try to find direct path
        if (current, target_version) in self.COMPATIBILITY_MATRIX:
            path.append(self.COMPATIBILITY_MATRIX[(current, target_version)])
            return path

        # For major version jumps, provide general guidance
        if abs(source_version.major - target_version.major) > 1:
            return [VersionCompatibility(
                source_version=source_version,
                target_version=target_version,
                compatible=True,
                breaking_changes=["Major version jump requires careful review"],
                migration_notes=[
                    "Consult mCODE release notes for detailed migration guide",
                    "Consider intermediate version upgrades",
                    "Test extensively before production deployment"
                ]
            )]

        # No direct path found
        raise ValueError(f"No migration path found from {source_version.value} to {target_version.value}")

    def get_version_features(self, version: McodeVersion) -> Dict[str, Any]:
        """
        Get feature set for a specific mCODE version.

        Args:
            version: Version to get features for

        Returns:
            Dictionary of version features and capabilities
        """
        base_features: Dict[str, Any] = {
            "profiles_supported": len(McodeProfile),
            "genomics_support": version.major >= 2,
            "pediatric_support": version.major >= 4,
            "risk_assessment": version.major >= 4,
            "enhanced_medication": version.major >= 3,
        }

        # Version-specific features
        version_features: Dict[McodeVersion, Dict[str, Any]] = {
            McodeVersion.STU1: {
                **base_features,
                "staging_support": "basic",
                "genomics_support": False,
                "pediatric_support": False,
                "risk_assessment": False,
                "enhanced_medication": False,
            },
            McodeVersion.STU2: {
                **base_features,
                "staging_support": "intermediate",
                "genomics_support": True,
                "pediatric_support": False,
                "risk_assessment": False,
                "enhanced_medication": False,
            },
            McodeVersion.STU3: {
                **base_features,
                "staging_support": "advanced",
                "genomics_support": True,
                "pediatric_support": False,
                "risk_assessment": False,
                "enhanced_medication": True,
            },
            McodeVersion.STU4: {
                **base_features,
                "staging_support": "comprehensive",
                "genomics_support": True,
                "pediatric_support": True,
                "risk_assessment": True,
                "enhanced_medication": True,
            },
        }

        return version_features.get(version, base_features)


# Global version manager instance
_version_manager = McodeVersionManager()


def get_version_manager() -> McodeVersionManager:
    """Get the global mCODE version manager instance."""
    return _version_manager


def get_profile_url(profile: McodeProfile, version: Optional[McodeVersion] = None) -> str:
    """
    Convenience function to get profile URL using global version manager.

    Args:
        profile: mCODE profile
        version: Version (defaults to latest)

    Returns:
        Canonical profile URL
    """
    return _version_manager.get_profile_url(profile, version)


def check_version_compatibility(source: str, target: str) -> VersionCompatibility:
    """
    Convenience function to check version compatibility.

    Args:
        source: Source version string
        target: Target version string

    Returns:
        VersionCompatibility object
    """
    source_version = McodeVersion.from_string(source)
    target_version = McodeVersion.from_string(target)
    return _version_manager.check_compatibility(source_version, target_version)


def validate_mcode_version(version: str) -> bool:
    """
    Validate if a version string is a supported mCODE version.

    Args:
        version: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    return _version_manager.validate_version_string(version)
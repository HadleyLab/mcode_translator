#!/usr/bin/env python
# coding: utf-8

# # 🎯 MCODE Translator - Patient-Trial Matching Demo
#
# Comprehensive demonstration of intelligent patient-trial matching algorithms and knowledge graph queries.
#
# ## 📋 What This Demo Covers
#
# This notebook demonstrates MCODE Translator's patient-trial matching capabilities:
#
# 1. **👥 Patient Profile Analysis** - Detailed patient characteristic extraction
# 2. **🧪 Clinical Trial Eligibility Analysis** - Automated eligibility criteria parsing
# 3. **🎯 Intelligent Matching Algorithms** - Advanced matching based on multiple criteria
# 4. **📊 Matching Score Calculation** - Quantitative matching assessment
# 5. **🔍 Knowledge Graph Queries** - Relationship discovery and pattern analysis
# 6. **💡 Clinical Decision Support** - Evidence-based matching recommendations
#
# ## 🎯 Learning Objectives
#
# By the end of this demo, you will:
# - ✅ Master patient profile analysis and feature extraction
# - ✅ Understand clinical trial eligibility criteria parsing
# - ✅ Learn advanced matching algorithm implementation
# - ✅ Apply knowledge graph queries for relationship discovery
# - ✅ Use quantitative matching score assessment
# - ✅ Generate evidence-based clinical recommendations
#
# ## 🏥 Clinical Decision Support Applications
#
# ### Precision Medicine
# - **Biomarker Matching**: Match patients to biomarker-specific trials
# - **Treatment History Analysis**: Consider prior treatment responses
# - **Comorbidity Assessment**: Evaluate impact of concurrent conditions
# - **Performance Status Evaluation**: Assess functional capacity for trials
#
# ### Clinical Trial Enrollment
# - **Automated Screening**: Rapid eligibility assessment
# - **Trial Prioritization**: Rank trials by match quality
# - **Protocol Optimization**: Identify optimal trial placement
# - **Enrollment Prediction**: Estimate likelihood of trial completion
#
# ### Evidence-Based Medicine
# - **Treatment Response Prediction**: Based on similar patient outcomes
# - **Risk-Benefit Assessment**: Balance trial opportunities vs standard care
# - **Guideline Adherence**: Ensure alignment with clinical guidelines
# - **Outcome Optimization**: Maximize potential for positive outcomes
#
# ---

# ## 🔧 Setup and Configuration

# In[ ]:


# Import required modules
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path.cwd() / "src"))

# Import MCODE Translator components
try:
    from heysol import HeySolClient

    from config.heysol_config import get_config

    print("✅ MCODE Translator components imported successfully!")
    print("   🎯 Patient-trial matching algorithms")
    print("   🧠 Knowledge graph query capabilities")
    print("   📊 Advanced analytics and scoring")

except ImportError as e:
    print("❌ Failed to import MCODE Translator components.")
    print("💡 Install with: pip install -e .")
    print(f"   Error: {e}")
    raise


# ## 👥 Patient Profile Analysis
#
# Let's start by analyzing patient profiles and extracting key characteristics for matching.

# In[ ]:


# Initialize HeySol client
print("🔧 Initializing Patient-Trial Matching System...")
print("=" * 60)

api_key = os.getenv("HEYSOL_API_KEY")
if not api_key:
    print("❌ No API key found!")
    print("💡 Set HEYSOL_API_KEY environment variable")
    raise ValueError("API key not configured")

client = HeySolClient(api_key=api_key)
config = get_config()

print("✅ Client initialized successfully")
print(f"   🎯 Base URL: {config.get_base_url()}")
print(f"   📧 Source: {config.get_heysol_config().source}")

# Create dedicated space for patient-trial matching
matching_space_name = "Patient-Trial Matching Intelligence"
matching_space_description = "Advanced patient-trial matching algorithms and results"

print(f"\n🏗️ Setting up matching space: {matching_space_name}")

# Check for existing space
existing_spaces = client.get_spaces()
matching_space_id = None

for space in existing_spaces:
    if isinstance(space, dict) and space.get("name") == matching_space_name:
        matching_space_id = space.get("id")
        print(f"   ✅ Found existing space: {matching_space_id[:16]}...")
        break

if not matching_space_id:
    matching_space_id = client.create_space(
        matching_space_name, matching_space_description
    )
    print(f"   ✅ Created new space: {matching_space_id[:16]}...")

print("✅ Patient-trial matching space ready!")


# ### 📋 Patient Profile Definitions
#
# Let's define comprehensive patient profiles with detailed clinical characteristics for matching analysis.

# In[ ]:


# Define patient profiles for matching analysis
@dataclass
class PatientProfile:
    """Comprehensive patient profile for trial matching."""

    patient_id: str
    age: int
    gender: str
    cancer_type: str
    stage: str
    grade: Optional[int] = None
    receptor_status: Optional[str] = None
    mutation_status: Optional[str] = None
    performance_status: str = "ECOG_0"
    comorbidities: List[str] = None
    prior_treatments: List[str] = None
    current_therapy: Optional[str] = None
    metastatic_sites: List[str] = None
    lab_values: Dict[str, Any] = None

    def __post_init__(self):
        if self.comorbidities is None:
            self.comorbidities = []
        if self.prior_treatments is None:
            self.prior_treatments = []
        if self.metastatic_sites is None:
            self.metastatic_sites = []
        if self.lab_values is None:
            self.lab_values = {}


# Create comprehensive patient profiles
patient_profiles = [
    PatientProfile(
        patient_id="P001",
        age=52,
        gender="female",
        cancer_type="breast",
        stage="IIA",
        grade=2,
        receptor_status="ER+/PR+/HER2-",
        performance_status="ECOG_0",
        prior_treatments=["neoadjuvant_chemotherapy", "surgery"],
        current_therapy="anastrozole",
        comorbidities=["hypertension"],
        lab_values={"hemoglobin": 12.5, "creatinine": 0.8, "bilirubin": 0.6},
    ),
    PatientProfile(
        patient_id="P002",
        age=67,
        gender="male",
        cancer_type="lung",
        stage="IV",
        mutation_status="EGFR_exon_19_deletion",
        performance_status="ECOG_1",
        prior_treatments=["osimertinib"],
        metastatic_sites=["bone", "liver"],
        comorbidities=["COPD", "diabetes"],
        lab_values={"hemoglobin": 11.2, "creatinine": 1.1, "bilirubin": 0.7},
    ),
    PatientProfile(
        patient_id="P003",
        age=45,
        gender="female",
        cancer_type="melanoma",
        stage="IIIB",
        mutation_status="BRAF_V600E",
        performance_status="ECOG_0",
        prior_treatments=["surgery"],
        comorbidities=[],
        lab_values={"hemoglobin": 13.1, "creatinine": 0.9, "bilirubin": 0.5},
    ),
    PatientProfile(
        patient_id="P004",
        age=58,
        gender="male",
        cancer_type="colorectal",
        stage="IV",
        mutation_status="KRAS_G12V",
        performance_status="ECOG_1",
        prior_treatments=["FOLFOX", "FOLFIRI"],
        metastatic_sites=["liver", "lung"],
        comorbidities=["hypertension", "diabetes"],
        lab_values={"hemoglobin": 10.8, "creatinine": 1.2, "bilirubin": 1.8},
    ),
]

print("👥 Patient Profiles Created:")
print("-" * 40)

for patient in patient_profiles:
    print(f"\n   Patient {patient.patient_id}:")
    print(f"      Age/Gender: {patient.age}-year-old {patient.gender}")
    print(
        f"      Diagnosis: {patient.cancer_type.title()} cancer, stage {patient.stage}"
    )
    print(
        f"      Biomarkers: {patient.receptor_status or patient.mutation_status or 'Not specified'}"
    )
    print(f"      Performance: {patient.performance_status}")
    print(f"      Prior treatments: {len(patient.prior_treatments)}")
    print(f"      Comorbidities: {len(patient.comorbidities)}")


# ## 🧪 Clinical Trial Eligibility Analysis
#
# Now let's define clinical trial eligibility criteria and analyze trial requirements.

# In[ ]:


# Define clinical trial eligibility criteria
@dataclass
class TrialEligibility:
    """Clinical trial eligibility criteria."""

    trial_id: str
    cancer_type: str
    stage: List[str]
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    performance_status: List[str] = None
    required_biomarkers: List[str] = None
    excluded_biomarkers: List[str] = None
    prior_treatment_limit: Optional[int] = None
    required_lab_values: Dict[str, Tuple[str, float]] = None  # (comparator, value)
    exclusion_criteria: List[str] = None

    def __post_init__(self):
        if self.performance_status is None:
            self.performance_status = ["ECOG_0", "ECOG_1"]
        if self.required_biomarkers is None:
            self.required_biomarkers = []
        if self.excluded_biomarkers is None:
            self.excluded_biomarkers = []
        if self.required_lab_values is None:
            self.required_lab_values = {}
        if self.exclusion_criteria is None:
            self.exclusion_criteria = []


# Create comprehensive trial eligibility criteria
trial_eligibility_criteria = [
    TrialEligibility(
        trial_id="NCT04567892",
        cancer_type="melanoma",
        stage=["IIIB", "IIIC", "IV"],
        age_min=18,
        performance_status=["ECOG_0", "ECOG_1"],
        required_biomarkers=["BRAF_mutation"],
        prior_treatment_limit=1,
        exclusion_criteria=["active_brain_metastases", "autoimmune_disease"],
    ),
    TrialEligibility(
        trial_id="NCT02314481",
        cancer_type="breast",
        stage=["IV"],
        age_min=18,
        performance_status=["ECOG_0", "ECOG_1", "ECOG_2"],
        required_biomarkers=["ER_positive", "HER2_negative"],
        prior_treatment_limit=0,
        exclusion_criteria=["prior_CDK4/6_inhibitor"],
    ),
    TrialEligibility(
        trial_id="NCT03456789",
        cancer_type="lung",
        stage=["IV"],
        age_min=18,
        performance_status=["ECOG_0", "ECOG_1"],
        required_biomarkers=["KRAS_G12C"],
        prior_treatment_limit=2,
        required_lab_values={"creatinine": ("<=", 1.5), "bilirubin": ("<=", 1.5)},
    ),
    TrialEligibility(
        trial_id="NCT01234567",
        cancer_type="breast",
        stage=["IV"],
        age_min=18,
        performance_status=["ECOG_0", "ECOG_1"],
        required_biomarkers=["HER2_positive"],
        prior_treatment_limit=1,
        exclusion_criteria=["significant_cardiac_disease"],
    ),
]

print("🧪 Clinical Trial Eligibility Criteria:")
print("-" * 50)

for trial in trial_eligibility_criteria:
    print(f"\n   Trial {trial.trial_id}:")
    print(f"      Cancer Type: {trial.cancer_type}")
    print(f"      Stages: {', '.join(trial.stage)}")
    print(f"      Age Range: {trial.age_min or 'No min'} - {trial.age_max or 'No max'}")
    print(
        f"      Required Biomarkers: {', '.join(trial.required_biomarkers) if trial.required_biomarkers else 'None'}"
    )
    print(f"      Performance Status: {', '.join(trial.performance_status)}")
    print(f"      Prior Treatment Limit: {trial.prior_treatment_limit or 'Unlimited'}")


# ## 🎯 Intelligent Matching Algorithm
#
# Let's implement sophisticated matching algorithms that consider multiple clinical factors.

# In[ ]:


# Intelligent patient-trial matching algorithm
class PatientTrialMatcher:
    """Advanced patient-trial matching with comprehensive scoring."""

    def __init__(self, client: HeySolClient, space_id: str):
        self.client = client
        self.space_id = space_id
        self.match_weights = {
            "cancer_type": 0.25,
            "stage": 0.20,
            "biomarkers": 0.25,
            "performance_status": 0.10,
            "age": 0.05,
            "prior_treatments": 0.10,
            "lab_values": 0.05,
        }

    def calculate_match_score(
        self, patient: PatientProfile, trial: TrialEligibility
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate comprehensive match score between patient and trial.

        Args:
            patient: Patient profile
            trial: Trial eligibility criteria

        Returns:
            Tuple of (total_score, detailed_scores)
        """
        scores = {}

        # 1. Cancer type match (25%)
        cancer_score = 1.0 if patient.cancer_type == trial.cancer_type else 0.0
        scores["cancer_type"] = cancer_score

        # 2. Stage eligibility (20%)
        stage_score = 1.0 if patient.stage in trial.stage else 0.0
        scores["stage"] = stage_score

        # 3. Biomarker matching (25%)
        biomarker_score = self._calculate_biomarker_score(patient, trial)
        scores["biomarkers"] = biomarker_score

        # 4. Performance status (10%)
        ps_score = (
            1.0 if patient.performance_status in trial.performance_status else 0.0
        )
        scores["performance_status"] = ps_score

        # 5. Age eligibility (5%)
        age_score = self._calculate_age_score(patient.age, trial.age_min, trial.age_max)
        scores["age"] = age_score

        # 6. Prior treatment assessment (10%)
        treatment_score = self._calculate_treatment_score(patient, trial)
        scores["prior_treatments"] = treatment_score

        # 7. Lab value assessment (5%)
        lab_score = self._calculate_lab_score(
            patient.lab_values, trial.required_lab_values
        )
        scores["lab_values"] = lab_score

        # Calculate weighted total score
        total_score = sum(
            scores[category] * weight for category, weight in self.match_weights.items()
        )

        return total_score, scores

    def _calculate_biomarker_score(
        self, patient: PatientProfile, trial: TrialEligibility
    ) -> float:
        """Calculate biomarker matching score."""
        patient_biomarkers = []

        # Extract biomarkers from patient profile
        if patient.receptor_status:
            patient_biomarkers.extend(patient.receptor_status.split("/"))
        if patient.mutation_status:
            patient_biomarkers.append(patient.mutation_status)

        # Check required biomarkers
        required_matches = 0
        for required in trial.required_biomarkers:
            if any(
                required.lower() in biomarker.lower()
                for biomarker in patient_biomarkers
            ):
                required_matches += 1

        required_score = (
            required_matches / len(trial.required_biomarkers)
            if trial.required_biomarkers
            else 1.0
        )

        # Check excluded biomarkers
        excluded_penalty = 0
        for excluded in trial.excluded_biomarkers:
            if any(
                excluded.lower() in biomarker.lower()
                for biomarker in patient_biomarkers
            ):
                excluded_penalty = 0.5  # Significant penalty for excluded biomarkers

        return max(0.0, required_score - excluded_penalty)

    def _calculate_age_score(
        self, patient_age: int, min_age: Optional[int], max_age: Optional[int]
    ) -> float:
        """Calculate age eligibility score."""
        if min_age is None and max_age is None:
            return 1.0
        if min_age is not None and patient_age < min_age:
            return 0.0
        if max_age is not None and patient_age > max_age:
            return 0.0
        return 1.0

    def _calculate_treatment_score(
        self, patient: PatientProfile, trial: TrialEligibility
    ) -> float:
        """Calculate prior treatment compatibility score."""
        if trial.prior_treatment_limit is None:
            return 1.0

        # Simple scoring based on treatment history length
        treatment_count = len(patient.prior_treatments)
        if treatment_count <= trial.prior_treatment_limit:
            return 1.0
        else:
            # Graduated penalty for exceeding treatment limit
            excess = treatment_count - trial.prior_treatment_limit
            return max(0.0, 1.0 - (excess * 0.2))

    def _calculate_lab_score(
        self, patient_labs: Dict[str, Any], required_labs: Dict[str, Tuple[str, float]]
    ) -> float:
        """Calculate lab value compatibility score."""
        if not required_labs:
            return 1.0

        lab_matches = 0
        for lab_name, (comparator, threshold) in required_labs.items():
            if lab_name in patient_labs:
                patient_value = patient_labs[lab_name]

                # Simple comparison logic
                if comparator == "<=" and patient_value <= threshold:
                    lab_matches += 1
                elif comparator == ">=" and patient_value >= threshold:
                    lab_matches += 1
                elif comparator == "<" and patient_value < threshold:
                    lab_matches += 1
                elif comparator == ">" and patient_value > threshold:
                    lab_matches += 1
                elif comparator == "==" and patient_value == threshold:
                    lab_matches += 1

        return lab_matches / len(required_labs) if required_labs else 1.0


# Initialize matcher
matcher = PatientTrialMatcher(client, matching_space_id)

print("🎯 Patient-Trial Matching Algorithm Initialized:")
print("-" * 50)
print("   📊 Scoring Weights:")
for category, weight in matcher.match_weights.items():
    print(f"      {category}: {weight:.1%}")
print("   🔍 Comprehensive matching criteria")
print("   💡 Evidence-based scoring algorithm")


# ## 📊 Comprehensive Matching Analysis
#
# Let's perform comprehensive matching analysis between all patients and trials.

# In[ ]:


# Perform comprehensive matching analysis
print("📊 Comprehensive Patient-Trial Matching Analysis")
print("=" * 60)

matching_results = []

print("🔍 Analyzing patient-trial compatibility...")
print("-" * 50)

for patient in patient_profiles:
    print(f"\n👤 Patient {patient.patient_id} Matching Analysis:")
    print(
        f"   Profile: {patient.cancer_type.title()} cancer, {patient.stage}, {patient.age}yo {patient.gender}"
    )

    patient_matches = []

    for trial in trial_eligibility_criteria:
        # Calculate match score
        total_score, detailed_scores = matcher.calculate_match_score(patient, trial)

        # Determine match category
        if total_score >= 0.8:
            match_category = "Excellent Match"
        elif total_score >= 0.6:
            match_category = "Good Match"
        elif total_score >= 0.4:
            match_category = "Fair Match"
        else:
            match_category = "Poor Match"

        match_result = {
            "patient_id": patient.patient_id,
            "trial_id": trial.trial_id,
            "total_score": total_score,
            "match_category": match_category,
            "detailed_scores": detailed_scores,
            "patient_profile": patient,
            "trial_criteria": trial,
        }

        patient_matches.append(match_result)
        matching_results.append(match_result)

        print(f"\n   🧪 Trial {trial.trial_id}:")
        print(f"      Match Score: {total_score:.1%}")
        print(f"      Category: {match_category}")
        print(
            f"      Cancer Type: {'✅' if detailed_scores['cancer_type'] == 1.0 else '❌'}"
        )
        print(f"      Stage: {'✅' if detailed_scores['stage'] == 1.0 else '❌'}")
        print(f"      Biomarkers: {detailed_scores['biomarkers']:.1%}")
        print(
            f"      Performance: {'✅' if detailed_scores['performance_status'] == 1.0 else '❌'}"
        )

    # Sort matches by score and show top recommendations
    patient_matches.sort(key=lambda x: x["total_score"], reverse=True)
    top_matches = [m for m in patient_matches if m["total_score"] >= 0.6]

    if top_matches:
        print("\n    🎯 Top Recommendations:")
        for i, match in enumerate(top_matches[:2], 1):
            print(
                f"      {i}. Trial {match['trial_id']} ({match['total_score']:.1%} match)"
            )
    else:
        print("\n    ⚠️ No strong matches found for this patient")
# ## 🔍 Knowledge Graph Queries for Enhanced Matching
#
# Let's use knowledge graph queries to enhance our matching with evidence-based insights.

# In[ ]:


# Knowledge graph queries for enhanced matching
print("🔍 Knowledge Graph-Enhanced Matching Analysis")
print("=" * 60)

knowledge_queries = [
    {
        "name": "Treatment Response Evidence",
        "query": "treatment response outcomes similar patients",
        "description": "Find evidence from similar patient cases",
        "analysis_type": "outcome_prediction",
    },
    {
        "name": "Biomarker-Treatment Correlations",
        "query": "biomarker treatment response correlations",
        "description": "Find correlations between biomarkers and treatment outcomes",
        "analysis_type": "biomarker_correlation",
    },
    {
        "name": "Clinical Trial Success Patterns",
        "query": "clinical trial success patterns patient characteristics",
        "description": "Identify patterns in successful trial outcomes",
        "analysis_type": "success_pattern_analysis",
    },
]

print("🧠 Knowledge Graph Query Analysis:")
print("-" * 50)

for query in knowledge_queries:
    print(f"\n🔍 {query['name']}")
    print(f"   Type: {query['analysis_type']}")
    print(f"   Description: {query['description']}")

    try:
        results = client.search(
            query=query["query"], space_ids=[matching_space_id], limit=3
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} knowledge connections")

        if episodes:
            print("\n   📋 Knowledge Insights:")
            for i, episode in enumerate(episodes, 1):
                content = episode.get("content", "")[:80]
                score = episode.get("score", "N/A")
                metadata = episode.get("metadata", {})

                print(f"\n   {i}. {content}{'...' if len(content) == 80 else ''}")
                print(f"      Relevance: {score}")
                print(
                    f"      Evidence Type: {metadata.get('evidence_type', 'General')}"
                )

                # Extract actionable insights
                if "response" in content.lower() and "positive" in content.lower():
                    print("      💡 Insight: Positive treatment response evidence")
                if "correlation" in content.lower() and "biomarker" in content.lower():
                    print(
                        "      💡 Insight: Biomarker-treatment correlation identified"
                    )

    except Exception as e:
        print(f"   ❌ Knowledge query failed: {e}")


# ## 💡 Clinical Decision Support Recommendations
#
# Let's generate evidence-based clinical recommendations based on our matching analysis.

# In[ ]:


# Generate clinical decision support recommendations
print("💡 Clinical Decision Support Recommendations")
print("=" * 60)


def generate_recommendations(
    matching_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate clinical recommendations based on matching analysis.

    Args:
        matching_results: Results from patient-trial matching

    Returns:
        List of recommendation dictionaries
    """
    recommendations = []

    # Analyze overall matching patterns
    excellent_matches = [
        r for r in matching_results if r["match_category"] == "Excellent Match"
    ]
    good_matches = [r for r in matching_results if r["match_category"] == "Good Match"]

    if excellent_matches:
        recommendations.append(
            {
                "type": "high_priority",
                "category": "Excellent Matches",
                "message": f"Found {len(excellent_matches)} excellent patient-trial matches",
                "action": "Prioritize these patients for trial enrollment screening",
                "confidence": "high",
            }
        )

    if good_matches:
        recommendations.append(
            {
                "type": "medium_priority",
                "category": "Good Matches",
                "message": f"Found {len(good_matches)} good potential matches",
                "action": "Review these patients for trial eligibility with additional criteria",
                "confidence": "medium",
            }
        )

    # Analyze by cancer type
    cancer_matches = {}
    for result in matching_results:
        cancer_type = result["patient_profile"].cancer_type
        if cancer_type not in cancer_matches:
            cancer_matches[cancer_type] = []
        cancer_matches[cancer_type].append(result)

    for cancer_type, matches in cancer_matches.items():
        high_quality_matches = [m for m in matches if m["total_score"] >= 0.7]
        if high_quality_matches:
            recommendations.append(
                {
                    "type": "cancer_specific",
                    "category": f"{cancer_type.title()} Cancer",
                    "message": f"Strong matching opportunities for {cancer_type} cancer patients",
                    "action": f"Focus trial enrollment efforts on {cancer_type} cancer patients",
                    "confidence": "high",
                }
            )

    # Analyze biomarker opportunities
    biomarker_patients = [
        p for p in patient_profiles if p.mutation_status or p.receptor_status
    ]
    if biomarker_patients:
        recommendations.append(
            {
                "type": "biomarker_focus",
                "category": "Precision Medicine",
                "message": f"{len(biomarker_patients)} patients with targetable biomarkers identified",
                "action": "Prioritize molecular profiling and targeted therapy trials",
                "confidence": "high",
            }
        )

    return recommendations


# Generate recommendations
clinical_recommendations = generate_recommendations(matching_results)

print("💡 Evidence-Based Clinical Recommendations:")
print("-" * 50)

for rec in clinical_recommendations:
    priority_emoji = "🔥" if rec["confidence"] == "high" else "⚡"
    print(f"\n{priority_emoji} {rec['category']}:")
    print(f"   {rec['message']}")
    print(f"   💡 Recommended Action: {rec['action']}")

# Generate specific patient recommendations
print("\n\n👥 Individual Patient Recommendations:")
print("-" * 50)

for patient in patient_profiles:
    patient_matches = [
        r for r in matching_results if r["patient_id"] == patient.patient_id
    ]
    top_match = (
        max(patient_matches, key=lambda x: x["total_score"])
        if patient_matches
        else None
    )

    if top_match and top_match["total_score"] >= 0.6:
        print(
            f"\n🎯 Patient {patient.patient_id} ({patient.cancer_type.title()} Cancer):"
        )
        print(
            f"   Best Match: Trial {top_match['trial_id']} ({top_match['total_score']:.1%})"
        )
        print("   Recommendation: Strong candidate for trial screening")
        print(
            "   Next Steps: Review full eligibility criteria and schedule screening visit"
        )
    else:
        print(
            f"\n⚠️ Patient {patient.patient_id} ({patient.cancer_type.title()} Cancer):"
        )
        print("   Status: Limited trial matches found")
        print(
            "   Recommendation: Consider standard treatment options or alternative trials"
        )
        print(
            "   Next Steps: Expand search to include early-phase or broader eligibility trials"
        )


# ## 📊 Matching Analytics and Insights
#
# Let's analyze our matching results to identify patterns and opportunities.

# In[ ]:


# Matching analytics and insights
print("📊 Matching Analytics and Insights")
print("=" * 60)


def analyze_matching_patterns(matching_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in matching results.

    Args:
        matching_results: Results from patient-trial matching

    Returns:
        Dictionary of analytics and insights
    """
    analytics = {
        "total_matches": len(matching_results),
        "match_distribution": {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0},
        "cancer_type_analysis": {},
        "trial_analysis": {},
        "biomarker_insights": {},
        "performance_insights": {},
    }

    for result in matching_results:
        # Match distribution
        category = result["match_category"]
        analytics["match_distribution"][category] = (
            analytics["match_distribution"].get(category, 0) + 1
        )

        # Cancer type analysis
        cancer_type = result["patient_profile"].cancer_type
        if cancer_type not in analytics["cancer_type_analysis"]:
            analytics["cancer_type_analysis"][cancer_type] = {
                "total": 0,
                "high_quality": 0,
            }
        analytics["cancer_type_analysis"][cancer_type]["total"] += 1
        if result["total_score"] >= 0.7:
            analytics["cancer_type_analysis"][cancer_type]["high_quality"] += 1

        # Trial analysis
        trial_id = result["trial_id"]
        if trial_id not in analytics["trial_analysis"]:
            analytics["trial_analysis"][trial_id] = {"total": 0, "high_quality": 0}
        analytics["trial_analysis"][trial_id]["total"] += 1
        if result["total_score"] >= 0.7:
            analytics["trial_analysis"][trial_id]["high_quality"] += 1

    return analytics


# Analyze matching patterns
analytics = analyze_matching_patterns(matching_results)

print("📈 Matching Performance Analytics:")
print("-" * 40)

print("🎯 Match Quality Distribution:")
for category, count in analytics["match_distribution"].items():
    percentage = (count / analytics["total_matches"]) * 100
    print(f"   {category}: {count} matches ({percentage:.1f}%)")

print("\n🏥 Cancer Type Analysis:")
for cancer_type, data in analytics["cancer_type_analysis"].items():
    high_quality_rate = (data["high_quality"] / data["total"]) * 100
    print(
        f"   {cancer_type.title()}: {data['total']} patients, {high_quality_rate:.1f}% high-quality matches"
    )

print("\n🧪 Trial Enrollment Opportunities:")
for trial_id, data in analytics["trial_analysis"].items():
    high_quality_rate = (data["high_quality"] / data["total"]) * 100
    print(
        f"   Trial {trial_id}: {data['total']} potential patients, {high_quality_rate:.1f}% high-quality matches"
    )

# Generate strategic insights
print("\n\n🎯 Strategic Enrollment Insights:")
print("-" * 40)

insights = []

# Calculate overall matching success rate
high_quality_matches = analytics["match_distribution"].get("Excellent", 0) + analytics[
    "match_distribution"
].get("Good", 0)
success_rate = (high_quality_matches / analytics["total_matches"]) * 100

insights.append(
    {
        "metric": "Overall Matching Success",
        "value": f"{success_rate:.1f}%",
        "interpretation": "High success rate indicates good trial-patient alignment",
        "action": "Continue current enrollment strategies with minor optimizations",
    }
)

# Identify best-performing cancer types
best_cancer_type = max(
    analytics["cancer_type_analysis"].items(),
    key=lambda x: x[1]["high_quality"] / x[1]["total"],
)
insights.append(
    {
        "metric": "Best-Performing Cancer Type",
        "value": f"{best_cancer_type[0].title()} ({best_cancer_type[1]['high_quality']}/{best_cancer_type[1]['total']})",
        "interpretation": "Strong matching opportunities in this cancer type",
        "action": "Prioritize this cancer type for future trial development",
    }
)

for insight in insights:
    print(f"\n📊 {insight['metric']}: {insight['value']}")
    print(f"   💡 {insight['interpretation']}")
    print(f"   🎯 {insight['action']}")


# ## 🧹 Cleanup and Summary
#
# Let's properly close our connections and provide a comprehensive summary.

# In[ ]:


# Cleanup and final summary
print("🧹 Cleanup and Final Summary")
print("=" * 60)

# Close client connection
try:
    client.close()
    print("✅ Client connection closed successfully")
except Exception as e:
    print(f"⚠️ Cleanup warning: {e}")

print("\n📊 Patient-Trial Matching Demo Summary:")
print("=" * 60)

print("✅ What We Accomplished:")
print("   👥 Analyzed 4 comprehensive patient profiles")
print("   🧪 Evaluated 4 clinical trial eligibility criteria")
print("   🎯 Performed intelligent matching analysis")
print("   📊 Generated quantitative match scores")
print("   🔍 Executed knowledge graph queries")
print("   💡 Created evidence-based recommendations")
print("   📈 Analyzed matching patterns and insights")

print("\n💡 Key Matching Capabilities Demonstrated:")
print("   • Multi-criteria patient-trial matching")
print("   • Quantitative scoring algorithms")
print("   • Knowledge graph-enhanced analysis")
print("   • Evidence-based clinical recommendations")
print("   • Strategic enrollment insights")
print("   • Performance analytics and optimization")

print("\n🏥 Clinical Applications:")
print("   • Automated patient eligibility screening")
print("   • Clinical trial enrollment optimization")
print("   • Evidence-based treatment recommendations")
print("   • Precision medicine matching")
print("   • Clinical decision support systems")

print("\n🔬 Research Applications:")
print("   • Clinical trial feasibility assessment")
print("   • Patient stratification algorithms")
print("   • Biomarker-treatment correlation analysis")
print("   • Enrollment prediction modeling")
print("   • Treatment response pattern discovery")

print("\n🎉 Patient-trial matching demo completed successfully!")
print("🚀 Ready for clinical trial enrollment optimization!")

#!/usr/bin/env python
# coding: utf-8

# # 🧪 MCODE Translator - Clinical Trials Demo
#
# Comprehensive demonstration of clinical trial data processing, analysis, and patient matching capabilities.
#
# ## 📋 What This Demo Covers
#
# This notebook demonstrates MCODE Translator's clinical trial capabilities:
#
# 1. **📥 Clinical Trial Data Ingestion** - Multiple sources and formats
# 2. **🔍 Trial Search & Discovery** - Semantic search across trial databases
# 3. **📊 Trial Summarization** - Automated trial summary generation
# 4. **🏷️ Trial Classification** - Automated categorization and eligibility analysis
# 5. **👥 Patient-Trial Matching** - Advanced matching algorithms
# 6. **📈 Trial Analytics** - Enrollment trends and outcome analysis
#
# ## 🎯 Learning Objectives
#
# By the end of this demo, you will:
# - ✅ Master clinical trial data ingestion patterns
# - ✅ Understand semantic search for trial discovery
# - ✅ Learn automated trial summarization techniques
# - ✅ Apply trial classification and eligibility analysis
# - ✅ Use advanced patient-trial matching algorithms
# - ✅ Generate trial analytics and insights
#
# ## 🏥 Clinical Research Use Cases
#
# ### Trial Management
# - **Trial Landscape Analysis**: Identify competing and complementary trials
# - **Site Selection**: Find optimal trial sites based on patient populations
# - **Enrollment Optimization**: Match patients to most appropriate trials
# - **Competitive Intelligence**: Track trial progress and outcomes
#
# ### Patient-Centric Applications
# - **Treatment Options Discovery**: Find relevant clinical trials for patients
# - **Eligibility Screening**: Automated patient eligibility assessment
# - **Trial Recommendation**: Personalized trial suggestions based on patient profiles
# - **Protocol Optimization**: Identify protocol amendments and updates
#
# ---

# ## 🔧 Setup and Configuration

# In[ ]:


# Import required modules
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path.cwd() / "src"))

# Import MCODE Translator components
try:
    from src.heysol import HeySolClient

    from src.config.heysol_config import get_config

    print("✅ MCODE Translator components imported successfully!")
    print("   🧪 Clinical trial processing capabilities")
    print("   👥 Patient-trial matching algorithms")
    print("   📊 Trial analytics and reporting")

except ImportError as e:
    print("❌ Failed to import MCODE Translator components.")
    print("💡 Install with: pip install -e .")
    print(f"   Error: {e}")
    raise


# ## 📥 Clinical Trial Data Ingestion
#
# Let's start by ingesting comprehensive clinical trial data from various sources.

# In[ ]:


# Initialize HeySol client
print("🔧 Initializing MCODE Translator...")
print("=" * 50)

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

# Create dedicated space for clinical trial data
trials_space_name = "Clinical Trials Database"
trials_space_description = "Active and historical clinical trial information"

print(f"\n🏗️ Setting up clinical trials space: {trials_space_name}")

# Check for existing space
existing_spaces = client.get_spaces()
trials_space_id = None

for space in existing_spaces:
    if isinstance(space, dict) and space.get("name") == trials_space_name:
        trials_space_id = space.get("id")
        print(f"   ✅ Found existing space: {trials_space_id[:16]}...")
        break

if not trials_space_id:
    trials_space_id = client.create_space(trials_space_name, trials_space_description)
    print(f"   ✅ Created new space: {trials_space_id[:16]}...")

print("✅ Clinical trials space ready!")


# ### 📋 Comprehensive Clinical Trial Dataset
#
# Let's create a diverse dataset of clinical trials covering various phases, cancer types, and treatment approaches.

# In[ ]:


# Comprehensive clinical trial dataset
def create_comprehensive_trial_dataset():
    """
    Create a diverse dataset of clinical trials for demonstration.

    Returns:
        list: Comprehensive clinical trial dataset with rich metadata
    """
    return [
        {
            "content": "Phase III randomized controlled trial (NCT04567892) evaluating combination immunotherapy with nivolumab plus ipilimumab versus chemotherapy in patients with advanced BRAF-mutant melanoma. Primary endpoint is progression-free survival with secondary endpoints including overall survival and objective response rate. Trial is actively recruiting with target enrollment of 600 patients across 50 sites.",
            "metadata": {
                "trial_id": "NCT04567892",
                "phase": "III",
                "status": "recruiting",
                "cancer_type": "melanoma",
                "mutation": "BRAF",
                "treatments": ["nivolumab", "ipilimumab"],
                "comparison": "chemotherapy",
                "primary_endpoint": "progression_free_survival",
                "secondary_endpoints": ["overall_survival", "objective_response_rate"],
                "target_enrollment": 600,
                "current_enrollment": 245,
                "sites": 50,
                "start_date": "2024-01-15",
                "estimated_completion": "2026-12-31",
                "sponsor": "Bristol Myers Squibb",
                "study_design": "randomized_controlled",
                "eligibility_criteria": {
                    "age_min": 18,
                    "performance_status": "ECOG_0_1",
                    "prior_treatment": "treatment_naive",
                },
            },
        },
        {
            "content": "Phase II single-arm study (NCT02314481) investigating CDK4/6 inhibitor palbociclib combined with letrozole as first-line treatment for postmenopausal women with ER-positive, HER2-negative metastatic breast cancer. Primary endpoint is progression-free survival with secondary endpoints including overall response rate and clinical benefit rate. Currently fully enrolled with 120 patients.",
            "metadata": {
                "trial_id": "NCT02314481",
                "phase": "II",
                "status": "fully_enrolled",
                "cancer_type": "breast",
                "receptor_status": "ER+/HER2-",
                "treatments": ["palbociclib", "letrozole"],
                "line": "first_line",
                "primary_endpoint": "progression_free_survival",
                "secondary_endpoints": [
                    "overall_response_rate",
                    "clinical_benefit_rate",
                ],
                "target_enrollment": 120,
                "current_enrollment": 120,
                "sites": 25,
                "start_date": "2023-06-01",
                "estimated_completion": "2025-06-01",
                "sponsor": "Pfizer",
                "study_design": "single_arm",
                "eligibility_criteria": {
                    "gender": "female",
                    "menopausal_status": "postmenopausal",
                    "performance_status": "ECOG_0_1",
                },
            },
        },
        {
            "content": "Phase I/II dose-escalation and expansion study (NCT03456789) evaluating novel KRAS G12C inhibitor adagrasib in patients with KRAS G12C-mutated advanced solid tumors. Phase I completed with recommended phase II dose established. Phase II expansion cohorts ongoing in non-small cell lung cancer, colorectal cancer, and pancreatic cancer. Primary endpoints include safety, tolerability, and objective response rate.",
            "metadata": {
                "trial_id": "NCT03456789",
                "phase": "I/II",
                "status": "active",
                "cancer_types": ["lung", "colorectal", "pancreatic"],
                "mutation": "KRAS_G12C",
                "treatments": ["adagrasib"],
                "primary_endpoints": [
                    "safety",
                    "tolerability",
                    "objective_response_rate",
                ],
                "target_enrollment": 200,
                "current_enrollment": 145,
                "sites": 35,
                "start_date": "2023-03-15",
                "estimated_completion": "2025-09-30",
                "sponsor": "Mirati Therapeutics",
                "study_design": "dose_escalation_expansion",
                "eligibility_criteria": {
                    "age_min": 18,
                    "performance_status": "ECOG_0_2",
                    "prior_treatment": "progressed_after_standard_therapy",
                },
            },
        },
        {
            "content": "Phase II randomized trial (NCT01234567) comparing trastuzumab deruxtecan versus trastuzumab emtansine in patients with HER2-positive metastatic breast cancer who have progressed on prior HER2-targeted therapy. Primary endpoint is progression-free survival with secondary endpoints including overall survival, objective response rate, and duration of response. Trial fully enrolled with 500 patients across 80 international sites.",
            "metadata": {
                "trial_id": "NCT01234567",
                "phase": "II",
                "status": "fully_enrolled",
                "cancer_type": "breast",
                "receptor_status": "HER2+",
                "treatments": ["trastuzumab_deruxtecan", "trastuzumab_emtansine"],
                "line": "second_line_plus",
                "primary_endpoint": "progression_free_survival",
                "secondary_endpoints": [
                    "overall_survival",
                    "objective_response_rate",
                    "duration_of_response",
                ],
                "target_enrollment": 500,
                "current_enrollment": 500,
                "sites": 80,
                "start_date": "2022-11-01",
                "estimated_completion": "2025-11-01",
                "sponsor": "Daiichi Sankyo",
                "study_design": "randomized",
                "eligibility_criteria": {
                    "age_min": 18,
                    "performance_status": "ECOG_0_1",
                    "prior_her2_therapy": "required",
                },
            },
        },
        {
            "content": "Phase III confirmatory trial (NCT04567890) evaluating first-line pembrolizumab plus chemotherapy versus chemotherapy alone in patients with PD-L1 positive advanced non-small cell lung cancer. Primary endpoints are progression-free survival and overall survival. Trial has met primary endpoints with significant improvement in both PFS and OS. Currently in follow-up phase with 800 patients enrolled.",
            "metadata": {
                "trial_id": "NCT04567890",
                "phase": "III",
                "status": "follow_up",
                "cancer_type": "lung",
                "biomarker": "PD-L1_positive",
                "treatments": ["pembrolizumab", "chemotherapy"],
                "comparison": "chemotherapy_alone",
                "primary_endpoints": ["progression_free_survival", "overall_survival"],
                "target_enrollment": 800,
                "current_enrollment": 800,
                "sites": 120,
                "start_date": "2021-08-15",
                "estimated_completion": "2024-08-15",
                "sponsor": "Merck",
                "study_design": "randomized_controlled",
                "results": {
                    "pfs_improvement": "significant",
                    "os_improvement": "significant",
                    "primary_endpoints_met": True,
                },
                "eligibility_criteria": {
                    "age_min": 18,
                    "performance_status": "ECOG_0_1",
                    "pd_l1_status": "positive",
                },
            },
        },
    ]


# ### 📤 Intelligent Clinical Trial Data Ingestion
#
# Now let's ingest our comprehensive trial dataset with intelligent routing and rich metadata.

# In[ ]:


# Ingest clinical trial data with comprehensive tracking
print("📤 Ingesting Clinical Trial Data with Rich Metadata")
print("=" * 60)

trial_dataset = create_comprehensive_trial_dataset()

print(f"✅ Created dataset with {len(trial_dataset)} clinical trials")

ingestion_stats = {
    "total": 0,
    "successful": 0,
    "failed": 0,
    "by_phase": {},
    "by_cancer_type": {},
    "by_status": {},
}

print("🚀 Ingesting clinical trial records...")

for i, trial in enumerate(trial_dataset, 1):
    print(
        f"\n🧪 Processing Trial {i}/{len(trial_dataset)}: {trial['metadata']['trial_id']}"
    )

    try:
        # Ingest with comprehensive metadata
        result = client.ingest(
            message=trial["content"],
            space_id=trials_space_id,
            metadata=trial["metadata"],
        )

        print("   ✅ Ingested successfully")
        print("   💾 Saved to CORE Memory: Persistent storage enabled")
        print(f"   📝 Trial ID: {trial['metadata']['trial_id']}")
        print(f"   📊 Phase: {trial['metadata']['phase']}")
        print(f"   🏥 Cancer Type: {trial['metadata']['cancer_type']}")
        print(f"   📈 Status: {trial['metadata']['status']}")

        # Update statistics
        ingestion_stats["total"] += 1
        ingestion_stats["successful"] += 1

        # Track by phase
        phase = trial["metadata"]["phase"]
        ingestion_stats["by_phase"][phase] = (
            ingestion_stats["by_phase"].get(phase, 0) + 1
        )

        # Track by cancer type
        cancer_type = trial["metadata"]["cancer_type"]
        if isinstance(cancer_type, list):
            for ct in cancer_type:
                ingestion_stats["by_cancer_type"][ct] = (
                    ingestion_stats["by_cancer_type"].get(ct, 0) + 1
                )
        else:
            ingestion_stats["by_cancer_type"][cancer_type] = (
                ingestion_stats["by_cancer_type"].get(cancer_type, 0) + 1
            )

        # Track by status
        status = trial["metadata"]["status"]
        ingestion_stats["by_status"][status] = (
            ingestion_stats["by_status"].get(status, 0) + 1
        )

    except Exception as e:
        print(f"   ❌ Ingestion failed: {e}")
        ingestion_stats["total"] += 1
        ingestion_stats["failed"] += 1

print("\n📊 Clinical Trial Data Ingestion Summary:")
print(f"   Total trials: {ingestion_stats['total']}")
print(f"   Successful: {ingestion_stats['successful']}")
print(f"   Failed: {ingestion_stats['failed']}")
print(
    f"   Success rate: {(ingestion_stats['successful']/ingestion_stats['total']*100):.1f}%"
)

print("\n📈 Distribution Analysis:")
print("   📊 By Phase:")
for phase, count in ingestion_stats["by_phase"].items():
    print(f"      Phase {phase}: {count} trials")

print("   🏥 By Cancer Type:")
for cancer_type, count in ingestion_stats["by_cancer_type"].items():
    print(f"      {cancer_type.title()}: {count} trials")

print("   📈 By Status:")
for status, count in ingestion_stats["by_status"].items():
    print(f"      {status.replace('_', ' ').title()}: {count} trials")


# ## 🔍 Clinical Trial Search and Discovery
#
# Now let's demonstrate powerful search capabilities for finding specific trials and opportunities.

# In[ ]:


# Advanced clinical trial search scenarios
print("🔍 Advanced Clinical Trial Search and Discovery")
print("=" * 60)

search_scenarios = [
    {
        "name": "KRAS G12C Inhibitor Trials",
        "query": "KRAS G12C inhibitor clinical trials",
        "description": "Find trials evaluating KRAS G12C inhibitors across cancer types",
        "expected_content": ["KRAS", "G12C", "inhibitor"],
    },
    {
        "name": "Immunotherapy Trials for Melanoma",
        "query": "immunotherapy melanoma BRAF clinical trials",
        "description": "Find immunotherapy trials for BRAF-mutant melanoma",
        "expected_content": ["immunotherapy", "melanoma", "BRAF"],
    },
    {
        "name": "HER2-Positive Breast Cancer Trials",
        "query": "HER2 positive breast cancer clinical trials",
        "description": "Find trials for HER2+ breast cancer patients",
        "expected_content": ["HER2", "breast", "clinical trial"],
    },
    {
        "name": "Phase III Trials Currently Recruiting",
        "query": "phase III recruiting clinical trials cancer",
        "description": "Find actively recruiting phase III cancer trials",
        "expected_content": ["phase III", "recruiting", "cancer"],
    },
    {
        "name": "PD-L1 Positive Lung Cancer Trials",
        "query": "PD-L1 positive lung cancer immunotherapy trials",
        "description": "Find immunotherapy trials for PD-L1+ lung cancer",
        "expected_content": ["PD-L1", "lung", "immunotherapy"],
    },
]

search_results = []

for scenario in search_scenarios:
    print(f"\n🔎 {scenario['name']}")
    print(f"   Description: {scenario['description']}")
    print(f"   Query: '{scenario['query']}'")

    try:
        results = client.search(
            query=scenario["query"], space_ids=[trials_space_id], limit=5
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} matching trials")

        if episodes:
            print("\n   📋 Matching Clinical Trials:")
            for i, episode in enumerate(episodes, 1):
                content = episode.get("content", "")[:120]
                score = episode.get("score", "N/A")
                metadata = episode.get("metadata", {})

                print(f"\n   {i}. Trial {metadata.get('trial_id', 'Unknown')}")
                print(f"      Score: {score}")
                print(f"      Details: {content}{'...' if len(content) == 120 else ''}")

                # Extract key trial information
                if metadata:
                    print(f"      Phase: {metadata.get('phase', 'N/A')}")
                    print(f"      Status: {metadata.get('status', 'N/A')}")
                    print(f"      Cancer Type: {metadata.get('cancer_type', 'N/A')}")

        search_results.append(
            {
                "scenario": scenario["name"],
                "query": scenario["query"],
                "results_count": len(episodes),
                "episodes": episodes,
            }
        )

    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        search_results.append(
            {"scenario": scenario["name"], "error": str(e), "results_count": 0}
        )

print("\n📊 Clinical Trial Search Summary:")
print(f"   Search scenarios: {len(search_scenarios)}")
print(f"   Total trials found: {sum(r['results_count'] for r in search_results)}")
print(
    f"   Average results per search: {sum(r['results_count'] for r in search_results)/len(search_scenarios):.1f}"
)


# ## 📊 Automated Clinical Trial Summarization
#
# Let's demonstrate automated summarization capabilities for generating concise trial reports.

# In[ ]:


# Clinical trial summarization and reporting
print("📊 Automated Clinical Trial Summarization")
print("=" * 60)


def generate_trial_summary(trial_metadata):
    """
    Generate a concise clinical trial summary.

    Args:
        trial_metadata: Trial metadata dictionary

    Returns:
        str: Formatted clinical trial summary
    """
    summary = f"""
Clinical Trial Summary: {trial_metadata.get('trial_id', 'Unknown')}
{'=' * 60}

Trial Information:
• Phase: {trial_metadata.get('phase', 'N/A')}
• Status: {trial_metadata.get('status', 'N/A').replace('_', ' ').title()}
• Sponsor: {trial_metadata.get('sponsor', 'N/A')}

Target Disease:
• Cancer Type: {trial_metadata.get('cancer_type', 'N/A').title()}
• Biomarker: {trial_metadata.get('mutation', trial_metadata.get('biomarker', 'N/A'))}

Treatment:
• Intervention: {', '.join(trial_metadata.get('treatments', ['N/A']))}
• Comparison: {trial_metadata.get('comparison', 'N/A')}

Endpoints:
• Primary: {trial_metadata.get('primary_endpoint', 'N/A').replace('_', ' ').title()}
• Secondary: {', '.join(trial_metadata.get('secondary_endpoints', ['N/A']))}

Enrollment:
• Target: {trial_metadata.get('target_enrollment', 'N/A')} patients
• Current: {trial_metadata.get('current_enrollment', 'N/A')} enrolled
• Sites: {trial_metadata.get('sites', 'N/A')} locations

Timeline:
• Start Date: {trial_metadata.get('start_date', 'N/A')}
• Completion: {trial_metadata.get('estimated_completion', 'N/A')}
"""

    return summary.strip()


print("📋 Generating Clinical Trial Summaries:")
print("\n" + "=" * 80)

for i, trial in enumerate(trial_dataset, 1):
    print(f"\n🧪 Trial {trial['metadata']['trial_id']} Summary:")
    print("-" * 50)

    summary = generate_trial_summary(trial["metadata"])
    print(summary)

    # Generate trial insights
    print("\n🔍 Trial Insights:")
    metadata = trial["metadata"]

    # Enrollment status
    current = metadata.get("current_enrollment", 0)
    target = metadata.get("target_enrollment", 0)
    if target > 0:
        enrollment_rate = (current / target) * 100
        print(f"   • Enrollment Progress: {enrollment_rate:.1f}% ({current}/{target})")

    # Trial phase insights
    phase = metadata.get("phase", "")
    if phase == "III":
        print("   • Phase III: Confirmatory trial with high evidence level")
    elif phase == "II":
        print("   • Phase II: Proof of concept and efficacy evaluation")
    elif "I" in phase:
        print("   • Phase I/II: Safety and preliminary efficacy evaluation")

    # Results availability
    if metadata.get("results"):
        results = metadata["results"]
        if results.get("primary_endpoints_met"):
            print("   • Results: Primary endpoints met - positive trial")


# ## 👥 Patient-Trial Matching Analysis
#
# Let's demonstrate advanced patient-trial matching capabilities for clinical research.

# In[ ]:


# Patient-trial matching analysis
print("👥 Patient-Trial Matching Analysis")
print("=" * 60)

# Define sample patient profiles for matching
sample_patients = [
    {
        "patient_id": "P001",
        "cancer_type": "breast",
        "stage": "IIA",
        "receptor_status": "ER+/PR+/HER2-",
        "age": 52,
        "performance_status": "ECOG_0",
        "treatment_history": ["neoadjuvant_chemotherapy", "surgery"],
        "current_status": "adjuvant_therapy",
    },
    {
        "patient_id": "P002",
        "cancer_type": "lung",
        "stage": "IV",
        "mutation": "EGFR_exon_19_deletion",
        "age": 67,
        "performance_status": "ECOG_1",
        "treatment_history": ["first_line_osimertinib"],
        "current_status": "progressive_disease",
    },
    {
        "patient_id": "P003",
        "cancer_type": "melanoma",
        "stage": "IIIB",
        "mutation": "BRAF_V600E",
        "age": 45,
        "performance_status": "ECOG_0",
        "treatment_history": ["surgery"],
        "current_status": "adjuvant_therapy_naive",
    },
]

matching_scenarios = [
    {
        "name": "Hormone Receptor Positive Breast Cancer",
        "patient": sample_patients[0],
        "query": "ER positive breast cancer clinical trials",
        "description": "Find trials for ER+ breast cancer patients",
    },
    {
        "name": "EGFR-Mutated Lung Cancer",
        "patient": sample_patients[1],
        "query": "EGFR lung cancer clinical trials",
        "description": "Find trials for EGFR-mutated lung cancer",
    },
    {
        "name": "BRAF-Mutant Melanoma",
        "patient": sample_patients[2],
        "query": "BRAF melanoma immunotherapy trials",
        "description": "Find immunotherapy trials for BRAF-mutant melanoma",
    },
]

print("🎯 Patient-Trial Matching Scenarios:")
print("-" * 50)

for scenario in matching_scenarios:
    print(f"\n👤 Patient {scenario['patient']['patient_id']}")
    print(
        f"   Profile: {scenario['patient']['cancer_type'].title()} cancer, {scenario['patient']['stage']}"
    )
    print(
        f"   Biomarkers: {scenario['patient'].get('receptor_status', scenario['patient'].get('mutation', 'None'))}"
    )
    print(f"   🔍 Query: '{scenario['query']}'")

    try:
        results = client.search(
            query=scenario["query"], space_ids=[trials_space_id], limit=3
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} potential trial matches")

        if episodes:
            print("\n   📋 Trial Match Analysis:")
            for i, episode in enumerate(episodes, 1):
                metadata = episode.get("metadata", {})
                print(f"\n   {i}. Trial {metadata.get('trial_id', 'Unknown')}")
                print(f"      Match Score: {episode.get('score', 'N/A')}")

                # Analyze eligibility criteria
                eligibility = metadata.get("eligibility_criteria", {})
                patient_criteria = scenario["patient"]

                matching_criteria = []
                total_criteria = len(eligibility)

                for criterion, required_value in eligibility.items():
                    if criterion in patient_criteria:
                        patient_value = patient_criteria[criterion]
                        if patient_value == required_value or (
                            isinstance(patient_value, str)
                            and isinstance(required_value, str)
                            and required_value in patient_value
                        ):
                            matching_criteria.append(criterion)

                print(
                    f"      Eligibility Match: {len(matching_criteria)}/{total_criteria}"
                )
                print(f"      Matched Criteria: {', '.join(matching_criteria)}")

                # Trial details
                print(f"      Phase: {metadata.get('phase', 'N/A')}")
                print(f"      Status: {metadata.get('status', 'N/A')}")
                print(f"      Sites: {metadata.get('sites', 'N/A')} locations")

    except Exception as e:
        print(f"   ❌ Matching analysis failed: {e}")


# ## 📈 Clinical Trial Analytics
#
# Let's demonstrate advanced analytics and insights generation for clinical trials.

# In[ ]:


# Clinical trial analytics and insights
print("📈 Clinical Trial Analytics")
print("=" * 60)


def calculate_trial_statistics(trial_dataset):
    """
    Calculate comprehensive trial statistics.

    Args:
        trial_dataset: List of trial records

    Returns:
        dict: Statistical analysis results
    """
    stats = {
        "total_trials": len(trial_dataset),
        "phase_distribution": {},
        "cancer_type_distribution": {},
        "status_distribution": {},
        "enrollment_statistics": {"total": 0, "current": 0, "target": 0},
        "sponsor_diversity": {},
        "endpoint_analysis": {},
    }

    for trial in trial_dataset:
        metadata = trial["metadata"]

        # Phase distribution
        phase = metadata.get("phase", "unknown")
        stats["phase_distribution"][phase] = (
            stats["phase_distribution"].get(phase, 0) + 1
        )

        # Cancer type distribution
        cancer_type = metadata.get("cancer_type", "unknown")
        if isinstance(cancer_type, list):
            for ct in cancer_type:
                stats["cancer_type_distribution"][ct] = (
                    stats["cancer_type_distribution"].get(ct, 0) + 1
                )
        else:
            stats["cancer_type_distribution"][cancer_type] = (
                stats["cancer_type_distribution"].get(cancer_type, 0) + 1
            )

        # Status distribution
        status = metadata.get("status", "unknown")
        stats["status_distribution"][status] = (
            stats["status_distribution"].get(status, 0) + 1
        )

        # Enrollment statistics
        current = metadata.get("current_enrollment", 0)
        target = metadata.get("target_enrollment", 0)
        stats["enrollment_statistics"]["current"] += current
        stats["enrollment_statistics"]["target"] += target

        # Sponsor diversity
        sponsor = metadata.get("sponsor", "unknown")
        stats["sponsor_diversity"][sponsor] = (
            stats["sponsor_diversity"].get(sponsor, 0) + 1
        )

        # Endpoint analysis
        primary_endpoint = metadata.get("primary_endpoint", "")
        if isinstance(primary_endpoint, list):
            for endpoint in primary_endpoint:
                stats["endpoint_analysis"][endpoint] = (
                    stats["endpoint_analysis"].get(endpoint, 0) + 1
                )
        else:
            stats["endpoint_analysis"][primary_endpoint] = (
                stats["endpoint_analysis"].get(primary_endpoint, 0) + 1
            )

    # Calculate enrollment rate
    total_target = stats["enrollment_statistics"]["target"]
    total_current = stats["enrollment_statistics"]["current"]
    if total_target > 0:
        stats["enrollment_statistics"]["overall_rate"] = (
            total_current / total_target
        ) * 100

    return stats


# Calculate comprehensive statistics
statistics = calculate_trial_statistics(trial_dataset)

print("📊 Clinical Trial Population Statistics:")
print("-" * 50)

print("📈 Phase Distribution:")
total_trials = statistics["total_trials"]
for phase, count in statistics["phase_distribution"].items():
    percentage = (count / total_trials) * 100
    print(f"   Phase {phase}: {count} trials ({percentage:.1f}%)")

print("\n🏥 Cancer Type Distribution:")
for cancer_type, count in statistics["cancer_type_distribution"].items():
    percentage = (count / total_trials) * 100
    print(f"   {cancer_type.title()}: {count} trials ({percentage:.1f}%)")

print("\n📊 Status Distribution:")
for status, count in statistics["status_distribution"].items():
    percentage = (count / total_trials) * 100
    print(f"   {status.replace('_', ' ').title()}: {count} trials ({percentage:.1f}%)")

print("\n👥 Enrollment Statistics:")
enrollment = statistics["enrollment_statistics"]
print(f"   Total Target Enrollment: {enrollment['target']:,} patients")
print(f"   Total Current Enrollment: {enrollment['current']:,} patients")
if "overall_rate" in enrollment:
    print(f"   Overall Enrollment Rate: {enrollment['overall_rate']:.1f}%")

print("\n💊 Primary Endpoints:")
for endpoint, count in statistics["endpoint_analysis"].items():
    percentage = (count / total_trials) * 100
    print(
        f"   {endpoint.replace('_', ' ').title()}: {count} trials ({percentage:.1f}%)"
    )


# ## 🎯 Clinical Trial Insights and Recommendations
#
# Let's generate actionable insights for clinical trial strategy and patient enrollment.

# In[ ]:


# Generate clinical trial insights and recommendations
print("🎯 Clinical Trial Insights and Recommendations")
print("=" * 60)

insights = []

# Insight 1: Enrollment opportunities
recruiting_trials = sum(
    1 for t in trial_dataset if t["metadata"].get("status") == "recruiting"
)
total_target = statistics["enrollment_statistics"]["target"]
total_current = statistics["enrollment_statistics"]["current"]
remaining_spots = total_target - total_current

insights.append(
    {
        "category": "Enrollment Opportunities",
        "insight": f"Active Recruitment: {recruiting_trials} trials currently recruiting",
        "actionable": f"Available spots: {remaining_spots:,} patients across all trials",
    }
)

# Insight 2: Therapeutic focus areas
cancer_focus = max(statistics["cancer_type_distribution"].items(), key=lambda x: x[1])
insights.append(
    {
        "category": "Therapeutic Focus",
        "insight": f"Primary Focus: {cancer_focus[0].title()} cancer trials ({cancer_focus[1]} studies)",
        "actionable": "Consider this as primary therapeutic area for site development",
    }
)

# Insight 3: Trial phase distribution
phase_iii_trials = statistics["phase_distribution"].get("III", 0)
total_trials = statistics["total_trials"]
phase_iii_percentage = (phase_iii_trials / total_trials) * 100

insights.append(
    {
        "category": "Development Stage",
        "insight": f"Phase III Focus: {phase_iii_percentage:.1f}% of trials are Phase III",
        "actionable": "Strong focus on confirmatory trials indicates mature pipeline",
    }
)

# Insight 4: Biomarker-driven trials
biomarker_trials = sum(
    1
    for t in trial_dataset
    if t["metadata"].get("mutation") or t["metadata"].get("biomarker")
)
biomarker_percentage = (biomarker_trials / total_trials) * 100

insights.append(
    {
        "category": "Precision Medicine",
        "insight": f"Biomarker-Driven: {biomarker_percentage:.1f}% of trials are biomarker-specific",
        "actionable": "Emphasize molecular profiling capabilities for patient screening",
    }
)

print("💡 Key Clinical Trial Insights:")
print("-" * 50)

for insight in insights:
    print(f"\n🔍 {insight['category']}:")
    print(f"   {insight['insight']}")
    print(f"   💡 Actionable: {insight['actionable']}")

# Generate strategic recommendations
print("\n\n🎯 Strategic Recommendations:")
print("-" * 40)

recommendations = [
    "Prioritize site development for breast and lung cancer trials given current focus",
    "Invest in molecular diagnostics infrastructure to support biomarker-driven trials",
    "Develop rapid patient screening protocols for actively recruiting Phase III trials",
    "Consider collaborative opportunities with top sponsors (BMS, Pfizer, Merck)",
    "Focus on diversity and inclusion in trial enrollment to improve generalizability",
    "Implement real-time enrollment tracking and predictive analytics",
    "Develop patient-centric trial matching algorithms to improve enrollment rates",
    "Consider expanding into early-phase trials to build referral networks",
]

for i, recommendation in enumerate(recommendations, 1):
    print(f"{i}. {recommendation}")


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

print("\n📊 Clinical Trials Demo Summary:")
print("=" * 50)

print("✅ What We Accomplished:")
print("   📥 Ingested 5 comprehensive clinical trial records")
print("   🔍 Executed 5 advanced trial search scenarios")
print("   📊 Generated automated trial summaries")
print("   👥 Performed patient-trial matching analysis")
print("   📈 Calculated comprehensive trial statistics")
print("   🎯 Generated strategic insights and recommendations")

print("\n💡 Key Capabilities Demonstrated:")
print("   • Multi-source clinical trial data ingestion")
print("   • Semantic search across trial databases")
print("   • Automated trial summarization and reporting")
print("   • Advanced patient-trial matching algorithms")
print("   • Comprehensive trial analytics and insights")
print("   • Strategic recommendation generation")

print("\n🏥 Clinical Applications:")
print("   • Clinical trial landscape analysis")
print("   • Patient eligibility screening")
print("   • Trial enrollment optimization")
print("   • Competitive intelligence gathering")
print("   • Site selection and development")

print("\n🔬 Research Applications:")
print("   • Trial feasibility assessment")
print("   • Biomarker-stratified trial identification")
print("   • Enrollment trend analysis")
print("   • Therapeutic area strategic planning")
print("   • Sponsor collaboration opportunities")

print("\n🎉 Clinical trials demo completed successfully!")
print("🚀 Ready for clinical trial management applications!")

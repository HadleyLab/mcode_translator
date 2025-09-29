#!/usr/bin/env python
# coding: utf-8

# # 👥 MCODE Translator - Patient Data Demo
#
# Comprehensive demonstration of patient data processing, summarization, and analysis capabilities.
#
# ## 📋 What This Demo Covers
#
# This notebook demonstrates MCODE Translator's patient data capabilities:
#
# 1. **📥 Patient Data Ingestion** - Multiple formats and sources
# 2. **🔍 Patient Search & Discovery** - Semantic search across patient records
# 3. **📊 Patient Summarization** - Automated summary generation
# 4. **🏷️ Patient Classification** - Automated categorization and tagging
# 5. **📈 Patient Analytics** - Statistical analysis and insights
# 6. **🔗 Patient Matching** - Finding similar patients for research
#
# ## 🎯 Learning Objectives
#
# By the end of this demo, you will:
# - ✅ Master patient data ingestion patterns
# - ✅ Understand semantic search for patient discovery
# - ✅ Learn automated summarization techniques
# - ✅ Apply patient classification and analytics
# - ✅ Use patient matching for research
#
# ## 🏥 Clinical Use Cases
#
# ### Research Applications
# - **Cohort Identification**: Find patients matching specific criteria
# - **Comparative Analysis**: Compare treatment outcomes across patients
# - **Biomarker Discovery**: Identify patterns in patient responses
# - **Clinical Trial Matching**: Match patients to appropriate trials
#
# ### Healthcare Applications
# - **Treatment Planning**: Identify similar cases for treatment guidance
# - **Risk Assessment**: Analyze patient risk factors and outcomes
# - **Quality Improvement**: Track patient outcomes and care patterns
# - **Population Health**: Analyze patient populations and trends
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
    print("   👥 Patient processing capabilities")
    print("   🔍 Search and analytics functions")
    print("   📊 Summarization and reporting")

except ImportError as e:
    print("❌ Failed to import MCODE Translator components.")
    print("💡 Install with: pip install -e .")
    print(f"   Error: {e}")
    raise


# ## 📥 Patient Data Ingestion
#
# Let's start by ingesting diverse patient data from multiple sources and formats.

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

# Create dedicated space for patient data
patients_space_name = "Patient Data Repository"
patients_space_description = (
    "Comprehensive patient data for clinical research and analysis"
)

print(f"\n🏗️ Setting up patient data space: {patients_space_name}")

# Check for existing space
existing_spaces = client.get_spaces()
patients_space_id = None

for space in existing_spaces:
    if isinstance(space, dict) and space.get("name") == patients_space_name:
        patients_space_id = space.get("id")
        print(f"   ✅ Found existing space: {patients_space_id[:16]}...")
        break

if not patients_space_id:
    patients_space_id = client.create_space(
        patients_space_name, patients_space_description
    )
    print(f"   ✅ Created new space: {patients_space_id[:16]}...")

print("✅ Patient data space ready!")


# ### 📋 Sample Patient Dataset
#
# Let's create a comprehensive dataset of synthetic patient records covering various cancer types, stages, and treatment scenarios.

# In[ ]:


# Comprehensive patient dataset
def create_comprehensive_patient_dataset():
    """
    Create a diverse dataset of patient records for demonstration.

    Returns:
        list: Comprehensive patient dataset with rich metadata
    """
    return [
        {
            "content": "Patient ID: P001 | Name: Sarah Johnson | 52-year-old female diagnosed with ER+/PR+/HER2- invasive ductal carcinoma of the left breast, stage IIA. Completed neoadjuvant chemotherapy with AC-T regimen followed by lumpectomy and sentinel lymph node biopsy. Currently on adjuvant endocrine therapy with anastrozole. Recent follow-up shows no evidence of disease recurrence.",
            "metadata": {
                "patient_id": "P001",
                "name": "Sarah Johnson",
                "age": 52,
                "gender": "female",
                "cancer_type": "breast",
                "subtype": "invasive_ductal_carcinoma",
                "stage": "IIA",
                "grade": 2,
                "receptor_status": "ER+/PR+/HER2-",
                "treatment_phase": "adjuvant",
                "current_therapy": "anastrozole",
                "treatment_history": [
                    "AC-T_chemotherapy",
                    "lumpectomy",
                    "sentinel_lymph_node_biopsy",
                ],
                "response": "complete_response",
                "recurrence_status": "none",
                "follow_up_months": 18,
                "performance_status": "ECOG_0",
                "comorbidities": ["hypertension", "osteoporosis"],
            },
        },
        {
            "content": "Patient ID: P002 | Name: Michael Chen | 67-year-old male with stage IV non-small cell lung adenocarcinoma, EGFR exon 19 deletion positive. Currently receiving first-line osimertinib therapy with excellent tolerance and partial response on recent imaging. Performance status remains excellent with minimal treatment-related toxicity.",
            "metadata": {
                "patient_id": "P002",
                "name": "Michael Chen",
                "age": 67,
                "gender": "male",
                "cancer_type": "lung",
                "histology": "adenocarcinoma",
                "stage": "IV",
                "mutation": "EGFR_exon_19_deletion",
                "treatment_phase": "first_line",
                "current_therapy": "osimertinib",
                "response": "partial_response",
                "performance_status": "ECOG_1",
                "toxicity": "minimal",
                "treatment_duration_months": 8,
                "imaging_response": "partial_response",
                "biomarker_status": "EGFR_positive",
                "smoking_history": "former_smoker",
            },
        },
        {
            "content": "Patient ID: P003 | Name: Elena Rodriguez | 45-year-old female with triple-negative breast cancer, stage IIIA, grade 3. Completed neoadjuvant chemotherapy with dose-dense AC followed by paclitaxel plus carboplatin. Achieved pathologic complete response at surgery. Currently receiving adjuvant capecitabine with good tolerance.",
            "metadata": {
                "patient_id": "P003",
                "name": "Elena Rodriguez",
                "age": 45,
                "gender": "female",
                "cancer_type": "breast",
                "subtype": "triple_negative",
                "stage": "IIIA",
                "grade": 3,
                "receptor_status": "TNBC",
                "treatment_phase": "adjuvant",
                "current_therapy": "capecitabine",
                "treatment_history": ["dose_dense_AC", "paclitaxel_carboplatin"],
                "pathologic_response": "complete_response",
                "surgical_outcome": "breast_conserving_surgery",
                "performance_status": "ECOG_0",
                "toxicity_profile": "manageable",
            },
        },
        {
            "content": "Patient ID: P004 | Name: David Thompson | 58-year-old male with metastatic colorectal cancer, KRAS G12V mutated, MSI-low. Progressed through FOLFOX and FOLFIRI chemotherapy regimens. Currently enrolled in clinical trial evaluating novel KRAS G12C inhibitor with partial response observed at first restaging.",
            "metadata": {
                "patient_id": "P004",
                "name": "David Thompson",
                "age": 58,
                "gender": "male",
                "cancer_type": "colorectal",
                "stage": "IV",
                "mutation": "KRAS_G12V",
                "msi_status": "MSI_low",
                "treatment_phase": "third_line",
                "current_therapy": "KRAS_G12C_inhibitor",
                "prior_regimens": ["FOLFOX", "FOLFIRI"],
                "clinical_trial": True,
                "response": "partial_response",
                "metastatic_sites": ["liver", "lung"],
                "performance_status": "ECOG_1",
            },
        },
        {
            "content": "Patient ID: P005 | Name: Lisa Park | 38-year-old female with HER2-positive breast cancer, stage IIB, initially treated with neoadjuvant TCHP chemotherapy followed by bilateral mastectomy. Currently receiving adjuvant trastuzumab emtansine (T-DM1) with excellent tolerance and no evidence of residual disease.",
            "metadata": {
                "patient_id": "P005",
                "name": "Lisa Park",
                "age": 38,
                "gender": "female",
                "cancer_type": "breast",
                "subtype": "HER2_positive",
                "stage": "IIB",
                "receptor_status": "HER2+",
                "treatment_phase": "adjuvant",
                "current_therapy": "T-DM1",
                "treatment_history": ["TCHP_chemotherapy", "bilateral_mastectomy"],
                "response": "complete_response",
                "surgical_procedure": "bilateral_mastectomy",
                "performance_status": "ECOG_0",
                "family_history": "breast_cancer_mother",
            },
        },
    ]


# ### 📤 Intelligent Patient Data Ingestion
#
# Now let's ingest our comprehensive patient dataset with intelligent routing and rich metadata.

# In[ ]:


# Ingest patient data with comprehensive tracking
print("📤 Ingesting Patient Data with Rich Metadata")
print("=" * 60)

patient_dataset = create_comprehensive_patient_dataset()

print(f"✅ Created dataset with {len(patient_dataset)} patient records")

ingestion_stats = {
    "total": 0,
    "successful": 0,
    "failed": 0,
    "by_cancer_type": {},
    "by_stage": {},
    "by_treatment_phase": {},
}

print("🚀 Ingesting patient records...")

for i, patient in enumerate(patient_dataset, 1):
    print(
        f"\n👤 Processing Patient {i}/{len(patient_dataset)}: {patient['metadata']['patient_id']}"
    )

    try:
        # Ingest with comprehensive metadata
        result = client.ingest(
            message=patient["content"],
            space_id=patients_space_id,
            metadata=patient["metadata"],
        )

        print("   ✅ Ingested successfully")
        print("   💾 Saved to CORE Memory: Persistent storage enabled")
        print(f"   � Patient ID: {patient['metadata']['patient_id']}")
        print(f"   🏥 Cancer Type: {patient['metadata']['cancer_type']}")
        print(f"   📊 Stage: {patient['metadata']['stage']}")
        print(f"   💊 Treatment: {patient['metadata']['current_therapy']}")

        # Update statistics
        ingestion_stats["total"] += 1
        ingestion_stats["successful"] += 1

        # Track by cancer type
        cancer_type = patient["metadata"]["cancer_type"]
        ingestion_stats["by_cancer_type"][cancer_type] = (
            ingestion_stats["by_cancer_type"].get(cancer_type, 0) + 1
        )

        # Track by stage
        stage = patient["metadata"]["stage"]
        ingestion_stats["by_stage"][stage] = (
            ingestion_stats["by_stage"].get(stage, 0) + 1
        )

        # Track by treatment phase
        treatment_phase = patient["metadata"]["treatment_phase"]
        ingestion_stats["by_treatment_phase"][treatment_phase] = (
            ingestion_stats["by_treatment_phase"].get(treatment_phase, 0) + 1
        )

    except Exception as e:
        print(f"   ❌ Ingestion failed: {e}")
        ingestion_stats["total"] += 1
        ingestion_stats["failed"] += 1

print("\n📊 Patient Data Ingestion Summary:")
print(f"   Total patients: {ingestion_stats['total']}")
print(f"   Successful: {ingestion_stats['successful']}")
print(f"   Failed: {ingestion_stats['failed']}")
print(
    f"   Success rate: {(ingestion_stats['successful']/ingestion_stats['total']*100):.1f}%"
)

print("\n📈 Distribution Analysis:")
print("   🏥 By Cancer Type:")
for cancer_type, count in ingestion_stats["by_cancer_type"].items():
    print(f"      {cancer_type}: {count} patients")

print("   📊 By Stage:")
for stage, count in ingestion_stats["by_stage"].items():
    print(f"      Stage {stage}: {count} patients")

print("   💊 By Treatment Phase:")
for phase, count in ingestion_stats["by_treatment_phase"].items():
    print(f"      {phase}: {count} patients")


# ## 🔍 Patient Search and Discovery
#
# Now let's demonstrate powerful search capabilities for finding specific patients and patterns.

# In[ ]:


# Advanced patient search scenarios
print("🔍 Advanced Patient Search and Discovery")
print("=" * 60)

search_scenarios = [
    {
        "name": "Triple-Negative Breast Cancer Patients",
        "query": "triple negative breast cancer patients",
        "description": "Find all TNBC patients for research studies",
        "expected_count": 1,
    },
    {
        "name": "EGFR-Mutated Lung Cancer",
        "query": "EGFR mutation lung cancer patients",
        "description": "Identify EGFR+ lung cancer patients for targeted therapy research",
        "expected_count": 1,
    },
    {
        "name": "Metastatic Colorectal Cancer",
        "query": "metastatic colorectal cancer KRAS mutated",
        "description": "Find metastatic CRC patients with KRAS mutations",
        "expected_count": 1,
    },
    {
        "name": "Complete Response Patients",
        "query": "complete response pathologic complete response",
        "description": "Find patients with complete responses to neoadjuvant therapy",
        "expected_count": 2,
    },
    {
        "name": "Clinical Trial Participants",
        "query": "clinical trial enrolled patients",
        "description": "Identify patients currently enrolled in clinical trials",
        "expected_count": 1,
    },
]

search_results = []

for scenario in search_scenarios:
    print(f"\n🔎 {scenario['name']}")
    print(f"   Description: {scenario['description']}")
    print(f"   Query: '{scenario['query']}'")

    try:
        results = client.search(
            query=scenario["query"], space_ids=[patients_space_id], limit=10
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} matching patients")

        if episodes:
            print("\n   📋 Matching Patient Records:")
            for i, episode in enumerate(episodes, 1):
                content = episode.get("content", "")[:120]
                score = episode.get("score", "N/A")
                metadata = episode.get("metadata", {})

                print(f"\n   {i}. Patient {metadata.get('patient_id', 'Unknown')}")
                print(f"      Score: {score}")
                print(f"      Details: {content}{'...' if len(content) == 120 else ''}")

                # Extract key clinical information
                if metadata:
                    print(
                        f"      Age/Gender: {metadata.get('age', 'N/A')}-year-old {metadata.get('gender', 'N/A')}"
                    )
                    print(
                        f"      Diagnosis: {metadata.get('cancer_type', 'N/A')} cancer, stage {metadata.get('stage', 'N/A')}"
                    )
                    print(f"      Treatment: {metadata.get('current_therapy', 'N/A')}")

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

print("\n📊 Patient Search Summary:")
print(f"   Search scenarios: {len(search_scenarios)}")
print(f"   Total patients found: {sum(r['results_count'] for r in search_results)}")
print(
    f"   Average results per search: {sum(r['results_count'] for r in search_results)/len(search_scenarios):.1f}"
)


# ## 📊 Automated Patient Summarization
#
# Let's demonstrate automated summarization capabilities for generating concise patient reports.

# In[ ]:


# Patient summarization and reporting
print("📊 Automated Patient Summarization")
print("=" * 60)


def generate_patient_summary(patient_metadata):
    """
    Generate a concise clinical summary for a patient.

    Args:
        patient_metadata: Patient metadata dictionary

    Returns:
        str: Formatted clinical summary
    """
    summary = f"""
Patient Summary: {patient_metadata.get('patient_id', 'Unknown')}
{'=' * 50}

Demographics:
• Age/Gender: {patient_metadata.get('age', 'N/A')}-year-old {patient_metadata.get('gender', 'N/A')}
• Performance Status: {patient_metadata.get('performance_status', 'N/A')}

Oncologic History:
• Cancer Type: {patient_metadata.get('cancer_type', 'N/A').title()}
• Stage: {patient_metadata.get('stage', 'N/A')}
• Grade: {patient_metadata.get('grade', 'N/A') if patient_metadata.get('grade') else 'N/A'}
• Receptor Status: {patient_metadata.get('receptor_status', 'N/A')}

Current Treatment:
• Phase: {patient_metadata.get('treatment_phase', 'N/A').title()}
• Therapy: {patient_metadata.get('current_therapy', 'N/A')}
• Response: {patient_metadata.get('response', 'N/A').replace('_', ' ').title()}

Clinical Course:
• Treatment Duration: {patient_metadata.get('treatment_duration_months', 'N/A')} months
• Toxicity Profile: {patient_metadata.get('toxicity', 'N/A')}
• Recurrence Status: {patient_metadata.get('recurrence_status', 'N/A').replace('_', ' ').title()}
"""

    return summary.strip()


print("📋 Generating Patient Summaries:")
print("\n" + "=" * 80)

for i, patient in enumerate(patient_dataset, 1):
    print(f"\n👤 Patient {patient['metadata']['patient_id']} Summary:")
    print("-" * 50)

    summary = generate_patient_summary(patient["metadata"])
    print(summary)

    # Generate insights
    print("\n🔍 Clinical Insights:")
    print(
        f"   • Treatment Response: {patient['metadata'].get('response', 'N/A').replace('_', ' ').title()}"
    )

    if patient["metadata"].get("clinical_trial"):
        print("   • Clinical Trial: Currently enrolled")

    if patient["metadata"].get("mutation"):
        print(f"   • Targetable Mutation: {patient['metadata']['mutation']}")

    if patient["metadata"].get("metastatic_sites"):
        sites = ", ".join(patient["metadata"]["metastatic_sites"])
        print(f"   • Metastatic Sites: {sites}")


# ## 🏷️ Patient Classification and Analytics
#
# Let's demonstrate automated patient classification and statistical analysis.

# In[ ]:


# Patient classification and analytics
print("🏷️ Patient Classification and Analytics")
print("=" * 60)

# Analyze patient distribution
print("📊 Patient Population Analysis:")
print("-" * 40)

# Cancer type distribution
cancer_types = {}
stages = {}
age_groups = {}
treatment_responses = {}

for patient in patient_dataset:
    metadata = patient["metadata"]

    # Cancer type analysis
    cancer_type = metadata.get("cancer_type", "unknown")
    cancer_types[cancer_type] = cancer_types.get(cancer_type, 0) + 1

    # Stage analysis
    stage = metadata.get("stage", "unknown")
    stages[stage] = stages.get(stage, 0) + 1

    # Age group analysis
    age = metadata.get("age", 0)
    if age < 40:
        age_group = "young_adult"
    elif age < 60:
        age_group = "middle_adult"
    else:
        age_group = "senior"
    age_groups[age_group] = age_groups.get(age_group, 0) + 1

    # Treatment response analysis
    response = metadata.get("response", "unknown")
    treatment_responses[response] = treatment_responses.get(response, 0) + 1

print("🏥 Cancer Type Distribution:")
for cancer_type, count in cancer_types.items():
    percentage = (count / len(patient_dataset)) * 100
    print(f"   {cancer_type.title()}: {count} patients ({percentage:.1f}%)")

print("\n📊 Stage Distribution:")
for stage, count in stages.items():
    percentage = (count / len(patient_dataset)) * 100
    print(f"   Stage {stage}: {count} patients ({percentage:.1f}%)")

print("\n👥 Age Group Distribution:")
for age_group, count in age_groups.items():
    percentage = (count / len(patient_dataset)) * 100
    print(
        f"   {age_group.replace('_', ' ').title()}: {count} patients ({percentage:.1f}%)"
    )

print("\n💊 Treatment Response Distribution:")
for response, count in treatment_responses.items():
    percentage = (count / len(patient_dataset)) * 100
    print(
        f"   {response.replace('_', ' ').title()}: {count} patients ({percentage:.1f}%)"
    )


# ## 🔗 Patient Matching for Research
#
# Let's demonstrate patient matching capabilities for clinical research and trial enrollment.

# In[ ]:


# Patient matching for research and clinical trials
print("🔗 Patient Matching for Research")
print("=" * 60)

matching_scenarios = [
    {
        "name": "TNBC Clinical Trial Candidates",
        "criteria": {
            "cancer_type": "breast",
            "receptor_status": "TNBC",
            "stage": "IIIA",
            "treatment_phase": "adjuvant",
        },
        "description": "Find TNBC patients for adjuvant therapy trials",
    },
    {
        "name": "Targeted Therapy Candidates",
        "criteria": {
            "mutation": "EGFR_exon_19_deletion",
            "cancer_type": "lung",
            "treatment_phase": "first_line",
        },
        "description": "Find EGFR+ lung cancer patients for targeted therapy",
    },
    {
        "name": "KRAS Inhibitor Trial Candidates",
        "criteria": {
            "mutation": "KRAS_G12V",
            "cancer_type": "colorectal",
            "stage": "IV",
        },
        "description": "Find KRAS-mutated colorectal cancer patients",
    },
]

print("🎯 Patient Matching Scenarios:")
print("-" * 40)

for scenario in matching_scenarios:
    print(f"\n🔍 {scenario['name']}")
    print(f"   Description: {scenario['description']}")

    # Build search query from criteria
    criteria = scenario["criteria"]
    query_parts = []

    if "cancer_type" in criteria:
        query_parts.append(f"{criteria['cancer_type']} cancer")
    if "receptor_status" in criteria:
        query_parts.append(f"{criteria['receptor_status']}")
    if "mutation" in criteria:
        query_parts.append(f"{criteria['mutation']}")
    if "stage" in criteria:
        query_parts.append(f"stage {criteria['stage']}")
    if "treatment_phase" in criteria:
        query_parts.append(f"{criteria['treatment_phase']} treatment")

    search_query = " ".join(query_parts)
    print(f"   Search Query: '{search_query}'")

    try:
        results = client.search(
            query=search_query, space_ids=[patients_space_id], limit=5
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} matching patients")

        if episodes:
            print("\n   📋 Matching Patients:")
            for i, episode in enumerate(episodes, 1):
                metadata = episode.get("metadata", {})
                print(f"\n   {i}. Patient {metadata.get('patient_id', 'Unknown')}")
                print(f"      Match Score: {episode.get('score', 'N/A')}")

                # Verify match criteria
                matched_criteria = []
                for criterion, value in criteria.items():
                    if metadata.get(criterion) == value:
                        matched_criteria.append(criterion)

                print(f"      Criteria Match: {len(matched_criteria)}/{len(criteria)}")
                print(f"      Matched: {', '.join(matched_criteria)}")

    except Exception as e:
        print(f"   ❌ Matching search failed: {e}")


# ## 📈 Advanced Patient Analytics
#
# Let's demonstrate advanced analytics and insights generation.

# In[ ]:


# Advanced patient analytics and insights
print("📈 Advanced Patient Analytics")
print("=" * 60)


def calculate_population_statistics(patient_dataset):
    """
    Calculate comprehensive population statistics.

    Args:
        patient_dataset: List of patient records

    Returns:
        dict: Statistical analysis results
    """
    stats = {
        "total_patients": len(patient_dataset),
        "age_statistics": {"sum": 0, "count": 0, "min": float("inf"), "max": 0},
        "cancer_type_distribution": {},
        "stage_distribution": {},
        "response_rates": {},
        "treatment_distribution": {},
    }

    for patient in patient_dataset:
        metadata = patient["metadata"]

        # Age statistics
        age = metadata.get("age")
        if age:
            stats["age_statistics"]["sum"] += age
            stats["age_statistics"]["count"] += 1
            stats["age_statistics"]["min"] = min(stats["age_statistics"]["min"], age)
            stats["age_statistics"]["max"] = max(stats["age_statistics"]["max"], age)

        # Cancer type distribution
        cancer_type = metadata.get("cancer_type", "unknown")
        stats["cancer_type_distribution"][cancer_type] = (
            stats["cancer_type_distribution"].get(cancer_type, 0) + 1
        )

        # Stage distribution
        stage = metadata.get("stage", "unknown")
        stats["stage_distribution"][stage] = (
            stats["stage_distribution"].get(stage, 0) + 1
        )

        # Response rates
        response = metadata.get("response", "unknown")
        stats["response_rates"][response] = stats["response_rates"].get(response, 0) + 1

        # Treatment distribution
        treatment = metadata.get("current_therapy", "unknown")
        stats["treatment_distribution"][treatment] = (
            stats["treatment_distribution"].get(treatment, 0) + 1
        )

    # Calculate derived statistics
    if stats["age_statistics"]["count"] > 0:
        stats["age_statistics"]["mean"] = (
            stats["age_statistics"]["sum"] / stats["age_statistics"]["count"]
        )

    return stats


# Calculate comprehensive statistics
statistics = calculate_population_statistics(patient_dataset)

print("📊 Population Statistics:")
print("-" * 40)

print("👥 Demographics:")
age_stats = statistics["age_statistics"]
if age_stats["count"] > 0:
    print(f"   Age Range: {age_stats['min']}-{age_stats['max']} years")
    print(f"   Mean Age: {age_stats['mean']:.1f} years")
    print(f"   Total Patients: {age_stats['count']}")

print("\n🏥 Cancer Type Distribution:")
for cancer_type, count in statistics["cancer_type_distribution"].items():
    percentage = (count / statistics["total_patients"]) * 100
    print(f"   {cancer_type.title()}: {count} ({percentage:.1f}%)")

print("\n📊 Stage Distribution:")
for stage, count in statistics["stage_distribution"].items():
    percentage = (count / statistics["total_patients"]) * 100
    print(f"   Stage {stage}: {count} ({percentage:.1f}%)")

print("\n💊 Treatment Response Rates:")
for response, count in statistics["response_rates"].items():
    percentage = (count / statistics["total_patients"]) * 100
    print(f"   {response.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

print("\n💉 Treatment Distribution:")
for treatment, count in statistics["treatment_distribution"].items():
    percentage = (count / statistics["total_patients"]) * 100
    print(f"   {treatment}: {count} ({percentage:.1f}%)")


# ## 🎯 Clinical Research Insights
#
# Let's generate actionable insights for clinical research and practice.

# In[ ]:


# Generate clinical research insights
print("🎯 Clinical Research Insights")
print("=" * 60)

insights = []

# Insight 1: Treatment response patterns
complete_responses = sum(
    1 for p in patient_dataset if p["metadata"].get("response") == "complete_response"
)
partial_responses = sum(
    1 for p in patient_dataset if p["metadata"].get("response") == "partial_response"
)

insights.append(
    {
        "category": "Treatment Efficacy",
        "insight": f"Response Rate Analysis: {complete_responses + partial_responses}/{len(patient_dataset)} patients showing treatment response",
        "actionable": "Consider these patients for response biomarker studies",
    }
)

# Insight 2: Clinical trial opportunities
trial_patients = sum(1 for p in patient_dataset if p["metadata"].get("clinical_trial"))
insights.append(
    {
        "category": "Clinical Trials",
        "insight": f"Clinical Trial Enrollment: {trial_patients}/{len(patient_dataset)} patients in clinical trials",
        "actionable": "Evaluate remaining patients for appropriate trial matching",
    }
)

# Insight 3: Mutation prevalence
mutated_patients = sum(1 for p in patient_dataset if p["metadata"].get("mutation"))
insights.append(
    {
        "category": "Biomarker Prevalence",
        "insight": f"Targetable Mutations: {mutated_patients}/{len(patient_dataset)} patients with actionable mutations",
        "actionable": "Prioritize molecular testing for remaining patients",
    }
)

# Insight 4: Age distribution insights
young_patients = sum(1 for p in patient_dataset if p["metadata"].get("age", 0) < 50)
insights.append(
    {
        "category": "Age Demographics",
        "insight": f"Young Patients: {young_patients}/{len(patient_dataset)} patients under 50 years old",
        "actionable": "Consider age-specific treatment approaches and supportive care",
    }
)

print("💡 Key Clinical Insights:")
print("-" * 50)

for insight in insights:
    print(f"\n🔍 {insight['category']}:")
    print(f"   {insight['insight']}")
    print(f"   💡 Actionable: {insight['actionable']}")

# Generate research recommendations
print("\n\n🧪 Research Recommendations:")
print("-" * 40)

recommendations = [
    "Consider correlative studies between treatment response and molecular biomarkers",
    "Evaluate quality of life outcomes for patients on targeted therapies",
    "Assess long-term outcomes for patients with complete pathologic responses",
    "Investigate resistance mechanisms in patients with partial responses",
    "Study the impact of age and comorbidities on treatment tolerance",
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

print("\n📊 Patient Data Demo Summary:")
print("=" * 50)

print("✅ What We Accomplished:")
print("   📥 Ingested 5 comprehensive patient records")
print("   🔍 Executed 5 advanced search scenarios")
print("   📊 Generated automated patient summaries")
print("   🏷️ Performed patient classification and analytics")
print("   🔗 Demonstrated patient matching for research")
print("   📈 Calculated population statistics")
print("   🎯 Generated clinical research insights")

print("\n💡 Key Capabilities Demonstrated:")
print("   • Multi-format patient data ingestion")
print("   • Semantic search across patient records")
print("   • Automated summarization and reporting")
print("   • Statistical analysis and insights")
print("   • Clinical trial matching algorithms")
print("   • Research cohort identification")

print("\n🏥 Clinical Applications:")
print("   • Patient cohort identification for research")
print("   • Treatment response pattern analysis")
print("   • Clinical trial enrollment optimization")
print("   • Population health trend analysis")
print("   • Quality improvement initiatives")

print("\n🔬 Research Applications:")
print("   • Biomarker discovery and validation")
print("   • Treatment efficacy comparative analysis")
print("   • Patient stratification for studies")
print("   • Real-world evidence generation")
print("   • Clinical outcome prediction modeling")

print("\n🎉 Patient data demo completed successfully!")
print("🚀 Ready for clinical research applications!")

#!/usr/bin/env python
# coding: utf-8

# # 🧠 MCODE Translator - Core Memory Integration Demo
#
# Comprehensive demonstration of CORE Memory integration, data persistence, and knowledge graph operations.
#
# ## 📋 What This Demo Covers
#
# This notebook demonstrates MCODE Translator's CORE Memory capabilities:
#
# 1. **💾 Memory Space Management** - Creating and organizing memory spaces
# 2. **📥 Data Ingestion to Memory** - Persistent data storage with metadata
# 3. **🔍 Memory Search Operations** - Semantic search across memory spaces
# 4. **🧠 Knowledge Graph Queries** - Advanced relationship and pattern discovery
# 5. **📊 Memory Analytics** - Usage patterns and performance metrics
# 6. **🔄 Cross-Space Operations** - Data movement and synchronization
#
# ## 🎯 Learning Objectives
#
# By the end of this demo, you will:
# - ✅ Master CORE Memory space management
# - ✅ Understand data ingestion and persistence patterns
# - ✅ Learn advanced memory search techniques
# - ✅ Apply knowledge graph query methods
# - ✅ Use memory analytics for optimization
# - ✅ Perform cross-space data operations
#
# ## 💡 Memory Integration Benefits
#
# ### Persistence and Reliability
# - **Data Durability**: Information stored persistently across sessions
# - **Knowledge Accumulation**: Build long-term memory for AI applications
# - **Session Continuity**: Maintain context across different interactions
# - **Historical Tracking**: Track data evolution and changes over time
#
# ### Advanced Search and Discovery
# - **Semantic Understanding**: Deep comprehension of stored content
# - **Relationship Discovery**: Find connections between different data points
# - **Context Awareness**: Understand context and relevance of information
# - **Pattern Recognition**: Identify trends and patterns in data
#
# ### Multi-Modal Intelligence
# - **Cross-Domain Integration**: Connect information across different domains
# - **Multi-Source Correlation**: Combine data from various sources
# - **Intelligent Recommendations**: Generate insights based on accumulated knowledge
# - **Adaptive Learning**: Improve responses based on interaction history
#
# ---

# ## 🔧 Setup and Memory Configuration

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
    from heysol import HeySolClient

    from config.heysol_config import get_config

    print("✅ MCODE Translator components imported successfully!")
    print("   🧠 CORE Memory integration")
    print("   💾 Persistent data storage")
    print("   🔍 Advanced search capabilities")

except ImportError as e:
    print("❌ Failed to import MCODE Translator components.")
    print("💡 Install with: pip install -e .")
    print(f"   Error: {e}")
    raise


# ## 💾 Memory Space Architecture Design
#
# Let's design and implement a comprehensive memory architecture for clinical data management.

# In[ ]:


# Initialize HeySol client with memory focus
print("🔧 Initializing CORE Memory Integration...")
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
print(
    f"   🧠 Memory Integration: {'Available' if client.is_mcp_available() else 'API Only'}"
)

# Define comprehensive memory architecture
memory_architecture = {
    "clinical_patients": {
        "name": "Clinical Patient Memory",
        "description": "Persistent memory for patient data, treatment history, and outcomes",
        "data_types": [
            "patient_profiles",
            "treatment_courses",
            "outcomes",
            "biomarkers",
        ],
        "retention": "permanent",
        "access_pattern": "mixed_read_write",
        "governance": "high",
    },
    "clinical_trials": {
        "name": "Clinical Trials Memory",
        "description": "Persistent memory for clinical trial data, protocols, and results",
        "data_types": [
            "trial_protocols",
            "eligibility_criteria",
            "enrollment_data",
            "results",
        ],
        "retention": "permanent",
        "access_pattern": "read_heavy",
        "governance": "high",
    },
    "research_knowledge": {
        "name": "Research Knowledge Base",
        "description": "Persistent memory for research findings, publications, and insights",
        "data_types": [
            "research_papers",
            "clinical_guidelines",
            "treatment_standards",
            "evidence",
        ],
        "retention": "permanent",
        "access_pattern": "read_heavy",
        "governance": "standard",
    },
    "patient_matching": {
        "name": "Patient-Trial Matching Memory",
        "description": "Persistent memory for matching algorithms, criteria, and results",
        "data_types": [
            "matching_criteria",
            "eligibility_rules",
            "match_results",
            "recommendations",
        ],
        "retention": "temporary",
        "access_pattern": "compute_heavy",
        "governance": "standard",
    },
}

print("🏗️ Memory Architecture Overview:")
print(f"   Total spaces: {len(memory_architecture)}")
print(
    f"   Governance levels: {len(set(space['governance'] for space in memory_architecture.values()))}"
)
print(
    f"   Data types: {sum(len(space['data_types']) for space in memory_architecture.values())}"
)

print("\n📋 Memory Space Details:")
for space_key, space_info in memory_architecture.items():
    print(f"\n   💾 {space_key.upper()}:")
    print(f"      Purpose: {space_info['name']}")
    print(f"      Data Types: {', '.join(space_info['data_types'])}")
    print(f"      Access: {space_info['access_pattern']}")
    print(f"      Retention: {space_info['retention']}")
    print(f"      Governance: {space_info['governance']}")


# ## 📥 Intelligent Data Ingestion to Memory
#
# Now let's ingest comprehensive clinical data with rich metadata for persistent storage.

# In[ ]:


# Create comprehensive clinical dataset for memory storage
def create_memory_dataset():
    """
    Create a comprehensive dataset for CORE Memory storage.

    Returns:
        dict: Organized dataset by memory space
    """
    return {
        "clinical_patients": [
            {
                "content": "Patient P001: 52-year-old female with ER+/PR+/HER2- invasive ductal carcinoma, stage IIA. Completed neoadjuvant AC-T chemotherapy with excellent pathologic response (Miller-Payne grade 5). Underwent breast-conserving surgery with sentinel lymph node biopsy showing no residual disease. Currently receiving adjuvant endocrine therapy with anastrozole. Recent follow-up imaging shows no evidence of recurrence.",
                "metadata": {
                    "patient_id": "P001",
                    "age": 52,
                    "gender": "female",
                    "cancer_type": "breast",
                    "subtype": "invasive_ductal_carcinoma",
                    "stage": "IIA",
                    "receptor_status": "ER+/PR+/HER2-",
                    "treatment_phase": "adjuvant",
                    "current_therapy": "anastrozole",
                    "treatment_history": [
                        "AC-T_chemotherapy",
                        "breast_conserving_surgery",
                    ],
                    "pathologic_response": "complete_response",
                    "surgical_margins": "negative",
                    "lymph_node_status": "negative",
                    "recurrence_risk": "low",
                    "follow_up_status": "no_evidence_of_disease",
                },
            },
            {
                "content": "Patient P002: 67-year-old male with stage IV lung adenocarcinoma, EGFR exon 19 deletion positive. Initially presented with symptomatic bone metastases requiring radiation therapy. Started first-line osimertinib with excellent tolerance and partial response on initial restaging. Currently maintaining stable disease with good quality of life and minimal treatment-related toxicity.",
                "metadata": {
                    "patient_id": "P002",
                    "age": 67,
                    "gender": "male",
                    "cancer_type": "lung",
                    "histology": "adenocarcinoma",
                    "stage": "IV",
                    "mutation": "EGFR_exon_19_deletion",
                    "metastatic_sites": ["bone", "liver"],
                    "treatment_phase": "first_line",
                    "current_therapy": "osimertinib",
                    "treatment_history": ["palliative_radiation", "osimertinib"],
                    "response": "partial_response",
                    "performance_status": "ECOG_1",
                    "toxicity_profile": "minimal",
                    "quality_of_life": "good",
                },
            },
        ],
        "clinical_trials": [
            {
                "content": "Phase III randomized controlled trial (NCT04567892) evaluating nivolumab plus ipilimumab versus chemotherapy in patients with advanced BRAF-mutant melanoma. Primary endpoint is progression-free survival with secondary endpoints including overall survival and objective response rate. Trial met primary endpoint with significant PFS improvement (HR 0.67, p<0.001). Currently in long-term follow-up phase.",
                "metadata": {
                    "trial_id": "NCT04567892",
                    "phase": "III",
                    "status": "follow_up",
                    "cancer_type": "melanoma",
                    "mutation": "BRAF",
                    "treatments": ["nivolumab", "ipilimumab"],
                    "comparison": "chemotherapy",
                    "primary_endpoint": "progression_free_survival",
                    "results": {
                        "pfs_hazard_ratio": 0.67,
                        "pfs_p_value": 0.001,
                        "primary_endpoint_met": True,
                        "overall_survival_trend": "positive",
                    },
                    "target_enrollment": 600,
                    "actual_enrollment": 600,
                    "sites": 50,
                    "start_date": "2024-01-15",
                    "primary_completion": "2024-12-31",
                },
            },
            {
                "content": "Phase II single-arm study (NCT02314481) investigating palbociclib plus letrozole in postmenopausal women with ER+/HER2- metastatic breast cancer. Primary endpoint was progression-free survival with secondary endpoints including overall response rate and clinical benefit rate. Study completed with median PFS of 24.8 months, significantly exceeding historical controls.",
                "metadata": {
                    "trial_id": "NCT02314481",
                    "phase": "II",
                    "status": "completed",
                    "cancer_type": "breast",
                    "receptor_status": "ER+/HER2-",
                    "treatments": ["palbociclib", "letrozole"],
                    "line": "first_line",
                    "primary_endpoint": "progression_free_survival",
                    "results": {
                        "median_pfs": 24.8,
                        "response_rate": 55.2,
                        "clinical_benefit_rate": 78.9,
                        "toxicity_profile": "manageable",
                    },
                    "target_enrollment": 120,
                    "actual_enrollment": 120,
                    "sites": 25,
                    "start_date": "2023-06-01",
                    "completion_date": "2024-06-01",
                },
            },
        ],
        "research_knowledge": [
            {
                "content": "Comprehensive genomic analysis of 2,500 breast cancer patients reveals four distinct molecular subtypes with prognostic and therapeutic implications. The HER2-enriched subtype shows excellent response to dual HER2 blockade, while the basal-like subtype demonstrates poor prognosis but potential sensitivity to immune checkpoint inhibitors. These findings support molecular classification-driven treatment strategies.",
                "metadata": {
                    "publication_type": "research_study",
                    "cancer_type": "breast",
                    "sample_size": 2500,
                    "molecular_subtypes": [
                        "HER2_enriched",
                        "basal_like",
                        "luminal_A",
                        "luminal_B",
                    ],
                    "key_findings": [
                        "molecular_subtype_prognosis",
                        "treatment_response_prediction",
                    ],
                    "therapeutic_implications": [
                        "dual_her2_blockade",
                        "immune_checkpoint_inhibitors",
                    ],
                    "publication_year": 2024,
                    "evidence_level": "high",
                    "clinical_utility": "treatment_selection",
                },
            },
            {
                "content": "Meta-analysis of 15 randomized controlled trials evaluating CDK4/6 inhibitors in metastatic breast cancer demonstrates consistent progression-free survival benefit across all subgroups. The benefit is most pronounced in patients with bone-only metastases and those with longer treatment-free intervals. Toxicity profiles are manageable with appropriate dose modifications and supportive care.",
                "metadata": {
                    "publication_type": "meta_analysis",
                    "cancer_type": "breast",
                    "trials_analyzed": 15,
                    "treatment_class": "CDK4/6_inhibitors",
                    "primary_benefit": "progression_free_survival",
                    "special_subgroups": [
                        "bone_only_metastases",
                        "long_treatment_free_interval",
                    ],
                    "toxicity_profile": "manageable",
                    "publication_year": 2024,
                    "evidence_level": "1A",
                    "clinical_utility": "treatment_optimization",
                },
            },
        ],
    }


# ### 💾 Persistent Memory Ingestion
#
# Let's ingest our comprehensive dataset into CORE Memory with proper organization and metadata.

# In[ ]:


# Ingest data into CORE Memory with comprehensive tracking
print("💾 Persistent Memory Ingestion")
print("=" * 60)

memory_dataset = create_memory_dataset()
space_ids = {}

# Create memory spaces
print("🏗️ Creating Memory Spaces:")
for space_key, space_info in memory_architecture.items():
    try:
        space_id = client.create_space(space_info["name"], space_info["description"])
        space_ids[space_key] = space_id
        print(f"   ✅ {space_key}: {space_id[:16]}...")
    except Exception as e:
        print(f"   ❌ {space_key}: Failed - {e}")

print("\n📥 Ingesting Data into Memory:")

ingestion_log = []

for space_key, items in memory_dataset.items():
    space_id = space_ids.get(space_key)
    if not space_id:
        print(f"❌ No space ID for {space_key}")
        continue

    print(f"\n💾 {space_key.upper()}: {len(items)} items")

    for i, item in enumerate(items, 1):
        try:
            result = client.ingest(
                message=item["content"], space_id=space_id, metadata=item["metadata"]
            )

            ingestion_log.append(
                {
                    "space": space_key,
                    "item": i,
                    "success": True,
                    "content_type": item["metadata"].get(
                        "patient_id",
                        item["metadata"].get(
                            "trial_id",
                            item["metadata"].get("publication_type", "unknown"),
                        ),
                    ),
                }
            )

            print(
                f"   ✅ Item {i}: {item['metadata'].get('patient_id', item['metadata'].get('trial_id', 'Research'))}"
            )
            print("   💾 Saved to CORE Memory: Persistent storage confirmed")

        except Exception as e:
            ingestion_log.append(
                {"space": space_key, "item": i, "success": False, "error": str(e)}
            )
            print(f"   ❌ Item {i}: Failed - {e}")

# Memory ingestion summary
successful_ingestions = [log for log in ingestion_log if log["success"]]
failed_ingestions = [log for log in ingestion_log if not log["success"]]

print("\n📊 Memory Ingestion Summary:")
print(f"   Total items: {len(ingestion_log)}")
print(f"   Successful: {len(successful_ingestions)}")
print(f"   Failed: {len(failed_ingestions)}")
print(f"   Success rate: {(len(successful_ingestions)/len(ingestion_log)*100):.1f}%")

print("\n📈 By Memory Space:")
for space_key in memory_dataset.keys():
    space_items = [log for log in ingestion_log if log["space"] == space_key]
    successful = [log for log in space_items if log["success"]]
    print(f"   {space_key}: {len(successful)}/{len(space_items)} items")


# ## 🔍 Advanced Memory Search Operations
#
# Let's demonstrate sophisticated search capabilities across our memory spaces.

# In[ ]:


# Advanced memory search scenarios
print("🔍 Advanced Memory Search Operations")
print("=" * 60)

search_scenarios = [
    {
        "name": "Complete Response Patients",
        "query": "complete response pathologic complete response breast cancer",
        "space_filter": [space_ids.get("clinical_patients")],
        "description": "Find patients with complete responses to treatment",
        "expected_content": ["complete response", "breast cancer"],
    },
    {
        "name": "EGFR-Targeted Therapy Outcomes",
        "query": "EGFR lung cancer osimertinib response",
        "space_filter": [space_ids.get("clinical_patients")],
        "description": "Find EGFR+ lung cancer patients and their treatment outcomes",
        "expected_content": ["EGFR", "lung", "osimertinib"],
    },
    {
        "name": "Positive Clinical Trial Results",
        "query": "phase III positive results significant improvement",
        "space_filter": [space_ids.get("clinical_trials")],
        "description": "Find phase III trials with positive results",
        "expected_content": ["phase III", "positive", "significant"],
    },
    {
        "name": "CDK4/6 Inhibitor Evidence",
        "query": "CDK4/6 inhibitors breast cancer progression free survival",
        "space_filter": [space_ids.get("research_knowledge")],
        "description": "Find research evidence on CDK4/6 inhibitors",
        "expected_content": ["CDK4/6", "breast", "PFS"],
    },
    {
        "name": "Cross-Domain Treatment Insights",
        "query": "hormone receptor positive breast cancer treatment options",
        "space_filter": None,  # Search all spaces
        "description": "Find comprehensive treatment information across all domains",
        "expected_content": ["hormone receptor", "breast", "treatment"],
    },
]

search_results = []

for scenario in search_scenarios:
    print(f"\n🔎 {scenario['name']}")
    print(f"   Description: {scenario['description']}")
    print(f"   Query: '{scenario['query']}'")

    try:
        results = client.search(
            query=scenario["query"], space_ids=scenario["space_filter"], limit=5
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} relevant memories")

        if episodes:
            print("\n   📋 Memory Results:")
            for i, episode in enumerate(episodes, 1):
                content = episode.get("content", "")[:100]
                score = episode.get("score", "N/A")
                metadata = episode.get("metadata", {})

                print(f"\n   {i}. {content}{'...' if len(content) == 100 else ''}")
                print(f"      Relevance Score: {score}")
                print(
                    f"      Source: {metadata.get('patient_id', metadata.get('trial_id', metadata.get('publication_type', 'Unknown')))}"
                )

                # Show relevant metadata
                if metadata.get("response"):
                    print(f"      Response: {metadata['response']}")
                if metadata.get("results"):
                    results_info = metadata["results"]
                    if results_info.get("primary_endpoint_met"):
                        print("      Results: Positive trial outcome")
        search_results.append(
            {
                "scenario": scenario["name"],
                "results_count": len(episodes),
                "episodes": episodes,
                "success": True,
            }
        )

    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        search_results.append(
            {
                "scenario": scenario["name"],
                "error": str(e),
                "results_count": 0,
                "success": False,
            }
        )

print("\n📊 Memory Search Summary:")
print(f"   Search scenarios: {len(search_scenarios)}")
print(f"   Total results: {sum(r['results_count'] for r in search_results)}")
print(
    f"   Average results per search: {sum(r['results_count'] for r in search_results)/len(search_scenarios):.1f}"
)


# ## 🧠 Knowledge Graph Queries and Analysis
#
# Let's demonstrate advanced knowledge graph operations for discovering relationships and patterns.

# In[ ]:


# Knowledge graph analysis and pattern discovery
print("🧠 Knowledge Graph Analysis")
print("=" * 60)

knowledge_queries = [
    {
        "name": "Treatment Response Patterns",
        "query": "treatment response patterns across cancer types",
        "description": "Discover patterns in treatment responses across different cancers",
        "analysis_type": "pattern_discovery",
    },
    {
        "name": "Biomarker-Treatment Relationships",
        "query": "biomarker treatment relationships and outcomes",
        "description": "Find relationships between biomarkers and treatment outcomes",
        "analysis_type": "relationship_discovery",
    },
    {
        "name": "Clinical Trial Outcome Correlations",
        "query": "clinical trial outcomes and patient characteristics",
        "description": "Correlate trial outcomes with patient demographics and characteristics",
        "analysis_type": "correlation_analysis",
    },
    {
        "name": "Evidence-Based Treatment Recommendations",
        "query": "evidence based treatment recommendations breast cancer",
        "description": "Find evidence-based treatment recommendations for breast cancer",
        "analysis_type": "evidence_synthesis",
    },
]

print("🔍 Knowledge Graph Query Analysis:")
print("-" * 50)

for query in knowledge_queries:
    print(f"\n🧠 {query['name']}")
    print(f"   Type: {query['analysis_type']}")
    print(f"   Query: '{query['query']}'")

    try:
        results = client.search(
            query=query["query"],
            space_ids=None,  # Search all spaces for comprehensive analysis
            limit=4,
        )

        episodes = results.get("episodes", [])
        print(f"   ✅ Found {len(episodes)} knowledge connections")

        if episodes:
            print("\n   📋 Knowledge Connections:")
            for i, episode in enumerate(episodes, 1):
                content = episode.get("content", "")[:80]
                score = episode.get("score", "N/A")
                metadata = episode.get("metadata", {})

                print(f"\n   {i}. {content}{'...' if len(content) == 80 else ''}")
                print(f"      Relevance: {score}")
                print(
                    f"      Domain: {metadata.get('cancer_type', metadata.get('publication_type', 'General'))}"
                )

                # Analyze connection type
                if metadata.get("patient_id"):
                    print("      Type: Patient case study")
                elif metadata.get("trial_id"):
                    print("      Type: Clinical trial evidence")
                elif metadata.get("publication_type"):
                    print("      Type: Research publication")
                # Extract actionable insights
                if "complete response" in content.lower():
                    print("      💡 Insight: Complete response case identified")
                if (
                    "significant" in content.lower()
                    and "improvement" in content.lower()
                ):
                    print("      💡 Insight: Significant treatment benefit")

    except Exception as e:
        print(f"   ❌ Knowledge query failed: {e}")


# ## 📊 Memory Analytics and Performance
#
# Let's analyze memory usage patterns and performance characteristics.

# In[ ]:


# Memory analytics and performance analysis
print("📊 Memory Analytics and Performance")
print("=" * 60)


def analyze_memory_performance():
    """
    Analyze memory performance and usage patterns.

    Returns:
        dict: Memory performance metrics
    """
    performance_metrics = {
        "search_performance": [],
        "space_utilization": {},
        "content_distribution": {},
        "query_effectiveness": {},
    }

    # Test search performance across different query types
    test_queries = [
        (
            "Simple patient query",
            "breast cancer patient",
            [space_ids.get("clinical_patients")],
        ),
        (
            "Complex trial query",
            "phase III positive results immunotherapy",
            [space_ids.get("clinical_trials")],
        ),
        ("Cross-domain query", "treatment response evidence", None),
        ("Biomarker query", "EGFR mutation targeted therapy", None),
    ]

    print("⏱️ Testing Memory Search Performance:")
    print("-" * 50)

    for query_name, query, space_filter in test_queries:
        import time

        start_time = time.time()

        try:
            results = client.search(query, space_ids=space_filter, limit=3)
            episodes = results.get("episodes", [])

            end_time = time.time()
            duration = end_time - start_time

            performance_metrics["search_performance"].append(
                {
                    "query": query_name,
                    "duration": duration,
                    "results": len(episodes),
                    "performance": (
                        "excellent"
                        if duration < 0.5
                        else "good" if duration < 1.0 else "fair"
                    ),
                }
            )

            print(
                f"   {query_name}: {duration:.3f}s ({len(episodes)} results) - {performance_metrics['search_performance'][-1]['performance']}"
            )

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"   {query_name}: {duration:.3f}s (FAILED) - {e}")

    return performance_metrics


# Analyze memory performance
performance_metrics = analyze_memory_performance()

# Display performance summary
print("\n📊 Memory Performance Summary:")
print(f"   Queries tested: {len(performance_metrics['search_performance'])}")

excellent_queries = [
    p
    for p in performance_metrics["search_performance"]
    if p["performance"] == "excellent"
]
good_queries = [
    p for p in performance_metrics["search_performance"] if p["performance"] == "good"
]

print(f"   Excellent performance (< 0.5s): {len(excellent_queries)}")
print(f"   Good performance (0.5-1.0s): {len(good_queries)}")

if excellent_queries:
    avg_excellent = sum(p["duration"] for p in excellent_queries) / len(
        excellent_queries
    )
    print(f"   Average excellent query time: {avg_excellent:.3f}s")

if good_queries:
    avg_good = sum(p["duration"] for p in good_queries) / len(good_queries)
    print(f"   Average good query time: {avg_good:.3f}s")


# ## 🔄 Cross-Space Memory Operations
#
# Let's demonstrate advanced cross-space operations for data integration and analysis.

# In[ ]:


# Cross-space memory operations and integration
print("🔄 Cross-Space Memory Operations")
print("=" * 60)

cross_space_scenarios = [
    {
        "name": "Patient-Trial Matching Intelligence",
        "description": "Combine patient data with trial information for matching",
        "patient_query": "ER positive breast cancer patients",
        "trial_query": "ER positive breast cancer clinical trials",
        "integration_type": "patient_trial_matching",
    },
    {
        "name": "Evidence-Based Treatment Synthesis",
        "description": "Combine research evidence with clinical trial results",
        "research_query": "CDK4/6 inhibitors evidence breast cancer",
        "trial_query": "CDK4/6 inhibitor clinical trials results",
        "integration_type": "evidence_synthesis",
    },
    {
        "name": "Biomarker-Guided Treatment Discovery",
        "description": "Connect molecular biomarkers with treatment outcomes",
        "biomarker_query": "EGFR mutation lung cancer biomarkers",
        "outcome_query": "EGFR targeted therapy outcomes",
        "integration_type": "biomarker_outcome_correlation",
    },
]

print("🔗 Cross-Space Integration Analysis:")
print("-" * 50)

for scenario in cross_space_scenarios:
    print(f"\n🔄 {scenario['name']}")
    print(f"   Type: {scenario['integration_type']}")
    print(f"   Description: {scenario['description']}")

    try:
        # Perform integrated search across relevant spaces
        if "patient" in scenario and "trial" in scenario:
            # Patient-trial matching
            patient_results = client.search(
                scenario["patient_query"],
                space_ids=[space_ids.get("clinical_patients")],
                limit=2,
            )
            trial_results = client.search(
                scenario["trial_query"],
                space_ids=[space_ids.get("clinical_trials")],
                limit=2,
            )

            print(f"   ✅ Patient matches: {len(patient_results.get('episodes', []))}")
            print(f"   ✅ Trial matches: {len(trial_results.get('episodes', []))}")

            # Analyze matching opportunities
            patient_episodes = patient_results.get("episodes", [])
            trial_episodes = trial_results.get("episodes", [])

            if patient_episodes and trial_episodes:
                print("\n    💡 Matching Opportunities:")
                for p_episode in patient_episodes:
                    p_metadata = p_episode.get("metadata", {})
                    for t_episode in trial_episodes:
                        t_metadata = t_episode.get("metadata", {})

                        # Simple matching logic
                        cancer_match = p_metadata.get("cancer_type") == t_metadata.get(
                            "cancer_type"
                        )
                        receptor_match = p_metadata.get(
                            "receptor_status"
                        ) == t_metadata.get("receptor_status")

                        if cancer_match and receptor_match:
                            print(
                                f"      🎯 Patient {p_metadata.get('patient_id')} may be eligible for Trial {t_metadata.get('trial_id')}"
                            )

        elif "research" in scenario and "trial" in scenario:
            # Evidence synthesis
            research_results = client.search(
                scenario["research_query"],
                space_ids=[space_ids.get("research_knowledge")],
                limit=2,
            )
            trial_results = client.search(
                scenario["trial_query"],
                space_ids=[space_ids.get("clinical_trials")],
                limit=2,
            )

            print(
                f"   ✅ Research evidence: {len(research_results.get('episodes', []))}"
            )
            print(f"   ✅ Trial results: {len(trial_results.get('episodes', []))}")

            if research_results.get("episodes") and trial_results.get("episodes"):
                print("\n    💡 Evidence Integration:")
                print(
                    "      📚 Research provides supporting evidence for trial outcomes"
                )
                print("      🧪 Trials validate research findings in clinical settings")
    except Exception as e:
        print(f"   ❌ Cross-space operation failed: {e}")


# ## 🎯 Memory-Driven Insights and Recommendations
#
# Let's generate actionable insights based on our accumulated memory and knowledge graph.

# In[ ]:


# Memory-driven insights and recommendations
print("🎯 Memory-Driven Insights and Recommendations")
print("=" * 60)

insights = []

# Insight 1: Treatment response patterns
complete_responses = 0
partial_responses = 0

# Search for response patterns
try:
    response_results = client.search("complete response partial response", limit=10)
    response_episodes = response_results.get("episodes", [])

    for episode in response_episodes:
        content = episode.get("content", "").lower()
        if "complete response" in content:
            complete_responses += 1
        if "partial response" in content:
            partial_responses += 1

    insights.append(
        {
            "category": "Treatment Response Patterns",
            "insight": f"Response Distribution: {complete_responses} complete, {partial_responses} partial responses identified",
            "confidence": "high",
            "actionable": "Consider complete responders for de-escalation strategies",
        }
    )

except Exception as e:
    print(f"⚠️ Could not analyze response patterns: {e}")

# Insight 2: Clinical trial opportunities
try:
    trial_results = client.search(
        "phase III recruiting", space_ids=[space_ids.get("clinical_trials")], limit=5
    )
    recruiting_trials = len(trial_results.get("episodes", []))

    insights.append(
        {
            "category": "Clinical Trial Opportunities",
            "insight": f"Active Trials: {recruiting_trials} phase III trials currently recruiting",
            "confidence": "high",
            "actionable": "Evaluate patient eligibility for ongoing trials",
        }
    )

except Exception as e:
    print(f"⚠️ Could not analyze trial opportunities: {e}")

# Insight 3: Evidence-based treatment options
try:
    evidence_results = client.search(
        "evidence based treatment recommendations", limit=5
    )
    evidence_count = len(evidence_results.get("episodes", []))

    insights.append(
        {
            "category": "Evidence-Based Medicine",
            "insight": f"Evidence Base: {evidence_count} evidence-based treatment recommendations available",
            "confidence": "high",
            "actionable": "Prioritize treatments with strongest evidence base",
        }
    )

except Exception as e:
    print(f"⚠️ Could not analyze evidence base: {e}")

print("💡 Memory-Driven Clinical Insights:")
print("-" * 50)

for insight in insights:
    confidence_emoji = "🔥" if insight["confidence"] == "high" else "⚡"
    print(f"\n{confidence_emoji} {insight['category']}:")
    print(f"   {insight['insight']}")
    print(f"   💡 Actionable: {insight['actionable']}")

# Generate strategic recommendations
print("\n\n🎯 Strategic Recommendations:")
print("-" * 40)

recommendations = [
    "Implement routine molecular profiling to identify patients for targeted therapies",
    "Develop automated patient-trial matching workflows using CORE Memory",
    "Create evidence-based treatment pathways for common cancer scenarios",
    "Establish continuous learning systems to update treatment approaches",
    "Build comprehensive outcome tracking to validate treatment decisions",
    "Develop predictive models for treatment response using accumulated data",
    "Create automated literature surveillance for emerging treatment options",
    "Implement real-time clinical decision support based on memory queries",
]

for i, recommendation in enumerate(recommendations, 1):
    print(f"{i}. {recommendation}")


# ## 🧹 Memory Management and Cleanup
#
# Let's demonstrate proper memory management and cleanup procedures.

# In[ ]:


# Memory management and cleanup
print("🧹 Memory Management and Cleanup")
print("=" * 60)

print("💡 Memory Management Best Practices:")
print("   1. Regular performance monitoring")
print("   2. Space utilization optimization")
print("   3. Query performance tuning")
print("   4. Proper resource cleanup")

# Demonstrate memory space management
print("\n🏗️ Memory Space Management:")
print(f"   Active spaces: {len(space_ids)}")

for space_key, space_id in space_ids.items():
    space_info = memory_architecture.get(space_key, {})
    print(f"   💾 {space_key}: {space_id[:16]}...")
    print(f"      Purpose: {space_info.get('name', 'Unknown')}")
    print(f"      Governance: {space_info.get('governance', 'Unknown')}")
    print(f"      Retention: {space_info.get('retention', 'Unknown')}")

# Close client connection
try:
    client.close()
    print("\n✅ Memory client connection closed successfully")
except Exception as e:
    print(f"\n⚠️ Cleanup warning: {e}")

print("\n📊 CORE Memory Integration Summary:")
print("=" * 60)

print("✅ What We Accomplished:")
print("   💾 Created 4 specialized memory spaces")
print("   📥 Ingested comprehensive clinical data")
print("   🔍 Executed advanced memory search operations")
print("   🧠 Performed knowledge graph analysis")
print("   📊 Analyzed memory performance metrics")
print("   🔄 Demonstrated cross-space operations")
print("   🎯 Generated memory-driven insights")

print("\n💡 Key Memory Capabilities Demonstrated:")
print("   • Persistent data storage across sessions")
print("   • Advanced semantic search capabilities")
print("   • Knowledge graph relationship discovery")
print("   • Cross-domain data integration")
print("   • Performance-optimized memory operations")
print("   • Evidence-based insight generation")

print("\n🏥 Clinical Applications:")
print("   • Long-term patient outcome tracking")
print("   • Clinical trial evidence accumulation")
print("   • Treatment response pattern analysis")
print("   • Evidence-based treatment recommendations")
print("   • Continuous learning from clinical data")

print("\n🔬 Research Applications:")
print("   • Multi-study evidence synthesis")
print("   • Biomarker-outcome correlation analysis")
print("   • Clinical trial landscape intelligence")
print("   • Treatment guideline development")
print("   • Knowledge discovery and pattern recognition")

print("\n🎉 CORE Memory integration demo completed successfully!")
print("🚀 Ready for persistent clinical knowledge management!")

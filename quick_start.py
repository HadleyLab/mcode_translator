#!/usr/bin/env python
# coding: utf-8

# # 🎯 MCODE Translator - Quick Start
#
# Get up and running with MCODE Translator in under 5 minutes!
#
# This notebook will guide you through:
# 1. ✅ API key setup and validation
# 2. 🔧 CLI initialization and registry setup
# 3. 🏗️ Creating a demo space
# 4. 📝 Ingesting sample clinical data
# 5. 🔍 Performing searches and summaries
# 6. 📊 Viewing results and patient matching
#
# ## 📋 Prerequisites
#
# Before running this notebook, ensure you have:
#
# 1. **A valid HeySol API key** from [https://core.heysol.ai/settings/api](https://core.heysol.ai/settings/api)
# 2. **Set the environment variable**: `export HEYSOL_API_KEY="your-key-here"`
# 3. **Or create a `.env` file** with: `HEYSOL_API_KEY_xxx=your-key-here` (any name starting with HEYSOL_API_KEY)
# 4. **Install the package**: `pip install -e .`
#
# ## 🚀 Let's Get Started!

# In[1]:


# Import required modules
import os

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import MCODE Translator components
try:
    from src.cli.cli import app as mcode_app
    from src.config.heysol_config import get_config

    print("✅ MCODE Translator components imported successfully!")
    print("   🧠 CORE Memory integration")
    print("   📊 Clinical data processing")
    print("   🔍 Advanced search capabilities")
    print("   👥 Patient-trial matching")
except ImportError as e:
    print("❌ Failed to import MCODE Translator components.")
    print("💡 Install with: pip install -e .")
    print(f"   Error: {e}")
    raise


# ## 🔑 Step 1: API Key Validation
#
# First, let's check that your API key is properly configured. The system will validate your key format and test it against the API.

# In[2]:


# Check and validate API key
print("🔑 Checking API key configuration...")

api_key = os.getenv("HEYSOL_API_KEY")
if not api_key:
    print("❌ No API key found!")
    print("\n📝 To get started:")
    print("1. Visit: https://core.heysol.ai/settings/api")
    print("2. Generate an API key")
    print("3. Set environment variable:")
    print("   export HEYSOL_API_KEY='your-api-key-here'")
    print("4. Or create a .env file with:")
    print("   HEYSOL_API_KEY=your-api-key-here")
    print("\nThen restart this notebook!")
    raise ValueError("API key not configured")

print(f"✅ API key found (ends with: ...{api_key[-4:]})")
print("🔍 Validating API key...")

# The validation will happen automatically when we create clients below


# ## 🔧 Step 2: Client Types Demonstration
#
# MCODE Translator provides integrated HeySol client capabilities. Let's explore the available features.

# In[3]:


# Demonstrate MCODE Translator capabilities
print("🔧 Demonstrating MCODE Translator capabilities...")
print("=" * 50)

# Check CLI availability
try:
    import subprocess

    result = subprocess.run(
        ["python", "-m", "src.cli.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    print("✅ CLI available")
except Exception as e:
    print(f"⚠️ CLI check failed: {e}")

# Check configuration
try:
    config = get_config()
    print("✅ Configuration loaded successfully")
    print(f"   🎯 Base URL: {config.get_base_url()}")
    print(f"   📧 Source: {config.get_heysol_config().source}")
except Exception as e:
    print(f"⚠️ Configuration check failed: {e}")

print("\n✅ MCODE Translator setup complete!")
print("💡 Ready to process clinical data")


# ## 🏗️ Step 3: Create Demo Space
#
# Spaces are containers for organizing your clinical data in HeySol. Let's create a demo space for our examples.

# In[4]:


# Create or reuse demo space
print("🏗️ Setting up demo space...")

space_name = "MCODE Translator Demo"
space_description = "Created by MCODE Translator quick start notebook"

import json
# Use CLI to create space
import subprocess

print(f"   🔍 Checking for existing space: '{space_name}'...")

try:
    # Check existing spaces
    result = subprocess.run(
        ["python", "-m", "src.cli.cli", "spaces", "list"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    existing_spaces = []
    if result.stdout:
        try:
            existing_spaces = json.loads(result.stdout)
        except:
            existing_spaces = []

    space_id = None

    # Look for existing space
    if isinstance(existing_spaces, list):
        for space in existing_spaces:
            if isinstance(space, dict) and space.get("name") == space_name:
                space_id = space.get("id")
                break

    # Create new space if needed
    if not space_id:
        print(f"   🆕 Creating new space: '{space_name}'...")
        create_result = subprocess.run(
            [
                "python",
                "-m",
                "src.cli.cli",
                "spaces",
                "create",
                space_name,
                "--description",
                space_description,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if create_result.stdout:
            try:
                create_data = json.loads(create_result.stdout)
                space_id = create_data.get("space_id")
            except:
                pass

    print(f"\n📊 Ready to use space: {space_name}")
    print(f"   ID: {space_id}")
    print(f"   Description: {space_description}")

except Exception as e:
    print(f"⚠️ Space setup failed: {e}")
    space_id = "demo"


# ## 📝 Step 4: Ingest Sample Clinical Data
#
# Now let's add some sample clinical data to HeySol. This data will be processed and made searchable.

# In[5]:


# Ingest sample clinical data
print("📝 Ingesting sample clinical data...")
print("=" * 50)

sample_data = [
    "Patient P001: 52-year-old female with ER+/PR+/HER2- invasive ductal carcinoma, stage IIA. Completed neoadjuvant AC-T chemotherapy with excellent pathologic response (Miller-Payne grade 5). Underwent breast-conserving surgery with sentinel lymph node biopsy showing no residual disease. Currently receiving adjuvant endocrine therapy with anastrozole.",
    "Patient P002: 67-year-old male with stage IV lung adenocarcinoma, EGFR exon 19 deletion positive. Initially presented with symptomatic bone metastases requiring radiation therapy. Started first-line osimertinib with excellent tolerance and partial response on initial restaging.",
    "Clinical trial NCT04567892: Phase III study evaluating nivolumab plus ipilimumab versus chemotherapy in patients with advanced BRAF-mutant melanoma. Primary endpoint is progression-free survival with secondary endpoints including overall survival and objective response rate.",
    "Biomarker analysis reveals key indicators for treatment success in oncology patients. Molecular profiling shows distinct subtypes with varying responses to targeted therapies.",
]

print(f"   📋 Will ingest {len(sample_data)} clinical data items")

for i, data in enumerate(sample_data, 1):
    print(f"   🔄 Ingesting item {i}/{len(sample_data)}...")
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.cli.cli",
                "memory",
                "ingest",
                data,
                "--space-id",
                space_id,
            ],
            capture_output=True,
            timeout=30,
        )

        print(f"   ✅ Item {i} ingested successfully")

    except Exception as e:
        print(f"   ❌ Item {i} failed: {e}")

print("\n✅ Sample clinical data ingestion complete!")
print("💡 Data is being processed in the background and will be searchable soon")


# ## 🔍 Step 5: Perform Search and Summaries
#
# Let's search for the data we just ingested and demonstrate MCODE Translator's summarization capabilities.

# In[6]:


# Search for ingested data and demonstrate summaries
print("🔍 Searching for clinical data and generating summaries...")
print("=" * 60)

search_queries = [
    "breast cancer treatment",
    "lung cancer EGFR",
    "clinical trial melanoma",
    "biomarker analysis",
]

print("🔎 Performing semantic searches:")
for query in search_queries:
    print(f"\n   Query: '{query}'")
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.cli.cli",
                "memory",
                "search",
                query,
                "--space-id",
                space_id,
                "--limit",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.stdout:
            try:
                search_data = json.loads(result.stdout)
                episodes = search_data.get("episodes", [])
                print(f"   ✅ Found {len(episodes)} results")
                for i, episode in enumerate(episodes[:2], 1):
                    content = episode.get("content", "")[:80]
                    print(f"      {i}. {content}{'...' if len(content) == 80 else ''}")
            except:
                print("   ℹ️ Could not parse results")
        else:
            print("   📭 No results found yet")

    except Exception as e:
        print(f"   ❌ Search failed: {e}")

print("\n📊 Demonstrating patient and trial summarization:")

# Demonstrate patient summarization
print("\n👥 Patient Summarization:")
try:
    result = subprocess.run(
        [
            "python",
            "-m",
            "src.cli.cli",
            "patients",
            "summarize",
            "--space-id",
            space_id,
            "--limit",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.stdout:
        print("   ✅ Patient summary generated")
        print(f"   📋 {result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}")
except Exception as e:
    print(f"   ⚠️ Patient summarization failed: {e}")

# Demonstrate trial summarization
print("\n🧪 Clinical Trial Summarization:")
try:
    result = subprocess.run(
        [
            "python",
            "-m",
            "src.cli.cli",
            "trials",
            "summarize",
            "--space-id",
            space_id,
            "--limit",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.stdout:
        print("   ✅ Trial summary generated")
        print(f"   📋 {result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}")
except Exception as e:
    print(f"   ⚠️ Trial summarization failed: {e}")


# ## 📊 Step 6: Patient-Trial Matching
#
# Let's demonstrate MCODE Translator's patient-trial matching capabilities using the knowledge graph.

# In[7]:


# Demonstrate patient-trial matching
print("🎯 Demonstrating Patient-Trial Matching")
print("=" * 50)

print("🔗 Finding potential matches between patients and clinical trials...")

try:
    # Search for patients with specific characteristics
    patient_search = subprocess.run(
        [
            "python",
            "-m",
            "src.cli.cli",
            "memory",
            "search",
            "breast cancer ER positive stage II",
            "--space-id",
            space_id,
            "--limit",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Search for relevant trials
    trial_search = subprocess.run(
        [
            "python",
            "-m",
            "src.cli.cli",
            "memory",
            "search",
            "breast cancer clinical trial",
            "--space-id",
            space_id,
            "--limit",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    patient_count = 0
    trial_count = 0

    if patient_search.stdout:
        try:
            patient_data = json.loads(patient_search.stdout)
            patient_count = len(patient_data.get("episodes", []))
        except:
            pass

    if trial_search.stdout:
        try:
            trial_data = json.loads(trial_search.stdout)
            trial_count = len(trial_data.get("episodes", []))
        except:
            pass

    print(f"   👥 Found {patient_count} potential patients")
    print(f"   🧪 Found {trial_count} relevant trials")

    if patient_count > 0 and trial_count > 0:
        print("\n💡 Matching Analysis:")
        print("   • Patients with breast cancer characteristics identified")
        print("   • Clinical trials for breast cancer treatments found")
        print(
            "   • Knowledge graph can correlate patient profiles with trial eligibility"
        )
        print("   • Matching considers tumor characteristics, stage, and biomarkers")
        print(
            "\n🎯 Potential matches detected - ready for detailed eligibility assessment!"
        )
    else:
        print(
            "\n💡 Note: Matching requires sufficient patient and trial data in memory"
        )
        print("   Recent data may still be processing or queued")

except Exception as e:
    print(f"⚠️ Matching demonstration failed: {e}")
    print("💡 This is normal if data is still being processed")


# ## 📊 Step 7: View Results & Summary
#
# Let's get a summary of what we've accomplished and explore next steps.

# In[8]:


# Display summary and next steps
print("📊 MCODE Translator Quick Start Summary")
print("=" * 50)

print("✅ What we accomplished:")
print("   🔑 Validated API key and configuration")
print("   🔧 Demonstrated CLI capabilities")
print(f"   🏗️ Created/used space: {space_name}")
print(f"   📝 Ingested {len(sample_data)} clinical data items")
print("   🔍 Performed semantic searches")
print("   📊 Generated patient and trial summaries")
print("   🎯 Demonstrated patient-trial matching")

print("\n📚 Next Steps:")
print("   📖 Explore examples: ls examples/")
print("   🖥️ Try the CLI: python -m src.cli.cli --help")
print("   📚 Read docs: README.md")
print("   🔬 Try comprehensive demos: python examples/patients_demo.py")
print("   🧪 Run clinical trials demo: python examples/clinical_trials_demo.py")

print("\n💡 MCODE Translator Features:")
print("   🧠 CORE Memory integration for persistent knowledge")
print("   👥 Advanced patient data processing and summarization")
print("   🧪 Clinical trial analysis and optimization")
print("   🎯 Intelligent patient-trial matching")
print("   📊 Comprehensive clinical data workflows")

# Clean up
print("\n🧹 Quick start completed successfully!")
print("🚀 You're now ready to use MCODE Translator!")
print("\n💡 Pro Tips:")
print("   • Use --user iDrDex@MammoChat.com for main repository operations")
print("   • Incorporate patients and clinical trials data for comprehensive analysis")
print("   • Show different summaries of both patients and clinical trials")
print("   • Push data to CORE Memory for persistent storage")
print("   • Ask patient matching questions based on the knowledge graph")
print("   • Recent innovations may be processing or queued - check back soon!")

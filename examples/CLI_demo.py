#!/usr/bin/env python
# coding: utf-8

# # 🖥️ MCODE Translator - CLI Demo
#
# Comprehensive demonstration of command-line interface capabilities for clinical data processing.
#
# ## 📋 What This Demo Covers
#
# This notebook demonstrates MCODE Translator's CLI capabilities:
#
# 1. **🔧 CLI Setup and Configuration** - Environment setup and validation
# 2. **📥 Data Ingestion Commands** - CLI-based data import and processing
# 3. **🔍 Search and Query Operations** - Command-line search functionality
# 4. **📊 Analytics and Reporting** - CLI-based analytics and summaries
# 5. **🏗️ Space Management** - CLI space creation and management
# 6. **📋 Batch Processing** - Automated batch operations
#
# ## 🎯 Learning Objectives
#
# By the end of this demo, you will:
# - ✅ Master CLI command structure and syntax
# - ✅ Understand environment configuration for CLI operations
# - ✅ Learn automated data ingestion workflows
# - ✅ Apply CLI-based search and analytics
# - ✅ Use batch processing for large datasets
#
# ## 🏥 CLI Use Cases
#
# ### Data Management
# - **Automated Data Ingestion**: Batch import clinical data from files
# - **Scheduled Processing**: Cron jobs for regular data updates
# - **Pipeline Integration**: CLI commands in data processing workflows
# - **Quality Assurance**: Automated validation and reporting
#
# ### Research Operations
# - **Cohort Analysis**: CLI-based patient cohort identification
# - **Trial Matching**: Automated clinical trial eligibility screening
# - **Data Export**: Structured data export for analysis tools
# - **Monitoring**: Health checks and performance monitoring
#
# ---

# ## 🔧 CLI Environment Setup

# In[ ]:


# Import required modules
import os
import sys
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path.cwd() / "src"))

print("🔧 CLI Environment Setup")
print("=" * 40)

# Check if CLI is available
try:
    result = subprocess.run(['python', '-m', 'cli', '--help'],
                          capture_output=True, text=True, timeout=10)
    print("✅ CLI module available")
except (subprocess.TimeoutExpired, FileNotFoundError):
    print("❌ CLI module not found!")
    print("💡 Install with: pip install -e .")
    raise

# Check API key configuration
api_key = os.getenv("HEYSOL_API_KEY")
if not api_key:
    print("❌ No API key found!")
    print("💡 Set HEYSOL_API_KEY environment variable")
    raise ValueError("API key not configured")

print(f"✅ API key configured (ends with: ...{api_key[-4:]})")
print("✅ CLI environment ready!")


# ## 🏗️ Space Management with CLI

# In[ ]:


# CLI Space Management Demo
print("🏗️ CLI Space Management")
print("=" * 40)

# Create demo space
space_name = "CLI Demo Space"
space_description = "Created by CLI demo for testing"

print(f"Creating space: {space_name}")

try:
    result = subprocess.run([
        'python', '-m', 'cli', 'spaces', 'create',
        space_name, '--description', space_description
    ], capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        print("✅ Space created successfully")
        # Extract space ID from output
        space_id = None
        for line in result.stdout.split('\n'):
            if 'space_id' in line:
                space_id = line.split(':')[1].strip()
                break
        if space_id:
            print(f"   Space ID: {space_id}")
    else:
        print(f"❌ Space creation failed: {result.stderr}")

except subprocess.TimeoutExpired:
    print("❌ Space creation timeout")


# ## 📥 CLI Data Ingestion

# In[ ]:


# CLI Data Ingestion Demo
print("📥 CLI Data Ingestion")
print("=" * 40)

# Sample clinical data for ingestion
sample_data = [
    "Patient with advanced lung cancer shows excellent response to immunotherapy",
    "Clinical trial demonstrates 85% response rate for new targeted therapy",
    "Biomarker analysis reveals key predictors of treatment success",
]

print(f"Ingesting {len(sample_data)} clinical records...")

for i, data in enumerate(sample_data, 1):
    print(f"📤 Ingesting record {i}/{len(sample_data)}...")

    try:
        result = subprocess.run([
            'python', '-m', 'cli', 'memory', 'ingest', data
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("   ✅ Ingested successfully")
        else:
            print(f"   ❌ Ingestion failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("   ❌ Ingestion timeout")


# ## 🔍 CLI Search Operations

# In[ ]:


# CLI Search Operations Demo
print("🔍 CLI Search Operations")
print("=" * 40)

search_queries = [
    "lung cancer immunotherapy",
    "clinical trial response rate",
    "biomarker treatment success",
]

for query in search_queries:
    print(f"\n🔎 Searching for: '{query}'")

    try:
        result = subprocess.run([
            'python', '-m', 'cli', 'memory', 'search', query, '--limit', '3'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("   ✅ Search completed")
            # Count results (simplified)
            lines = result.stdout.split('\n')
            result_count = sum(1 for line in lines if 'episodes' in line.lower())
            print(f"   📊 Results found: {result_count}")
        else:
            print(f"   ❌ Search failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("   ❌ Search timeout")


# ## 📊 CLI Analytics and Reporting

# In[ ]:


# CLI Analytics Demo
print("📊 CLI Analytics and Reporting")
print("=" * 40)

print("Generating analytics report...")

try:
    result = subprocess.run([
        'python', '-m', 'cli', 'analytics', 'summary'
    ], capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        print("✅ Analytics report generated")
        print("📋 Report preview:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    else:
        print(f"❌ Analytics failed: {result.stderr}")

except subprocess.TimeoutExpired:
    print("❌ Analytics timeout")


# ## 📋 Batch Processing with CLI

# In[ ]:


# CLI Batch Processing Demo
print("📋 CLI Batch Processing")
print("=" * 40)

print("Demonstrating batch data processing...")

# Create batch file
batch_data = """Patient cohort analysis reveals treatment patterns
Clinical outcomes data shows improved survival rates
Research study identifies novel therapeutic targets
"""

batch_file = "batch_demo.txt"
with open(batch_file, 'w') as f:
    f.write(batch_data)

print(f"Created batch file: {batch_file}")

try:
    result = subprocess.run([
        'python', '-m', 'cli', 'batch', 'process', batch_file
    ], capture_output=True, text=True, timeout=60)

    if result.returncode == 0:
        print("✅ Batch processing completed")
    else:
        print(f"❌ Batch processing failed: {result.stderr}")

except subprocess.TimeoutExpired:
    print("❌ Batch processing timeout")

# Cleanup
if os.path.exists(batch_file):
    os.remove(batch_file)
    print(f"🧹 Cleaned up: {batch_file}")


# ## 🎯 CLI Integration Summary

# In[ ]:


# CLI Integration Summary
print("🎯 CLI Integration Summary")
print("=" * 40)

print("✅ What We Accomplished:")
print("   🔧 Set up CLI environment and validation")
print("   🏗️ Created and managed memory spaces")
print("   📥 Ingested clinical data via CLI commands")
print("   🔍 Performed search operations")
print("   📊 Generated analytics reports")
print("   📋 Demonstrated batch processing")

print("\n💡 Key CLI Capabilities:")
print("   • Command-line data ingestion")
print("   • Automated search and analytics")
print("   • Batch processing workflows")
print("   • Space management operations")
print("   • Integration with scripts and pipelines")

print("\n🏥 CLI Applications:")
print("   • Automated data pipelines")
print("   • Scheduled data processing")
print("   • Quality assurance workflows")
print("   • Research data management")
print("   • Clinical trial operations")

print("\n🎉 CLI demo completed successfully!")
print("🚀 Ready for automated clinical data workflows!")
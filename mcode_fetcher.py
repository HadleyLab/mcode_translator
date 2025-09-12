#!/usr/bin/env python3
"""
mCODE Fetcher - Clinical T  # Concurrent processing with mCODE mapping
  mcode_fetcher.py --condition "breast cancer" --concurrent --process \
    --workers 8 -m deepseek-coder -p direct_mcode_evidence_based_concisel Data Fetcher with Concurrent Processing

A high-performance clinical trial data fetcher that searches ClinicalTrials.gov
and optionally processes results with mCODE mapping using concurrent processing.

Author: mCODE Translation Team
Version: 2.0.0
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
import os
from typing import Dict, Any, List, Optional

from src.pipeline.concurrent_fetcher import (
    ConcurrentFetcher, 
    ProcessingConfig,
    concurrent_search_and_process,
    concurrent_process_trials
)
from src.pipeline.fetcher import (
    search_trials,
    get_full_study,
    calculate_total_studies,
    ClinicalTrialsAPIError
)
from src.utils.config import Config, ConfigurationError
from src.utils.logging_config import get_logger, setup_logging
from src.utils.core_memory_client import CoreMemoryClient, CoreMemoryError


def store_in_core_memory(results: Dict[str, Any], api_key: str) -> None:
    """Store results in CORE Memory."""
    logger = get_logger(__name__)
    logger.info("🧠 Storing results in CORE Memory")

    try:
        client = CoreMemoryClient(api_key=api_key)
        
        # Handle different result structures
        trial_list = []
        if "successful_trials" in results:
            # Concurrent processing results
            trial_list = results["successful_trials"]
            summary_message = f"mCODE Fetcher processed {results.get('summary', {}).get('total_trials', 0)} trials. " \
                              f"Successful: {results.get('summary', {}).get('successful_trials', 0)}, " \
                              f"Failed: {results.get('summary', {}).get('failed_trials', 0)}"
        elif "trials" in results:
            # Sequential search results
            trial_list = results["trials"]
            summary_message = f"mCODE Fetcher processed {results.get('total_found', 0)} trials."
        elif "trial" in results:
            # Single trial results
            trial_list = [results["trial"]]
            nct_id = results.get("nct_id", "unknown")
            summary_message = f"mCODE Fetcher processed 1 trial (NCT ID: {nct_id})."
        else:
            # Unknown structure
            logger.warning("Unknown results structure, storing raw results")
            summary_message = f"mCODE Fetcher processed results: {json.dumps(results)}"
        
        # Log batch processing summary but don't store in CORE Memory (only store individual trials/patients)
        logger.info(f"Batch processing summary: {summary_message}")

        # Ingest each trial
        for trial in trial_list:
            nct_id = trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "unknown")
            brief_title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'N/A')
            mcode_results = trial.get('McodeResults', {})
            
            # Store trial in CORE Memory
            logger.info(f"💾 Storing trial {nct_id} in CORE Memory")
            
            # Create a detailed summary with mCODE mappings
            if mcode_results:
                # Extract mCODE mappings for detailed storage
                mappings = mcode_results.get('mcode_mappings', [])
                validation = mcode_results.get('validation', {})

                # Extract relevant information
                eligibility_criteria = trial.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', 'N/A')
                cancer_condition = next((m.get('value', 'N/A') for m in mappings if m.get('mcode_element') == 'CancerCondition'), 'N/A')
                # Check for TNMStage first, then CancerStage
                cancer_stage = 'N/A'
                for m in mappings:
                    if m.get('mcode_element') == 'TNMStage':
                        cancer_stage = m.get('value', 'N/A')
                        break
                if cancer_stage == 'N/A':
                    for m in mappings:
                        if m.get('mcode_element') == 'CancerStage':
                            cancer_stage = m.get('value', 'N/A')
                            break
                histology = next((m.get('value', 'N/A') for m in mappings if m.get('mcode_element') == 'HistologyMorphologyBehavior'), 'N/A')
                treatments_list = [m.get('value', 'N/A') for m in mappings if m.get('mcode_element') == 'CancerTreatment']
                treatments = ", ".join(treatments_list) if treatments_list else 'Not specified'
                
                # Extract sponsor and location for context
                sponsor = trial.get('protocolSection', {}).get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name', 'N/A')
                location = trial.get('protocolSection', {}).get('contactsLocationsModule', {}).get('locations', [{}])[0].get('facility', 'N/A')
                
                # Create a narrative summary with all mCODE mappings in plain English
                summary_parts = [
                    f"Clinical Trial {nct_id} titled '{brief_title}' is recruiting patients for breast cancer treatment at {location}, sponsored by {sponsor} (mCODE:ClinicalTrial).",
                    f"This study targets {cancer_condition.replace(' (disorder)', '')} patients (mCODE:CancerCondition)."
                ]
                
                # Add staging information
                if cancer_stage != 'N/A':
                    summary_parts.append(f"Eligible patients have {cancer_stage} disease (mCODE:TNMStage).")
                
                # Add treatment information
                if treatments != 'Not specified':
                    treatment_list = [t.strip() for t in treatments.split(',') if t.strip() != 'N/A']
                    if treatment_list:
                        treatments_narrative = ", ".join(treatment_list[:-1]) + " and " + treatment_list[-1] if len(treatment_list) > 1 else treatment_list[0]
                        summary_parts.append(f"The trial evaluates {treatments_narrative} as cancer treatments (mCODE:CancerTreatment).")
                
                # Add detailed mCODE mappings as narrative sentences with codes
                summary_parts.append("Key eligibility and treatment criteria from mCODE analysis include:")
                
                # Group and narrate the 25 mappings with mCODE codes
                mapping_groups = {
                    "patient demographics": [],
                    "cancer characteristics": [],
                    "biomarker requirements": [],
                    "treatment modalities": [],
                    "genomic testing": [],
                    "other criteria": []
                }
                
                # Standard medical codes for common clinical trial elements
                standard_codes = {
                    # Patient demographics
                    'PatientSex': {'Female': '248152002'},  # SNOMED: Female sex
                    'PatientAge': {'Adult': '263537006'},   # SNOMED: Adult age group
                    
                    # Cancer conditions - common breast cancer codes
                    'CancerCondition': {
                        'Breast Cancer': '254837009',  # SNOMED: Malignant neoplasm of breast
                        'Early-stage Breast Cancer': '254837009',
                        'HR+/HER2- breast cancer': '254837009',
                        'HR+/HER2- invasive breast cancer': '254837009',
                        'ER+/HER2- early invasive breast cancer': '254837009',
                        'hormone receptor-positive (HR+) breast cancer': '254837009',
                        'breast cancer': '254837009'
                    },
                    
                    # Staging
                    'TNMStage': {
                        'T2N1M0': '258215001'  # SNOMED: Stage I breast cancer (adjust as needed)
                    },
                    
                    # Biomarkers
                    'ERStatus': {'>50%': '108283007'},  # SNOMED: ER positive
                    'HER2Status': {'Negative': '260385009'},  # SNOMED: Negative HER2
                    'Ki67Index': {'≥20%': '419377000'},  # SNOMED: Ki67 proliferation index
                    
                    # Treatments
                    'CancerRelatedMedication': {
                        'palbociclib': '716061000',  # SNOMED: Palbociclib
                        'ribociclib': '763340007',   # SNOMED: Ribociclib
                        'letrozole': '386878001',    # SNOMED: Letrozole
                        'anastrozole': '386871004',  # SNOMED: Anastrozole
                        'CDK4/6 Inhibitors': '763340007',  # Generic CDK4/6
                        'Endocrine Therapy': '278850018',  # SNOMED: Endocrine therapy
                        'Neoadjuvant Endocrine Therapy': '278850018'
                    },
                    
                    # Genomic
                    'CancerGenomicVariant': {
                        'Multigene Risk Score': '363344003'  # SNOMED: Multigene analysis
                    },
                    
                    # Tumor markers
                    'TumorMarker': {
                        'Ki-67 expression': '419377000',  # SNOMED: Ki67
                        'Ki67 dynamic changes': '419377000',
                        'Ki-67 Dynamic Assessment': '419377000'
                    },
                    
                    # Tumor marker tests
                    'TumorMarkerTest': {
                        'Ki-67 Dynamic Assessment': '48676-1'  # LOINC: Tumor marker test
                    }
                }
                
                for mapping in mappings:
                    element = mapping.get('mcode_element', '')
                    value = mapping.get('value', '')
                    
                    if not value or value == 'N/A':
                        continue
                    
                    # Try to get specific code from trial data first
                    trial_code = None
                    if element == 'CancerCondition':
                        # Look for specific condition codes in trial data
                        conditions = trial.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])
                        if value in conditions:
                            trial_code = f"ICD10:C50"  # Generic breast cancer code
                    elif element == 'CancerRelatedMedication':
                        # Look for drug names in interventions
                        interventions = trial.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])
                        for intervention in interventions:
                            if intervention.get('name') == value:
                                trial_code = intervention.get('name', value)
                                break
                    
                    # Use standard codes if available, otherwise raise a warning
                    code_info = standard_codes.get(element, {}).get(value)
                    code_system = "SNOMED" if code_info and code_info.isdigit() else "LOINC" if code_info and '-' in code_info else "ICD-10" if code_info and 'C' in code_info else "Trial-Specific"
                    
                    if not code_info:
                        logger.warning(f"No specific code found for mCODE element '{element}' with value '{value}'. Skipping.")
                        continue
                    
                    # Create narrative with mCODE reference including the code system
                    mcode_ref = f"(mCODE:{element}, {code_system}: {code_info})"
                    
                    # Categorize and create narrative sentences
                    if element in ['PatientSex', 'PatientAge', 'Race', 'Ethnicity']:
                        mapping_groups["patient demographics"].append(f"{value} patients {mcode_ref}")
                    elif element in ['CancerCondition', 'CancerStage', 'TNMStage', 'HistologyMorphologyBehavior']:
                        mapping_groups["cancer characteristics"].append(f"{value.replace(' (disorder)', '')} {mcode_ref}")
                    elif element in ['ERStatus', 'HER2Status', 'Ki67Index', 'TumorMarker', 'TumorMarkerTest']:
                        mapping_groups["biomarker requirements"].append(f"{value} {mcode_ref}")
                    elif element in ['CancerTreatment', 'CancerRelatedMedication']:
                        mapping_groups["treatment modalities"].append(f"{value} {mcode_ref}")
                    elif element in ['CancerGenomicVariant', 'GenomicVariant']:
                        mapping_groups["genomic testing"].append(f"{value} {mcode_ref}")
                    else:
                        mapping_groups["other criteria"].append(f"{value} {mcode_ref}")
                
                # Create narrative from grouped mappings
                for category, items in mapping_groups.items():
                    if items:
                        # For multiple items, clean up and format properly
                        clean_items = []
                        for item in items:
                            # Remove any existing category reference that might have been added
                            if f", {category})" in item:
                                clean_item = item.replace(f", {category})", ")")
                            else:
                                clean_item = item
                            clean_items.append(clean_item)
                        
                        narrative_items = ", ".join(clean_items)
                        summary_parts.append(f"The trial requires {category}: {narrative_items}.")
                
                # Eligibility criteria are mapped to mCODE, so no need to include the raw text
                
                summary = " ".join(summary_parts)
                logger.debug(f"CORE Memory summary for trial {nct_id}: {summary}")
                logger.debug(f"CORE Memory summary for trial {nct_id}: {summary}")
            else:
                # Fallback to simple narrative summary
                sponsor = trial.get('protocolSection', {}).get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name', 'N/A')
                location = trial.get('protocolSection', {}).get('contactsLocationsModule', {}).get('locations', [{}])[0].get('facility', 'N/A')
                eligibility_criteria = trial.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', 'N/A')
                
                summary = (
                    f"Clinical Trial {nct_id} titled '{brief_title}' is sponsored by {sponsor} "
                    f"and conducted at {location}. "
                    f"The trial focuses on breast cancer treatment and is seeking {cancer_condition.lower()} patients. "
                    f"Eligibility criteria: {eligibility_criteria[:300]}... "
                    f"No detailed mCODE mappings available for patient matching."
                )
                logger.debug(f"CORE Memory summary for trial {nct_id}: {summary}")

            client.ingest(summary)

        logger.info("✅ Successfully stored results in CORE Memory")

    except CoreMemoryError as e:
        logger.error(f"❌ Failed to store results in CORE Memory: {e}")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred while storing to CORE Memory: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with shared argument patterns."""
    parser = argparse.ArgumentParser(
        description="mCODE Fetcher - Clinical trial data fetcher with concurrent processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search and export
  mcode_fetcher.py --condition "breast cancer" --limit 10 --output results.json
  
  # Fetch specific trials
  mcode_fetcher.py --nct-id NCT12345678 --output trial.json
  mcode_fetcher.py --nct-ids "NCT001,NCT002,NCT003" --output trials.json
  
  # Search with mCODE processing (sequential)
  mcode_fetcher.py --condition "lung cancer" --process -m deepseek-coder
  
  # Concurrent processing with mCODE mapping
  mcode_fetcher.py --condition "breast cancer" --concurrent --process \\
    --workers 8 --model deepseek-coder --prompt direct_mcode_evidence_based_concise
  
  # Count total available studies
  mcode_fetcher.py --condition "cancer" --count-only
  
  # With CORE Memory storage
  mcode_fetcher.py --condition "breast cancer" --process --store-in-core-memory -m deepseek-coder
  mcode_fetcher.py --nct-id NCT00616135 --process --store-in-core-memory -m deepseek-coder
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--condition", "-c",
        help="Medical condition to search for (e.g., 'breast cancer')"
    )
    input_group.add_argument(
        "--nct-id", "-n",
        help="Specific NCT ID to fetch (e.g., 'NCT12345678')"
    )
    input_group.add_argument(
        "--nct-ids",
        help="Comma-separated list of NCT IDs to process"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path (JSON format). If not specified, prints to stdout"
    )
    
    # Search parameters
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)"
    )
    
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count total studies matching condition (no data fetch)"
    )
    
    # Processing options
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process results with mCODE mapping pipeline"
    )
    
    # Shared mCODE processing arguments (consistent with mcode_translator.py)
    parser.add_argument(
        "-m", "--model",
        help="LLM model to use for mCODE processing (overrides config)"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        default="direct_mcode_evidence_based_concise",
        help="Prompt template to use for mCODE processing (default: evidence-based concise)"
    )
    
    # Concurrent processing options
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Use concurrent processing for improved performance"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of concurrent workers (default: CPU count, only with --concurrent)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for concurrent processing (default: 10)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress updates during concurrent processing"
    )

    # New: Split output into per-trial files
    parser.add_argument(
        "--split-output",
        action="store_true",
        help="Save each trial's result as a separate file with NCTID and model name in the filename (in fetcher_output directory)"
    )
    
    # CORE Memory options
    parser.add_argument(
        "--store-in-core-memory",
        action="store_true",
        help="Store results in CORE Memory"
    )

    # Universal flags
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (overrides default)"
    )
    return parser


async def fetch_and_process_concurrent(args: argparse.Namespace) -> Dict[str, Any]:
    """Handle concurrent fetching and processing."""
    get_logger(__name__).info("🚀 Starting concurrent processing")
    
    if args.condition:
        # Concurrent search and process
        result = await concurrent_search_and_process(
            condition=args.condition,
            limit=args.limit,
            max_workers=args.workers,
            batch_size=args.batch_size,
            process_criteria=args.process,
            process_trials=args.process,
            model_name=args.model or "deepseek-coder",
            prompt_name=args.prompt,
            export_path=None,  # We'll handle export separately
            progress_updates=not args.no_progress
        )
        
        return {
            "search_condition": args.condition,
            "processing_type": "concurrent_search_and_process",
            "summary": {
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "failed_trials": result.failed_trials,
                "success_rate": (result.successful_trials / result.total_trials * 100) if result.total_trials > 0 else 0,
                "duration_seconds": result.duration_seconds,
                "processing_rate": result.total_trials / result.duration_seconds if result.duration_seconds > 0 else 0
            },
            "task_statistics": result.task_stats,
            "successful_trials": result.results,
            "failed_trials": result.errors
        }
    
    elif args.nct_ids:
        # Concurrent process specific trials
        nct_id_list = [nct_id.strip() for nct_id in args.nct_ids.split(',')]
        
        result = await concurrent_process_trials(
            nct_ids=nct_id_list,
            max_workers=args.workers,
            batch_size=args.batch_size,
            process_criteria=args.process,
            process_trials=args.process,
            model_name=args.model,
            prompt_name=args.prompt,
            export_path=None,  # We'll handle export separately
            progress_updates=not args.no_progress
        )
        
        return {
            "nct_ids": nct_id_list,
            "processing_type": "concurrent_process_trials",
            "summary": {
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "failed_trials": result.failed_trials,
                "success_rate": (result.successful_trials / result.total_trials * 100) if result.total_trials > 0 else 0,
                "duration_seconds": result.duration_seconds,
                "processing_rate": result.total_trials / result.duration_seconds if result.duration_seconds > 0 else 0
            },
            "task_statistics": result.task_stats,
            "successful_trials": result.results,
            "failed_trials": result.errors
        }
    
    else:
        raise ValueError("Concurrent processing requires either --condition or --nct-ids")


def fetch_and_process_sequential(args: argparse.Namespace) -> Dict[str, Any]:
    """Handle sequential fetching and processing."""
    logger = get_logger(__name__)
    config = Config()
    
    if args.condition:
        # Search for trials
        logger.info(f"🔍 Searching for trials: '{args.condition}' (limit: {args.limit})")
        search_result = search_trials(args.condition, fields=None, max_results=args.limit)
        trials = search_result.get('studies', [])
        
        if not trials:
            logger.warning("No trials found")
            return {
                "search_condition": args.condition,
                "processing_type": "sequential_search",
                "total_found": 0,
                "trials": []
            }
        
        logger.info(f"📋 Found {len(trials)} trials")
        
        # Process with mCODE if requested
        if args.process:
            logger.info("🔬 Processing trials with mCODE pipeline")
            from src.pipeline import McodePipeline
            
            pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)
            processed_trials = []
            
            # Initialize CORE Memory client if needed for real-time storage
            core_memory_client = None
            if args.store_in_core_memory:
                try:
                    api_key = config.get_core_memory_api_key()
                    core_memory_client = CoreMemoryClient(api_key=api_key)
                except Exception as e:
                    logger.warning(f"Could not initialize CORE Memory client for real-time storage: {e}")
            
            for i, trial in enumerate(trials):
                try:
                    logger.info(f"Processing trial {i+1}/{len(trials)}")
                    result = pipeline.process_clinical_trial(trial)
                    
                    # Add mCODE results to trial
                    enhanced_trial = trial.copy()
                    enhanced_trial['McodeResults'] = {
                        'extracted_entities': result.extracted_entities,
                        'mcode_mappings': result.mcode_mappings,
                        'source_references': result.source_references,
                        'validation': result.validation_results,
                        'metadata': result.metadata,
                        'error': result.error
                    }
                    processed_trials.append(enhanced_trial)
                    
                    # Store in CORE Memory in real-time if requested
                    # Skip CORE Memory storage during sequential search processing
                    # Only store in trials processing (single trial and multiple trials)
                    
                except Exception as e:
                    logger.error(f"Failed to process trial {i+1}: {e}")
                    trial['McodeProcessingError'] = str(e)
                    processed_trials.append(trial)
            
            trials = processed_trials
        
        return {
            "search_condition": args.condition,
            "processing_type": "sequential_search",
            "total_found": len(trials),
            "trials": trials
        }
    
    elif args.nct_id:
        # Fetch single trial
        logger.info(f"📥 Fetching trial: {args.nct_id}")
        trial = get_full_study(args.nct_id)
        
        # Process with mCODE if requested
        if args.process:
            logger.info("🔬 Processing trial with mCODE pipeline")
            from src.pipeline import McodePipeline
            
            pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)
            result = pipeline.process_clinical_trial(trial)
            
            # Add mCODE results to trial
            trial['McodeResults'] = {
                'extracted_entities': result.extracted_entities,
                'mcode_mappings': result.mcode_mappings,
                'source_references': result.source_references,
                'validation': result.validation_results,
                'metadata': result.metadata,
                'error': result.error
            }
            
            # Store in CORE Memory in real-time if requested
            if args.store_in_core_memory:
                try:
                    api_key = config.get_core_memory_api_key()
                    core_memory_client = CoreMemoryClient(api_key=api_key)
                    
                    nct_id = trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "unknown")
                    brief_title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'N/A')
                    mcode_results = trial.get('McodeResults', {})
                    
                    summary = (
                        f"Trial: {nct_id} - {brief_title}\n"
                        f"Mappings: {len(mcode_results.get('mcode_mappings', []))}, "
                        f"Compliance: {mcode_results.get('validation', {}).get('compliance_score', 'N/A')}"
                    )
                    core_memory_client.ingest(summary)
                    logger.info(f"✅ Stored trial {nct_id} in CORE Memory in real-time")
                except Exception as e:
                    logger.warning(f"Failed to store trial {args.nct_id} in CORE Memory: {e}")
        
        return {
            "nct_id": args.nct_id,
            "processing_type": "sequential_single",
            "trial": trial
        }
    
    elif args.nct_ids:
        # Fetch multiple trials sequentially
        nct_id_list = [nct_id.strip() for nct_id in args.nct_ids.split(',')]
        logger.info(f"📥 Fetching {len(nct_id_list)} trials sequentially")
        
        trials = []
        failed_trials = []
        
        # Initialize CORE Memory client if needed for real-time storage
        core_memory_client = None
        if args.store_in_core_memory:
            try:
                api_key = config.get_core_memory_api_key()
                core_memory_client = CoreMemoryClient(api_key=api_key)
            except Exception as e:
                logger.warning(f"Could not initialize CORE Memory client for real-time storage: {e}")
        
        for nct_id in nct_id_list:
            try:
                trial = get_full_study(nct_id)
                
                # Process with mCODE if requested
                if args.process:
                    from src.pipeline import McodePipeline
                    
                    pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)
                    result = pipeline.process_clinical_trial(trial)
                    
                    # Add mCODE results to trial
                    trial['McodeResults'] = {
                        'extracted_entities': result.extracted_entities,
                        'mcode_mappings': result.mcode_mappings,
                        'source_references': result.source_references,
                        'validation': result.validation_results,
                        'metadata': result.metadata,
                        'error': result.error
                    }
                
                trials.append(trial)
                logger.info(f"✅ Successfully processed {nct_id}")
                
                # Store in CORE Memory in real-time if requested
                if core_memory_client and args.process:
                    try:
                        nct_id = trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "unknown")
                        brief_title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'N/A')
                        mcode_results = trial.get('McodeResults', {})
                        
                        summary = (
                            f"Trial: {nct_id} - {brief_title}\n"
                            f"Mappings: {len(mcode_results.get('mcode_mappings', []))}, "
                            f"Compliance: {mcode_results.get('validation', {}).get('compliance_score', 'N/A')}"
                        )
                        core_memory_client.ingest(summary)
                        logger.info(f"✅ Stored trial {nct_id} in CORE Memory in real-time")
                    except Exception as e:
                        logger.warning(f"Failed to store trial {nct_id} in CORE Memory: {e}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {nct_id}: {e}")
                failed_trials.append({"nct_id": nct_id, "error": str(e)})
        
        return {
            "nct_ids": nct_id_list,
            "processing_type": "sequential_multiple",
            "successful_count": len(trials),
            "failed_count": len(failed_trials),
            "trials": trials,
            "failed_trials": failed_trials
        }


def count_studies(args: argparse.Namespace) -> Dict[str, Any]:
    """Count total studies for a condition."""
    logger = get_logger(__name__)
    logger.info(f"📊 Counting studies for condition: '{args.condition}'")
    
    stats = calculate_total_studies(args.condition)
    
    return {
        "condition": args.condition,
        "total_studies": stats['total_studies'],
        "total_pages": stats['total_pages'],
        "page_size": stats['page_size']
    }


def output_results(results: Dict[str, Any], output_path: Optional[str]) -> None:
    """Output results to file or stdout."""
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to {output_path}")
    else:
        print(json.dumps(results, indent=2))


async def main() -> None:
    """Main entry point for the mCODE Fetcher CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    logger = get_logger(__name__)
    logger.info("🚀 mCODE Fetcher starting")
    
    try:
        # Initialize configuration
        config = Config()
        if args.config:
            logger.info(f"Custom config specified: {args.config}")

        # Handle count-only requests
        if args.count_only:
            if not args.condition:
                parser.error("--count-only requires --condition")
            results = count_studies(args)
            output_results(results, args.output)
            return

        # Handle concurrent vs sequential processing
        if args.concurrent and args.process:
            logger.info("🔄 Using concurrent processing with mCODE mapping")
            results = await fetch_and_process_concurrent(args)
        else:
            if args.concurrent:
                logger.warning("⚠️  --concurrent specified but --process not enabled. Using sequential processing.")
            logger.info("📝 Using sequential processing")
            results = fetch_and_process_sequential(args)

        # Output results
        output_results(results, args.output)

        # New: Split output into per-trial files if requested
        if args.split_output:
            import os
            output_dir = os.path.join("data", "fetcher_output")
            os.makedirs(output_dir, exist_ok=True)
            model_name = args.model if args.model else "model"
            # Support both concurrent and sequential output formats
            trial_list = []
            if "successful_trials" in results:
                trial_list = results["successful_trials"]
            elif "trials" in results:
                trial_list = results["trials"]
            elif "trial" in results:
                trial_list = [results["trial"]]
            for trial in trial_list:
                # Extract NCTID
                nct_id = None
                try:
                    nct_id = trial["protocolSection"]["identificationModule"]["nctId"]
                except Exception:
                    nct_id = "unknown"
                filename = f"{nct_id}.{model_name}.results.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w") as f:
                    json.dump(trial, f, indent=2)
                print(f"✅ Saved per-trial result: {filepath}")

        # Store in CORE Memory if requested
        if args.store_in_core_memory:
            try:
                api_key = config.get_core_memory_api_key()
                store_in_core_memory(results, api_key)
            except ConfigurationError as e:
                logger.error(f"❌ Could not store to CORE Memory: {e}")

        logger.info("✅ mCODE Fetcher completed successfully")

    except ClinicalTrialsAPIError as e:
        logger.error(f"❌ ClinicalTrials.gov API Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
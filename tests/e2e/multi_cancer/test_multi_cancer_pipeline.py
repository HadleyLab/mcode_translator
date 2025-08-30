#!/usr/bin/env python3
"""
Comprehensive Test Script for Cancer-Specific Prompt Optimization

This script demonstrates the framework's capabilities with the new
comprehensive cancer test cases, showing how it handles diverse
clinical scenarios across multiple cancer types.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path for proper src package discovery
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.optimization.strict_prompt_optimization_framework import (
    StrictPromptOptimizationFramework,
    PromptType,
    APIConfig,
    PromptVariant
)


class ComprehensiveCancerDemo:
    """Demonstration of comprehensive cancer test case capabilities"""
    
    def __init__(self):
        self.framework = StrictPromptOptimizationFramework()
        self.results = []
    
    def _mock_pipeline_callback(self, test_case: Dict[str, Any], prompt_template: str) -> Any:
        """Mock pipeline callback that simulates different responses based on cancer type"""
        title = test_case.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', '').lower()
        
        class MockResult:
            def __init__(self, cancer_type):
                # Simulate different extraction results based on cancer type
                if 'pancreatic' in cancer_type:
                    self.extracted_entities = [
                        {"text": "pancreatic cancer", "type": "condition", "confidence": 0.92},
                        {"text": "metastatic", "type": "stage", "confidence": 0.85},
                        {"text": "gemcitabine", "type": "treatment", "confidence": 0.88}
                    ]
                    self.mcode_mappings = [
                        {"resourceType": "Condition", "code": {"text": "pancreatic cancer"}},
                        {"resourceType": "MedicationAdministration", "code": {"text": "gemcitabine"}}
                    ]
                    self.validation_results = {"compliance_score": 0.87}
                    
                elif 'ovarian' in cancer_type:
                    self.extracted_entities = [
                        {"text": "ovarian cancer", "type": "condition", "confidence": 0.94},
                        {"text": "platinum-sensitive", "type": "biomarker", "confidence": 0.82},
                        {"text": "carboplatin", "type": "treatment", "confidence": 0.91}
                    ]
                    self.mcode_mappings = [
                        {"resourceType": "Condition", "code": {"text": "ovarian cancer"}},
                        {"resourceType": "Observation", "code": {"text": "platinum-sensitive"}}
                    ]
                    self.validation_results = {"compliance_score": 0.89}
                    
                elif 'glioblastoma' in cancer_type:
                    self.extracted_entities = [
                        {"text": "glioblastoma", "type": "condition", "confidence": 0.96},
                        {"text": "MGMT methylated", "type": "biomarker", "confidence": 0.79},
                        {"text": "temozolomide", "type": "treatment", "confidence": 0.93}
                    ]
                    self.mcode_mappings = [
                        {"resourceType": "Condition", "code": {"text": "glioblastoma"}},
                        {"resourceType": "Observation", "code": {"text": "MGMT methylation status"}}
                    ]
                    self.validation_results = {"compliance_score": 0.85}
                    
                else:
                    # Generic cancer response
                    self.extracted_entities = [
                        {"text": "cancer", "type": "condition", "confidence": 0.9},
                        {"text": "clinical trial", "type": "study_type", "confidence": 0.8}
                    ]
                    self.mcode_mappings = [
                        {"resourceType": "Condition", "code": {"text": "malignant neoplasm"}}
                    ]
                    self.validation_results = {"compliance_score": 0.82}
        
        # Detect cancer type from title
        cancer_type = self._detect_cancer_type(title)
        return MockResult(cancer_type)
    
    def _detect_cancer_type(self, title: str) -> str:
        """Detect cancer type from clinical trial title"""
        title_lower = title.lower()
        if 'pancreatic' in title_lower:
            return 'pancreatic'
        elif 'ovarian' in title_lower:
            return 'ovarian'
        elif 'glioblastoma' in title_lower:
            return 'glioblastoma'
        elif 'bladder' in title_lower:
            return 'bladder'
        elif 'multiple myeloma' in title_lower:
            return 'multiple_myeloma'
        elif 'head and neck' in title_lower or 'head & neck' in title_lower:
            return 'head_neck'
        elif 'gastric' in title_lower or 'stomach' in title_lower:
            return 'gastric'
        elif 'renal cell' in title_lower or 'kidney' in title_lower:
            return 'renal_cell'
        else:
            return 'other'
    
    def load_comprehensive_configurations(self) -> None:
        """Load comprehensive configurations for cancer testing"""
        print("📋 Loading comprehensive cancer test configurations...")
        
        # Add basic API config
        api_config = APIConfig(
            name="cancer_test_api",
            base_url="https://api.deepseek.com/v1",
            api_key="mock_key_for_testing",
            model="deepseek-coder",
            temperature=0.2,
            max_tokens=4000
        )
        self.framework.add_api_config(api_config)
        print("  ✅ Added API config: cancer_test_api")
        
        # Add comprehensive prompt variants
        extraction_variants = [
            PromptVariant(
                name="comprehensive_extraction",
                prompt_type=PromptType.NLP_EXTRACTION,
                template="Extract all clinical entities from: {text}",
                description="Comprehensive entity extraction for cancer trials",
                version="2.0.0"
            ),
            PromptVariant(
                name="targeted_extraction",
                prompt_type=PromptType.NLP_EXTRACTION,
                template="Focus on cancer-specific entities: {text}",
                description="Targeted extraction for oncology",
                version="1.5.0"
            )
        ]
        
        mapping_variants = [
            PromptVariant(
                name="comprehensive_mapping",
                prompt_type=PromptType.MCODE_MAPPING,
                template="Map clinical entities to mCODE: {entities}",
                description="Comprehensive mCODE mapping",
                version="2.0.0"
            ),
            PromptVariant(
                name="cancer_specific_mapping",
                prompt_type=PromptType.MCODE_MAPPING,
                template="Map oncology entities to mCODE standards: {entities}",
                description="Cancer-specific mCODE mapping",
                version="1.8.0"
            )
        ]
        
        for variant in extraction_variants + mapping_variants:
            self.framework.add_prompt_variant(variant)
            print(f"  ✅ Added prompt variant: {variant.name}")
        
        # Load comprehensive cancer test cases
        test_case_path = Path("tests/data/test_cases/multi_cancer.json")
        if test_case_path.exists():
            self.framework.load_test_cases_from_file(str(test_case_path))
            print(f"  ✅ Loaded {len(self.framework.test_cases)} comprehensive cancer test cases")
            
            # Show cancer type distribution
            cancer_counts = self._analyze_cancer_types()
            print(f"  🎯 Cancer types distribution:")
            for cancer_type, count in cancer_counts.items():
                print(f"     - {cancer_type}: {count} cases")
        else:
            print("  ⚠️  Comprehensive cancer test cases not found")
            return False
        
        return True
    
    def _analyze_cancer_types(self) -> Dict[str, int]:
        """Analyze cancer types in loaded test cases"""
        cancer_counts = {}
        for test_data in self.framework.test_cases.values():
            title = test_data.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', '')
            cancer_type = self._detect_cancer_type(title)
            cancer_counts[cancer_type] = cancer_counts.get(cancer_type, 0) + 1
        return cancer_counts
    
    async def run_comprehensive_benchmark(self) -> None:
        """Run comprehensive benchmark across cancer types"""
        print("\n🧪 Running comprehensive cancer benchmark...")
        
        # Get available configurations
        api_configs = list(self.framework.api_configs.keys())
        prompt_variants = list(self.framework.prompt_variants.keys())
        test_cases = list(self.framework.test_cases.keys())
        
        print(f"  Configurations:")
        print(f"  - API Configs: {len(api_configs)}")
        print(f"  - Prompt Variants: {len(prompt_variants)}")
        print(f"  - Test Cases: {len(test_cases)}")
        
        # Run NLP extraction benchmark on subset of cases
        print("\n🔬 Running NLP Extraction Benchmark...")
        nlp_results = self.framework.run_comprehensive_benchmark(
            prompt_type=PromptType.NLP_EXTRACTION,
            pipeline_callback=self._mock_pipeline_callback,
            api_config_names=api_configs,
            prompt_variant_ids=[v for v in prompt_variants if 'extraction' in v],
            test_case_ids=test_cases[:5]  # First 5 test cases
        )
        
        print(f"  ✅ Completed {len(nlp_results)} NLP extraction experiments")
        self.results.extend(nlp_results)
        
        # Run MCode mapping benchmark on subset of cases
        print("\n🧬 Running MCode Mapping Benchmark...")
        mcode_results = self.framework.run_comprehensive_benchmark(
            prompt_type=PromptType.MCODE_MAPPING,
            pipeline_callback=self._mock_pipeline_callback,
            api_config_names=api_configs,
            prompt_variant_ids=[v for v in prompt_variants if 'mapping' in v],
            test_case_ids=test_cases[:3]  # First 3 test cases
        )
        
        print(f"  ✅ Completed {len(mcode_results)} MCode mapping experiments")
        self.results.extend(mcode_results)
    
    def analyze_results(self) -> None:
        """Analyze and report benchmark results"""
        print("\n📊 Analyzing comprehensive benchmark results...")
        
        # Load all results
        self.framework.load_benchmark_results()
        df = self.framework.get_results_dataframe()
        
        if not df.empty:
            print(f"  📈 Overall Statistics:")
            print(f"    Total experiments: {len(df)}")
            print(f"    Success rate: {df['success'].mean():.1%}")
            print(f"    Average duration: {df['duration_ms'].mean():.1f} ms")
            
            if 'entities_extracted' in df.columns:
                print(f"    Average entities extracted: {df['entities_extracted'].mean():.1f}")
            
            if 'compliance_score' in df.columns:
                print(f"    Average compliance score: {df['compliance_score'].mean():.3f}")
        
        # Generate performance report
        report = self.framework.generate_performance_report()
        if 'error' not in report:
            print(f"\n🏆 Performance Report:")
            print(f"  Total experiments: {report.get('total_experiments', 0)}")
            print(f"  Overall success rate: {report.get('success_rate', 0):.1%}")
            
            if 'best_configs' in report:
                for category, config in report['best_configs'].items():
                    print(f"  Best {category}: {config.get('name', 'N/A')} "
                          f"(score: {config.get('score', 0):.3f})")
        
        # Export to CSV
        csv_path = self.framework.export_results_to_csv()
        print(f"\n📋 Results exported to CSV: {csv_path}")
    
    async def run_complete_demo(self) -> None:
        """Run the complete comprehensive cancer demo"""
        print("=" * 70)
        print("🩺 COMPREHENSIVE CANCER TEST CASE DEMONSTRATION")
        print("=" * 70)
        
        try:
            # Load configurations
            if not self.load_comprehensive_configurations():
                print("❌ Failed to load configurations")
                return False
            
            # Run benchmarks
            await self.run_comprehensive_benchmark()
            
            # Analyze results
            self.analyze_results()
            
            print("\n" + "=" * 70)
            print("🎉 COMPREHENSIVE CANCER DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("\nNext steps:")
            print("1. Review the generated results and reports")
            print("2. Launch web interface: python -m src.optimization.optimization_ui")
            print("3. Explore cancer-specific visualizations in the UI")
            print("4. Run additional benchmarks with different configurations")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main function to run the comprehensive cancer demo"""
    demo = ComprehensiveCancerDemo()
    success = await demo.run_complete_demo()
    
    if success:
        print("\n✅ Comprehensive cancer testing framework is ready!")
        print("   Access the web interface at http://localhost:8081")
    else:
        print("\n❌ Comprehensive cancer demo encountered errors")

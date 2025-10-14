#!/usr/bin/env python3
"""
Expert Panel Demonstration - Comprehensive comparison of individual vs combined expert performance.

This script demonstrates the expert panel system by comparing:
1. Individual expert performance (clinical reasoning, pattern recognition, comprehensive analyst)
2. Expert panel combined performance with ensemble decision making
3. Real-world patient-trial matching examples with detailed analysis
4. Performance metrics and optimization insights
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from src.matching.clinical_expert_agent import ClinicalExpertAgent
from src.matching.expert_panel_manager import ExpertPanelManager
from src.utils.config import Config
from src.utils.logging_config import get_logger


class ExpertPanelDemo:
    """Demonstration class for expert panel system."""

    def __init__(self):
        """Initialize the demonstration."""
        self.logger = get_logger(__name__)
        self.config = Config()

        # Initialize individual experts
        self.individual_experts = {}
        self._initialize_individual_experts()

        # Initialize expert panel
        self.expert_panel = ExpertPanelManager(
            model_name="deepseek-coder",
            config=self.config,
            max_concurrent_experts=3,
            enable_diversity_selection=True
        )

        # Demo data
        self.patient_cases = self._create_demo_patient_cases()
        self.trial_cases = self._create_demo_trial_cases()

    def _initialize_individual_experts(self):
        """Initialize individual expert agents for comparison."""
        expert_types = ["clinical_reasoning", "pattern_recognition", "comprehensive_analyst"]

        for expert_type in expert_types:
            try:
                expert = ClinicalExpertAgent(
                    model_name="deepseek-coder",
                    expert_type=expert_type,
                    config=self.config
                )
                self.individual_experts[expert_type] = expert
                self.logger.info(f"✅ Initialized individual {expert_type} expert")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {expert_type} expert: {e}")

    def _create_demo_patient_cases(self) -> List[Dict[str, Any]]:
        """Create realistic patient cases for demonstration."""
        return [
            {
                "id": "patient_001",
                "age": 45,
                "gender": "female",
                "conditions": ["breast cancer"],
                "stage": "II",
                "biomarkers": {"ER": "positive", "PR": "positive", "HER2": "negative"},
                "performance_status": 0,
                "comorbidities": ["hypertension"],
                "current_medications": ["lisinopril"],
                "treatment_history": ["lumpectomy", "radiation therapy"]
            },
            {
                "id": "patient_002",
                "age": 62,
                "gender": "male",
                "conditions": ["lung cancer", "COPD"],
                "stage": "III",
                "biomarkers": {"EGFR": "positive", "ALK": "negative"},
                "performance_status": 1,
                "comorbidities": ["COPD", "diabetes", "hypertension"],
                "current_medications": ["metformin", "insulin", "tiotropium"],
                "treatment_history": ["lobectomy", "chemotherapy"]
            },
            {
                "id": "patient_003",
                "age": 58,
                "gender": "female",
                "conditions": ["ovarian cancer"],
                "stage": "IV",
                "biomarkers": {"BRCA1": "positive", "BRCA2": "negative"},
                "performance_status": 2,
                "comorbidities": ["anemia", "neuropathy"],
                "current_medications": ["carboplatin", "paclitaxel"],
                "treatment_history": ["debulking surgery", "multiple chemotherapy regimens"]
            }
        ]

    def _create_demo_trial_cases(self) -> List[Dict[str, Any]]:
        """Create realistic clinical trial cases for demonstration."""
        return [
            {
                "trial_id": "NCT00123456",
                "title": "Phase II Study of Targeted Therapy in ER+ Breast Cancer",
                "conditions": ["breast cancer"],
                "eligibilityCriteria": """
                Inclusion Criteria:
                - Postmenopausal women with ER-positive, HER2-negative breast cancer
                - Stage II-III disease
                - ECOG performance status 0-1
                - Adequate organ function
                - No prior chemotherapy for metastatic disease

                Exclusion Criteria:
                - Significant cardiovascular disease
                - Uncontrolled hypertension
                - Active infection
                """,
                "phase": "Phase II",
                "age_min": 40,
                "age_max": 70,
                "biomarkers": {"ER": "positive", "HER2": "negative"},
                "performance_status_max": 1
            },
            {
                "trial_id": "NCT00234567",
                "title": "Phase III EGFR Targeted Therapy in NSCLC",
                "conditions": ["lung cancer", "NSCLC"],
                "eligibilityCriteria": """
                Inclusion Criteria:
                - EGFR-positive non-small cell lung cancer
                - Stage III-IV disease
                - ECOG performance status 0-2
                - Adequate pulmonary function
                - No untreated brain metastases

                Exclusion Criteria:
                - Severe COPD requiring oxygen
                - Uncontrolled diabetes
                - Recent myocardial infarction
                """,
                "phase": "Phase III",
                "age_min": 18,
                "age_max": 75,
                "biomarkers": {"EGFR": "positive"},
                "performance_status_max": 2
            },
            {
                "trial_id": "NCT00345678",
                "title": "Phase II PARP Inhibitor Trial in BRCA+ Ovarian Cancer",
                "conditions": ["ovarian cancer"],
                "eligibilityCriteria": """
                Inclusion Criteria:
                - BRCA1/2 positive ovarian cancer
                - Stage III-IV disease
                - ECOG performance status 0-2
                - Platinum-sensitive disease
                - Adequate bone marrow function

                Exclusion Criteria:
                - Severe anemia requiring transfusion
                - Grade 3-4 neuropathy
                - Prior PARP inhibitor therapy
                """,
                "phase": "Phase II",
                "age_min": 18,
                "age_max": 80,
                "biomarkers": {"BRCA1": "positive"},
                "performance_status_max": 2
            }
        ]

    async def run_individual_expert_comparison(self, patient_idx: int = 0, trial_idx: int = 0) -> Dict[str, Any]:
        """Run comparison of individual expert performances."""
        self.logger.info("🔬 Running Individual Expert Performance Comparison")
        self.logger.info("=" * 60)

        patient_data = self.patient_cases[patient_idx]
        trial_criteria = self.trial_cases[trial_idx]

        self.logger.info(f"📋 Patient: {patient_data['id']} - {patient_data['conditions']} (Age: {patient_data['age']})")
        self.logger.info(f"🧪 Trial: {trial_criteria['trial_id']} - {trial_criteria['title']}")
        self.logger.info("")

        # Test individual experts
        individual_results = {}

        for expert_type, expert in self.individual_experts.items():
            self.logger.info(f"🔍 Testing {expert_type.replace('_', ' ').title()} Expert")
            self.logger.info("-" * 40)

            start_time = time.time()
            try:
                result = await expert.assess_match(patient_data, trial_criteria)
                processing_time = time.time() - start_time

                individual_results[expert_type] = {
                    "result": result,
                    "processing_time": processing_time,
                    "success": True
                }

                self.logger.info(f"⏱️  Processing Time: {processing_time:.2f}s")
                self.logger.info(f"🎯 Match Decision: {'✅ MATCH' if result.get('is_match') else '❌ NO MATCH'}")
                self.logger.info(f"📊 Confidence Score: {result.get('confidence_score', 0):.3f}")
                self.logger.info(f"💬 Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
                self.logger.info("")

            except Exception as e:
                processing_time = time.time() - start_time
                individual_results[expert_type] = {
                    "result": None,
                    "processing_time": processing_time,
                    "success": False,
                    "error": str(e)
                }
                self.logger.error(f"❌ {expert_type} assessment failed: {e}")
                self.logger.info("")

        return individual_results

    async def run_expert_panel_assessment(self, patient_idx: int = 0, trial_idx: int = 0) -> Dict[str, Any]:
        """Run expert panel assessment for comparison."""
        self.logger.info("🎭 Running Expert Panel Assessment")
        self.logger.info("=" * 60)

        patient_data = self.patient_cases[patient_idx]
        trial_criteria = self.trial_cases[trial_idx]

        self.logger.info(f"📋 Patient: {patient_data['id']} - {patient_data['conditions']} (Age: {patient_data['age']})")
        self.logger.info(f"🧪 Trial: {trial_criteria['trial_id']} - {trial_criteria['title']}")
        self.logger.info("")

        start_time = time.time()
        try:
            panel_result = await self.expert_panel.assess_with_expert_panel(
                patient_data, trial_criteria
            )
            processing_time = time.time() - start_time

            self.logger.info(f"⏱️  Panel Processing Time: {processing_time:.2f}s")
            self.logger.info(f"🎯 Ensemble Decision: {'✅ MATCH' if panel_result.get('is_match') else '❌ NO MATCH'}")
            self.logger.info(f"📊 Ensemble Confidence: {panel_result.get('confidence_score', 0):.3f}")
            self.logger.info(f"🤝 Consensus Level: {panel_result.get('consensus_level', 'unknown')}")
            self.logger.info(f"👥 Experts Used: {len(panel_result.get('expert_assessments', []))}")
            self.logger.info(f"💬 Ensemble Reasoning: {panel_result.get('reasoning', 'N/A')[:100]}...")
            self.logger.info("")

            return {
                "result": panel_result,
                "processing_time": processing_time,
                "success": True
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Expert panel assessment failed: {e}")
            return {
                "result": None,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }

    def compare_results(self, individual_results: Dict[str, Any], panel_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare individual expert results with panel results."""
        self.logger.info("⚖️  Results Comparison Analysis")
        self.logger.info("=" * 60)

        comparison = {
            "individual_analysis": {},
            "panel_analysis": {},
            "performance_comparison": {},
            "decision_consistency": {},
            "insights": []
        }

        # Analyze individual results
        for expert_type, data in individual_results.items():
            if data["success"] and data["result"]:
                result = data["result"]
                comparison["individual_analysis"][expert_type] = {
                    "decision": result.get("is_match"),
                    "confidence": result.get("confidence_score", 0),
                    "processing_time": data["processing_time"],
                    "matched_criteria_count": len(result.get("matched_criteria", [])),
                    "unmatched_criteria_count": len(result.get("unmatched_criteria", []))
                }

        # Analyze panel result
        if panel_result["success"] and panel_result["result"]:
            result = panel_result["result"]
            comparison["panel_analysis"] = {
                "decision": result.get("is_match"),
                "confidence": result.get("confidence_score", 0),
                "processing_time": panel_result["processing_time"],
                "consensus_level": result.get("consensus_level"),
                "experts_used": len(result.get("expert_assessments", [])),
                "matched_criteria_count": len(result.get("matched_criteria", [])),
                "unmatched_criteria_count": len(result.get("unmatched_criteria", []))
            }

        # Performance comparison
        if comparison["individual_analysis"] and comparison["panel_analysis"]:
            individual_times = [data["processing_time"] for data in individual_results.values()
                              if data["success"]]
            avg_individual_time = sum(individual_times) / len(individual_times)

            comparison["performance_comparison"] = {
                "avg_individual_time": avg_individual_time,
                "panel_time": comparison["panel_analysis"]["processing_time"],
                "time_savings": avg_individual_time - comparison["panel_analysis"]["processing_time"],
                "parallel_efficiency": (avg_individual_time - comparison["panel_analysis"]["processing_time"]) / avg_individual_time * 100
            }

        # Decision consistency analysis
        individual_decisions = [data["decision"] for data in comparison["individual_analysis"].values()]
        panel_decision = comparison["panel_analysis"]["decision"]

        match_votes = sum(1 for decision in individual_decisions if decision)
        consistency_ratio = match_votes / len(individual_decisions) if individual_decisions else 0

        comparison["decision_consistency"] = {
            "individual_decisions": individual_decisions,
            "panel_decision": panel_decision,
            "agreement_ratio": consistency_ratio,
            "consensus_achieved": consistency_ratio >= 0.67  # 2/3 majority
        }

        # Generate insights
        insights = []

        if comparison["performance_comparison"]:
            perf = comparison["performance_comparison"]
            if perf["parallel_efficiency"] > 0:
                insights.append(f"🚀 Panel achieved {perf['parallel_efficiency']:.1f}% better performance through parallelization")
            else:
                insights.append("⚠️  Panel overhead reduced parallelization benefits")

        if comparison["decision_consistency"]:
            consistency = comparison["decision_consistency"]
            if consistency["consensus_achieved"]:
                insights.append(f"✅ Experts achieved {consistency['agreement_ratio']:.1%} consensus")
            else:
                insights.append(f"⚠️  Low consensus ({consistency['agreement_ratio']:.1%}) - panel decision may need review")

        comparison["insights"] = insights

        # Log comparison results
        self._log_comparison_results(comparison)

        return comparison

    def _log_comparison_results(self, comparison: Dict[str, Any]):
        """Log detailed comparison results."""
        self.logger.info("📊 Individual Expert Results:")
        for expert_type, data in comparison["individual_analysis"].items():
            self.logger.info(f"  {expert_type.replace('_', ' ').title()}:")
            self.logger.info(f"    Decision: {'✅' if data['decision'] else '❌'} (Confidence: {data['confidence']:.3f})")
            self.logger.info(f"    Time: {data['processing_time']:.2f}s")
            self.logger.info(f"    Criteria: +{data['matched_criteria_count']} -{data['unmatched_criteria_count']}")

        self.logger.info("\n🎭 Expert Panel Results:")
        panel = comparison["panel_analysis"]
        self.logger.info(f"  Decision: {'✅' if panel['decision'] else '❌'} (Confidence: {panel['confidence']:.3f})")
        self.logger.info(f"  Time: {panel['processing_time']:.2f}s")
        self.logger.info(f"  Consensus: {panel['consensus_level']}")
        self.logger.info(f"  Experts Used: {panel['experts_used']}")
        self.logger.info(f"  Criteria: +{panel['matched_criteria_count']} -{panel['unmatched_criteria_count']}")

        if comparison["performance_comparison"]:
            perf = comparison["performance_comparison"]
            self.logger.info("\n⚡ Performance Comparison:")
            self.logger.info(f"  Avg Individual Time: {perf['avg_individual_time']:.2f}s")
            self.logger.info(f"  Panel Time: {perf['panel_time']:.2f}s")
            self.logger.info(f"  Time Savings: {perf['time_savings']:.2f}s ({perf['parallel_efficiency']:.1f}%)")

        if comparison["decision_consistency"]:
            consistency = comparison["decision_consistency"]
            self.logger.info("\n🤝 Decision Consistency:")
            self.logger.info(f"  Individual Decisions: {consistency['individual_decisions']}")
            self.logger.info(f"  Panel Decision: {consistency['panel_decision']}")
            self.logger.info(f"  Agreement Ratio: {consistency['agreement_ratio']:.1%}")
            self.logger.info(f"  Consensus Achieved: {'✅' if consistency['consensus_achieved'] else '❌'}")

        if comparison["insights"]:
            self.logger.info("\n💡 Key Insights:")
            for insight in comparison["insights"]:
                self.logger.info(f"  {insight}")

    async def run_full_demonstration(self):
        """Run the complete expert panel demonstration."""
        self.logger.info("🚀 Starting Expert Panel Demonstration")
        self.logger.info("=" * 80)

        # Test different patient-trial combinations
        test_cases = [
            (0, 0, "Breast Cancer Case"),
            (1, 1, "Lung Cancer Case"),
            (2, 2, "Ovarian Cancer Case")
        ]

        all_results = {}

        for patient_idx, trial_idx, case_name in test_cases:
            self.logger.info(f"\n🧪 Test Case: {case_name}")
            self.logger.info("=" * 60)

            # Run individual expert comparison
            individual_results = await self.run_individual_expert_comparison(patient_idx, trial_idx)

            # Run expert panel assessment
            panel_result = await self.run_expert_panel_assessment(patient_idx, trial_idx)

            # Compare results
            comparison = self.compare_results(individual_results, panel_result)

            all_results[f"case_{patient_idx}_{trial_idx}"] = {
                "individual_results": individual_results,
                "panel_result": panel_result,
                "comparison": comparison
            }

        # Generate summary
        await self.generate_demonstration_summary(all_results)

        return all_results

    async def generate_demonstration_summary(self, all_results: Dict[str, Any]):
        """Generate comprehensive summary of the demonstration."""
        self.logger.info("\n📋 DEMONSTRATION SUMMARY")
        self.logger.info("=" * 80)

        total_cases = len(all_results)
        successful_panels = 0
        total_time_savings = 0
        consensus_achieved = 0

        for case_name, case_data in all_results.items():
            if case_data["panel_result"]["success"]:
                successful_panels += 1

            comparison = case_data["comparison"]
            if comparison["performance_comparison"]:
                total_time_savings += comparison["performance_comparison"]["time_savings"]

            if comparison["decision_consistency"]:
                if comparison["decision_consistency"]["consensus_achieved"]:
                    consensus_achieved += 1

        self.logger.info(f"📊 Test Results Summary:")
        self.logger.info(f"  Total Cases Tested: {total_cases}")
        self.logger.info(f"  Successful Panels: {successful_panels}/{total_cases}")
        self.logger.info(f"  Success Rate: {(successful_panels/total_cases)*100:.1f}%")
        self.logger.info(f"  Total Time Savings: {total_time_savings:.2f}s")
        self.logger.info(f"  Consensus Achieved: {consensus_achieved}/{total_cases}")
        self.logger.info(f"  Consensus Rate: {(consensus_achieved/total_cases)*100:.1f}%")

        # Performance insights
        self.logger.info("\n💡 Key Performance Insights:")
        if total_time_savings > 0:
            self.logger.info(f"  ✅ Parallel processing saved {total_time_savings:.2f} of processing time")
        else:
            self.logger.info("  ⚠️  Parallel processing overhead observed - may need optimization")

        if consensus_achieved >= total_cases * 0.8:
            self.logger.info("  ✅ High expert consensus demonstrates reliable decision making")
        else:
            self.logger.info("  ⚠️  Variable consensus suggests need for expert panel tuning")

        # Cache performance
        self.logger.info("\n💾 Cache Performance:")
        panel_stats = self.expert_panel._get_panel_cache_stats()
        self.logger.info(f"  Panel Hit Rate: {panel_stats['panel_hit_rate']:.2%}")
        self.logger.info(f"  Time Saved by Cache: {panel_stats['panel_total_time_saved_seconds']:.2f}s")

        # Recommendations
        self.logger.info("\n🎯 Recommendations:")
        if successful_panels == total_cases:
            self.logger.info("  ✅ Expert panel system is working reliably")
        else:
            self.logger.info("  ⚠️  Review failed cases for system improvements")

        if panel_stats['panel_hit_rate'] > 0.5:
            self.logger.info("  ✅ Caching system is providing good performance benefits")
        else:
            self.logger.info("  ⚠️  Consider cache optimization for better hit rates")

        self.logger.info("\n🎉 Expert Panel Demonstration Complete!")
    def shutdown(self):
        """Shutdown the demonstration and cleanup resources."""
        self.logger.info("🔄 Shutting down Expert Panel Demonstration")

        if self.expert_panel:
            self.expert_panel.shutdown()

        self.logger.info("✅ Demonstration shutdown complete")


async def main():
    """Main demonstration function."""
    demo = ExpertPanelDemo()

    try:
        await demo.run_full_demonstration()
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
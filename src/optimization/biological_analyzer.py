"""
Biological Analysis Module - Analyzes biological content of clinical trials.

This module handles the analysis of trial biology including conditions,
interventions, demographics, and mCODE element extraction patterns.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from shared.models import McodeElement
from utils.logging_config import get_logger


class BiologicalAnalyzer:
    """
    Analyzes the biological content of clinical trial data.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def analyze_trial_biology(self, trials_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the biological content of trial data with robust error handling."""
        biology_stats: Dict[str, Any] = {
            "total_trials": len(trials_data),
            "conditions": {},
            "interventions": {},
            "phases": {},
            "study_types": {},
            "ages": [],
            "genders": {},
            "locations": {},
        }

        for trial in trials_data:
            try:
                # Extract conditions - handle nested structure safely
                if isinstance(trial, dict) and "protocolSection" in trial:
                    protocol = trial["protocolSection"]
                    if isinstance(protocol, dict):

                        # Conditions
                        if "conditionsModule" in protocol and isinstance(
                            protocol["conditionsModule"], dict
                        ):
                            conditions = protocol["conditionsModule"].get("conditions", [])
                            if isinstance(conditions, list):
                                for condition in conditions:
                                    if isinstance(condition, dict):
                                        cond_name = condition.get("condition", "Unknown")
                                        if isinstance(cond_name, str):
                                            conditions_dict = biology_stats["conditions"]
                                            if isinstance(conditions_dict, dict):
                                                conditions_dict[cond_name] = (
                                                    conditions_dict.get(cond_name, 0) + 1
                                                )

                        # Interventions
                        if "armsInterventionsModule" in protocol and isinstance(
                            protocol["armsInterventionsModule"], dict
                        ):
                            interventions = protocol["armsInterventionsModule"].get(
                                "interventions", []
                            )
                            if isinstance(interventions, list):
                                for intervention in interventions:
                                    if isinstance(intervention, dict):
                                        int_type = intervention.get("type", "Unknown")
                                        if isinstance(int_type, str):
                                            interventions_dict = biology_stats["interventions"]
                                            if isinstance(interventions_dict, dict):
                                                interventions_dict[int_type] = (
                                                    interventions_dict.get(int_type, 0) + 1
                                                )

                        # Study phase
                        if "designModule" in protocol and isinstance(
                            protocol["designModule"], dict
                        ):
                            phases = protocol["designModule"].get("phases", [])
                            if isinstance(phases, list) and phases:
                                phase = phases[0] if isinstance(phases[0], str) else "Unknown"
                                biology_stats["phases"][phase] = (
                                    biology_stats["phases"].get(phase, 0) + 1
                                )

                        # Study type
                        if "identificationModule" in protocol and isinstance(
                            protocol["identificationModule"], dict
                        ):
                            study_type = protocol["identificationModule"].get(
                                "studyType", "Unknown"
                            )
                            if isinstance(study_type, str):
                                biology_stats["study_types"][study_type] = (
                                    biology_stats["study_types"].get(study_type, 0) + 1
                                )

                        # Eligibility criteria (age, gender)
                        if "eligibilityModule" in protocol and isinstance(
                            protocol["eligibilityModule"], dict
                        ):
                            eligibility = protocol["eligibilityModule"]

                            # Gender
                            gender = eligibility.get("sex", "Unknown")
                            if isinstance(gender, str):
                                biology_stats["genders"][gender] = (
                                    biology_stats["genders"].get(gender, 0) + 1
                                )

                            # Age extraction (simplified)
                            criteria = eligibility.get("eligibilityCriteria", "")
                            if isinstance(criteria, str) and "years" in criteria.lower():
                                # Simple age extraction - could be enhanced
                                biology_stats["ages"].append("Has age criteria")

            except Exception as e:
                # Skip malformed trial data but continue processing
                self.logger.debug(f"Skipping malformed trial data: {e}")
                continue

        return biology_stats

    def analyze_mcode_elements(
        self,
        combo_results: Dict[int, Dict[str, Any]],
        combinations: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Analyze mCODE elements generated across all combinations using proper McodeElement models."""
        analysis = {}

        for combo_idx, combo in enumerate(combinations):
            combo_key = f"{combo['model']}_{combo['prompt']}"
            raw_elements = combo_results[combo_idx]["mcode_elements"]

            # Convert raw elements to McodeElement objects for proper analysis
            mcode_elements = []
            for raw_element in raw_elements:
                try:
                    # Ensure we have the required element_type field
                    if isinstance(raw_element, dict) and "element_type" in raw_element:
                        element = McodeElement(**raw_element)
                        mcode_elements.append(element)
                except Exception as e:
                    self.logger.debug(f"Skipping invalid mCODE element: {e}")
                    continue

            combo_analysis: Dict[str, Any] = {
                "total_elements": len(mcode_elements),
                "element_types": {},
                "confidence_distribution": [],
                "evidence_quality": [],
                "biological_categories": {
                    "cancer_conditions": 0,
                    "treatments": 0,
                    "patient_characteristics": 0,
                    "tumor_staging": 0,
                    "genetic_markers": 0,
                    "other": 0,
                },
                "top_conditions": {},
                "top_treatments": {},
                "evidence_sources": {},
            }

            for element in mcode_elements:
                # Element type distribution
                elem_type = element.element_type
                element_types_dict = combo_analysis["element_types"]
                if isinstance(element_types_dict, dict):
                    element_types_dict[elem_type] = element_types_dict.get(elem_type, 0) + 1

                # Confidence scores
                confidence = element.confidence_score or 0.0
                confidence_list = combo_analysis["confidence_distribution"]
                if isinstance(confidence_list, list):
                    confidence_list.append(confidence)

                # Biological categorization using proper element types
                biological_categories = combo_analysis["biological_categories"]
                if isinstance(biological_categories, dict):
                    if elem_type in [
                        "CancerCondition",
                        "PrimaryCancerCondition",
                        "SecondaryCancerCondition",
                    ]:
                        biological_categories["cancer_conditions"] += 1
                        condition = element.display or "Unknown"
                        top_conditions = combo_analysis["top_conditions"]
                        if isinstance(top_conditions, dict):
                            top_conditions[condition] = top_conditions.get(condition, 0) + 1
                    elif elem_type in [
                        "CancerTreatment",
                        "ChemotherapyTreatment",
                        "TargetedTherapy",
                        "RadiationTreatment",
                    ]:
                        biological_categories["treatments"] += 1
                        treatment = element.display or "Unknown"
                        top_treatments = combo_analysis["top_treatments"]
                        if isinstance(top_treatments, dict):
                            top_treatments[treatment] = top_treatments.get(treatment, 0) + 1
                    elif elem_type in ["PatientDemographics", "Patient"]:
                        biological_categories["patient_characteristics"] += 1
                    elif elem_type in ["TNMStage", "CancerStage"]:
                        biological_categories["tumor_staging"] += 1
                    elif element.display and any(
                        marker in element.display.upper()
                        for marker in ["HER2", "ER+", "ER-", "PR+", "PR-", "BRCA"]
                    ):
                        biological_categories["genetic_markers"] += 1
                    else:
                        biological_categories["other"] += 1

                # Evidence quality assessment using proper evidence_text field
                evidence = element.evidence_text or ""
                if evidence:
                    # Enhanced evidence quality scoring
                    quality_score = 0
                    if len(evidence) > 50:
                        quality_score += 1  # Substantial evidence
                    if any(
                        keyword in evidence.lower()
                        for keyword in [
                            "patient",
                            "treatment",
                            "cancer",
                            "study",
                            "clinical",
                        ]
                    ):
                        quality_score += 1
                    if any(
                        term in evidence.lower()
                        for term in [
                            "stage",
                            "grade",
                            "metastatic",
                            "recurrent",
                            "neoadjuvant",
                            "adjuvant",
                        ]
                    ):
                        quality_score += 1
                    if element.confidence_score and element.confidence_score > 0.8:
                        quality_score += 1  # High confidence
                    evidence_quality_list = combo_analysis["evidence_quality"]
                    if isinstance(evidence_quality_list, list):
                        evidence_quality_list.append(quality_score)

                    # Evidence source tracking
                    evidence_sources = combo_analysis["evidence_sources"]
                    if isinstance(evidence_sources, dict):
                        if "patients" in evidence.lower() or "eligibility" in evidence.lower():
                            evidence_sources["patient_criteria"] = (
                                evidence_sources.get("patient_criteria", 0) + 1
                            )
                        elif "treatment" in evidence.lower() or "intervention" in evidence.lower():
                            evidence_sources["treatment_info"] = (
                                evidence_sources.get("treatment_info", 0) + 1
                            )
                        elif "cancer" in evidence.lower() or "condition" in evidence.lower():
                            evidence_sources["condition_info"] = (
                                evidence_sources.get("condition_info", 0) + 1
                            )
                        elif "study" in evidence.lower() or "trial" in evidence.lower():
                            evidence_sources["study_design"] = (
                                evidence_sources.get("study_design", 0) + 1
                            )
                        else:
                            evidence_sources["other"] = evidence_sources.get("other", 0) + 1

            analysis[combo_key] = combo_analysis

        return analysis

    def generate_comparative_analysis(
        self, mcode_analysis: Dict[str, Any], combinations: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Generate comparative analysis across models and prompts."""
        comparative: Dict[str, Any] = {
            "model_comparison": {},
            "prompt_comparison": {},
            "best_performers": {
                "most_elements": None,
                "highest_quality": None,
                "best_conditions": None,
                "best_treatments": None,
            },
        }

        # Group by model and prompt
        model_stats: Dict[str, Dict[str, Any]] = {}
        prompt_stats: Dict[str, Dict[str, Any]] = {}

        for combo_key, analysis in mcode_analysis.items():
            model = combo_key.split("_")[0]
            prompt = "_".join(combo_key.split("_")[1:])

            # Model aggregation
            if model not in model_stats:
                model_stats[model] = {
                    "combinations": 0,
                    "total_elements": 0,
                    "avg_confidence": 0,
                    "quality_scores": [],
                }
            model_stats[model]["combinations"] += 1
            model_stats[model]["total_elements"] += analysis["total_elements"]
            if analysis["confidence_distribution"]:
                model_stats[model]["avg_confidence"] = sum(
                    analysis["confidence_distribution"]
                ) / len(analysis["confidence_distribution"])
            model_stats[model]["quality_scores"].extend(analysis["evidence_quality"])

            # Prompt aggregation
            if prompt not in prompt_stats:
                prompt_stats[prompt] = {
                    "combinations": 0,
                    "total_elements": 0,
                    "avg_confidence": 0,
                    "quality_scores": [],
                }
            prompt_stats[prompt]["combinations"] += 1
            prompt_stats[prompt]["total_elements"] += analysis["total_elements"]
            if analysis["confidence_distribution"]:
                prompt_stats[prompt]["avg_confidence"] = sum(
                    analysis["confidence_distribution"]
                ) / len(analysis["confidence_distribution"])
            prompt_stats[prompt]["quality_scores"].extend(analysis["evidence_quality"])

        comparative["model_comparison"] = model_stats
        comparative["prompt_comparison"] = prompt_stats

        # Find best performers
        if mcode_analysis:
            # Most elements
            comparative["best_performers"]["most_elements"] = max(
                mcode_analysis.items(), key=lambda x: x[1]["total_elements"]
            )

            # Highest quality (average confidence)
            comparative["best_performers"]["highest_quality"] = max(
                mcode_analysis.items(),
                key=lambda x: (
                    sum(x[1]["confidence_distribution"]) / len(x[1]["confidence_distribution"])
                    if x[1]["confidence_distribution"]
                    else 0
                ),
            )

            # Best condition coverage
            comparative["best_performers"]["best_conditions"] = max(
                mcode_analysis.items(),
                key=lambda x: x[1]["biological_categories"]["cancer_conditions"],
            )

            # Best treatment coverage
            comparative["best_performers"]["best_treatments"] = max(
                mcode_analysis.items(),
                key=lambda x: x[1]["biological_categories"]["treatments"],
            )

        return comparative

    def generate_markdown_report(
        self,
        trial_biology: Dict[str, Any],
        mcode_analysis: Dict[str, Any],
        comparative_analysis: Dict[str, Any],
        combinations: List[Dict[str, str]],
    ) -> str:
        """Generate comprehensive markdown report."""
        report = []

        # Header
        report.append("# mCODE Translation Optimization Report")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Combinations Tested:** {len(combinations)}")
        report.append("")

        # Trial Biology Overview
        report.append("## Trial Data Biology Overview")
        report.append(f"- **Total Trials:** {trial_biology['total_trials']}")
        report.append(f"- **Primary Conditions:** {len(trial_biology['conditions'])} unique")
        report.append(f"- **Intervention Types:** {len(trial_biology['interventions'])} types")
        report.append("")

        # Top conditions
        if trial_biology["conditions"]:
            report.append("### Most Common Conditions")
            sorted_conditions = sorted(
                trial_biology["conditions"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            for condition, count in sorted_conditions:
                report.append(f"- {condition}: {count} trials")
            report.append("")

        # mCODE Analysis Summary
        report.append("## mCODE Generation Analysis")
        report.append("")

        # Summary table
        report.append(
            "| Combination | Elements | Conditions | Treatments | Avg Confidence | Quality Score |"
        )
        report.append(
            "|-------------|----------|------------|------------|----------------|---------------|"
        )

        for combo_key, analysis in mcode_analysis.items():
            elements = analysis["total_elements"]
            conditions = analysis["biological_categories"]["cancer_conditions"]
            treatments = analysis["biological_categories"]["treatments"]
            avg_conf = (
                sum(analysis["confidence_distribution"]) / len(analysis["confidence_distribution"])
                if analysis["confidence_distribution"]
                else 0
            )
            quality = (
                sum(analysis["evidence_quality"]) / len(analysis["evidence_quality"])
                if analysis["evidence_quality"]
                else 0
            )

            report.append(
                f"| {combo_key} | {elements} | {conditions} | {treatments} | {avg_conf:.2f} | {quality:.2f} |"
            )

        report.append("")

        # Comparative Analysis
        report.append("## Comparative Analysis")
        report.append("")

        # Model comparison
        if comparative_analysis["model_comparison"]:
            report.append("### Model Performance")
            for model, stats in comparative_analysis["model_comparison"].items():
                avg_elements = stats["total_elements"] / stats["combinations"]
                avg_quality = (
                    sum(stats["quality_scores"]) / len(stats["quality_scores"])
                    if stats["quality_scores"]
                    else 0
                )
                report.append(
                    f"- **{model}**: {avg_elements:.1f} avg elements, {stats['avg_confidence']:.2f} avg confidence, {avg_quality:.2f} quality"
                )
            report.append("")

        # Best performers
        if comparative_analysis["best_performers"]["most_elements"]:
            best_elements = comparative_analysis["best_performers"]["most_elements"]
            report.append("### Best Performers")
            report.append(
                f"- **Most Elements:** {best_elements[0]} ({best_elements[1]['total_elements']} elements)"
            )

            best_quality = comparative_analysis["best_performers"]["highest_quality"]
            avg_conf = (
                sum(best_quality[1]["confidence_distribution"])
                / len(best_quality[1]["confidence_distribution"])
                if best_quality[1]["confidence_distribution"]
                else 0
            )
            report.append(
                f"- **Highest Quality:** {best_quality[0]} ({avg_conf:.2f} avg confidence)"
            )

            best_conditions = comparative_analysis["best_performers"]["best_conditions"]
            report.append(
                f"- **Best Condition Coverage:** {best_conditions[0]} ({best_conditions[1]['biological_categories']['cancer_conditions']} conditions)"
            )

            best_treatments = comparative_analysis["best_performers"]["best_treatments"]
            report.append(
                f"- **Best Treatment Coverage:** {best_treatments[0]} ({best_treatments[1]['biological_categories']['treatments']} treatments)"
            )
            report.append("")

        # Detailed Element Analysis
        report.append("## Detailed Element Analysis")
        report.append("")

        for combo_key, analysis in mcode_analysis.items():
            report.append(f"### {combo_key}")
            report.append(f"- **Total Elements:** {analysis['total_elements']}")

            # Element type breakdown
            if analysis["element_types"]:
                report.append("- **Element Types:**")
                for elem_type, count in sorted(
                    analysis["element_types"].items(), key=lambda x: x[1], reverse=True
                ):
                    report.append(f"  - {elem_type}: {count}")

            # Biological categories
            report.append("- **Biological Categories:**")
            for category, count in analysis["biological_categories"].items():
                if count > 0:
                    report.append(f"  - {category.replace('_', ' ').title()}: {count}")

            # Top conditions and treatments
            if analysis["top_conditions"]:
                report.append("- **Top Conditions:**")
                for condition, count in sorted(
                    analysis["top_conditions"].items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    report.append(f"  - {condition}: {count}")

            if analysis["top_treatments"]:
                report.append("- **Top Treatments:**")
                for treatment, count in sorted(
                    analysis["top_treatments"].items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    report.append(f"  - {treatment}: {count}")

            report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        if comparative_analysis["best_performers"]["highest_quality"]:
            best_combo = comparative_analysis["best_performers"]["highest_quality"][0]
            model, prompt = best_combo.split("_", 1)
            report.append(f"1. **Recommended Configuration:** {model} + {prompt.replace('_', ' ')}")
            report.append("   - Highest quality mCODE mappings with best evidence support")

        if comparative_analysis["best_performers"]["most_elements"]:
            most_combo = comparative_analysis["best_performers"]["most_elements"][0]
            report.append(f"2. **High Volume Option:** {most_combo.replace('_', ' + ')}")
            report.append("   - Generates the most comprehensive mCODE elements")

        report.append("")
        report.append("---")
        report.append("*Report generated by mCODE Translation Optimizer*")

        return "\n".join(report)

    def generate_biological_analysis_report(
        self,
        combo_results: Dict[int, Dict[str, Any]],
        combinations: List[Dict[str, str]],
        trials_data: List[Dict[str, Any]],
    ) -> None:
        """Generate comprehensive biological and mCODE analysis report."""
        try:
            # Analyze trial biology
            trial_biology = self.analyze_trial_biology(trials_data)

            # Analyze mCODE elements
            mcode_analysis = self.analyze_mcode_elements(combo_results, combinations)

            # Generate comparative analysis
            comparative_analysis = self.generate_comparative_analysis(mcode_analysis, combinations)

            # Generate markdown report
            report = self.generate_markdown_report(
                trial_biology, mcode_analysis, comparative_analysis, combinations
            )

            # Save report
            report_path = (
                Path("optimization_runs")
                / f"biological_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            self.logger.info(f"📊 Biological analysis report saved to: {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate biological analysis report: {e}")
            raise

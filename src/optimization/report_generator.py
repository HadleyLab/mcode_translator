"""
Report Generation Module - Generates comprehensive optimization reports.

This module handles the generation of various reports including mega reports,
biological analysis reports, and comparative analysis reports.
"""

from typing import Dict, Any
from datetime import datetime

from src.utils.logging_config import get_logger


class ReportGenerator:
    """
    Generates comprehensive optimization reports.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def generate_mega_report(
        self,
        analysis: Dict[str, Any],
        biological_content: str = "",
        inter_rater_content: str = "",
    ) -> str:
        """Generate a comprehensive mega report aggregating all optimization runs."""

        # Get best performers
        best_model = None
        if analysis.get("model_stats"):
            best_model_data = max(
                analysis["model_stats"].items(),
                key=lambda x: x[1]["successful"] / max(x[1]["runs"], 1),
            )
            best_model = best_model_data[0]

        # Extract mCODE coverage from biological analysis
        mcode_coverage = self._extract_mcode_coverage(biological_content)

        report = f"""# mCODE Translation Optimization - Actionable Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Executive Summary & Recommendations

### Best Configuration for Production Use
"""

        if best_model:
            report += f"**🏆 Recommended Model:** `{best_model}`\n"
            model_stats = analysis["model_stats"][best_model]
            success_rate = model_stats["successful"] / max(model_stats["runs"], 1) * 100
            report += f"- **Success Rate:** {success_rate:.1f}%\n"
            report += f"- **Average Score:** {model_stats['avg_score']:.3f}\n"
            report += f"- **Expected mCODE Coverage:** {mcode_coverage.get('avg_elements', 'Unknown')}\n\n"

        report += f"""### Expected mCODE Element Coverage
- **Average Elements per Trial:** {analysis.get('performance_stats', {}).get('avg_elements', 0):.1f}
- **Most Common Elements:** Cancer Conditions, Treatments, Patient Demographics
- **Reliability:** {self._extract_reliability_summary(inter_rater_content)}

### Key Findings
- **Total Optimization Runs:** {analysis.get('total_runs', 0)}
- **Success Rate:** {analysis.get('successful_runs', 0)/max(analysis.get('total_runs', 1), 1)*100:.1f}%
- **Time Range:** {analysis.get('time_range', {}).get('earliest', 'N/A')} to {analysis.get('time_range', {}).get('latest', 'N/A')}

## 📊 Model Performance & Reliability

### Top Performing Models
| Rank | Model | Success Rate | Avg Score | Elements | Reliability |
|------|-------|-------------|-----------|----------|-------------|
"""

        # Sort models by success rate and score
        models = []
        for model, stats in analysis.get("model_stats", {}).items():
            success_rate = stats["successful"] / max(stats["runs"], 1) * 100
            models.append((model, success_rate, stats["avg_score"], stats["runs"]))

        models.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for i, (model, success_rate, avg_score, runs) in enumerate(models[:5], 1):
            reliability = self._get_model_reliability(model, inter_rater_content)
            elements = mcode_coverage.get("model_elements", {}).get(model, "N/A")
            report += f"| {i} | {model} | {success_rate:.1f}% | {avg_score:.3f} | {elements} | {reliability} |\n"

        report += "\n### Provider Comparison\n"
        report += "| Provider | Models | Success Rate | Avg Score | Cost ($/run) |\n"
        report += "|----------|--------|-------------|-----------|-------------|\n"

        for provider, stats in analysis.get("provider_stats", {}).items():
            success_rate = stats["successful"] / max(stats["runs"], 1) * 100
            avg_cost = stats.get("avg_cost", 0)
            models_count = len(stats.get("models", []))
            report += f"| {provider} | {models_count} | {success_rate:.1f}% | {stats['avg_score']:.3f} | ${avg_cost:.4f} |\n"

        # mCODE Element Mapping Across Combinations
        report += "\n## 🗺️ mCODE Element Mapping by Configuration\n\n"
        report += "### Element Coverage Matrix\n"
        report += "| Configuration | Total Elements | Cancer Conditions | Treatments | Demographics | Staging | Biomarkers |\n"
        report += "|---------------|----------------|------------------|------------|-------------|---------|------------|\n"

        # Extract combination data from biological content
        combinations_data = self._extract_combinations_data(biological_content)
        for combo_key, data in combinations_data.items():
            total = data.get("total_elements", 0)
            conditions = data.get("biological_categories", {}).get(
                "cancer_conditions", 0
            )
            treatments = data.get("biological_categories", {}).get("treatments", 0)
            demographics = data.get("biological_categories", {}).get(
                "patient_characteristics", 0
            )
            staging = data.get("biological_categories", {}).get("tumor_staging", 0)
            biomarkers = data.get("biological_categories", {}).get("genetic_markers", 0)
            report += f"| {combo_key.replace('_', ' + ')} | {total} | {conditions} | {treatments} | {demographics} | {staging} | {biomarkers} |\n"

        # Inter-rater Reliability Section
        if inter_rater_content:
            report += "\n## 🤝 Inter-Rater Reliability Analysis\n\n"
            # Extract key metrics from inter-rater report
            reliability_metrics = self._extract_reliability_metrics(inter_rater_content)
            report += f"""### Agreement Metrics
- **Presence Agreement:** {reliability_metrics.get('presence_agreement', 'N/A')}
- **Values Agreement:** {reliability_metrics.get('values_agreement', 'N/A')}
- **Confidence Agreement:** {reliability_metrics.get('confidence_agreement', 'N/A')}
- **Fleiss' Kappa:** {reliability_metrics.get('fleiss_kappa', 'N/A')}

### Rater Performance
"""
            rater_performance = self._extract_rater_performance(inter_rater_content)
            for rater, stats in rater_performance.items():
                report += f"- **{rater}:** {stats.get('success_rate', 'N/A')} success, {stats.get('avg_elements', 'N/A')} elements\n"

        # Error Analysis
        report += "\n## ⚠️ Error Analysis & Troubleshooting\n\n"
        error_analysis = analysis.get("error_analysis", {})
        if error_analysis:
            total_errors = sum(error_analysis.values())
            report += f"**Total Errors:** {total_errors}\n\n"
            report += "| Error Type | Count | Percentage | Action Required |\n"
            report += "|------------|-------|------------|----------------|\n"

            error_actions = {
                "quota_exceeded": "Increase API limits or reduce concurrent requests",
                "json_parsing": "Fix prompt formatting and JSON parsing logic",
                "rate_limit": "Implement exponential backoff and request throttling",
                "auth_error": "Check API keys and authentication setup",
                "network_error": "Improve error handling and retry logic",
                "timeout": "Increase timeout limits or optimize processing",
                "api_error": "Check API compatibility and update client libraries",
                "model_error": "Verify model availability and update model names",
                "other": "Investigate logs for specific error patterns",
            }

            for error_type, count in sorted(
                error_analysis.items(), key=lambda x: x[1], reverse=True
            ):
                if count > 0:
                    percentage = count / max(total_errors, 1) * 100
                    action = error_actions.get(error_type, "Review error logs")
                    report += f"| {error_type.replace('_', ' ').title()} | {count} | {percentage:.1f}% | {action} |\n"

        # Biological Analysis Summary
        if biological_content:
            report += "\n## 🔬 Biological Content Analysis\n\n"
            # Extract key biological insights
            bio_insights = self._extract_biological_insights(biological_content)
            report += f"""### Trial Characteristics
- **Total Trials Analyzed:** {bio_insights.get('total_trials', 'N/A')}
- **Primary Conditions:** {', '.join(bio_insights.get('top_conditions', [])[:3])}
- **Intervention Types:** {', '.join(bio_insights.get('intervention_types', [])[:3])}

### Most Extracted mCODE Elements
"""
            for element_type, count in bio_insights.get(
                "element_distribution", {}
            ).items():
                if count > 0:
                    report += f"- **{element_type}:** {count} extractions\n"

        # Actionable Recommendations
        report += "\n## 🎯 Actionable Recommendations\n\n"

        if best_model:
            report += "### 1. Production Deployment\n"
            report += f"   - Use **{best_model}** for production mCODE extraction\n"
            report += f"   - Expected reliability: {self._get_model_reliability(best_model, inter_rater_content)}\n"
            report += "   - Monitor for the error patterns identified above\n\n"

        report += "### 2. Performance Optimization\n"
        if analysis.get("model_stats"):
            fastest_model = min(
                analysis["model_stats"].items(),
                key=lambda x: x[1].get("avg_processing_time", float("inf")),
            )
            report += f"   - Fastest model: **{fastest_model[0]}** ({fastest_model[1].get('avg_processing_time', 0):.1f}s avg)\n"

            cheapest_model = min(
                analysis["model_stats"].items(),
                key=lambda x: x[1].get("avg_cost", float("inf")),
            )
            report += f"   - Most cost-effective: **{cheapest_model[0]}** (${cheapest_model[1].get('avg_cost', 0):.4f} avg)\n\n"

        report += "### 3. Quality Assurance\n"
        report += "   - Implement inter-rater reliability checks for new models\n"
        report += "   - Monitor element coverage against expected baselines\n"
        report += "   - Set up automated error pattern detection\n\n"

        report += "### 4. Future Improvements\n"
        report += "   - Focus optimization on top-performing model families\n"
        report += (
            "   - Investigate reliability gaps in underperforming configurations\n"
        )
        report += "   - Expand biological validation with clinical expert review\n\n"

        report += "---\n*Generated by mCODE Translation Optimizer - Actionable Intelligence for Production Use*"

        return report

    def _extract_mcode_coverage(self, biological_content: str) -> Dict[str, Any]:
        """Extract mCODE coverage information from biological report."""
        coverage = {"avg_elements": "Unknown", "model_elements": {}}

        if not biological_content:
            return coverage

        lines = biological_content.split("\n")
        for line in lines:
            # Look for average elements
            if "avg elements" in line.lower():
                try:
                    # Extract number from line like "deepseek-coder: 50.0 avg elements"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        num_str = parts[1].strip().split()[0]
                        if num_str.replace(".", "").isdigit():
                            coverage["avg_elements"] = float(num_str)
                except:
                    pass

            # Look for model-specific element counts
            if "elements |" in line and "|" in line:
                # Parse table rows
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 2:
                    model = parts[0].replace(" + ", "_")
                    try:
                        elements = int(parts[1])
                        coverage["model_elements"][model] = elements
                    except:
                        pass

        return coverage

    def _extract_reliability_summary(self, inter_rater_content: str) -> str:
        """Extract a summary of inter-rater reliability."""
        if not inter_rater_content:
            return "Not analyzed"

        # Look for key metrics
        lines = inter_rater_content.split("\n")
        for line in lines:
            if "presence agreement" in line.lower():
                try:
                    # Extract percentage
                    if "%" in line:
                        pct = line.split("%")[0].split()[-1]
                        if pct.replace(".", "").isdigit():
                            return f"{pct}% agreement"
                except:
                    pass

        return "Analysis available"

    def _get_model_reliability(self, model: str, inter_rater_content: str) -> str:
        """Get reliability rating for a specific model."""
        if not inter_rater_content:
            return "Unknown"

        # Look for model in rater performance section
        lines = inter_rater_content.split("\n")
        in_rater_section = False

        for line in lines:
            if "### Rater Performance" in line:
                in_rater_section = True
                continue
            elif in_rater_section and line.startswith("##"):
                break

            if in_rater_section and model.lower() in line.lower():
                # Extract success rate
                if "%" in line:
                    try:
                        pct = line.split("%")[0].split()[-1]
                        if pct.replace(".", "").isdigit():
                            return f"{pct}%"
                    except:
                        pass

        return "Unknown"

    def _extract_combinations_data(self, biological_content: str) -> Dict[str, Dict]:
        """Extract combination data from biological report."""
        combinations = {}

        if not biological_content:
            return combinations

        lines = biological_content.split("\n")
        in_table = False

        for line in lines:
            if "| Combination | Elements |" in line:
                in_table = True
                continue
            elif (
                in_table
                and line.startswith("| ")
                and "|" in line
                and not line.startswith("|---")
            ):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 6:
                    combo_key = parts[0].replace(" + ", "_")
                    try:
                        total_elements = int(parts[1])
                        cancer_conditions = int(parts[2])
                        treatments = int(parts[3])
                        demographics = int(parts[4])
                        staging = int(parts[5])
                        biomarkers = int(parts[6]) if len(parts) > 6 else 0

                        combinations[combo_key] = {
                            "total_elements": total_elements,
                            "biological_categories": {
                                "cancer_conditions": cancer_conditions,
                                "treatments": treatments,
                                "patient_characteristics": demographics,
                                "tumor_staging": staging,
                                "genetic_markers": biomarkers,
                            },
                        }
                    except:
                        pass
            elif in_table and not line.startswith("|"):
                break

        return combinations

    def _extract_reliability_metrics(self, inter_rater_content: str) -> Dict[str, str]:
        """Extract reliability metrics from inter-rater report."""
        metrics = {}

        if not inter_rater_content:
            return metrics

        lines = inter_rater_content.split("\n")
        for line in lines:
            line_lower = line.lower()
            if "presence agreement" in line_lower:
                if "%" in line:
                    try:
                        pct = line.split("%")[0].split()[-1]
                        metrics["presence_agreement"] = f"{pct}%"
                    except:
                        pass
            elif "values agreement" in line_lower:
                if "%" in line:
                    try:
                        pct = line.split("%")[0].split()[-1]
                        metrics["values_agreement"] = f"{pct}%"
                    except:
                        pass
            elif "confidence agreement" in line_lower:
                if "%" in line:
                    try:
                        pct = line.split("%")[0].split()[-1]
                        metrics["confidence_agreement"] = f"{pct}%"
                    except:
                        pass
            elif "fleiss" in line_lower and "kappa" in line_lower:
                try:
                    # Extract kappa value
                    kappa_part = line.split(":")[-1].strip()
                    metrics["fleiss_kappa"] = kappa_part
                except:
                    pass

        return metrics

    def _extract_rater_performance(self, inter_rater_content: str) -> Dict[str, Dict]:
        """Extract rater performance data."""
        performance = {}

        if not inter_rater_content:
            return performance

        lines = inter_rater_content.split("\n")
        in_performance = False

        for line in lines:
            if "### Rater Performance" in line:
                in_performance = True
                continue
            elif in_performance and line.startswith("##"):
                break

            if in_performance and line.startswith("- **"):
                try:
                    # Parse line like "- **model+prompt:** 85.3% success, 12.5 elements"
                    content = line[4:-2]  # Remove "- **" and "**"
                    if ":" in content:
                        rater, stats = content.split(":", 1)
                        rater = rater.strip()
                        stats = stats.strip()

                        performance[rater] = {}
                        if "%" in stats:
                            pct = stats.split("%")[0].split()[-1]
                            if pct.replace(".", "").isdigit():
                                performance[rater]["success_rate"] = f"{pct}%"

                        # Extract avg elements
                        if "elements" in stats:
                            try:
                                elements_part = stats.split("elements")[0].split()[-1]
                                if elements_part.replace(".", "").isdigit():
                                    performance[rater]["avg_elements"] = float(
                                        elements_part
                                    )
                            except:
                                pass
                except:
                    pass

        return performance

    def _extract_biological_insights(self, biological_content: str) -> Dict[str, Any]:
        """Extract biological insights from biological report."""
        insights = {
            "total_trials": "N/A",
            "top_conditions": [],
            "intervention_types": [],
            "element_distribution": {},
        }

        if not biological_content:
            return insights

        lines = biological_content.split("\n")
        for line in lines:
            # Extract total trials
            if "total trials:" in line.lower():
                try:
                    num = line.split(":")[-1].strip()
                    if num.isdigit():
                        insights["total_trials"] = int(num)
                except:
                    pass

            # Extract top conditions
            elif "malignant neoplasm of breast" in line.lower():
                insights["top_conditions"].append("Breast Cancer")

            # Extract intervention types
            elif "chemotherapy" in line.lower() and ":" in line:
                insights["intervention_types"].append("Chemotherapy")

            # Extract element distribution
            elif line.startswith("- **") and ":**" in line:
                try:
                    element_type = line.split(":**")[0][4:]  # Remove "- **"
                    count_part = line.split(":**")[1].strip()
                    if count_part.isdigit():
                        insights["element_distribution"][element_type] = int(count_part)
                except:
                    pass

        return insights

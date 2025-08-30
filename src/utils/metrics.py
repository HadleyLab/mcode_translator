from collections import defaultdict
from typing import Dict, List, Any
from src.utils.logging_config import get_logger

class MatchingMetrics:
    """
    Tracks metrics for breast cancer patient-trial matching
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_patients = 0
        self.total_trials = 0
        self.total_matches = 0
        self.match_reasons = defaultdict(int)
        self.gene_match_counts = defaultdict(int)
        self.biomarker_match_counts = defaultdict(int)
        self.stage_match_counts = defaultdict(int)
        self.treatment_match_counts = defaultdict(int)
    
    def record_match(self, match_reasons: List[str], genomic_variants: List[Dict]):
        """Record metrics for a successful match"""
        self.total_matches += 1
        
        for reason in match_reasons:
            self.match_reasons[reason] += 1
            
            # Track specific match types
            if reason.startswith("Biomarker match:"):
                bio_name = reason.split(':')[1].split('(')[0].strip()
                self.biomarker_match_counts[bio_name] += 1
            elif reason.startswith("Variant match:"):
                gene = reason.split(':')[1].split()[0].strip()
                self.gene_match_counts[gene] += 1
            elif reason.startswith("Stage match:") or reason.startswith("Stage compatible:"):
                stage = reason.split(':')[1].split()[0].strip()
                self.stage_match_counts[stage] += 1
            elif reason.startswith("Shared treatments:"):
                treatments = reason.split(':')[1].strip()
                for treatment in treatments.split(','):
                    self.treatment_match_counts[treatment.strip()] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Return summary of metrics"""
        return {
            "total_patients": self.total_patients,
            "total_trials": self.total_trials,
            "total_matches": self.total_matches,
            "match_rate": self.total_matches / max(1, self.total_patients * self.total_trials),
            "top_match_reasons": sorted(self.match_reasons.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_genes": sorted(self.gene_match_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_biomarkers": sorted(self.biomarker_match_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_stages": sorted(self.stage_match_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_treatments": sorted(self.treatment_match_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def log_summary(self):
        """Log metrics summary"""
        summary = self.get_summary()
        self.logger.info("Matching Metrics Summary:")
        self.logger.info(f"Patients processed: {summary['total_patients']}")
        self.logger.info(f"Trials processed: {summary['total_trials']}")
        self.logger.info(f"Total matches: {summary['total_matches']}")
        self.logger.info(f"Match rate: {summary['match_rate']:.2%}")
        
        self.logger.info("Top match reasons:")
        for reason, count in summary['top_match_reasons']:
            self.logger.info(f"  - {reason}: {count}")
            
        self.logger.info("Top matching genes:")
        for gene, count in summary['top_genes']:
            self.logger.info(f"  - {gene}: {count}")
            
        self.logger.info("Top matching biomarkers:")
        for biomarker, count in summary['top_biomarkers']:
            self.logger.info(f"  - {biomarker}: {count}")
            
        self.logger.info("Top matching stages:")
        for stage, count in summary['top_stages']:
            self.logger.info(f"  - {stage}: {count}")
            
        self.logger.info("Top matching treatments:")
        for treatment, count in summary['top_treatments']:
            self.logger.info(f"  - {treatment}: {count}")
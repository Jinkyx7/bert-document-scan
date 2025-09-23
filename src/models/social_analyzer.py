"""
Social aspect analyzer using ESGBERT/SocRoBERTa-social model.

This module implements social responsibility analysis for annual reports,
identifying sentences that discuss social governance, employee welfare,
community impact, diversity, and other social ESG factors.
"""

from .base_analyzer import BaseAnalyzer


class SocialAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying social responsibility content in annual reports.
    
    Uses the ESGBERT/SocRoBERTa-social model, which is specifically trained
    to classify text as related to social governance factors including:
    - Employee relations and welfare
    - Community engagement and impact  
    - Diversity and inclusion initiatives
    - Human rights and labor practices
    - Social sustainability programs
    """
    
    def __init__(self, threshold: float = 0.7, batch_size: int = 32):
        """
        Initialize the social responsibility analyzer.
        
        Args:
            threshold: Classification threshold for social content (0.0-1.0)
                      Higher values = more conservative, fewer false positives
                      Lower values = more liberal, fewer false negatives
            batch_size: Number of sentences to process simultaneously
                       Larger batches = faster processing but more memory usage
        """
        # Initialize with the specialized social ESG model
        super().__init__(
            model_name="ESGBERT/SocRoBERTa-social",
            threshold=threshold,
            batch_size=batch_size
        )
    
    def _find_target_class_index(self) -> int:
        """
        Find the index of the 'social' class in the model's output.
        
        The ESGBERT/SocRoBERTa-social model may use different label names
        depending on its configuration. This method searches for common
        variations of "social" labels.
        
        Returns:
            Integer index of the social class (typically 0 or 1 for binary models)
        """
        # Get the model's label configuration
        id2label = getattr(self.model.config, "id2label", {})
        
        # Search for various ways "social" might be labeled in the model
        social_synonyms = ["social", "label_1", "esg_social", "soc"]
        
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if any(synonym in label_lower for synonym in social_synonyms):
                return int(idx)
        
        # Fallback: assume binary classification where class 1 = social
        return 1
    
    def _get_output_column_name(self) -> str:
        """Get the name for the score column in output CSV files."""
        return "social_score"
    
    def _get_output_prefix(self) -> str:
        """Get the prefix for output file names and internal columns."""
        return "social"
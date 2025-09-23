"""
Social aspect analyzer using ESGBERT/SocRoBERTa-social model.
"""

from .base_analyzer import BaseAnalyzer


class SocialAnalyzer(BaseAnalyzer):
    """Analyzer for social aspects in annual reports."""
    
    def __init__(self, threshold: float = 0.7, batch_size: int = 32):
        """
        Initialize the social analyzer.
        
        Args:
            threshold: Classification threshold for social content
            batch_size: Batch size for inference
        """
        super().__init__(
            model_name="ESGBERT/SocRoBERTa-social",
            threshold=threshold,
            batch_size=batch_size
        )
    
    def _find_target_class_index(self) -> int:
        """Find the index of the 'social' class."""
        id2label = getattr(self.model.config, "id2label", {})
        
        # Look for social-related labels
        social_synonyms = ["social", "label_1", "esg_social", "soc"]
        
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if any(synonym in label_lower for synonym in social_synonyms):
                return int(idx)
        
        # Fallback to class 1 for binary classification
        return 1
    
    def _get_output_column_name(self) -> str:
        """Get the name for the score column."""
        return "social_score"
    
    def _get_output_prefix(self) -> str:
        """Get the prefix for output files."""
        return "social"
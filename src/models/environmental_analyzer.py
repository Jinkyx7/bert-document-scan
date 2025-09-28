"""
Environmental aspect analyzer using ESGBERT/EnvRoBERTa-environmental model.

This module implements environmental impact analysis for annual reports,
identifying sentences that discuss climate change, sustainability initiatives,
environmental compliance, and other environmental ESG factors.
"""

from .base_analyzer import BaseAnalyzer


class EnvironmentalAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying environmental sustainability content in annual reports.
    
    Uses the ESGBERT/EnvRoBERTa-environmental model, which is specifically trained
    to classify text as related to environmental governance factors including:
    - Climate change mitigation and adaptation
    - Carbon emissions and greenhouse gas management
    - Renewable energy and clean technology initiatives
    - Waste management and circular economy practices
    - Water conservation and pollution prevention
    - Biodiversity and ecosystem protection
    - Environmental compliance and regulations
    """
    
    def __init__(self, threshold: float = 0.7, batch_size: int = 32):
        """
        Initialize the environmental sustainability analyzer.
        
        Args:
            threshold: Classification threshold for environmental content (0.0-1.0)
                      Higher values = more conservative, fewer false positives
                      Lower values = more liberal, fewer false negatives
            batch_size: Number of sentences to process simultaneously
                       Larger batches = faster processing but more memory usage
        """
        # Initialize with the specialized environmental ESG model
        super().__init__(
            model_name="ESGBERT/EnvRoBERTa-environmental",
            threshold=threshold,
            batch_size=batch_size
        )
    
    def _find_target_class_index(self) -> int:
        """
        Find the index of the 'environmental' class in the model's output.
        
        The ESGBERT/EnvRoBERTa-environmental model may use different label names
        depending on its configuration. This method searches for common
        variations of "environmental" labels.
        
        Returns:
            Integer index of the environmental class (typically 0 or 1 for binary models)
        """
        # Get the model's label configuration
        id2label = getattr(self.model.config, "id2label", {})
        
        # Search for various ways "environmental" might be labeled in the model
        env_synonyms = ["environment", "environmental", "env", "label_1", "esg_environmental"]
        
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if any(synonym in label_lower for synonym in env_synonyms):
                return int(idx)
        
        # Fallback: assume binary classification where class 1 = environmental
        return 1
    
    def _get_output_column_name(self) -> str:
        """Get the name for the score column in output CSV files."""
        return "environment_score"
    
    def _get_output_prefix(self) -> str:
        """Get the prefix for output file names and internal columns."""
        return "environment"
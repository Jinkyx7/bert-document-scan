"""
Base class for BERT-based text analyzers.
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BaseAnalyzer(ABC):
    """Base class for BERT text analysis models."""
    
    def __init__(self, model_name: str, threshold: float = 0.7, batch_size: int = 32):
        """
        Initialize the analyzer.
        
        Args:
            model_name: HuggingFace model name
            threshold: Classification threshold
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Find target class index
        self.target_idx = self._find_target_class_index()
    
    @abstractmethod
    def _find_target_class_index(self) -> int:
        """Find the index of the target class in the model output."""
        pass
    
    @abstractmethod
    def _get_output_column_name(self) -> str:
        """Get the name for the score column in output."""
        pass
    
    @abstractmethod
    def _get_output_prefix(self) -> str:
        """Get the prefix for output file names."""
        pass
    
    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score a list of sentences using the model.
        
        Args:
            sentences: List of sentences to score
            
        Returns:
            List of probability scores for the target class
        """
        probabilities = []
        
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, self.target_idx]
            probabilities.extend(probs.detach().cpu().tolist())
            
            # Clean up memory
            del encoded, logits, probs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return probabilities
    
    def analyze_report(self, sentences_data: List[Dict], report_name: str, output_dir: str) -> Dict[str, Any]:
        """
        Analyze sentences from a report and save results.
        
        Args:
            sentences_data: List of dicts with 'page' and 'sentence' keys
            report_name: Name of the report
            output_dir: Directory to save results
            
        Returns:
            Summary statistics
        """
        if not sentences_data:
            return {
                "report": report_name,
                "total_sentences": 0,
                f"{self._get_output_prefix()}_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }
        
        # Create DataFrame
        df = pd.DataFrame(sentences_data)
        
        # Score sentences
        scores = self.score_sentences(df["sentence"].tolist())
        score_column = self._get_output_column_name()
        df[score_column] = scores
        df[f"is_{self._get_output_prefix()}"] = df[score_column] >= self.threshold
        
        # Save all results
        os.makedirs(output_dir, exist_ok=True)
        all_csv = os.path.join(output_dir, f"{self._get_output_prefix()}_scores_{report_name}.csv")
        df[["page", score_column, "sentence"]].to_csv(all_csv, index=False)
        
        # Save high-confidence hits
        hits = (df[df[f"is_{self._get_output_prefix()}"]]
                .sort_values(score_column, ascending=False))
        hits_csv = os.path.join(output_dir, f"{self._get_output_prefix()}_hits_{report_name}_min{str(self.threshold).replace('.', '_')}.csv")
        hits[["page", score_column, "sentence"]].to_csv(hits_csv, index=False)
        
        # Return summary
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            f"{self._get_output_prefix()}_candidates": int(df[f"is_{self._get_output_prefix()}"].sum()),
            "all_csv": all_csv,
            "hits_csv": hits_csv
        }
"""
Base class for BERT-based text analyzers.

This module provides a common foundation for ESG (Environmental, Social, Governance)
text classification using BERT models. It handles model loading, batched inference,
and standardized output formatting.
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BaseAnalyzer(ABC):
    """
    Abstract base class for BERT-based text analysis models.
    
    This class provides common functionality for loading BERT models, processing
    text in batches, and generating standardized outputs. Subclasses implement
    specific logic for different types of analysis (social, environmental, etc.).
    """
    
    def __init__(self, model_name: str, threshold: float = 0.7, batch_size: int = 32):
        """
        Initialize the BERT analyzer with model loading and configuration.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "ESGBERT/SocRoBERTa-social")
            threshold: Classification threshold for positive predictions (0.0-1.0)
            batch_size: Number of sentences to process simultaneously
            
        Note:
            Models are automatically moved to GPU if available, otherwise CPU.
            The target class index is determined by subclass implementation.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        
        # Automatically detect and use GPU if available for faster inference
        # Priority: CUDA (NVIDIA) > MPS (Apple Silicon M1/M2/M3/M4) > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load pre-trained tokenizer and model from HuggingFace
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to appropriate device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()  # Disable dropout, batch norm updates, etc.
        
        # Determine which output class represents the target category
        # (implementation varies by model - see subclasses)
        self.target_idx = self._find_target_class_index()
    
    @abstractmethod
    def _find_target_class_index(self) -> int:
        """
        Find the index of the target class in the model's output.
        
        Different BERT models may use different label mappings (e.g., "social" vs "label_1").
        Subclasses implement model-specific logic to identify the correct class index.
        
        Returns:
            Integer index of the target class in the model's output tensor
        """
        pass
    
    @abstractmethod
    def _get_output_column_name(self) -> str:
        """
        Get the name for the score column in CSV output.
        
        Returns:
            Column name for scores (e.g., "social_score", "environment_score")
        """
        pass
    
    @abstractmethod
    def _get_output_prefix(self) -> str:
        """
        Get the prefix for output file names and columns.
        
        Returns:
            Prefix string used in filenames and column names (e.g., "social", "environment")
        """
        pass
    
    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score a list of sentences using the loaded BERT model.
        
        This method processes sentences in batches for memory efficiency and speed.
        Each sentence gets a probability score for the target class (e.g., "social").
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of probability scores (0.0-1.0) for the target class,
            one score per input sentence in the same order
            
        Note:
            Uses @torch.no_grad() decorator to disable gradient computation
            for faster inference and reduced memory usage.
        """
        probabilities = []
        
        # Process sentences in batches to manage memory usage
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]
            
            # Tokenize the batch of sentences
            # - padding=True: Pad shorter sentences to match longest in batch
            # - truncation=True: Cut off sentences longer than max_length
            # - max_length=256: BERT's typical sequence length limit
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"  # Return PyTorch tensors
            ).to(self.device)
            
            # Forward pass through BERT model
            logits = self.model(**encoded).logits  # Raw model outputs
            
            # Convert logits to probabilities using softmax
            # Extract only the target class probabilities
            probs = torch.softmax(logits, dim=-1)[:, self.target_idx]
            
            # Move results back to CPU and convert to Python list
            probabilities.extend(probs.detach().cpu().tolist())
            
            # Clean up GPU memory to prevent out-of-memory errors
            del encoded, logits, probs
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
        
        return probabilities

    def _preview_results(self, hits_df: pd.DataFrame, score_column: str, report_name: str, preview_count: int = 70):
        """
        Preview high-confidence results before saving to CSV.

        Args:
            hits_df: DataFrame containing high-confidence matches, sorted by score
            score_column: Name of the score column
            report_name: Name of the report being processed
            preview_count: Number of results to preview (default: 70)
        """
        if hits_df.empty:
            print(f"\n📋 Preview for {report_name}: No high-confidence matches found above threshold")
            return

        print(f"\n📋 Preview: Top {min(preview_count, len(hits_df))} {self._get_output_prefix()} results for {report_name}")
        print("=" * 80)

        # # Show top results up to preview_count
        # preview_df = hits_df.head(preview_count)

        # for idx, (_, row) in enumerate(preview_df.iterrows(), 1):
        #     score = row[score_column]
        #     page = row['page']
        #     sentence = row['sentence']

        #     # Truncate very long sentences for display
        #     display_sentence = sentence[:120] + "..." if len(sentence) > 120 else sentence

        #     print(f"{idx:2d}. [Page {page:2d}] Score: {score:.3f} | {display_sentence}")

        #     # Add a separator every 10 items for readability
        #     if idx % 10 == 0 and idx < len(preview_df):
        #         print("-" * 80)

        # Show summary if there are more results
        total_hits = len(hits_df)
        if total_hits > preview_count:
            print(f"\n... and {total_hits - preview_count} more results (showing top {preview_count})")

        print("=" * 80)

    def analyze_report(self, sentences_data: List[Dict], report_name: str, output_dir: str) -> Dict[str, Any]:
        """
        Analyze sentences from a report and save results to CSV files.
        
        This method orchestrates the complete analysis pipeline:
        1. Score all sentences using the BERT model
        2. Apply threshold to identify high-confidence matches
        3. Save comprehensive results and filtered hits to CSV files
        4. Return summary statistics
        
        Args:
            sentences_data: List of dictionaries with 'page' and 'sentence' keys
            report_name: Clean report identifier for output filenames
            output_dir: Directory where CSV files will be saved
            
        Returns:
            Dictionary containing:
            - "report": Report name
            - "total_sentences": Total number of sentences processed
            - "{prefix}_candidates": Number of high-confidence matches
            - "all_csv": Path to file with all sentences and scores
            - "hits_csv": Path to file with only high-confidence matches
        """
        # Handle edge case where no sentences were extracted
        if not sentences_data:
            return {
                "report": report_name,
                "total_sentences": 0,
                f"{self._get_output_prefix()}_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }
        
        # Convert sentence data to pandas DataFrame for easier manipulation
        df = pd.DataFrame(sentences_data)
        
        # Run BERT inference on all sentences
        scores = self.score_sentences(df["sentence"].tolist())
        
        # Add scores and threshold-based classification to DataFrame
        score_column = self._get_output_column_name()
        df[score_column] = scores
        df[f"is_{self._get_output_prefix()}"] = df[score_column] >= self.threshold
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare filtered results: high-confidence matches, sorted by score
        hits = (df[df[f"is_{self._get_output_prefix()}"]]
                .sort_values(score_column, ascending=False))

        # Preview results before saving to CSV
        self._preview_results(hits, score_column, report_name)

        # Save complete results: all sentences with their scores
        # Using finbert naming convention: {model}_results_all_{report}.csv
        all_csv = os.path.join(output_dir, f"{self._get_output_prefix()}_results_all_{report_name}.csv")
        df[["page", score_column, "sentence"]].to_csv(all_csv, index=False)

        # Using finbert naming convention: {model}_results_hits_{report}.csv (no threshold in filename)
        hits_csv = os.path.join(output_dir, f"{self._get_output_prefix()}_results_hits_{report_name}.csv")
        hits[["page", score_column, "sentence"]].to_csv(hits_csv, index=False)
        
        # Return summary statistics for tracking progress
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            f"{self._get_output_prefix()}_candidates": int(df[f"is_{self._get_output_prefix()}"].sum()),
            "all_csv": all_csv,
            "hits_csv": hits_csv
        }
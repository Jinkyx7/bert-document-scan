"""
Environmental aspect analyzer using ESGBERT/EnvRoBERTa-environmental model.

This module implements environmental impact analysis for annual reports,
identifying sentences that discuss climate change, sustainability initiatives,
environmental compliance, and other environmental ESG factors.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline


class EnvironmentalAnalyzer:
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
        self.threshold = threshold
        self.batch_size = batch_size

        # Initialize ESGBERT/EnvRoBERTa-environmental pipeline
        print("Loading ESGBERT/EnvRoBERTa-environmental model...")
        self.classifier = pipeline(
            task="text-classification",
            model="ESGBERT/EnvRoBERTa-environmental",
            tokenizer="ESGBERT/EnvRoBERTa-environmental",
            return_all_scores=True,  # Return scores for all classes
            truncation=True          # Handle long sentences automatically
        )

    def score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score a list of sentences for environmental content using ESGBERT.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of probability scores (0.0-1.0) for environmental content,
            one score per input sentence in the same order
        """
        environmental_scores = []

        # Process sentences in batches for efficiency
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]

            # Run ESGBERT inference on the batch
            outputs = self.classifier(batch)

            # Process each sentence's results
            for output in outputs:
                # Find the environmental score from the pipeline output
                environmental_score = 0.0
                for result in output:
                    # Look for environmental-related labels (model might use different names)
                    label = result["label"].lower()
                    if ("environment" in label or "environmental" in label or
                        "label_1" in label or "env" in label or "esg" in label):
                        environmental_score = result["score"]
                        break

                environmental_scores.append(environmental_score)

        return environmental_scores

    def analyze_report(self, sentences_data: List[Dict], report_name: str, output_dir: str) -> Dict[str, Any]:
        """
        Analyze sentences from a report and save results to CSV files.

        Args:
            sentences_data: List of dictionaries with 'page' and 'sentence' keys
            report_name: Clean report identifier for output filenames
            output_dir: Directory where CSV files will be saved

        Returns:
            Dictionary containing summary statistics and file paths
        """
        # Handle edge case where no sentences were extracted
        if not sentences_data:
            return {
                "report": report_name,
                "total_sentences": 0,
                "environmental_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }

        # Convert sentence data to pandas DataFrame
        df = pd.DataFrame(sentences_data)

        # Run ESGBERT inference on all sentences
        scores = self.score_sentences(df["sentence"].tolist())

        # Add scores and threshold-based classification to DataFrame
        df["environment_score"] = scores
        df["is_environmental"] = df["environment_score"] >= self.threshold

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save complete results: all sentences with their scores
        all_csv = os.path.join(output_dir, f"environmental_results_all_{report_name}.csv")
        df[["page", "environment_score", "sentence"]].to_csv(all_csv, index=False)

        # Save filtered results: only high-confidence matches, sorted by score
        hits = (df[df["is_environmental"]]
                .sort_values("environment_score", ascending=False))

        hits_csv = os.path.join(output_dir, f"environmental_results_hits_{report_name}.csv")
        hits[["page", "environment_score", "sentence"]].to_csv(hits_csv, index=False)

        # Return summary statistics
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            "environmental_candidates": int(df["is_environmental"].sum()),
            "all_csv": all_csv,
            "hits_csv": hits_csv
        }
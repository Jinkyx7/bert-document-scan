"""
Social aspect analyzer using ESGBERT/SocRoBERTa-social model.

This module implements social responsibility analysis for annual reports,
identifying sentences that discuss social governance, employee welfare,
community impact, diversity, and other social ESG factors.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline


class SocialAnalyzer:
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
        self.threshold = threshold
        self.batch_size = batch_size

        # Initialize ESGBERT/SocRoBERTa-social pipeline
        print("Loading ESGBERT/SocRoBERTa-social model...")
        self.classifier = pipeline(
            task="text-classification",
            model="ESGBERT/SocRoBERTa-social",
            tokenizer="ESGBERT/SocRoBERTa-social",
            return_all_scores=True,  # Return scores for all classes
            truncation=True          # Handle long sentences automatically
        )

    def score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score a list of sentences for social content using ESGBERT.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of probability scores (0.0-1.0) for social content,
            one score per input sentence in the same order
        """
        social_scores = []

        # Process sentences in batches for efficiency
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]

            # Run ESGBERT inference on the batch
            outputs = self.classifier(batch)

            # Process each sentence's results
            for output in outputs:
                # Find the social score from the pipeline output
                social_score = 0.0
                for result in output:
                    # Look for social-related labels (model might use different names)
                    label = result["label"].lower()
                    if "social" in label or "label_1" in label or "esg" in label:
                        social_score = result["score"]
                        break

                social_scores.append(social_score)

        return social_scores

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
                "social_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }

        # Convert sentence data to pandas DataFrame
        df = pd.DataFrame(sentences_data)

        # Run ESGBERT inference on all sentences
        scores = self.score_sentences(df["sentence"].tolist())

        # Add scores and threshold-based classification to DataFrame
        df["social_score"] = scores
        df["is_social"] = df["social_score"] >= self.threshold

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save complete results: all sentences with their scores
        all_csv = os.path.join(output_dir, f"social_results_all_{report_name}.csv")
        df[["page", "social_score", "sentence"]].to_csv(all_csv, index=False)

        # Save filtered results: only high-confidence matches, sorted by score
        hits = (df[df["is_social"]]
                .sort_values("social_score", ascending=False))

        hits_csv = os.path.join(output_dir, f"social_results_hits_{report_name}.csv")
        hits[["page", "social_score", "sentence"]].to_csv(hits_csv, index=False)

        # Return summary statistics
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            "social_candidates": int(df["is_social"].sum()),
            "all_csv": all_csv,
            "hits_csv": hits_csv
        }
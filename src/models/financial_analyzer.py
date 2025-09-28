"""
Financial sentiment analyzer using ProsusAI/finbert model.

This module implements financial sentiment analysis for annual reports,
identifying positive, negative, and neutral financial sentiment in text.
Unlike the ESG analyzers, this uses a different architecture (pipeline)
and focuses on sentiment rather than topic classification.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline


class FinancialAnalyzer:
    """
    Analyzer for financial sentiment in annual reports.
    
    Uses the ProsusAI/finbert model, which is specifically trained to classify
    financial text sentiment into three categories:
    - Positive: Optimistic outlook, good performance, growth opportunities
    - Negative: Concerns, risks, poor performance, challenges
    - Neutral: Factual statements without clear sentiment direction
    
    Note: This analyzer uses a different architecture than the ESG analyzers,
    using transformers.pipeline instead of direct model inference.
    """
    
    def __init__(self, threshold: float = 0.6, batch_size: int = 32, top_n: int = 50):
        """
        Initialize the financial sentiment analyzer.
        
        Args:
            threshold: Classification threshold for positive sentiment (0.0-1.0)
                      Higher values = more conservative, only very positive text
                      Lower values = more liberal, includes mildly positive text
            batch_size: Number of sentences to process simultaneously
                       Larger batches = faster processing but more memory usage
            top_n: Maximum number of top positive sentences to save in hits file
                  Controls the size of the filtered results
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.top_n = top_n
        
        # Initialize FinBERT pipeline for sentiment classification
        # Uses transformers.pipeline for simplified inference
        print("Loading FinBERT model for financial sentiment analysis...")
        self.classifier = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,  # Return scores for all three classes
            truncation=True          # Handle long sentences automatically
        )
    
    def _softmax_scores(self, label_scores: List[Dict]) -> Dict[str, float]:
        """
        Convert pipeline output to normalized sentiment scores.
        
        The transformers pipeline returns a list of dictionaries with labels
        and scores. This method normalizes them into a consistent format.
        
        Args:
            label_scores: List of {"label": str, "score": float} from pipeline
            
        Returns:
            Dictionary with keys "positive", "neutral", "negative" and float scores
            
        Note:
            Pipeline scores are already normalized (sum to 1.0), so we just
            reorganize the data structure for easier processing.
        """
        scores = {}
        
        # Extract scores from pipeline output format
        for item in label_scores:
            scores[item["label"].lower()] = float(item["score"])
        
        # Ensure all three sentiment categories are present with default values
        # This prevents KeyError exceptions if model output format changes
        for sentiment in ("positive", "neutral", "negative"):
            scores.setdefault(sentiment, 0.0)
        
        return scores
    
    def score_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Score a list of sentences for financial sentiment using FinBERT.
        
        This method processes sentences in batches and returns comprehensive
        sentiment analysis including all three sentiment scores and the
        dominant sentiment classification.
        
        Args:
            sentences: List of sentences to analyze for financial sentiment
            
        Returns:
            List of dictionaries, one per sentence, containing:
            - "positive": Probability score for positive sentiment (0.0-1.0)
            - "neutral": Probability score for neutral sentiment (0.0-1.0)  
            - "negative": Probability score for negative sentiment (0.0-1.0)
            - "label": Dominant sentiment ("positive", "neutral", or "negative")
            - "score": Probability score of the dominant sentiment
        """
        results = []
        
        # Process sentences in batches for efficiency
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]
            
            # Run FinBERT inference on the batch
            outputs = self.classifier(batch)
            
            # Process each sentence's results
            for output in outputs:
                # Convert pipeline output to standardized format
                scores = self._softmax_scores(output)
                
                # Determine which sentiment has the highest confidence
                # This follows the same logic as the original notebook
                if (scores["positive"] >= scores["neutral"] and 
                    scores["positive"] >= scores["negative"]):
                    label = "positive"
                    max_score = scores["positive"]
                elif scores["negative"] >= scores["neutral"]:
                    label = "negative"
                    max_score = scores["negative"]
                else:
                    label = "neutral"
                    max_score = scores["neutral"]
                
                # Store comprehensive results for this sentence
                results.append({
                    "positive": scores["positive"],
                    "neutral": scores["neutral"],
                    "negative": scores["negative"],
                    "label": label,
                    "score": max_score
                })
        
        return results

    def _preview_results(self, hits_df: pd.DataFrame, report_name: str, preview_count: int = 70):
        """
        Preview high-confidence financial sentiment results before saving to CSV.

        Args:
            hits_df: DataFrame containing high-positive sentiment matches, sorted by score
            report_name: Name of the report being processed
            preview_count: Number of results to preview (default: 70)
        """
        if hits_df.empty:
            print(f"\n📋 Preview for {report_name}: No positive financial sentiment above threshold")
            return

        print(f"\n📋 Preview: Top {min(preview_count, len(hits_df))} positive financial sentiment results for {report_name}")
        print("=" * 80)

        # Show top results up to preview_count
        preview_df = hits_df.head(preview_count)

        for idx, (_, row) in enumerate(preview_df.iterrows(), 1):
            positive_score = row['positive']
            page = row['page']
            sentence = row['sentence']

            # Truncate very long sentences for display
            display_sentence = sentence[:120] + "..." if len(sentence) > 120 else sentence

            print(f"{idx:2d}. [Page {page:2d}] Positive: {positive_score:.3f} | {display_sentence}")

            # Add a separator every 10 items for readability
            if idx % 10 == 0 and idx < len(preview_df):
                print("-" * 80)

        # Show summary if there are more results
        total_hits = len(hits_df)
        if total_hits > preview_count:
            print(f"\n... and {total_hits - preview_count} more results (showing top {preview_count})")

        print("=" * 80)

    def analyze_report(self, sentences_data: List[Dict], report_name: str, output_dir: str) -> Dict[str, Any]:
        """
        Analyze sentences from a report for financial sentiment and save results.
        
        This method orchestrates the complete financial sentiment analysis:
        1. Score all sentences for positive/neutral/negative sentiment
        2. Apply threshold to identify strongly positive sentences
        3. Save comprehensive results and top positive hits to CSV files
        4. Return summary statistics
        
        Args:
            sentences_data: List of dictionaries with 'page' and 'sentence' keys
            report_name: Clean report identifier for output filenames
            output_dir: Directory where CSV files will be saved
            
        Returns:
            Dictionary containing:
            - "report": Report name
            - "total_sentences": Total number of sentences processed
            - "positive_candidates": Number of sentences above positive threshold
            - "threshold_positive_min": The threshold value used
            - "all_csv": Path to file with all sentences and sentiment scores
            - "hits_csv": Path to file with top positive sentences only
        """
        # Handle edge case where no sentences were extracted
        if not sentences_data:
            return {
                "report": report_name,
                "total_sentences": 0,
                "positive_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }
        
        # Convert sentence data to pandas DataFrame for easier manipulation
        df = pd.DataFrame(sentences_data)
        
        # Run FinBERT sentiment analysis on all sentences
        scores = self.score_sentences(df["sentence"].tolist())
        
        # Add sentiment scores and classifications to DataFrame
        # This dynamically adds columns for all score types
        for i, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                if key not in df.columns:
                    df[key] = None  # Initialize column if it doesn't exist
                df.at[i, key] = value
        
        # Create positive sentiment flag based on threshold
        df["is_positive"] = df["positive"] >= self.threshold
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare filtered results: top N most positive sentences
        # Sort by positive score first, then by overall confidence score
        top_positive = (df[df["is_positive"]]
                       .sort_values(["positive", "score"], ascending=False)
                       .head(self.top_n))  # Limit to top N results

        # Preview results before saving to CSV
        self._preview_results(top_positive, report_name)

        # Save complete results: all sentences with comprehensive sentiment data
        # Using consistent naming convention: financial_results_all_{report}.csv
        all_csv = os.path.join(output_dir, f"financial_results_all_{report_name}.csv")
        df[["page", "label", "score", "positive", "neutral", "negative", "sentence"]].to_csv(
            all_csv, index=False
        )

        # Using consistent naming convention: financial_results_hits_{report}.csv
        hits_csv = os.path.join(output_dir, f"financial_results_hits_{report_name}.csv")
        top_positive[["page", "positive", "sentence"]].to_csv(hits_csv, index=False)
        
        # Return summary statistics for tracking progress
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            "positive_candidates": int(df["is_positive"].sum()),
            "threshold_positive_min": float(self.threshold),
            "all_csv": all_csv,
            "hits_csv": hits_csv
        }
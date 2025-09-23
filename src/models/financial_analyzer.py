"""
Financial sentiment analyzer using ProsusAI/finbert model.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline


class FinancialAnalyzer:
    """Analyzer for financial sentiment in annual reports."""
    
    def __init__(self, threshold: float = 0.6, batch_size: int = 32, top_n: int = 50):
        """
        Initialize the financial analyzer.
        
        Args:
            threshold: Classification threshold for positive sentiment
            batch_size: Batch size for inference
            top_n: Number of top positive sentences to save
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.top_n = top_n
        
        # Initialize FinBERT pipeline
        self.classifier = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True
        )
    
    def _softmax_scores(self, label_scores: List[Dict]) -> Dict[str, float]:
        """
        Convert pipeline output to normalized scores.
        
        Args:
            label_scores: List of label-score dictionaries from pipeline
            
        Returns:
            Dictionary with positive, neutral, negative scores
        """
        scores = {}
        for item in label_scores:
            scores[item["label"].lower()] = float(item["score"])
        
        # Ensure all three sentiments are present
        for sentiment in ("positive", "neutral", "negative"):
            scores.setdefault(sentiment, 0.0)
        
        return scores
    
    def score_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Score a list of sentences for financial sentiment.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of dictionaries with sentiment scores and labels
        """
        results = []
        
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]
            outputs = self.classifier(batch)
            
            for output in outputs:
                scores = self._softmax_scores(output)
                
                # Determine dominant sentiment
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
                
                results.append({
                    "positive": scores["positive"],
                    "neutral": scores["neutral"],
                    "negative": scores["negative"],
                    "label": label,
                    "score": max_score
                })
        
        return results
    
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
                "positive_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }
        
        # Create DataFrame
        df = pd.DataFrame(sentences_data)
        
        # Score sentences
        scores = self.score_sentences(df["sentence"].tolist())
        
        # Add scores to DataFrame
        for i, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                if key not in df.columns:
                    df[key] = None
                df.at[i, key] = value
        
        # Create positive flag
        df["is_positive"] = df["positive"] >= self.threshold
        
        # Save all results
        os.makedirs(output_dir, exist_ok=True)
        all_csv = os.path.join(output_dir, f"finbert_results_all_{report_name}.csv")
        df[["page", "label", "score", "positive", "neutral", "negative", "sentence"]].to_csv(
            all_csv, index=False
        )
        
        # Save top positive hits
        top_positive = (df[df["is_positive"]]
                       .sort_values(["positive", "score"], ascending=False)
                       .head(self.top_n))
        hits_csv = os.path.join(output_dir, f"finbert_results_hits_{report_name}.csv")
        top_positive[["page", "positive", "sentence"]].to_csv(hits_csv, index=False)
        
        # Return summary
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            "positive_candidates": int(df["is_positive"].sum()),
            "threshold_positive_min": float(self.threshold),
            "all_csv": all_csv,
            "hits_csv": hits_csv
        }
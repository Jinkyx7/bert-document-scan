#!/usr/bin/env python3
"""
Main entry point for BERT-based annual report analysis.

This script processes PDF files in a specified directory using three different BERT models:
- Social analysis: ESGBERT/SocRoBERTa-social
- Environmental analysis: ESGBERT/EnvRoBERTa-environmental  
- Financial sentiment: ProsusAI/finbert

Usage:
    python main.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--model MODEL]
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_pdf_files, extract_sentences_with_pages, safe_report_name
from src.models import SocialAnalyzer, EnvironmentalAnalyzer, FinancialAnalyzer


def process_reports_with_model(pdf_files, analyzer, model_name, output_dir):
    """Process all PDF files with a specific analyzer."""
    print(f"\n{'='*60}")
    print(f"Processing {len(pdf_files)} reports with {model_name}")
    print(f"{'='*60}")
    
    summaries = []
    
    for pdf_path in pdf_files:
        report_name = safe_report_name(pdf_path)
        print(f"\nProcessing: {pdf_path} (report: {report_name})")
        
        try:
            # Extract sentences from PDF
            sentences_data = extract_sentences_with_pages(pdf_path)
            
            if not sentences_data:
                print(f"No sentences extracted from {pdf_path}. Skipping.")
                continue
            
            # Analyze with the model
            summary = analyzer.analyze_report(sentences_data, report_name, output_dir)
            summaries.append(summary)
            
            print(f"Results:")
            print(f"  - Total sentences: {summary['total_sentences']}")
            if 'positive_candidates' in summary:
                print(f"  - Positive candidates: {summary['positive_candidates']}")
            else:
                key = [k for k in summary.keys() if k.endswith('_candidates')][0]
                print(f"  - {key.replace('_', ' ').title()}: {summary[key]}")
            print(f"  - All results: {summary['all_csv']}")
            print(f"  - Hits: {summary['hits_csv']}")
            
        except Exception as e:
            print(f"ERROR processing {pdf_path}: {e}")
            continue
    
    # Save summary across all reports
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_file = os.path.join(output_dir, f"{model_name.lower().replace('/', '_')}_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary: {summary_file}")
    
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Analyze annual reports with BERT models")
    parser.add_argument(
        "--data-dir", 
        default="./data", 
        help="Directory containing PDF files (default: ./data)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./outputs", 
        help="Directory to save results (default: ./outputs)"
    )
    parser.add_argument(
        "--model", 
        choices=["social", "environmental", "financial", "all"],
        default="all",
        help="Which model to run (default: all)"
    )
    parser.add_argument(
        "--social-threshold",
        type=float,
        default=0.7,
        help="Threshold for social classification (default: 0.7)"
    )
    parser.add_argument(
        "--env-threshold",
        type=float,
        default=0.7,
        help="Threshold for environmental classification (default: 0.7)"
    )
    parser.add_argument(
        "--fin-threshold",
        type=float,
        default=0.6,
        help="Threshold for financial positive sentiment (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        print("Please create the directory and add PDF files to analyze.")
        return 1
    
    # Get PDF files
    pdf_files = get_pdf_files(args.data_dir)
    if not pdf_files:
        print(f"No PDF files found in '{args.data_dir}'.")
        print("Please add PDF files to the directory.")
        return 1
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis based on selected model
    all_summaries = {}
    
    if args.model in ["social", "all"]:
        analyzer = SocialAnalyzer(threshold=args.social_threshold)
        output_subdir = os.path.join(args.output_dir, "social")
        summaries = process_reports_with_model(pdf_files, analyzer, "Social", output_subdir)
        all_summaries["social"] = summaries
    
    if args.model in ["environmental", "all"]:
        analyzer = EnvironmentalAnalyzer(threshold=args.env_threshold)
        output_subdir = os.path.join(args.output_dir, "environmental")
        summaries = process_reports_with_model(pdf_files, analyzer, "Environmental", output_subdir)
        all_summaries["environmental"] = summaries
    
    if args.model in ["financial", "all"]:
        analyzer = FinancialAnalyzer(threshold=args.fin_threshold)
        output_subdir = os.path.join(args.output_dir, "financial")
        summaries = process_reports_with_model(pdf_files, analyzer, "Financial", output_subdir)
        all_summaries["financial"] = summaries
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
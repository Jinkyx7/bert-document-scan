#!/usr/bin/env python3
"""
Main entry point for BERT-based annual report analysis.

This script processes PDF files in a specified directory using three different BERT models:
- Social analysis: ESGBERT/SocRoBERTa-social (identifies social governance content)
- Environmental analysis: ESGBERT/EnvRoBERTa-environmental (identifies environmental content)
- Financial sentiment: ProsusAI/finbert (analyzes financial sentiment: positive/negative/neutral)

The application automatically discovers PDF files in the data directory, extracts sentences
with page numbers, runs BERT analysis, and saves structured CSV results.

Usage:
    python main.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--model MODEL]

Examples:
    python main.py                                    # Analyze all PDFs in ./data with all models
    python main.py --model social                     # Run only social analysis
    python main.py --data-dir /path/to/pdfs           # Use custom data directory
    python main.py --social-threshold 0.8             # Use higher threshold for social analysis
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add src directory to Python path for imports
# This allows importing from src.utils and src.models regardless of where script is run
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_pdf_files, extract_sentences_with_pages, safe_report_name, extract_company_code
from src.models import SocialAnalyzer, EnvironmentalAnalyzer, FinancialAnalyzer, MaoriAnalyzer, MaoriXLMAnalyzer, MaoriMDeBERTaAnalyzer


def process_reports_with_model(pdf_files, analyzer, model_name, output_dir):
    """
    Process all PDF files with a specific BERT analyzer and save results.
    
    This function orchestrates the complete analysis pipeline for a single model:
    1. Extracts sentences from each PDF with page tracking
    2. Runs BERT analysis on all sentences
    3. Saves individual report results and aggregate summary
    4. Provides progress feedback to user
    
    Args:
        pdf_files: List of PDF file paths to process
        analyzer: Initialized analyzer instance (Social/Environmental/Financial)
        model_name: Human-readable model name for display
        output_dir: Directory to save all results
        
    Returns:
        List of summary dictionaries, one per processed report
    """
    print(f"\n{'='*60}")
    print(f"Processing {len(pdf_files)} reports with {model_name}")
    print(f"{'='*60}")
    
    summaries = []
    
    # Process each PDF file individually
    for pdf_path in pdf_files:
        # Generate clean report name for output files
        report_name = safe_report_name(pdf_path)
        print(f"\nProcessing: {pdf_path} (report: {report_name})")
        
        try:
            # Step 1: Extract text content from PDF
            sentences_data = extract_sentences_with_pages(pdf_path)

            # Skip files where no meaningful text was extracted
            if not sentences_data:
                print(f"No sentences extracted from {pdf_path}. Skipping.")
                continue

            # Step 2: Run BERT analysis and save results
            summary = analyzer.analyze_report(sentences_data, report_name, output_dir)
            summaries.append(summary)
            
            # Step 3: Display results summary to user
            print(f"Results:")
            print(f"  - Total sentences: {summary['total_sentences']}")
            
            # Handle different summary formats between models
            if 'positive_candidates' in summary:
                print(f"  - Positive candidates: {summary['positive_candidates']}")
            else:
                # Find the candidates key (social_candidates, environment_candidates, etc.)
                candidates_key = [k for k in summary.keys() if k.endswith('_candidates')][0]
                display_name = candidates_key.replace('_', ' ').title()
                print(f"  - {display_name}: {summary[candidates_key]}")
            
            print(f"  - All results: {summary['all_csv']}")
            print(f"  - Hits: {summary['hits_csv']}")
            
        except Exception as e:
            # Continue processing other files even if one fails
            print(f"ERROR processing {pdf_path}: {e}")
            continue
    
    # Save aggregate summary across all reports for this model
    if summaries:
        summary_df = pd.DataFrame(summaries)
        # Create safe filename from model name
        safe_model_name = model_name.lower().replace('/', '_')
        
        # Extract company code from PDF files for summary filename
        company_code = extract_company_code(pdf_files)
        summary_file = os.path.join(output_dir, f"{safe_model_name}_summary_{company_code}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary: {summary_file}")
    
    return summaries


def main():
    """
    Main function that handles command-line arguments and orchestrates the analysis.
    
    Sets up argument parsing, validates inputs, discovers PDF files, and runs
    the selected BERT models on all discovered files.
    
    Returns:
        0 for success, 1 for error (following Unix convention)
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze annual reports with BERT models for ESG and sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run all models on ./data
  python main.py --model social                     # Run only social analysis
  python main.py --data-dir /path/to/pdfs           # Use custom input directory
  python main.py --social-threshold 0.8             # Use stricter social threshold
        """
    )
    
    # Directory arguments
    parser.add_argument(
        "--data-dir", 
        default="./data", 
        help="Directory containing PDF files to analyze (default: ./data)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./outputs", 
        help="Directory to save analysis results (default: ./outputs)"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        choices=["social", "environmental", "financial", "maori", "maori_xlm", "maori_mdeberta", "all"],
        default="all",
        help="Which BERT model(s) to run (default: all)"
    )
    
    # Classification thresholds for each model
    parser.add_argument(
        "--social-threshold",
        type=float,
        default=0.7,
        help="Threshold for social governance classification (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--env-threshold",
        type=float,
        default=0.7,
        help="Threshold for environmental classification (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--fin-threshold",
        type=float,
        default=0.6,
        help="Threshold for financial positive sentiment (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--maori-threshold",
        type=float,
        default=0.7,
        help="Threshold for Māori wellbeing classification (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--maori-xlm-threshold",
        type=float,
        default=0.7,
        help="Threshold for Māori wellbeing XLM-RoBERTa classification (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--maori-mdeberta-threshold",
        type=float,
        default=0.7,
        help="Threshold for Māori wellbeing mDeBERTa classification (0.0-1.0, default: 0.7)"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate that data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        print("Please create the directory and add PDF files to analyze.")
        return 1
    
    # Discover PDF files in the data directory
    pdf_files = get_pdf_files(args.data_dir)
    if not pdf_files:
        print(f"No PDF files found in '{args.data_dir}'.")
        print("Please add PDF files to the directory.")
        return 1
    
    # Display discovered files to user
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis based on selected model(s)
    # Store results for potential future use or debugging
    all_summaries = {}
    
    # Social governance analysis
    if args.model in ["social", "all"]:
        print(f"\nInitializing Social Analyzer (threshold: {args.social_threshold})...")
        analyzer = SocialAnalyzer(threshold=args.social_threshold)
        output_subdir = os.path.join(args.output_dir, "social")
        summaries = process_reports_with_model(pdf_files, analyzer, "Social", output_subdir)
        all_summaries["social"] = summaries
    
    # Environmental sustainability analysis
    if args.model in ["environmental", "all"]:
        print(f"\nInitializing Environmental Analyzer (threshold: {args.env_threshold})...")
        analyzer = EnvironmentalAnalyzer(threshold=args.env_threshold)
        output_subdir = os.path.join(args.output_dir, "environmental")
        summaries = process_reports_with_model(pdf_files, analyzer, "Environmental", output_subdir)
        all_summaries["environmental"] = summaries
    
    # Financial sentiment analysis
    if args.model in ["financial", "all"]:
        print(f"\nInitializing Financial Analyzer (threshold: {args.fin_threshold})...")
        analyzer = FinancialAnalyzer(threshold=args.fin_threshold)
        output_subdir = os.path.join(args.output_dir, "financial")
        summaries = process_reports_with_model(pdf_files, analyzer, "Financial", output_subdir)
        all_summaries["financial"] = summaries

    # Māori wellbeing analysis (BART)
    if args.model in ["maori", "all"]:
        print(f"\nInitializing Māori Wellbeing Analyzer (BART) (threshold: {args.maori_threshold})...")
        analyzer = MaoriAnalyzer(threshold=args.maori_threshold)
        output_subdir = os.path.join(args.output_dir, "maori")
        summaries = process_reports_with_model(pdf_files, analyzer, "Māori Wellbeing (BART)", output_subdir)
        all_summaries["maori"] = summaries

    # Māori wellbeing analysis (XLM-RoBERTa)
    if args.model in ["maori_xlm", "all"]:
        print(f"\nInitializing Māori Wellbeing Analyzer (XLM-RoBERTa) (threshold: {args.maori_xlm_threshold})...")
        analyzer = MaoriXLMAnalyzer(threshold=args.maori_xlm_threshold)
        output_subdir = os.path.join(args.output_dir, "maori_xlm")
        summaries = process_reports_with_model(pdf_files, analyzer, "Māori Wellbeing (XLM-RoBERTa)", output_subdir)
        all_summaries["maori_xlm"] = summaries

    # Māori wellbeing analysis (mDeBERTa)
    if args.model in ["maori_mdeberta", "all"]:
        print(f"\nInitializing Māori Wellbeing Analyzer (mDeBERTa) (threshold: {args.maori_mdeberta_threshold})...")
        analyzer = MaoriMDeBERTaAnalyzer(threshold=args.maori_mdeberta_threshold)
        output_subdir = os.path.join(args.output_dir, "maori_mdeberta")
        summaries = process_reports_with_model(pdf_files, analyzer, "Māori Wellbeing (mDeBERTa)", output_subdir)
        all_summaries["maori_mdeberta"] = summaries

    # Display completion message
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    # Entry point when script is run directly
    # Exit with the return code from main() for proper shell integration
    exit(main())
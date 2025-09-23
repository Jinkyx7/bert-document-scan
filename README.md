# BERT Document Scan - Annual Report Analysis

A Python application for analyzing company annual reports using multiple BERT models to extract social, environmental, and financial insights.

## Features

- **Multi-model Analysis**: Uses three specialized BERT models:
  - **Social Analysis**: ESGBERT/SocRoBERTa-social
  - **Environmental Analysis**: ESGBERT/EnvRoBERTa-environmental  
  - **Financial Sentiment**: ProsusAI/finbert

- **Batch Processing**: Processes all PDF files in a directory automatically
- **Structured Output**: Generates consistent CSV outputs for all models
- **Configurable Thresholds**: Adjustable classification thresholds
- **Memory Efficient**: Handles large documents with batched processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bert-document-scan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place your PDF annual reports in the `data/` directory and run:

```bash
python main.py
```

This will analyze all PDF files with all three models and save results to `outputs/`.

### Advanced Usage

```bash
# Analyze with specific model only
python main.py --model social

# Use custom directories
python main.py --data-dir /path/to/pdfs --output-dir /path/to/results

# Adjust classification thresholds
python main.py --social-threshold 0.8 --env-threshold 0.75 --fin-threshold 0.65
```

### Command Line Options

- `--data-dir`: Directory containing PDF files (default: `./data`)
- `--output-dir`: Directory to save results (default: `./outputs`)
- `--model`: Which model to run (`social`, `environmental`, `financial`, or `all`)
- `--social-threshold`: Threshold for social classification (default: 0.7)
- `--env-threshold`: Threshold for environmental classification (default: 0.7)
- `--fin-threshold`: Threshold for financial positive sentiment (default: 0.6)

## Output Structure

The application creates organized output directories:

```
outputs/
├── social/
│   ├── social_scores_COMPANY2024.csv      # All sentences with scores
│   ├── social_hits_COMPANY2024_min0_7.csv # High-confidence hits
│   └── social_summary.csv                 # Summary across all reports
├── environmental/
│   ├── environment_scores_COMPANY2024.csv
│   ├── environment_hits_COMPANY2024_min0_7.csv
│   └── environmental_summary.csv
└── financial/
    ├── finbert_results_all_COMPANY2024.csv
    ├── finbert_results_hits_COMPANY2024.csv
    └── financial_summary.csv
```

### Output File Formats

**Scores Files** (all sentences):
- `page`: Page number in PDF
- `[model]_score`: Probability score for target class
- `sentence`: Original sentence text

**Hits Files** (high-confidence matches):
- Same format as scores files, filtered and sorted by confidence

**Summary Files**:
- `report`: Report name
- `total_sentences`: Total sentences processed
- `[category]_candidates`: Number of high-confidence matches

## Project Structure

```
bert-document-scan/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── data/                      # Place PDF files here
├── outputs/                   # Analysis results
└── src/
    ├── models/
    │   ├── base_analyzer.py      # Base class for analyzers
    │   ├── social_analyzer.py    # Social aspect analysis
    │   ├── environmental_analyzer.py  # Environmental analysis
    │   └── financial_analyzer.py     # Financial sentiment
    └── utils/
        └── pdf_processor.py      # PDF text extraction utilities
```

## Models Used

1. **ESGBERT/SocRoBERTa-social**: Identifies social responsibility content
2. **ESGBERT/EnvRoBERTa-environmental**: Identifies environmental content  
3. **ProsusAI/finbert**: Analyzes financial sentiment (positive/negative/neutral)

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- PyMuPDF
- BlingFire
- pandas
- tqdm

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
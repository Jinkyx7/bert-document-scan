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

### Running Specific Models

You can run individual models or all models together:

```bash
# Run all models (default)
python main.py

# Run only social analysis
python main.py --model social

# Run only environmental analysis
python main.py --model environmental

# Run only financial sentiment analysis
python main.py --model financial
```

### Advanced Configuration

```bash
# Use custom directories
python main.py --data-dir /path/to/pdfs --output-dir /path/to/results

# Adjust classification thresholds
python main.py --social-threshold 0.8 --env-threshold 0.75 --fin-threshold 0.65

# Combine model selection with custom settings
python main.py --model social --social-threshold 0.9 --data-dir ./reports
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

## Adding New Models

To add a new BERT model for analysis, follow these steps:

### 1. Create a New Analyzer Class

Create a new analyzer in `src/models/` using the pipeline approach:

```python
# src/models/your_analyzer.py
import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline

class YourAnalyzer:
    def __init__(self, threshold: float = 0.7, batch_size: int = 32):
        self.threshold = threshold
        self.batch_size = batch_size

        # Initialize pipeline
        self.classifier = pipeline(
            task="text-classification",
            model="huggingface/your-model-name",
            tokenizer="huggingface/your-model-name",
            return_all_scores=True,
            truncation=True
        )

    def score_sentences(self, sentences: List[str]) -> List[float]:
        # Implementation similar to social/environmental analyzers
        pass

    def analyze_report(self, sentences_data: List[Dict], report_name: str, output_dir: str) -> Dict[str, Any]:
        # Implementation similar to social/environmental analyzers
        pass
```

### 2. Update Main Script

Add your model to `main.py`:

```python
# In the imports section
from src.models.your_analyzer import YourAnalyzer

# In the model selection logic
elif args.model == 'your_model':
    analyzer = YourAnalyzer()
    # Add threshold argument if needed
```

### 3. Add Command Line Arguments

Update the argument parser in `main.py`:

```python
parser.add_argument('--model', choices=['social', 'environmental', 'financial', 'your_model', 'all'])
parser.add_argument('--your-threshold', type=float, default=0.7)
```

### Model Requirements

- Model should be available on Hugging Face Hub
- All analyzers now use the transformers pipeline approach for consistency
- Models should support text classification task
- Handle label extraction by checking for appropriate label names in the pipeline output

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
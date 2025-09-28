"""
PDF processing utilities for extracting text and sentences from annual reports.

This module provides functions for:
- Handling local PDF files
- Extracting text from PDFs with page number tracking
- Splitting text into clean sentences for BERT analysis
- Generating standardized report names from file paths
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF - for PDF text extraction
# from blingfire import text_to_sentences  # Microsoft's fast sentence segmentation

def clean_extracted_text(text: str) -> str:
    """
    Clean up common encoding issues in PDF text extraction.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned text with proper character replacements
    """
    if not text:
        return text


    # Try different approaches
    cleaned_text = text

    # Approach 1: Direct string replacement
    simple_replacements = {
        'We‚Äôre': "We're",
        'We‚Äôve': "We've",
        '‚Äôs': "'s",
        'Äô': "'", 'Äì': "-", '≈ç': 'fi', 'Â': '',
        '‚Äô': "'",  # Common encoding issue for apostrophes
        '‚Äì': "-",  # Common encoding issue for dashes
        '‚Äù': "'",  # Another apostrophe variant
        '‚Äú': "'",  # Another apostrophe variant
    }

    for old_char, new_char in simple_replacements.items():
        count = cleaned_text.count(old_char)
        if count > 0:
            cleaned_text = cleaned_text.replace(old_char, new_char)

    return cleaned_text


def text_to_sentences(text: str) -> str:
    """
    Simple but effective sentence segmentation with abbreviation protection.
    
    This function splits text into sentences while protecting common abbreviations
    and applying basic validation to filter out obvious fragments.
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Clean and normalize the text
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    # Step 2: Protect common abbreviations that should NOT break sentences
    protected_text = text
    
    # Key abbreviations found in business/financial documents
    abbreviations = [
        r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Corp|Co|vs|etc|est|approx|govt)\.',
        r'\b(?:U\.S|U\.K|U\.N|E\.U)\.',  # Countries/regions
        r'\b(?:e\.g|i\.e|cf|viz)\.',     # Latin abbreviations
        r'\b[A-Z]\.',                    # Single capital letters (A., B., etc.)
        r'\$\d+\.\d+',                   # Money amounts ($100.50)
        r'\b\d+\.\d+',                   # Decimal numbers (3.14, 15.5)
    ]
    
    # Replace dots in abbreviations with placeholder
    for pattern in abbreviations:
        protected_text = re.sub(pattern, lambda m: m.group(0).replace('.', '<DOT>'), protected_text, flags=re.IGNORECASE)
    
    # Step 3: Split on sentence boundaries
    # Look for period/!/? followed by space and capital letter (or end)
    sentence_boundaries = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', protected_text)
    
    # Step 4: Simple validation - keep sentences that look complete
    valid_sentences = []
    
    for sentence in sentence_boundaries:
        if not sentence:
            continue
            
        # Restore dots in abbreviations
        sentence = sentence.replace('<DOT>', '.').strip()
        
        # Simple validation criteria:
        if (len(sentence) >= 10 and                                    # Minimum length (lowered from 20)
            re.search(r'[a-zA-Z]', sentence) and                      # Contains letters
            not sentence.lower().rstrip('.!?').endswith((             # Doesn't end with clear fragments
                'with', 'and', 'or', 'but', 'during', 'since', 'until', 'while',
                'the', 'a', 'an', 'this', 'that', 'which', 'we', 'they'
            ))):
            valid_sentences.append(sentence)
    
    return '\n'.join(valid_sentences)


def fetch_pdf(pdf_source: str) -> str:
    """
    Return local file path for processing.
    
    This function simply returns the provided local file path.
    
    Args:
        pdf_source: Local file path to PDF
        
    Returns:
        Local file path to the PDF
    """
    return pdf_source


def extract_sentences_with_pages(pdf_path: str, min_len: int = 30, max_len: int = 600) -> List[Dict]:
    """
    Extract clean sentences from PDF with page number tracking.
    
    This function processes each page of a PDF to extract meaningful sentences
    suitable for BERT analysis. It performs text cleaning and filtering to
    remove very short/long sentences that may not be useful for classification.
    
    Args:
        pdf_path: Path to the PDF file to process
        min_len: Minimum sentence length (chars) to include (default: 30)
        max_len: Maximum sentence length (chars) to include (default: 600)
        
    Returns:
        List of dictionaries, each containing:
        - 'page': Page number (1-indexed) where sentence was found
        - 'sentence': Clean sentence text ready for BERT processing
        
    Note:
        Uses BlingFire for robust sentence segmentation, which handles
        abbreviations and edge cases better than simple regex splitting.
    """
    # Open PDF document using PyMuPDF
    doc = fitz.open(pdf_path)
    sentences = []
    
    # Process each page individually to track page numbers
    for page_num in range(len(doc)):
        # Extract raw text from the page
        text = doc[page_num].get_text("text")
        
        # Clean up encoding issues first
        text = clean_extracted_text(text)
        
        # Clean up whitespace and formatting issues common in PDF extraction
        text = re.sub(r"[ \t]+", " ", text)        # Normalize multiple spaces/tabs to single space
        text = re.sub(r"\s+\n", "\n", text)        # Clean up space before newlines
        text = re.sub(r"\n{2,}", "\n", text)       # Collapse multiple newlines to single
        
        # Use BlingFire for robust sentence segmentation
        # BlingFire handles abbreviations, decimal numbers, etc. better than regex
        for sentence in text_to_sentences(text).split("\n"):
            # Perform an additional cleaning pass on each sentence after
            # segmentation to remove any residual encoding artefacts.  This
            # ensures that the financial and ESG analyzers receive clean
            # sentences even if the segmentation introduces new boundaries.
            sentence = clean_extracted_text(sentence.strip())

            # Filter sentences by length to exclude:
            # - Very short text (likely headers, page numbers, etc.)
            # - Very long text (likely tables, lists that aren't proper sentences)
            if min_len <= len(sentence) <= max_len:
                sentences.append({
                    "page": page_num + 1,  # Convert to 1-indexed page numbers
                    "sentence": sentence
                })
    
    # Clean up PyMuPDF document to free memory
    doc.close()
    return sentences


def safe_report_name(path_or_url: str) -> str:
    """
    Extract a clean, standardized report name from file path or URL.
    
    This function attempts to create consistent report names for output files
    by looking for common annual report naming patterns (e.g., "AIR2024", "FSF2023").
    
    Args:
        path_or_url: File path or URL to extract name from
        
    Returns:
        Clean report name suitable for use in output filenames
        Examples: 'AIR2024', 'FSF2023', 'COMPANY_REPORT'
        
    Note:
        The function prioritizes extracting ticker/company codes followed by years,
        which is a common pattern in annual report filenames.
    """
    # Extract filename without extension from path/URL
    name = Path(path_or_url).stem
    
    # Look for common annual report patterns: 2-4 letters followed by 4-digit year
    # Examples: AIR2024, FSF2023, AAPL2024, MSFT2023
    match = re.search(r"[A-Za-z]{2,}\d{4}", name)
    if match:
        return match.group(0).upper()
    
    # Fallback: sanitize the entire filename for safe use in output paths
    # Replace any non-alphanumeric characters with underscores
    clean_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return clean_name or "REPORT"


def extract_company_code(pdf_files: List[str]) -> str:
    """
    Extract company code from a list of PDF files.
    
    This function identifies the common company code from PDF filenames,
    useful for naming summary files with the company identifier.
    
    Args:
        pdf_files: List of PDF file paths
        
    Returns:
        Company code (e.g., "CEN", "AIR", "FSF") or "MIXED" if multiple companies
        
    Examples:
        ["CEN2024.pdf", "CEN2023.pdf"] → "CEN"
        ["AIR2024.pdf", "FSF2023.pdf"] → "MIXED" 
    """
    if not pdf_files:
        return "UNKNOWN"
    
    # Extract potential company codes from all files
    company_codes = set()
    for pdf_file in pdf_files:
        filename = Path(pdf_file).stem
        # Look for pattern: letters followed by year (e.g., CEN2024, AIR2023)
        match = re.search(r"([A-Za-z]{2,})(\d{4})", filename)
        if match:
            company_codes.add(match.group(1).upper())
        else:
            # Fallback: try to extract first few letters before numbers
            letters_match = re.search(r"^([A-Za-z]+)", filename)
            if letters_match and len(letters_match.group(1)) >= 2:
                company_codes.add(letters_match.group(1).upper())
    
    # Return single code if all files are from same company, otherwise "MIXED"
    if len(company_codes) == 1:
        return list(company_codes)[0]
    elif len(company_codes) > 1:
        return "MIXED"
    else:
        return "UNKNOWN"


def get_pdf_files(directory: str) -> List[str]:
    """
    Scan directory for PDF files and return sorted list of paths.
    
    This function finds all PDF files in the specified directory and returns
    them in alphabetical order for consistent processing order.
    
    Args:
        directory: Directory path to scan for PDF files
        
    Returns:
        Sorted list of absolute paths to PDF files
        
    Note:
        Only looks for files with .pdf extension (case-insensitive).
        Does not search subdirectories recursively.
    """
    pdf_files = []
    
    # Use pathlib to find all PDF files in the directory
    for file_path in Path(directory).glob("*.pdf"):
        # Verify it's actually a file (not a directory with .pdf in name)
        if file_path.is_file():
            pdf_files.append(str(file_path))  # Convert Path object to string
    
    # Return sorted list for consistent processing order
    return sorted(pdf_files)
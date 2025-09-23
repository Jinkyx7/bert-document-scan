"""
PDF processing utilities for extracting text and sentences from annual reports.
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from blingfire import text_to_sentences


def fetch_pdf(pdf_source: str, temp_dir: str = "/tmp") -> str:
    """
    Download PDF from URL or return local path.
    
    Args:
        pdf_source: Local file path or HTTP URL
        temp_dir: Directory for downloaded files
        
    Returns:
        Local file path to the PDF
    """
    if pdf_source.lower().startswith("http"):
        filename = os.path.basename(pdf_source.split("?")[0]) or "report.pdf"
        output_path = os.path.join(temp_dir, filename)
        
        response = requests.get(pdf_source, timeout=120)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    
    return pdf_source


def extract_sentences_with_pages(pdf_path: str, min_len: int = 30, max_len: int = 600) -> List[Dict]:
    """
    Extract sentences from PDF with page numbers.
    
    Args:
        pdf_path: Path to PDF file
        min_len: Minimum sentence length
        max_len: Maximum sentence length
        
    Returns:
        List of dictionaries with 'page' and 'sentence' keys
    """
    doc = fitz.open(pdf_path)
    sentences = []
    
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        
        # Clean up text
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text)
        
        # Split into sentences
        for sentence in text_to_sentences(text).split("\n"):
            sentence = sentence.strip()
            if min_len <= len(sentence) <= max_len:
                sentences.append({
                    "page": page_num + 1,
                    "sentence": sentence
                })
    
    doc.close()
    return sentences


def safe_report_name(path_or_url: str) -> str:
    """
    Extract a clean report name from file path or URL.
    
    Args:
        path_or_url: File path or URL
        
    Returns:
        Clean report name (e.g., 'AIR2024')
    """
    name = Path(path_or_url).stem
    
    # Look for pattern like AIR2024, FSF2023, etc.
    match = re.search(r"[A-Za-z]{2,}\d{4}", name)
    if match:
        return match.group(0).upper()
    
    # Fallback: sanitize the entire name
    clean_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return clean_name or "REPORT"


def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files from a directory.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    for file_path in Path(directory).glob("*.pdf"):
        if file_path.is_file():
            pdf_files.append(str(file_path))
    
    return sorted(pdf_files)
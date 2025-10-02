"""
Māori wellbeing analyzer using facebook/bart-large-mnli zero-shot classification.

This module identifies sentences related to Māori wellbeing, kaupapa Māori services,
and Māori health frameworks using a hybrid approach combining keyword matching
and zero-shot classification.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import pipeline


class MaoriAnalyzer:
    """
    Analyzer for identifying Māori wellbeing content in annual reports.

    Uses a hybrid approach:
    1. Keyword matching with Māori health lexicon (frameworks, practices, terminology)
    2. Zero-shot classification using facebook/bart-large-mnli
    3. Combined scoring to improve accuracy

    Key areas covered:
    - Māori health frameworks (Te Whare Tapa Whā, Whānau Ora)
    - Traditional practices (Rongoā Māori, tikanga, karakia)
    - Cultural concepts (mana whenua, kaitiakitanga, wairua)
    - Service delivery (kaupapa Māori, Māori-led providers)
    """

    def __init__(self, threshold: float = 0.7, batch_size: int = 8):
        """
        Initialize the Māori wellbeing analyzer.

        Args:
            threshold: Hybrid score threshold for filtering hits (0.0-1.0)
            batch_size: Number of sentences to process simultaneously
                       Note: Zero-shot models are slower; use smaller batches
        """
        self.threshold = threshold
        self.batch_size = batch_size

        # Zero-shot configuration
        self.zshot_min_score = 0.65  # Minimum zero-shot score for consideration
        self.keyword_min_hits = 0    # Minimum keyword hits (0 = all sentences scored)

        # Define zero-shot labels for Māori wellbeing classification
        self.zshot_labels = [
            "This sentence describes Māori wellbeing or kaupapa Māori services.",
            "This sentence discusses Māori culture or Māori implementation in services.",
            "This sentence is about Whānau Ora, Rongoā Māori, or Te Whare Tapa Whā.",
        ]

        # Initialize Māori health lexicon
        self._init_lexicon()

        # Load zero-shot classification model
        print("Loading facebook/bart-large-mnli zero-shot model...")
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )

        # Disable tokenizer parallelism to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _init_lexicon(self):
        """
        Initialize the Māori wellbeing lexicon with core terms and variants.

        Creates both exact and macron-insensitive versions of each term to ensure
        matching regardless of whether macrons are used in the source text.
        """
        # Core Māori health and wellbeing terms
        seed_terms = [
            # Frameworks & models
            "kaupapa Māori", "Mātauranga Māori", "Te Whare Tapa Whā",
            "Whānau Ora", "whānau-centred", "whānau centered", "whānau-led",

            # Health dimensions (Te Whare Tapa Whā)
            "taha wairua", "taha hinengaro", "taha tinana", "taha whānau",
            "wairua", "hinengaro", "tinana", "whānau", "wharenui",

            # Practices / services
            "Rongoā Māori", "tikanga", "kawa", "karakia", "waiata", "tohunga",

            # People / identity / governance
            "Māori-led", "Māori provider", "mana whenua", "tangata whenua",
            "iwi", "hapū", "marae", "rangatahi", "tamariki", "kaumātua", "kuia",
            "kaitiakitanga", "manaakitanga", "rangatiratanga", "te ao Māori",

            # Programmes / agencies
            "Te Whatu Ora", "Access and Choice",
        ]

        # Generate macron-insensitive variants
        self.lexicon = self._expand_variants(seed_terms)

        # Precompile regex patterns for efficient matching
        self.kw_patterns = [self._kw_pattern(term) for term in self.lexicon]

    def _expand_variants(self, terms: List[str]) -> List[str]:
        """
        Generate macron-insensitive and spacing variants of terms.

        Args:
            terms: List of base terms with proper macrons

        Returns:
            Expanded list with both original and variant forms
        """
        macron_map = str.maketrans({
            "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
            "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U"
        })

        variants = set()
        for term in terms:
            # Normalize whitespace
            normalized = re.sub(r"\s+", " ", term.strip())
            variants.add(normalized)

            # Add macron-removed version
            variants.add(normalized.translate(macron_map))

            # Add hyphen/space variants (e.g., "whānau-centred" vs "whānau centred")
            variants.add(normalized.replace("-", " "))
            variants.add(normalized.translate(macron_map).replace("-", " "))

        return sorted(variants, key=str.lower)

    def _kw_pattern(self, term: str) -> re.Pattern:
        """
        Create a word-boundary regex pattern for a keyword.

        Args:
            term: The keyword to match

        Returns:
            Compiled regex pattern with case-insensitive word boundaries
        """
        escaped = re.escape(term)
        return re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")

    def _keyword_hits(self, sentence: str) -> List[str]:
        """
        Find all lexicon terms that match in the sentence.

        Args:
            sentence: Text to search for keywords

        Returns:
            List of matched keyword terms
        """
        hits = []
        for term, pattern in zip(self.lexicon, self.kw_patterns):
            if pattern.search(sentence):
                hits.append(term)
        return hits

    def _zero_shot_score(self, sentence: str) -> float:
        """
        Calculate zero-shot classification score for Māori wellbeing.

        Args:
            sentence: Text to classify

        Returns:
            Maximum score across all candidate labels (0.0-1.0)
        """
        try:
            result = self.classifier(
                sentence,
                candidate_labels=self.zshot_labels,
                multi_label=True
            )
            # Take the best label score
            return float(max(result["scores"])) if "scores" in result else 0.0
        except Exception as e:
            print(f"Warning: Zero-shot scoring failed for sentence: {str(e)[:100]}")
            return 0.0

    def _hybrid_score(self, kw_count: int, zshot_score: float) -> float:
        """
        Combine keyword and zero-shot scores into a hybrid metric.

        Keyword matches provide a boost to the zero-shot score, improving
        precision when cultural terminology is present.

        Args:
            kw_count: Number of keyword matches
            zshot_score: Zero-shot classification score

        Returns:
            Combined score (0.0-1.0), capped at 1.0
        """
        # Boost formula: saturating increase based on keyword count
        # kw_count >= 3 → +0.1, >= 6 → +0.2, capped at +0.25
        boost = min(0.25, 0.05 * kw_count)
        return min(1.0, zshot_score + boost)

    def score_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Score sentences using hybrid keyword + zero-shot approach.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of dictionaries with keys:
            - 'keyword_hits': List of matched keywords
            - 'kw_count': Number of keyword matches
            - 'zshot_score': Zero-shot classification score
            - 'hybrid_score': Combined score
        """
        results = []

        # Process sentences in batches for zero-shot model
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]

            for sentence in batch:
                # Step 1: Keyword matching
                hits = self._keyword_hits(sentence)
                kw_count = len(hits)

                # Step 2: Zero-shot classification (only if keywords present or no min required)
                if kw_count >= self.keyword_min_hits:
                    zshot_score = self._zero_shot_score(sentence)
                    hybrid_score = self._hybrid_score(kw_count, zshot_score)
                else:
                    zshot_score = 0.0
                    hybrid_score = 0.0

                results.append({
                    'keyword_hits': hits,
                    'kw_count': kw_count,
                    'zshot_score': zshot_score,
                    'hybrid_score': hybrid_score
                })

        return results

    def analyze_report(self, sentences_data: List[Dict], report_name: str, output_dir: str) -> Dict[str, Any]:
        """
        Analyze sentences from a report and save results to CSV files.

        Creates two output files:
        1. All sentences with all scores (maori_results_all_*.csv)
        2. High-confidence hits above threshold (maori_results_hits_*.csv)

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
                "maori_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }

        # Convert sentence data to pandas DataFrame
        df = pd.DataFrame(sentences_data)

        # Run hybrid scoring on all sentences
        print(f"Analyzing {len(df)} sentences for Māori wellbeing content...")
        score_results = self.score_sentences(df["sentence"].tolist())

        # Add scoring results to DataFrame
        df["keyword_hits"] = ["; ".join(r['keyword_hits']) for r in score_results]
        df["kw_count"] = [r['kw_count'] for r in score_results]
        df["zshot_score"] = [round(r['zshot_score'], 4) for r in score_results]
        df["hybrid_score"] = [round(r['hybrid_score'], 4) for r in score_results]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save complete results: all sentences with all scores
        all_csv = os.path.join(output_dir, f"maori_results_all_{report_name}.csv")
        df[["page", "zshot_score", "hybrid_score", "kw_count", "keyword_hits", "sentence"]].to_csv(
            all_csv, index=False, encoding="utf-8"
        )

        # Filter high-confidence hits using hybrid score threshold
        # Also require minimum zero-shot score to avoid pure keyword matches
        hits = df[
            (df["hybrid_score"] >= self.threshold) |
            (df["zshot_score"] >= self.zshot_min_score)
        ].sort_values("hybrid_score", ascending=False)

        # Save filtered results: only high-confidence matches
        hits_csv = os.path.join(output_dir, f"maori_results_hits_{report_name}.csv")
        hits[["page", "zshot_score", "hybrid_score", "kw_count", "keyword_hits", "sentence"]].to_csv(
            hits_csv, index=False, encoding="utf-8"
        )

        # Return summary statistics
        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            "maori_candidates": int(len(hits)),
            "all_csv": all_csv,
            "hits_csv": hits_csv,
            "avg_zshot_score": round(hits["zshot_score"].mean(), 4) if len(hits) > 0 else 0.0,
            "avg_kw_count": round(hits["kw_count"].mean(), 2) if len(hits) > 0 else 0.0
        }

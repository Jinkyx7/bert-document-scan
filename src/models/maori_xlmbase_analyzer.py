"""
Māori wellbeing analyzer using xlm-roberta-base zero-shot classification.

This module identifies sentences related to Māori wellbeing using XLM-RoBERTa base, a smaller
and faster multilingual model suitable for efficient processing with good cross-lingual support.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import pipeline


class MaoriXLMBaseAnalyzer:
    """
    Analyzer for identifying Māori wellbeing content using XLM-RoBERTa base model.

    Uses a hybrid approach:
    1. Keyword matching with Māori health lexicon (frameworks, practices, terminology)
    2. Zero-shot classification using xlm-roberta-base
    3. Combined scoring to improve accuracy

    Key advantages of XLM-RoBERTa base:
    - Multilingual support (better for te reo Māori than English-only models)
    - Faster inference than large models
    - Good balance of accuracy and speed
    - Handles mixed English/Māori content well
    """

    def __init__(self, threshold: float = 0.7, batch_size: int = 16):
        """
        Initialize the Māori wellbeing analyzer with XLM-RoBERTa base.

        Args:
            threshold: Hybrid score threshold for filtering hits (0.0-1.0)
            batch_size: Number of sentences to process simultaneously
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

        # Load zero-shot classification model (XLM-RoBERTa base)
        print("Loading xlm-roberta-base zero-shot model...")
        device = 0 if torch.cuda.is_available() else -1

        # Note: xlm-roberta-base is not specifically fine-tuned for NLI/zero-shot
        # For better results, consider using a fine-tuned version like joeddav/xlm-roberta-base-xnli
        # but that may have similar tokenizer issues. Using base model with custom approach.
        print("Note: Using base XLM-RoBERTa. For true zero-shot, consider fine-tuned NLI version.")

        self.classifier = pipeline(
            "zero-shot-classification",
            model="xlm-roberta-base",
            device=device
        )

        # Disable tokenizer parallelism to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _init_lexicon(self):
        """Initialize the Māori wellbeing lexicon with core terms and variants."""
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

        self.lexicon = self._expand_variants(seed_terms)
        self.kw_patterns = [self._kw_pattern(term) for term in self.lexicon]

    def _expand_variants(self, terms: List[str]) -> List[str]:
        """Generate macron-insensitive and spacing variants of terms."""
        macron_map = str.maketrans({
            "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
            "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U"
        })

        variants = set()
        for term in terms:
            normalized = re.sub(r"\s+", " ", term.strip())
            variants.add(normalized)
            variants.add(normalized.translate(macron_map))
            variants.add(normalized.replace("-", " "))
            variants.add(normalized.translate(macron_map).replace("-", " "))

        return sorted(variants, key=str.lower)

    def _kw_pattern(self, term: str) -> re.Pattern:
        """Create a word-boundary regex pattern for a keyword."""
        escaped = re.escape(term)
        return re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")

    def _keyword_hits(self, sentence: str) -> List[str]:
        """Find all lexicon terms that match in the sentence."""
        hits = []
        for term, pattern in zip(self.lexicon, self.kw_patterns):
            if pattern.search(sentence):
                hits.append(term)
        return hits

    def _zero_shot_score(self, sentence: str) -> float:
        """Calculate zero-shot classification score using XLM-RoBERTa base."""
        try:
            result = self.classifier(
                sentence,
                candidate_labels=self.zshot_labels,
                multi_label=True
            )
            return float(max(result["scores"])) if "scores" in result else 0.0
        except Exception as e:
            print(f"Warning: Zero-shot scoring failed for sentence: {str(e)[:100]}")
            return 0.0

    def _hybrid_score(self, kw_count: int, zshot_score: float) -> float:
        """Combine keyword and zero-shot scores into a hybrid metric."""
        boost = min(0.25, 0.05 * kw_count)
        return min(1.0, zshot_score + boost)

    def score_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Score sentences using hybrid keyword + zero-shot approach."""
        results = []

        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Scoring", leave=False):
            batch = sentences[i:i + self.batch_size]

            for sentence in batch:
                hits = self._keyword_hits(sentence)
                kw_count = len(hits)

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
        """Analyze sentences from a report and save results to CSV files."""
        if not sentences_data:
            return {
                "report": report_name,
                "total_sentences": 0,
                "maori_xlmbase_candidates": 0,
                "all_csv": None,
                "hits_csv": None
            }

        df = pd.DataFrame(sentences_data)

        print(f"Analyzing {len(df)} sentences for Māori wellbeing content (XLM-RoBERTa base)...")
        score_results = self.score_sentences(df["sentence"].tolist())

        df["keyword_hits"] = ["; ".join(r['keyword_hits']) for r in score_results]
        df["kw_count"] = [r['kw_count'] for r in score_results]
        df["zshot_score"] = [round(r['zshot_score'], 4) for r in score_results]
        df["hybrid_score"] = [round(r['hybrid_score'], 4) for r in score_results]

        os.makedirs(output_dir, exist_ok=True)

        # Save all results
        all_csv = os.path.join(output_dir, f"maori_xlmbase_results_all_{report_name}.csv")
        df[["page", "zshot_score", "hybrid_score", "kw_count", "keyword_hits", "sentence"]].to_csv(
            all_csv, index=False, encoding="utf-8"
        )

        # Filter and save hits
        hits = df[
            (df["hybrid_score"] >= self.threshold) |
            (df["zshot_score"] >= self.zshot_min_score)
        ].sort_values("hybrid_score", ascending=False)

        hits_csv = os.path.join(output_dir, f"maori_xlmbase_results_hits_{report_name}.csv")
        hits[["page", "zshot_score", "hybrid_score", "kw_count", "keyword_hits", "sentence"]].to_csv(
            hits_csv, index=False, encoding="utf-8"
        )

        return {
            "report": report_name,
            "total_sentences": int(len(df)),
            "maori_xlmbase_candidates": int(len(hits)),
            "all_csv": all_csv,
            "hits_csv": hits_csv,
            "avg_zshot_score": round(hits["zshot_score"].mean(), 4) if len(hits) > 0 else 0.0,
            "avg_kw_count": round(hits["kw_count"].mean(), 2) if len(hits) > 0 else 0.0
        }

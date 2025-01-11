# src/eduai_detector/utils/text_analysis.py

from typing import List, Dict
import numpy as np
from collections import Counter

def calculate_text_statistics(text: str) -> Dict[str, float]:
    """Calculate basic statistical features of the text."""
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        "avg_word_length": np.mean([len(word) for word in words]),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "vocabulary_size": len(set(words)),
        "type_token_ratio": len(set(words)) / len(words) if words else 0,
    }
    return stats

def analyze_sentence_patterns(text: str) -> Dict[str, float]:
    """Analyze patterns in sentence structure."""
    sentences = text.split('.')
    patterns = {
        "sentence_length_variance": np.var([len(s.split()) for s in sentences if s.strip()]),
        "sentence_complexity": _calculate_sentence_complexity(sentences),
    }
    return patterns

def _calculate_sentence_complexity(sentences: List[str]) -> float:
    """Helper function to calculate sentence complexity."""
    # Placeholder for more sophisticated complexity analysis
    complexities = [len(s.split()) * len(set(s.split())) for s in sentences if s.strip()]
    return np.mean(complexities) if complexities else 0

def get_word_frequency_distribution(text: str) -> Dict[str, int]:
    """Calculate word frequency distribution."""
    words = text.lower().split()
    return dict(Counter(words))
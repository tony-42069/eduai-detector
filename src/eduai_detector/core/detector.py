# src/eduai_detector/core/detector.py

from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
import re
import math

class AITextDetector:
    def __init__(self):
        """Initialize the AI text detector with necessary components."""
        # Initialize detection thresholds
        self.thresholds = {
            "repetition": 0.3,
            "entropy": 4.2,
            "complexity": 3.0,
            "vocabulary_diversity": 0.85
        }

    def detect(self, text: str) -> Tuple[bool, Dict[str, float], str]:
        """Main detection method that determines if text is AI-generated."""
        if not text.strip():
            return False, {}, "No text provided"

        # Calculate metrics
        metrics = self._calculate_metrics(text)
        
        # Determine if text is AI-generated based on metrics
        indicators = [
            metrics["repetition_score"] > self.thresholds["repetition"],
            metrics["entropy_score"] > self.thresholds["entropy"],
            metrics["complexity_score"] > self.thresholds["complexity"],
            metrics["vocabulary_diversity"] > self.thresholds["vocabulary_diversity"]
        ]
        
        is_ai_generated = bool(sum(indicators) >= 3)
        explanation = self._generate_explanation(metrics, indicators, is_ai_generated)
        
        return is_ai_generated, metrics, explanation

    def _calculate_metrics(self, text: str) -> Dict[str, float]:
        """Calculate all metrics for the given text."""
        words = text.lower().split()
        
        metrics = {
            "repetition_score": float(self._calculate_repetition(words)),
            "entropy_score": float(self._calculate_entropy(text)),
            "complexity_score": float(self._calculate_complexity(text)),
            "vocabulary_diversity": float(self._calculate_vocabulary_diversity(words))
        }
        
        return metrics

    def _calculate_repetition(self, words: List[str]) -> float:
        """Calculate repetition score based on bigrams."""
        if len(words) < 2:
            return 0.0
        
        # Create bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        # Calculate repetition score
        total_bigrams = len(bigrams)
        repeated_bigrams = sum(count > 1 for count in bigram_counts.values())
        
        return repeated_bigrams / total_bigrams if total_bigrams > 0 else 0.0

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text."""
        if not text:
            return 0.0
            
        # Calculate character frequency
        freq = Counter(text.lower())
        length = len(text)
        
        # Calculate entropy
        entropy = -sum((count/length) * math.log2(count/length) 
                      for count in freq.values())
        
        return entropy

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity based on sentence and word length."""
        if not text:
            return 0.0
            
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.lower().split()
        
        if not sentences or not words:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Combine metrics
        return (avg_word_length * 0.5 + avg_sentence_length * 0.05)

    def _calculate_vocabulary_diversity(self, words: List[str]) -> float:
        """Calculate vocabulary diversity (unique words / total words)."""
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _generate_explanation(self, metrics: Dict[str, float], 
                            indicators: List[bool], 
                            is_ai_generated: bool) -> str:
        """Generate a detailed explanation of the analysis."""
        parts = []
        
        if is_ai_generated:
            parts.append("This text shows multiple characteristics of AI-generated content:")
            
            if indicators[0]:
                parts.append("- Higher than normal repetition patterns")
            if indicators[1]:
                parts.append("- Unusual entropy patterns in text structure")
            if indicators[2]:
                parts.append("- Higher than typical text complexity")
            if indicators[3]:
                parts.append("- Unusually consistent vocabulary usage")
        else:
            parts.append("This text shows characteristics more typical of human-written content:")
            parts.append("- Natural variation in writing patterns")
            parts.append("- Expected levels of text complexity")
            parts.append("- Normal language patterns")
        
        parts.append("\nDetailed metrics:")
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            parts.append(f"- {formatted_key}: {value:.3f}")
        
        return "\n".join(parts)
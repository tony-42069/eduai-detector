# src/eduai_detector/core/detector.py

from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import math

class AITextDetector:
    def __init__(self):
        """Initialize the AI text detector with statistical analysis tools."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english'
        )
        
        # Initialize detection thresholds
        self.repetition_threshold = 0.3
        self.complexity_threshold = 0.7
        self.entropy_threshold = 4.0

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text

    def calculate_metrics(self, text: str) -> Dict[str, float]:
        """Calculate various statistical metrics for AI detection."""
        # Get basic text units
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        
        metrics = {
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]),
            "sentence_length_variance": np.var([len(s.split()) for s in sentences]),
            "vocabulary_diversity": len(set(words)) / len(words) if words else 0,
            "repetition_score": self._calculate_repetition(words),
            "entropy_score": self._calculate_entropy(text),
            "complexity_score": self._calculate_complexity(text)
        }
        
        return metrics

    def detect(self, text: str) -> Tuple[bool, Dict[str, float], str]:
        """
        Main detection method that returns whether text is likely AI-generated.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple containing:
            - bool: True if likely AI-generated, False otherwise
            - Dict[str, float]: Detailed metrics
            - str: Explanation of the decision
        """
        processed_text = self.preprocess_text(text)
        metrics = self.calculate_metrics(processed_text)
        
        # Implement detection logic based on statistical patterns
        indicators = [
            metrics["repetition_score"] > self.repetition_threshold,
            metrics["complexity_score"] > self.complexity_threshold,
            metrics["entropy_score"] > self.entropy_threshold,
            metrics["vocabulary_diversity"] > 0.8,  # Unusually high vocabulary diversity
            metrics["sentence_length_variance"] < 0.5  # Too consistent sentence structure
        ]
        
        # Text is flagged as AI-generated if it triggers multiple indicators
        is_ai_generated = sum(indicators) >= 3
        
        explanation = self._generate_explanation(metrics, indicators, is_ai_generated)
        
        return is_ai_generated, metrics, explanation

    def _calculate_repetition(self, words: List[str]) -> float:
        """Calculate repetition patterns in text."""
        if not words:
            return 0.0
            
        # Count bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        # Calculate repetition score based on bigram frequency
        total_bigrams = len(bigrams)
        repeated_bigrams = sum(count > 1 for count in bigram_counts.values())
        
        return repeated_bigrams / total_bigrams if total_bigrams > 0 else 0

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text."""
        if not text:
            return 0.0
            
        # Calculate character frequency
        freq = Counter(text)
        length = len(text)
        
        # Calculate entropy
        entropy = -sum((count/length) * math.log2(count/length) 
                      for count in freq.values())
        
        return entropy

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        if not text:
            return 0.0
            
        # Split into sentences and words
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Calculate average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Calculate average sentence length
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        
        # Combine metrics into complexity score
        complexity = (avg_word_length * 0.5 + avg_sentence_length * 0.05)
        
        return complexity

    def _generate_explanation(self, metrics: Dict[str, float], 
                            indicators: List[bool], 
                            is_ai_generated: bool) -> str:
        """Generate detailed explanation of the detection decision."""
        parts = []
        
        if is_ai_generated:
            parts.append("This text shows multiple characteristics of AI-generated content:")
            
            if indicators[0]:
                parts.append("- Higher than normal repetition patterns in language use")
            if indicators[1]:
                parts.append("- Unusually consistent text complexity")
            if indicators[2]:
                parts.append("- Abnormal entropy patterns in text structure")
            if indicators[3]:
                parts.append("- Suspiciously diverse vocabulary usage")
            if indicators[4]:
                parts.append("- Unnaturally consistent sentence lengths")
                
            parts.append("\nDetailed metrics:")
            parts.append(f"- Repetition score: {metrics['repetition_score']:.3f}")
            parts.append(f"- Complexity score: {metrics['complexity_score']:.3f}")
            parts.append(f"- Entropy score: {metrics['entropy_score']:.3f}")
            parts.append(f"- Vocabulary diversity: {metrics['vocabulary_diversity']:.3f}")
            
        else:
            parts.append("This text shows characteristics more typical of human-written content:")
            parts.append("- Natural variation in language patterns")
            parts.append("- Expected levels of text complexity and structure")
            parts.append("- Normal entropy and repetition patterns")
        
        return "\n".join(parts)
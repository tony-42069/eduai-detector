# src/eduai_detector/core/detector.py

from typing import Dict, List, Tuple
from collections import Counter
import re
import math
from textblob import TextBlob

class AITextDetector:
    def __init__(self):
        """Initialize the AI text detector with necessary components."""
        # Initialize detection weights - must sum to 1.0
        self.metric_weights = {
            "repetition_score": 0.15,    # Increased as it's more reliable without NLTK
            "entropy_score": 0.20,       # Increased as it's more reliable
            "complexity_score": 0.15,    # Increased for simpler calculation
            "vocabulary_diversity": 0.20, # Keep same
            "sentence_variation": 0.15,   # Reduced due to simpler calculation
            "readability": 0.15          # Increased as it's more reliable
        }

        # Initialize baseline thresholds
        self.thresholds = {
            "repetition_score": 0.25,    # Repetition in text
            "entropy_score": 4.3,        # Word distribution entropy
            "complexity_score": 3.5,     # Text complexity
            "vocabulary_diversity": 0.45, # Unique words ratio
            "sentence_variation": 0.30,   # Sentence length variation
            "readability": 0.60          # TextBlob sentiment deviation
        }

    def detect(self, text: str) -> Tuple[bool, float, Dict[str, float], str]:
        """Detect if the text is AI-generated."""
        # Calculate metrics
        metrics = self._calculate_metrics(text)
        
        # Calculate weighted score
        weighted_score = sum(
            self.metric_weights[metric] * self._normalize_metric(metric, value)
            for metric, value in metrics.items()
        )

        # Determine if text is AI-generated based on weighted score
        is_ai_generated = weighted_score > 0.6  # Conservative threshold

        # Generate explanation
        explanation = self._generate_explanation(metrics, weighted_score, is_ai_generated)

        return is_ai_generated, weighted_score, metrics, explanation

    def _calculate_metrics(self, text: str) -> Dict[str, float]:
        """Calculate all metrics for the text."""
        return {
            "repetition_score": self._calculate_repetition(text),
            "entropy_score": self._calculate_entropy(text),
            "complexity_score": self._calculate_complexity(text),
            "vocabulary_diversity": self._calculate_vocabulary_diversity(text),
            "sentence_variation": self._calculate_sentence_variation(text),
            "readability": self._calculate_readability(text)
        }

    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize metric value to a 0-1 scale."""
        if metric in ["vocabulary_diversity", "sentence_variation"]:
            return value  # These are already 0-1 scaled
        elif metric in ["repetition_score", "entropy_score", "complexity_score"]:
            return min(value / self.thresholds[metric], 1.0)
        elif metric == "readability":
            return value  # Already normalized
        return value

    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition score using word pairs."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        # Use simple word pairs instead of NLTK trigrams
        pairs = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        pair_freq = Counter(pairs)
        
        # Calculate repetition score
        total_pairs = len(pairs)
        repeated_pairs = sum(freq - 1 for freq in pair_freq.values() if freq > 1)
        
        return repeated_pairs / total_pairs if total_pairs > 0 else 0.0

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        word_freq = Counter(words)
        total_words = len(words)
        
        entropy = 0
        for count in word_freq.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)
            
        return entropy

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.0
        
        # Calculate average sentence length and word length
        words = text.split()
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return (avg_sentence_length * 0.5 + avg_word_length * 0.5)

    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (type-token ratio)."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words

    def _calculate_sentence_variation(self, text: str) -> float:
        """Calculate sentence length variation."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]
        
        # Calculate variation coefficient
        mean_length = sum(lengths) / len(lengths)
        variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        variation = std_dev / mean_length if mean_length > 0 else 0
        return min(variation, 1.0)  # Normalize to 0-1

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using TextBlob."""
        try:
            blob = TextBlob(text)
            # Use polarity and subjectivity as indicators
            polarity = (blob.sentiment.polarity + 1) / 2  # Convert -1,1 to 0,1
            subjectivity = blob.sentiment.subjectivity
            return (polarity + subjectivity) / 2  # Combined score
        except:
            return 0.5  # Default to neutral if calculation fails

    def _generate_explanation(self, metrics: Dict[str, float], confidence_score: float, is_ai_generated: bool) -> str:
        """Generate a detailed explanation of the analysis."""
        explanation = []
        
        # Overall result
        if is_ai_generated:
            explanation.append(f"This text shows characteristics of AI-generated content (Confidence: {confidence_score:.1%}).")
        else:
            explanation.append(f"This text appears to be human-written (Confidence: {(1-confidence_score):.1%}).")
        
        explanation.append("\nKey indicators:")
        
        # Explain significant metrics
        if metrics["vocabulary_diversity"] < self.thresholds["vocabulary_diversity"]:
            explanation.append("- Lower vocabulary diversity than typical human writing")
        else:
            explanation.append("- Natural vocabulary diversity typical of human writing")
            
        if metrics["sentence_variation"] < self.thresholds["sentence_variation"]:
            explanation.append("- Unusually consistent sentence lengths")
        else:
            explanation.append("- Natural variation in sentence structure")
            
        if metrics["entropy_score"] > self.thresholds["entropy_score"]:
            explanation.append("- Highly uniform word distribution")
        else:
            explanation.append("- Natural word distribution patterns")
            
        if metrics["complexity_score"] > self.thresholds["complexity_score"]:
            explanation.append("- Unusually complex sentence structures")
        else:
            explanation.append("- Natural sentence complexity")
        
        # Add metric values
        explanation.append("\nDetailed metrics:")
        for metric, value in metrics.items():
            formatted_metric = metric.replace('_', ' ').title()
            explanation.append(f"- {formatted_metric}: {value:.3f}")
        
        return "\n".join(explanation)
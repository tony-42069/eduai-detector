# src/eduai_detector/core/detector.py

from typing import Dict, List, Tuple
from statistics import mean, stdev
from collections import Counter
import re
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.util import ngrams
from textblob import TextBlob

class AITextDetector:
    def __init__(self):
        """Initialize the AI text detector with necessary components."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')

        # Initialize detection weights - must sum to 1.0
        self.metric_weights = {
            "repetition_score": 0.10,    # Keep low as it's less reliable
            "entropy_score": 0.15,       # Reduced as it can be misleading for academic writing
            "complexity_score": 0.10,    # Reduced as academic writing is naturally complex
            "vocabulary_diversity": 0.20, # Increased as it's more reliable
            "sentence_variation": 0.20,   # Increased as it's more reliable
            "transition_patterns": 0.10,  # Keep same
            "pos_distribution": 0.10,     # Keep same
            "readability": 0.05          # Keep low
        }

        # Initialize baseline thresholds - adjusted to be more conservative
        self.thresholds = {
            "repetition_score": 0.25,    # Increased to allow more repetition in human writing
            "entropy_score": 4.3,        # Increased as academic writing can have high entropy
            "complexity_score": 3.5,     # Increased for academic writing
            "vocabulary_diversity": 0.45, # Lowered threshold makes it harder to flag as AI
            "sentence_variation": 0.30,   # Lowered to account for academic style
            "transition_patterns": 0.03,  # Lowered as even human writing can have few transitions
            "pos_distribution": 0.85,    # Increased to be more conservative
            "readability": 50.0          # Increased to be more conservative
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
            "transition_patterns": self._calculate_transition_patterns(text),
            "pos_distribution": self._calculate_pos_distribution(text),
            "readability": self._calculate_readability(text)
        }

    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize metric value to a 0-1 scale."""
        if metric in ["vocabulary_diversity", "sentence_variation", "transition_patterns"]:
            return value  # These are already 0-1 scaled
        elif metric == "repetition_score":
            return min(value / self.thresholds[metric], 1.0)
        elif metric == "entropy_score":
            return min(value / self.thresholds[metric], 1.0)
        elif metric == "complexity_score":
            return min(value / self.thresholds[metric], 1.0)
        elif metric == "pos_distribution":
            return value
        elif metric == "readability":
            return min(value / 100.0, 1.0)  # Normalize to 0-1 scale
        return value

    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition score using trigrams."""
        words = word_tokenize(text.lower())
        if len(words) < 3:
            return 0.0
        
        trigrams = list(ngrams(words, 3))
        trigram_freq = Counter(trigrams)
        
        # Calculate repetition score
        total_trigrams = len(trigrams)
        repeated_trigrams = sum(freq - 1 for freq in trigram_freq.values() if freq > 1)
        
        return repeated_trigrams / total_trigrams if total_trigrams > 0 else 0.0

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text."""
        words = word_tokenize(text.lower())
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
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        # Calculate average sentence length and word length
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_sentence_length = mean(sentence_lengths) if sentence_lengths else 0
        
        words = word_tokenize(text)
        avg_word_length = mean([len(word) for word in words]) if words else 0
        
        return (avg_sentence_length * 0.5 + avg_word_length * 0.5)

    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (type-token ratio)."""
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words

    def _calculate_sentence_variation(self, text: str) -> float:
        """Calculate sentence length variation."""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        try:
            variation = stdev(lengths) / mean(lengths) if lengths else 0
            return min(variation, 1.0)  # Normalize to 0-1
        except:
            return 0.0

    def _calculate_transition_patterns(self, text: str) -> float:
        """Calculate transition words frequency."""
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
            'thus', 'consequently', 'meanwhile', 'subsequently', 'conversely'
        }
        
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
        
        transition_count = sum(1 for word in words if word in transition_words)
        return transition_count / len(words)

    def _calculate_pos_distribution(self, text: str) -> float:
        """Calculate uniformity of POS tag distribution."""
        words = word_tokenize(text)
        if not words:
            return 0.0
        
        pos_tags = [tag for _, tag in pos_tag(words)]
        tag_freq = Counter(pos_tags)
        
        # Calculate normalized entropy of POS distribution
        total_tags = len(pos_tags)
        entropy = 0
        for count in tag_freq.values():
            prob = count / total_tags
            entropy -= prob * math.log2(prob)
            
        # Normalize to 0-1 scale (assuming max entropy ≈ 4.5 for English POS tags)
        return min(entropy / 4.5, 1.0)

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity + 0.5  # Convert -1 to 1 range to 0 to 1
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
            
        if metrics["pos_distribution"] > self.thresholds["pos_distribution"]:
            explanation.append("- Very uniform parts of speech patterns")
        else:
            explanation.append("- Natural variation in parts of speech")
        
        # Add metric values
        explanation.append("\nDetailed metrics:")
        for metric, value in metrics.items():
            formatted_metric = metric.replace('_', ' ').title()
            explanation.append(f"- {formatted_metric}: {value:.3f}")
        
        return "\n".join(explanation)
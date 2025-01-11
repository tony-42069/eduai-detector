# src/eduai_detector/core/detector.py

from typing import Dict, List, Tuple
import numpy as np
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

        # Common transition words
        self.transition_words = set([
            'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
            'thus', 'consequently', 'meanwhile', 'subsequently', 'alternatively',
            'although', 'despite', 'whereas', 'while', 'hence', 'additionally',
            'similarly', 'likewise', 'in contrast', 'conversely', 'instead',
            'rather', 'still', 'yet', 'anyway', 'besides', 'further', 'next',
            'finally', 'lastly', 'first', 'second', 'third', 'then', 'also'
        ])

    def detect(self, text: str) -> Tuple[bool, Dict[str, float], str, float]:
        """Main detection method that determines if text is AI-generated."""
        if not text.strip():
            return False, {}, "No text provided", 0.0

        # Calculate metrics
        metrics = self._calculate_metrics(text)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(metrics)
        
        # More conservative threshold (70%) for classification
        is_ai_generated = confidence_score > 0.70  # Increased threshold
        
        explanation = self._generate_explanation(metrics, confidence_score, is_ai_generated)
        
        return is_ai_generated, metrics, explanation, confidence_score

    def _calculate_metrics(self, text: str) -> Dict[str, float]:
        """Calculate all metrics for the given text."""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        metrics = {
            "repetition_score": float(self._calculate_repetition(words)),
            "entropy_score": float(self._calculate_entropy(text)),
            "complexity_score": float(self._calculate_complexity(text)),
            "vocabulary_diversity": float(self._calculate_vocabulary_diversity(words)),
            "sentence_variation": float(self._calculate_sentence_variation(sentences)),
            "transition_patterns": float(self._calculate_transition_patterns(words)),
            "pos_distribution": float(self._calculate_pos_distribution(text)),
            "readability": float(self._calculate_readability(text))
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

    def _calculate_sentence_variation(self, sentences: List[str]) -> float:
        """Calculate variation in sentence structure and length."""
        if not sentences:
            return 0.0
        
        # Calculate sentence lengths
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        
        # Calculate coefficient of variation (standard deviation / mean)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        return std_length / mean_length if mean_length > 0 else 0.0

    def _calculate_transition_patterns(self, words: List[str]) -> float:
        """Analyze the use of transition words and phrases."""
        if not words:
            return 0.0
        
        transition_count = sum(1 for word in words if word.lower() in self.transition_words)
        return transition_count / len(words)

    def _calculate_pos_distribution(self, text: str) -> float:
        """Calculate distribution of parts of speech."""
        if not text.strip():
            return 0.0
        
        # Get POS tags
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Count POS tag frequencies
        pos_counts = Counter(tag for _, tag in pos_tags)
        total_tags = len(pos_tags)
        
        # Calculate distribution entropy
        entropy = -sum((count/total_tags) * math.log2(count/total_tags) 
                      for count in pos_counts.values())
        
        # Normalize to [0,1]
        return min(entropy / 4.0, 1.0)  # 4.0 is approximate max entropy for POS

    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        if not text.strip():
            return 0.0
        
        blob = TextBlob(text)
        sentences = len(blob.sentences)
        words = len(blob.words)
        syllables = sum(self._count_syllables(word) for word in blob.words)
        
        if words == 0 or sentences == 0:
            return 0.0
        
        # Flesch Reading Ease score
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0.0, min(100.0, score))

    def _count_syllables(self, word: str) -> int:
        """Helper method to count syllables in a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
            
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2:
            count += 1
        if count == 0:
            count = 1
            
        return count

    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall confidence score using weighted metrics."""
        score = 0.0
        
        for metric_name, metric_value in metrics.items():
            threshold = self.thresholds[metric_name]
            weight = self.metric_weights[metric_name]
            
            # Normalize and adjust direction based on metric type
            if metric_name in ['vocabulary_diversity', 'sentence_variation', 'transition_patterns']:
                # For these metrics, lower values are more AI-like
                normalized_value = 1.0 - (metric_value / threshold) if threshold > 0 else 0.0
            else:
                # For other metrics, higher values are more AI-like
                normalized_value = metric_value / threshold if threshold > 0 else 0.0
            
            # Clip normalized value to [0, 1] range
            normalized_value = max(0.0, min(1.0, normalized_value))
            
            # Add weighted contribution
            score += normalized_value * weight
        
        # Final score should be between 0 and 1
        return max(0.0, min(1.0, score))

    def _generate_explanation(self, metrics: Dict[str, float], 
                            confidence_score: float, 
                            is_ai_generated: bool) -> str:
        """Generate a detailed explanation of the analysis."""
        parts = []
        
        confidence_percent = confidence_score * 100
        parts.append(f"Analysis Confidence: {confidence_percent:.1f}%\n")
        
        if is_ai_generated:
            parts.append("This text shows characteristics of AI-generated content:")
            
            if metrics["repetition_score"] > self.thresholds["repetition_score"]:
                parts.append("- Higher than normal repetition patterns")
            if metrics["entropy_score"] > self.thresholds["entropy_score"]:
                parts.append("- Unusual entropy patterns in text structure")
            if metrics["complexity_score"] > self.thresholds["complexity_score"]:
                parts.append("- Higher than typical text complexity")
            if metrics["vocabulary_diversity"] < self.thresholds["vocabulary_diversity"]:
                parts.append("- Limited vocabulary diversity")
            if metrics["sentence_variation"] < self.thresholds["sentence_variation"]:
                parts.append("- Uniform sentence structures")
            if metrics["transition_patterns"] < self.thresholds["transition_patterns"]:
                parts.append("- Limited use of transition words")
            if metrics["pos_distribution"] > self.thresholds["pos_distribution"]:
                parts.append("- Unusual distribution of parts of speech")
            if metrics["readability"] > self.thresholds["readability"]:
                parts.append("- Unusually consistent readability level")
        else:
            parts.append("This text shows characteristics more typical of human-written content:")
            if confidence_score < 0.35:
                parts.append("- Natural variation in writing patterns")
                parts.append("- Diverse vocabulary usage")
                parts.append("- Varied sentence structures")
                parts.append("- Natural transition usage")
                parts.append("- Typical language patterns")
            else:
                parts.append("- Some AI-like patterns detected but not enough for confident classification")
                parts.append("- Mixed characteristics of both human and AI writing")
        
        parts.append("\nDetailed metrics (with thresholds and weights):")
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            threshold = self.thresholds[key]
            weight = self.metric_weights[key]
            
            # Add interpretation hint
            if key in ['vocabulary_diversity', 'sentence_variation', 'transition_patterns']:
                hint = "Lower values suggest AI"
            else:
                hint = "Higher values suggest AI"
            
            parts.append(f"- {formatted_key}: {value:.3f} (threshold: {threshold:.2f}, weight: {weight:.2f}, {hint})")
        
        return "\n".join(parts)
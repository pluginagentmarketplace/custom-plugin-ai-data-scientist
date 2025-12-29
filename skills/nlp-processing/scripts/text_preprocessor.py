#!/usr/bin/env python3
"""
NLP Text Preprocessing Pipeline
Comprehensive text cleaning and preprocessing utilities
"""

import re
import unicodedata
from typing import List, Optional, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = False
    remove_extra_whitespace: bool = True
    min_token_length: int = 2
    expand_contractions: bool = True
    unicode_normalize: bool = True


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline."""

    # Common contractions
    CONTRACTIONS = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._build_pipeline()

    def _build_pipeline(self):
        """Build preprocessing pipeline based on config."""
        self.pipeline: List[Callable[[str], str]] = []

        if self.config.unicode_normalize:
            self.pipeline.append(self._normalize_unicode)

        if self.config.lowercase:
            self.pipeline.append(str.lower)

        if self.config.remove_html:
            self.pipeline.append(self._remove_html)

        if self.config.remove_urls:
            self.pipeline.append(self._remove_urls)

        if self.config.remove_emails:
            self.pipeline.append(self._remove_emails)

        if self.config.expand_contractions:
            self.pipeline.append(self._expand_contractions)

        if self.config.remove_numbers:
            self.pipeline.append(self._remove_numbers)

        if self.config.remove_punctuation:
            self.pipeline.append(self._remove_punctuation)

        if self.config.remove_extra_whitespace:
            self.pipeline.append(self._remove_extra_whitespace)

    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline to text."""
        if not text:
            return ""

        for step in self.pipeline:
            text = step(text)

        return text.strip()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def _remove_html(text: str) -> str:
        """Remove HTML tags."""
        clean = re.compile('<.*?>')
        return re.sub(clean, ' ', text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, ' ', text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        """Remove email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, ' ', text)

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions (e.g., don't -> do not)."""
        for contraction, expansion in self.CONTRACTIONS.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _remove_numbers(text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', ' ', text)

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation."""
        return re.sub(r'[^\w\s]', ' ', text)

    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace."""
        return ' '.join(text.split())


class Tokenizer:
    """Simple tokenization utilities."""

    @staticmethod
    def word_tokenize(text: str) -> List[str]:
        """Basic word tokenization."""
        return text.split()

    @staticmethod
    def sentence_tokenize(text: str) -> List[str]:
        """Basic sentence tokenization."""
        # Simple sentence boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def ngrams(tokens: List[str], n: int) -> List[tuple]:
        """Generate n-grams from tokens."""
        return list(zip(*[tokens[i:] for i in range(n)]))


class TextStatistics:
    """Calculate text statistics."""

    @staticmethod
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    @staticmethod
    def char_count(text: str, include_spaces: bool = False) -> int:
        """Count characters in text."""
        if include_spaces:
            return len(text)
        return len(text.replace(' ', ''))

    @staticmethod
    def sentence_count(text: str) -> int:
        """Count sentences in text."""
        return len(Tokenizer.sentence_tokenize(text))

    @staticmethod
    def avg_word_length(text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)

    @staticmethod
    def vocabulary_size(text: str) -> int:
        """Count unique words."""
        return len(set(text.lower().split()))

    @staticmethod
    def lexical_diversity(text: str) -> float:
        """Calculate type-token ratio (lexical diversity)."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)


def main():
    """Demo NLP preprocessing."""
    print("NLP Text Preprocessing Demo")
    print("=" * 50)

    # Sample text with various issues
    sample_text = """
    <p>Hello! Check out https://example.com for more info!</p>

    I can't believe it's already 2024... Contact us at info@example.com.

    This is AMAZING!!! Don't you think so?

    We've got 100+ products    with    extra   spaces.
    """

    print("Original text:")
    print(sample_text)
    print("-" * 50)

    # Create preprocessor
    config = PreprocessingConfig(
        lowercase=True,
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        remove_numbers=False,
        remove_punctuation=False,
        expand_contractions=True
    )

    preprocessor = TextPreprocessor(config)
    cleaned = preprocessor.preprocess(sample_text)

    print("\nCleaned text:")
    print(cleaned)
    print("-" * 50)

    # Text statistics
    print("\nText Statistics:")
    print(f"  Word count: {TextStatistics.word_count(cleaned)}")
    print(f"  Sentence count: {TextStatistics.sentence_count(cleaned)}")
    print(f"  Avg word length: {TextStatistics.avg_word_length(cleaned):.2f}")
    print(f"  Vocabulary size: {TextStatistics.vocabulary_size(cleaned)}")
    print(f"  Lexical diversity: {TextStatistics.lexical_diversity(cleaned):.3f}")

    print("\n[SUCCESS] NLP preprocessing complete!")


if __name__ == '__main__':
    main()

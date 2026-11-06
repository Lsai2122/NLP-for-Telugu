# -*- coding: utf-8 -*-
"""
Telugu Text Normalization and Cleaning Module.
Handles Unicode NFC normalization, Indic NLP library normalization,
Telugu script ratio detection, and noise removal.
"""
import re
import unicodedata
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

class TeluguPreprocessor:
    """
    A robust preprocessor for Telugu text.
    Handles Unicode NFC canonical composition, ZWJ/ZWNJ cleanup,
    script detection based on Telugu character density, and noise filtering.
    """

    def __init__(self):
        factory = IndicNormalizerFactory()
        self.normalizer = factory.get_normalizer("te")
        # Pre-compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.html_pattern = re.compile(r'<.*?>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.telugu_range_pattern = re.compile(r'[\u0C00-\u0C7F]')
        self.punctuation_pattern = re.compile(r'[^\w\s\u0C00-\u0C7F]', re.UNICODE)

    def normalize(self, text: str) -> str:
        """
        Normalizes Telugu text:
        1. Unicode NFC composition (canonical equivalence)
        2. Indic NLP normalization (cleans invisible zero-width characters, fixes matras)
        """
        if not text:
            return ""
        
        # Step 1: Unicode NFC normalization
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Indic NLP library normalization
        text = self.normalizer.normalize(text)
        
        return text

    def clean(self, text: str) -> str:
        """
        Cleans the input text by stripping URLs, HTML tags, and redundant whitespace.
        """
        if not text:
            return ""
            
        # Remove URLs and links
        text = self.url_pattern.sub('', text)
        
        # Remove HTML markup
        text = self.html_pattern.sub('', text)
        
        # Normalize whitespace (replace tabs, newlines, multiple spaces with a single space)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text

    def is_telugu(self, text: str, threshold: float = 0.4) -> bool:
        """
        Determines whether the given text is predominantly written in Telugu script.
        Uses a character density threshold (default 40% of non-whitespace characters).
        This distinguishes Telugu script from Tenglish / English text.
        """
        if not text:
            return False
            
        non_space_chars = text.replace(' ', '')
        if not non_space_chars:
            return False
            
        telugu_count = len(self.telugu_range_pattern.findall(non_space_chars))
        ratio = telugu_count / len(non_space_chars)
        return ratio >= threshold

    def has_telugu_characters(self, text: str) -> bool:
        """
        Returns True if the text contains at least one Telugu Unicode character.
        """
        if not text:
            return False
        return bool(self.telugu_range_pattern.search(text))

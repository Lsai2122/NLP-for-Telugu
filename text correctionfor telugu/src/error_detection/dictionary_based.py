# -*- coding: utf-8 -*-
"""
Dictionary-Based Error Detection Module for Telugu.
Checks whether words exist in the curated 100K Telugu vocabulary.
Properly handles Telugu Unicode combining marks and punctuation stripping.
"""
import os
import json
import re

class DictionaryErrorDetector:
    """
    Detects spelling and out-of-vocabulary (OOV) errors by consulting a
    curated Telugu vocabulary frequency dictionary.
    """
    
    def __init__(self, vocab_path: str = None):
        self.vocab = set()
        self.word_freqs = {}
        # Pattern to isolate Telugu letters/digits vs punctuation
        self.punctuation_cleaner = re.compile(r'^[^\u0C00-\u0C7F\w]+|[^\u0C00-\u0C7F\w]+$', re.UNICODE)
        self.telugu_char_pattern = re.compile(r'[\u0C00-\u0C7F]')
        
        # Default path to processed telugu_vocab.json
        if vocab_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "processed", "telugu_vocab.json"
            )
            if os.path.exists(default_path):
                vocab_path = default_path
                
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)

    def load_vocab(self, path: str):
        """
        Loads vocabulary from a JSON file (dict of {word: frequency} or list of words).
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.word_freqs = data
                    self.vocab = set(data.keys())
                elif isinstance(data, list):
                    self.vocab = set(data)
                    self.word_freqs = {w: 1 for w in data}
            print(f"[DictionaryErrorDetector] Successfully loaded {len(self.vocab):,} vocabulary words from {path}.")
        except Exception as e:
            print(f"[DictionaryErrorDetector] Error loading vocabulary from {path}: {e}")

    def clean_token(self, token: str) -> str:
        """
        Strips surrounding punctuation from a token while preserving internal Telugu characters.
        """
        return self.punctuation_cleaner.sub('', token).strip()

    def is_word_error(self, token: str) -> bool:
        """
        Checks if a single token is a spelling error:
        1. Pure punctuation or numbers are NOT errors.
        2. Non-Telugu tokens are ignored here (handled by script detector).
        3. If it contains Telugu characters and is NOT in vocabulary -> Error.
        """
        cleaned = self.clean_token(token)
        if not cleaned:
            return False
            
        # If token does not contain any Telugu letters, do not flag as Telugu spelling error
        if not self.telugu_char_pattern.search(cleaned):
            return False
            
        # If vocabulary is loaded, check membership
        if self.vocab:
            return cleaned not in self.vocab
            
        return False

    def detect_errors(self, sentence: str, tokenizer=None) -> list:
        """
        Detects words in the sentence that are not in the dictionary.
        Returns a list of dicts:
        [{'word': word, 'clean_word': clean_word, 'position': idx, 'is_error': bool}, ...]
        """
        if tokenizer is not None and hasattr(tokenizer, 'word_tokenize'):
            words = tokenizer.word_tokenize(sentence)
        else:
            words = sentence.split()
            
        results = []
        for idx, word in enumerate(words):
            cleaned = self.clean_token(word)
            has_error = self.is_word_error(word)
            
            results.append({
                "word": word,
                "clean_word": cleaned,
                "position": idx,
                "is_error": has_error
            })
            
        return results

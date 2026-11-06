# -*- coding: utf-8 -*-
from indicnlp.tokenize import indic_tokenize
from indicnlp.syllable import syllabifier

class TeluguTokenizer:
    """
    A class for tokenizing Telugu text at the word and syllable (Akshara) levels.
    """
    
    def __init__(self):
        self.lang = 'te'

    def word_tokenize(self, text: str) -> list:
        """
        Tokenizes text into words using indic_tokenize.
        """
        if not text:
            return []
        return indic_tokenize.trivial_tokenize(text, self.lang)

    def syllable_tokenize(self, word: str) -> list:
        """
        Tokenizes a word into syllables (Aksharas).
        """
        if not word:
            return []
        return syllabifier.orthographic_syllabify(word, self.lang)

    def tokenize_sentence(self, text: str) -> list:
        """
        Tokenizes text into a list of words.
        """
        return self.word_tokenize(text)

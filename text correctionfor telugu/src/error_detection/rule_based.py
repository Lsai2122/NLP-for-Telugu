# -*- coding: utf-8 -*-
import re

class RuleBasedErrorDetector:
    """
    Rule-based error detector for Telugu text.
    """
    
    def __init__(self):
        # Telugu Unicode ranges and rules
        self.vowels = r'[\u0C05-\u0C14\u0C60\u0C61]'
        self.consonants = r'[\u0C15-\u0C39\u0C58\u0C59]'
        self.matras = r'[\u0C3E-\u0C4C\u0C55\u0C56\u0C62\u0C63]'
        self.halant = r'\u0C4D'
        
    def detect_errors(self, sentence: str) -> list:
        """
        Detects rule-based errors in a sentence.
        """
        words = sentence.split()
        errors = []
        
        for idx, word in enumerate(words):
            is_error = False
            
            # Rule 1: Two consecutive matras on a consonant
            if re.search(f"{self.consonants}{self.matras}{{2,}}", word):
                is_error = True
                
            # Rule 2: Matra applied directly to independent vowel
            if re.search(f"{self.vowels}{self.matras}", word):
                is_error = True
                
            if is_error:
                errors.append({
                    "word": word,
                    "position": idx,
                    "error_type": "rule_violation"
                })
                
        return errors

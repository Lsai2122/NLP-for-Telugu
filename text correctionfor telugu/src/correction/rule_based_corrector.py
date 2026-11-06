# -*- coding: utf-8 -*-
"""
Rule-Based Telugu Text Corrector.
Leverages Telugu linguistic phonology rules (matra confusions, consonant confusions)
and validates generated candidates against the Telugu vocabulary dictionary.
"""
import os
import json
import re

class RuleBasedCorrector:
    """
    Applies deterministic phonetic and orthographic rules tailored to Telugu:
    1. Matra Length Confusions (short <-> long vowels)
    2. Consonant Confusions (sibilants, retroflex nasals, voiced/unvoiced stops)
    3. Direct dictionary validation: Only suggests a rule-based change if the
       resulting word is attested in the vocabulary.
    """
    
    def __init__(self, vocab_path: str = None):
        self.vocab = set()
        self.word_freqs = {}
        
        # Primary Telugu Matra Swap Rules (both directions)
        self.matra_rules = {
            'ి': ['ీ'],
            'ీ': ['ి'],
            'ు': ['ూ'],
            'ూ': ['ు'],
            'ె': ['ే'],
            'ే': ['ె'],
            'ొ': ['ో'],
            'ో': ['ొ']
        }
        
        # Primary Telugu Consonant Confusion Rules
        self.consonant_rules = {
            'ణ': ['న'],
            'న': ['ణ'],
            'శ': ['ష', 'స'],
            'ష': ['శ', 'స'],
            'స': ['శ', 'ష'],
            'ట': ['డ'],
            'డ': ['ట'],
            'బ': ['వ'],
            'వ': ['బ']
        }
        
        # Common irregular / colloquial word mappings
        self.common_word_fixes = {
            "తేలుగు": "తెలుగు",
            "తెలుగి": "తెలుగు",
            "బాష": "భాష",
            "భాశ": "భాష",
            "చాల": "చాలా",
            "అందమైణది": "అందమైనది",
            "విద్యార్ధి": "విద్యార్థి",
            "పరిక్ష": "పరీక్ష",
            "పరిక్షలు": "పరీక్షలు",
            "ధన్యవాదాలు": "ధన్యవాదాలు",
            "నమసకారం": "నమస్కారం"
        }
        
        if vocab_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "processed", "telugu_vocab.json"
            )
            if os.path.exists(default_path):
                vocab_path = default_path
                
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)

    def load_vocab(self, path: str):
        """Loads vocabulary dictionary for validation."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.word_freqs = data
                    self.vocab = set(data.keys())
                elif isinstance(data, list):
                    self.vocab = set(data)
                    self.word_freqs = {w: 1 for w in data}
        except Exception as e:
            print(f"[RuleBasedCorrector] Error loading vocabulary from {path}: {e}")

    def generate_rule_candidates(self, word: str) -> list:
        """
        Generates potential candidate words by applying 1-hop Telugu phonetic rules.
        """
        candidates = set()
        chars = list(word)
        
        # 1. Try single matra substitutions
        for i, ch in enumerate(chars):
            if ch in self.matra_rules:
                for replacement in self.matra_rules[ch]:
                    new_word = "".join(chars[:i] + [replacement] + chars[i+1:])
                    candidates.add(new_word)
                    
        # 2. Try single consonant substitutions
        for i, ch in enumerate(chars):
            if ch in self.consonant_rules:
                for replacement in self.consonant_rules[ch]:
                    new_word = "".join(chars[:i] + [replacement] + chars[i+1:])
                    candidates.add(new_word)
                    
        # Filter: only keep candidates that are attested in the Telugu vocabulary
        valid_candidates = [c for c in candidates if c in self.vocab]
        
        # Rank by corpus frequency
        valid_candidates.sort(key=lambda c: self.word_freqs.get(c, 1), reverse=True)
        return valid_candidates

    def correct_word(self, word: str) -> str:
        """
        Applies rules to correct a single word.
        If word is already valid or no rule produces a valid word, returns original.
        """
        clean_word = word.strip(" ,.!?;:\"'()[]{}")
        if not clean_word:
            return word
            
        # If the word is already valid, do not corrupt it
        if clean_word in self.vocab:
            return word
            
        # Check direct lookup fixes first
        if clean_word in self.common_word_fixes:
            fixed = self.common_word_fixes[clean_word]
            return word.replace(clean_word, fixed, 1)
            
        # Try generating candidates using linguistic rules
        candidates = self.generate_rule_candidates(clean_word)
        if candidates:
            best_candidate = candidates[0]
            return word.replace(clean_word, best_candidate, 1)
            
        return word

    def correct_sentence(self, sentence: str) -> str:
        """
        Corrects an entire sentence using Telugu phonetic and orthographic rules.
        """
        if not sentence:
            return ""
            
        words = sentence.split()
        corrected_words = [self.correct_word(w) for w in words]
        return " ".join(corrected_words)

# -*- coding: utf-8 -*-
"""
Tenglish (Romanized Telugu) to Telugu Script Conversion Module.
Leverages Google Dakshina transliteration lexicon (69,980 pairs)
with phonetic fallback heuristics.
"""
import os
import json
import re

class TenglishConverter:
    """
    Converts Romanized Telugu (Tenglish) words and sentences to Telugu script.
    Uses Dakshina transliteration dataset for direct mapping with top candidate support.
    """
    
    def __init__(self, lexicon_path: str = None, vocab_path: str = None):
        self.lexicon = {}  # {roman_word: [telugu_candidates]}
        self.word_freqs = {}  # {telugu_word: corpus frequency} for tie-breaking

        # Default path to processed Dakshina lexicon
        if lexicon_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "processed", "dakshina_transliteration.json"
            )
            if os.path.exists(default_path):
                lexicon_path = default_path

        # Corpus frequency dictionary used to disambiguate multi-candidate roman forms.
        # Dakshina attests ~2.8 romanizations per word type, so a bare "first candidate"
        # lookup silently returns rare forms (e.g. 'ela' -> 'ఈల' instead of 'ఏలా').
        if vocab_path is None:
            vocab_default = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "processed", "telugu_vocab.json"
            )
            if os.path.exists(vocab_default):
                vocab_path = vocab_default

        if vocab_path and os.path.exists(vocab_path):
            self._load_frequencies(vocab_path)

        if lexicon_path and os.path.exists(lexicon_path):
            self.load_lexicon(lexicon_path)
        else:
            # Fallback starter lexicon if file not found
            self._load_fallback_lexicon()

        # Always merge the curated fallback entries for any word Dakshina does not
        # attest. Previously these were applied ONLY when the lexicon file was
        # entirely absent, so curated coverage for common colloquial forms never
        # took effect once the real Dakshina lexicon was present.
        self._load_fallback_lexicon()

    def _load_frequencies(self, json_path: str):
        """Loads the Telugu corpus frequency map used to rank ambiguous candidates."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.word_freqs = data
        except Exception as e:
            print(f"[TenglishConverter] Warning: could not load frequencies from {json_path}: {e}")

    def _best_candidate(self, candidates: list) -> str:
        """
        Chooses among multiple attested Telugu forms for one romanized word.
        Ranks by corpus frequency (descending), falling back to lexicon order when a
        candidate is absent from the frequency dictionary.
        """
        if len(candidates) == 1:
            return candidates[0]
        return max(
            candidates,
            key=lambda w: (self.word_freqs.get(w, 0), -candidates.index(w))
        )

    def load_lexicon(self, json_path: str):
        """
        Loads Roman -> Telugu mappings from the processed Dakshina JSON.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
                
            for item in pairs:
                roman = item['roman'].strip().lower()
                telugu = item['telugu'].strip()
                if roman not in self.lexicon:
                    self.lexicon[roman] = []
                if telugu not in self.lexicon[roman]:
                    self.lexicon[roman].append(telugu)
                    
            print(f"[TenglishConverter] Loaded {len(self.lexicon):,} unique Roman words into transliteration lexicon.")
        except Exception as e:
            print(f"[TenglishConverter] Error loading lexicon from {json_path}: {e}")
            self._load_fallback_lexicon()

    def _load_fallback_lexicon(self):
        fallback = {
            "namaskaram": ["నమస్కారం"],
            "ela": ["ఎలా"],
            "unnaru": ["ఉన్నారు"],
            "bagunnara": ["బాగున్నారా"],
            "bagunnanu": ["బాగున్నాను"],
            "nenu": ["నేను"],
            "meeru": ["మీరు"],
            "chala": ["చాలా"],
            "telugu": ["తెలుగు"],
            "bhasha": ["భాష"],
            "andaru": ["అందరూ"],
            "dhanyavadalu": ["ధన్యవాదాలు"],
            "em": ["ఏం", "ఎం"],
            "chestunnav": ["చేస్తున్నావ్", "చేస్తున్నావ"],
            "ekkada": ["ఎక్కడ"],
            "eppudu": ["ఎప్పుడు"],
            "enduku": ["ఎందుకు"],
            "evaru": ["ఎవరు"]
        }
        for k, v in fallback.items():
            if k not in self.lexicon:
                self.lexicon[k] = v

    def convert_word(self, word: str) -> str:
        """
        Converts a single Romanized word to Telugu.
        Punctuation is preserved around the word.
        """
        clean_word = word.strip().lower()
        # Separate leading/trailing punctuation
        match = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", clean_word, re.UNICODE)
        if match:
            prefix, core, suffix = match.groups()
            if core in self.lexicon:
                return f"{prefix}{self._best_candidate(self.lexicon[core])}{suffix}"
            return word
        return self._best_candidate(self.lexicon.get(clean_word, [word]))

    def convert(self, roman_text: str) -> str:
        """
        Converts a Tenglish sentence into Telugu script word by word.
        """
        if not roman_text:
            return ""
            
        words = roman_text.split()
        converted_words = [self.convert_word(w) for w in words]
        return " ".join(converted_words)

    def get_candidates(self, word: str, top_k: int = 5) -> list:
        """
        Returns the top_k transliteration candidate Telugu words.
        """
        clean_word = word.strip().lower()
        if clean_word in self.lexicon:
            return self.lexicon[clean_word][:top_k]
        return [word]

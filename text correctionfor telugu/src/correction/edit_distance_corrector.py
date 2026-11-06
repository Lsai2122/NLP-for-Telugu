# -*- coding: utf-8 -*-
"""
Optimized Edit-Distance Corrector for Telugu Text.
Uses Levenshtein distance with length-bucketed pruning, early score cutoffs,
and corpus word frequency weighting.
"""
import os
import json
from collections import defaultdict
from rapidfuzz.distance import Levenshtein

class EditDistanceCorrector:
    """
    Finds the closest dictionary correction for misspelled Telugu words.
    Key optimizations:
    1. Length-based bucketing: Avoids comparing words with impossible length differentials.
    2. Score cutoffs: Short-circuits Levenshtein calculation when distance exceeds max_dist.
    3. Frequency weighting: Breaks edit-distance ties using real corpus frequencies.
    """
    
    def __init__(self, vocab_path: str = None, max_edit_distance: int = 2):
        self.vocab = set()
        self.word_freqs = {}
        self.vocab_by_len = defaultdict(list)
        self.max_dist = max_edit_distance
        
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
        Loads vocabulary and builds length buckets.
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
                    
            self._build_buckets()
            print(f"[EditDistanceCorrector] Loaded {len(self.vocab):,} words into length buckets.")
        except Exception as e:
            print(f"[EditDistanceCorrector] Error loading vocabulary: {e}")

    def set_vocab(self, vocab, freqs=None):
        """
        Sets vocabulary in-memory directly.
        """
        if isinstance(vocab, dict):
            self.word_freqs = vocab
            self.vocab = set(vocab.keys())
        else:
            self.vocab = set(vocab)
            self.word_freqs = freqs or {w: 1 for w in vocab}
        self._build_buckets()

    def _build_buckets(self):
        self.vocab_by_len.clear()
        # Iterate a deterministic order. self.vocab is a set, and Python randomises
        # str hashing per process (PYTHONHASHSEED), so iterating the set directly
        # made bucket order - and therefore tie-breaking - differ between runs.
        # Ranking by (-freq, word) makes ties resolve to the most frequent word,
        # then alphabetically, which is both stable and a better tie-break.
        for word in sorted(self.vocab, key=lambda w: (-self.word_freqs.get(w, 1), w)):
            self.vocab_by_len[len(word)].append(word)

    def get_candidates(self, word: str, top_k: int = 5) -> list:
        """
        Finds the closest valid words in vocabulary within max_edit_distance.
        Returns list of candidate words sorted by (distance ASC, frequency DESC).
        """
        if not self.vocab or not word:
            return []
            
        w_len = len(word)
        candidates = []
        
        # Search length buckets within [w_len - max_dist, w_len + max_dist]
        min_len = max(1, w_len - self.max_dist)
        max_len = w_len + self.max_dist
        
        for l in range(min_len, max_len + 1):
            for v_word in self.vocab_by_len[l]:
                # Rapidfuzz score_cutoff: aborts if distance > max_dist
                dist = Levenshtein.distance(word, v_word, score_cutoff=self.max_dist)
                if dist <= self.max_dist:
                    freq = self.word_freqs.get(v_word, 1)
                    candidates.append((v_word, dist, freq))
                    
        if not candidates:
            return []
            
        # Primary sort: edit distance (lower is closer)
        # Secondary sort: corpus frequency (higher is more common)
        # Tertiary sort: alphabetical, so equal (distance, frequency) pairs resolve
        # deterministically instead of depending on set iteration order.
        candidates.sort(key=lambda x: (x[1], -x[2], x[0]))
        
        return [w for w, d, f in candidates[:top_k]]

    def correct_word(self, word: str) -> str:
        """
        Returns the single most probable dictionary candidate for a misspelled word.
        If no candidate is found within max_edit_distance, returns original word.
        """
        candidates = self.get_candidates(word, top_k=1)
        if candidates:
            return candidates[0]
        return word

    def correct_sentence(self, sentence: str, detector=None, tokenizer=None) -> str:
        """
        Corrects errors in a sentence.
        Uses detector if provided; otherwise checks words against internal vocabulary.
        """
        if tokenizer is not None and hasattr(tokenizer, 'word_tokenize'):
            tokens = tokenizer.word_tokenize(sentence)
            # A real tokenizer splits punctuation into standalone tokens.
            use_smart_join = True
        else:
            tokens = sentence.split()
            # Whitespace split keeps punctuation attached to its word, so a plain
            # space join is already correct here and must not be altered.
            use_smart_join = False
            
        corrected_tokens = []
        
        for token in tokens:
            # Clean token to test error status
            clean_token = token.strip(" ,.!?;:\"'()[]{}")
            if not clean_token:
                corrected_tokens.append(token)
                continue
                
            # If word is in vocabulary, leave it untouched
            if clean_token in self.vocab:
                corrected_tokens.append(token)
                continue
                
            # If detector is provided, check if it flagged this token
            if detector is not None and not detector.is_word_error(token):
                corrected_tokens.append(token)
                continue
                
            # Find correction for core word
            best_cand = self.correct_word(clean_token)
            
            # Preserve surrounding punctuation
            if best_cand != clean_token:
                # Replace core while preserving surrounding punctuation
                corrected_word = token.replace(clean_token, best_cand, 1)
                corrected_tokens.append(corrected_word)
            else:
                corrected_tokens.append(token)
                
        return self._detokenize(corrected_tokens) if use_smart_join else " ".join(corrected_tokens)

    @staticmethod
    def _detokenize(tokens: list) -> str:
        """
        Rejoins tokens, gluing punctuation back onto the preceding word.

        A plain '" ".join(...)' is wrong whenever a real tokenizer has split
        `తరలించామనీ,` into `['తరలించామనీ', ',']` - rejoining with spaces produced
        `తరలించామని ,`, which turns one word into two and roughly doubles the
        measured WER. Standalone punctuation carries no word content, so it must
        attach to the previous token rather than sit behind a space.
        """
        out = []
        for tok in tokens:
            has_word_char = any(ch.isalnum() for ch in tok)
            if out and not has_word_char:
                out[-1] += tok          # punctuation glues to previous token
            else:
                out.append(tok)
        return " ".join(out)

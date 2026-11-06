# -*- coding: utf-8 -*-
from collections import Counter
import matplotlib.pyplot as plt

class ErrorAnalyzer:
    """
    Analyzes and classifies errors between original and corrected text.
    """
    
    def __init__(self):
        self.error_counts = Counter()
        self.matras = set(list("ాిీుూెేైొోౌంః"))
        
    def classify_error(self, original_word: str, corrected_word: str) -> str:
        """
        Classifies the type of error between two words.
        """
        if original_word == corrected_word:
            return "none"
            
        # Simple heuristics for classification
        diff_len = len(original_word) - len(corrected_word)
        
        if diff_len > 0:
            return "deletion"
        elif diff_len < 0:
            return "insertion"
            
        # If lengths are same, check if only matras differ
        orig_chars = set(original_word)
        corr_chars = set(corrected_word)
        
        diff_chars = orig_chars.symmetric_difference(corr_chars)
        
        if all(c in self.matras for c in diff_chars):
            return "matra"
            
        return "consonant_or_other"
        
    def analyze_dataset(self, pairs: list):
        """
        Analyzes a dataset of (original, corrected) pairs.
        pairs: list of tuples (original_sentence, corrected_sentence)
        """
        self.error_counts.clear()
        
        for orig, corr in pairs:
            orig_words = orig.split()
            corr_words = corr.split()
            
            # Simplified: assuming 1:1 mapping for analysis
            min_len = min(len(orig_words), len(corr_words))
            for i in range(min_len):
                err_type = self.classify_error(orig_words[i], corr_words[i])
                if err_type != "none":
                    self.error_counts[err_type] += 1
                    
        return dict(self.error_counts)
        
    def plot_error_distribution(self):
        """
        Plots the distribution of error types.
        """
        if not self.error_counts:
            print("No data to plot.")
            return
            
        labels = list(self.error_counts.keys())
        counts = list(self.error_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')
        plt.title('Error Type Distribution in Telugu Text')
        plt.xlabel('Error Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

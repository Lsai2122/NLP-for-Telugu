# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings

class NeuralErrorDetector:
    """
    Neural error detector using MuRIL MLM to score words in context.
    Note: It is recommended to run this on a GPU (e.g., in Google Colab).
    """
    
    def __init__(self, model_name: str = "google/muril-base-cased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load {model_name}: {e}")
            self.model = None
            self.tokenizer = None
            
    def detect_errors(self, sentence: str, threshold: float = 0.01) -> list:
        """
        Scores each word in context and flags low-probability words as errors.
        """
        if not self.model or not self.tokenizer:
            return []
            
        words = sentence.split()
        results = []
        
        for i, word in enumerate(words):
            # Create masked sentence
            masked_words = words[:]
            masked_words[i] = self.tokenizer.mask_token
            masked_sentence = " ".join(masked_words)
            
            inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            
            if len(mask_token_index) > 0:
                mask_token_logits = logits[0, mask_token_index[0], :]
                probs = torch.softmax(mask_token_logits, dim=-1)
                
                # Get the probability of the actual word tokens
                word_tokens = self.tokenizer.tokenize(word)
                if word_tokens:
                    first_token_id = self.tokenizer.convert_tokens_to_ids(word_tokens[0])
                    word_prob = probs[first_token_id].item()
                    
                    if word_prob < threshold:
                        results.append({
                            "word": word,
                            "position": i,
                            "probability": word_prob,
                            "is_error": True
                        })
                        
        return results

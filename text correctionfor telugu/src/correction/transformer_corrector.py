# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

class TransformerCorrector:
    """
    Transformer-based text corrector for Telugu.
    Uses models like IndicBART or mT5.
    """
    
    def __init__(self, model_path: str = "ai4bharat/IndicBART"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model(model_path)
        
    def load_model(self, path: str):
        """
        Loads the Seq2Seq model from local path or HF hub.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load transformer model from {path}: {e}")
            
    def correct(self, text: str, num_beams: int = 5) -> str:
        """
        Generates corrections using beam search for a single text.
        """
        if not self.model or not self.tokenizer:
            return text
            
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=128, 
                num_beams=num_beams, 
                early_stopping=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def batch_correct(self, texts: list, num_beams: int = 5) -> list:
        """
        Generates corrections for a batch of texts.
        """
        if not self.model or not self.tokenizer:
            return texts
            
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, max_length=128, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=num_beams,
                early_stopping=True
            )
            
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

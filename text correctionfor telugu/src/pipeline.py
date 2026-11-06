# -*- coding: utf-8 -*-
"""
End-to-End Telugu Automatic Text Correction Pipeline.
Integrates:
1. Script Detection & Transliteration (Tenglish -> Telugu)
2. Normalization & Preprocessing (NFC, ZWJ/ZWNJ cleanup)
3. Dictionary-Based Error Detection
4. Multi-Strategy Correction (Rule-Based, Edit-Distance, or Hybrid)
"""
from src.preprocessing.normalizer import TeluguPreprocessor
from src.preprocessing.tokenizer import TeluguTokenizer
from src.preprocessing.transliteration import TenglishConverter
from src.error_detection.dictionary_based import DictionaryErrorDetector
from src.correction.rule_based_corrector import RuleBasedCorrector
from src.correction.edit_distance_corrector import EditDistanceCorrector

class TeluguTextCorrectionPipeline:
    """
    Main orchestrator for the Telugu Text Correction system.
    """
    
    def __init__(self, mode: str = "hybrid"):
        """
        mode: 'hybrid' (rules first, then edit-distance), 'edit_distance', or 'rule_based'
        """
        self.mode = mode
        self.preprocessor = TeluguPreprocessor()
        self.tokenizer = TeluguTokenizer()
        self.transliteration = TenglishConverter()
        self.detector = DictionaryErrorDetector()
        self.rule_corrector = RuleBasedCorrector()
        self.edit_corrector = EditDistanceCorrector()
        
    def correct(self, text: str, mode: str = None) -> dict:
        """
        Processes a single input text through the pipeline.
        Returns a dictionary with intermediate states and final corrected text.
        """
        if not text:
            return {
                "original": "",
                "transliterated": "",
                "normalized": "",
                "corrected": "",
                "errors_detected": []
            }
            
        use_mode = mode or self.mode
        
        # Step 1: Clean raw whitespace and noise
        cleaned = self.preprocessor.clean(text)
        
        # Step 2: Script detection - transliterate if Tenglish
        is_tel = self.preprocessor.is_telugu(cleaned)
        if not is_tel:
            transliterated = self.transliteration.convert(cleaned)
        else:
            transliterated = cleaned
            
        # Step 3: Unicode NFC & IndicNLP normalization
        normalized = self.preprocessor.normalize(transliterated)
        
        # Step 4: Detect spelling errors
        errors = self.detector.detect_errors(normalized, tokenizer=self.tokenizer)
        
        # Step 5: Correction based on chosen strategy
        if use_mode == "rule_based":
            corrected = self.rule_corrector.correct_sentence(normalized)
        elif use_mode == "edit_distance":
            corrected = self.edit_corrector.correct_sentence(
                normalized, detector=self.detector
            )
        else:
            # Hybrid strategy:
            # 1. First pass with deterministic linguistic rules
            rule_pass = self.rule_corrector.correct_sentence(normalized)
            # 2. Second pass with edit distance for remaining uncorrected words
            #
            # NOTE: the tokenizer is deliberately NOT passed to the corrector.
            # IndicNLP splits punctuation off as standalone tokens, and any
            # tokenize-then-rejoin cycle cannot faithfully reproduce the original
            # spacing (e.g. GT 'వినిపించగా,విశ్వనాథ' has no space after the comma).
            # Whitespace splitting is lossless, so the corrector uses that and the
            # detector keeps the IndicNLP tokenizer for detection only.
            corrected = self.edit_corrector.correct_sentence(
                rule_pass, detector=self.detector
            )
            
        return {
            "original": text,
            "was_transliterated": not is_tel,
            "transliterated": transliterated,
            "normalized": normalized,
            "corrected": corrected,
            "errors_detected": [e for e in errors if e.get("is_error", False)]
        }
        
    def batch_correct(self, texts: list, mode: str = None) -> list:
        """
        Processes a list of sentences.
        """
        return [self.correct(t, mode=mode) for t in texts]

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    pipeline = TeluguTextCorrectionPipeline(mode="hybrid")
    
    # Test cases
    test_inputs = [
        "namaskaram ela unnaru",
        "తేలుగు భాశ చాల అందమైణది",
        "విద్యార్ధులు పరిక్షలు రాసారు"
    ]
    
    for t in test_inputs:
        res = pipeline.correct(t)
        print(f"\nInput     : {res['original']}")
        print(f"Corrected : {res['corrected']}")
        print(f"Errors    : {[e['word'] for e in res['errors_detected']]}")

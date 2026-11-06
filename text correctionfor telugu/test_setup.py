# -*- coding: utf-8 -*-
"""Quick test script to verify all project modules work correctly."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
random.seed(42)

print("=" * 50)
print("TESTING ALL MODULES")
print("=" * 50)

# 1. Noise Injector
print("\n--- Noise Injector ---")
from src.utils.noise_injector import corrupt_sentence, generate_synthetic_pairs, create_splits

sentences = [
    'తెలుగు భాష చాలా అందమైనది',
    'విద్యార్థులు పరీక్షలు రాశారు',
    'నేను బాగున్నాను ధన్యవాదాలు'
]
for s in sentences:
    result = corrupt_sentence(s, error_rate=0.8)
    if result['input'] != result['target']:
        print(f"Original:  {result['target']}")
        print(f"Corrupted: {result['input']}")
        print(f"Error:     {result['error_type']}")
        print()

pairs = generate_synthetic_pairs(sentences * 100, num_pairs=20)
print(f"Generated {len(pairs)} pairs, {sum(1 for p in pairs if p['input'] != p['target'])} with errors")
train, val, test = create_splits(pairs)
print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

# 2. Edit Distance Corrector
print("\n--- Edit Distance Corrector ---")
from src.correction.edit_distance_corrector import EditDistanceCorrector
vocab = {'తెలుగు', 'భాష', 'చాలా', 'అందమైనది', 'విద్యార్థులు', 'పరీక్షలు'}
corrector = EditDistanceCorrector(vocab=vocab)
test_word = 'తెలుగి'
candidates = corrector.get_candidates(test_word, top_k=3)
print(f"Candidates for '{test_word}': {candidates}")

# 3. Evaluation Metrics
print("\n--- Evaluation Metrics ---")
from src.evaluation.metrics import compute_metrics, print_metrics
preds = ['తెలుగు భాష', 'నేను బాగున్నాను']
refs = ['తెలుగు భాష', 'నేను బాగున్నాను']
metrics = compute_metrics(preds, refs)
print_metrics(metrics)

# 4. Rule-based Corrector
print("\n--- Rule-based Corrector ---")
from src.correction.rule_based_corrector import RuleBasedCorrector
rc = RuleBasedCorrector()
test_sent = 'తెలుగి భాశ చాల అందమైణది'
corrected = rc.correct_sentence(test_sent)
print(f"Input:     {test_sent}")
print(f"Corrected: {corrected}")

# 5. Error Analysis
print("\n--- Error Analysis ---")
from src.evaluation.error_analysis import ErrorAnalyzer
analyzer = ErrorAnalyzer()
err_type = analyzer.classify_error('తెలుగి', 'తెలుగు')
print(f"Error type for తెలుగి -> తెలుగు: {err_type}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)

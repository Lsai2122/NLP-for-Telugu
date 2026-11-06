# -*- coding: utf-8 -*-
"""
Evaluation Metrics Module for Automatic Text Correction.
Computes:
1. Word Error Rate (WER) via jiwer
2. Character Error Rate (CER) via jiwer
3. Exact Match Accuracy (EM)
4. BLEU Score via sacrebleu
5. Token-Level Precision, Recall, and F1 (multiset overlap)
"""
from collections import Counter
import jiwer
import sacrebleu

def compute_metrics(predictions: list, references: list) -> dict:
    """
    Computes standard NLP text correction evaluation metrics:
    - WER, CER: Lower is better (0.0 = perfect)
    - ExactMatch, BLEU, Precision, Recall, F1: Higher is better (1.0 or 100.0 = perfect)
    """
    if not predictions or not references or len(predictions) != len(references):
        return {}

    # 1. Word Error Rate (WER) and Character Error Rate (CER)
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    
    # 2. Exact Match Accuracy (Sentence-level)
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    exact_match_acc = exact_matches / len(predictions)
    
    # 3. Corpus BLEU Score
    try:
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        bleu_score = bleu.score
    except Exception:
        bleu_score = 0.0
        
    # 4. Token-Level Multiset Precision, Recall, and F1
    total_tp = 0
    total_pred_tokens = 0
    total_ref_tokens = 0
    
    for p, r in zip(predictions, references):
        p_tokens = p.split()
        r_tokens = r.split()
        
        c_pred = Counter(p_tokens)
        c_ref = Counter(r_tokens)
        
        # Intersection of multisets
        overlap = sum((c_pred & c_ref).values())
        
        total_tp += overlap
        total_pred_tokens += len(p_tokens)
        total_ref_tokens += len(r_tokens)
        
    precision = (total_tp / total_pred_tokens) if total_pred_tokens > 0 else 0.0
    recall = (total_tp / total_ref_tokens) if total_ref_tokens > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "WER": wer,
        "CER": cer,
        "ExactMatch": exact_match_acc,
        "BLEU": bleu_score,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Total_Samples": len(predictions)
    }

def print_metrics(metrics_dict: dict, title: str = "Evaluation Metrics"):
    """
    Pretty prints the evaluation metrics table.
    """
    print("=" * 45)
    print(f" {title.upper()}")
    print("=" * 45)
    # Only genuine proportions are rendered as percentages. Throughput and timing
    # are raw quantities - printing them as "1079.18%" was nonsense.
    PERCENT_METRICS = {"WER", "CER", "ExactMatch", "Precision", "Recall", "F1"}
    for k, v in metrics_dict.items():
        if k == "Total_Samples":
            print(f"  {k:<16s}: {v}")
        elif k == "BLEU":
            print(f"  {k:<16s}: {v:.2f}")
        elif k == "Inference_Time_Sec":
            print(f"  {k:<16s}: {v:.2f} s")
        elif k == "Sentences_Per_Sec":
            print(f"  {k:<16s}: {v:.2f} sentences/s")
        elif k in PERCENT_METRICS and isinstance(v, float):
            print(f"  {k:<16s}: {v:.4f} ({v*100:.2f}%)")
        elif isinstance(v, float):
            print(f"  {k:<16s}: {v:.4f}")
        else:
            print(f"  {k:<16s}: {v}")
    print("=" * 45)

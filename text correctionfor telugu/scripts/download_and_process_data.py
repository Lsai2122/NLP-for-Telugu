# -*- coding: utf-8 -*-
"""
Download and process REAL Telugu data from multiple sources.
Builds vocabulary and generates synthetic training pairs.
"""
import os
import sys
import json
import re
import random
from collections import Counter
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r"t:\B-Tech\SEMESTER\Sem 5\Mini proj\telugu-text-correction")
sys.path.append(os.getcwd())

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from src.utils.noise_injector import generate_synthetic_pairs, save_pairs_to_json, create_splits

normalizer = IndicNormalizerFactory().get_normalizer("te")

def is_good_telugu(text):
    """Check if text is primarily Telugu and long enough."""
    if not text or len(text) < 20:
        return False
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    total_chars = len(text.replace(' ', ''))
    return total_chars > 0 and (telugu_chars / total_chars) > 0.5

def clean_sentence(text):
    """Clean a single Telugu sentence."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)  # Remove wiki markup [1], [edit] etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical refs
    text = normalizer.normalize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_sentences(text):
    """Split a paragraph into individual sentences."""
    # Split on Telugu sentence endings and common delimiters
    raw_sents = re.split(r'[.।!?]', text)
    sentences = []
    for s in raw_sents:
        s = clean_sentence(s)
        if is_good_telugu(s) and 20 < len(s) < 300:
            sentences.append(s)
    return sentences

# ============================================================
print("=" * 60)
print("DOWNLOADING REAL TELUGU DATA")
print("=" * 60)

all_sentences = []

# SOURCE 1: Telugu Wikipedia (most reliable)
print("\n[1/3] Loading Telugu Wikipedia...")
try:
    from datasets import load_dataset
    ds = load_dataset('wikimedia/wikipedia', '20231101.te', split='train', streaming=True)
    wiki_count = 0
    for item in ds:
        text = item.get('text', '')
        sents = extract_sentences(text)
        all_sentences.extend(sents)
        wiki_count += 1
        if len(all_sentences) >= 60000:
            break
        if wiki_count % 1000 == 0:
            print(f"  Processed {wiki_count} articles, {len(all_sentences)} sentences so far...")
    print(f"  ✅ Got {len(all_sentences)} sentences from {wiki_count} Wikipedia articles")
except Exception as e:
    print(f"  ❌ Wikipedia failed: {e}")

# SOURCE 2: Sangraha/IndicCorp Telugu
print("\n[2/3] Loading Sangraha Telugu...")
try:
    ds2 = load_dataset('ai4bharat/sangraha', 'verified', split='tel', streaming=True)
    sangraha_count = 0
    for item in ds2:
        text = item.get('text', '')
        sents = extract_sentences(text)
        all_sentences.extend(sents)
        sangraha_count += 1
        if len(all_sentences) >= 100000:
            break
        if sangraha_count % 2000 == 0:
            print(f"  Processed {sangraha_count} docs, {len(all_sentences)} total sentences...")
    print(f"  ✅ Got {len(all_sentences)} total sentences (added from {sangraha_count} Sangraha docs)")
except Exception as e:
    print(f"  ⚠️ Sangraha failed: {e} — continuing with Wikipedia data")

# SOURCE 3: Dakshina transliteration data (already downloaded)
print("\n[3/3] Loading Dakshina transliteration data...")
dakshina_pairs = []
dakshina_dir = 'data/raw/dakshina_dataset_v1.0/te'
if os.path.exists(dakshina_dir):
    for fname in os.listdir(dakshina_dir):
        if 'lexicon' in fname and fname.endswith('.tsv'):
            fpath = os.path.join(dakshina_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        dakshina_pairs.append({
                            'telugu': parts[0],
                            'roman': parts[1]
                        })
    print(f"  ✅ Loaded {len(dakshina_pairs)} transliteration pairs from Dakshina")
    # Save transliteration pairs
    with open('data/processed/dakshina_transliteration.json', 'w', encoding='utf-8') as f:
        json.dump(dakshina_pairs, f, ensure_ascii=False, indent=2)
else:
    print(f"  ⚠️ Dakshina not found at {dakshina_dir}")

# ============================================================
# DEDUPLICATE AND SAVE CORPUS
print("\n" + "=" * 60)
print("PROCESSING CORPUS")
print("=" * 60)

# Deduplicate
unique_sentences = list(set(all_sentences))
random.shuffle(unique_sentences)
print(f"Total unique sentences: {len(unique_sentences)}")

# Save corpus
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/indiccorp_telugu_100k.txt', 'w', encoding='utf-8') as f:
    for s in unique_sentences:
        f.write(s + '\n')
print(f"Saved to data/processed/indiccorp_telugu_100k.txt")

# Corpus stats
lengths = [len(s) for s in unique_sentences]
print(f"Avg sentence length: {sum(lengths)/len(lengths):.0f} chars")
print(f"Min: {min(lengths)}, Max: {max(lengths)}")

# ============================================================
# BUILD VOCABULARY
print("\n" + "=" * 60)
print("BUILDING VOCABULARY")
print("=" * 60)

word_counts = Counter()
for sentence in tqdm(unique_sentences, desc="Tokenizing"):
    words = indic_tokenize.trivial_tokenize(sentence, "te")
    # Filter: only Telugu words (no punctuation, numbers, etc.)
    telugu_words = [w for w in words if any('\u0C00' <= c <= '\u0C7F' for c in w)]
    word_counts.update(telugu_words)

top_100k = dict(word_counts.most_common(100000))
with open('data/processed/telugu_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(top_100k, f, ensure_ascii=False, indent=2)

print(f"Total unique Telugu words: {len(word_counts)}")
print(f"Saved top {len(top_100k)} words to telugu_vocab.json")
print(f"Top 20 words: {list(top_100k.keys())[:20]}")

# ============================================================
# GENERATE SYNTHETIC TRAINING PAIRS
print("\n" + "=" * 60)
print("GENERATING SYNTHETIC TRAINING DATA")
print("=" * 60)

os.makedirs('data/synthetic', exist_ok=True)
pairs = generate_synthetic_pairs(unique_sentences, 50000)
train, val, test = create_splits(pairs)

save_pairs_to_json(train, 'data/synthetic/train.json')
save_pairs_to_json(val, 'data/synthetic/val.json')
save_pairs_to_json(test, 'data/synthetic/test.json')

total_pairs = len(pairs)
with_errors = sum(1 for p in pairs if p['input'] != p['target'])

error_types = Counter()
for p in pairs:
    if p['error_type'] != 'none':
        for et in p['error_type'].split(','):
            error_types[et.strip()] += 1

print(f"Total pairs: {total_pairs}")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Pairs with actual errors: {with_errors} ({with_errors/total_pairs*100:.1f}%)")
print(f"Error type distribution:")
for et, count in error_types.most_common():
    print(f"  {et}: {count} ({count/with_errors*100:.1f}%)")

# ============================================================
# SHOW SAMPLE PAIRS
print("\n" + "=" * 60)
print("SAMPLE SYNTHETIC PAIRS")
print("=" * 60)

shown = 0
for p in pairs:
    if p['input'] != p['target']:
        print(f"  Input:  {p['input'][:80]}")
        print(f"  Target: {p['target'][:80]}")
        print(f"  Error:  {p['error_type']}")
        print()
        shown += 1
        if shown >= 5:
            break

print("\n✅ ALL DATA PROCESSING COMPLETE!")

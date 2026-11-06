# -*- coding: utf-8 -*-
import sys
import json
import os

sys.stdout.reconfigure(encoding='utf-8')

base = r't:\B-Tech\SEMESTER\Sem 5\Mini proj\telugu-text-correction\data\raw\dakshina_dataset_v1.0'
out_dir = r't:\B-Tech\SEMESTER\Sem 5\Mini proj\telugu-text-correction\data\processed'
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Load Transliteration Lexicon Pairs (Word Level)
# ---------------------------------------------------------
word_pairs = []
for split in ['train', 'dev', 'test']:
    fpath = os.path.join(base, 'te', 'lexicons', f'te.translit.sampled.{split}.tsv')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word_pairs.append({
                        'telugu': parts[0],
                        'roman': parts[1],
                        'split': split
                    })

print(f'Total Dakshina word-level pairs: {len(word_pairs):,}')
print('Sample word pairs:')
for p in word_pairs[:5]:
    print(f"  {p['roman']:20s} -> {p['telugu']}")

with open(os.path.join(out_dir, 'dakshina_transliteration.json'), 'w', encoding='utf-8') as f:
    json.dump(word_pairs, f, ensure_ascii=False, indent=2)
print(f'Saved to {os.path.join(out_dir, "dakshina_transliteration.json")}')

# ---------------------------------------------------------
# 2. Load Full Romanized Sentence Pairs (Sentence Level)
# ---------------------------------------------------------
sentence_pairs = []

for split in ['dev', 'test']:
    fn_native = os.path.join(base, 'te', 'romanized', f'te.romanized.rejoined.{split}.native.txt')
    fn_roman = os.path.join(base, 'te', 'romanized', f'te.romanized.rejoined.{split}.roman.txt')
    
    if os.path.exists(fn_native) and os.path.exists(fn_roman):
        with open(fn_native, 'r', encoding='utf-8') as f_nat, open(fn_roman, 'r', encoding='utf-8') as f_rom:
            for native_line, roman_line in zip(f_nat, f_rom):
                native_text = native_line.strip()
                roman_text = roman_line.strip()
                if native_text and roman_text:
                    sentence_pairs.append({
                        'telugu': native_text,
                        'roman': roman_text,
                        'split': split,
                        'num_words': len(native_text.split())
                    })

print(f'\nTotal Romanized Full Sentence Pairs: {len(sentence_pairs):,}')
print('Sample full sentences:')
for i, p in enumerate(sentence_pairs[:3], 1):
    print(f"[{i}] [{p['split'].upper()}] ({p['num_words']} words)")
    print(f"   Roman  : {p['roman']}")
    print(f"   Telugu : {p['telugu']}\n")

with open(os.path.join(out_dir, 'dakshina_romanized_sentences.json'), 'w', encoding='utf-8') as f:
    json.dump(sentence_pairs, f, ensure_ascii=False, indent=2)
print(f'Saved to {os.path.join(out_dir, "dakshina_romanized_sentences.json")}')
print('\n✅ Dakshina data processed successfully!')

# -*- coding: utf-8 -*-
import random
import json
import os

# Confusions dictionaries for Telugu
MATRA_CONFUSIONS = {
    "ి": "ీ",
    "ీ": "ి",
    "ు": "ూ",
    "ూ": "ు",
    "ె": "ే",
    "ే": "ె",
    "ొ": "ో",
    "ో": "ొ"
}

CONSONANT_CONFUSIONS = {
    "ణ": "న",
    "న": "ణ",
    "శ": "ష",
    "ష": "స",
    "స": "శ",
    "ట": "డ",
    "డ": "ట",
    "బ": "వ",
    "వ": "బ"
}

def inject_matra_error(word: str) -> str:
    chars = list(word)
    for i, c in enumerate(chars):
        if c in MATRA_CONFUSIONS and random.random() < 0.5:
            chars[i] = MATRA_CONFUSIONS[c]
            return "".join(chars)
    return word

def inject_consonant_error(word: str) -> str:
    chars = list(word)
    for i, c in enumerate(chars):
        if c in CONSONANT_CONFUSIONS and random.random() < 0.5:
            chars[i] = CONSONANT_CONFUSIONS[c]
            return "".join(chars)
    return word

def inject_space_error(sentence: str) -> str:
    words = sentence.split()
    if len(words) > 1:
        idx = random.randint(0, len(words) - 2)
        # Merge two words
        words[idx] = words[idx] + words[idx+1]
        del words[idx+1]
    return " ".join(words)

def inject_deletion(word: str) -> str:
    if len(word) > 2:
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx+1:]
    return word
    
def inject_swap(word: str) -> str:
    if len(word) > 2:
        idx = random.randint(1, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        return "".join(chars)
    return word

def corrupt_sentence(sentence: str, error_rate: float = 0.3) -> dict:
    words = sentence.split()
    corrupted_words = []
    error_types = []
    
    # 10% chance for space error
    if random.random() < 0.1:
        return {"input": inject_space_error(sentence), "target": sentence, "error_type": "space"}
        
    for word in words:
        if random.random() < error_rate:
            r = random.random()
            if r < 0.4:
                corrupted_words.append(inject_matra_error(word))
                error_types.append("matra")
            elif r < 0.7:
                corrupted_words.append(inject_consonant_error(word))
                error_types.append("consonant")
            elif r < 0.85:
                corrupted_words.append(inject_deletion(word))
                error_types.append("deletion")
            else:
                corrupted_words.append(inject_swap(word))
                error_types.append("swap")
        else:
            corrupted_words.append(word)
            
    err_type = ",".join(set(error_types)) if error_types else "none"
    return {
        "input": " ".join(corrupted_words),
        "target": sentence,
        "error_type": err_type
    }

def generate_synthetic_pairs(clean_sentences: list, num_pairs: int) -> list:
    pairs = []
    for _ in range(num_pairs):
        sentence = random.choice(clean_sentences)
        pairs.append(corrupt_sentence(sentence))
    return pairs

def save_pairs_to_json(pairs: list, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

def create_splits(pairs: list) -> tuple:
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]

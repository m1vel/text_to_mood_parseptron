import re
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s!?.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_vector(text: str, dictionary: List[str], emotional_words: Dict) -> np.ndarray:
    text = preprocess_text(text)
    words = text.split()
    vector = np.zeros(len(dictionary), dtype=np.float32)
    word_counts = Counter(words)
    for i, word in enumerate(dictionary):
        if word in word_counts:
            base_value = 1.0
            for mood, mood_words in emotional_words.items():
                if word in mood_words:
                    base_value *= mood_words[word]
                    break
            frequency = word_counts[word]
            freq_factor = min(frequency * 0.3, 2.0)
            vector[i] = base_value * freq_factor
            if base_value > 1.5:
                vector[i] *= 1.2
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def extend_dictionary_from_texts(texts: List[Tuple[str, str]], existing_dict: List[str]) -> List[str]:
    new_words = set()
    for text, _ in texts:
        text = preprocess_text(text)
        words = text.split()
        new_words.update(words)
    
    existing_words = set(existing_dict)
    return existing_dict + [word for word in new_words if word not in existing_words]
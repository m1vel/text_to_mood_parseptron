import random
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from emotional_words import em_words
    emotional_words = em_words()
except ImportError:
    sys.exit(1)

try:
    from traindata import re_train_data, get_test_data
    train_data = re_train_data()
except ImportError:
    sys.exit(1)

try:
    from text_work import preprocess_text, text_to_vector, extend_dictionary_from_texts
except ImportError:
    sys.exit(1)

try:
    from Parseptron import EmotionPerceptron
    emotional_words = em_words()
except ImportError:
    sys.exit(1)
moods = list(emotional_words.keys())

all_words = set()
for mood_words in emotional_words.values():
    all_words.update(mood_words.keys())

dictionary = list(all_words)
dictionary = extend_dictionary_from_texts(train_data, dictionary)

perceptrons = {mood: EmotionPerceptron(len(dictionary), lr=0.015, momentum=0.92) 
               for mood in moods}

def train_with_validation(train_set: List[Tuple[str, str]], 
                         val_set: List[Tuple[str, str]], 
                         epochs: int = 300,
                         early_stopping_patience: int = 30) -> Dict[str, Any]:
    best_accuracy = 0
    patience_counter = 0
    best_weights = {mood: (p.weights.copy(), p.bias) for mood, p in perceptrons.items()}
    
    for epoch in range(epochs):
        shuffled_data = train_set.copy()
        random.shuffle(shuffled_data)
        total_error = 0
        correct_train = 0
        for text, answer in shuffled_data:
            x = text_to_vector(text, dictionary, emotional_words)
            scores = {}
            for mood, perceptron in perceptrons.items():
                scores[mood] = perceptron.activate(x)
            predicted = max(scores, key=scores.get)
            if predicted == answer:
                correct_train += 1
            for mood, perceptron in perceptrons.items():
                y = 1 if mood == answer else 0
                error = perceptron.train(x, y)
                total_error += error
        correct_val = 0
        val_predictions = []
        for text, answer in val_set:
            predicted, scores = select_mood(text)
            val_predictions.append((text, answer, predicted, scores))
            if predicted == answer:
                correct_val += 1
        train_accuracy = correct_train / len(train_set)
        val_accuracy = correct_val / len(val_set)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = {mood: (p.weights.copy(), p.bias) for mood, p in perceptrons.items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"эпоха {epoch:3d}/{epochs}"
                  f"ошибки: {total_error:.4f}"
                  f"точность: {train_accuracy:.2%}")
        if patience_counter >= early_stopping_patience:
            break
    for mood, (weights, bias) in best_weights.items():
        perceptrons[mood].weights = weights
        perceptrons[mood].bias = bias
    return {
        'best_accuracy': best_accuracy,
        'final_val_accuracy': val_accuracy,
        'epochs_trained': min(epoch, epochs)
    }

def select_mood(text: str) -> Tuple[str, Dict[str, float]]:
    x = text_to_vector(text, dictionary, emotional_words)
    scores = {}
    for mood, perceptron in perceptrons.items():
        scores[mood] = perceptron.activate(x)
    exp_scores = {k: np.exp(v) for k, v in scores.items()}
    sum_exp = sum(exp_scores.values())
    if sum_exp == 0:
        probabilities = {k: 1/len(scores) for k in scores}
    else:
        probabilities = {k: v/sum_exp for k, v in exp_scores.items()}
    best_mood = max(probabilities, key=probabilities.get)
    return best_mood, probabilities

def analyze_text(text: str, verbose: bool = True) -> Tuple[str, Dict[str, float]]:
    text_clean = preprocess_text(text)
    mood, probabilities = select_mood(text)
    if verbose:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_probs:
            indicator = ".!." if k == mood else ".-."
            print(f"  [{indicator}] {k:12} : {v:7.2%}")
        words = text_clean.split()
        found_emotional_words = []
        for word in words:
            if word in dictionary:
                related_emotions = []
                for emotion, words_dict in emotional_words.items():
                    if word in words_dict:
                        related_emotions.append((emotion, words_dict[word]))
                if related_emotions:
                    emotion, weight = max(related_emotions, key=lambda x: x[1])
                    found_emotional_words.append((word, emotion, weight))
        if found_emotional_words:
            found_emotional_words.sort(key=lambda x: x[2], reverse=True)
            for word, emotion, weight in found_emotional_words:
                strength = "слабенько" if weight < 1.3 else "средненько" if weight < 1.7 else "сильненько"
                print(f"  '{word}' - {emotion:12} ({strength}, вес: {weight:.2f})")
                print(f"конечное настроение: {mood.upper()} ({probabilities[mood]:.1%})")
        else:
            print("нет значимых слов")
        print(f"конечное настроение: {mood.upper()} ({probabilities[mood]:.1%})")
    return mood, probabilities

def test_model(test_phrases: List[str]):   
    results = []
    for phrase in test_phrases:
        if isinstance(phrase, tuple):
            phrase = phrase[0]
        elif not isinstance(phrase, str):
            phrase = str(phrase)
        mood, probs = analyze_text(phrase, verbose=False)
        max_prob = max(probs.values())
        results.append((phrase, mood, max_prob))
    return results

def show_model_statistics():
    for mood, perceptron in perceptrons.items():
        important = perceptron.get_important_features(dictionary, top_n=5)
        if important:
            words_str = ", ".join([f"{w} ({weight:+.3f})" for w, weight in important])
            print(f"{mood:12}:{words_str}")

test_phrases = get_test_data()

def main():
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.8)
    train_set = train_data[:split_idx]
    val_set = train_data[split_idx:]
    results = train_with_validation(
        train_set, 
        val_set, 
        epochs=400,
        early_stopping_patience=100
    )
    print(results)
    test_results = test_model(test_phrases)
    print(test_results)
    show_model_statistics()
    while True:
        try:
            user_input = input("фраза: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                break
            
            elif user_input.lower() in ['стат', 'статистика', 'stats']:
                show_model_statistics()
                continue
            
            elif user_input.lower() in ['тест', 'test']:
                test_model(test_phrases)
                continue
            mood, probs = analyze_text(user_input, verbose=True)
            
        except KeyboardInterrupt:
            break
        except Exception:
            continue

if __name__ == "__main__":
    main()
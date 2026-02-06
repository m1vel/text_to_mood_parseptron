import re
import random
from Parseptron import Parseptron as PS

moods = [
    'neutral',
    'sadness',
    'fear', 
    'joy',
    'anger',
    'irritation'
]

dictionary = [
    'класс', 'рад', 'супер', 'отлично', 'хорошо', 'прекрасно', 'доволен', 'счастлив',
    'злюсь', 'бесит', 'ненавижу', 'злость', 'ярост', 'гнев', 'возмущен',
    'грустно', 'печально', 'тоска', 'уныло', 'плачу', 'скорбь',
    'страшно', 'боюсь', 'ужас', 'пугает', 'испуг', 'тревога',
    'раздражает', 'досадно', 'бесит', 'надоело', 'достало',
    'ладно', 'нормально', 'ок', 'пойдет', 'обычно', 'стандартно'
]

def text_to_vector(text: str, dictionary: list[str]) -> list[int]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    word_set = set(words)
    return [1 if word in word_set else 0 for word in dictionary]

def extend_dictionary_from_texts(texts):
    new_words = set()
    for text, _ in texts:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        words = text.split()
        new_words.update(words)
    
    existing_words = set(dictionary)
    return dictionary + [word for word in new_words if word not in existing_words]
parseptrons = {
    mood: PS(len(dictionary))
    for mood in moods
}

train_data = [
    ("мне очень классно", "joy"),
    ("я рад и доволен", "joy"),
    ("супер, отлично", "joy"),
    ("хорошее настроение", "joy"),
    ("я счастлив сегодня", "joy"),
    ("все прекрасно", "joy"),

    ("я злюсь", "anger"),
    ("меня это бесит", "anger"),
    ("ненавижу это", "anger"),
    ("это вызывает злость", "anger"),
    ("я в ярости", "anger"),

    ("мне грустно", "sadness"),
    ("очень печально", "sadness"),
    ("на душе тоска", "sadness"),
    ("чувствую уныние", "sadness"),

    ("мне страшно", "fear"),
    ("я боюсь", "fear"),
    ("это ужасно", "fear"),
    ("меня пугает", "fear"),

    ("это раздражает", "irritation"),
    ("меня это бесит и раздражает", "irritation"),
    ("достало уже", "irritation"),
    ("надоело это", "irritation"),

    ("ну ладно", "neutral"),
    ("нормально, ок", "neutral"),
    ("все в порядке", "neutral"),
    ("ничего особенного", "neutral"),
]

dictionary = extend_dictionary_from_texts(train_data)

parseptrons = {
    mood: PS(len(dictionary))
    for mood in moods
}

epochs = 100
for epoch in range(epochs):
    shuffled_data = train_data.copy()
    random.shuffle(shuffled_data)
    
    total_error = 0
    for text, answer in shuffled_data:
        x = text_to_vector(text, dictionary)
        
        for mood, parseptron in parseptrons.items():
            y = 1 if mood == answer else 0
            error = parseptron.train(x, y)
            total_error += error
    
    if epoch % 20 == 0:
        print(f"  эпоха {epoch}/{epochs}, ошибка: {total_error}")

def select_mood(text):
    x = text_to_vector(text, dictionary)
    activates = {}
    
    for mood, parseptron in parseptrons.items():
        activates[mood] = parseptron.activate(x)
    
    best_mood = max(activates, key=activates.get)
    return best_mood, activates
def show_weights_for_text(text, mood):
    x = text_to_vector(text, dictionary)
    
    print("\nвеса:")
    
    found_words_with_weights = []
    for i, (word, present) in enumerate(zip(dictionary, x)):
        if present == 1:
            weight = parseptrons[mood].weights[i]
            found_words_with_weights.append((word, weight))
    
    if found_words_with_weights:
        sorted_words = sorted(found_words_with_weights, key=lambda x: abs(x[1]), reverse=True)
        
        for word, weight in sorted_words:
            influence = "+" if weight > 0 else "-" if weight < 0 else "пох"
    else:
        print("не найдено ключевых слов из словаря")

while True:
    user_input = input("\nфраза: ").strip()
    if user_input.lower() in ['выход', 'exit', 'quit']:
        print("\n")
        break
    if not user_input:
        continue
    mood, all_scores = select_mood(user_input)
    print(f"\nнастроение: {mood.upper()}")
    print("все настроения:")
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_scores:
        indicator = "+" if k == mood else "-"
        print(f"  {indicator} {k:12}: {v:7.2f}")
    show_weights_for_text(user_input, mood)
    
    text_lower = user_input.lower()
    text_lower = re.sub(r"[^\w\s]", "", text_lower)
    words_in_text = set(text_lower.split())
    words_in_dict = [word for word in words_in_text if word in dictionary]
    
    if words_in_dict:
        print(f"\nфразы в словаре: {', '.join(words_in_dict)}")

print("\nстатистика модели:")

print("\nважные слова для каждого настроения:")
for mood, perc in parseptrons.items():
    word_weights = list(zip(dictionary, perc.weights))
    positive_words = sorted([(w, weight) for w, weight in word_weights if weight > 0], 
                           key=lambda x: x[1], reverse=True)[:3]
    if positive_words:
        top_words = ", ".join([w for w, _ in positive_words])
        print(f"  {mood:12}: {top_words}")
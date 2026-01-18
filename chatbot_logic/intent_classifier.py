import json
import random
import nltk
import joblib
from nltk.tokenize import word_tokenize
from collections import defaultdict

#The Use Case the chatbot will be focused on is for the user to see top-rated books
nltk.download('punkt', quiet=True)

def extract_features(text):
    features = {}
    words = word_tokenize(text.lower())
    for word in words:
        features[f'has({word})'] = True
    return features

try:
    with open('intents_data.json', 'r') as file:
        training_data = json.load(file)
except FileNotFoundError:
    print("intents_data.json not found!")
    exit()

feature_sets = [(extract_features(item["text"]), item["intent"]) 
                for item in training_data]

random.shuffle(feature_sets)
split_point = int(len(feature_sets) * 0.8)
train_set = feature_sets[:split_point]
test_set = feature_sets[split_point:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

# Accuracy
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(f"Overall Accuracy: {accuracy:.2%}")

print("\nMost Informative Features:")
classifier.show_most_informative_features(5)

# Get predictions for the test set
refsets = defaultdict(set) # Reference (Actual) Intents
testsets = defaultdict(set) # Predicted Intents


for i, (features, intent_actual) in enumerate(test_set):
    intent_predicted = classifier.classify(features)
    refsets[intent_actual].add(i)
    testsets[intent_predicted].add(i)

intents = classifier.labels()

for intent in intents:
    if intent in refsets:

        precision = nltk.scores.precision(refsets[intent], testsets[intent])

        recall = nltk.scores.recall(refsets[intent], testsets[intent])

        f1_score = nltk.scores.f_measure(refsets[intent], testsets[intent])

        # Using 0.0 if one of matrices is None
        print(f"Intent: {intent}")
        print(f"Precision: {(precision if precision else 0.0):.2f}")
        print(f"Recall: {(recall if recall else 0.0):.2f}")
        print(f"F1-Score: {(f1_score if f1_score else 0.0):.2f}")


joblib.dump(classifier, 'models/book_intent_model.joblib')

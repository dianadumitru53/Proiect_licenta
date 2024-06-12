from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from keras.models import load_model
import re
import spacy
import random
from nltk.tokenize import word_tokenize

# Încarcă modelul și datele
nlp = spacy.load("ro_core_news_md")
stop_words = nlp.Defaults.stop_words
model = load_model('chatbot_model.h5')
data = json.loads(open('intents_r.json').read())

# Funcțiile tale de preprocesare
def remove_diacritics(text):
    diacritic_map = {
        'ă': 'a', 'Ă': 'A',
        'â': 'a', 'Â': 'A',
        'ș': 's', 'Ș': 'S',
        'ț': 't', 'Ț': 'T',
        'î': 'i', 'Î': 'I',
        'ţ': 't', 'Ţ': 'T'
    }
    pattern = re.compile('|'.join(diacritic_map.keys()))
    return pattern.sub(lambda match: diacritic_map[match.group(0)], text)

def clean_sentence(sentence):
    lem = nlp(sentence)
    lemmatized_tokens = [token.lemma_.lower() for token in lem if token.lemma_ not in stop_words]
    sentence_no_diacritics = remove_diacritics(' '.join(lemmatized_tokens))
    sentence_no_punctuation = re.sub(r'[^\w\s]', '', sentence_no_diacritics)
    sentence_ascii_only = re.sub(r'[^\x00-\x7F]+', ' ', sentence_no_punctuation)
    sentence_words = word_tokenize(sentence_ascii_only)
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

words = sorted(list(set([word for intent in data["intents"] for pattern in intent["patterns"] for word in clean_sentence(pattern)])))
classes = sorted(list(set(intent['tag'] for intent in data['intents'])))

def predict_class(sentence, words, model):
    ERROR_THRESHOLD = 0.25
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = []
    for i, r in enumerate(res):
        if r > ERROR_THRESHOLD:
            results.append({"intent": classes[i], "probability": str(r)})
    return results

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json['message']
    ints = predict_class(msg, words, model)
    res = getResponse(ints, data)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)

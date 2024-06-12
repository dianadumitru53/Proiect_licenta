import json
import keras
import numpy as np
import re
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import random
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from conversation import respond
from keras.optimizers import Adam, SGD
import spacy
from keras.regularizers import l2, l1, l1_l2

# with open('stop.txt', 'r', encoding='utf-8') as file:
#     stop_words = list(file.read().strip().split(','))

nlp = spacy.load("ro_core_news_md")
stop_words = nlp.Defaults.stop_words


url = 'intents_r.json'
words = []
documents = []
classes = []
data = json.loads(open(url).read())

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

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words.extend(clean_sentence(pattern))
        documents.append((pattern, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = bow(doc[0], words)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

from sklearn.model_selection import train_test_split
train_x1, test_x, train_y1, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(70, input_shape=(len(train_x[0]),),activation='relu',kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
model.add(Dropout(0.5))
# model.add(Dense(75,activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dropout(0.6))
# model.add(Dense(512,activation='relu',kernel_regularizer=l2(0.01)))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(np.array(train_x1), np.array(train_y1), epochs=1000, batch_size=5, verbose=1, validation_split=0.1)
model.save('chatbot_model.h5')
print("Model saved")

loss, accuracy = model.evaluate(np.array(test_x), np.array(test_y))
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

mymodel = load_model('chatbot_model.h5')
tag_list = []

def predict_class(sentence, words, model):
    ERROR_THRESHOLD = 0.25
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    for i, r in enumerate(res):
        if r > ERROR_THRESHOLD:
            results = [[i, r]]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            tag_list.append(return_list[0]['intent'])
            print(return_list)
            return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = respond(list_of_intents, tag_list, tag)
    return result

def chatbot_response(text):
    ints = predict_class(text, words, mymodel)
    res = getResponse(ints, data)
    return res

import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plotează graficul antrenării
plot_training_history(history)

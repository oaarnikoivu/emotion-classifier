import os
import pickle

basepath = os.path.abspath("./models/")


def load_tfidf_model():
    # load models
    with open(basepath + '/tfidf/logistic_anger.pkl', 'rb') as logistic_anger_file:
        logistic_anger_model = pickle.load(logistic_anger_file)
    with open(basepath + '/tfidf/logistic_anticipation.pkl', 'rb') as logistic_anticipation_file:
        logistic_anticipation_model = pickle.load(logistic_anticipation_file)
    with open(basepath + '/tfidf/logistic_disgust.pkl', 'rb') as logistic_disgust_file:
        logistic_disgust_model = pickle.load(logistic_disgust_file)
    with open(basepath + '/tfidf/logistic_fear.pkl', 'rb') as logistic_fear_file:
        logistic_fear_model = pickle.load(logistic_fear_file)
    with open(basepath + '/tfidf/logistic_joy.pkl', 'rb') as logistic_joy_file:
        logistic_joy_model = pickle.load(logistic_joy_file)
    with open(basepath + '/tfidf/logistic_love.pkl', 'rb') as logistic_love_file:
        logistic_love_model = pickle.load(logistic_love_file)
    with open(basepath + '/tfidf/logistic_optimism.pkl', 'rb') as logistic_optimism_file:
        logistic_optimism_model = pickle.load(logistic_optimism_file)
    with open(basepath + '/tfidf/logistic_pessimism.pkl', 'rb') as logistic_pessimism_file:
        logistic_pessimism_model = pickle.load(logistic_pessimism_file)
    with open(basepath + '/tfidf/logistic_sadness.pkl', 'rb') as logistic_sadness_file:
        logistic_sadness_model = pickle.load(logistic_sadness_file)
    with open(basepath + '/tfidf/logistic_surprise.pkl', 'rb') as logistic_surprise_file:
        logistic_surprise_model = pickle.load(logistic_surprise_file)
    with open(basepath + '/tfidf/logistic_trust.pkl', 'rb') as logistic_trust_file:
        logistic_trust_model = pickle.load(logistic_trust_file)

    return {
        'anger': logistic_anger_model,
        'anticipation': logistic_anticipation_model,
        'disgust': logistic_disgust_model,
        'fear': logistic_fear_model,
        'joy': logistic_joy_model,
        'love': logistic_love_model,
        'optimism': logistic_optimism_model,
        'pessimism': logistic_pessimism_model,
        'sadness': logistic_sadness_model,
        'surprise': logistic_surprise_model,
        'trust': logistic_trust_model
    }
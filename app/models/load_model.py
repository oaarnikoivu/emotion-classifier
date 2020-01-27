import os
import pickle

basepath = os.path.abspath("./models/")


def load_bert_model():
    with open(basepath + '/bert/log_bert_anger.pkl', 'rb') as log_anger_file:
        log_anger_model = pickle.load(log_anger_file)
    with open(basepath + '/bert/log_bert_anticipation.pkl', 'rb') as log_anticipation_file:
        log_anticipation_model = pickle.load(log_anticipation_file)
    with open(basepath + '/bert/log_bert_disgust.pkl', 'rb') as log_disgust_file:
        log_disgust_model = pickle.load(log_disgust_file)
    with open(basepath + '/bert/log_bert_fear.pkl', 'rb') as log_fear_file:
        log_fear_model = pickle.load(log_fear_file)
    with open(basepath + '/bert/log_bert_joy.pkl', 'rb') as log_joy_file:
        log_joy_model = pickle.load(log_joy_file)
    with open(basepath + '/bert/log_bert_love.pkl', 'rb') as log_love_file:
        log_love_model = pickle.load(log_love_file)
    with open(basepath + '/bert/log_bert_optimism.pkl', 'rb') as log_optimism_file:
        log_optimism_model = pickle.load(log_optimism_file)
    with open(basepath + '/bert/log_bert_pessimism.pkl', 'rb') as log_pessimism_file:
        log_pessimism_model = pickle.load(log_pessimism_file)
    with open(basepath + '/bert/log_bert_sadness.pkl', 'rb') as log_sadness_file:
        log_sadness_model = pickle.load(log_sadness_file)
    with open(basepath + '/bert/log_bert_surprise.pkl', 'rb') as log_surprise_file:
        log_surprise_model = pickle.load(log_surprise_file)
    with open(basepath + '/bert/log_bert_trust.pkl', 'rb') as log_trust_file:
        log_trust_model = pickle.load(log_trust_file)

    return {
        'anger': log_anger_model,
        'anticipation': log_anticipation_model,
        'disgust': log_disgust_model,
        'fear': log_fear_model,
        'joy': log_joy_model,
        'love': log_love_model,
        'optimism': log_optimism_model,
        'pessimism': log_pessimism_model,
        'sadness': log_sadness_model,
        'surprise': log_surprise_model,
        'trust': log_trust_model
    }


def load_tfidf_model():
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

import os
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
basepath = os.path.abspath("./")

# load vectorizer
with open(basepath + '/models/vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)

# load models
with open(basepath + '/models/logistic_anger.pkl', 'rb') as logistic_anger_file:
    logistic_anger_model = pickle.load(logistic_anger_file)
with open(basepath + '/models/logistic_anticipation.pkl', 'rb') as logistic_anticipation_file:
    logistic_anticipation_model = pickle.load(logistic_anticipation_file)
with open(basepath + '/models/logistic_disgust.pkl', 'rb') as logistic_disgust_file:
    logistic_disgust_model = pickle.load(logistic_disgust_file)
with open(basepath + '/models/logistic_fear.pkl', 'rb') as logistic_fear_file:
    logistic_fear_model = pickle.load(logistic_fear_file)
with open(basepath + '/models/logistic_joy.pkl', 'rb') as logistic_joy_file:
    logistic_joy_model = pickle.load(logistic_joy_file)
with open(basepath + '/models/logistic_love.pkl', 'rb') as logistic_love_file:
    logistic_love_model = pickle.load(logistic_love_file)
with open(basepath + '/models/logistic_optimism.pkl', 'rb') as logistic_optimism_file:
    logistic_optimism_model = pickle.load(logistic_optimism_file)
with open(basepath + '/models/logistic_pessimism.pkl', 'rb') as logistic_pessimism_file:
    logistic_pessimism_model = pickle.load(logistic_pessimism_file)
with open(basepath + '/models/logistic_sadness.pkl', 'rb') as logistic_sadness_file:
    logistic_sadness_model = pickle.load(logistic_sadness_file)
with open(basepath + '/models/logistic_surprise.pkl', 'rb') as logistic_surprise_file:
    logistic_surprise_model = pickle.load(logistic_surprise_file)
with open(basepath + '/models/logistic_trust.pkl', 'rb') as logistic_trust_file:
    logistic_trust_model = pickle.load(logistic_trust_file)


@app.route('/')
def my_form():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def form_post():
    """
        Takes text submitted by user, applies TF-IDF
        vectorizer to it and predicts emotion using trained logistic regression models.
    """

    text = request.form['text']

    comment_term_doc = vectorizer.transform([text])

    dict_preds = {'pred_anger': logistic_anger_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_anticipation': logistic_anticipation_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_disgust': logistic_disgust_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_fear': logistic_fear_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_joy': logistic_joy_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_love': logistic_love_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_optimism': logistic_optimism_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_pessimism': logistic_pessimism_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_sadness': logistic_sadness_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_surprise': logistic_sadness_model.predict_proba(comment_term_doc)[:, 1][0],
                  'pred_trust': logistic_trust_model.predict_proba(comment_term_doc)[:, 1][0]
                  }

    max_val = max(dict_preds, key=dict_preds.get)

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = '{0:.2f}%'.format(perc)

    return render_template('main.html', text=text,
                           pred_anger=dict_preds['pred_anger'],
                           pred_anticipation=dict_preds['pred_anticipation'],
                           pred_disgust=dict_preds['pred_disgust'],
                           pred_fear=dict_preds['pred_fear'],
                           pred_joy=dict_preds['pred_joy'],
                           pred_love=dict_preds['pred_love'],
                           pred_optimism=dict_preds['pred_optimism'],
                           pred_pessimism=dict_preds['pred_pessimism'],
                           pred_sadness=dict_preds['pred_sadness'],
                           pred_surprise=dict_preds['pred_surprise'],
                           pred_trust=dict_preds['pred_trust'],
                           emotion_val=max_val)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

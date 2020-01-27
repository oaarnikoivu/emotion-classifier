import os
import pickle
import torch
import transformers as ppb
from models.load_model import load_tfidf_model
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
basepath = os.path.abspath("./")

# load vectorizer
with open(basepath + '/models/tfidf/vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)


@app.route('/emotions')
def show_emotions():
    my_dict = {
        'Test': 'test'
    }
    return jsonify(my_dict)


@app.route('/')
def my_form():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def form_post():
    """
        Takes text submitted by user, applies TF-IDF
        vectorizer to it and predicts emotion using trained logistic regression models.
    """

    text = request.json

    comment_term_doc = vectorizer.transform([text])

    dict_preds = {'pred_anger': load_tfidf_model()['anger'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_anticipation': load_tfidf_model()['anticipation'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_disgust': load_tfidf_model()['disgust'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_fear': load_tfidf_model()['fear'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_joy': load_tfidf_model()['joy'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_love': load_tfidf_model()['love'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_optimism': load_tfidf_model()['optimism'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_pessimism': load_tfidf_model()['pessimism'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_sadness': load_tfidf_model()['sadness'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_surprise': load_tfidf_model()['surprise'].predict_proba(comment_term_doc)[:, 1][0],
                  'pred_trust': load_tfidf_model()['trust'].predict_proba(comment_term_doc)[:, 1][0]
                  }

    prediction = max(dict_preds.keys(), key=(lambda k: dict_preds[k]))

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = '{0:.2f}%'.format(perc)

    return jsonify(prediction)

    # return render_template('main.html', text=text,
    #                        pred_anger=dict_preds['pred_anger'],
    #                        pred_anticipation=dict_preds['pred_anticipation'],
    #                        pred_disgust=dict_preds['pred_disgust'],
    #                        pred_fear=dict_preds['pred_fear'],
    #                        pred_joy=dict_preds['pred_joy'],
    #                        pred_love=dict_preds['pred_love'],
    #                        pred_optimism=dict_preds['pred_optimism'],
    #                        pred_pessimism=dict_preds['pred_pessimism'],
    #                        pred_sadness=dict_preds['pred_sadness'],
    #                        pred_surprise=dict_preds['pred_surprise'],
    #                        pred_trust=dict_preds['pred_trust'],
    #                        emotion_val=max_val)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

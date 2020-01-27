import os
import pickle

import torch

from transformers import DistilBertModel, DistilBertTokenizer
from flask import Flask, request, render_template, jsonify

from models.load_model import load_bert_model, load_tfidf_model

app = Flask(__name__)
basepath = os.path.abspath("./")

pretrained_weights = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
model = DistilBertModel.from_pretrained(pretrained_weights)

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

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    with torch.no_grad():
        last_hidden_states = model(input_ids)
        features = last_hidden_states[0][0:, 0, :].numpy()


    # comment_term_doc = vectorizer.transform([text])
    #
    # dict_preds = {'pred_anger': load_tfidf_model()['anger'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_anticipation': load_tfidf_model()['anticipation'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_disgust': load_tfidf_model()['disgust'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_fear': load_tfidf_model()['fear'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_joy': load_tfidf_model()['joy'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_love': load_tfidf_model()['love'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_optimism': load_tfidf_model()['optimism'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_pessimism': load_tfidf_model()['pessimism'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_sadness': load_tfidf_model()['sadness'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_surprise': load_tfidf_model()['surprise'].predict_proba(comment_term_doc)[:, 1][0],
    #               'pred_trust': load_tfidf_model()['trust'].predict_proba(comment_term_doc)[:, 1][0]
    #               }

    dict_preds = {'pred_anger': load_bert_model()['anger'].predict_proba(features)[:, 1][0],
                  'pred_anticipation': load_bert_model()['anticipation'].predict_proba(features)[:, 1][0],
                  'pred_disgust': load_bert_model()['disgust'].predict_proba(features)[:, 1][0],
                  'pred_fear': load_bert_model()['fear'].predict_proba(features)[:, 1][0],
                  'pred_joy': load_bert_model()['joy'].predict_proba(features)[:, 1][0],
                  'pred_love': load_bert_model()['love'].predict_proba(features)[:, 1][0],
                  'pred_optimism': load_bert_model()['optimism'].predict_proba(features)[:, 1][0],
                  'pred_pessimism': load_bert_model()['pessimism'].predict_proba(features)[:, 1][0],
                  'pred_sadness': load_bert_model()['sadness'].predict_proba(features)[:, 1][0],
                  'pred_surprise': load_bert_model()['surprise'].predict_proba(features)[:, 1][0],
                  'pred_trust': load_bert_model()['trust'].predict_proba(features)[:, 1][0]
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

import os
import pickle

import torch
from flask import Flask, request, jsonify
from collections import Counter

from models.load_model import load_bert_model

app = Flask(__name__)
basepath = os.path.abspath("./")

# load vectorizer
with open(basepath + '/models/tfidf/vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)

# load distilbert tokenizer
with open(basepath + '/models/bert/distilbert_tokenizer.pkl', 'rb') as tz_file:
    tokenizer = pickle.load(tz_file)

# load distilbert model
with open(basepath + '/models/bert/distilbert_model.pkl', 'rb') as db_model_file:
    model = pickle.load(db_model_file)


@app.route('/update_preds', methods=['POST'])
def update_predictions():
    updated_preds = request.json
    print(updated_preds)
    return jsonify('Received!')


@app.route('/', methods=['POST'])
def form_post():
    text = request.json

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    with torch.no_grad():
        last_hidden_states = model(input_ids)
        features = last_hidden_states[0][0:, 0, :].numpy()

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

    c = Counter(dict_preds)
    mc = c.most_common(5)

    return jsonify(mc)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

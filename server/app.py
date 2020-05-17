import numpy as np
import torch
import transformers
import csv

from flask import Flask, jsonify, request
from transformers import BertModel, BertTokenizer
from flask_cors import CORS
from collections import Counter
from models.attention.attention_lstm import AttentionBiLSTM
from args import args

app = Flask(__name__)
CORS(app)

LOCAL_MODEL_PATH = '/Users/olive/github/dissertation/server/models/attention/bert-lstm-model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_COLS = ['pred_anger', 'pred_anticipation', 'pred_disgust', 'pred_fear', 'pred_joy',
              'pred_love', 'pred_optimism', 'pred_pessimism', 'pred_sadness', 'pred_surprise', 'pred_trust']

tokenizer = BertTokenizer.from_pretrained(args['bert_tokenizer'])

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id

max_input_length = tokenizer.max_model_input_sizes[args['bert_tokenizer']]

model = AttentionBiLSTM(
    hidden_size=args['hidden_size'],
    num_layers=args['num_layers'],
    dropout=args['dropout'],
    fc_dropout=args['fc_dropout'],
    emb_layer_dropout=args['embed_dropout'],
    num_classes=args['output_dim'],
)


model.load_state_dict(torch.load(LOCAL_MODEL_PATH,
                                 map_location='cpu'))


def predict_emotion(tweet, model, tokenizer, max_input_length, init_token_idx, eos_token_idx):
    preds = []
    model.eval()
    tokens = tokenizer.tokenize(tweet)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + \
        tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    predictions, attn_weights = model(tensor)
    preds.append(torch.sigmoid(predictions).detach().cpu().numpy())
    return preds, attn_weights, tokens


@app.route('/update_preds', methods=['POST'])
def update_predictions():
    updated_preds = request.json

    prediction = updated_preds['id']
    text = updated_preds['text']
    is_correct = updated_preds['correct']

    with open('new_data.csv', 'a') as f:
        writer = csv.writer(f)
        if prediction == 'pred_anger' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Anger')])
        if prediction == 'pred_anticipation' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Anticipation')])
        if prediction == 'pred_disgust' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Disgust')])
        if prediction == 'pred_fear' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Fear')])
        if prediction == 'pred_joy' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Joy')])
        if prediction == 'pred_love' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Love')])
        if prediction == 'pred_optimism' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Optimism')])
        if prediction == 'pred_pessimism' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Pessimism')])
        if prediction == 'pred_sadness' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Sadness')])
        if prediction == 'pred_surprise' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Surprise')])
        if prediction == 'pred_trust' and is_correct:
            writer.writerow(["{} {}".format(text.encode(), 'Trust')])

    return jsonify('Updated!')


@app.route('/')
def server_home_page():
    return 'Hello, world!'


@app.route('/predictions', methods=['POST'])
def form_post():

    text = request.json[0]
    text_len = request.json[1]

    preds, attn_weights, tokens = predict_emotion(
        text, model, tokenizer, max_input_length, init_token_idx, eos_token_idx)

    pred_values = []
    for p in preds[0]:
        for v in p:
            pred_values.append(v)

    dict_preds = {}

    for i, label in enumerate(LABEL_COLS):
        dict_preds[LABEL_COLS[i]] = float(str(pred_values[i]))

    pred_c = Counter(dict_preds)
    mc_preds = pred_c.most_common(3)

    attention_weights = []
    for aw in attn_weights[0]:
        for v in aw:
            attention_weights.append(v.detach().cpu().numpy())

    attention_weights = np.array(attention_weights[1:-1])

    attn_dict = {}
    for i in range(len(attention_weights)):
        attn_dict[tokens[i]] = float(str(attention_weights[i]))

    weight_c = Counter(attn_dict)
    if text_len >= 15:
        mc_weights = weight_c.most_common(8)
    else:
        mc_weights = weight_c.most_common(4)

    return jsonify(mc_preds, mc_weights)


if __name__ == '__main__':
    # Local
    app.run(port=5000, debug=True)
    # Server
    # app.run(host='0.0.0.0', debug=True)

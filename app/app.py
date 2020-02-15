import os
import boto3
import numpy as np


import torch
from flask import Flask, request, jsonify
from collections import Counter
from transformers import BertModel, BertTokenizer
from models.attention.attention_lstm import AttentionBiLSTM

app = Flask(__name__)
basepath = os.path.abspath("./")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert = BertModel.from_pretrained('bert-base-uncased')

model = AttentionBiLSTM(
    bert=bert,
    hidden_size=768,
    num_layers=2,
    dropout=0.5,
    fc_dropout=0.5,
    embed_dropout=0.2,
    num_classes=11
)

s3 = boto3.resource('s3')
s3.Bucket('attentionlstm').download_file('bert-lstm-model.pt', '/tmp/model.pth')

model.load_state_dict(torch.load('/tmp/model.pth'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

LABEL_COLS = ['pred_anger', 'pred_anticipation', 'pred_disgust', 'pred_fear', 'pred_joy',
              'pred_love', 'pred_optimism', 'pred_pessimism', 'pred_sadness', 'pred_surprise', 'pred_trust']


def predict_emotion(tweet):
    preds = []
    model.eval()
    tokens = tokenizer.tokenize(tweet)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    predictions, attn_weights = model(tensor)
    preds.append(torch.sigmoid(predictions).detach().cpu().numpy())
    return preds, attn_weights, tokens


@app.route('/update_preds', methods=['POST'])
def update_predictions():
    updated_preds = request.json
    print(updated_preds)
    return jsonify('Received!')


@app.route('/', methods=['POST'])
def form_post():
    text = request.json[0]
    text_len = request.json[1]

    preds, attn_weights, tokens = predict_emotion(text)

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
    app.run(port=5000, debug=True)

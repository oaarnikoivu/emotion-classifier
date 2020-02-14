import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

from build_dataset import build_dataset
from evaluate import evaluate
from model.text_cnn import BertCNN
from model.attention_lstm import AttentionLSTM
from train import train
from utils import epoch_time
from opts import parse_opt

opt = parse_opt()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(opt.bert_model)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes[opt.bert_model]

DATA_PATH = Path('/Users/olive/github/dissertation/experiments/data/')

LABEL_COLS = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
              'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']


def tokenize_and_cut(tweet):
    tokens = tokenizer.tokenize(tweet)
    tokens = tokens[:max_input_length-2]
    return tokens


train_iterator, valid_iterator, test_iterator = build_dataset(
    tokenize=tokenize_and_cut,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=init_token_idx,
    eos_token=eos_token_idx,
    pad_token=pad_token_idx,
    unk_token=unk_token_idx,
    data_path=DATA_PATH,
    batch_size=opt.batch_size,
    device=device
)

# Batch wrapper
iaux = 0
for batch in valid_iterator:
    iaux += 1
    aux = batch
    aux2 = torch.stack([getattr(batch, label) for label in LABEL_COLS])
    if aux == 20:
        break

bert = BertModel.from_pretrained(opt.bert_model)

# model = BertCNN(
#     bert=bert,
#     opt=opt
# )

model = AttentionLSTM(bert=bert, opt=opt)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def run():
    best_valid_loss = float('inf')

    train_history = []
    valid_history = []

    for epoch in range(opt.epochs):

        start_time = time.time()

        train_loss, train_metrics = train(
            model, train_iterator, optimizer, criterion, LABEL_COLS)
        valid_loss, valid_metrics = evaluate(
            model, valid_iterator, criterion, LABEL_COLS)

        train_history.append(train_loss)
        valid_history.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'bert-cnn-model.pt')

        train_acc = train_metrics['acc']
        train_micro = train_metrics['f1_micro']
        train_macro = train_metrics['f1_macro']

        valid_acc = valid_metrics['acc']
        valid_micro = valid_metrics['f1_micro']
        valid_macro = valid_metrics['f1_macro']

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1 Micro: {train_micro*100:.2f}'
              f'% | Train F1 Macro: {train_macro*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%  | Val. F1 Micro: {valid_micro*100:.2f}'
            f'%  | Val. F1 Macro: {valid_macro*100:.2f}%')


if __name__ == "__main__":
    run()

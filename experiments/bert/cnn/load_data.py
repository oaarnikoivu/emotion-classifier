import torch
from transformers import BertTokenizer
from torchtext import data
from pathlib import Path
from .utils import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(args['bert_model'])

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes[args['bert_model']]


def tokenize(text):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_input_length - 2]
    return tokens


def load_dataset():

    text = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    label = data.LabelField(sequential=False,
                            use_vocab=False,
                            pad_token=None,
                            unk_token=None,
                            dtype=torch.float)

    data_fields = {"Tweet": ("Tweet", text),
                   'anger': ("anger", label),
                   'anticipation': ("anticipation", label),
                   'disgust': ("disgust", label),
                   'fear': ("fear", label),
                   'joy': ("joy", label),
                   'love': ("love", label),
                   'optimism': ("optimism", label),
                   'pessimism': ("pessimism", label),
                   'sadness': ("sadness", label),
                   'surprise': ("surprise", label),
                   'trust': ("trust", label)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path=Path(args['data_dir']),
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        format='csv',
        fields=data_fields
    )

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: len(x.Tweet),
        sort_within_batch=True,
        batch_size=args['batch_size'],
        device=device
    )

    return train_iterator, valid_iterator, test_iterator

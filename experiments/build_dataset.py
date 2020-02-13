import torch
from torchtext import data


def build_dataset(tokenize, preprocessing, init_token, eos_token, pad_token, unk_token, data_path, batch_size, device):
    TEXT = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=tokenize,
        preprocessing=preprocessing,
        init_token=init_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token
    )

    LABEL = data.LabelField(sequential=False,
                            use_vocab=False,
                            pad_token=None,
                            unk_token=None,
                            dtype=torch.float)

    dataFields = {"Tweet": ("Tweet", TEXT),
                  'anger': ("anger", LABEL),
                  'anticipation': ("anticipation", LABEL),
                  'disgust': ("disgust", LABEL),
                  'fear': ("fear", LABEL),
                  'joy': ("joy", LABEL),
                  'love': ("love", LABEL),
                  'optimism': ("optimism", LABEL),
                  'pessimism': ("pessimism", LABEL),
                  'sadness': ("sadness", LABEL),
                  'surprise': ("surprise", LABEL),
                  'trust': ("trust", LABEL)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        format='csv',
        fields=dataFields
    )

    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of validation examples: {len(valid_data)}")
    print(f"Number of testing examples: {len(test_data)}")

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: len(x.Tweet),
        sort_within_batch=True,
        batch_size=batch_size,
        device=device
    )

    return train_iterator, valid_iterator, test_iterator

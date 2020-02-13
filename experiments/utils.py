args = {
    "bert_model": "bert-base-uncased",
    "seed": 1234,
    "batch_size": 64,
    "num_filters": 100,
    "filter_sizes": [3, 4, 5],
    "output_dim": 11,
    "dropout": 0.5,
    "epochs": 10
}


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

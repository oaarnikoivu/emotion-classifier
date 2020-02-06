import time

import torch
from torch import optim, nn

from bert.cnn.evaluate import evaluate
from bert.cnn.load_data import load_dataset
from bert.cnn.model import BertTextCNN
from bert.cnn.train import train
from bert.cnn.utils import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = load_dataset()

iaux = 0
for batch in valid_iterator:
    iaux += 1
    aux = batch
    aux2 = torch.stack([getattr(batch, label) for label in args['label_cols']])
    if aux == 20:
        break

model = BertTextCNN(n_filters=args['num_filters'],
                    filter_sizes=args['filter_sizes'],
                    output_dim=args['output_dim'],
                    dropout=args['dropout'])

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def predict():
    best_valid_loss = float('inf')

    train_history = []
    valid_history = []

    for epoch in range(args['epochs']):

        start_time = time.time()

        train_loss, train_metrics = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_metrics = evaluate(model, valid_iterator, criterion)

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

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% '
            f'| Train F1 Micro: {train_micro * 100:.2f}% | Train F1 Macro: {train_macro * 100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%  '
            f'| Val. F1 Micro: {valid_micro * 100:.2f}%  | Val. F1 Macro: {valid_macro * 100:.2f}%')


if __name__ == '__main__':
    predict()

    model.load_state_dict(torch.load('bert-cnn-model.pt'))

    test_loss, test_metrics = evaluate(model, test_iterator, criterion)

    test_acc = test_metrics['acc']
    test_micro = test_metrics['f1_micro']
    test_macro = test_metrics['f1_macro']

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% '
          f'| Test F1 Micro: {test_micro * 100:.2f}% | Test F1 Macro: {test_macro * 100:.2f}%')


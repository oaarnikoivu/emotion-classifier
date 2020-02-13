import numpy as np
import torch

from metrics import metricize


def train(model, iterator, optimizer, criterion, label_cols):
    print('\nTraining...')

    epoch_loss = 0

    model.train()

    preds_list = []
    labels_list = []

    for i, batch in enumerate(iterator):

        optimizer.zero_grad()

        predictions = model(batch.Tweet).squeeze(1)

        batch_labels = torch.stack([getattr(batch, label)
                                    for label in label_cols])
        batch_labels = torch.transpose(batch_labels, 0, 1)

        loss = criterion(predictions, batch_labels)

        loss.backward()

        optimizer.step()

        preds_list += [torch.sigmoid(predictions).detach().cpu().numpy()]
        labels_list += [batch_labels.cpu().numpy()]

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), metricize(np.vstack(preds_list),
                                                 np.vstack(labels_list))

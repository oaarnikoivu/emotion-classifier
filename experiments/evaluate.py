import torch
import numpy as np

from metrics import metricize


def evaluate(model, iterator, criterion, label_cols):
    print('\nEvaluating...')

    epoch_loss = 0

    model.eval()

    preds_list = []
    labels_list = []
    attns = []

    with torch.no_grad():

        for batch in iterator:
            predictions = model(batch.Tweet).squeeze(1)

            batch_labels = torch.stack([getattr(batch, label)
                                        for label in label_cols])
            batch_labels = torch.transpose(batch_labels, 0, 1)

            loss = criterion(predictions, batch_labels)

            epoch_loss += loss.item()

            preds_list += [torch.sigmoid(predictions).detach().cpu().numpy()]
            labels_list += [batch_labels.cpu().numpy()]

    return epoch_loss / len(iterator), metricize(np.vstack(preds_list),
                                                 np.vstack(labels_list))

from sklearn.metrics import f1_score, roc_auc_score


def metricize(preds, y):
    f1_macro = f1_score(y, preds.round(), average='macro')
    f1_micro = f1_score(y, preds.round(), average='micro')
    acc = roc_auc_score(y, preds)

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'acc': acc
    }

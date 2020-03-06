import numpy as np
from sklearn import metrics


def MacroF1Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = metrics.f1_score(labels, preds, average='macro')
    return ('MacroF1Metric', score, True)


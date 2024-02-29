import numpy as np


def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    # print(threshold)
    return threshold
from sklearn.metrics import confusion_matrix, precision_recall_curve
def find_threshold_micro_v2(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]

    print(dev_yhat_raw.shape)
    print(dev_y.shape)
    best_thresholds = []
    alphas = []
    for cat_id in range(dev_y.shape[1]):
        predict_one = dev_yhat_raw[:, cat_id]
        target_one = dev_y[:, cat_id]
        precision, recall, thresholds = precision_recall_curve(target_one, predict_one)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-9)
        best_thresholds.append(thresholds[np.argmax(f1_scores)])
        alphas.append(np.sum(target_one))
    alphas = np.array(alphas, dtype=float)
    alphas /= np.max(alphas)
    best_thresholds = np.array(best_thresholds)
    best_thresholds = alphas*best_thresholds+(1-alphas)*threshold
    return best_thresholds
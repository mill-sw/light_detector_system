import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tensorflow.math import confusion_matrix


def draw_CM(label, predicted):
    cm = confusion_matrix(label, predicted)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # true : false rate
    true = 0
    false = 0
    for i, j in enumerate(label):
        if j != predicted[i]:
            false += 1
        else:
            true += 1

    classification_report = metrics.classification_report(label, predicted)
    multilabel_to_binary_matrics = metrics.multilabel_confusion_matrix(label, predicted)

    return plt.show(), print('true rate: ', true), print('false rate: ', false), print(), \
           print('=' * 10, 'classification_report: ', '\n', classification_report), \
           print('=' * 10, 'multilabel_to_binary_matrics by class_num: ', '\n', '[[TN / FP] [FN / TP]]', '\n',
                 multilabel_to_binary_matrics)

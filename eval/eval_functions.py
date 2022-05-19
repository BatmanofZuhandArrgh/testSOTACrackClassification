import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues, save_path = None):
    #cm output from sklearn.confusion_matrix
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, np.round(cm[i, j], 2),
                          ha="center", va="center", color="k")
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)

def iou(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    
def evaluate(y_truth, y_pred, labels, 
        save_confusion_matrix_path = 'evaluation_results/sample.png', 
        save_classification_report_path = 'evaluation_results/sample.csv'
        ): 
    labels = [str(x) for x in labels]

    print('Classification Report')
    cls_report = classification_report(y_truth, y_pred,labels = [x for x in range(len(labels))], target_names=labels)
    if save_classification_report_path != None:
        df = pd.DataFrame(cls_report).transpose()
        df.to_csv(save_classification_report_path)
    print(cls_report)

    print('Confusion Matrix')
    plot_confusion_matrix(confusion_matrix(y_truth, y_pred, normalize = 'true'), labels = labels, save_path = save_confusion_matrix_path)

if __name__ == '__main__':
    y_test = [0,1,0]
    y_pred = [1,0,0]
    labels = np.unique(y_test+y_pred).tolist()
    evaluate(y_test, y_pred, labels)

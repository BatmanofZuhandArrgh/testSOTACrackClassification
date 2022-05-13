import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.show()
    
def evaluate(y_truth, y_pred, labels): 
    labels = [str(x) for x in labels]
    print('Classification Report')
    print(classification_report(y_truth, y_pred,labels = [x for x in range(len(labels))], target_names=labels))
    print('Confusion Matrix')
    plot_confusion_matrix(confusion_matrix(y_truth, y_pred, normalize = 'true'), labels = labels)

if __name__ == '__main__':
    y_test = [0,1,0]
    y_pred = [1,0,0]
    labels = np.unique(y_test+y_pred).tolist()
    evaluate(y_test, y_pred, labels)

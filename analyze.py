import os
import cv2
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import compress
from tqdm import tqdm

from eval.eval_functions import evaluate
from sklearn.metrics import f1_score


def regression_chart(label_values,plot_save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,10))

    # creating the bar plot
    bins= [x for x in range(min(label_values),max(label_values),int((max(label_values) - min(label_values))/25))]
    counts, edges, bars = ax1.hist(label_values, bins=bins, edgecolor="k")
    ax1.bar_label(bars)
    ax1.set_xticks(bins)
    ax1.set_xticklabels(bins, rotation = 75)
    
    ax2.boxplot(label_values)
    # fig.show()
    # plt.show()
    fig.savefig(plot_save_path)

def EDA_regression(label_values, plot_save_path):
    print(f'Mean: {sum(label_values)/len(label_values)}')
    print(f'StdDev: {np.std(label_values)}')
    print(f'Max: {max(label_values)}') 
    print(f'Min: {min(label_values)}')
    regression_chart(label_values, plot_save_path)

def sort_by_file_name(value_list):
    name_list = [os.path.basename(os.path.splitext(x)[0]) for x in value_list]
    name_list = [x.split('_fused')[0] if 'fused' in x else x for x in name_list ]
    new_list = [x for _,x in sorted(zip(name_list,value_list))]
    return new_list

def get_pos_pred(pred_list, conf_threshold):
    num_pos_pred = []
    for pred_path in tqdm(pred_list):
        pred = cv2.imread(pred_path)[:,:,0]
        pred = pred[pred>conf_threshold]
        values, counts = np.unique(pred, return_counts=True)
        num_pos_pred.append(sum(counts))
    return num_pos_pred

def test_conf_threshold(cracked_train, not_cracked_train, list_of_threshold = np.arange(0,1, 0.1)):
    '''
    Get the threshold for confidence score that will maximize 
    avg count of crack pixels cracked_train, and avg count of crack pixels in not_cracked_train
    list_of_threshold
    '''

    list_of_pixel_threshold = [255*x for x in list_of_threshold]
    avg_count_cracked = []
    avg_count_noncracked = []
    print("Finding best conf_score threshold: ")
    for threshold in tqdm(list_of_pixel_threshold):
        num_pos_pred_cracked = get_pos_pred(cracked_train, threshold)
        num_pos_pred_noncracked = get_pos_pred(not_cracked_train, threshold)

        avg_count_cracked.append(sum(num_pos_pred_cracked)/len(num_pos_pred_cracked))
        avg_count_noncracked.append(sum(num_pos_pred_noncracked)/len(num_pos_pred_noncracked))
    print(np.argmax(avg_count_cracked), np.argmin(avg_count_noncracked))
    print('avg counts of positive pred in cracked img: ', avg_count_cracked)
    print('avg counts of positive pred in noncracked img: ', avg_count_noncracked)
    diff = np.array(avg_count_cracked) - np.array(avg_count_noncracked) 
    largest_diff_index = np.argmax(diff)
    return list_of_threshold[largest_diff_index]

binary_thresholding = lambda value_list, threshold : [1 if x>threshold else 0 for x in value_list]

def test_pixcount_threshold(bin_train_pixcount_groundtruth, train_pixcount_pred, list_of_pixcount_thresholds):
    f1s = [] 
    print("Finding pixel count threshold for a correct prediction")
    for threshold in tqdm(list_of_pixcount_thresholds):
        bin_train_pixcount_pred = binary_thresholding(train_pixcount_pred, threshold)
        f1s.append(f1_score(bin_train_pixcount_groundtruth, bin_train_pixcount_pred, 'micro'))
    max_f1_index = np.argmax(f1s)
    print('Best f1 is: ', max(f1s))
    return list_of_pixcount_thresholds[max_f1_index]

def evaluate_classification(pred_paths, labels, conf_threshold, pixcount_threshold, save_confusion_matrix_path, save_classification_report_path):
    pred_pixcount = get_pos_pred(pred_paths, conf_threshold=conf_threshold)
    assert len(pred_pixcount) == len(labels)
    pred = binary_thresholding(value_list=pred_pixcount, threshold=pixcount_threshold)
    evaluate(labels, pred, ['0','1'], 
        save_confusion_matrix_path, 
        save_classification_report_path,
        )

if __name__ == '__main__':
    imgs = glob.glob('datasets/DATA_Maguire_20180517_ALL/test_img/**/*.jpg', recursive=True)
    preds =glob.glob('results/SDNET2018/**/*fused.png', recursive=True)

    #Sort by file name
    imgs = sort_by_file_name(imgs)
    preds = sort_by_file_name(preds)
    
    #Get labels
    labels = [1 if 'U' not in x else 0 for x in imgs]
    opp_labels = [1-x for x in labels]

    #Get imgs and preds paths based on positive and negative labels
    cracked_imgs = list(compress(imgs, labels))
    not_cracked_imgs = list(compress(imgs, opp_labels))
    cracked_preds = list(compress(preds, labels))
    not_cracked_preds = list(compress(preds, opp_labels))

    #Split by 50/50 to test thresholds 
    cracked_train, cracked_test = \
        train_test_split(cracked_preds, test_size = .5, random_state=42)

    not_cracked_train, not_cracked_test = \
        train_test_split(not_cracked_preds, test_size = .5, random_state=42)
    
    #Find confidence threshold where the difference between the mean of positive pixel counts is largest
    best_conf_threshold = test_conf_threshold(cracked_train, not_cracked_train)
    print("Best confidence score threshold: ", best_conf_threshold)

    best_conf_threshold *= 255
    num_pos_pred_cracked = get_pos_pred(cracked_train, best_conf_threshold)
    num_pos_pred_noncracked = get_pos_pred(not_cracked_train, best_conf_threshold)
    
    EDA_regression(num_pos_pred_cracked, 'evaluation_results/cracked_positive_pixel_counts.png')
    EDA_regression(num_pos_pred_noncracked, 'evaluation_results/not_cracked_positive_pixel_counts.png')

    #Find the pixel count threshold where the f1_score is largest
    bin_train_pixcount_groundtruth = [1 for x in cracked_train] + [0 for x in not_cracked_train]
    train_pixcount_pred = num_pos_pred_cracked + num_pos_pred_noncracked  

    list_of_pixcount_thresholds = np.arange(0, max(train_pixcount_pred), 50)

    best_pixcount_threshold = test_pixcount_threshold(bin_train_pixcount_groundtruth, train_pixcount_pred, list_of_pixcount_thresholds)       

    print("Best pixel count threshold: ", best_pixcount_threshold)
    
    #Evaluate both train_set and test_set
    bin_test_pixcount_groundtruth = [1 for x in cracked_test] + [0 for x in not_cracked_test]
    evaluate_classification(
        cracked_test + not_cracked_test, 
        bin_test_pixcount_groundtruth, 
        best_conf_threshold, 
        best_pixcount_threshold, 
        save_confusion_matrix_path = 'evaluation_results/test_SDNet2018_cm.png', 
        save_classification_report_path = 'evaluation_results/test_SDNet2018_cr.csv')

    evaluate_classification(
        cracked_train + not_cracked_train, 
        bin_train_pixcount_groundtruth, 
        best_conf_threshold, 
        best_pixcount_threshold, 
        save_confusion_matrix_path = 'evaluation_results/train_SDNet2018_cm.png', 
        save_classification_report_path = 'evaluation_results/train_SDNet2018_cr.csv')

    evaluate_classification(
        cracked_train + not_cracked_train + cracked_test + not_cracked_test, 
        bin_train_pixcount_groundtruth + bin_test_pixcount_groundtruth, 
        best_conf_threshold, 
        best_pixcount_threshold, 
        save_confusion_matrix_path = 'evaluation_results/SDNet2018_cm.png', 
        save_classification_report_path = 'evaluation_results/SDNet2018_cr.csv')
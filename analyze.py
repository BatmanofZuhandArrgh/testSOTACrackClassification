import os
import cv2
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import compress
from tqdm import tqdm


def regression_chart(label_values):
    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    bins= [x for x in range(0,70,2)]
    counts, edges, bars = plt.hist(label_values, bins=bins, edgecolor="k")
    plt.bar_label(bars)
    plt.xticks(bins)
    plt.show()
    
    plt.boxplot(label_values)
    plt.show()

def EDA_regression(label_values):
    print(f'Mean: {sum(label_values)/len(label_values)}')
    print(f'StdDev: {np.std(label_values)}')
    print(f'Max: {max(label_values)}') 
    print(f'Min: {min(label_values)}')
    regression_chart(label_values)

def sort_by_file_name(value_list):
    name_list = [os.path.basename(os.path.splitext(x)[0]) for x in value_list]
    name_list = [x.split('_fused')[0] if 'fused' in x else x for x in name_list ]
    # print(name_list)
    new_list = [x for _,x in sorted(zip(name_list,value_list))]
    return new_list

def get_pos_pred(pred_list):
    num_pos_pred = []
    for pred_path in tqdm(pred_list):
        pred = cv2.imread(pred_path)[:,:,0]
        pred = pred[pred>threshold]
        values, counts = np.unique(pred, return_counts=True)
        num_pos_pred.append(sum(counts))
    print(len(num_pos_pred))
    return num_pos_pred

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

    #Set confidence threshold of 
    threshold = 0.85 * 255

    #Split by 50/50 to test thresholds 
    cracked_train, cracked_test = \
        train_test_split(cracked_preds, test_size = .5, random_state=42)

    not_cracked_train, not_cracked_test = \
        train_test_split(not_cracked_preds, test_size = .5, random_state=42)
    
    num_pos_pred_cracked = get_pos_pred(cracked_train)
    num_pos_pred_noncracked = get_pos_pred(not_cracked_train)
    
    # avg_num_pos_pred_noncracked = sum(num_pos_pred_noncracked)/len(num_pos_pred_noncracked)
    # num_pos_pred_cracked = sum(num_pos_pred_cracked)/len(num_pos_pred_cracked)
    # print(avg_num_pos_pred_noncracked, num_pos_pred_cracked)
    EDA_regression(num_pos_pred_noncracked)
    EDA_regression(num_pos_pred_cracked)

    
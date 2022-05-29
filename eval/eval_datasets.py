import os
import glob as glob
import cv2
import numpy as np
from tqdm import tqdm

from eval_functions import evaluate, iou

def sort_by_file_name(value_list):
    name_list = [os.path.basename(os.path.splitext(x)[0]) for x in value_list]
    name_list = [x.split('_fused')[0] if 'fused' in x else x for x in name_list ]
    # print(name_list)
    new_list = [x for _,x in sorted(zip(name_list,value_list))]
    return new_list

binary_thresholding = lambda value_list, threshold : [1 if x>threshold else 0 for x in value_list]

class Eval_dataset:
    def __init__(
        self,
        dataset_names: list,
        img_data_roots: list,
        lab_data_roots: list,
        pred_data_roots: list,
        og_ext = 'jpg', #Original image extension
        lab_ext = 'png', #Ground truth and predict image extension
        ) -> None:
        '''
        lab_data_roots: list, #List of data directory for labels
        pred_data_roots: list,#List of data directory for prediction
        img_data_roots: list, #List of data directory for images
        og_ext = 'jpg' : Original image extension
        lab_ext = 'png': Ground truth and predict image extension
        '''
        self.img_paths = [glob.glob(f'{img_data_root}/**/*.{og_ext}', recursive=True) for img_data_root in img_data_roots]
        self.pred_paths= [glob.glob(f'{pred_data_root}/**/*.{lab_ext}', recursive=True) for pred_data_root in pred_data_roots] 
        self.lab_paths = [glob.glob(f'{lab_data_root}/**/*.{lab_ext}', recursive=True) for lab_data_root in lab_data_roots]
        self.dataset_names = dataset_names

        for index in range(len(img_data_roots)):
            self.pred_paths[index] = [x for x in self.pred_paths[index] if 'fused' in x]
        
        # #Sort
        self.paths = {}
        self.paths['img'] = [sort_by_file_name(x) for x in self.img_paths]
        self.paths['pred'] = [sort_by_file_name(x) for x in self.pred_paths]
        self.paths['lab'] = [sort_by_file_name(x) for x in self.lab_paths] 

        # self.check_all_img_name_match()
    
    def check_all_img_name_match(self):
        for i in range(len(img_data_roots)):            
            for j in tqdm(range(len(self.img_paths[i]))):
                img_name = os.path.basename(os.path.splitext(self.paths['img'][i][j])[0])
                pred_name = os.path.basename(os.path.splitext(self.paths['pred'][i][j])[0])
                
                if img_name not in pred_name and pred_name not in img_name:  
                    print(self.paths['img'][i][j], self.paths['pred'][i][j])                  
                    print(img_name, pred_name)
                    raise ValueError

                img = cv2.imread(self.paths['img'][i][j])
                pred = cv2.imread(self.paths['pred'][i][j])

                assert img.shape[:2] == pred.shape[:2]
                if self.paths['pred'] != []:
                    lab = cv2.imread(self.paths['pred'][i][j])
                    assert lab.shape[:2] == pred.shape[:2]

    def evaluate_segmentation(
        self, 
        fig_save_path = 'evaluation_results/segmentation_results.png', 
        conf_score_threshold = 0.5
        ):
        for key in self.paths.keys():
            print(key, [len(x) for x in self.paths[key]])


        for i in range(len(self.paths['pred'])):
            all_pred = []
            all_label = []
            all_iou = []
            for j in tqdm(range(len(self.paths['pred'][i]))):
                pred_path = self.paths['pred'][i][j]
                label_path = self.paths['lab'][i][j]
                
                pred = cv2.imread(pred_path)[:,:,0]
                label = cv2.imread(label_path)[:,:,0]
                
                # pred = np.reshape(pred, (-1,1)).tolist()
                # label = np.reshape(label, (-1,1)).tolist()
                pred = pred.ravel()
                label= label.ravel()

                pred = binary_thresholding(pred, conf_score_threshold * 255)           
                if 255 in np.unique(label):
                    label = binary_thresholding(label, conf_score_threshold * 255)

                individual_iou = iou(target = label, prediction = pred)

                all_pred.extend(pred)
                all_label.extend(label)
                all_iou.append(individual_iou)
            
            print(self.dataset_names[i])
            print(f'Average iou: {np.mean(all_iou)}')

            dataset_fig_save_path = os.path.join(os.path.dirname(fig_save_path),  self.dataset_names[i]+ '_' + os.path.basename(fig_save_path))
            evaluate(all_label, all_pred, ['0', '1'], 
                save_confusion_matrix_path = dataset_fig_save_path, 
                save_classification_report_path = None
                )

    def evaluate_segmentation2classification(
        self,
        pixcount_threshold = 950, 
        conf_score_threshold = 0.2,
        fig_save_path = 'evaluation_results/seg2class.png', 
        ):
        all_pred = []
        total_count_imgs = 0
        for i in range(len(self.paths['pred'])):
            total_count_imgs += len(self.paths['pred'][i])
            for j in tqdm(range(len(self.paths['pred'][i]))):
                pred_path = self.paths['pred'][i][j]                
                pred = cv2.imread(pred_path)[:,:,0]

                pred = pred.ravel()
                pred = binary_thresholding(pred, conf_score_threshold * 255)
                num_positive_pred = len([x for x in pred if x == 1])
                isPredAsCrack = 1 if num_positive_pred > pixcount_threshold else 0
                all_pred.append(isPredAsCrack)
            
            # print(self.dataset_names[i])    
            # dataset_fig_save_path = os.path.join(os.path.dirname(fig_save_path),  self.dataset_names[i]+ '_' + os.path.basename(fig_save_path))
        
        all_label = [1 for x in range(total_count_imgs)]
        evaluate(all_label, all_pred, ['0', '1'], 
            save_confusion_matrix_path = fig_save_path, 
            save_classification_report_path = None
            )


if __name__ == '__main__':
    dataset_names = [
        # 'AEL',
        # 'CFD',
        # # 'SDNET2018',
        # 'CrackTree200',
        # 'CRACK500',
        'GAPs384'
    ]
    lab_data_roots = [
        # 'datasets/AEL/gt',
        # 'datasets/CFD/cfd_gt',
        # # None,
        # 'datasets/CrackTree200/cracktree200_gt',
        # 'datasets/CRACK500/test_img',
        'datasets/GAPs384/croppedgt'
    ]
    pred_data_roots = [
        # 'results/AEL',
        # 'results/CFD',
        # # 'results/SDNET2018',
        # 'results/CrackTree200',
        # 'results/CRACK500',
        'results/GAPs384',
    ]
    img_data_roots = [
        # 'datasets/AEL/test_img',
        # 'datasets/CFD/test_img',
        # # 'datasets/DATA_Maguire_20180517_ALL/test_img',
        # 'datasets/CrackTree200/test_img',
        # 'datasets/CRACK500/test_img',
        'datasets/GAPs384/test_img'
    ]
    evalDataset = Eval_dataset(
        dataset_names = dataset_names,
        img_data_roots=img_data_roots,
        lab_data_roots=lab_data_roots,
        pred_data_roots=pred_data_roots,
    )

    # evalDataset.evaluate_segmentation2classification(conf_score_threshold=0.5)
    evalDataset.evaluate_segmentation(conf_score_threshold=0.2)

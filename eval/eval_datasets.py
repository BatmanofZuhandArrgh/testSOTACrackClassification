import os
import glob as glob
import cv2
from tqdm import tqdm

def sort_by_file_name(value_list):
    name_list = [os.path.basename(os.path.splitext(x)[0]) for x in value_list]
    name_list = [x.split('_fused')[0] if 'fused' in x else x for x in name_list ]
    # print(name_list)
    new_list = [x for _,x in sorted(zip(name_list,value_list))]
    return new_list

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

            
    def evaluate(self):
        pass

if __name__ == '__main__':
    dataset_names = [
        'AEL',
        'CFD',
        'SDNET2018',
        'CrackTree200',
        'CRACK500',
        'GAPs384'
    ]
    lab_data_roots = [
        'datasets/AEL/gt',
        'datasets/CFD/cfd_gt/seg_gt',
        None,
        'datasets/CrackTree200/cracktree200_gt',
        'datasets/CRACK500/lab_img',
        'datasets/GAPs384/cropgt'
    ]
    pred_data_roots = [
        'results/AEL',
        'results/CFD',
        'results/SDNET2018',
        'results/CrackTree200',
        'results/CRACK500',
        'results/GAPs384',
    ]
    img_data_roots = [
        'datasets/AEL/test_img',
        'datasets/CFD/test_img',
        'datasets/DATA_Maguire_20180517_ALL/test_img',
        'datasets/CrackTree200/test_img',
        'datasets/CRACK500/test_img'
        'datasets/GAPs384/test_img'
    ]
    evalDataset = Eval_dataset(
        dataset_names = dataset_names,
        img_data_roots=img_data_roots,
        lab_data_roots=lab_data_roots,
        pred_data_roots=pred_data_roots,
    )

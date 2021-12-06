# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:03:13 2021

@author: abhia
"""

'''
Performs pre-processing for a given Dataset, while splitting it into Train-Val-Test
Folders before it is leveraged for the Unet segmentation model trainng
The pre-processing involves:
    (i) Load the CXRs from various sources in the given folder
    (ii) Peforms appropriate medical image enhancement equalising the whole 
    range of cxr images using image processing techniques
            - CLAHE or Histogram equalization
            - filtering/Blurring/denoising
    (iv) Resize based on appropraite interpolation
    (v) save it to the a desired directory in the following fashion:
            #Train set-
                - CXRs
                - Lung masks
            #Val set-
                - CXRs
                - Lung masks
            #Test set-
                - CXRs
                - Lung masks
    

# =============================================================================
# # Things to change-
# # 1. in_path : fixed, no need to change
# # 2. out_path : as desired
# # 3. image pre-processings : refer prep_cxr_segmtn function in Methods_img_proceess.py
# =============================================================================
   
'''

import cv2
from tqdm import tqdm 
import pandas as pd
from Methods_img_proceess import prep_cxr_segmtn, makemydir,mergeMasks



def write_in_path(ids, in_dirr, out_dirr,mask):
    '''

    Parameters
    ----------
    id_ : ID labels list of the source images
    in_dirr : home directory of the images folders
    out_dirr : path for writing the processed images

    Returns
    -------
    Returns none but write the images in a new folder in the home diretory

    '''

    makemydir(out_dirr)
    print('Pre-processing and writing the CXRs images')
    i=1
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):   
    
        name = id_ + '.png'
        image_path = in_dirr + '/' + name 
        #img = cv2.imread(image_path,0)
        img = prep_cxr_segmtn(image_path, img_width, mask)
        #plt.imshow(img, cmap='gray')
        cv2.imwrite(out_dirr + name ,img)

        i=i+1     
 

img_width = 256
img_height = 256
img_channels = 1

in_path = '/home/aa086655/Seg dataset/208_added_raw256/'

out_path = '/data/CXR/Seg dataset/phase3/CLAHE-Gausblur3_3Xaug/'


cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'

# To perform mask merging of Left and Right lungs
#mergeMasks()  

# To load the ids dorectly from the path, use this snippet
#cxr_ids = getImagesAndLabels(in_path + cxrs)
#lung_masks_ids = getImagesAndLabels(in_path + lung_masks)

# For grabbing image IDs from csv file
colnames = ["Train","Test", "Val"]
data = pd.read_csv('208_added_train_test_val_ids_cxr_masks.csv', names=colnames)
train_ids = data.Train.tolist()
val_ids = data.Val.tolist()
val_ids = [x for x in val_ids if str(x) != 'nan']
test_ids = data.Test.tolist()
test_ids = [x for x in test_ids if str(x) != 'nan']


#write_in_path(cxr_ids, in_path + cxrs, out_path+ cxrs)
#write_in_path(cxr_ids , in_path + lung_masks,out_path + lung_masks)

write_in_path(train_ids, in_path + cxrs, out_path + 'train/' + cxrs, mask = False)
write_in_path(train_ids , in_path + lung_masks ,out_path + 'train/' + lung_masks, mask = True)

write_in_path(val_ids, in_path + cxrs, out_path + 'val/' + cxrs, mask = False)
write_in_path(val_ids , in_path + lung_masks ,out_path + 'val/' + lung_masks, mask = True)
write_in_path(test_ids, in_path + cxrs, out_path + 'test/' + cxrs, mask = False)
write_in_path(test_ids , in_path + lung_masks, out_path + 'test/' + lung_masks, mask = True)


# For Augmentation
#aug_dir= path + 'Cxrs_preprocessed/aug/'
#makemydir(aug_dir)
#img = cv2.imread(path,0)
#aug( img ,aug_dir,20)


cv2.waitKey(0)
cv2.destroyAllWindows()



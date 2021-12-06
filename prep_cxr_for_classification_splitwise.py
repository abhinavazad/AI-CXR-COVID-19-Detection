# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 01:50:07 2021

@author: AA086655
"""

'''
Performs pre-processing for a given Dataset, while splitting it into Train-Val-Test
Folders before it is leveraged for the model trainng
The pre-processing involves:
    (i) Load the disoriented CXRs from various sources in the given folder
    (ii) Crop the CXR around the Thoracic/Lung region(RoI) using the self trained UNet model
    (iii) Peforms appropriate medical image enhancement using image processing techniques
            - CLAHE or Histogram equalization
            - filtering/Blurring/denoising
    (iv) Resize based on appropraite interpolation
    (v) save it to the a desired directory in the following fashion:
            #Train set:-
                - Class1
                - Class2
            #Val set:-
                - Class1
                - Class2
            #Test set:-
                - Class1
                - Class2
    
Steps:
    (i) Load the CXRs and Unet Lung segmentation model 
    (ii) Predict the Lung segmentation mask for each CXR using the Unet model
            - Set the Threshold for prediction
    (iii) Contour detection around the lung mask and crop only around the lung contours(based on area)
    (iv) Image enhancement and resizing
    (v) Save to out_path

# =============================================================================
# # Things to change-
# 1. in_path : directory of the original image datasets
# 2. out_path : directory of the output when a new folder will be created 
#               with a unique name which you need to decide.:
# 3. modelpath : path of the segementaion model to be loaded for mask predictions
# 4. post_process = True or False in case you want to post process the mask prdicitons
# 5. maskout option = True or False in the pre-processing function
# 6. Adaptive equalisation = 2nd last line of the pre-processing finction, optional to use.
# 7. dim = Output dimensions of the pre-processing, you can also comment the resizing fucniton in the end of the pre-processing funciton incase you want the original size onlt.
# 8. Choose a split type: Train/Val/Test or Train/Test 
# 9. Choose the Split ratio and tune them as per the Split type chosen
# 10.image preprocessings in Adap_resize function 
# =============================================================================

'''

import cv2
import numpy as np
import os
from tqdm import tqdm 
from skimage.transform import resize
import os.path

from keras.models import model_from_yaml

from Methods_img_proceess import adap_equalize, morph
from methods_model_training import load_seg_model, get_folders_only,  grabLungsbox, getImagesAndLabels,make_split_dir_classwise_train_val_test, make_split_dir_classwise_train_test


def adap_resize(in_path, out_dirr, idss, dim):
    '''
    Performs CLHAE and resizing for the given image ids from in_path and saves at out_path

    Parameters
    ----------
    in_path
    out_dirr
    idss : list of image ids to be pre-processed from the given in_path
    dim : resized to dim

    Returns
    -------
    Directly saves the images in out_dirr

    '''
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(idss), total=len(idss)):
        name = id_ + '.png'
        print(name)
        img_out = cv2.imread(in_path + name, 0)#[:,IMG_CHANNELS]    

        # Final Resizing of the images        
        if img_out.shape != (dim, dim):
            interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>dim else  cv2.INTER_CUBIC
            img_out = cv2.resize(img_out, (dim, dim), interpolation = interpolation_type)
        
        # Adaptive Histogram equalization
        img_out = adap_equalize(img_out)
    
        #img_out = cv2.resize(img_out, (dim, dim), interpolation = cv2.INTER_CUBIC) #for cxrs : INTER_CUBIC 
        cv2.imwrite(out_dirr  + name ,img_out)
        
        
def prep_cxr_classification(in_path, out_dirr, idss, seg_model,dim):
    """
    This is the all in one pre-processing fucntion for the classification training
    Optional features: maskout = True if you only want predicted lungs region to be visible
                     : adaptive histogram equalization
                     : Denoising using blur filters
                     

    Parameters
    ----------
    in_path : Dataset folder with all the Source images together in folders classwose
    out_dirr : Directory for the split datset
    idss : id names list of all the original images in in_path directory 
    seg_model : Lungs segmentation model
    dim : Output dimensions of the preprocessed images

    Returns
    -------
    An array of all the preprocessed images

    """
    x = np.zeros((len(idss), dim, dim), dtype=np.uint8)
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(idss), total=len(idss)):
        name = id_ + '.png'
        print(name)
        # if the images need to be pre-processed or resized!
        #img = prep_cxr_segmtn(path + name, img_width)
        image = cv2.imread(in_path + name, 0)
        #print(in_path + name)
        size = image.shape
        img = image
        if img.shape != (256, 256): #Input shape required for Mask prediction seg_model
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
        img = adap_equalize(img)
        img = np.expand_dims(np.array(img), axis = 0)
        preds = seg_model.predict(img, verbose=1)
        preds_t = (preds > 0.5).astype(np.uint8)
        preds_t = np.squeeze(preds_t)
        post_process = False
        if post_process:
                preds_t = morph(morph, 1)
                preds_t = morph(morph, 2)
                preds_t = morph(morph, 1)
        #plt.imshow(preds_t)
        #plt.show()
        
        
        mask = resize(preds_t, size , mode='constant',  preserve_range=True)
        mask = (mask > 0).astype(np.uint8)
        #plt.imshow(mask)
        #plt.show()
        img_out, ratio = grabLungsbox(image, mask, maskout=False) # maskout= False for whole CXR of the lung region
        if ratio == 0:
            print("Cropping failed for: ", name)
        '''
        # Plot and check the processing here
        img_out1 = adap_equalize(img_out1)
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(img_out, cmap='gray')
        
        plt.subplot(232)
        plt.title('After pre-processing')
        plt.imshow(img_out1, cmap='gray')
        '''
        # Final Resizing of the images
        if img_out.shape != (dim, dim):
            interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>dim else  cv2.INTER_CUBIC
            img_out = cv2.resize(img_out, (dim, dim), interpolation = interpolation_type)
        
        # Adaptive Histogram equalization
        #img_out = adap_equalize(img_out)
    
        #img_out = cv2.resize(img_out, (dim, dim), interpolation = cv2.INTER_CUBIC) #for cxrs : INTER_CUBIC 
        cv2.imwrite(out_dirr  + name ,img_out)
        x[n] = img_out  #Fill empty x with values from img
        return x


# Setting the model names
modelpath1 = os.getcwd() +'/256_b32_X3726_09-08__07-40-52/'
modelname1 = "DoubleCLAHE_3Xaug_unet_lungs_segmtn"

modelpath2 = os.getcwd() + "/256_b32_X3726_09-08__07-41-52/"
modelname2 = "CLAHE-Gausblur3_3Xaug_unet_lungs_segmtn"



# load model
model = load_seg_model(modelpath1,modelname1)

# input dimension
dim = 256


# All raw cropped non-resized images
in_path = '/data/CXR/Classificatn_phase2/NoSplit_allclass/Raw_cropped_2CLAHE/' ; cropped = True

out_path0 = '/data/CXR/Classificatn_phase2/Training_split_2c/' +'Raw' + str(dim) +'_3Xaug'
out_path = '/data/CXR/Classificatn_phase2/Training_split_2c/' +'Clahe' + str(dim) +'_3Xaug'


classes = get_folders_only(in_path)
# Or directly give the list of classes to be trianed for
#classes = ['COVID', 'Normal']


for n, class_ in tqdm(enumerate(classes), total=len(classes)):

    # CHOOSE AS PER THE SPLIT TYPE YOU WANT
    class_train_out, class_val_out, class_test_out = make_split_dir_classwise_train_val_test(out_path, class_)
    #class_train_out, class_test_out = make_split_dir_classwise_train_test(out_path, class_)

    class_in_path = in_path + str(class_) + '/'
    
    '''
    # TO RANDOMLY ALLOCATE TRAIN-VAL-TEST IDS
    ids = getImagesAndLabels(class_in_path)
    random.shuffle(ids)
    
    train_ids, test_val_ids  = train_test_split(ids, test_size=0.4, shuffle=True)
    val_ids, test_ids  = train_test_split(test_val_ids, test_size=0.5, shuffle=True)
    '''
    
    # TO ALLOCATE SAME TRAIN-VAL-TEST IDS as contained in "out_path0"
    class_train_in, class_val_in, class_test_in = make_split_dir_classwise_train_val_test(out_path0, class_)
    train_ids = getImagesAndLabels(class_train_in)
    val_ids = getImagesAndLabels(class_val_in)
    test_ids =getImagesAndLabels(class_test_in)

  
    if cropped == False:
        prep_cxr_classification(class_in_path, class_train_out, train_ids, model, dim)
        prep_cxr_classification(class_in_path, class_val_out, val_ids, model, dim)
        prep_cxr_classification(class_in_path, class_test_out, test_ids, model, dim)
     
    else:
        adap_resize(class_in_path, class_train_out, train_ids, dim)
        adap_resize(class_in_path, class_val_out, val_ids, dim)
        adap_resize(class_in_path, class_test_out, test_ids, dim)
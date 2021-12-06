# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 01:50:07 2021

@author: AA086655
"""


'''
Performs pre-processing of a given Dataset for a cross-validation training, 
while splitting the data it into Train-Val-Test for K fold in which every fold has
Folders before it is leveraged for the model trainng
The pre-processing involves:
    (i) Load the disoriented CXRs from various sources in the given folder
    (ii) Crop the CXR around the Thoracic/Lung region(RoI) using the self trained UNet model
    (iii) Peforms appropriate medical image enhancement using image processing techniques
            - CLAHE or Histogram equalization
            - filtering/Blurring/denoising
    (iv) Resize based on appropraite interpolation
    (v) save it to the a desired directory in the following fashion:
            #Fold 1:
                #Train set-
                    - Class1
                    - Class2
                #Val set-
                    - Class1
                    - Class2
                #Test set-
                    - Class1
                    - Class2
            #Fold 2:
                ...
            #Fold 3:
                ...
            #Fold 4:
                ...
            #Fold 5:
                ...
    
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
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import random
import pandas as pd
from tqdm import tqdm 
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import model_from_yaml
from sklearn.model_selection import KFold, StratifiedKFold

from Methods_img_proceess import morph, adap_equalize
from methods_model_training import makemydir, load_seg_model, get_folders_only, getImagesAndLabels, grabLungsbox, make_split_dir_classwise_train_val_test_crossVal





def test_img(img_path):
    '''
    Test the image preprocessing involving cropping of the Lung RoI

    '''
    image = cv2.imread(img_path , 0)

    size = image.shape
    img = image
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    img = adap_equalize(img)
    img = np.expand_dims(np.array(img), axis = 0)
    preds = model.predict(img, verbose=1)
    preds_t = (preds > 0.5).astype(np.uint8)
    preds_t = np.squeeze(preds_t)
    #plt.imshow(preds_t)
    #plt.show()
    
    mask = resize(preds_t, size , mode='constant',  preserve_range=True)
    mask = (mask > 0).astype(np.uint8)
    plt.imshow(mask)
    plt.show()
    img_out, ratio = grabLungsbox(image, mask)




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
        if img_out.shape != (dim, dim):
            interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>dim else  cv2.INTER_CUBIC
            img_out = cv2.resize(img_out, (dim, dim), interpolation = interpolation_type)
        
        img_out = adap_equalize(img_out)
    
        #img_out = cv2.resize(img_out, (dim, dim), interpolation = cv2.INTER_CUBIC) #for cxrs : INTER_CUBIC 
        cv2.imwrite(out_dirr  + name ,img_out)
       
        
def prep_cxr_classification(in_path, out_dirr, ids, seg_model,dim):
    """
    This is the all in one pre-processing fucntion for the calssification training
    Optional features: maskout = True if you only want predicted lungs region to be visible
                     : adaptive histogram equalization
                     

    Parameters
    ----------
    in_path : Dataset folder with all the Source images together in folders classwose
    out_dirr : Directory for the split datset
    ids : id names list of all the original images in in_path directory 
    seg_model : Lungs segmentation model
    dim : Output dimensions of the preprocessed imahes

    Returns
    -------
    An array of all the preprocessed images

    """
    x = np.zeros((len(ids), dim, dim), dtype=np.uint8)
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        name = id_ + '.png'
        print(name)
        # if the images need to be pre-processed or resized!
        #img = prep_cxr_segmtn(path + name, img_width)
        image = cv2.imread(in_path + name, 0)#[:,IMG_CHANNELS]
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
        img_out1 = adap_equalize(img_out1)
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(img_out, cmap='gray')
        
        plt.subplot(232)
        plt.title('After pre-processing')
        plt.imshow(img_out1, cmap='gray')
        '''
        if img_out.shape != (dim, dim):
            interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>dim else  cv2.INTER_CUBIC
            img_out = cv2.resize(img_out, (dim, dim), interpolation = interpolation_type)
        
        img_out = adap_equalize(img_out)
    
        #img_out = cv2.resize(img_out, (dim, dim), interpolation = cv2.INTER_CUBIC) #for cxrs : INTER_CUBIC 
        cv2.imwrite(out_dirr  + name ,img_out)
        x[n] = img_out  #Fill empty x with values from img
        return x



modelpath = os.getcwd() +'/b32-e150_X5703_06-19__21-53-43/'
modelname = "256_b32-e150_X5703_Raw_blur_prep_model_lungs_segmtn_06-19__21-53-43"

#modelpath = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Selected results/b8_X6079_06-17__08-54-16/"
#modelname = "256_b8_X6079_Raw_blur_prep_model_lungs_segmtn_06-17__08-54-16"

# load model
model = load_seg_model(modelpath,modelname)
print("Loaded model from disk")


dim = 256

in_path = '/data/CXR/Orignals/COVID-19 Radiography Database/COVID-19_Radiography_Dataset 3616/'

# All raw cropped non-resized images
in_path = '/data/CXR/NoSplit_4c/Raw_cropped/'
out_path = '/data/CXR/Crossval Training_split_2c/'+ str(dim) +'_80-20_train_test/'


classes = get_folders_only(in_path)
# Or directly give the list of classes to be trianed for
#classes = ['COVID', 'Normal']

# When shuffle is True, random_state affects the ordering of the indices.
# Pass an int to random_state for reproducible output across multiple function calls.
skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True) 

for n, class_ in tqdm(enumerate(classes), total=len(classes)):

    class_in_path = in_path + str(class_) + '/'
    ids = getImagesAndLabels(class_in_path)
    random.shuffle(ids)

    ids_data = {'ids': ids}
    ids_df =  pd.DataFrame(ids_data)
    
    # if val split type is not chosen then "test_val_ids" will be used as the final test ids and "val_ids, test_ids" wont be used(see Try-expect loop)
    train_val_ids, test_ids  = train_test_split(ids, test_size=0.2, shuffle=True)
    #val_ids, test_ids  = train_test_split(test_val_ids, test_size=0.5, shuffle=True)


    fold_var = 1
    for train_index, val_index in skf.split(train_val_ids,np.zeros(len(train_val_ids))):
        training_data_ids = ids_df.iloc[train_index]['ids']
        validation_data_ids = ids_df.iloc[val_index]['ids']
        
        if fold_var ==1:
            train_ids_data = {'train_ids' + str(fold_var): training_data_ids}
            train_ids_df =  pd.DataFrame(training_data_ids)
            
            val_ids_data = {'val_ids' + str(fold_var): validation_data_ids}
            val_ids_df =  pd.DataFrame(val_ids_data)
            
        else:
            train_ids_df['train_ids' + str(fold_var)] = training_data_ids
       
            val_ids_df['val_ids' + str(fold_var)] = validation_data_ids

        print(validation_data_ids)
        
        class_train_out, class_val_out, class_test_out = make_split_dir_classwise_train_val_test_crossVal(out_path, class_, fold_var)

    
        #prep_cxr_classification(class_in_path, class_train_out, training_data_ids, model, dim)
        adap_resize(class_in_path, class_train_out, training_data_ids, dim)

        #prep_cxr_classification(class_in_path, class_val_out, validation_data_ids, model, dim)
        adap_resize(class_in_path, class_val_out, validation_data_ids, dim)

       	# LOAD BEST MODEL to evaluate the performance of the model
        #model.load_weights(save_dir + str(fold_var)+  modelname +".h5")
    
       	
        fold_var += 1

    #prep_cxr_classification(class_in_path, class_test_out, test_ids, model, dim)
    adap_resize(class_in_path, class_test_out, test_ids, dim)



    
#img_path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19_Radiography_Dataset 3616/COVID/COVID-2521.png"
#test_img(img_path)


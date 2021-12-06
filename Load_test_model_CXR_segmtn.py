"""
Created on Wed May 12 13:17:28 2021

@author: abhia
"""

'''
Load and test the segmentaion model on a given test image
    > Test and comparison among various models trained differently for the same segmentation
    > Helps understand which model is out performing others
We have trained a Unet model for the segmentation task

Step:
    (i) load the models
    (ii) Load the image and makes the segmentaion predictions from the loaded model
    (iii) Peforms RoI extraction around the lungs masks
    (iv) Returns a cropped images around RoI


# =============================================================================
# # Things to change-
# # 1. in_path : dirrectory of the train/val data to be added
# # 2. result_dir : No need to change. based on os.getcwd
# # 3. modelname : path of the model with final name based on prep-process features
# # 4. prep_cxr_segmtn(in_img_path, dim, mask) : in Methods_img_process.py
# # 5. batchsize
# =============================================================================
   
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_yaml

from Methods_img_process import prep_cxr_segmtn, morph, adap_equalize
from methods_model_training import load_seg_model, makemydir, grabLungsbox, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, plot_test_maskout3, plot_maskout3, plot_maskonly2
from methods_model_training import load_img_fromIDs_no_prep, load_img_fromIDs_1chale_gausblur3, load_img_fromIDs_1claha, load_img_fromIDs_2chale, load_img_fromIDs_1he_1claha, load_img_fromIDs_3clahe, plot_maskonly_compare4

import errno
from datetime import datetime



mydir = os.getcwd()+ "/LoadResults/"+datetime.now().strftime('%m-%d__%H-%M-%S') + "/"
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..


#initializing a random seed
seed = 38
np.random.seed = seed

# Assigninig Image width, hight and chanel(1 for Grayscale)
img_width = 256
img_hieght = 256
img_channels = 1

cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'


# Test set
test_path =  "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/cxr_lung_seg_covid_detection_/Training split 2c/test/Normal/"
test_ids = getImagesAndLabels(test_path)# + cxrs)



# 1xCLAHE 3Xaug
modelpath1 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/SEG_Selected results/SegResults_208covid_added/256_b32_X3726_09-09__03-26-53/'
modelname1 = "1ClAHE_3Xaug_unet_lungs_segmtn"

# 2xCLAHE 3Xaug
modelpath2 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/SEG_Selected results/SegResults_208covid_added/256_b32_X3726_09-08__07-40-52/'
modelname2 = "DoubleCLAHE_3Xaug_unet_lungs_segmtn"

# 3xCLAHE 3Xaug
modelpath3 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/SEG_Selected results/SegResults_208covid_added/256_b32_X3726_09-08__07-41-16/'
modelname3 = "TripleCLAHE_3Xaug_unet_lungs_segmtn"

# 1xAdap-1xCLAHE 3Xaug
modelpath4 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/SEG_Selected results/SegResults_208covid_added/256_b32_X3726_09-08__07-41-52/'
modelname4 = "CLAHE-Gausblur3_3Xaug_unet_lungs_segmtn"

#modelpath5 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/SEG_Selected results/SegResults_208covid_added/256_b32_X3060_08-30__17-10-45'
#modelname5 = 'Adap_Gausblur_3Xaug_unet_lungs_segmtn'


x_test_raw = load_img_fromIDs_no_prep(test_path, test_ids, 256)

x_test1 = load_img_fromIDs_1claha(test_path, test_ids, 256)
x_test2 = load_img_fromIDs_2chale(test_path, test_ids, 256)
x_test3 = load_img_fromIDs_3clahe(test_path, test_ids, 256)
#x_test4 = load_img_fromIDs_1he_1claha(test_path, test_ids, 256)
#x_test4 = load_img_fromIDs_1chale_gausblur3(test_path, test_ids, 256)
x_test4 = x_test1


# load model1
model1 = load_seg_model(modelpath1,modelname1)
print("Loaded model 1 from disk")

# load model2
model2 = load_seg_model(modelpath2,modelname2)
print("Loaded model 2 from disk")

# load model3
model3 = load_seg_model(modelpath3,modelname3)
print("Loaded model 3 from disk")

# load model4
model4 = load_seg_model(modelpath4,modelname4)
print("Loaded model 4 from disk")

#model.summary()

def preds_mask(model, x_test, p):
    '''
    Predicts segmentation masks for a given model and array of the image array

    Parameters
    ----------
    model : trained Segmentation model
    x_test : pre-processed array of the input image tobe tested
    p : Threshold for making the segmentaion based on model predictions

    Returns
    -------
    preds_test_t : segmentation mask prediciton for the input image

    '''
    preds_test = model.predict(x_test, verbose=1)
    preds_test_t = (preds_test > p).astype(np.uint8)
    preds_test_t = np.squeeze(preds_test_t)
    return preds_test_t

# Prediction for a single image after required pre-processing of its data strcuture
#preds_test = model.predict(np.expand_dims(np.array(x_test[1]), axis = 0), verbose=1)


def masks_comparison(p):
    '''
    This function collates all the segmentation predictions for all the model variations

    Parameters
    ----------
    p : Threshold for making the segmentaion based on model predictions

    Returns
    -------
    preds_ts_tuple : a tuple of lung segmentaion mask for all the model prections

    '''
    preds_test_t1 = preds_mask(model1, x_test1, p)
    preds_test_t2 = preds_mask(model2, x_test2, p)
    preds_test_t3 = preds_mask(model3, x_test3, p)
    preds_test_t4 = preds_mask(model4, x_test4, p)
    #preds_test_t5 = preds_mask(model5, x_test5, p)

    preds_ts_tuple = (preds_test_t1, preds_test_t2, preds_test_t3,preds_test_t4)
    
    #plot_maskonly_compare4(preds_ts_tuple, x_test_raw,test_ids[ix], ix,False)
    return preds_ts_tuple

# Set a threshold p for mask predicitons
p=0.35
preds_ts_tuple = masks_comparison(p)

out_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Seg_mask_results/'

for i in range(len(x_test_raw)):
    save_path = out_path+'/' +str(p)+'/Comparisons'
    makemydir(save_path)
    plot_maskonly_compare4(preds_ts_tuple, x_test_raw,test_ids[i], i, save_path)
    plt.show()
    plt.cla()
    plt.clf()

# Perform a check on some random test samples
ix = 2 #random.randint(0, len(preds_test_t))
#plot_maskonly2(preds_test_t1[ix], x_test_raw[3])

for j in range(len(preds_ts_tuple)):
    print(test_ids[i])
    for i in range(len(x_test_raw)):
        save_path = out_path+'/' +str(p)+'/Covid_masks/model'+ str(j+1)+'/'

        crop, r = grabLungsbox(x_test_raw[i], preds_ts_tuple[j][i],test_ids[i], False)
        mask = preds_ts_tuple[j][i]*255
        makemydir(save_path)
        cv2.imwrite(save_path+ test_ids[i] + ".png",mask)
        #cv2.imwrite(save_path+ test_ids[i] + ".png",crop)
        
        plt.show()
        plt.cla()
        plt.clf()
    print(":::::::::::::")
    
i=3
crop, r = grabLungsbox(x_test_raw[i], preds_ts_tuple[1][i],test_ids[i], True)
crop, r = grabLungsbox(x_test1[i], preds_ts_tuple[1][i],test_ids[i], True)


# Snippet for Raw vs CLAHE comparison
i=3
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title('Raw ' + str(test_ids[i]))
plt.imshow(x_test_raw[i], cmap='gray')

plt.subplot(122)
plt.title('After CLAHE')
plt.imshow(x_test1[i], cmap='gray')

#plot_maskout3(preds_test_t[ix], x_test[ix], mydir, test_ids[ix] )
#grabLungsbox( x_test[ix], preds_test_t[ix])
#lot_test_maskout3(preds_test_t[ix],x_test[ix], y_test[ix], mydir, test_ids[ix] )

'''
# To find array id for any specific image based on its real name
for i in range(3600):
    if test_ids[i] == "COVID-2479":
        print(i)
    else:
        pass
'''

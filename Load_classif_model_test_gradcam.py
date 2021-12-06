# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:12:56 2021

@author: AA086655
"""
'''
Load and performs comparative gradcam study among various pretrained models 
on a given test images. 

The result will be a combined plot of all the gradcams overlayed over the test images

We have deployed the following models:
    a. vgg16_model, 
    b. Xception_model
    c. resnet_model, 
    d. InceptionV3_model, 
    e. densenet201_model.
    
Step:
    (i) load all the models
    (ii) Select an image to perform Gradcam upon
    (iii) Select layers for each model whose Gradcam to be extracted and ploted - very cruicial step
    (iv) Select Gradcam Type Simple(choose gradcam_simple) or with ac color template(choose GradCam)
    (v) Load the test path and image ID to be tested
    
PREDICTION LABELS[0,1]:
    > [0,1] - is for COVID NEGATIVE
    > [1,0] - is for COVID POSITIVE

# =============================================================================
# # Things to change-
# # 1. in_path : dirrectory of the train/val data
# # 2. result_dir :Rename the Top subfolder, rest is Based on os. getcwd
# # 3. modelname : only the unique name(based on feature and class) of the model to be added
# # 4. model layers whose gradcam to be plotted

# =============================================================================
'''

import cv2
import os

import matplotlib
from sys import platform
if platform == "linux":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from methods_model_training import makemydir, Load_model, append_multiple_lines, plot_class_confusion_matrix, eval, makemydir, getImagesAndLabels, confusion_mat_seg
from Gradcam_module import gradcam_simple, get_activation_maps, GradCam

# SETTING VARABLES 
# Assigninig Image width, hight and chanel(1 for Grayscale)
dim = 256

batch_size = 64
epochs= 20

img_width = dim
img_height = dim
IMG_CHANNELS = 1

# INPUT layer size, re-size all the images to this
IMAGE_SIZE = [img_width, img_height]

result_dir = os.getcwd()
#makemydir(result_dir)

# VGG16
modelpath1 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Phase2 _class2_selectred_Results/Results_raw_prep/VGG16_dl256mode1_22class_model256RAW_aug__ts09-12__19-36-48/'
modelname1 = "VGG16_dl256mode1_22class_model256RAW_aug"

# RESNET50
modelpath2 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Phase2 _class2_selectred_Results/Results_raw_prep/Resnet50_dl256bn1_2class_model256RAW_aug__ts09-13__10-56-41/'
modelname2 = "Resnet50_dl256bn1_2class_model256RAW_aug"

# DENSENET
modelpath3 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Phase2 _class2_selectred_Results/Results_raw_prep/Densenet_dl256_2class_model256RAW_aug__ts09-12__07-57-51/'
modelname3 = "Densenet_dl256_2class_model256RAW_aug"


# INCEPTIONV3
modelpath4 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Phase2 _class2_selectred_Results/Results_raw_prep/InceptionV3_dl256bn1_2class_model256RAW_aug__ts09-12__19-30-39/'
modelname4 = "InceptionV3_dl256bn1_2class_model256RAW_aug"

# XCEPTION
modelpath5 = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Phase2 _class2_selectred_Results/Results_raw_prep/Xception_dl256bn1_2class_model256RAW_aug__ts09-22__04-04-36/'
modelname5 = 'Xception_dl256bn1_2class_model256RAW_aug'
#########################################################


# load model1
model1 = Load_model(modelpath1,modelname1)
print("Loaded model 1 from disk")

# load model2
model2 = Load_model(modelpath2,modelname2)
print("Loaded model 2 from disk")

# load model3
model3 = Load_model(modelpath3,modelname3)
print("Loaded model 3 from disk")

# load model4
model4 = Load_model(modelpath4,modelname4)
print("Loaded model 4 from disk")

# load model4
model5 = Load_model(modelpath5,modelname5)
print("Loaded model 5 from disk")

# summarize model.
# view the final structure of the model
model1.summary()
model2.summary()
model3.summary()
model4.summary()
model5.summary()


# view the trainable layers of the model
a =[]
for i, layers in enumerate(model3.layers):
    a.append([i,layers.name, "-", layers.trainable])
    print(i,layers.name, "-", layers.trainable)
    

# TEST PATH
# Plot for a all the images at test_path
test_path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19_Radiography_Dataset 3616"+ '/COVID/'
path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Figures/gradcams/Comparative gradcams all model/"
test_covid_ids = getImagesAndLabels(test_path)


def run_gradcams_green(test_id):
    '''
    Simple Gradcam without any color scheme
    > Extracts the Gradcam of the given models for a specific layer(selected out priorly after proper analysis)
    > Puts all the gradcams side by side ina single horizontal plot.

    Parameters
    ----------
    test_id : id of the cxr whos gradcams are to plotted

    Returns
    -------
    None.

    '''
    # For VGG16
    gradcam1,layer1, pred_index1 = gradcam_simple(model1, test_path + test_id +'.png', "block5_conv3", result_dir,None)
    
    # For Resnet
    #preds = GradCam(test_path + '/COVID/' + test_covid_ids[i] +'.png', model2,"conv5_block3_add", result_dir, None)
    gradcam2,layer2, pred_index2 = gradcam_simple(model2, test_path + test_id +'.png', "conv5_block2_2_bn", result_dir,None)
    
    # For DENSENET
    #preds = GradCam(test_path + '/COVID/' + test_covid_ids[i] +'.png', model3,"conv5_block32_concat", result_dir, None)
    gradcam3,layer3, pred_index3 = gradcam_simple(model3, test_path + test_id +'.png', "conv5_block28_2_conv", result_dir,None)
    
    # For InceptionV3
    #preds = GradCam(test_path + '/COVID/' + test_covid_ids[i] +'.png', model4,"activation_59", result_dir, None)
    gradcam4,layer4, pred_index4 = gradcam_simple(model4, test_path  + test_id +'.png', "batch_normalization_42", result_dir,None)
    
    # For Xception
    #preds = GradCam(test_path + '/COVID/' + test_covid_ids[i] +'.png', model5,"block14_sepconv1", result_dir, None)
    gradcam5,layer5, pred_index5 = gradcam_simple(model5, test_path + test_id +'.png', "block14_sepconv2_bn", result_dir,None)


    #plt.cla()
    #plt.clf() 
    plt.figure(figsize=(32, 8))
    plt.subplot(161)
    plt.title(str(test_id))
    img = cv2.imread( test_path + test_id +'.png',0)
    plt.imshow(img,cmap="gray")
    
        
    plt.subplot(162)
    plt.title('VG16-'+layer1+'='+ str(pred_index1))
    plt.imshow(gradcam1)
    
    plt.subplot(163)
    plt.title('RES50-'+layer2+'='+ str(pred_index2))
    plt.imshow(gradcam2)

    plt.subplot(164)
    plt.title('DENS-'+layer3+'='+ str(pred_index3))
    plt.imshow(gradcam3)

    plt.subplot(165)
    plt.title('INCE-'+layer4+'='+ str(pred_index4))
    plt.imshow(gradcam4)
    
    plt.subplot(166)
    plt.title('XCEP-'+layer5+'='+ str(pred_index5))
    plt.imshow(gradcam5)
    
    out_path= path + '/Gradcam_depth_analysis2/greens/'
    makemydir(out_path)
    plt.savefig(out_path + str(test_id) +'.png', bbox_inches='tight')


def run_gradcams_rainbow(test_id):
    '''
    Gradcam with rainbow color scheme - colour scheme can be changed in the the Gradcam module
    > Extracts the Gradcam of the given models for a specific layer(selected out priorly after proper analysis)
    > Puts all the gradcams side by side ina single horizontal plot.

    Parameters
    ----------
    test_id : id of the cxr whos gradcams are to plotted

    Returns
    -------
    None.

    '''
    # For VGG16
    gradcam1,layer1, pred_index1  = GradCam(test_path + test_id +'.png', model1,"block5_conv3", result_dir, None)
    #gradcam_simple(model1, test_path + '/COVID/' + test_covid_ids[i] +'.png', "block5_conv2", result_dir,None)
    
    # For Resnet50
    gradcam2,layer2, pred_index2 = GradCam(test_path + test_id +'.png', model2,"conv5_block2_2_bn", result_dir, None)
    #gradcam_simple(model2, test_path + '/COVID/' + test_covid_ids[i] +'.png', "conv5_block3_add", result_dir,None)
    
    # For DENSENET
    gradcam3,layer3, pred_index3 = GradCam(test_path  + test_id +'.png', model3,"conv5_block28_2_conv", result_dir, None)
    #gradcam_simple(model3, test_path + '/COVID/' + test_covid_ids[i] +'.png', "conv5_block32_concat", result_dir,None)
    
    # For InceptionV3
    gradcam4,layer4,pred_index4 = GradCam(test_path  + test_id +'.png', model4,"batch_normalization_42", result_dir, None)
    #gradcam_simple(model4, test_path + '/COVID/' + test_covid_ids[i] +'.png', "activation_59", result_dir,None)
    
    # For Xception
    gradcam5,layer5, pred_index5 = GradCam(test_path  + test_id +'.png', model5,"block14_sepconv2_bn", result_dir, None)
    #gradcam_simple(model5, test_path + '/COVID/' + test_covid_ids[i] +'.png', "block14_sepconv1", result_dir,None)


    #plt.cla()
    #plt.clf() 
    plt.figure(figsize=(32, 8))
    plt.subplot(161)
    plt.title(str(test_id))
    in_image = cv2.imread( test_path + test_id +'.png',0)
    img_size = [256,256]
    if in_image.shape != img_size:
        interpolation_type =  cv2.INTER_AREA if in_image.shape[1]>img_size[1] else  cv2.INTER_CUBIC
        img = cv2.resize(in_image, img_size, interpolation = interpolation_type)
    print("Image shape:", in_image.shape, "resized to :", img.shape)
    
    plt.imshow(img,cmap="gray")
    
        
    plt.subplot(162)
    plt.title('VG16-'+layer1+'='+ str(pred_index1))
    plt.imshow(gradcam1)
    
    plt.subplot(163)
    plt.title('RES50-'+layer2+'='+ str(pred_index2))
    plt.imshow(gradcam2)

    plt.subplot(164)
    plt.title('DENS-'+layer3+'='+ str(pred_index3))
    plt.imshow(gradcam3)

    plt.subplot(165)
    plt.title('INCE-'+layer4+'='+ str(pred_index4))
    plt.imshow(gradcam4)
    
    plt.subplot(166)
    plt.title('XCEP-'+layer5+'='+ str(pred_index5))
    plt.imshow(gradcam5)
    
    out_path= path + '/Gradcam_depth_analysis2_no_prep/rainbows/'
    makemydir(out_path)
    plt.savefig(out_path + str(test_id) +'.png', bbox_inches='tight')

##############################################################
# plot the gradcams
##############################################################


# Plot comparative gradcam of all the given models for a single image
run_gradcams_rainbow("COVID-100")
run_gradcams_rainbow("COVID-101")
run_gradcams_rainbow("COVID-220")

# Plot only a single simple gradcam for a given layer for a single model using gradcam_simple function
layer5= "block14_sepconv2_bn"
gradcam,layer5, pred_index5 = gradcam_simple(model5, test_path + test_covid_ids[110] +'.png', layer5, result_dir,None)
plt.imshow(gradcam); print(layer5)

# Plot only a single Rainbow template gradcam for a given layer for a single model using gradcam_simple function
gradcam,layer, pred_index = GradCam(test_path + "COVID-1425" +'.png', model5,layer5, result_dir, None)
plt.imshow(gradcam); print(layer5)

# Makes comparative Gradcam of various models and saves them at a given path for all the test ids extracted from a directory
"""
for id_ in reversed(test_covid_ids):
    #run_gradcams_green(ids)
    run_gradcams_rainbow(id_)
"""



"""
######################################################################
# Expreiment 1: Using get_activation_maps we can plot the whole activation maps
# for a given layer of a model
######################################################################

i = 1644
img_path = test_path + test_covid_ids[i]+'.png'
get_activation_maps(model3,img_path, "conv5_block32_2_conv")

######################################################################
"""


"""
######################################################################
# Expreiment 1: Here we plot Gradcams for all the possible plotable layer of the model
# to help us indentify which layer activations are extracting the feature of out interest
######################################################################


model = model3
for layer in model.layers:
    print(layer.name)
    layer_name = layer.name
    try:
        #plt.clf()
        gradcam,layer_name, pred_index = GradCam(test_path + '/COVID/' +  test_covid_ids[100] +'.png', model,layer_name, result_dir, None)
        # call plt.figure to instantiate a new figure
        plt.figure()
        plt.title('Dense-'+layer_name+'='+ str(pred_index))
        plt.imshow(gradcam)
        print("GRADCAM sucess! - ", layer.name)
    except Exception as e:
        print(e)
        print("NO GRADCAM for: ", layer.name)
        
######################################################################
"""

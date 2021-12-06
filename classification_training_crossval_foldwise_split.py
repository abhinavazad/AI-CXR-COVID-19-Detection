# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:42:17 2021

@author: AA086655
"""

'''
Cross-validation Classification training script for Covid vs Normal CXR detection.
Using Sklearn's Stratified K fold method for trainig
We have deployed the following models among which, one maybe selected for a particular training:
    a. vgg16_model, 
    b. vgg19_model, 
    c. resnet_model, 
    d. InceptionV3_model, 
    e. densenet201_model, 
    f. mobilenet_model, 
    g. Xception_model
    

# =============================================================================
Steps:-
# 1. read image ids from the dataset path(Rrain, test, Val)
# 2. Resize to 256 for getting mask predictions
# 3. Load segmentation model and get masks
# 4. Resize the mask to the original CXR img dimens : WHATSOEVER
# 5. Cropping RoI in the input CXR img about the lungs as per cnt detection in maks.
# 6. Resize the cropped CXR img to the input image dimens : 512
# 7. Data Aug
# 8. Denoising - optional preprocessing integrated in Data Aug
# 9. Feed in to the model and train
    - Early stopping
# 10. Confusion matrix and evaluation on the various metrices
# =============================================================================
# =============================================================================
# # Tweakable parameters:-
# # 1. in_path : dirrectory of the train/val data
# # 2. result_dir :Rename the Top subfolder, rest is Based on os. getcwd
# # 3. batchsize
# # 4. Epoch
# # 5. modelname : only the unique name(based on feature and class) of the model to be added
# # 6. dim : input dimensions of the images for the training.
# # 7a. For OTG Aug: Use the "train_datagen" datagenerator with augmentation features on
# # 7b. For No OTG Aug: Use the "test_datagen" datagenerator with all augmentation features off and only a pre-processing feature(rescale=1./255)
# =============================================================================
  
    
Data folder structure Requirement:
Fold 1:
    Train set-
        - Class1
        - Class2
    Val set-
        - Class1
        - Class2
    Test set-
        - Class1
        - Class2
FOld 2:
    ...
Fold 3:
    ...
Fold 4:
    ...
Fold 5:
    ...
    

'''

import os
import pandas as pd
from glob import glob
from datetime import datetime
from contextlib import redirect_stdout
import pandas
import time
import collections
import tqdm

import matplotlib
from sys import platform
if platform == "linux":
    matplotlib.use('Agg')
    
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import layers
from sklearn.model_selection import KFold, StratifiedKFold

from methods_model_training import eval, getImagesAndLabels,makemydir,plot_model_hist,append_multiple_lines
from Transfer_learning_models import vgg16_model, vgg19_model, resnet_model, InceptionV3_model, densenet201_model, mobilenet_model, Xception_model


# Assigninig Image width, hight and chanel(1 for Grayscale)
dim = 256 #256

batch_size = 64 #64
epochs= 20 #50

img_width = dim
img_height = dim
IMG_CHANNELS = 1

# INPUT layer size, re-size all the images to this
IMAGE_SIZE = [img_width, img_height]


path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 2c'
path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Crossval Training split 2c/"

#path = '/data/CXR/Crossval Training_split_3c/256adap_aug_80-20_train_test/'
path = '/data/CXR/Crossval Training_split_2c/256adap_aug_80-20_train_test/'


#folder_name = os.path.basename(path)

train_val_folds_path = path + '/train_val_fold_splits/'

test_path = path + '/test/'



# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function= None) #Put : adap_equalize for equalizing the inputs
test_datagen = ImageDataGenerator(rescale=1./255,
                                  preprocessing_function= None) #Put : adap_equalize for equalizing the inputs


# With on-the-go Augmentaion for training set
#train_set = train_datagen.flow_from_directory(train_path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', interpolation = "bicubic")

# No on-the-go Augmentaion
#train_set = test_datagen.flow_from_directory(train_path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', interpolation = "bicubic")
#val_set = test_datagen.flow_from_directory(val_path, shuffle=True,target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical',interpolation = "bicubic")
test_set = test_datagen.flow_from_directory(test_path, shuffle=False, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', interpolation = "bicubic")


########################################################
# SETTING VARABLES
########################################################
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
TEST_ACCURACY = []
TEST_LOSS = []
callbacks = []


train_len =  'Crossval' #+ str(train_data_generator.n) #str(steps_per_epoch*batch_size) #str(len(x_train))
suffix = str(dim) +"_b"+ str(batch_size) +"_e"+ str(epochs) + "X"+ train_len + '_ts' + datetime.now().strftime('%m-%d__%H-%M-%S') 

result_dir = str(os.getcwd()+ "/Res50_crossval_results_c2_60-40/"+ suffix + '/')


# save_dir just for a common place to save just the weights of all the models
save_dir = str(result_dir + '/saved_model_weights/')
makemydir(save_dir)
fold_var = 1


kfolds = glob(train_val_folds_path + '/*')


print('Lets start training.')
start0 = time.time()

#########################################################
# Cross validation Kfold MODEL TRAINING 
#########################################################

kf = KFold(n_splits = 5)
skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True) 


for n, fold_ in tqdm(enumerate(kfolds), total=len(kfolds)):
    print(':::: KFold ', n+1, ' path: ', fold_)

   
for n, fold_ in tqdm(enumerate(kfolds), total=len(kfolds)):
    fold_ = train_val_folds_path + os.path.basename(fold_)
    train_path =  fold_ + '/train/'
    val_path =  fold_ + '/val/'
    
    # Shuffle=True during training
    train_data_generator = test_datagen.flow_from_directory(train_path, shuffle=True, 
                               target_size=IMAGE_SIZE, batch_size= batch_size, 
                               class_mode='categorical', interpolation = "bicubic")
    valid_data_generator = test_datagen.flow_from_directory(val_path, shuffle=True, 
                               target_size=IMAGE_SIZE, batch_size= 1, 
                               class_mode='categorical',interpolation = "bicubic")


    
    res_dir_this_fold = result_dir + "/fold" + str(n+1) +'/'
    modelname =  "kfold_" + str(fold_var)
    modelpath = res_dir_this_fold +'/' + modelname
    
    makemydir(res_dir_this_fold + '/saved_model')
    #keras_model_dir = result_dir + '/model_'  + str(img_width) +"_b"+ str(batch_size) + "_X"+ train_len+ '/'


    steps_per_epoch = train_data_generator.samples//train_data_generator.batch_size #(len(x_train))//batch_size
    validation_steps = valid_data_generator.samples//valid_data_generator.batch_size #(len(x_val))//batch_size

    # Refreshing the model variable to create a new model for every fold
    print("::::==Refreshing the model to new-untrained==::::")
    print(':::: KFold ', n+1, ' path: ', fold_)

    # for getting number of output classes
    class_folders = glob(train_path + '/*')
    
    #Set dense layer dimension and modes
    # initially all the model vars are set to False.
    dense=256;mode=1; inceptV3=False; vgg16=False;  vgg19=False; xception=False;  resnet=False; densenet=False;mobilenet=False; 
    
    #Select a model by assigning "True" to it
    inceptV3=True #vgg16=True;  vgg19=True; xception=True;  resnet=True; densenet=True;mobilenet=True; 
    # if any additional information to be added in the model name prefix
    add_info = ""
    
    ########################################################
    # SETTING VARABLES and loaing the selected model
    if vgg16==True:
        model = vgg16_model(IMAGE_SIZE, class_folders,mode,dense)
        modelname = modelname + "VGG16_dl"+ add_info +str(dense)+"mode_" + str(mode) + "_"+ str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
    elif vgg19==True:
        model = vgg19_model(IMAGE_SIZE, class_folders,mode,dense)
        modelname = modelname + "VGG19_dl"+ add_info +str(dense)+"mode_" + str(mode) + "_"+ str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
    elif resnet==True:
        model = resnet_model(IMAGE_SIZE, class_folders,mode,dense)
        modelname = modelname + "Resnet50_dl"+ add_info +str(dense) + "bn" + str(mode) +"_"  + str(len(class_folders))+"class_model" + str(dim)+ "clahe_aug" #"RAW_aug"
    elif inceptV3==True:
        model = InceptionV3_model(IMAGE_SIZE, class_folders,mode,dense)
        modelname = modelname + "InceptionV3_dl"+ add_info +str(dense)+ "bn" + str(mode) +"_"  +str(len(class_folders))+"class_model" + str(dim)+ "clahe_aug" #"RAW_aug"
    elif densenet==True:
        model = densenet201_model(IMAGE_SIZE, class_folders,mode,dense)
        modelname = modelname + "Densenet_dl" + add_info +str(dense) + "bn" + str(mode) + "_"  +str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
    elif mobilenet==True:
        model = mobilenet_model(IMAGE_SIZE, class_folders,mode,dense,0.4)
        modelname = modelname + "Mobilenet_dl"+ add_info +str(dense) + "bn" + str(mode) + "_"  +str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
    elif xception==True:
        model = Xception_model(IMAGE_SIZE, class_folders,mode,dense)
        modelname = modelname + "Xception_dl"+ add_info +str(dense) + "bn" + str(mode) + "_"  +str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"


    # view the final structure of the model
    if n == 0:
        model.summary()
    
    # view the trainable layers of the model
    a =[]
    for i, layers_ in enumerate(model.layers):
        a.append([i,layers.name, "-", layers_.trainable])
        print(i,layers_.name, "-", layers_.trainable)

    
    # tell the model what cost and optimization method to use
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    common_model_weight_paths = (save_dir + '/'+ modelname +".hdf5")
    ##Modelcheckpoint
    checkpoint = ModelCheckpoint( common_model_weight_paths , monitor='val_accuracy', verbose=1, save_best_only=True,  mode='max')
    #This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3,  mode='max', verbose=1, restore_best_weights = True) #min_delta=0.5
    #CSVLogger logs epoch, acc, loss, val_acc, val_loss
    log_csv = CSVLogger(res_dir_this_fold + '/my_logs.csv', separator=',', append=False)
    callbacks_list = [checkpoint, early_stop, log_csv]
 
    print("Cross validation training for Kfold ", n, " starts::::::::::::")
    start = time.time()
   	# FIT THE MODEL
    history = model.fit(train_data_generator, validation_data=valid_data_generator,
                        epochs=epochs, steps_per_epoch= steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list)
    end = time.time()
    print('Training finished!')
    print("CrossVal for Kfold ", n+1, ' :::::::Total Cross validation training Time taken: ', (end-start)/60,  'minutes ========')

    
    # Shuffle=False after training for evaluation funtions to work wll
    train_data_generator = test_datagen.flow_from_directory(train_path, shuffle=False, 
                               target_size=IMAGE_SIZE, batch_size= batch_size, 
                               class_mode='categorical', interpolation = "bicubic")
    valid_data_generator = test_datagen.flow_from_directory(val_path, shuffle=False, 
                               target_size=IMAGE_SIZE, batch_size= 1, 
                               class_mode='categorical',interpolation = "bicubic")


    #model.save(res_dir_this_fold + '/saved_model')
    
    # Saving the model to disk
    # serialize model to YAML, saves the model architecture to YAML

    model_yaml = model.to_yaml()
    with open( modelpath + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(modelpath + ".h5")

    print("Saved model to disk")


    
   	#PLOT HISTORY
    plot_model_hist(history, res_dir_this_fold )
   	
   	# LOAD BEST MODEL to evaluate the performance of the model
    #model.load_weights(common_model_weight_paths)
   	
    val_results = model.evaluate(valid_data_generator, steps = valid_data_generator.samples)
    val_results = dict(zip(model.metrics_names,val_results))
   	
    VALIDATION_ACCURACY.append(val_results['accuracy'])
    VALIDATION_LOSS.append(val_results['loss'])
    
    test_results = model.evaluate(test_set, steps = test_set.samples)
    test_results = dict(zip(model.metrics_names,test_results))
   	
    TEST_ACCURACY.append(test_results['accuracy'])
    TEST_LOSS.append(test_results['loss'])
    
    
    ################################################
    # SAVING LOGS of training
    summary_path = (res_dir_this_fold + '/modelsummary.txt')
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
    with open(summary_path, 'a') as output:
        for row in a:
            output.write(str(row) + '\n')
        
    file = open(summary_path, 'a')       
    line = ['========Time taken for model training: ', str((end-start)/60),  'minutes ========',
            '\nTraining Dataset path: ', str(fold_), '\nTrainng Classes: ',str(test_set.class_indices), 
            '\nTotal train samples: ', str(train_data_generator.n),
            '\nClasswise train sample support: ', str(collections.Counter(train_data_generator.labels)),
            '\nresult_dir path: ', str(res_dir_this_fold), 
            '\nINPUT IMAGE_SIZE: ',str(IMAGE_SIZE), '\nset EPOCHS: ',str(epochs), 
            '\nvalidation_steps: ',str(validation_steps), '\nTrain batch_size: ',
            str(batch_size), '\nsteps_per_epoch: ',str(steps_per_epoch)]
    
    append_multiple_lines(summary_path, line)
    
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file =res_dir_this_fold +  'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    

    eval(model, test_set, "test_set ", res_dir_this_fold)
    model.evaluate(test_set, steps = test_set.samples)
    #model.predict_classes(test_set,batch_size)
    
    eval(model, valid_data_generator, "val_set ", res_dir_this_fold)
    model.evaluate(valid_data_generator, steps = valid_data_generator.samples)
    
    
    test_ids = getImagesAndLabels(test_path + '/COVID/')
    
    '''
    if resnet==False:
        # For VGG
        preds = GradCam(test_path + '/COVID/' + test_ids[11] +'.png', model,"block5_conv3", res_dir_this_fold, None)
    else:
        # For Resnet
        preds = GradCam(test_path + '/COVID/' + test_ids[11] +'.png', model,"conv5_block3_add", res_dir_this_fold, None)
    '''

   	
    tf.keras.backend.clear_session()
   	
    fold_var += 1
   	
   
end0 = time.time()
print('Training finished!')
print('========Total Cross validation training Time taken: ', (end0-start0)/60,  'minutes ========')

################################################



df = pandas.DataFrame({'Val acc': VALIDATION_ACCURACY, 'Val loss': VALIDATION_LOSS,
                       'Test acc': TEST_ACCURACY, 'Test loss': TEST_LOSS})
df.to_csv(result_dir +"all_acc_test" +".csv")  


# Validation accuray and loss csv
print(df)


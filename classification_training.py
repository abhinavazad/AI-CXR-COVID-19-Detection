# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:14:55 2021

@author: AA086655

"""
'''
The main Classification training script for Covid vs Normal CXR detection.
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
# 6. Resize the cropped CXR img to the input image dimens : 256
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
    Train set-
        - Class1
        - Class2
    Val set-
        - Class1
        - Class2
    Test set-
        - Class1
        - Class2
    
'''

import os
import pandas as pd
from glob import glob
from datetime import datetime
from contextlib import redirect_stdout
import pandas

import matplotlib
from sys import platform
if platform == "linux":
    matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import layers

import time
import collections



from methods_model_training import append_multiple_lines, eval, makemydir, plot_model_hist
from Transfer_learning_models import vgg16_model, vgg19_model, resnet_model, InceptionV3_model, densenet201_model, mobilenet_model, Xception_model


batch_size = 64
epochs= 20

dim = 256
img_width = dim
img_height = dim
IMG_CHANNELS = 1

# INPUT layer size, re-size all the images to this
IMAGE_SIZE = [img_width, img_height]


path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 2c"
print('::::: PATH :', path)

folder_name = os.path.basename(path)
train_path = path + '/train/'
val_path = path + '/val/'
test_path = path + '/test/'

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
    modelname = "VGG16_dl"+ add_info +str(dense)+"mode_" + str(mode) + "_"+ str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
elif vgg19==True:
    model = vgg19_model(IMAGE_SIZE, class_folders,mode,dense)
    modelname = "VGG19_dl"+ add_info +str(dense)+"mode_" + str(mode) + "_"+ str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
elif resnet==True:
    model = resnet_model(IMAGE_SIZE, class_folders,mode,dense)
    modelname = "Resnet50_dl"+ add_info +str(dense) + "bn" + str(mode) +"_"  + str(len(class_folders))+"class_model" + str(dim)+ "clahe_aug" #"RAW_aug"
elif inceptV3==True:
    model = InceptionV3_model(IMAGE_SIZE, class_folders,mode,dense)
    modelname = "InceptionV3_dl"+ add_info +str(dense)+ "bn" + str(mode) +"_"  +str(len(class_folders))+"class_model" + str(dim)+ "clahe_aug" #"RAW_aug"
elif densenet==True:
    model = densenet201_model(IMAGE_SIZE, class_folders,mode,dense)
    modelname = "Densenet_dl" + add_info +str(dense) + "bn" + str(mode) + "_"  +str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
elif mobilenet==True:
    model = mobilenet_model(IMAGE_SIZE, class_folders,mode,dense,0.4)
    modelname = "Mobilenet_dl"+ add_info +str(dense) + "bn" + str(mode) + "_"  +str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"
elif xception==True:
    model = Xception_model(IMAGE_SIZE, class_folders,mode,dense)
    modelname = "Xception_dl"+ add_info +str(dense) + "bn" + str(mode) + "_"  +str(len(class_folders))+"class_model" + str(dim) + "clahe_aug" #"RAW_aug"



# view the final structure of the model
model.summary()

# view the trainable layers of the model
a =[]
'''
for i, layers_ in enumerate(model.layers):
    a.append([i,layers.name, "-", layers_.trainable])
    print(i,layers_.name, "-", layers_.trainable)
'''


# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function= None) #Put : adap_equalize for equalizing the inputs
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function= None) #Put : adap_equalize for equalizing the inputs

# With on-the-go Augmentaion for training set
#train_set = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical')

# No on-the-go Augmentaion
train_set = test_datagen.flow_from_directory(train_path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', interpolation = "bicubic")
val_set = test_datagen.flow_from_directory(val_path, shuffle=False,target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical',interpolation = "bicubic")
test_set = test_datagen.flow_from_directory(test_path, shuffle=False, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', interpolation = "bicubic")


steps_per_epoch = train_set.samples//train_set.batch_size #(len(x_train))//batch_size
validation_steps = val_set.samples//val_set.batch_size #(len(x_val))//batch_size



train_len = str(steps_per_epoch*batch_size) #str(len(x_train))
suffix =  '_ts' + datetime.now().strftime('%m-%d__%H-%M-%S') 

result_dir = os.getcwd()+ '/Results_clahe_prep/'+ modelname + '_'+ suffix + '/'
makemydir(result_dir)
#keras_model_dir = result_dir + '/model_'  + str(img_width) +"_b"+ str(batch_size) + "_X"+ train_len+ '/'

modelpath = result_dir + modelname
#########################################################


#########################################################
# MODEL TRAINING 

# tell the model what cost and optimization method to use
print("[INFO] compiling model...", modelname)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


##Modelcheckpoint
checkpoint = ModelCheckpoint(modelpath + ".hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,  mode='max')
#This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
early_stop = EarlyStopping(monitor='val_accuracy', patience=3,  mode='max', verbose=1, restore_best_weights = True) #min_delta=0.5
#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger(result_dir + 'my_logs.csv', separator=',', append=False)
callbacks_list = [checkpoint, early_stop, log_csv]


print('Lets start training.')
start = time.time()
# fit the model
#########################
history = model.fit(train_set, validation_data=test_set, epochs=epochs, steps_per_epoch= steps_per_epoch, validation_steps=validation_steps,callbacks=callbacks_list)
#########################
#history = model.fit(train_set, validation_data=test_set, epochs=epochs, batch_size = batch_size) #steps_per_epoch=len(train_set), validation_steps=len(val_set))
end = time.time()
print('Training finished! ->',modelname)
print('========Time taken: ', (end-start)/60,  'minutes ========')

################################################
# Saving the model to disk
# serialize model to YAML, saves the model architecture to YAML
model_yaml = model.to_yaml()
with open( modelpath + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(modelpath + ".h5")
model.save(result_dir + 'saved_model')
print("Saved model to disk")

################################################
# SAVING LOGS of training
with open(result_dir + 'modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        
with open(result_dir + 'modelsummary.txt', 'a') as output:
    for row in a:
        output.write(str(row) + '\n')


file = open(result_dir + 'modelsummary.txt', 'a')       
line = ['========Time taken for model training: ', str((end-start)/60),  'minutes ========',
        '\nTraining Dataset path: ', str(path), '\nTrainng Classes: ',str(train_set.class_indices), 
        '\nTotal train samples: ', (train_len), ' ::Out of: ', str(train_set.samples),
        '\nClasswise train sample support: ', str(collections.Counter(train_set.labels)),
        '\nresult_dir path: ', str(result_dir), 
        '\nINPUT IMAGE_SIZE: ',str(IMAGE_SIZE), '\nset EPOCHS: ',str(epochs), 
        '\nvalidation_steps: ',str(validation_steps), '\nTrain batch_size: ',
        str(batch_size), '\nsteps_per_epoch: ',str(steps_per_epoch)]

append_multiple_lines(result_dir + 'modelsummary.txt', line)


# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
# save to csv: 
hist_csv_file =result_dir +  'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# Pnot the model history: accuracy and loss
try:
    plot_model_hist(history, result_dir)
except Exception as e:
    print("Exception due to ERROR: ", e)
    
##############################################################
# TESTING

# redefining Datagenerators with shuffle off
train_set = test_datagen.flow_from_directory(train_path, shuffle=False, target_size=IMAGE_SIZE,
                                             batch_size=batch_size, class_mode='categorical',
                                             interpolation = "bicubic")
val_set = test_datagen.flow_from_directory(val_path, shuffle=False,target_size=IMAGE_SIZE, 
                                           batch_size=1, class_mode='categorical',
                                           interpolation = "bicubic")
test_set = test_datagen.flow_from_directory(test_path, shuffle=False, target_size=IMAGE_SIZE,
                                            batch_size=1, class_mode='categorical',
                                            interpolation = "bicubic")


# confusion matrix and midel evalusaiton
eval(model, test_set, "test_set ", result_dir)
model.evaluate(test_set, steps = test_set.samples, verbose =1)
#model.predict_classes(test_set,batch_size)

eval(model, val_set, "val_set ", result_dir)
model.evaluate(val_set, steps = val_set.samples)



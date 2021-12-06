# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:04:03 2021

@author: abhia
"""

'''
Unet Lungs segementation model training script

# =============================================================================
Steps:-
# 1. read image ids from the dataset path(Rrain, test, Val)
# 2. Resize to 256 for getting mask predictions
# 3. Feed in to the model and train
    - Early stopping
# 4. Confusion matrix and evaluation on the various metrices
# =============================================================================
# =============================================================================
# # Tweakable parameters:-
# # 1. in_path : dirrectory of the train/val/test data
# # 2. result_dir :Rename the Top subfolder, rest is Based on os. getcwd
# # 3. batchsize
# # 4. Epoch
# # 5. modelname : only the unique name(based on feature and class) of the model to be added
# # 6. dim : input dimensions of the images for the training.
# =============================================================================
  
Data folder structure Requirement:
    Train set-
        - CXRs
        - Lung masks
    Val set-
        - CXRs
        - Lung masks
    Test set-
        - CXRs
        - Lung masks
    
'''


import os
import random
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from datetime import datetime
from contextlib import redirect_stdout

import importlib.util
import time


from methods_model_training import append_multiple_lines, makemydir, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat_seg, plot_test_maskout3

# import the custum Unet model architecture
from Unet_Model_CXR_Lungs_seg import get_model


       

size = 256
# Assigninig Image width, hight and chanel(1 for Grayscale)
img_width = size
img_height = size
IMG_CHANNELS = 1

# Assigning Dataset paths
path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Seg_trial_split'; name = '1ClAHE_3Xaug'


print('::::: PATH :', path)
train_path = path + '/train/'
test_path =  path + '/test/'
val_path =  path + '/val/'

cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'



# USE WHEN TRAIN, VAL AND TEST DATASET ARE SEPERATE
# get the image labels out from the given foler path
train_ids = getImagesAndLabels(train_path + cxrs)
test_ids = getImagesAndLabels(test_path + cxrs)
val_ids = getImagesAndLabels(val_path + cxrs)


'''
# For grabbing image IDs from csv file
colnames = ["Train", "Val","Test"]
data = pd.read_csv('train_test_val_ids.csv', names=colnames)
train_ids = data.Train.tolist()
val_ids = data.Val.tolist()
val_ids = [x for x in val_ids if str(x) != 'nan']
test_ids = data.Test.tolist()
test_ids = [x for x in test_ids if str(x) != 'nan']
'''


covid_df = pd.read_csv("C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/working codes/cleaned codes/main/files/21_covid_ids.csv")
covid208_test_ids = covid_df['ids'].tolist()

#############################################

x_train, y_train = load_img_n_masks_fromIDs(train_path, train_ids, size)
x_val, y_val = load_img_n_masks_fromIDs(val_path, val_ids, size)
x_test, y_test = load_img_n_masks_fromIDs(test_path, test_ids, size)

#x_test_covid, y_test_covid = load_img_n_masks_fromIDs("C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Seg prep_all_together", covid208_test_ids, size)


# Preprocessing for image_data_generator.fit function
x_train = np.expand_dims(np.array(x_train), axis = -1)
y_train = np.expand_dims(np.array(y_train), axis = -1)

x_val = np.expand_dims(np.array(x_val), axis = -1)
y_val = np.expand_dims(np.array(y_val), axis = -1)

x_test = np.expand_dims(np.array(x_test), axis = -1)
y_test = np.expand_dims(np.array(y_test), axis = -1)


#x_test_covid = np.expand_dims(np.array(x_test_covid), axis = -1)
#y_test_covid = np.expand_dims(np.array(y_test_covid), axis = -1)


# Get the model defined from get_model.py 
model = get_model(img_height,img_width,IMG_CHANNELS)
model.summary()

# view the trainable layers of the model
a =[]
for i, layers in enumerate(model.layers):
    a.append([i,layers.name, "-", layers.trainable])
    print(i,layers.name, "-", layers.trainable)



batch_size = 32
steps_per_epoch = (len(x_train))//batch_size
validation_steps = (len(x_val))//batch_size
epochs=50  

result_dir = os.getcwd()+ "/SegResults_208covid_added/"+ str(img_width) +"_b"+ str(batch_size) + "_X"+ str(len(x_train)) + "_" + datetime.now().strftime('%m-%d__%H-%M-%S') +'/'
modelname = result_dir +  name + '_unet_lungs_segmtn'   #"Adap_Gausblur_noaug_unet_lungs_segmtn" 
makemydir(result_dir)

#Use Mode = max for accuracy and min for loss. 

# #Modelcheckpoint
checkpoint = ModelCheckpoint(modelname + ".hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,  mode='auto')
#This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.00001,  mode='auto', verbose=1, restore_best_weights = True)
#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger(result_dir + 'my_logs.csv', separator=',', append=False)
callbacks_list = [checkpoint, early_stop, log_csv]

print('Lets start training.')
start = time.time()
#history = model.fit(x_train, y_train, validation_data=(x_val,y_val), steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs) 
history = model.fit(x_train, y_train, validation_data=(x_val,y_val),epochs=epochs, batch_size = batch_size, callbacks=callbacks_list) 
end = time.time()
print('Training finished!')
print('========Time taken: ', (end-start)/60,  'minutes ========')



# Saving the model to disk
# serialize model to YAML
model_yaml = model.to_yaml()
with open( modelname + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(modelname + ".h5")
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
        '\nTraining Dataset path: ', str(path),
        '\nTotal train samples: ', str(len(x_train)),
        '\nresult_dir path: ', str(result_dir), 
        '\nINPUT IMAGE_SIZE: ',str(img_width), '\nset EPOCHS: ',str(epochs), 
        '\nvalidation_steps: ',str(validation_steps), '\nTrain batch_size: ',
        str(batch_size), '\nsteps_per_epoch: ',str(steps_per_epoch)]

append_multiple_lines(result_dir + 'modelsummary.txt', line)


# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
# save to csv: 
hist_csv_file =result_dir +  'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


try:
    plot_model_hist(history, result_dir)
except Exception as e:
    print("Exception due to ERROR: ", e)


# Make prediction on thetraineid model for further evaluations
preds_train = model.predict(x_train, verbose=1)
preds_val = model.predict(x_val, verbose=1)
preds_test = model.predict(x_test, verbose=1)

# Load test covid images for seperate test just of COVID positive CXRs
#preds_test_covid =  model.predict(x_test_covid, verbose=1)

# Set threshold for final segmentation based on the prediction, ideally threshold = 0.5
p=0.5
preds_train_t = (preds_train > p).astype(np.uint8)
preds_val_t = (preds_val > p).astype(np.uint8)
preds_test_t = (preds_test >p).astype(np.uint8)

#preds_test_covid_t = (preds_test_covid > 0.5).astype(np.uint8)




#results

#print("\nCOVID test_res: ")
#covid_test_res = confusion_mat_seg(y_test_covid, preds_test_covid_t, 'preds_test_covid_t', result_dir)
#print(covid_test_res)

print("\ntest_res: ")
test_res = confusion_mat_seg(y_test, preds_test_t, 'preds_test_t', result_dir)
print(test_res)

print("\nval_res: ")
val_res = confusion_mat_seg(y_val, preds_val_t, 'preds_val_t', result_dir)

print("\ntrain_res: ")
train_res = confusion_mat_seg(y_train, preds_train_t, 'preds_train_t', result_dir)
print(train_res)

df1 = pd.DataFrame([train_res + [len(y_train)] +[batch_size, epochs, steps_per_epoch], val_res + [len(y_val)], test_res + [len(y_test)]],
                   index=["train","val","test"], 
                   columns=['IoU score', 'F1 score','Precision:','Sensitivity','Specificity',
                            'Accuracy', 'Datapoints','Batchsize', 'Epochs', 'StepsPerEpoch'])

df1.to_excel(result_dir + "/output"+ datetime.now().strftime('%m-%d__%H-%M-%S') +".xlsx")  
       
print(df1)
print("Saved the results to Dataframe")

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t)-1)
plot_test_maskout3(preds_train_t[ix],x_train[ix], y_train[ix],result_dir, train_ids[ix])

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t)-1)
plot_test_maskout3(preds_val_t[ix],x_val[ix], y_val[ix], result_dir, val_ids[ix] )

# Perform a check on some random test samples
ix = random.randint(0, len(preds_test_t)-1)
plot_test_maskout3(preds_test_t[ix],x_test[ix], y_test[ix], result_dir, test_ids[ix] )

#ix = random.randint(0, len(preds_test_covid_t)-1)
#plot_test_maskout3(preds_test_covid_t[ix],x_test_covid[ix], y_test_covid[ix], result_dir, covid208_test_ids[ix] )





# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:29:35 2021

@author: AA086655
"""

'''
This module consists of several popular pretrained models on imagenet weights
   - Additional pre-processing layers have been added for custom input
   - layers to the end are chopped and custom layer are introduced with a mix of
     Dropout, conv and fully connected dense layers to meet the required classficaiton task

The following models maybe imported form this module to implement transfer learning:
    a. vgg16_model, 
    b. vgg19_model, 
    c. resnet_model, 
    d. InceptionV3_model, 
    e. densenet201_model, 
    f. mobilenet_model, 
    g. Xception_model
    
    
'''

import tensorflow as tf
import matplotlib
from sys import platform
if platform == "linux":
    matplotlib.use('Agg')

import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras import layers



def vgg16_model(IMAGE_SIZE, out_classes, mode,dense):
    # Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG16
    # Here we will be using imagenet weights
    # Similary you can use the same template for Vgg 19, Resnet50, Mobilenet. All you have to import the library. Below are the examples
    vgg16 = VGG16(input_shape=IMAGE_SIZE + [3],
                weights='imagenet', include_top=False)
    
    # don't train existing weights
    for layer in vgg16.layers:
        layer.trainable = False
    
    if mode == 0:
        # Flatenning outputs from the pretraineid model and adding a Softmax dense layer - you can add more if you want
        x = vgg16.layers[-2].output
        #x = vgg16.output
        x = Flatten()(x)
        prediction = Dense(len(out_classes), activation='softmax')(x)
    
    
    # create a model object
    #model = Model(inputs=vgg16.input, outputs=prediction)
    elif mode == 1: 
        new_model = vgg16.layers[-2].output
        new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
        new_model = Flatten(name="flatten")(new_model)
        new_model = Dense(dense, activation="relu")(new_model)
        new_model = Dropout(0.3)(new_model)
        prediction = Dense(len(out_classes), activation="softmax")(new_model)
    
    model = Model(inputs=vgg16.input, outputs= prediction)
    return model


def vgg19_model(IMAGE_SIZE, out_classes, mode,dense):
    # Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG19
    # Here we will be using imagenet weights
    # Similary you can use the same template for Vgg 19, Resnet50, Mobilenet. All you have to import the library. Below are the examples
    vgg19 = VGG19(input_shape=IMAGE_SIZE + [3],
                weights='imagenet', include_top=False)
    
    # don't train existing weights
    for layer in vgg19.layers:
        layer.trainable = False
    
    if mode == 0:
        # Flatenning outputs from the pretraineid model and adding a Softmax dense layer - you can add more if you want
        x = vgg19.layers[-2].output
        #x = vgg16.output
        x = Flatten()(x)
        prediction = Dense(len(out_classes), activation='softmax')(x)
    
    
    # create a model object
    #model = Model(inputs=vgg16.input, outputs=prediction)
    elif mode == 1: 
        new_model = vgg19.layers[-2].output
        new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
        new_model = Flatten(name="flatten")(new_model)
        new_model = Dense(dense, activation="relu")(new_model)
        new_model = Dropout(0.3)(new_model)
        prediction = Dense(len(out_classes), activation="softmax")(new_model)
    
    model = Model(inputs=vgg19.input, outputs= prediction)
    return model

def InceptionV3_model(IMAGE_SIZE, out_classes,mode,dense):
    # Import the InceptionV3 library and add preprocessing layer to the front the predefined layers
    # Here we will be using imagenet weights

    baseModel = InceptionV3(input_shape=IMAGE_SIZE + [3],
                weights='imagenet', include_top=False)

    if mode == 0: 
        for layer in baseModel.layers:
        	layer.trainable = False
        
    elif mode == 1: 
        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True #False for no optimzation 
            else:
                layer.trainable = False
                
    last_layer = baseModel.get_layer('mixed7')
    last_output = last_layer.output
    
    new_model = layers.MaxPooling2D(pool_size=(4, 4))(last_output) 
    new_model = layers.Flatten()(new_model)
    new_model = layers.Dense(dense, activation='relu')(new_model) #1024
    new_model = layers.Dropout(0.4)(new_model)
    prediction = layers.Dense(len(out_classes), activation='softmax')(new_model)
    '''
    #  without chopping off the last set of layers
    new_model = _inceptionv3.layers[-2].output
    #new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
    new_model = Flatten(name="flatten")(new_model)
    new_model = Dense(64, activation="relu")(new_model)
    new_model = Dropout(0.3)(new_model)
    prediction = Dense(len(out_classes), activation="softmax")(new_model)
    '''

    model = Model(inputs=baseModel.input, outputs= prediction)
    return model

def densenet201_model(IMAGE_SIZE, out_classes,mode,dense):
    # Import the Densenet201 library and add preprocessing layer to the front of Densenet201
    # Here we will be using imagenet weights

    baseModel = DenseNet201(input_shape=IMAGE_SIZE + [3], 
                            weights='imagenet', include_top=False)
    
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    if mode == 0: 
        for layer in baseModel.layers:
        	layer.trainable = False
        
    elif mode == 1: 
        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True #False for no optimzation 
            else:
                layer.trainable = False
    # construct the head of the model that will be placed on top of the  the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(dense, activation="relu")(headModel)
    headModel = Dropout(0.4)(headModel)
    headModel = Dense(len(out_classes), activation="softmax")(headModel)
    
    # place the head model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    return model



def resnet_model(IMAGE_SIZE, out_classes, mode,dense):
    # Import the ResNet50 library as shown below and add preprocessing layer to the front of the Resnet
    # Here we will be using imagenet weights
    Resnet50 = ResNet50(input_shape=IMAGE_SIZE + [3],
                weights='imagenet', include_top=False)
        
    if mode == 0:
        # don't train existing weights
        for layer in Resnet50.layers:
            layer.trainable = False
        # Flatenning outputs from the pretraineid model and adding a Softmax dense layer - you can add more if you want

    
    # create a model object
    #model = Model(inputs=ResNet50.input, outputs=prediction)
    elif mode == 1: 
        
        for layer in Resnet50.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True #False for no optimzation 
            else:
                layer.trainable = False
    
    #new_model = Resnet50.output
    new_model = Resnet50.layers[-2].output
    new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
    new_model = Flatten(name="flatten")(new_model)
    new_model = Dense(dense, activation="relu")(new_model)
    new_model = Dropout(0.4)(new_model) #changed from 0.3->0.4 because train acc shows a bit overfit like 0.99 acc
    prediction = Dense(len(out_classes), activation="softmax")(new_model)

    model = Model(inputs=Resnet50.input, outputs= prediction)
    
    return model

def mobilenet_model(IMAGE_SIZE, out_classes, mode,dense,drop):
    # Import the Mobilenet library and add preprocessing layer to the front of Mobilenet
    # Here we will be using imagenet weights
   
    baseModel = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + [3],
                                               include_top=False,
                                               weights='imagenet')   
    if mode == 0: 
        for layer in baseModel.layers:
        	layer.trainable = False
        
    elif mode == 1: 
        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True #False for no optimzation 
            else:
                layer.trainable = False

    
    new_model = baseModel.layers[-2].output
    new_model=AveragePooling2D()(new_model)
    new_model = Flatten(name="flatten")(new_model)
    new_model=Dense(4096,activation='relu')(new_model) 
    new_model=Dense(2048,activation='relu')(new_model) #we add dense layers so that the model can learn more complex functions and classify for better results.
    new_model=Dense(dense,activation='relu')(new_model) 
    new_model = Dropout(drop)(new_model)#dense layer 3
    prediction=Dense(2,activation='softmax')(new_model) #final layer with softmax activation

    '''
    new_model = baseModel.layers[-2].output
    new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
    new_model = Flatten(name="flatten")(new_model)
    new_model = Dense(dense, activation="relu")(new_model)
    new_model = Dropout(drop)(new_model) #changed from 0.3->0.4 because train acc shows a bit overfit like 0.99 acc
    prediction = Dense(len(out_classes), activation="softmax")(new_model)
    '''
    model = Model(inputs=baseModel.input, outputs= prediction)
    
    return model

def Xception_model(IMAGE_SIZE, out_classes,mode,dense):
    baseModel = Xception(input_shape=IMAGE_SIZE + [3], 
                            weights='imagenet', include_top=False)
    
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    if mode == 0: 
        for layer in baseModel.layers:
        	layer.trainable = False
        
    elif mode == 1: 
        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True #False for no optimzation 
            else:
                layer.trainable = False
    # construct the head of the model that will be placed on top of the  the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(dense, activation="relu")(headModel)
    headModel = Dropout(0.4)(headModel)
    headModel = Dense(len(out_classes), activation="softmax")(headModel)
    
    # place the head model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    return model


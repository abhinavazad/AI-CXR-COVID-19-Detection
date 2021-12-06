# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:48:18 2021

@author: AA086655
"""
import cv2
import numpy as np
 
import tensorflow as tf


import keras

import matplotlib.cm as cm



def get_img_array(img_path, size):
    '''
    Prepares image array as per the required dimensionality to fit the framework of gradcam function

    Parameters
    ----------
    img_path : path of the input image
    size : Required size

    Returns
    -------
    img_array : Pre-proccesed image array for Gradcam

    '''
    # `img` is a PIL image of size of the input layer
    img = keras.preprocessing.image.load_img(img_path, target_size=size, interpolation='bicubic')
    #plt.imshow(img)
    #print(type(img))

    # `array` is a float32 Numpy array of shape of the input layer
    img_array = keras.preprocessing.image.img_to_array(img)
    #print("PIL image shape:", img_array.shape)
    #img_array = adap_equalize(img_array)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 256, 256, 3)
    #img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    #print("expanded keras image shape:", img_array.shape)
    
    return img_array

def save_and_display_gradcam(img_path, heatmap, img_size, cam_path, alpha=0.3, beta = 0.7):
    '''
    displays and save the gradcam 

    Parameters
    ----------
    img_path : input image path
    heatmap : input heatmap for the gradcam
    img_size : input path
    cam_path : 
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.3.
    beta : TYPE, optional
        DESCRIPTION. The default is 0.7.

    Returns
    -------
    superimposed_img : Gradcam imposed on the test image

    '''
    # Load the original image
    #img = cv2.imread(img_path,0)
    img = keras.preprocessing.image.load_img(img_path, target_size=img_size, interpolation='bicubic')
    img = keras.preprocessing.image.img_to_array(img)


    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, img_size)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet") #rainbow gnuplot2

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    #jet_heatmap = heatmap

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    #superimposed_img = jet_heatmap * alpha + img *beta
    #superimposed_img = jet_heatmap * img 
    superimposed_img = cv2.addWeighted(jet_heatmap, 0.3, img, 0.7, 0)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))
    return superimposed_img




def make_gradcam_heatmap(img_array, model, conv_layer_name, pred_index=None):
    '''
    Makes Gradcam heatmap for the given layer

    Parameters
    ----------
    img_array : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    conv_layer_name : TYPE
        DESCRIPTION.
    pred_index : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    pred_index : Label for which the Gradcam is to be extracted

    '''
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer or the given conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
            # IF pred_index is not defined, make_gradcam_heatmap will automatically pick up the activation map for max predicted label using np.argmax()
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    print("pred_index: ", pred_index)
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(),pred_index


def GradCam(img_path, model, conv_layer, out_path, pred_index):
    '''
    Accepts a model with a given Conv_layer and makes Gradcam for a particular label(0 or 1)
        - Gradcam can produce activation visualisation for both the labels
        - Simple
    Parameters
    ----------
    img_path : path of th einput image to be tested
    model : model whose gradcam is to be plotted
    conv_layer : Convolutional layer whose gradcam is to be plotted
    out_path : Save path
    pred_index : Label for which the Gradcam is to be extracted

    Returns
    -------
    superimposed_img :  Gradcam imposed on the test image
    conv_layer : string name of the convolutional layer whose gardcam is plotted

    '''
    # The convolution layer, whose activation is to be visulised using gradcam
    conv_layer_name = conv_layer #"block5_conv3"
    
    shape = model.input.shape[1]
    #print("model shape:", shape)
    img_size = (shape, shape)

    # Remove last layer's softmax
    #model.layers[-1].activation = None
    
    # Preparing image as per the Keras input format : (n_samples, height, width, channels)
    # 2 ways:
    # Way 1 using opencv, reading the image as Grayscale and then adds 3 channels after given preprocessings-> Resizing-> and then expanding the axis at 0th index
    in_image = cv2.imread(img_path,0)
    
    # Add all the pre-processings, that you want:
    #in_image = adap_equalize(in_image)
    #in_image = adap_equalize(in_image)
    in_image = cv2.merge((in_image,in_image,in_image))
    
    if in_image.shape != img_size:
        interpolation_type =  cv2.INTER_AREA if in_image.shape[1]>shape else  cv2.INTER_CUBIC
        img = cv2.resize(in_image, img_size, interpolation = interpolation_type)
    print("Image shape:", in_image.shape, "resized to :", img.shape)
    
    img = img/255.0 
    img_array = np.expand_dims(img, axis = 0)
    #print(img_array)
    #print(img_array[0][50][1]==img_array[0][50][2])
    #print("expanded image shape:", img_array.shape)
    
    #################################
    '''
    # Way 2: using Keras,preprcessing to load the image as a PIL image object-> Converting into array-> and then expanding the axis
    # Not able to edit the PIL image using openCV fucntions...
    img_array = get_img_array(img_path, size=img_size)/255.0
    #print(img_array[0][1].shape)
    #print(img_array[0][50][1]==img_array[0][50][1])
    '''
    #print(img_array) 
    preds = model.predict(img_array)
    print(preds)

    n = (len(np.asarray(preds)[0]))
    # IF pred_index is set, make_gradcam_heatmap will pick up the activation map for the given label only.
    '''
    for i in range(n):
        heatmap = make_gradcam_heatmap(img_array, model, conv_layer_name, pred_index=i)
        save_and_display_gradcam(img_path, heatmap, out_path + "gradcam.jpg")
    '''
    heatmap,pred = make_gradcam_heatmap(img_array, model, conv_layer_name, pred_index=pred_index)
    superimposed_img = save_and_display_gradcam(img_path, heatmap,img_size, out_path + "gradcam.jpg")        

    return superimposed_img, conv_layer, pred.numpy()

   


def gradcam_simple(model, img_path, conv_layer_name, output_path_gradcam,pred_index) :
    '''
    Accepts a model with a given Conv_layer and makes Gradcam for a particular label(0 or 1)
        - Gradcam can produce activation visualisation for both the labels
        - makes a simple Gradcam

    Parameters
    ----------
    img_path : path of th einput image to be tested
    model : model whose gradcam is to be plotted
    conv_layer_name : Convolutional layer whose gradcam is to be plotted
    output_path_gradcam : Save path
    pred_index : Label for which the Gradcam is to be extracted
    
    Returns
    -------
    superimposed_img :  Gradcam imposed on the test image
    conv_layer : string name of the convolutional layer whose gardcam is plotted

    '''
    
    shape = model.input.shape[1]
    img_size = (shape, shape)

    img_array = get_img_array(img_path, size=img_size)/255.0
    img = img_array
    
    predict = model.predict(img)
    target_class = np.argmax(predict[0])
    last_conv = model.get_layer(conv_layer_name)


    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output])

    #pred_index = None
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
            # IF pred_index is not defined, make_gradcam_heatmap will automatically pick up the activation map for max predicted label using np.argmax()
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    print("pred_index: ", pred_index)
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    #plt.imshow(heatmap)
    

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # doesnt work with tf.max..

    #plt.imshow(heatmap)
    
    img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
    upsampled_heatmap = cv2.resize(heatmap, img_size)
    
    #superimposed_img = upsample * img_gray
    superimposed_img = cv2.addWeighted(upsampled_heatmap, 0.4, img_gray, 0.6, 0)
    #superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img) # Not needed adn gives error with this

    #plt.imshow(gradcam)
    #output_path = output_path_gradcam + '/gradcam2.jpg'
    #plt.imsave(output_path, upsample * img_gray)
    return superimposed_img,conv_layer_name, pred_index.numpy()


def get_activation_maps(model,img_path, layer):
    '''
    Makes Activation map of the given layer
    - Helps understand feature exraction and the layer 

    Parameters
    ----------
    model : the model with loaded weights
    img_path : path of the image
    layer : Name of the layer to be visualized
            -> if None: then it will print for all the layers

    Returns
    -------
    None.

    '''
    import keract
    shape = model.input.shape[1]
    img_size = (shape, shape)
    
    img_array = get_img_array(img_path, size=img_size)/255.0
    img = img_array
    activations = keract.get_activations(model, img, layer_names=layer)
    keract.display_activations(activations, save=False)



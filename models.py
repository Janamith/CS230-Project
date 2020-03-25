import numpy as np
import pydicom
import os
import random
import imageio
import h5py
import matplotlib.pyplot as plt
import cv2
import subprocess
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, concatenate, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform, Constant
import scipy.misc
from scipy.io import loadmat
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps
from scipy.io import loadmat
from data_processing import load_dataset

import keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


_NUM_VALIDATION = 1000

_RANDOM_SEED = 0

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='output', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(conv_5x5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def Inception_V1(input_shape):

	input_layer = Input(shape=input_shape)

	x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(input_layer)
	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
	x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
	x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

	x = inception_module(x,
						filters_1x1=64,
						filters_3x3_reduce=96,
						filters_3x3=128,
						filters_5x5_reduce=16,
						filters_5x5=32,
						filters_pool_proj=32,
						name='inception_3a')

	x = inception_module(x,
						filters_1x1=128,
						filters_3x3_reduce=128,
						filters_3x3=192,
						filters_5x5_reduce=32,
						filters_5x5=96,
						filters_pool_proj=64,
						name='inception_3b')

	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

	x = inception_module(x,
						filters_1x1=192,
						filters_3x3_reduce=96,
						filters_3x3=208,
						filters_5x5_reduce=16,
						filters_5x5=48,
						filters_pool_proj=64,
						name='inception_4a')


	x1 = AveragePooling2D((5, 5), strides=3)(x)
	x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
	x1 = Flatten()(x1)
	x1 = Dense(1024, activation='relu')(x1)
	x1 = Dropout(0.7)(x1)
	x1 = Dense(1, activation='sigmoid', name='auxilliary_output_1')(x1)

	x = inception_module(x,
						filters_1x1=160,
						filters_3x3_reduce=112,
						filters_3x3=224,
						filters_5x5_reduce=24,
						filters_5x5=64,
						filters_pool_proj=64,
						name='inception_4b')

	x = inception_module(x,
						filters_1x1=128,
						filters_3x3_reduce=128,
						filters_3x3=256,
						filters_5x5_reduce=24,
						filters_5x5=64,
						filters_pool_proj=64,
						name='inception_4c')

	x = inception_module(x,
						filters_1x1=112,
						filters_3x3_reduce=144,
						filters_3x3=288,
						filters_5x5_reduce=32,
						filters_5x5=64,
						filters_pool_proj=64,
						name='inception_4d')


	x2 = AveragePooling2D((5, 5), strides=3)(x)
	x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
	x2 = Flatten()(x2)
	x2 = Dense(1024, activation='relu')(x2)
	x2 = Dropout(0.7)(x2)
	x2 = Dense(1, activation='sigmoid', name='auxilliary_output_2')(x2)

	x = inception_module(x,
						filters_1x1=256,
						filters_3x3_reduce=160,
						filters_3x3=320,
						filters_5x5_reduce=32,
						filters_5x5=128,
						filters_pool_proj=128,
						name='inception_4e')

	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

	x = inception_module(x,
						filters_1x1=256,
						filters_3x3_reduce=160,
						filters_3x3=320,
						filters_5x5_reduce=32,
						filters_5x5=128,
						filters_pool_proj=128,
						name='inception_5a')

	x = inception_module(x,
						filters_1x1=384,
						filters_3x3_reduce=192,
						filters_3x3=384,
						filters_5x5_reduce=48,
						filters_5x5=128,
						filters_pool_proj=128,
						name='inception_5b')

	x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

	x = Dropout(0.4)(x)

	x = Dense(1, activation='sigmoid', name='output')(x)

	model = Model(input_layer, [x, x1, x2], name='inception_v1')

	return model

def Inception_small_V1(input_shape, classes):

	input_layer = Input(shape=input_shape)

	x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(value=0.2))(input_layer)
	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
	x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
	x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

	x = inception_module(x,
						filters_1x1=64,
						filters_3x3_reduce=96,
						filters_3x3=128,
						filters_5x5_reduce=16,
						filters_5x5=32,
						filters_pool_proj=32,
						name='inception_3a')

	x = inception_module(x,
						filters_1x1=128,
						filters_3x3_reduce=128,
						filters_3x3=192,
						filters_5x5_reduce=32,
						filters_5x5=96,
						filters_pool_proj=64,
						name='inception_3b')

	x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

	x = inception_module(x,
						filters_1x1=192,
						filters_3x3_reduce=96,
						filters_3x3=208,
						filters_5x5_reduce=16,
						filters_5x5=48,
						filters_pool_proj=64,
						name='inception_4a')


	x1 = AveragePooling2D((5, 5), strides=3)(x)
	x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
	x1 = Flatten()(x1)
	x1 = Dense(1024, activation='relu')(x1)
	x1 = Dropout(0.7)(x1)
	x1 = Dense(1, activation='sigmoid', name='output')(x1)

	model = Model(input_layer, x1, name='inception_v1_small')

	return model


def LeNet(input_shape):
	X_input = Input(input_shape)

	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides=(2, 2))(X)
	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides=(2, 2))(X)
	X = Flatten()(X)
	X = Dense(500, activation='relu', name='fc1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = Dense(1, activation='sigmoid', name='fc2', kernel_initializer = glorot_uniform(seed=0))(X)

	model = Model(inputs = X_input, outputs = X, name='LeNet')

	return model


def convert_to_one_hot(array):
	new_array = np.empty((array.shape[0], 2))
	for i in range(array.shape[0]):
		if (array[i] == 0):
			new_array[i][0] = 0
			new_array[i][1] = 1
		else:
			new_array[i][0] = 1
			new_array[i][1] = 0
	return new_array

def run(dataset_path):

	model = Inception_V1(input_shape = (256, 256, 1))
	#model = LeNet(input_shape = (256,256,1))
	#model = ResNet50(input_shape = (256, 256, 1))
	#model = Inception_small_V1(input_shape = (256,256,1), classes=2)

	model.compile(optimizer='adam', loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])
	#model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])

	print("Loading dataset")
	X_train, Y_train, X_valid, Y_valid = load_dataset(dataset_path)

	#np.save('./X_train.npy', X_train)
	#np.save('./Y_train.npy', Y_train)
	#np.save('./X_valid.npy', X_valid)
	#np.save('./Y_valid.npy', Y_valid)


	#extract_masks('./segments', './ground_truth', './data')
	
    #X_train = np.load('./X_train.npy')
	#Y_train = np.load('./Y_train.npy')
	#X_valid = np.load('./X_valid.npy')
	#Y_valid = np.load('./Y_valid.npy')

	
	#directories, dir_names = get_patient_directories(dataset_path)
	#get_lung_masks(directories, dir_names)
	

	#segment = np.load('./segment.npy')
	#print(dataset.pixel_array[250])

	Y_train = convert_to_one_hot(Y_train)
	Y_valid = convert_to_one_hot(Y_valid)
	
	class_weight = {0: 1., 1:1.}

	#model.fit(X_train, [Y_train, Y_train, Y_train], epochs = 1, batch_size = 32, class_weight=class_weight)
	model.fit(X_train, Y_train, epochs = 1, batch_size = 32, class_weight=class_weight, validation_split = 0.2)

	results = model.predict(X_valid)
	output = []
	for i in range(len(results)):
		if (results[i] < 0.5):
			output.append(0)
		else:
			output.append(1)
	num_equal = 0

	for i in range(len(output)):
		if (output[i] == Y_valid[i].astype(int)):
			num_equal += 1

	print(num_equal / len(output))

run('./data_png_final')




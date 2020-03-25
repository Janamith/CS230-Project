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

import keras.backend as K
import tensorflow as tf


def get_patient_directories(dataset_path):
	directories = []
	dir_names = []
	for filename in os.listdir(dataset_path):
		path = os.path.join(dataset_path, filename)
		if os.path.isdir(path):
			dir_names.append(filename)
			directories.append(path)
	return directories, dir_names

def get_lung_masks(directories, dir_names):
	counter = 0
	for directory in directories:
		output = './segments/' + dir_names[counter] + '.dcm'
		subprocess.run(["lungmask", directory, output])
		counter += 1

def extract_label(example):
	if 1.0 in example:
		return 1
	else:
		return 0

def write_images(final_images, labels):
	example_count = 0
	for i in range(0, len(final_images)):
		im = Image.fromarray(final_images[i])
		im.save('./data_png_final/out' + str(example_count) + '_' + str(labels[i]) + '.tiff')
		print(example_count)
		example_count += 1

def extract_masks(segment_path, ground_truth_path, image_path):
	image_filenames = get_filenames(segment_path)
	final_images = []
	labels = []
 	
	for i in range(0, len(image_filenames)):
		masks = pydicom.dcmread(image_filenames[i]).pixel_array
		truths = loadmat(ground_truth_path + '/' + image_filenames[i][-10:-4] + '.mat')['Mask']
		for j in range(0, masks.shape[0]):
			if (j + 1 < 10):
				image_full_path = image_path + '/' + image_filenames[i][-10:-4] + '/D000' + str(j + 1) + '.dcm'
			elif (j + 1 < 100):
				image_full_path = image_path + '/' + image_filenames[i][-10:-4] + '/D00' + str(j + 1) + '.dcm'
			else:
				image_full_path = image_path + '/' + image_filenames[i][-10:-4] + '/D0' + str(j + 1) + '.dcm'
			image = pydicom.dcmread(image_full_path).pixel_array
			truth = truths[:, :, j]
			label = extract_label(truth)
			mask_left = masks[j] == 1
			mask_right = masks[j] == 2
			mask = np.where(mask_left + mask_right, image, 0)
			final_images.append(mask)
			labels.append(label)

	write_images(final_images, labels)

def convert_to_png():
	example_count = 0
	for i in range(1, 35):
		if (i + 1 < 10):
			path_base = './data/PAT00' + str(i+1)
			ground_truth_str = './ground_truth/PAT00' + str(i+1) + '.mat'
		else:
			path_base = './data/PAT0' + str(i+1)
			ground_truth_str = './ground_truth/PAT0' + str(i+1) + '.mat'

		path_str = path_base + '/D0001.dcm'
		ext = 1
		while (path.exists(path_str)):
			dataset = rescale(pydicom.dcmread(path_str).pixel_array, 0.25, anti_aliasing=False)
			print(dataset)
			im = Image.fromarray(dataset)
			im.save('./data_png/image' + str(example_count) + '.png')
			ext += 1
			if (ext < 10):
				path_str = path_base + '/D000' + str(ext) + '.dcm'
			elif (ext < 100):
				path_str = path_base + '/D00' + str(ext) + '.dcm'
			else:
				path_str = path_base + '/D0' + str(ext) + '.dcm'
			example_count += 1
			print(example_count)


def get_filenames(dataset_path):
	filenames = []
	for filename in os.listdir(dataset_path):
		path = os.path.join(dataset_path, filename)
		filenames.append(path)
	return filenames


def load_dataset(dataset_path):
	image_filenames = get_filenames(dataset_path)
	random.shuffle(image_filenames)
	training_filenames = image_filenames[_NUM_VALIDATION:]
	validation_filenames = image_filenames[:_NUM_VALIDATION]

	training_labels = get_labels(training_filenames)
	validation_labels = get_labels(validation_filenames)

	X_train = np.empty((len(training_filenames) + (sum(training_labels) * 3), 256, 256, 1))
	Y_train = np.empty((len(training_filenames) + (sum(training_labels) * 3), 1))
	X_valid = np.empty((len(validation_filenames), 256, 256, 1))
	Y_valid = np.empty((len(validation_filenames), 1))


	j = 0
	for i in range(0, len(training_filenames)):
		print("loading training image" + str(i))
		if not training_filenames[i].endswith('_0.tiff') and not training_filenames[i].endswith('_1.tiff'):
			continue
		dataset = Image.open(training_filenames[i])
		dataset_mirror = ImageOps.mirror(dataset)
		dataset_flip = ImageOps.flip(dataset)
		dataset_rotated = dataset.rotate(45)
		if (training_labels[i] == 1):
			X_train[j] = np.reshape(np.array(dataset), (512,512,1))[0::2, 0::2]
			X_train[j+1] = np.reshape(np.array(dataset_mirror), (512,512,1))[0::2, 0::2]
			X_train[j+2] = np.reshape(np.array(dataset_flip), (512,512,1))[0::2, 0::2]
			X_train[j+3] = np.reshape(np.array(dataset_rotated), (512,512,1))[0::2, 0::2]
			Y_train[j] = training_labels[i]
			Y_train[j+1] = training_labels[i]
			Y_train[j+2] = training_labels[i]
			Y_train[j+3] = training_labels[i]
			j += 4
		else:
			X_train[j] = np.reshape(np.array(dataset), (512,512,1))[0::2, 0::2]
			Y_train[j] = training_labels[i]
			j += 1

	j = 0
	for i in range(0, len(validation_filenames)):
		print("loading validation image" + str(i))
		if not validation_filenames[i].endswith('_0.tiff') and not validation_filenames[i].endswith('_1.tiff'):
			continue
		dataset = Image.open(validation_filenames[i])
		dataset_mirror = ImageOps.mirror(dataset)
		dataset_flip = ImageOps.flip(dataset)
		dataset_rotated = dataset.rotate(45)
		if (validation_labels[i] == 1):
			X_valid[j] = np.reshape(np.array(dataset), (512,512,1))[0::2, 0::2]
			Y_valid[j] = validation_labels[i]
			j += 1
		else:
			X_valid[j] = np.reshape(np.array(dataset), (512,512,1))[0::2, 0::2]
			Y_valid[j] = validation_labels[i]
			j += 1
	return X_train, Y_train, X_valid, Y_valid
import cv2
import pathlib

import scipy
import computer_vision
import BitVector

import matplotlib.pylab as plt
# import matplotlib as mpl
import numpy as np
# import math

import os
import glob
import time
import torch

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
torch.set_grad_enabled(False)

from vgg_and_resnet import VGG19, CustomResNet


class FeatureExtractor:
	"""Implements functionality to extract texture based features from the
	image and save them to the path specified.
	"""
	def __init__(self, categories, data_dir, sub_dir='data') -> None:
		self.categories = np.array(categories)
		self.data_dir = data_dir
		self.sub_dir = sub_dir

		self.output_dir = os.path.join(self.data_dir, 'features')
		# Create the directory (and any necessary parent directories)
		os.makedirs(self.output_dir, exist_ok=True)

	def get_train_test_splits(self, method='LBP'):
		"""Returns data as train-test splits"""
		train_features, test_features = self.get_features(method=method)
		train_x, train_y, train_image_names = FeatureExtractor.get_inputs_and_labels(train_features, self.categories)

		test_x, test_y, test_image_names = FeatureExtractor.get_inputs_and_labels(test_features, self.categories)
		return train_x, train_y, test_x, test_y, test_features, test_image_names

	def get_features(self, method='LBP'):
		"""Returns features for the method specified.
		
		Args:
			method: choices = ['LBP', 'VGG', 'RESNET_coarse', 'RESNET_fine', 'AdaIN']
		"""
		choices = ['LBP', 'VGG', 'RESNET_coarse', 'RESNET_fine', 'AdaIN']
		assert method in choices, print(f"Select method from {choices}...")
		print(f"Reading features for '{method}' from disk...")
		filepath = os.path.join(self.output_dir, f'{method}_training.npz')
		train_features = np.load(filepath, allow_pickle=True)['features'].item()
		
		filepath = os.path.join(self.output_dir, f'{method}_testing.npz')
		test_features = np.load(filepath, allow_pickle=True)['features'].item()
		
		return train_features, test_features
	
	def extract_and_save_AdaIN_features(self):
		"""Extracts AdaIN features and saves them to the disk."""
		print(f"For AdaIN features using VGG features...")
		vgg = VGG19()
		vgg.load_weights('vgg_normalized.pth')    
		splits = ['training', 'testing']
		for split in splits:
			print(f"For '{split}'...")
			images_dir =  os.path.join(self.data_dir, self.sub_dir, split)
			
			filepaths = glob.glob(os.path.join(images_dir, "*.jpg"))
			AdaIN_features = {}

			for filepath in filepaths:
	
				try:
					img = FeatureExtractor.read_image_from_path(filepath)
					AdaIN_feat = FeatureExtractor.get_AdaIN_features_for_image(img, vgg)

					image_name = os.path.basename(filepath)
					image_name = image_name.rsplit('.', 1)[0]	            
					AdaIN_features[image_name] = AdaIN_feat
				except:
					print(f"Error loading image: {os.path.basename(filepath)}")

			out_path = os.path.join(self.output_dir, f'AdaIN_{split}')
			np.savez_compressed(out_path, features=AdaIN_features)
			print(f"AdaIN features saved to {out_path}.npz")
	
	def get_AdaIN_features_for_image(img, vgg):
		"""Extracts VGG features for the given image.
		"""
		x = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
		# Obtain the output feature map
		vgg_feature = vgg(x)
		vgg_feature = vgg_feature.reshape(vgg_feature.shape[0], -1)
		mu = np.mean(vgg_feature, axis=1)
		sigma = np.std(vgg_feature, axis=1)

		return np.concatenate([mu, sigma])


	
	def extract_and_save_RESNET50_features(self):
		"""Extracts LBP features and saves them to the disk."""
		print(f"Using RESNET-50 to extract features...")
		encoder_name='resnet50'
		resnet = CustomResNet(encoder=encoder_name)   
		splits = ['training', 'testing']
		for split in splits:
			print(f"For '{split}'...")
			images_dir =  os.path.join(self.data_dir, self.sub_dir, split)
			
			filepaths = glob.glob(os.path.join(images_dir, "*.jpg"))
			RESNET_coarse_features = {}
			RESNET_fine_features = {}

			for filepath in filepaths:
	
				try:
					img = FeatureExtractor.read_image_from_path(filepath)
					renset_coarse, resnet_fine = FeatureExtractor.get_RESNET50_features_for_image(img, resnet)

					image_name = os.path.basename(filepath)
					image_name = image_name.rsplit('.', 1)[0]	            
					RESNET_coarse_features[image_name] = renset_coarse
					RESNET_fine_features[image_name] = resnet_fine
				except:
					print(f"Error loading image: {os.path.basename(filepath)}")

			# saving coarse features
			out_path = os.path.join(self.output_dir, f'RESNET_coarse_{split}')
			np.savez_compressed(out_path, features=RESNET_coarse_features)
			print(f"RESNET_coarse features saved to {out_path}.npz")

			# saving fine features
			out_path = os.path.join(self.output_dir, f'RESNET_fine_{split}')
			np.savez_compressed(out_path, features=RESNET_fine_features)
			print(f"RESNET_fine features saved to {out_path}.npz")
	
	def get_RESNET50_features_for_image(img, resnet):
		"""Extracts RESNET50 features for the given image.
		"""
		x = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
		# Obtain the output feature map

		resnet_feat_coarse, resnet_feat_fine = resnet(x)     

		resnet_feat_coarse = resnet_feat_coarse.reshape(resnet_feat_coarse.shape[0], -1)
		gram_mat_coarse = np.matmul(resnet_feat_coarse, resnet_feat_coarse.T)
		gram_mat_coarse = cv2.resize(gram_mat_coarse, (32, 32), interpolation=cv2.INTER_AREA)

		resnet_feat_fine = resnet_feat_fine.reshape(resnet_feat_fine.shape[0], -1)
		gram_mat_fine = np.matmul(resnet_feat_fine, resnet_feat_fine.T)
		gram_mat_fine = cv2.resize(gram_mat_fine, (32, 32), interpolation=cv2.INTER_AREA)
		return gram_mat_coarse, gram_mat_fine

	def extract_and_save_VGG_features(self):
		"""Extracts VGG features and saves them to the disk."""
		print(f"Using VGG to extract features...")
		vgg = VGG19()
		vgg.load_weights('vgg_normalized.pth')    
		splits = ['training', 'testing']
		for split in splits:
			print(f"For '{split}'...")
			images_dir =  os.path.join(self.data_dir, self.sub_dir, split)
			
			filepaths = glob.glob(os.path.join(images_dir, "*.jpg"))
			VGG_features = {}

			for filepath in filepaths:
	
				try:
					img = FeatureExtractor.read_image_from_path(filepath)
					VGG_feature = FeatureExtractor.get_VGG_features_for_image(img, vgg)

					image_name = os.path.basename(filepath)
					image_name = image_name.rsplit('.', 1)[0]	            
					VGG_features[image_name] = VGG_feature
				except:
					print(f"Error loading image: {os.path.basename(filepath)}")

			out_path = os.path.join(self.output_dir, f'VGG_{split}')
			np.savez_compressed(out_path, features=VGG_features)
			print(f"VGG features saved to {out_path}.npz")
	
	def get_VGG_features_for_image(img, vgg):
		"""Extracts VGG features for the given image.
		"""
		x = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
		# Obtain the output feature map
		vgg_feature = vgg(x)
		vgg_feature = vgg_feature.reshape(vgg_feature.shape[0], -1)
		gram_mat = np.matmul(vgg_feature, vgg_feature.T)
		gram_mat = cv2.resize(gram_mat, (32, 32), interpolation=cv2.INTER_AREA)
		return gram_mat


	def extract_and_save_LBP_features(self, R=1, P=8, resize=True):
		"""Extracts LBP features and saves them to the disk."""
		splits = ['training', 'testing']
		for split in splits:
			print(f"For '{split}'...")
			images_dir =  os.path.join(self.data_dir, self.sub_dir, split)
			
			filepaths = glob.glob(os.path.join(images_dir, "*.jpg"))
			LBP_features = {}

			for filepath in filepaths:
	
				try:
					img = FeatureExtractor.read_image_from_path(filepath)
					LBP = FeatureExtractor.get_LBP_features_for_image(img, R=R, P=P, resize=resize)

					image_name = os.path.basename(filepath)
					image_name = image_name.rsplit('.', 1)[0]	            
					LBP_features[image_name] = LBP
				except:
					print(f"Error loading image: {os.path.basename(filepath)}")

			out_path = os.path.join(self.output_dir, f'LBP_{split}')
			np.savez_compressed(out_path, features=LBP_features)
			print(f"LBP features saved to {out_path}.npz")

	def get_LBP_features_for_image(img, R=1, P=8, resize=True):
		"""Extracts 'local binary pattern (LBP)' features 
		for the given image.
		"""
		if resize:
			# downsample image to 64x64
			new_dimensions = (64, 64)
			downsampled_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
			HSI_img = FeatureExtractor.RGB_to_HSI(downsampled_image)
		else:
			HSI_img = FeatureExtractor.RGB_to_HSI(img)
		H = HSI_img[...,0]
		### Local binary pattern
		LBP_features = np.zeros_like(H)

		neighbor_offsets = np.array([(R*np.cos(2*np.pi*p/P), R*np.sin(2*np.pi*p/P)) for p in range(P)]) 
		pixel_y, pixel_x = np.meshgrid(range(1, H.shape[0]-1), range(1, H.shape[1]-1))
		pixels = np.stack([pixel_x.flatten(), pixel_y.flatten()])

		neighboring_pixels = pixels[None,:,:] + neighbor_offsets[:,:,None]

		central_values, _ = FeatureExtractor.bilinear_interpolation(H, pixels)
		neightboring_values = np.zeros((P, neighboring_pixels.shape[-1]))
		for i in range(P):
			img_values, _ = FeatureExtractor.bilinear_interpolation(H, neighboring_pixels[i])
			neightboring_values[i] = img_values.squeeze()

		binary_patterns = np.zeros(neightboring_values.shape, dtype=np.bool)
		binary_patterns[neightboring_values > central_values] = 1

		encodings = FeatureExtractor.get_LBP_encodings(binary_patterns)

		LBP_features[pixel_y.flatten(), pixel_x.flatten()] = encodings
		
		return np.histogram(LBP_features.flatten(), bins=np.arange(P+2))[0]
	

	@staticmethod
	def get_LBP_encodings(binary_patterns):
		"""Returns encodings for the input binary pattens.
		Args:
			binary_patterns: (P, N)

		Returns:
			np array of size N
		"""
		P, N = binary_patterns.shape
		encodings = []
		for n in range(N):
			bv = BitVector.BitVector(bitlist=binary_patterns[...,n])

			minbv = BitVector.BitVector(intVal = min([int(bv<<1) for p in range(P)]), size=P)
			runs = minbv.runs()
			if len(runs) == 1 and runs[0] == 0:
				enc = 0
			if len(runs) == 1 and runs[0] != 0:
				enc = P
			elif len(runs) == 2:
				enc = len(runs[1])
			else:
				enc = P+1

			encodings.append(enc)    
		return np.array(encodings)
	
	@staticmethod
	def RGB_to_HSI(RGB_img):
		"""Converts RGB for the input image, to HSI"""
		RGB_img = RGB_img/255.0
		R = RGB_img[...,0]
		G = RGB_img[...,1]
		B = RGB_img[...,2]

		H = np.zeros(RGB_img.shape[:2])
		S = np.zeros(RGB_img.shape[:2])
		M = np.max(RGB_img, axis=2)
		m = np.min(RGB_img, axis=2)
		c = M - m
		
		mask = (M == R) & (c != 0)
		H[mask] = 60*(((G[mask] - B[mask])/c[mask]) % 6)


		mask = (M == G) & (c != 0)
		H[mask] = 60*(((B[mask] - R[mask])/c[mask]) + 2)

		mask = (M == B) & (c != 0)
		H[mask] = 60*(((R[mask] - G[mask])/c[mask]) + 4)

		I = (R + G + B)/3

		mask = c != 0
		S[mask] = 1 - m[mask]/c[mask]
		
		return np.stack([H,S,I], axis=-1)
	
	@staticmethod
	def read_image_from_path(image_path):
		"""Reads images from the subdir"""
		img_bgr = cv2.imread(image_path)
		# Convert the image from BGR to RGB format
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		return img_rgb
	
	@staticmethod
	def bilinear_interpolation(img, coordinates):
		"""For the given img and coordinates, uses bilinear interpolation
		and returns img values for the input coordinates.
		Args:
			transformed_coordinates = (2, n)
		"""
		# making sure img is 3 dimensional....
		if img.ndim ==2:
			img = img[...,None]
		img_height, img_width = img.shape[:2]
		# Extract integer and fractional parts of coordinates
		x_trans = coordinates[0]
		y_trans = coordinates[1]
		x_floor = np.floor(x_trans).astype(int)
		y_floor = np.floor(y_trans).astype(int)
		x_ceil = np.ceil(x_trans).astype(int)
		y_ceil = np.ceil(y_trans).astype(int)

		# Clip coordinates to image boundaries
		x_floor = np.clip(x_floor, 0, img_width - 1)
		y_floor = np.clip(y_floor, 0, img_height - 1)
		x_ceil = np.clip(x_ceil, 0, img_width - 1)
		y_ceil = np.clip(y_ceil, 0, img_height - 1)

		# Bilinear interpolation weights
		wx = x_trans - x_floor  # fractional part in x
		wy = y_trans - y_floor  # fractional part in y

		# Perform bilinear interpolation
		img_values = ((1 - wx) * (1 - wy))[:, None] * img[y_floor, x_floor] + \
					(wx * (1 - wy))[:, None] * img[y_floor, x_ceil] + \
					((1 - wx) * wy)[:, None] * img[y_ceil, x_floor] + \
					(wx * wy)[:, None] * img[y_ceil, x_ceil]
		# Create a mask for valid (within image bounds) coordinates
		mask_coordinates = (x_trans >= 0) & (x_trans < img_width) & \
						(y_trans >= 0) & (y_trans < img_height)
		return img_values.squeeze(), mask_coordinates.squeeze()

	@staticmethod
	def get_inputs_and_labels(features, categories):
		x = []
		y = []
		image_names = {}
		for i, (image_name, feat) in enumerate(features.items()):
			if feat.ndim == 2:
				# extract upper trianlge elements...
				n,m =feat.shape
				feat = feat[np.triu_indices(n, m=m)]
			x.append(feat)
			y.append(next((i for i, label in enumerate(categories) if label in image_name), -1))

			image_names[i] = image_name
		x = np.array(x)
		y = np.array(y)
		return x, y, image_names
	

class Classifier:
	def __init__(self, method, categories, output_dir, data_dir, subdir='data') -> None:
		"""Creates an instance of feature extractor and uses that to get 
		train/test splits of data. In addition, it implements linear fit 
		and provides plotting functions.
		"""
		choices = ['LBP', 'VGG', 'RESNET_coarse', 'RESNET_fine', 'AdaIN']
		assert method in choices, print(f"Select method from {choices}!")
		
		feature_extractor = FeatureExtractor(categories, data_dir, sub_dir=subdir)
		
		train_x, train_y, test_x, test_y, test_features, test_image_names = feature_extractor.get_train_test_splits(method=method)
		self.test_x, self.test_y, self.test_features = test_x, test_y, test_features
		self.test_image_names = test_image_names

		self.svm_classifier = LinearSVC()
		print(f"Fitting classifier for {method}...")
		self.svm_classifier.fit(train_x, train_y)

		self.categories = categories
		self.method = method
		self.output_dir = output_dir
		self.data_dir = data_dir
		self.subdir = subdir
		
	def plot_confusion_matrix(self, savefig=True):
		"""Using the trained linear classifier, and test data,
		plots confusion matrix as well as computes accuracy.
		"""
		pred_y = self.svm_classifier.predict(self.test_x)

		# Evaluate the classifier
		acc = accuracy_score(self.test_y, pred_y)
		conf_matrix = confusion_matrix(self.test_y, pred_y)
		fig, ax = plt.subplots(figsize=(4,3))
		sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
				xticklabels=categories, yticklabels=categories, ax=ax)
		
		ax.set_title(f"{self.method}, acc: {acc*100:.0f}%")
		if savefig:
			filepath = os.path.join(self.output_dir, f'{self.method}_c_matrix.png')
			plt.savefig(filepath)
			print(f"For {self.method}: \nconfusion matrix saved to: {filepath}")

			
	def plot_example_features(self):
		"""plots randomly choosen example features for all categories"""
		figsize = (4,3)
		filenames = np.array(list(self.test_features.keys()))
		print(f"For {self.method}: ")
		for cat in self.categories:
			filename = np.random.choice(filenames[np.char.find(filenames, cat) != -1])
			features = self.test_features[str(filename)]
			fig, ax = plt.subplots(figsize=figsize)
			if self.method == 'LBP':
				ax.bar(np.arange(features.shape[0]), features)
			else:
				features = 255*(features - np.min(features))/(np.max(features) - np.min(features))
				mappable = ax.imshow(features.astype(np.uint8), cmap='viridis')
				plt.colorbar(mappable)
			ax.set_title(f"{self.method}, '{cat}'")
			filepath = os.path.join(self.output_dir, f'{self.method}_features_{cat}.png')
			plt.savefig(filepath)
			print(f"\t {cat} example features saved to: {filepath}")

	def plot_qualitative_results(self, figsize=(4,3)):
		"""plots one correctly classified and one mis-classified images"""

		pred_y = self.svm_classifier.predict(self.test_x)

		# correct example
		ind_correct = int(np.random.choice(np.where(pred_y == self.test_y)[0]))
		filename = self.test_image_names[ind_correct]
		ground_truth_label = self.categories[self.test_y[ind_correct]]
		predicted_label = self.categories[pred_y[ind_correct]]
		self.display_image(filename, ground_truth_label, predicted_label, correct=True, figsize=figsize)

		# incorrect example
		ind_correct = int(np.random.choice(np.where(pred_y != self.test_y)[0]))
		filename = self.test_image_names[ind_correct]
		ground_truth_label = self.categories[self.test_y[ind_correct]]
		predicted_label = self.categories[pred_y[ind_correct]]
		self.display_image(filename, ground_truth_label, predicted_label, correct=False, figsize=figsize)

	def display_image(self, filename, ground_truth, predicted_label, correct=True, figsize=(4,3)):
		"""Displays image with the given filename, adds ground truth 
		and predicted labels.
		"""
		filepath = os.path.join(data_dir, subdir, 'testing', f"{filename}.jpg")
		img = FeatureExtractor.read_image_from_path(filepath)
		fig, ax = plt.subplots(figsize=(4,4))
		ax.imshow(img)
		ax.set_title(f"Ground truth: {ground_truth}\n Predicted label: {predicted_label}")
		if correct:
			ind = 'correct'
		else:
			ind = 'missclassified'
		filepath = os.path.join(self.output_dir, f"{self.method}_example_{ind}.png")
		plt.savefig(filepath)
		print(f"\t example of {ind} result saved to: {filepath}")


if __name__ == '__main__':

	categories = ['cloudy', 'rain', 'shine', 'sunrise']
	output_dir = r'C:\Users\ahmedb\projects\computer-vision\images\hw7'
	data_dir = r'C:\Users\ahmedb\projects\computer-vision\HW7-Auxilliary'
	subdir = 'data'

	# loading features and saving to disk.
	feature_extractor = FeatureExtractor(categories, data_dir, sub_dir=subdir)
	feature_extractor.extract_and_save_AdaIN_features()
	feature_extractor.extract_and_save_LBP_features()
	feature_extractor.extract_and_save_VGG_features()
	feature_extractor.extract_and_save_RESNET50_features()

	# using the saved features, implements classifier and generates results
	methods = ['LBP', 'VGG', 'RESNET_coarse', 'RESNET_fine', 'AdaIN']
	for method in methods:
		classifier = Classifier(method, categories, output_dir, data_dir, subdir)

		if method != 'AdaIN':
		    # plot example features
		    classifier.plot_example_features()

		# confusion matrix
		classifier.plot_confusion_matrix()

		# plot qualititive results
		classifier.plot_qualitative_results(figsize=(4,3))


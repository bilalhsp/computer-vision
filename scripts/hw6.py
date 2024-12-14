import cv2
import pathlib
# import skimage
import scipy
import computer_vision

import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import math


def read_image(subdir, image_name):
	"""Reads images from the subdir"""
	path = pathlib.Path(computer_vision.__path__[0])
	image_path = path.parents[0] / 'images' /subdir/image_name
	img_bgr = cv2.imread(image_path)
	# Convert the image from BGR to RGB format
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	return img_rgb

def save_image(image, subdir, image_name, grayscale=False):
	"""Saves image to the subdir"""
	path = pathlib.Path(computer_vision.__path__[0])
	image_path = path.parents[0] / 'images' /subdir/image_name
	# Save the image with specific quality
	if not grayscale:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(image_path, image)
	

def otsu_single_itr(img, L=256, mask=None):
	"""Performs one iteration of otsu algorithm.
	
	Args:
		img: single level image
		L: number of of gray levels
		mask: mask to focus on image regions

	Returns:
		threshold
		gray_level_index
	"""
	epsilon = 1.e-6
	if mask is None:
		mask = np.where(img > 0)
	bins = np.linspace(0, 255, num=L)
	prob_dist, bins = np.histogram(img[mask].flatten(), bins=bins, density=True)
	# getting upper bin edges
	gray_levels = bins[1:]
	mu_t = np.sum(gray_levels*prob_dist)
	max_var = 0
	threshold = 0
	for k in range(1, L-1):
		mu_k = np.sum(gray_levels[:k]*prob_dist[:k])
		w_k = np.sum(prob_dist[:k])
		var_b = ((mu_t*w_k - mu_k)**2)/(w_k*(1-w_k)+epsilon)		# OTSU threshold selection method: Eq. 18
		
		if var_b > max_var:
			max_var = var_b
			threshold = gray_levels[k]

	# print(f"threshold: {threshold}, max var_b: {max_var}")
	return threshold

def compute_prob_foreground(img, threshold, flip=False, L=256):
	"""Given the image and threshold, computes the probability
	of pixel values greater than threshold i.e. prob of foreground object.
	"""
	bins = np.linspace(0, 255, num=L)
	prob_dist, bins = np.histogram(img.flatten(), bins=bins, density=True)
	gray_levels = bins[1:]
	if flip:
		return np.sum(prob_dist[np.where(gray_levels < threshold)])
	else:      
		return np.sum(prob_dist[np.where(gray_levels > threshold)])


def otsu_single_channel(
		img, L = 256, max_itr=10, flip=False):
	"""Performs multiple iterations of otsu algorithm, stops if 
	number of iterations is equal to the max_itr or probability of foreground
	object is less than or equal to the given estimate of probability.
	 
	Args:
		img: input image (single channel)
		L: number of gray levels, default=256.
		max_itr: max number of iterations, default=5
		flip: If yes, smaller pixels values are considered as foreground.
	"""
	mask = None
	for itr in range(max_itr):
		
		threshold = otsu_single_itr(img, L=L, mask=mask)
		if flip:
			mask = np.where(img < threshold)
		else:
			mask = np.where(img > threshold)

	print(f"Selected threshold: {threshold}")
	segmented_img = np.zeros_like(img)
	segmented_img[mask] = 255
	return segmented_img.astype(np.uint8)


def get_segmentation_mask_using_RGB(img, flip=None, L=256, max_itr=None):
	"""computes segmentation mask using RGB channels"""
	# mask using channel R
	if flip is None:
		flip = [False, False, False]
	if max_itr is None:
		max_itr = [1,1,1]
	mask_R = otsu_single_channel(
		img[...,0], flip=flip[0], L=L, max_itr=max_itr[0]
		)
	
	# mask using channel R
	mask_G = otsu_single_channel(
		img[...,1], flip=flip[1], L=L, max_itr=max_itr[1]
		)
	
	# mask using channel R
	mask_B = otsu_single_channel(
		img[...,2], flip=flip[2], L=L, max_itr=max_itr[2]
		)
	
	mask_combined = mask_R & mask_G & mask_B

	return mask_combined, mask_R, mask_G, mask_B



def dilation(img, kernel_size=3):
	"""Dilation implemented for single channel images,
	convert image to grayscale if it is 3 channel 
	"""
	if len(img.shape) ==3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height, width = img.shape
	k = kernel_size//2
	dilated_img = np.zeros_like(img)
	for i in range(k, height - k):
		for j in range(k, width - k):
			dilated_img[i,j] = np.max(img[i-k:i+k+1, j-k:j+k+1])
			
	return dilated_img

def erosion(img, kernel_size=3):
	"""Erosion implemented for single channel images,
	convert image to grayscale if it is 3 channel 
	"""
	if len(img.shape) ==3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height, width = img.shape
	k = kernel_size//2
	eroded_img = np.zeros_like(img)
	for i in range(k, height - k):
		for j in range(k, width - k):
			eroded_img[i,j] = np.min(img[i-k:i+k+1, j-k:j+k+1])
	return eroded_img

def get_contour_mask(img):
	"""Expects a binary image at the input.
	"""
	if len(img.shape) ==3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height, width = img.shape
	k = 1
	img = img.copy().astype(np.int32)
	contour_mask = np.zeros_like(img)
	for i in range(k, height - k):
		for j in range(k, width - k):
			if (img[i,j] > 0) and (np.sum(img[i-k:i+k+1, j-k:j+k+1]) < 9*img[i,j]):
				# if first condition is true, the sum of the pixel (i,j) and its 8
				# neighbors can only be less than 9, when at least one of its neighbors is 0
				# which the condition we wanted to check for contour extraction.
				contour_mask[i,j] = img[i,j]
	return contour_mask.astype(np.uint8)

def get_zero_padded_image(img, k):
    """zero pads input
    """
    height, width = img.shape[:2]
    zero_padded_img = np.zeros((height+2*k, width+2*k))
    zero_padded_img[k:k+height, k:k+width] = img
    return zero_padded_img

def get_texture_map(img, kernel_size):
	"""Returns the texture map of the image, as the variance of 
	pixels in the NxN neighborhood.
	"""

	if len(img.shape) ==3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height, width = img.shape
	k = kernel_size//2
	img = img.copy().astype(np.float32)
	texture_map = np.zeros_like(img)
	zero_padded_img = get_zero_padded_image(img, k)
	for i in range(height):
		for j in range(width):
			local_var = np.var(zero_padded_img[i-k:i+k+1, j-k:j+k+1])
			texture_map[i,j] = local_var
	texture_map = np.where(np.isnan(texture_map), 0, texture_map)
	texture_map = 255*texture_map/(np.max(texture_map) - np.min(texture_map))
	return texture_map.astype(np.uint8)

def extract_contour(img, kernel_size, morphological_operator='grad'):

	if morphological_operator == 'grad':

		dilated = dilation(img, kernel_size)
		erroded = erosion(img, kernel_size)
		morph_out = dilated - erroded
	elif morphological_operator =='open':
		erroded = erosion(img, kernel_size)
		morph_out = dilation(erroded, kernel_size)
		
	elif morphological_operator == 'close':
		dilated = dilation(img, kernel_size)
		morph_out = erosion(dilated, kernel_size)

	contour_map = get_contour_mask(morph_out)
	return contour_map


def closing(img, kernel_size):
	output_img = img.copy()
	output_img = dilation(output_img, kernel_size)
	output_img = erosion(output_img, kernel_size)
	return output_img

def opening(img, kernel_size):
	output_img = img.copy()
	output_img = erosion(output_img, kernel_size)
	output_img = dilation(output_img, kernel_size)
	return output_img

def get_segmentation_using_texture(img, kernel_sizes, L=256, flip=None, max_itr=None):

	texture_maps = [get_texture_map(img, kernel_size=kernel_size) for kernel_size in kernel_sizes]
	if flip is None:
		flip = [False, False, False,]
	if max_itr is None:
		max_itr = [2,2,2]
	masks = [otsu_single_channel(
		texture_map, flip=flip[i], L=L, max_itr=max_itr[i]
		) for i, texture_map in enumerate(texture_maps)]

	mask_combined = masks[0] & masks[1] & masks[2]
	return mask_combined, masks[0], masks[1], masks[2]



if __name__ == '__main__':
	
	# Task-1: flower_small
	# segmentation mask using RGB
	subdir = 'hw6'
	image_name = 'flower_small'
	# reading images...
	img = read_image(subdir, f"{image_name}.jpg")
	L = 256
	max_itr = [2,2,2]
	flip=[False, False, False]
	mask_combined, mask_R, mask_G, mask_B = get_segmentation_mask_using_RGB(
		img, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_R, subdir, f'{image_name}-mask-R.jpg', True)
	save_image(mask_G, subdir, f'{image_name}-mask-G.jpg', True)
	save_image(mask_B, subdir, f'{image_name}-mask-B.jpg', True)
	save_image(mask_combined, subdir, f'{image_name}-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(mask_combined, morphological_kernel, 'grad')
	save_image(contour_map, subdir, f'{image_name}-contour-map.jpg', True)


	texture_kernels = [9, 11, 13]
	max_itr = [1,1,1]
	flip=[False, False, False]
	texture_mask_combined, mask_1, mask_2, mask_3 = get_segmentation_using_texture(
		img, texture_kernels, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_1, subdir, f'{image_name}-texture-mask-1.jpg', True)
	save_image(mask_2, subdir, f'{image_name}-texture-mask-2.jpg', True)
	save_image(mask_3, subdir, f'{image_name}-texture-mask-3.jpg', True)
	save_image(texture_mask_combined, subdir, f'{image_name}-texture-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(texture_mask_combined, morphological_kernel, 'close')
	save_image(contour_map, subdir, f'{image_name}-texture-contour-map.jpg', True)

	# Task-1: dog
	# segmentation mask using RGB
	subdir = 'hw6'
	image_name = 'dog_small'
	# reading images...
	img = read_image(subdir, f"{image_name}.jpg")
	L = 256
	max_itr = [2, 2, 2]
	flip=[True, True, True]
	mask_combined, mask_R, mask_G, mask_B = get_segmentation_mask_using_RGB(
		img, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_R, subdir, f'{image_name}-mask-R.jpg', True)
	save_image(mask_G, subdir, f'{image_name}-mask-G.jpg', True)
	save_image(mask_B, subdir, f'{image_name}-mask-B.jpg', True)
	save_image(mask_combined, subdir, f'{image_name}-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(mask_combined, morphological_kernel, 'open')
	save_image(contour_map, subdir, f'{image_name}-contour-map.jpg', True)


	texture_kernels = [5,7, 11]
	max_itr = [3, 2, 2]
	flip=[True, True, True]
	texture_mask_combined, mask_1, mask_2, mask_3 = get_segmentation_using_texture(
		img, texture_kernels, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_1, subdir, f'{image_name}-texture-mask-1.jpg', True)
	save_image(mask_2, subdir, f'{image_name}-texture-mask-2.jpg', True)
	save_image(mask_3, subdir, f'{image_name}-texture-mask-3.jpg', True)
	save_image(texture_mask_combined, subdir, f'{image_name}-texture-mask-combined.jpg', True)

	morphological_kernel = 5
	contour_map = extract_contour(texture_mask_combined, morphological_kernel, 'open')
	save_image(contour_map, subdir, f'{image_name}-texture-contour-map.jpg', True)

	# Task-2: Fire-hydrant
	subdir = 'hw6'
	image_name = 'fire_hydrant'
	print(f"Running for figure: {image_name}")
	# reading images...
	img = read_image(subdir, f"{image_name}.jpg")
	foreground_prob = 0.15
	L = 256
	max_itr = [2,2,1]
	flip=[False, False, False]
	mask_combined, mask_R, mask_G, mask_B = get_segmentation_mask_using_RGB(
		img, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_R, subdir, f'{image_name}-mask-R.jpg', True)
	save_image(mask_G, subdir, f'{image_name}-mask-G.jpg', True)
	save_image(mask_B, subdir, f'{image_name}-mask-B.jpg', True)
	save_image(mask_combined, subdir, f'{image_name}-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(mask_combined, morphological_kernel, 'open')
	save_image(contour_map, subdir, f'{image_name}-contour-map.jpg', True)


	texture_kernels = [3,5,7]
	max_itr = [1, 1, 1]
	flip=[False, False, False]
	texture_mask_combined, mask_1, mask_2, mask_3 = get_segmentation_using_texture(
		img, texture_kernels, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_1, subdir, f'{image_name}-texture-mask-1.jpg', True)
	save_image(mask_2, subdir, f'{image_name}-texture-mask-2.jpg', True)
	save_image(mask_3, subdir, f'{image_name}-texture-mask-3.jpg', True)
	save_image(texture_mask_combined, subdir, f'{image_name}-texture-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(texture_mask_combined, morphological_kernel, 'grad')
	save_image(contour_map, subdir, f'{image_name}-texture-contour-map.jpg', True)


	# Task-2: flowers
	subdir = 'hw6'
	image_name = 'flowers'
	print(f"Running for figure: {image_name}")
	# reading images...
	img = read_image(subdir, f"{image_name}.jpg")
	foreground_prob = 0.001
	L = 256
	max_itr = [1,1,1]
	flip=[False, True, False]
	mask_combined, mask_R, mask_G, mask_B = get_segmentation_mask_using_RGB(
		img, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_R, subdir, f'{image_name}-mask-R.jpg', True)
	save_image(mask_G, subdir, f'{image_name}-mask-G.jpg', True)
	save_image(mask_B, subdir, f'{image_name}-mask-B.jpg', True)
	save_image(mask_combined, subdir, f'{image_name}-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(mask_combined, morphological_kernel, 'close')
	save_image(contour_map, subdir, f'{image_name}-contour-map.jpg', True)


	texture_kernels = [3,5,7]
	max_itr = [1, 1, 1]
	flip=[False, False, False]
	texture_mask_combined, mask_1, mask_2, mask_3 = get_segmentation_using_texture(
		img, texture_kernels, flip=flip, L=L, max_itr=max_itr
	)

	save_image(mask_1, subdir, f'{image_name}-texture-mask-1.jpg', True)
	save_image(mask_2, subdir, f'{image_name}-texture-mask-2.jpg', True)
	save_image(mask_3, subdir, f'{image_name}-texture-mask-3.jpg', True)
	save_image(texture_mask_combined, subdir, f'{image_name}-texture-mask-combined.jpg', True)

	morphological_kernel = 3
	contour_map = extract_contour(texture_mask_combined, morphological_kernel, 'close')
	save_image(contour_map, subdir, f'{image_name}-texture-contour-map.jpg', True)




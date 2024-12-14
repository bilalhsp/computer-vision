import cv2
import pathlib
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

def save_image(image, subdir, image_name):
	"""Saves image to the subdir"""
	path = pathlib.Path(computer_vision.__path__[0])
	image_path = path.parents[0] / 'images' /subdir/image_name
	# Save the image with specific quality
	img_bgr_for_saving = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(image_path, img_bgr_for_saving)
	
def harris_corner_detection(img, sigma, k):


	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img_gray = img_gray.astype(np.float32)


	M = math.ceil(4*sigma)
	if M%2 != 0:
		M += 1
		
	# getting the haar kernels 
	haar_x = np.ones((M,M))
	haar_x[:,:M//2] = -1

	haar_y = np.ones((M,M))
	haar_y[:M//2] = -1

	dx = cv2.filter2D(img_gray, -1, haar_x)
	dy = cv2.filter2D(img_gray, -1, haar_y)

	dxx = dx*dx
	dxy = dx*dy
	dyy = dy*dy
	ksize = int(5*sigma)
	sum_filter = np.ones((ksize,ksize))
	Sxx = cv2.filter2D(dxx, -1, sum_filter)
	Sxy = cv2.filter2D(dxy, -1, sum_filter)
	Syy = cv2.filter2D(dyy, -1, sum_filter)


	# Harris response ..
	det_M = Sxx*Syy - Sxy**2
	trace_M = Sxx + Syy
	R = det_M - k*(trace_M**2)

	corners, R_norm = non_maximum_suppression(R, 2*ksize)
	return corners, R_norm
	

# Non-maximum suppression...
def non_maximum_suppression(R, k):
	"""For every window of size k x k, retain only the pixel
	Args:
		R: Harris reponse
		k: window size
	"""
	top_k = 500
	R_max = np.max(R)
	R_min = np.min(R)
	R_norm = (R-R_min)/(R_max - R_min)
	R_threshold = np.mean(R_norm)
	corners = []
	for i in range(k, R_norm.shape[0] - k):
		for j in range(k, R_norm.shape[1] - k):
			window = R_norm[i-k//2: i+k//2, j-k//2:j+k//2]
			local_max = np.max(window)
			if R_norm[i,j] == local_max and R_norm[i,j] > R_threshold:
				corners.append([i, j, R_norm[i,j]])
	corners = np.array(corners)
	print(corners.shape)
	if corners.shape[0] > top_k:
		percentile = 100 - 100*top_k/corners.shape[0]
		R_values = np.sort(corners[:,2])
		threshold = np.percentile(corners[:,2], percentile)
		mask = np.where(corners[:,2] > threshold)[0]
		corners = corners[mask,:]
		print(corners.shape)
	return corners, R_norm

def add_corner_points_to_image(
		img, corners,
		color=(255,0,0),
		radius=2,
		thickness=2,
		):
	# create an copy to avoid updating the original image.
	output_image = np.copy(img)
	for corner in corners:
		# note the cv2.circle expects point as cartesian coordinates (x,y),
		# whereas our coordinates of corners are saved as (row, col) index.
		point = (int(corner[1]), int(corner[0]))
		cv2.circle(output_image, point, radius=radius, color=color,
			 thickness=thickness)
		
	return output_image
	
def correspondence_using_SSD(
		img1, img2, interest_points1, interest_points2, M=20, top_k=100):

	features1, interest_points1 = extract_features(img1, interest_points1, M)
	features2, interest_points2 = extract_features(img2, interest_points2, M)
	ssd = np.sum((features1[:,None,:] - features2[None,:,:])**2, axis=2)

	ssd_norm = (ssd - np.min(ssd))/(np.max(ssd) - np.min(ssd))
	percentile = 100*top_k/ssd_norm.size
	threshold = np.percentile(ssd_norm, percentile)
	indices = np.where(ssd_norm < threshold)

	corresponding_points = {
		'img1': interest_points1[indices[0]],
		'img2': interest_points2[indices[1]],
	}
	
	return corresponding_points

def correspondence_using_NCC(
		img1, img2, interest_points1, interest_points2, M=20, top_k=100
		):
	features1, interest_points1 = extract_features(img1, interest_points1, M)
	features2, interest_points2 = extract_features(img2, interest_points2, M)

	demean_f1 = features1 - np.mean(features1, axis=1)[:,None]
	demean_f2 = features2 - np.mean(features2, axis=1)[:,None]

	numerator = np.sum(demean_f1[:,None,:]*demean_f2[None,:,:], axis=2)
	denom = np.sqrt(np.sum(demean_f1**2, axis=-1)[:,None]*np.sum(demean_f2**2, axis=-1)[None, :])

	epsilon = 1e-6
	ncc = numerator/(denom + epsilon)
	ncc_dist = 1 - ncc
	percentile = 100*top_k/ncc_dist.size
	threshold = np.percentile(ncc_dist, percentile)
	indices = np.where(ncc_dist < threshold)
	corresponding_points = {
		'img1': interest_points1[indices[0]],
		'img2': interest_points2[indices[1]],
	}
	
	return corresponding_points

def extract_features(img, interest_points, M):
	features = []
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	valid_interest_points = []
	for point in interest_points:
		i = int(point[0])
		j = int(point[1])
		if i >= M//2 and i< img_gray.shape[0]-M//2 and \
			j >= M//2 and j< img_gray.shape[1]-M//2:
			# print(img_gray[i-M//2:i+1+M//2, j-M//2:j+1+M//2].shape) 
			features.append(img_gray[i-M//2:i+1+M//2, j-M//2:j+1+M//2].flatten())
			valid_interest_points.append(point)

	return np.array(features), np.array(valid_interest_points)

def display_corresponding_points(
		img1, img2, corresponding_points,
		thickness = 2,
		):
	img_combined = np.concatenate([img1, img2], axis=1)
	half = corresponding_points['img1'].shape[0]
	for ind, (pt1, pt2) in enumerate(
			zip(corresponding_points['img1'], corresponding_points['img2'])):
		point1 = (int(pt1[1]), int(pt1[0]))
		point2 = (int(img1.shape[1]+pt2[1]), int(pt2[0]))
		# Randomly generate RGB values (each ranging from 0 to 255)
		color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
		cv2.line(
			img_combined, point1, point2, color=color, thickness=thickness
			)
		if ind > half:
			break

	plt.imshow(img_combined)
	return img_combined


def get_kps_and_correspondences_using_harris_corner_detection(filename1, filename2, sigma=1, subdir='hw4'):

	k = 0.05
	M = 20
	top_k=100
	lw = 1

	# reading images...
	img1 = read_image(subdir, filename1)
	img2 = read_image(subdir, filename2)
	
	corners1, R_norm = harris_corner_detection(img1, sigma=sigma, k=k)
	corners2, R_norm = harris_corner_detection(img2, sigma=sigma, k=k)

	# adding interest points to the img
	img_w_kps1 = add_corner_points_to_image(img1, corners=corners1)
	img_w_kps2 = add_corner_points_to_image(img2, corners=corners2)

	save_image(img_w_kps1, subdir, f'harris_kps_sigma_{10*sigma:02.0f}_{filename1[:-4]}.png')
	save_image(img_w_kps2, subdir, f'harris_kps_sigma_{10*sigma:02.0f}_{filename2[:-4]}.png')

	# correspondence using NCC
	corresponding_points = correspondence_using_NCC(
		img1, img2, corners1, corners2, M=M, top_k=top_k
		)
	combined_img = display_corresponding_points(img1, img2, corresponding_points, thickness=lw)
	save_image(combined_img, subdir, f'harris_correspondence_using_NCC_sigma_{10*sigma:02.0f}_{filename1[:-5]}.png')

	# correspondence using SSD
	corresponding_points = correspondence_using_SSD(
		img1, img2, corners1, corners2, M=M, top_k=top_k
		)
	combined_img = display_corresponding_points(img1, img2, corresponding_points, thickness=lw)
	save_image(combined_img, subdir, f'harris_correspondence_using_SSD_sigma_{10*sigma:02.0f}_{filename1[:-5]}.png')

	

def get_kps_and_correspondences_using_SIFT(
		filename1, filename2, subdir='hw4', top_k=100
		):
    # reading images...
	img1 = read_image(subdir, filename1)
	img2 = read_image(subdir, filename2)

	img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

	sift = cv2.SIFT_create()

	kp1, des1 = sift.detectAndCompute(img1_gray, None)
	kp2, des2 = sift.detectAndCompute(img2_gray, None)

	bf = cv2.BFMatcher()
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)

	out_img = np.zeros_like(img2)
	out_img = cv2.drawMatches(
		img1, kp1, img2, kp2, matches[:top_k], out_img,
		flags=2
	)
	save_image(out_img, subdir, f'SIFT_{filename1[:-5]}.png')

	

if __name__ == "__main__":
	
	image_pairs = {
		'pair_1': ('hovde_2.jpg', 'hovde_3.jpg'),
		'pair_2': ('temple_1.jpg', 'temple_2.jpg'),
		'pair_3': ('painting_1.png', 'painting_2.png'),
		'pair_4': ('fuel_1.png', 'fuel_2.png'),
	}
	
	sigmas = [0.8, 1.2, 1.6, 2.0]
	
	for pair, (filename1, filename2) in image_pairs.items():
		print(f"Running for {pair}")
		for sigma in sigmas:
			get_kps_and_correspondences_using_harris_corner_detection(
				filename1, filename2, sigma=sigma
			)

		get_kps_and_correspondences_using_SIFT(filename1, filename2, 'hw4')
		
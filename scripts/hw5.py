import os
import pandas as pd
import cv2
import pathlib
# import skimage
import scipy
import computer_vision

import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
from functools import reduce

import torch
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
						  make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


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
	
def save_geometric_distance_output(data):
	filepath = os.path.join(os.getcwd(), 'geometric_distances.csv')
	if not os.path.exists(filepath):
		df = pd.DataFrame(columns=list(data.keys()))
	else:
		df = pd.read_csv(filepath)
	df = df.reset_index(drop=True)
	# New data as a dictionary
	new_df = pd.DataFrame([data])

	# Append new data using pd.concat
	df = pd.concat([df, new_df], ignore_index=True)

	df.to_csv(filepath, index=False)
	print(f"distances saved to {filepath}")

def save_pairwise_matches_and_compute_geometric_distances(
		hw_dir, task_id, n=5, sigma=1.2, distance_threshold = 0.5, use_SIFT=True
		):
	"""This function computes pairwise homographies and saves the
	images alongwith inlier and outlier matches displayed on top.
	"""
	if task_id ==1:
		task_offet = 0
	elif task_id ==2:
		task_offet = 10

		
	images = [read_image(subdir=hw_dir, image_name=f'{task_offet+i}.jpg', )  for i in range(1, 6)]
	for i in range(len(images)-1):
		
		estimator = HomographyEstimator(
			images[i], images[i+1], distance_threshold=distance_threshold, use_SIFT=use_SIFT)

		H, matches, geometric_distance = estimator.automatic_homography_estimation(
			n, sigma=sigma, precondition=True, use_lm=True, use_scipy=False
			)
		distances = {
			'task_id': task_id,
			'image_pairs': f"{task_offet+i+1}--{task_offet+i+2}",
			'distance_without_LM': geometric_distance[0],
			'distance_with_LM': geometric_distance[1],
		}
		save_geometric_distance_output(distances)
		
		inlier_set, outlier_set = matches
		img_with_matches = display_corresponding_points(
			images[i], images[i+1], inlier_set, outlier_set[:20], thickness=2
			)
		filename = f'combined_{task_offet+i+1}_{task_offet+i+2}.png'
		save_image(img_with_matches, 'hw5', filename)


def display_corresponding_points(
		img1, img2, inlier_set, outlier_set,
		thickness = 2,
		):
	img_combined = np.concatenate([img1, img2], axis=1)
	# overlay inliers in green...
	color = color = (0, 255, 0)
	img_combined = overlay_correspondences(img_combined, inlier_set, img1.shape[1], color=color, thickness=thickness)
	
	# overlay outliers in red...
	color = color = (255, 0, 0)
	img_combined = overlay_correspondences(img_combined, outlier_set, img1.shape[1], color=color, thickness=thickness)

	plt.imshow(img_combined)
	return img_combined

def overlay_correspondences(img_combined, corresponding_points, img1_width, color=None, thickness=2):
	"""Given the combined image and correspondences, overlays
	the correspondences on the image.
	"""
	# half = corresponding_points['img1'].shape[0]
	for ind, correspondence in enumerate(corresponding_points):
		x, y = correspondence[0]
		x_p, y_p = correspondence[1]
		point1 = (int(x), int(y))
		point2 = (int(img1_width+x_p), int(y_p))
		
		if color is None:
			# Randomly generate RGB values (each ranging from 0 to 255)
			color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
		cv2.line(
			img_combined, point1, point2, color=color, thickness=thickness
			)
	return img_combined
	


class HomographyEstimator:
	"""Implements automatic Homography estimation pipeline.
	Steps of the pipeline are;
		- key points extraction
		- Finding correspondences (matches)
		- RANSAC for homography estimation using Linear Least-Square
			and outlier rejection.
		- Homograpy refinement using Non-linear Least-Square (LM)
	
	"""
	def __init__(self, img1, img2, distance_threshold=0.75, use_SIFT=True) -> None:
		"""Creates estimator for automatic homographic estimation.
		
		Args:
			img1: domain image
			img2: range image
			distance_threshol: threshold to filter the keypoint matches,
				used for ratio test.
			use_SIFT: If True, use SIFT for extracting key points and 
				matches, otherwise use SuperPoint and SuperGlue.
		"""
		self.img1 = img1
		self.img2 = img2
		self.distance_threshold = distance_threshold
		self.use_SIFT = use_SIFT
		self.kp1, self.kp2, self.matches = self.get_key_points()
		self.total_samples = len(self.matches)

		self.all_data_points = self.get_all_correspondences()
		self.T, self.T_prime = self.get_pre_conditioning_matrices()
		

	def get_superGlue_matches(self, img1, img2):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		config = {
				'superpoint': {
					'nms_radius': 8,
					'keypoint_threshold': 0.005,
					'max_keypoints': -1
				},
				'superglue': {
					'weights': 'outdoor',
					'sinkhorn_iterations': 20,
					'match_threshold': 0.2,
				}
			}
		matching = Matching(config).eval().to(device)
		data = {
			'image0': frame2tensor(img1, device),
			'image1': frame2tensor(img2, device),
		}
		
		results = matching(data)
		kp1, des1 = results['keypoints0'][0].numpy(), results['descriptors0'][0].numpy().T
		kp2, des2 = results['keypoints1'][0].numpy(), results['descriptors1'][0].numpy().T
		return kp1, kp2, des1, des2
		
	

	def get_key_points(self):
		img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
		img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)

		if self.use_SIFT:
			sift = cv2.SIFT_create()
			kp1, des1 = sift.detectAndCompute(img1_gray, None)
			kp2, des2 = sift.detectAndCompute(img2_gray, None)
		else:
			kp1, kp2, des1, des2 = self.get_superGlue_matches(img1_gray, img2_gray)
			

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1, des2, k=2)

		# Apply ratio test
		retained_matches = []
		for m, n in matches:
			if m.distance < self.distance_threshold * n.distance:  # Ratio threshold, can tweak this value
				retained_matches.append(m)

		return kp1, kp2, retained_matches
	
	def get_all_correspondences(self):
		"""Returns all point correspondences.
		
		Returns:
			(N, 2, 2): (num_corres, point, coordinates)

		"""
		pt_correspondences = []
		for id in range(self.total_samples):
			corresp = self.matches[id] 
			x = self.kp1[corresp.queryIdx]
			x_prime = self.kp2[corresp.trainIdx]
			if self.use_SIFT:
				x = x.pt
				x_prime = x_prime.pt

			pt_correspondences.append((x, x_prime))
		pt_correspondences  = np.array(pt_correspondences)
		return pt_correspondences
	

	def sample_data_points(self, n):
		"""Randomly select 'n' samples from the full data points."""
		random_ids = np.random.randint(0, self.total_samples, n)
		return self.all_data_points[random_ids]



	def get_pre_conditioning_matrices(self):
		"""Isotropic scalling to pre-condition points."""
		img1_h, img1_w, _ = self.img1.shape
		pixels_x, pixels_y = np.meshgrid(np.arange(img1_w), np.arange(img1_h))
		dom_pixels = np.stack([pixels_x.flatten(), pixels_y.flatten()], axis=1)

		T = HomographyEstimator.compute_pre_condtioning_transform(dom_pixels)

		img2_h, img2_w, _ = self.img2.shape
		pixels_x, pixels_y = np.meshgrid(np.arange(img2_w), np.arange(img2_h))
		range_pixels = np.stack([pixels_x.flatten(), pixels_y.flatten()], axis=1)

		T_prime = HomographyEstimator.compute_pre_condtioning_transform(range_pixels)
		return T, T_prime


	@staticmethod
	def compute_pre_condtioning_transform(points_2d):
		# compute centroid
		centroid = np.mean(points_2d, axis=0)
		# avg distance of point from centroid
		d = np.sqrt(np.sum((points_2d - centroid[None, :])**2, axis=1))
		d_avg = np.mean(d)

		# scale factor to set avg distance to sqrt(2)
		s = np.sqrt(2) / d_avg

		# pre-conditioning matrix
		T = np.eye(3)
		T[0,0] = s
		T[1,1] = s
		T[:2, -1] = -s*centroid
		return T


	@staticmethod
	def pre_condition_points(points_2d, T):
		"""Isotropic scalling to pre-condition points."""

		points_hc = np.concat([points_2d, np.ones(points_2d.shape[0])[:,None]], axis=1)
		points_norm = np.matmul(T, points_hc.T)
		return points_norm.T
	
	@staticmethod
	def get_coefficients_of_linear_equations(x, x_prime):
		"""Return the coefficients of linear equations, 
		for the given correspondence.

		Args:
			x = (3,)
			x_prime = (3,)
		"""
		return np.array([[0, 0, 0, -x_prime[2]*x[0], -x_prime[2]*x[1], -x_prime[2]*x[2], x_prime[1]*x[0], x_prime[1]*x[1], x_prime[1]*x[2]],
				[x_prime[2]*x[0], x_prime[2]*x[1], x_prime[2]*x[2], 0, 0, 0, -x_prime[0]*x[0], -x_prime[0]*x[1], -x_prime[0]*x[2]]])
	
	def get_coeffient_matrix(self, dom_pts, range_pts):
		"""For the given data points, get the coefficients matrix
		A for homogeneous system of equations.

		Args:
			dom_pts: (N, 3)
			range_pts: (N, 3)
		"""
		A = []
		for dom_pt, range_pt in zip(dom_pts, range_pts):

			A_i = HomographyEstimator.get_coefficients_of_linear_equations(dom_pt, range_pt)
			A.append(A_i)
		return np.concat(A, axis=0)
	
	@staticmethod
	def lst_square(A):
		"""Returns linear least square solution to homogeneous
		system of linear equations, Ah = 0.
		Since Rank(A) = n-1, solution is the eigen vector
		corresponding to the smallest singular value of A
		i.e. last row of vh.
		"""
		u, s, vh = np.linalg.svd(A)
		return vh[-1]
		
	
	def fit(self, data_points, precondition=False):
		"""For the given data points, compute homography matrix
		using linear least square fit.

		Args:
			data_points: (n, 2, 2) = n correspondences, 
				[:,0,:] domain pts, [:,1,:] range pts
		"""
		# print(data_points.shape)
		dom_pts = data_points[:,0]
		range_pts = data_points[:,1]
		if precondition:
			dom_pts = HomographyEstimator.pre_condition_points(dom_pts, self.T)
			range_pts = HomographyEstimator.pre_condition_points(range_pts, self.T_prime)
			# A = self.get_coeffient_matrix(dom_pts_norm, range_pts_norm)
		else:   
			dom_pts = np.concat([dom_pts, np.ones(dom_pts.shape[0])[:, None]], axis=1)
			range_pts = np.concat([range_pts, np.ones(range_pts.shape[0])[:, None]], axis=1)

		A = self.get_coeffient_matrix(dom_pts, range_pts)
		h = HomographyEstimator.lst_square(A)

		h /= h[-1]
		H = h.reshape(3,3)
		if precondition:
			H_tilde = np.matmul(H, self.T)
			H = np.matmul(np.linalg.pinv(self.T_prime), H_tilde)
			H /= H[-1,-1]
			
		return H
	
	@staticmethod
	def apply_homography(H, physical_pts):
		"""Apply homography matrix H, to physical points.
		Args:   
			H= (3 x 3) homography matrix
			physical_pts= (N, 2) physical coordinates
		"""
		hc_pts = np.concat([physical_pts, np.ones(physical_pts.shape[0])[:, None]], axis=1).transpose()
		hc_pts_prime = np.matmul(H, hc_pts)
		hc_pts_prime /= hc_pts_prime[-1]
		return hc_pts_prime[:2].transpose()
	
	def get_inlier_set(self, H, inlier_threshold):
		"""Returns the inlier set for the given homography H,
		and threshold.

		Returns:
			inlier_set:
			agg_distance: total distance for the inlier set. 
		"""
		
		dom_pts = self.all_data_points[:,0]
		range_pts = self.all_data_points[:,1]

		# deviation in range space..
		pred_range_pts = HomographyEstimator.apply_homography(H, dom_pts)
		d1 = np.sum((range_pts - pred_range_pts)**2, axis=1)
		
		# deviation in dom space..
		H_inv = np.linalg.pinv(H)
		pred_dom_pts = HomographyEstimator.apply_homography(H_inv, range_pts)
		d2 = np.sum((dom_pts - pred_dom_pts)**2, axis=1)

		dev = np.sqrt(d1 + d2)
		inlier_mask = np.where(dev < inlier_threshold)[0]
		# print(f"mask shape: {inlier_mask.shape}")
		 
		inlier_set = self.all_data_points[inlier_mask]
		outlier_set = self.all_data_points[~inlier_mask]
		agg_distance = np.sum(dev[inlier_mask])
		return inlier_set, agg_distance, outlier_set


	@staticmethod
	def cost_fn(h, point_correspondences):
		"""computes the cost for the given point correspondences
		and parameter h.
		"""
		residuals = HomographyEstimator.compute_residuals(h, point_correspondences)
		return np.sum(residuals**2)

	
	@staticmethod
	def compute_residuals(h, point_correspondences):
		"""computes the residuals for the given point correspondences
		and parameter h.
		"""
		residuals = []
		for corresp in point_correspondences:
			x = corresp[0]
			x_prime = corresp[1]
			denom = h[6]*x[0] + h[7]*x[1] + 1

			res_0 = x_prime[0] - (h[0]*x[0] + h[1]*x[1] + h[2])/denom
			res_1 = x_prime[1] - (h[3]*x[0] + h[4]*x[1] + h[5])/denom
			residuals.append(res_0)
			residuals.append(res_1)
		return np.array(residuals)

	@staticmethod
	def jacobian_f(h, point_correspondences):
		"""computes the cost for the given point correspondences
		and parameter h.
		"""
		jacobians = []
		for corresp in point_correspondences:
			x = corresp[0]
			numer1 =  h[0]*x[0] + h[1]*x[1] + h[2]
			numer2 = h[3]*x[0] + h[4]*x[1] + h[5]
			denom = h[6]*x[0] + h[7]*x[1] + 1
			jac_f1 = [x[0]/denom, x[1]/denom, 1/denom, 0, 0, 0, -x[0]*numer1/(denom**2), -x[1]*numer1/(denom**2)]
			jac_f2 = [0, 0, 0, x[0]/denom, x[1]/denom, 1/denom, -x[0]*numer2/(denom**2), -x[1]*numer2/(denom**2)]
			jacobians.append(jac_f1)
			jacobians.append(jac_f2)
		return np.array(jacobians)

	def RANSAC(self, n, sigma, p=0.99, precondition=True):
		"""
		
		Args:
			n = number of data points to fil the estimate.
			sigma = std. deviation of inlier noise
			epsilon: = probability of outlier.
			p = probability that at least one trial will be free of outliers.
		"""
		epsilon = 0.90  # starting with worst-case guess..
		n_total = self.total_samples
		N = np.log(1 - p)/np.log(1 - (1- epsilon)**n)
		M = (1 - epsilon)*n_total
		inlier_threshold = 3*sigma
		print(f"Total samples: {n_total}")
		print(f"Acceptable size of inlier set: {M}")
		print(f"Number of iterations: {N}")
		print(f"Inlier threshold: {inlier_threshold:.2f}")

		bestErr = None
		bestInSet = None
		sizeOfInset = 0
		outSet = None
		bestFit = None
		itr_count = 0
		while N > itr_count:
			data_points = self.sample_data_points(n)
			model = self.fit(data_points, precondition=precondition)
			inlier_set, modelErr, outlier_set = self.get_inlier_set(model, inlier_threshold)
			print(f"itr: {itr_count}, inliet set: {inlier_set.shape[0]}, modelErr: {modelErr}")
			# print(f"itr: {i}, modelErr: {modelErr}")
			if inlier_set.shape[0] > sizeOfInset:
				# bigger inset is found..
				bestFit = model
				bestErr = modelErr
				bestInSet = inlier_set
				outSet = outlier_set
				sizeOfInset = inlier_set.shape[0]
		
				# update estimate of iterations...
				epsilon = 1 - inlier_set.shape[0]/n_total
				N = np.log(1 - p)/np.log(1 - (1- epsilon)**n)
				M = (1 - epsilon)*n_total
				print(f"epsilon: {epsilon:.2f}")
				print(f"Estimate of required iterations: {N}")
			elif bestErr is not None and inlier_set.shape[0] == sizeOfInset and modelErr < bestErr:
				# better inset of the same size is found..
				bestFit = model
				bestErr = modelErr
				bestInSet = inlier_set
				outSet = outlier_set
					
			itr_count += 1

		if bestFit is not None:
			# Now fit using the entire inlier_set, to refine the bestFit
			bestFit = self.fit(bestInSet, precondition=precondition)
			bestInSet, bestErr, outSet = self.get_inlier_set(bestFit, inlier_threshold)
			bestFit = self.fit(bestInSet, precondition=precondition)
		return bestFit, bestInSet, outSet, bestErr
	
	@staticmethod
	def get_refined_homography(H, inlier_set, use_scipy=False, t=0.01, convergence_th=0.01, max_itr=50):
		"""Refine homography estimate using non-linear least squares
		for the correspondences from the inlier set.
		
		Args:
			H: starting H, (initial guess)
			inlier_set: set of (noisy) correspondences
		"""
		if use_scipy:
			h = scipy.optimize.least_squares(
					HomographyEstimator.compute_residuals, H.reshape(-1)[:-1], method='lm', args=(inlier_set, )
					).x
		else:
			h = HomographyEstimator.optimize_using_LM(
				H.reshape(-1)[:-1], inlier_set, t=t, convergence_th=convergence_th, 
				max_itr=max_itr
				)
		H = np.concat([h, np.ones(1)]).reshape(3,3)
		return H
	
	@staticmethod
	def optimize_using_LM(h, inlier_set, t=0.01, convergence_th=0.01, max_itr=50):
		
		J = HomographyEstimator.jacobian_f(h, inlier_set)
		mu = t*max(np.diag(np.matmul(J.transpose(), J)))

		for itr in range(max_itr):
			J = HomographyEstimator.jacobian_f(h, inlier_set)
			E = HomographyEstimator.compute_residuals(h, inlier_set)

			C_p = HomographyEstimator.cost_fn(h, inlier_set)
			delta_h = np.linalg.pinv(J.T@J + mu*np.eye(h.size))@J.T @ E
			C_p1 = HomographyEstimator.cost_fn(h + delta_h, inlier_set)

			rho = (C_p - C_p1) / (delta_h.T @J.T @ E +  delta_h.T @ (mu*np.eye(h.size)) @ delta_h)
			# mu update
			mu = mu*max(1/3, 1- (2*rho -1)**3)
			h = h + delta_h
			if np.linalg.norm(delta_h, 1) < convergence_th:
				break

		print(f"H converged in {itr+1} steps")
		return h

	def automatic_homography_estimation(
			self, n, sigma, precondition=True, use_lm=True, use_scipy=False, convergence_th=0.01):
		"""Automatically estimates homography"""
	
		p = 0.99
		H, inlier_set, outlier_set, model_error = self.RANSAC(n, sigma, p=p, precondition=precondition)
		
		initial_geometric_distance = HomographyEstimator.cost_fn(H.reshape(-1)[:-1], inlier_set)
		if use_lm:
			H = HomographyEstimator.get_refined_homography(
				H, inlier_set, use_scipy=use_scipy, convergence_th=convergence_th
				)
		improved_geometric_distance = HomographyEstimator.cost_fn(H.reshape(-1)[:-1], inlier_set)

		return H, (inlier_set, outlier_set), (initial_geometric_distance, improved_geometric_distance)


class Panorama:
	def __init__(
			self, images, homography_config, ref_img_index=None, max_size=4000,
			) -> None:
		self.images = images
		self.num_images = len(images)
		if ref_img_index is None:
			ref_img_index = len(images)//2
		self.ref_index = ref_img_index

		self.max_size = max_size
		self.homography_config = homography_config


	def get_combined_H_ref(self):

		H_mats_pairwise = self.compute_pairwise_homographies()
		H_mats_ref = []
		for i, img in enumerate(self.images):
			H = self.get_H_to_ref_frame(i, H_mats_pairwise)
			H_mats_ref.append(H)

		return H_mats_ref


	def get_H_to_ref_frame(self, index, homography_matrices):
		"""
		Computes a homography matrix to transform an image at `index` to the `ref_frame_index`.

		Args:
			index (int): The index of the image.
			ref_frame_index (int): The reference frame index.
			homography_matrices (list of np.ndarray): Pairwise homographies between consecutive images.

		Returns:
			np.ndarray: The combined homography matrix to map the image at `index` to the reference frame.
		"""
		H_list = []
		ref_index = self.ref_index
		while index != ref_index:
			if index < ref_index:
				H_list = [homography_matrices[index]] + H_list
				index += 1 
			elif index > ref_index:
				H_list.append(np.linalg.pinv(homography_matrices[index-1]))
				index -= 1
		if len(H_list) > 0:
			combined_H = reduce(np.matmul, H_list)
		else:
			combined_H = np.eye(3)
		return combined_H
	
	def compute_pairwise_homographies(self):
		"""
		Computes pairwise homography matrices between consecutive images.


		Returns:
			list of np.ndarray: Pairwise homography matrices between consecutive images.
		"""
		n = self.homography_config.pop('n', 5)
		sigma = self.homography_config.pop('sigma', 2)
		distance_threshold = self.homography_config.pop('distance_threshold', 0.75)
		use_SIFT = self.homography_config.pop('use_SIFT', True)
		precondition = self.homography_config.pop('precondition', True)
		use_lm = self.homography_config.pop('use_lm', True)
		use_scipy = self.homography_config.pop('use_scipy', False)

		H_mats_pairwise = []
 
		for i in range(self.num_images-1):

			estimator = HomographyEstimator(
				self.images[i], self.images[i+1], distance_threshold=distance_threshold,
				use_SIFT=use_SIFT
				)
			H, matches, geometric_distance = estimator.automatic_homography_estimation(
				n, sigma=sigma, precondition=precondition, use_lm=use_lm,
				use_scipy=use_scipy, 

				
				)
			
			H_mats_pairwise.append(H)
		return H_mats_pairwise
	
	@staticmethod
	def get_corner_points(H, img):
		"""Get the corners of the transformed img."""
		h,w,_ = img.shape
		corners = np.array([[0,0], [w, 0], [w, h], [0, h]])
		corners_hc = np.concat([corners, np.ones(corners.shape[0])[:,None]], axis=1)

		corners_tr = np.matmul(H, corners_hc.T)
		corners_tr /= corners_tr[-1]

		x_min, y_min = np.min(corners_tr[:-1], axis=1)
		x_max, y_max = np.max(corners_tr[:-1], axis=1)
		return x_min, y_min, x_max, y_max
	

	def get_canvas_dims(self, H_mats):

		ref_img = self.images[self.ref_index]
		ref_h, ref_w, _ = ref_img.shape
		canvas_x_min, canvas_x_max, canvas_y_min, canvas_y_max = 0, ref_w, 0, ref_h   
		for i, img in enumerate(self.images):
			x_min, y_min, x_max, y_max= Panorama.get_corner_points(H_mats[i], img=img)

			if x_min < canvas_x_min:
				canvas_x_min = x_min
			if y_min < canvas_y_min:
				canvas_y_min = y_min
			if x_max > canvas_x_max:
				canvas_x_max = x_max
			if y_max > canvas_y_max:
				canvas_y_max = y_max
			
		canvas_h =  canvas_y_max - canvas_y_min
		canvas_w = canvas_x_max - canvas_x_min

		# adjust origin for unequal streching on both sides of origin...
		offset_x = (abs(canvas_x_max) - abs(canvas_x_min))/2
		offset_y = (abs(canvas_y_max) - abs(canvas_y_min))/2
		origin_x = canvas_w/2 - offset_x
		origin_y = canvas_h/2 - offset_y

		scale_factor = max(max(canvas_h/self.max_size, canvas_w/self.max_size), 1)
		canvas_h /= scale_factor
		canvas_w /= scale_factor
		if scale_factor > 1:
			print(f"Output canvas dim adjusted to have max dim of {self.max_size}")
		return (int(canvas_h), int(canvas_w)), (int(origin_x), int(origin_y)), scale_factor
	

	def get_panorama_view(self):


		H_mats_ref = self.get_combined_H_ref()
		canvas_dims, origin, scale_factor = self.get_canvas_dims(H_mats_ref)

		canvas = np.zeros((*canvas_dims, 3), dtype=np.uint)

		for i, img in enumerate(self.images):
			if i != self.ref_index:
				canvas = self.project_img_onto_canvas(H_mats_ref[i], img, canvas, origin, scale_factor)
		i = self.ref_index
		canvas = self.project_img_onto_canvas(H_mats_ref[i], self.images[i], canvas, origin, scale_factor)
		return canvas.astype(np.uint8)
	
	def project_img_onto_canvas(self, H, img, canvas, origin, scale_factor):
		"""
		Projects an image onto a canvas using a homography matrix with bilinear interpolation.

		Args:
			H (np.ndarray): Homography matrix.
			img (np.ndarray): Image to project.
			canvas (np.ndarray): Canvas to project onto.
			origin (tuple): Origin coordinates for placement on the canvas.
			scale_factor (float): Scaling factor for transformation.

		Returns:
			None. The canvas is modified in-place.
		"""
		# Adjust homography matrix
		H = Panorama.adjust_H_for_output_size(H, origin, scale_factor)
		canvas_h, canvas_w, _ = canvas.shape
		# img_height, img_width, _ = img.shape

		# Inverse homography
		H_inv = np.linalg.pinv(H)

		# Create homogeneous coordinates for canvas pixels
		pixel_y, pixel_x = np.meshgrid(range(canvas_h), range(canvas_w))
		hc_coordinates = np.stack([pixel_x.flatten(), pixel_y.flatten(), np.ones_like(pixel_x).flatten()])
		
		
		# Transform canvas coordinates to image space
		transformed_coordinates = np.matmul(H_inv, hc_coordinates)
		transformed_coordinates /= transformed_coordinates[-1]
		
		img_values, mask_coordinates = Panorama.bilinear_interpolation(img, transformed_coordinates[:-1])
		# Project onto canvas
		canvas[pixel_y.flatten()[mask_coordinates], pixel_x.flatten()[mask_coordinates]] = img_values[mask_coordinates]
		return canvas.astype(np.uint8)
	
	@staticmethod
	def bilinear_interpolation(img, coordinates):
		"""For the given img and coordinates, uses bilinear interpolation
		and returns img values for the input coordinates.
		Args:
			transformed_coordinates = (2, n)
		"""
		img_height, img_width, _ = img.shape
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
		return img_values, mask_coordinates



	
	@staticmethod
	def adjust_H_for_output_size(H, origin, scale_factor):
		"""Adjust the homography matrix to set the desired output size 
		and origin.

		Args:
			H: Homography matrix H
			origin: desired origin (in the projected output size space).
			scale_factor: specifies adjusted to proj size.
		"""
		T = np.eye(3)
		T[:2, -1] = origin[0], origin[1]
		H = np.matmul(T, H)
		
		H_resize = np.eye(3)
		H_resize[0,0] = 1/scale_factor
		H_resize[1,1] = 1/scale_factor
		H = np.matmul(H_resize, H)
		return H
	




if __name__ == '__main__':
	
	geometric_distances_file = os.path.join(os.getcwd(), 'geometric_distances.csv')
	if os.path.exists(geometric_distances_file):
		os.remove(geometric_distances_file)
	
	# Config for mosaic image.
	homography_config = {
		'n': 5,
		'sigma': 1.2,
		'distance_threshold': 0.75,
		'use_SIFT': True,
		'precondition': True,
		'use_lm': True,
		'use_scipy': False
	}
	max_size = 6000

	for task_id in [1,2]:
		# Task-1
		# task_id = 1
		hw_dir = 'hw5'
		n=5
		sigma=1.2
		distance_threshold = 0.5
		use_SIFT=True
		save_pairwise_matches_and_compute_geometric_distances(
			hw_dir=hw_dir, task_id=task_id, n=n, sigma=sigma,
			distance_threshold=distance_threshold,
			use_SIFT=use_SIFT 
		)

		if task_id ==1:
			task_offet = 0
		elif task_id ==2:
			task_offet = 10
		images = [read_image(subdir=hw_dir, image_name=f'{task_offet+i}.jpg', )  for i in range(1, 6)]

		homography_config['use_lm'] = True
		panorama = Panorama(images, homography_config=homography_config, max_size=max_size)
		img_mosaic = panorama.get_panorama_view()

		filename = f'mosaic_image_task{task_id}_using_lm.png'
		save_image(img_mosaic, 'hw5', filename)

		homography_config['use_lm'] = False
		panorama = Panorama(images, homography_config=homography_config, max_size=max_size)
		img_mosaic = panorama.get_panorama_view()

		filename = f'mosaic_image_task{task_id}_without_lm.png'
		save_image(img_mosaic, 'hw5', filename)
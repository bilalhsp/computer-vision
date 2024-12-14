import cv2
import pathlib
# import skimage
import sklearn.cluster
import scipy
import computer_vision
import BitVector

import matplotlib.pylab as plt
import matplotlib as mpl
import plotly.graph_objects as go
import numpy as np
import math

import os
import glob
import time
import torch
from models.matching import Matching
from operator import itemgetter
from models.utils import (AverageTimer, VideoStreamer,
						  make_matching_plot_fast, frame2tensor)


from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

torch.set_grad_enabled(False)
import matplotlib

def read_image_from_path(image_path):
		"""Reads images from the subdir"""
		img_bgr = cv2.imread(image_path)
		# Convert the image from BGR to RGB format
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		return img_rgb


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


class ClusterManager:
	def __init__(self, num_horz_lines=10, num_vert_lines = 8) -> None:
		self.num_horz_lines = num_horz_lines
		self.num_vert_lines = num_vert_lines

	def euclidean_distance(self, line1, line2):
		"""Calculates difference between theta values"""
		return np.sqrt(np.sum((line1 - line2)**2))

	
	def get_clustered_lines(self, lines):
		"""Given all the line parameters, applies clustering separately 
		for horizontal and vertical lines to return the parameters of useful lines."""
		horz_lines_mask = (lines[:,1] > (np.pi/4)) & (lines[:,1] < (3*np.pi/4))

		horz_lines = lines[horz_lines_mask]
		vert_lines = lines[~horz_lines_mask]

		horz_lines = self.get_cartesian_coordinates(horz_lines)
		vert_lines = self.get_cartesian_coordinates(vert_lines)

		horz_useful_lines = self.cluster_adaptively(
			horz_lines, self.num_horz_lines, self.euclidean_distance
			)
		vert_useful_lines = self.cluster_adaptively(
			vert_lines, self.num_vert_lines, self.euclidean_distance
			)
		
		horz_useful_lines = self.get_polar_coordinates(horz_useful_lines)
		vert_useful_lines = self.get_polar_coordinates(vert_useful_lines)

		return horz_useful_lines, vert_useful_lines
	

	def cluster_adaptively(self, lines, n_clusters, distance_metric, num_itr=100, initial_eps=50):
		"""Adaptively picks the threshold to cluster the data points into n_clusters."""
		num = 1
		eps = initial_eps
		if distance_metric is None:
			distance_metric = self.rho_distance
		while num_itr>0 and num != n_clusters:
			num_itr -= 1
			clustered_lines = self.density_based_clustering(lines, eps=eps, distance_metric=distance_metric)

			num = clustered_lines.shape[0]
			if num < n_clusters:
				eps = eps*0.5
			elif num > n_clusters:
				eps = eps*1.5
		print(f"eps: {eps}")
		return clustered_lines
	
	def density_based_clustering(self, lines, eps=50, distance_metric=None):
		"""Density based clustering using the threshold eps"""
		if distance_metric is None:
			distance_metric = self.rho_distance
		
		cluster_obj = sklearn.cluster.DBSCAN(eps=eps, min_samples=1, metric=distance_metric).fit(lines)
		clustered_lines = []
		for label in np.unique(cluster_obj.labels_):
			if label != -1:
				cluster_lines = lines[np.where(cluster_obj.labels_==label)[0]]
				clustered_lines.append(np.median(cluster_lines, axis=0))

		return np.array(clustered_lines)
	
	
	def get_cartesian_coordinates(self, lines):
		"""Given polar coordinates, returns cartesian coordinates"""
		rhos = lines[:,0]
		angles = lines[:,1]
		x = rhos*np.cos(angles)
		y = rhos*np.sin(angles)
		return np.stack([x,y], axis=1)
	
	def get_polar_coordinates(self, lines):
		"""Given cartesian coordinates, returns polar coordinates"""
		x_coord = lines[:,0]
		y_coord = lines[:,1]
		
		angles = np.arctan2(y_coord, x_coord)
		rhos = np.sqrt(x_coord**2 + y_coord**2)

		mask = (angles < 0)
		angles[mask] = angles[mask] + np.pi
		rhos[mask] *= -1
		
		return np.stack([rhos, angles], axis=1)
	

class CornersDetector:
	def __init__(self, voting_th=40, canny_th1=200, canny_th2=500,
			min_rho_diff = 10, min_theta_diff=2, output_dir=None) -> None:
		# self.data_dir = data_dir
		self.canny_th1 = canny_th1
		self.canny_th2 = canny_th2
		self.voting_th = voting_th

		self.min_rho_diff = min_rho_diff
		self.min_theta_diff = min_theta_diff

		self.cluster_manager = ClusterManager()
		self.plotter = None
		if output_dir is not None:
			self.plotter = Plotter(output_dir)

	def get_corner_pts(self, image):
		"""Returns coordinates of corners points of the calibation pattern."""
		pts_horz_lines, pts_vert_lines = self.get_Hough_lines(image)
		pts = CornersDetector.get_intersection_pts(pts_horz_lines, pts_vert_lines)
		return pts

	def visualize_Hough_lines_extraction(self, image, image_name, subdir=None):
		""" """
		assert self.plotter is not None, "plotter object not availble, make sure initialize contructor with output dir"
		img1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		mask = otsu_single_channel(
			img1_gray, L = 256, max_itr=1, flip=True
			)
		mask = erosion(mask)
		mask = dilation(mask)
		mask = 255 - mask

		self.plotter.save_visualization(mask, image_name=image_name, identifier='mask', sub_dir=subdir)
		edges = cv2.Canny(mask, self.canny_th1, self.canny_th2, apertureSize=7, L2gradient=True)

		self.plotter.save_visualization(edges, image_name=image_name, identifier='edges', sub_dir=subdir)
		lines = cv2.HoughLines(edges, 1, 0.1*np.pi / 180, self.voting_th, None, 0, 0).squeeze()

		pts_on_lines = CornersDetector.get_pts_on_lines(lines)
		self.plotter.display_img_with_lines(
			image, pts=pts_on_lines , image_name=image_name, identifier='all-lines',
			sub_dir=subdir
			)

		horz_lines, vert_lines = self.cluster_manager.get_clustered_lines(lines)

		horz_lines = np.array(sorted(horz_lines, key=lambda line: line[0]*np.sin(line[1])))
		vert_lines = np.array(sorted(vert_lines, key=lambda line: line[0]*np.cos(line[1])))
		pts_horz_lines = CornersDetector.get_pts_on_lines(horz_lines)
		pts_vert_lines = CornersDetector.get_pts_on_lines(vert_lines)
		self.plotter.display_img_with_lines(
			image, pts=pts_horz_lines, image_name=image_name, identifier='horz-lines',
			sub_dir=subdir
			)
		self.plotter.display_img_with_lines(
			image, pts=pts_vert_lines, image_name=image_name, identifier='vert-lines',
			sub_dir=subdir
			)
		
		pts = CornersDetector.get_intersection_pts(pts_horz_lines, pts_vert_lines)
		self.plotter.display_img_with_pts(
			image, pts=pts, image_name=image_name, identifier='corners',
			sub_dir=subdir, label_pts=True
			)
	
		
	def get_Hough_lines(self, image):
		""" """
		img1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		mask = otsu_single_channel(
			img1_gray, L = 256, max_itr=1, flip=True
			)
		mask = erosion(mask)
		mask = dilation(mask)

		mask = 255 - mask
		edges = cv2.Canny(mask, self.canny_th1, self.canny_th2, apertureSize=7, L2gradient=True)

		lines = cv2.HoughLines(edges, 1, 0.1*np.pi / 180, self.voting_th, None, 0, 0).squeeze()

		# return lines

		print(f"Number of lines detected: {lines.shape[0]}")

		horz_lines, vert_lines = self.cluster_manager.get_clustered_lines(lines)

		horz_lines = np.array(sorted(horz_lines, key=lambda line: line[0]*np.sin(line[1])))
		vert_lines = np.array(sorted(vert_lines, key=lambda line: line[0]*np.cos(line[1])))
		pts_horz_lines = CornersDetector.get_pts_on_lines(horz_lines)
		pts_vert_lines = CornersDetector.get_pts_on_lines(vert_lines)

		return pts_horz_lines, pts_vert_lines

		

	def filter_duplicate_lines(self, lines):
		filtered_lines = []
		min_rho_diff = 10    
		min_theta_diff = 2*np.pi / 180  

		for i in range(len(lines)):
			rho, theta = lines[i]
			duplicate = False
			for rho2, theta2 in filtered_lines:
				circular_theta_diff = min(abs(theta - theta2), 2*np.pi - abs(theta - theta2))
				if abs(rho - rho2) < min_rho_diff and circular_theta_diff < min_theta_diff:
					duplicate = True
					break
			if not duplicate:
				filtered_lines.append((rho, theta))
		return np.stack(filtered_lines)
	
	

	@staticmethod
	def get_intersection_pts(horz_pts, vert_pts):

		horz_lines = CornersDetector.get_homogenous_lines(horz_pts)
		vert_lines = CornersDetector.get_homogenous_lines(vert_pts)

		pts_hc = np.cross(horz_lines[:,None,:], vert_lines[None,:,:])
		pts_hc = pts_hc.reshape(-1, pts_hc.shape[-1])
		pts_hc /= pts_hc[:,-1][:,None]
		return pts_hc[:,:-1]


	@staticmethod
	def get_homogenous_lines(pts):
		"""Given coordinates of pts, returns homogenous coordinates of lines"""
		pts_1 = np.concatenate([pts[:,:2], np.ones(pts.shape[0])[:,None]], axis=1)
		pts_2 = np.concatenate([pts[:,2:], np.ones(pts.shape[0])[:,None]], axis=1)
		lines = np.cross(pts_1, pts_2)
		return lines

	@staticmethod
	def get_pts_on_lines(lines):
		"""Given coordinates of lines as rho, theta coordinates.
		returns two points on the lines.
		"""
		rhos = lines[:,0]
		thetas = lines[:,1]

		x_0s = rhos*np.cos(thetas)
		y_0s = rhos*np.sin(thetas)

		x1s = x_0s + 1000*(np.sin(-thetas))
		y1s = y_0s + 1000*(np.cos(thetas))
		x2s = x_0s - 1000*(np.sin(-thetas))
		y2s = y_0s - 1000*(np.cos(thetas))

		pts = np.stack([x1s, y1s, x2s, y2s], axis=-1)
		return pts
	

class HomographyEstimator:
	"""Implements Homography estimation.	
	"""
	def __init__(self) -> None:
		"""Creates estimator for homographic estimation.
		"""
		pass

	@staticmethod
	def compute_pre_condtioning_transform(points_2d):

		centroid = np.mean(points_2d, axis=0)
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
		
	
	def get_homography(self, dom_pts, range_pts, precondition=False):
		"""Compute homography matrix
		using linear least square fit.
		"""
		if precondition:
			T = HomographyEstimator.compute_pre_condtioning_transform(dom_pts)
			T_prime = HomographyEstimator.compute_pre_condtioning_transform(range_pts)
			dom_pts = HomographyEstimator.pre_condition_points(dom_pts, T)
			range_pts = HomographyEstimator.pre_condition_points(range_pts, T_prime)
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


class Calibrator:
	def __init__(self, images, cell_length = 2.96, output_dir=None) -> None:
		"""Camera calibration objects
		
		Args:
			images: list of open cv image objects
			cell_length: length of each cell on calibration pattern
		"""
		# cell_length = 2.96 # cm or (1.1667 in)
		x_coord = np.arange(8)
		y_coord = np.arange(10)
		grid = np.meshgrid(x_coord, y_coord)
		self.physical_corners = cell_length*np.stack([grid[0].flatten(), grid[1].flatten()], axis=1)
		self.homograpy_estimator = HomographyEstimator()
		self.detector = CornersDetector(60, 100, 400, output_dir=output_dir)

		self.images = images
		self.num_images = len(images)
		self.homographies = []
		self.img_corner_pts = []
		for image in images:
			img_pts = self.detector.get_corner_pts(image)
			H = self.homograpy_estimator.get_homography(self.physical_corners, img_pts)
			self.homographies.append(H)
			self.img_corner_pts.append(img_pts)
	

	@staticmethod
	def get_coefficient_vector(h1, h2):
		"""Return the coefficients of linear equations, 
		for the given .

		Args:
			h1: column of homography matrix
			h2: column of homography matrix
		"""
		return np.array([h1[0]*h2[0], h1[0]*h2[1]+h1[1]*h2[0], h1[1]*h2[1],
					h1[2]*h2[0]+h1[0]*h2[2], h1[2]*h2[1]+h1[1]*h2[2], h1[2]*h2[2]])
	
	@staticmethod
	def get_coefficient_for_Homography(H):
		"""Return the matrix of coefficients for Vb=0 equation.

		Args:
			H: Homography matrix    
		"""
		h1 = H[:,0]
		h2 = H[:,1]
		v12 = Calibrator.get_coefficient_vector(h1,h2)
		v11 = Calibrator.get_coefficient_vector(h1,h1)
		v22 = Calibrator.get_coefficient_vector(h2,h2)
		V = np.array([v12,v11 - v22])
		return V
	
	def get_coeffient_matrix(self):
		"""For the given data points, get the coefficients matrix
		A for homogeneous system of equations.

		Args:
			dom_pts: (N, 3)
			range_pts: (N, 3)
		"""
		V = []
		for H in self.homographies:
			V_i = Calibrator.get_coefficient_for_Homography(H)
			V.append(V_i)

		return np.concat(V, axis=0)
	
	def calculate_intrinsic_parameters(self):
		"""Using homographies from all available images, compute
		intrinsic parameters.
		"""
		V = self.get_coeffient_matrix()
		b = HomographyEstimator.lst_square(V)

		y0 = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]*b[1])
		lmbda = b[5] - (b[3]*b[3] + y0*(b[1]*b[3] - b[0]*b[4]))/b[0]
		alpha_x = np.sqrt(lmbda/b[0])
		alpha_y = np.sqrt(lmbda*b[0]/(b[0]*b[2] - b[1]*b[1]))
		s = -b[1]*alpha_x*alpha_x*alpha_y/lmbda
		x0 = (s*y0/alpha_y) - (b[3]*alpha_x*alpha_x/lmbda) 

		K = np.array([[alpha_x, s, x0], 
					[0, alpha_y, y0], 
					[0, 0, 1]])
		return K
	
	@staticmethod
	def calculate_extrinsic_parameters(K, H):
		"""compute extrinsic parameters"""
		K_inv = np.linalg.pinv(K)
		r1 = K_inv@H[:,0]
		normalizer = np.linalg.norm(r1)
		r1 = r1/normalizer
		r2 = K_inv@H[:,1]/normalizer
		r3 = np.cross(r1, r2)
		t = K_inv@H[:,2]/normalizer
		
		R = np.stack([r1, r2, r3], axis=1)
		R = Calibrator.condition_orthonormal(R)
		return R, t
	
	@staticmethod
	def condition_orthonormal(R):
		"""singular value condioning of R to make sure its
		columns are orthogonal."""
		u, s, vh = np.linalg.svd(R)
		R_cond = u@np.eye(s.size)@vh
		return R_cond
	
	@staticmethod
	def R_to_w(R):
		"""Capture the 3 DoF of rotation matrix into 3-vector,
		using 'rodrigues_transform'
		"""
		phi = np.arccos((np.trace(R)-1)/2)
		w = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])*phi/(2*np.sin(phi))
		return w

	@staticmethod
	def w_to_R(w):
		"""Get rotation matrix R back from 'rodrigues_transform' w
		"""
		phi = np.linalg.norm(w)
		w_cross = np.array([[0, -w[2], w[1]], 
							[w[2], 0, -w[0]],
							[-w[1], w[0], 0]])
		R = np.eye(3) + (np.sin(phi)/phi)*w_cross + ((1-np.cos(phi))/phi**2)*(w_cross@w_cross)
		return R

	@staticmethod
	def get_free_param(K, R_matrices, t_vectors):
		"""Capture the free parameters in a vector"""
		param = []
		param = [K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]]
		for R,t in zip(R_matrices, t_vectors):
			w = Calibrator.R_to_w(R)
			param.extend([*w])
			param.extend([*t])
		
		return np.array(param)

	@staticmethod
	def get_calibration_parameters(param):
		"""returns calibration parameters from free parameters."""
		R_matrices = []
		t_vectors = []
		
		K = np.eye(3)
		K[0,:] = param[:3]
		K[1,1:] = param[3:5]

		param = param[5:]
		num_param = param.size
		for i in range(num_param//6):
			p = param[6*i:6*(i+1)]
			w = p[:3]
			t = p[3:]
			R = Calibrator.w_to_R(w)
			R_matrices.append(R)
			t_vectors.append(t)
		return K, R_matrices, t_vectors
	
	@staticmethod
	def project_pts(K, R, t, pts):
		"""Given calibration parameters and physical points,
		returns projected points.
		"""
		pts_hc = np.concatenate(
			[pts, np.ones((pts.shape[0],1))],
			axis=1).transpose()
		
		H = K@np.stack([R[:,0],R[:,1], t], axis=1)

		proj_pts = H@pts_hc
		proj_pts /= proj_pts[-1]
		return proj_pts[:2].transpose()
		
	
	def compute_residuals(self, param):
		"""Returns residuals for non-linear least squares."""
		residuals = []
		num_pts = self.physical_corners.shape[0]
		physical_corners_hc = np.concatenate(
			[self.physical_corners, np.ones((num_pts,1))],
			axis=1).transpose()
		K, R_matrices, t_vectors = Calibrator.get_calibration_parameters(param)
		for i in range(self.num_images):
			R = R_matrices[i]
			t = t_vectors[i]
			H = K@np.stack([R[:,0],R[:,1], t], axis=1)

			proj_pts = H@physical_corners_hc
			proj_pts /= proj_pts[-1]

			diff = self.img_corner_pts[i] - proj_pts[:2].transpose()
			residuals.extend(diff[:,0])
			residuals.extend(diff[:,1])
		
		return np.array(residuals)
	

	def compute_geometric_distance(self, param):
		"""Returns geometric distance for the given parameters."""
		residuals = self.compute_residuals(param)
		return np.sum(residuals**2)
	
	def calibrate_camera(self):
		"""Returns calibration parameters, both linear and refined using LM."""
		
		K = self.calculate_intrinsic_parameters()
		R_matrices = []
		t_vectors = []
		for H in self.homographies:
			R, t = self.calculate_extrinsic_parameters(K, H)
			R_matrices.append(R)
			t_vectors.append(t)

		print(f"Refining calibration parameters using non-linear least squares.")
		param = Calibrator.get_free_param(K, R_matrices, t_vectors)
		refined_param = scipy.optimize.least_squares(self.compute_residuals, param, method='lm').x
		K_ref, R_mats_ref, t_vec_ref = Calibrator.get_calibration_parameters(refined_param)

		return (K, R_matrices, t_vectors), (K_ref, R_mats_ref, t_vec_ref)
	
	def visualize_reprojections(self, index, calibration_param: tuple, lm_param=False):
		"""Using the calibration parameters, visually compares detected and
		reprojected corner points.
		"""
		if lm_param:
			identifier = f"using-LM-{index}"
		else:
			identifier = f"linear-{index}"

		img_pts = self.img_corner_pts[index]
		K, R_matrices, t_vectors = calibration_param
		proj_pts = Calibrator.project_pts(K, R_matrices[index], t_vectors[index], self.physical_corners)

		geo_error = np.sum((proj_pts - img_pts)**2)
		self.detector.plotter.visualize_projected_pts(
			image=self.images[index], pts=img_pts,proj_pts=proj_pts, geo_error=geo_error,
			identifier=identifier
			)



class Plotter:
	def __init__(self, output_dir=None) -> None:
		self.output_dir = output_dir

	def save_image(self, output_name, sub_dir=None):
		if sub_dir is None:
			filepath = os.path.join(self.output_dir, output_name)
		else:
			dir_path = os.path.join(self.output_dir, sub_dir)
			os.makedirs(dir_path, exist_ok=True)
			filepath = os.path.join(self.output_dir, sub_dir, output_name)

		plt.savefig(filepath)
		print(f"saved to path: {filepath}")
		plt.close()
		
		
	
	def display_img_with_lines(self, img, pts, ax=None, image_name=None, identifier=None, sub_dir=None):
		# Draw the lines
		if ax is None:
			fig, ax = plt.subplots()
		img = img.copy()
		pts = pts.astype(np.int32)
		for i in range(pts.shape[0]):
			pt = pts[i]
			cv2.line(img, (pt[0], pt[1]), (pt[2], pt[3]), (255,0,0), 2, cv2.LINE_AA)

		ax.imshow(img)
		if image_name is not None:
			if identifier is not None:
				title = f"{identifier}: {image_name[:-4]}"
				output_name = f'{identifier}-{image_name}'
			else:
				title = f"{image_name[:-4]}"
				output_name = f'lines-{image_name}'
			plt.title(title)
			self.save_image(output_name, sub_dir=sub_dir)


	def display_img_with_pts(
			self, img, pts, ax=None, image_name=None, fontsize=0.4,
			color=(255,0,0), label_pts=False, identifier=None, sub_dir=None,
			):
	
		if ax is None:
			fig, ax = plt.subplots()
		# Draw the lines
		img = img.copy()
		pts = pts.astype(np.int32)
		for i in range(pts.shape[0]):
			x,y = pts[i]
			cv2.circle(img, (x, y), radius=3, color=color, thickness=-1)

			if label_pts:
				# Add the numbering next to the dot
				cv2.putText(img, f"{i+1}", (x + 5, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
							fontScale=fontsize, color=color, thickness=1)

		ax.imshow(img)
		if image_name is not None:
			if identifier is not None:
				title = f"{identifier}: {image_name[:-4]}"
				output_name = f'{identifier}-{image_name}'
			else:
				title = f"{image_name[:-4]}"
				output_name = f'points_{image_name}'
			plt.title(title)

			self.save_image(output_name, sub_dir=sub_dir)
		return img



	def save_visualization(
			self, img, image_name=None, identifier=None, ax=None, sub_dir=None, ):
		if ax is None:
			fig, ax = plt.subplots()
		
		# Draw the lines
		ax.imshow(img, cmap='gray')
		if image_name is not None:
			if identifier is not None:
				title = f"{identifier}: {image_name[:-4]}"
				output_name = f'{identifier}-{image_name}'
			else:
				title = f"{image_name[:-4]}"
				output_name = f'{image_name}'
			plt.title(title)
			self.save_image(output_name, sub_dir=sub_dir)


	def visualize_projected_pts(self, image, pts, proj_pts, geo_error, identifier):
		"""visualizing corners points and projected points, 
		also adding error in the title.
		"""
		sub_dir = 'reprojection'
		# Draw the lines
		img = 255*np.ones_like(image)
		pts = pts.astype(np.int32)
		pts_color = (255,0,0)
		proj_pts_color = (0,0,255)
		for i in range(pts.shape[0]):
			x,y = pts[i]
			cv2.circle(img, (x, y), radius=3, color=pts_color, thickness=-1)
			cv2.putText(img, f"{i+1}", (x - 17, y+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
							fontScale=0.4, color=pts_color, thickness=1)

			# projected points
			x,y = proj_pts[i].astype(np.uint16)
			cv2.circle(img, (x, y), radius=3, color=proj_pts_color, thickness=-1)
			cv2.putText(img, f"{i+1}", (x + 2, y+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
							fontScale=0.4, color=proj_pts_color, thickness=1)

		title = f"{identifier}, geometric error: {geo_error:.2f}"
		output_name = f'reprojection-{identifier}.jpg'
		plt.imshow(img)
		plt.title(title)
		self.save_image(output_name, sub_dir=sub_dir)


	@staticmethod
	def world_to_cam_coord(X, R, t):
		X_cam = R@X + t
		return X_cam
	
	@staticmethod
	def cam_to_world_coord(X_cam, R, t):
		
		C = -R.T@t
		X = R.T@X_cam + C
		return X 

	@staticmethod
	def get_line_in_3D(origin, head, color='r', width=5, name='X'):   
		origin = origin
		head = head
		# Define the line from start to end for each vector
		vector_line = go.Scatter3d(
			x=[origin[0], head[0]],
			y=[origin[1], head[1]],
			z=[origin[2], head[2]],
			mode='lines',
			line=dict(color=color, width=width),  # Set color and width of the vector line
			name=name,
		)

		# adding arrow to line
		cone_arrows = go.Cone(
			x=[head[0]], 
			y=[head[1]],  
			z=[head[2]],  
			u=[head[0]-origin[0]],  
			v=[head[1]-origin[1]],  
			w=[head[2]-origin[2]],  
			colorscale=[[0, color], [1, color]],
			sizemode="absolute",
			showscale=False,
			sizeref=5,  # Adjust size as needed
		)
		# cone_arrows=None

		return vector_line, cone_arrows
	
	@staticmethod
	def random_color_with_alpha(alpha=0.5):
		return f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {alpha})'

	@staticmethod
	def create_camera_principle_plane(R, t, alpha=0.6, size=15, index=0):
		"""Create a camera plane for the given pos"""
		x = np.linspace(-size, size, 2*size)
		y = np.linspace(-size, size, 2*size)
		X, Y = np.meshgrid(x, y)
		Z = np.zeros_like(X)

		X = X - t[0]
		Y = Y - t[1]
		Z = Z - t[2]

		# Apply the rotation to the X, Y coordinates
		coordinates = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
		rotated_coordinates = np.dot(R.T, coordinates)

		# # Reshape the rotated coordinates back to 2D
		X = rotated_coordinates[0].reshape(X.shape)
		Y = rotated_coordinates[1].reshape(Y.shape)
		Z = rotated_coordinates[2].reshape(Z.shape)
		
		surface = go.Surface(
			z=Z,
			x=X,
			y=Y,
			surfacecolor=np.full_like(X, index),  # You can set surfacecolor based on Z values for variation
			colorscale=[[0, Plotter.random_color_with_alpha(alpha)], [1, Plotter.random_color_with_alpha(alpha)]],
			showscale=False  # Hide the color scale bar
		)

		C = -R.T@t
		# print(f"Center: {C}")
		X_cam_x = np.array([size, 0, 0])
		X_x = Plotter.cam_to_world_coord(X_cam_x, R, t)
		line_x, cone_x = Plotter.get_line_in_3D(C, X_x, 'red', name='X')

		# y line
		X_cam_y = np.array([0, size, 0])
		X_y = Plotter.cam_to_world_coord(X_cam_y, R, t)
		line_y, cone_y = Plotter.get_line_in_3D(C, X_y, 'green', name='Y')

		# z line
		X_cam_z = np.array([0,0, size])
		X_z = Plotter.cam_to_world_coord(X_cam_z, R, t)
		line_z, cone_z = Plotter.get_line_in_3D(C, X_z, 'blue', name='Z')

		return surface, [line_x, line_y, line_z], [cone_x, cone_y, cone_z]

	@staticmethod
	def create_calibration_pattern(size_scaler=3, alpha=0.5):
		s = size_scaler
		x = np.linspace(0, 7*s, 100)  # 10 points along x-axis from 1 to 2
		y = np.linspace(0, 9*s, 100)  # 10 points along y-axis from 1 to 2
		X, Y = np.meshgrid(x, y)   # Create a mesh grid for X and Y

		mask_x = (X>0)&(X<1*s)|(X>2*s)&(X<3*s)|(X>4*s)&(X<5*s)|(X>6*s)&(X<7*s)
		mask_y = (Y>0)&(Y<1*s)|(Y>2*s)&(Y<3*s)|(Y>4*s)&(Y<5*s)|(Y>6*s)&(Y<7*s)|(Y>8*s)&(Y<9*s)
		mask = mask_x & mask_y
		# Z values for the surface (constant value, making it flat)
		Z = np.zeros_like(X)
		C = np.zeros_like(X)
		C[mask] = 1
		
		# Create the surface plots
		surface = go.Surface(
			z=Z,
			x=X,
			y=Y,
			surfacecolor=C,
			colorscale=[[0, 'white'], [1, 'black']],  
			showscale=False  
		)
		return surface



	def make_3D_plot(self, R_matrices, t_vectors, dataset_name, alpha=0.5, size=20, area_size=80,calib_pattern_size=5):
		all_surfaces = []
		all_lines = []
		all_cones = []
		for index, (R, t) in enumerate(zip(R_matrices, t_vectors)):
			surface, lines, cones = Plotter.create_camera_principle_plane(R, t, alpha=alpha, size=size, index=index)
			all_lines.extend(lines)
			all_cones.extend(cones)
			all_surfaces.append(surface)

		all_surfaces.append(Plotter.create_calibration_pattern(calib_pattern_size, alpha=alpha))
		fig = go.Figure(data=all_surfaces+all_lines+all_cones)

		# Set the layout options
		fig.update_layout(
			showlegend=False,
			scene=dict(
				xaxis_title='X',
				yaxis_title='Y',
				zaxis_title='Z',
				xaxis=dict(range=[-60, area_size]),
				yaxis=dict(range=[-20, area_size]),
				zaxis=dict(range=[-area_size, 5]),  # Adjust Z range to include all surfaces
				camera=dict(
					eye=dict(x=1, y=-1.25, z=-1),  # Position of the camera (e.g., diagonally above)
					center=dict(x=0.5, y=0, z=0),          # Center point the camera looks at (scene center)
					up=dict(x=0, y=-1, z=0)               # Defines the "up" direction of the camera
				)
			),
			title=f'Camera poses in 3D ({dataset_name})'
		)
		fig.show()
		# image_path = os.path.join(self.output_dir, f'3D-view-camera-poses-{dataset_name}.png')
		# fig.write_image(image_path, width=600, height=400, scale=1)  
		# print(f"3D view of poses saved to: {image_path}")
		

if __name__ == '__main__':
	calibration_datasets = [
		r'C:\Users\ahmedb\projects\computer-vision/Auxilliary/calibration_dataset1',
		r'C:\Users\ahmedb\projects\computer-vision/Auxilliary/calibration_dataset2'
	]

	output_dirs = [
		r'C:\Users\ahmedb\projects\computer-vision/images/hw8/dataset1',
		r'C:\Users\ahmedb\projects\computer-vision/images/hw8/dataset2',
	]
	fixed_images = ['Pic_13.jpg', 'Pic_1.jpg']
	
	for i, (calibration_dataset, output_dir) in enumerate(zip(calibration_datasets, output_dirs)):

		image_filepaths = glob.glob(os.path.join(calibration_dataset, '*.jpg'))


		# visualize corner detection steps 
		image_files = [os.path.basename(file) for file in image_filepaths]
		detector = CornersDetector(60, 100, 400, output_dir=output_dir)
		for _ in range(5):
			image_name = np.random.choice(image_files)
			image_path = os.path.join(calibration_dataset, image_name)
			print(f"Reading image: {image_name}...", end='')
			image = read_image_from_path(image_path)

			detector.visualize_Hough_lines_extraction(image, image_name=image_name, subdir='corner_detection')


		# Calibration using linear and LM
		images = [read_image_from_path(image_path) for image_path in image_filepaths]
		calibrator = Calibrator(images, cell_length=2.96, output_dir=output_dir)

		linear_param, lm_param = calibrator.calibrate_camera()
		print(f"K: {linear_param[0]}")
		print(f"K (LM): {lm_param[0]}")
		for _ in range(5):
			index = np.random.randint(len(images))
			calibrator.visualize_reprojections(index=index, calibration_param=linear_param, lm_param=False)
			print(f"index: {index}")
			print(f"R: {linear_param[1][index]}")
			print(f"t: {linear_param[2][index]}")
			calibrator.visualize_reprojections(index=index, calibration_param=lm_param, lm_param=True)
			print(f"R: {lm_param[1][index]}")
			print(f"t: {lm_param[2][index]}")

		index = 0
		print(f"Fixed pose")
		calibrator.visualize_reprojections(index=index, calibration_param=linear_param, lm_param=False)
		print(f"R: {linear_param[1][index]}")
		print(f"t: {linear_param[2][index]}")
		calibrator.visualize_reprojections(index=index, calibration_param=lm_param, lm_param=True)
		print(f"R: {lm_param[1][index]}")
		print(f"t: {lm_param[2][index]}")
		print(f"Camera center = {-lm_param[1][index].T@lm_param[2][index]}")

		# visualizing 3D camera view
		K, R_matrices, t_vectors = lm_param
		calibrator.detector.plotter.make_3D_plot(
			R_matrices, t_vectors, 'dataset1', alpha=0.4, size=15, calib_pattern_size=5)
		


import cv2
import pathlib
# import skimage
import sklearn.cluster
import scipy
import computer_vision
import BitVector
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import matplotlib.pylab as plt
import matplotlib as mpl
import plotly.graph_objects as go
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import proj3d
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




# Helper functions..

def read_image_from_path(image_path):
		"""Reads images from the subdir"""
		img_bgr = cv2.imread(image_path)
		# Convert the image from BGR to RGB format
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		return img_rgb


def read_coords(img, num_pts=1):
	# Display the image
	if matplotlib.get_backend() != 'tkagg':
		matplotlib.use('TkAgg')  
	plt.imshow(img)
	plt.title(f"Click on {num_pts} points to get their coordinates")

	try:
		coords = plt.ginput(num_pts, timeout=100)  
		print(f"Coordinates selected: {coords}")
		coords = np.array(coords)
	except Exception as e:
		coords = None
		print(f"Error occurred: {e}. Did you close the window accidentally?")
	plt.close()
	return coords

def manual_correspondences(img1, img2, num_pts=8):
	
	pts_v1 = read_coords(img1, num_pts)
	pts_v2 = read_coords(img2, num_pts)

	assert (pts_v1 is not None) and (pts_v2 is not None), print(f"Points not recorded..!")
	return pts_v1, pts_v2


def plot_corresponding_keypoints(img1, img2, kp1, kp2):
	"""
	Plots two images side by side and draws lines between corresponding keypoints.

	Parameters:
	- img1, img2: The two input images to be plotted side by side.
	- kp1, kp2: The corresponding keypoints in the form of lists of tuples [(y1, x1), (y2, x2), ...].
	"""
	# Combine both images horizontally to create a side-by-side view
	combined_img = np.hstack((img1, img2))  # Horizontally stack img1 and img2

	# Create a figure for plotting
	# fig, ax = plt.subplots(figsize=(12, 6))
	plt.imshow(combined_img)
	# ax.axis('off')  # Hide axes for better visual appeal

	# Plot corresponding keypoints and draw lines between them
	for (x1, y1), (x2, y2) in zip(kp1, kp2):
		color = [random.random() for _ in range(3)]  # Random color for each line
		
		# Plot keypoints on the left image (img1)
		plt.scatter(x1, y1, color=color, s=50, marker='o')
		
		# Plot keypoints on the right image (img2) with an offset for the x-coordinate
		plt.scatter(x2 + img1.shape[1], y2, color=color, s=50, marker='o')  # Offset by width of img1
		
		# Draw a line between the corresponding keypoints
		plt.plot([x1, x2 + img1.shape[1]], [y1, y2], color=color, lw=1)  # Line between the two keypoints

	# plt.show()
	

import scipy.linalg


class Rectifier:
	def __init__(self, images_dir, img1_pts, img2_pts, save_img_pts=False) -> None:
		self.img1_name = 'image-1.jpg'
		self.img2_name = 'image-2.jpg'

		self.images_dir = images_dir
		
		img1 = read_image_from_path(os.path.join(self.images_dir, self.img1_name))
		img2 = read_image_from_path(os.path.join(self.images_dir, self.img2_name))
		
		self.img1 = img1
		self.img2 = img2

		self.x = img1_pts
		self.x_prime = img2_pts
		self.num_pts = img1_pts.shape[0]

		height, width, _ = self.img1.shape
		center = np.array([width//2, height//2])[None,:]
		self.T, self.T_prime = self.get_pre_conditioning_matrices()
		# self.x = self.x - center
		# self.x_prime = self.x_prime - center
		if save_img_pts:
			self.save_img_with_pts()

	def get_pre_conditioning_matrices(self):
		"""Isotropic scalling to pre-condition points."""
		img1_h, img1_w, _ = self.img1.shape
		pixels_x, pixels_y = np.meshgrid(np.arange(img1_w), np.arange(img1_h))
		dom_pixels = np.stack([pixels_x.flatten(), pixels_y.flatten()], axis=1)

		T = Rectifier.compute_pre_condtioning_transform(dom_pixels)

		img2_h, img2_w, _ = self.img2.shape
		pixels_x, pixels_y = np.meshgrid(np.arange(img2_w), np.arange(img2_h))
		range_pixels = np.stack([pixels_x.flatten(), pixels_y.flatten()], axis=1)

		T_prime = Rectifier.compute_pre_condtioning_transform(range_pixels)
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

		
	def save_img_with_pts(self):
		"""Save images with points to disk"""

		output_name = f'img-pts-{self.img1_name}'
		Rectifier.add_pts_to_img(self.img1, self.x)
		self.save_fig(output_name)

		output_name = f'img-pts-{self.img2_name}'
		Rectifier.add_pts_to_img(self.img2, self.x_prime)
		self.save_fig(output_name)

		
		
	def save_fig(self, output_name):
		
		filepath = os.path.join(self.images_dir, output_name)
		plt.savefig(filepath)
		print(f"saved to path: {filepath}")
		plt.close()


	@staticmethod
	def add_pts_to_img(img, pts):
		pts = pts.astype(np.int32)
		pts_color = (255,0,0)
		img = img.copy()
		for i in range(pts.shape[0]):
			x,y = pts[i]
			cv2.circle(img, (x, y), radius=5, color=pts_color, thickness=-1)
			cv2.putText(img, f"{i+1}", (x - 10, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
							fontScale=0.8, color=pts_color, thickness=3)
		title = f"image with recorded pts"
		plt.imshow(img)
		plt.title(title)


	@staticmethod
	def get_coefficient_vector(x, x_prime):
		"""Return the coefficients of linear equations, 
		for the given .

		Args:
			x: pt on 1st image
			x: pt on 2nd image
		"""
		return np.array([x_prime[0]*x[0], x_prime[0]*x[1], x_prime[0],
				   x_prime[1]*x[0], x_prime[1]*x[1], x_prime[1],
				   x[0], x[1], 1])
	
	def get_coeffient_matrix(self, x_pts, x_prime_pts):
		"""For the given data points, get the coefficients matrix
		A for homogeneous system of equations.
		"""
		V = []
		for i in range(self.num_pts):
			V_i = Rectifier.get_coefficient_vector(x_pts[i], x_prime_pts[i])
			V.append(V_i[None,:])
		return np.concat(V, axis=0)
	

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
	
	@staticmethod
	def pre_condition_points(points_2d, T):
		"""Isotropic scalling to pre-condition points."""

		points_hc = np.concat([points_2d, np.ones(points_2d.shape[0])[:,None]], axis=1)
		points_norm = np.matmul(T, points_hc.T)
		return points_norm.T
	
	def get_initial_F(self, precondition=True):

		if precondition:
			x_norm = Rectifier.pre_condition_points(self.x, self.T)
			x_prime_norm = Rectifier.pre_condition_points(self.x_prime, self.T_prime)

		A = self.get_coeffient_matrix(x_norm, x_prime_norm)
		F = Rectifier.lst_square(A).reshape(3,3)

		F = Rectifier.condition_F(F)
		if precondition:
			F_tilde = np.matmul(F, self.T)
			F = np.matmul(np.linalg.pinv(self.T_prime), F_tilde)
			F /= F[-1,-1]
		return F
	
	@staticmethod
	def compute_epipolar_pts(F):
		e = Rectifier.lst_square(F)
		e /= e[-1]
		e_prime = Rectifier.lst_square(F.T)
		e_prime /= e_prime[-1]
		return e, e_prime
	
	def canonical_cameras(self, F, e_prime):

		P = np.concatenate([np.eye(3), np.zeros((3,1))], axis=1)
		e_prime_cross = Rectifier.cross_prod_matrix(e_prime)
		P_prime = np.concatenate([e_prime_cross@F, e_prime[:, None]], axis=1)
		return P, P_prime

	@staticmethod
	def condition_F(F):
		"""Condition F to make it rank-2"""
		u, s, vh = np.linalg.svd(F)
		s[-1] = 0
		F = u@(s*np.eye(3))@vh
		return F
	
	@staticmethod
	def cross_prod_matrix(e):
		return np.array([[0, -e[2], e[1]],
				[e[2], 0, -e[0]],
				[-e[1], e[0], 0]])
	

	@staticmethod
	def get_world_pts(x, x_prime, P, P_prime):
		A = np.zeros((4,4))
		A[0] = x[0]*P[2] - P[0]
		A[1] = x[1]*P[2] - P[1]
		A[2] = x_prime[0]*P_prime[2] - P_prime[0]
		A[3] = x_prime[1]*P_prime[2] - P_prime[1]

		X_world = Rectifier.lst_square(A)
		X_world /= X_world[-1]
		return X_world[:-1]
	

	def compute_residuals(self, params):

		P = np.concatenate([np.eye(3), np.zeros((3,1))], axis=1)
		P_prime = params[:12].reshape(3,4)

		x_hat = []
		x_prime_hat = []
		residuals = []
		for i in range(self.num_pts):
			world_x = params[12+3*i:12+3*(i+1)]

			x_hat = Rectifier.get_img_pt(P, world_x)
			x_prime_hat = Rectifier.get_img_pt(P_prime, world_x)
			residuals.append(self.x[i] - x_hat)
			residuals.append(self.x_prime[i] - x_prime_hat)


		return np.array(residuals).flatten()

	@staticmethod
	def get_img_pt(P, X):
		X = np.concatenate([X, np.ones(1)])
		x = P@X
		x /= x[-1]
		return x[:-1]


	
	def get_refined_estimates(self, P, P_prime):

		params = np.zeros(12 + 3*(self.num_pts))
		params[:12] = P_prime.reshape(-1)
		for i in range(self.num_pts):
			world_x = Rectifier.get_world_pts(self.x[i], self.x_prime[i], P, P_prime)
			params[12+3*i:12+3*(i+1)] = world_x[:3]

		params = scipy.optimize.least_squares(self.compute_residuals, params, method='trf').x
		P_prime = params[:12].reshape(3,4)
		e_prime = P_prime[:,-1]

		F = Rectifier.cross_prod_matrix(e_prime)@P_prime[:,:3]
		F /= F[-1,-1]
		return F, P_prime

	def compute_H_prime(self, e_prime):

		e_prime /= e_prime[-1]
		height, width, _ = self.img2.shape
		x0 = width/1
		y0 = height/1
		
		T1 = np.eye(3)
		T1[0,2] = -x0
		T1[1,2] = -y0

		theta = np.arctan(-(e_prime[1]-y0)/(e_prime[0]-x0))
		R = np.eye(3)
		R[0,0] = np.cos(theta)
		R[0,1] = -np.sin(theta)
		R[1,0] = np.sin(theta)
		R[1,1] = np.cos(theta)

		f = np.abs((e_prime[0] - x0)*np.cos(theta) - (e_prime[1] - y0)*np.sin(theta))

		G = np.eye(3)
		G[2,0] = -1/f

		T2 = np.eye(3)
		T2[0,2] = x0
		T2[1,2] = y0

		H_prime = T2@G@R@T1

		return H_prime/H_prime[-1,-1]
	
	def compute_H(self, P, P_prime, H_prime):

		P_pinv = P.T @ np.linalg.inv(P @ P.T)
		M = P_prime @ P_pinv
		H0 = H_prime @ M

		H0 /= H0[-1,-1]

		# Transform points
		x_hat = Rectifier.transform_pixel_coordinate(H0, self.x)
		x_prime_hat = Rectifier.transform_pixel_coordinate(H_prime, self.x_prime)


		# Solve for Ha
		A = np.ones((x_hat.shape[0], 3))
		A[:, :-1] = x_hat
		b = x_prime_hat[:, 0]
		# Solve Ax - b = 0 using the left pseudo-inverse
		a = (np.linalg.inv(A.T @ A) @ A.T) @ b
		Ha = np.eye(3)
		Ha[0, :] = a

		print(f'Ha:\n{Ha}')
		
		# Compute final H
		H = Ha @ H0
		return H, H0

	
	@staticmethod
	def transform_pixel_coordinate(H, pts):
		# pts = np.random.randn(8,2)
		hc_coords = np.concatenate([pts, np.ones(pts.shape[0])[:, None]], axis=1).transpose()
	
		transformed_hc = np.matmul(H, hc_coords) # gives me (3,n)
		transformed_hc /= transformed_hc[-1]#[None, :]
		return transformed_hc[:-1].transpose()
	


class EdgeDetector:
	def __init__(self, rect_img1, rect_img2) -> None:
		self.rect_img1 = rect_img1
		self.rect_img2 = rect_img2
		self.canny_th1 = 0.4
		self.canny_th2 = 0.6
		self.kp1 = None  # Keypoints from the first image
		self.kp2 = None  # Keypoints from the second image
		self.ssd_scores = None  # SSD scores for the keypoints

	def canny_edges(self, kernel_size=5, num_rows=3):
		# Convert images to grayscale and apply Canny edge detection
		img1_gray = cv2.cvtColor(self.rect_img1, cv2.COLOR_RGB2GRAY)
		img2_gray = cv2.cvtColor(self.rect_img2, cv2.COLOR_RGB2GRAY)
		
		edges1 = cv2.Canny(img1_gray, self.canny_th1, self.canny_th2, apertureSize=7, L2gradient=True)
		edges2 = cv2.Canny(img2_gray, self.canny_th1, self.canny_th2, apertureSize=7, L2gradient=True)

		self.kp1, self.kp2, self.ssd_scores = self.compute_ssd(
			edges1, edges2, kernel_size=kernel_size, num_rows=num_rows
			)

	def compute_ssd(self, edges1, edges2, kernel_size=5, num_rows=3):
		"""Searches for matching key points only in 3 rows"""
		# Assuming edges1 and edges2 are binary edge maps (True for edges, False for non-edges)
		kernel_half = kernel_size // 2


		kp1 = []  # Keypoints from the first image
		kp2 = []  # Keypoints from the second image
		ssd_scores = []  # Corresponding SSD scores

		for i in range(kernel_half+num_rows//2, edges1.shape[0] - kernel_half - num_rows//2):
			for j in range(kernel_half, edges1.shape[1] - kernel_half):
				if edges1[i, j]:
					# restricts search to only within the mask
					ssd_values = []
					# search for match only in 3 rows
					for x in range(i-num_rows//2, i+1+num_rows//2):
						for y in range(kernel_half, edges2.shape[1] - kernel_half):
							if edges2[x,y]:
								# print(x,y)
								region1 = edges1[i-kernel_half:i+kernel_half+1, j-kernel_half:j+kernel_half+1]
								region2 = edges2[x-kernel_half:x+kernel_half+1, y-kernel_half:y+kernel_half+1]

								# print(region1.shape)
								# print(region2.shape)
								ssd_value = np.sum((region1 - region2) ** 2)
								ssd_values.append(((j, i), (y, x), ssd_value))

					min_ssd = min(ssd_values, key=lambda x: x[2])  # Sort by SSD value
					kp1.append(min_ssd[0])
					kp2.append(min_ssd[1])
					ssd_scores.append(min_ssd[2])

		return np.array(kp1), np.array(kp2), np.array(ssd_scores)



	def get_top_keypoints(self, top_percentage=10, max_points=100):

		scores = self.ssd_scores
		num_points = len(scores)

		# Compute the number of top points based on the given percentage or max_points
		num_top_points = min(int(num_points * top_percentage / 100), max_points)

		# Sort the keypoints by SSD score in descending order
		sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)

		# Select the top keypoints and their scores
		top_kp1 = np.array([self.kp1[i] for i in sorted_indices[:num_top_points]])
		top_kp2 = np.array([self.kp2[i] for i in sorted_indices[:num_top_points]])
		top_ssd_scores = np.array([scores[i] for i in sorted_indices[:num_top_points]])

		return top_kp1, top_kp2, top_ssd_scores




class WorldProjector:
	def __init__(self, left_image, right_image, x, x_prime, P, P_prime):

		self.left_image = left_image
		self.right_image = right_image
		self.x = x
		self.x_prime = x_prime
		self.P = P  
		self.P_prime = P_prime  #
		
		self.num_pts = self.x.shape[0]
		self.corner_ids = np.array([0, 1, 14,15], dtype=np.int32)


	@staticmethod
	def get_world_pts(x, x_prime, P, P_prime):
		A = np.zeros((4,4))
		A[0] = x[0]*P[2] - P[0]
		A[1] = x[1]*P[2] - P[1]
		A[2] = x_prime[0]*P_prime[2] - P_prime[0]
		A[3] = x_prime[1]*P_prime[2] - P_prime[1]

		X_world = Rectifier.lst_square(A)
		X_world /= X_world[-1]
		return X_world[:-1]
	
	@staticmethod
	def get_img_pt(P, X):
		X = np.concatenate([X, np.ones(1)])
		x = P@X
		x /= x[-1]
		return x[:-1]
	

	def get_refined_world_pts(self):

		params = np.zeros(3*(self.num_pts))
		for i in range(self.num_pts):
			world_x = Rectifier.get_world_pts(self.x[i], self.x_prime[i], P, P_prime)
			params[3*i:3*(i+1)] = world_x[:3]

		params = scipy.optimize.least_squares(self.compute_residuals, params, method='trf').x
		world_pts = [] 
		for i in range(self.num_pts):
			world_pts.append(params[3*i:3*(i+1)])

		return np.array(world_pts)
	

	def compute_residuals(self, params):
		"""Params are the world points."""
		x_hat = []
		x_prime_hat = []
		residuals = []
		for i in range(self.num_pts):
			world_x = params[3*i:3*(i+1)]

			x_hat = Rectifier.get_img_pt(self.P, world_x)
			x_prime_hat = Rectifier.get_img_pt(self.P_prime, world_x)
			residuals.append(self.x[i] - x_hat)
			residuals.append(self.x_prime[i] - x_prime_hat)

		return np.array(residuals).flatten()


	def plot(self, view_elevation=10, view_azimuth=30):
		"""
		Plot the 3D points and their corresponding projections on the 2D images.
		"""
		world_pts = self.get_refined_world_pts()
		# points_3d = self.project_to_3d()

		# 1. 3D Plot of world points
		fig = plt.figure(figsize=(18, 6))
		ax_3d = fig.add_subplot(132, projection='3d')
		ax_3d.scatter(world_pts[:, 0], world_pts[:, 1], world_pts[:, 2], color='r', marker='o')
		ax_3d.set_title("3D World Points")
		ax_3d.set_xlabel('X')
		ax_3d.set_ylabel('Y')
		ax_3d.set_zlabel('Z')
		# Set view angle
		ax_3d.view_init(elev=view_elevation, azim=view_azimuth)  # 30 degrees elevation, 45 degrees azimuth

		# 2. Plot the 2D projections on the left image
		ax_left = fig.add_subplot(131)
		ax_left.imshow(self.left_image)
		ax_left.set_title("Left Image Projections")
		ax_left.scatter(self.x[:, 0], self.x[:, 1], color='g', s=50, marker='x')

		# 3. Plot the 2D projections on the right image
		ax_right = fig.add_subplot(133)
		ax_right.imshow(self.right_image)
		ax_right.set_title("Right Image Projections")
		ax_right.scatter(self.x_prime[:, 0], self.x_prime[:, 1], color='b', s=50, marker='x')

		# Connect the 3D points to their projections
		left_img_lines = []
		right_img_lines = []
		world_lines = []
		colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:gray','tab:cyan','tab:brown', 'tab:pink']
		# possible_colors = ['tab:blue','tab:orange','tab:green','tab:red', 'tab:gray', 'tab:cyan']
		for ii in range(len(self.corner_ids)):
			for jj in range(ii+1, len(self.corner_ids)):
				corner1 = self.corner_ids[ii]
				corner2 = self.corner_ids[jj]
				# if corner1 != corner2:
				x1, y1 = self.x[corner1]
				x2, y2 = self.x[corner2]
				left_img_lines.append([[x1, x2], [y1, y2]])
				x1, y1 = self.x_prime[corner1]
				x2, y2 = self.x_prime[corner2]
				right_img_lines.append([[x1, x2], [y1, y2]])
				x1, y1, z1 = world_pts[corner1]
				x2, y2, z2 = world_pts[corner2]
				world_lines.append([[x1, x2], [y1, y2], [z1, z2]])
					# colors.append(np.random.rand(3,))
		# colors = possible_colors
		for i, line in enumerate(left_img_lines):
			ax_left.plot(line[0], line[1], color=colors[i], linewidth=3)

		for i, line in enumerate(right_img_lines):
			ax_right.plot(line[0], line[1], color=colors[i], linewidth=3)

		for i, line in enumerate(world_lines):
			ax_3d.plot(line[0], line[1], line[2],color=colors[i], linewidth=3)

		# adding lines across subplots
		for c_id in self.corner_ids:
			xyleft = self.x[c_id]
			xyright = self.x_prime[c_id]
			xyworld = proj3d.proj_transform(*world_pts[c_id][:3], ax_3d.get_proj())[:2]
			
			con = ConnectionPatch(
				xyA=xyleft, xyB=xyworld, coordsA=ax_left.transData, coordsB=ax_3d.transData,
				color='red', linestyle="--", linewidth=2
				
			)
			fig.add_artist(con)
			con = ConnectionPatch(
				xyA=xyworld, xyB=xyright, coordsA=ax_3d.transData, coordsB=ax_right.transData,
				color='red', linestyle="--", linewidth=2
				
			)

			fig.add_artist(con)
		plt.tight_layout()









# Dense Stereo Map implementation..
class DSM:
	def __init__(self, images_dir, window=(7,7)):
		self.images_dir = images_dir
		self.left_img = read_image_from_path(os.path.join(images_dir, 'im2.png'))
		self.right_img = read_image_from_path(os.path.join(images_dir, 'im6.png'))
		self.left_disp_map = DSM.adjust_disp_map(
			read_image_from_path(os.path.join(images_dir, 'disp2.png'))
			)
		self.right_disp_map = DSM.adjust_disp_map(
			read_image_from_path(os.path.join(images_dir, 'disp6.png'))
			)

		self.row_offset = window[0]//2 
		self.col_offset = window[1]//2
		
	@staticmethod
	def adjust_disp_map(disp_map):
		disp_map = disp_map.astype(np.float32)/4
		return disp_map.astype(np.uint8)
	

	def get_bitvector(self, img, row, col):
		img_window = img[row:row+2*self.row_offset+1, col:col+2*self.col_offset+1]
		wind_bitvector = img_window > img[row, col]
		return wind_bitvector
	


	def compute_left_disparity_map(self):

		row_offset = self.row_offset
		col_offset = self.col_offset

		d_max = np.max(self.left_disp_map)

		img_left_g = cv2.cvtColor(self.left_img, cv2.COLOR_RGB2GRAY)
		img_right_g = cv2.cvtColor(self.right_img, cv2.COLOR_RGB2GRAY)

		padded_left = np.pad(img_left_g, (row_offset, col_offset), mode='reflect')
		padded_right = np.pad(img_right_g, (row_offset, col_offset), mode='reflect')

		height, width = img_left_g.shape
		disparity_map = np.zeros((height, width), dtype=np.int32)


		for row in range(height):
				for col in range(width):
					left_bitvector = self.get_bitvector(padded_left, row, col)
					data_costs = []
					for d in range(d_max):
						# print(d)
						d_col = col - d
						if d_col>=0:
							right_bitvector = self.get_bitvector(padded_right, row, d_col)
							cost = np.bitwise_xor(left_bitvector, right_bitvector).sum().item()
							data_costs.append((d, cost))
							# print(d_col, cost)

					min_cost = min(data_costs, key=lambda x: x[1])  # Sort by data cost
					disparity_map[row, col] = min_cost[0]
					# print(min_cost)
					# break

		ground_truth = cv2.cvtColor(self.left_disp_map , cv2.COLOR_RGB2GRAY)
		non_black_mask = ground_truth> 0
		diff = np.abs(ground_truth - disparity_map)
		acc = np.count_nonzero(diff[non_black_mask] < 2) / np.count_nonzero(non_black_mask)
		print(f"Accuracy: {acc}")

		error_mask = np.zeros_like(non_black_mask)
		error_mask[diff < 2] = True
		error_mask[np.bitwise_not(non_black_mask)] = False
		return disparity_map.astype(np.uint8), error_mask.astype(np.uint8), acc


if __name__ == "__main__":

	# Task-1: Image rectification
	images_dir = r'C:\Users\ahmedb\projects\computer-vision\images\hw9'

	img1_pts = np.array([(208, 110), (493, 104), (211, 163), (498, 157),
						(211, 209), (502, 198), (212, 255), (507, 241),
						(219, 312), (360, 267), (514, 319), (221, 369),
						(405, 516), (520, 369), (221, 574), (540, 557)])
	img2_pts = np.array([(147, 132), (418, 103), (146, 182), (420, 154),
						(144, 222), (422, 198), (144, 264), (423, 239),
						(145, 313), (277, 270), (429, 316), (144, 370),
						(308, 513), (433, 364), (135, 560), (443, 559)])

	rectifier = Rectifier(images_dir, img1_pts, img2_pts)
	
	plot_corresponding_keypoints(rectifier.img1, rectifier.img2, img1_pts, img2_pts)
	filepath = os.path.join(images_dir, 'images-with-manual-pts.jpg')
	plt.savefig(filepath)
	plt.close()	
	print(f"image saved to {filepath}")


	F = rectifier.get_initial_F()
	e, e_prime = Rectifier.compute_epipolar_pts(F)
	P, P_prime = rectifier.canonical_cameras(F, e_prime)

	F, P_prime = rectifier.get_refined_estimates(P, P_prime)
	e, e_prime = Rectifier.compute_epipolar_pts(F)

	print(f"e: {e}")
	print(f"e_prime: {e_prime}")

	H_prime = rectifier.compute_H_prime(e_prime)
	H, H0 = rectifier.compute_H(P, P_prime, H_prime)
	print(f"Hp: {H_prime}")
	print(f"H: {H}")
	print(f"H0: {H0}")

	rectified_img1 = cv2.warpPerspective(rectifier.img1 , H, (600, 1200))
	rectified_img2 = cv2.warpPerspective(rectifier.img2 , H_prime, (600, 1200))

	trans_x = Rectifier.transform_pixel_coordinate(H, rectifier.x)
	trans_x_prime = Rectifier.transform_pixel_coordinate(H_prime, rectifier.x_prime)
	plot_corresponding_keypoints(rectified_img1, rectified_img2, trans_x, trans_x_prime)
	filepath = os.path.join(images_dir, 'rectified-images-with-manual-pts.jpg')
	plt.savefig(filepath)
	plt.close()	
	print(f"image saved to {filepath}")


	# Detecting key points
	edge_detector = EdgeDetector(rectified_img1, rectified_img2)

	# Apply Canny edge detection
	edge_detector.canny_edges(kernel_size=21)

	top_kp1, top_kp2, top_ssd_scores = edge_detector.get_top_keypoints(max_points=500)

	plot_corresponding_keypoints(rectified_img1, rectified_img2, top_kp1[:200], top_kp2[:200])
	filepath = os.path.join(images_dir, 'rectified-canny-pairs.jpg')
	plt.savefig(filepath)
	plt.close()
	print(f"image saved to {filepath}")


	img_kp1 = Rectifier.transform_pixel_coordinate(np.linalg.pinv(H) , top_kp1)
	img_kp2 = Rectifier.transform_pixel_coordinate(np.linalg.pinv(H_prime) , top_kp2)

	img_kp1 = np.concatenate([rectifier.x, img_kp1])
	img_kp2 = np.concatenate([rectifier.x_prime, img_kp2])

	projector = WorldProjector(rectifier.img1, rectifier.img2, img_kp1[:200], img_kp2[:200], P, P_prime)
	# world_pts = projector.get_refined_world_pts()
	projector.plot(45, 30)
	filepath = os.path.join(images_dir, 'image-keypoints-projected-world-3d-view-1.jpg')
	plt.savefig(filepath)
	plt.close()
	print(f"image saved to {filepath}")


	projector.plot(60, 0)
	filepath = os.path.join(images_dir, 'image-keypoints-projected-world-3d-view-2.jpg')
	plt.savefig(filepath)
	plt.close()
	print(f"image saved to {filepath}")



	# Task-3: Dense Stereo Map
	images_dir = r'C:\Users\ahmedb\projects\computer-vision\images\hw9'

	print(f"Task-3: Dense Stereo Map")
	windows = [(11,11), (9,9)]
	for i, window in enumerate(windows):
		print(f"Running for window: {window}")
		dsm = DSM(images_dir, window=window)

		disparity_map, error_map, acc = dsm.compute_left_disparity_map()

		plt.imshow(disparity_map, cmap='gray')
		plt.title(f"window= {window}, Acc: {acc:.3f}")
		filepath = os.path.join(images_dir, f'dense-stereo-map-wind{i+1}.jpg')
		plt.savefig(filepath)
		plt.close()	

		plt.imshow(error_map, cmap='gray')
		plt.title(f"error map for window={window}")
		filepath = os.path.join(images_dir, f'stereo-error-map-wind{i+1}.jpg')
		plt.savefig(filepath)
		plt.close()	
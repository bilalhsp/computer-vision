import cv2
import pathlib
import skimage
import computer_vision

import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np

def compute_general_homography(point_pairs):
	"""Takes 4 pairs of points in two images, where 
	each pair of points is a pixel location in one image and corresponding
	pixel location in the 2nd image. 
	Computes general planar projective transform that maps pixel locations
	in the image 1 to pixel locations in the image 2.
	
	Args:
		point_pairs: list(list(tuples)): Each pair of point is represented 
			as [(x,y), (x', y')]. Function expects 4 such pairs of points.
	"""
	num_points = len(point_pairs)
	assert (num_points) >=4, f"expects at least 4 pairs of points but got {num_points}"

	A = []
	y = []
	# np.array([x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_p4, y_p4])[:, None]
	for i, ((x1, y1), (x_p1, y_p1)) in enumerate(point_pairs):
		A.append([x1, y1, 1, 0, 0, 0, -x1*x_p1, -y1*x_p1])
		A.append([0, 0, 0, x1, y1, 1, -x1*y_p1, -y1*y_p1])
		y.extend([x_p1, y_p1])

	A = np.array(A)
	y = np.array(y)[:,None]

	h = np.linalg.lstsq(A, y)[0]
	h = np.append(h, 1)
	# homography matrix would be...
	H = h.reshape(3,3)
	return H

def transform_pixel_coordinate(H, pixel_coordinates):
	"""Applies homography H to the 2D pixel coordinates
	and returns transformed 2D pixel coordinates.
	
	Args:
		H: shape (3, 3): Homography matrix
		pixel_coordinates: shape (2, n): n being the number of coordinates.
	"""
	# pixel_hc = np.concatenate([pixel_coordinates, np.ones(pixel_coordinates.shape[1])[None, :]], axis=0)
	pixel_hc = np.array([*pixel_coordinates, 1])
	transformed_hc = np.matmul(H, pixel_hc) # gives me (3,n)
	transformed_hc /= transformed_hc[-1]#[None, :]
	return transformed_hc[:-1]

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
	# plt.imshow(image)
	# plt.savefig(image_path)
	# Save the image with specific quality
	img_bgr_for_saving = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(image_path, img_bgr_for_saving)
	
def get_physical_coordinates(x_hc):
	x = x_hc[0]/x_hc[2]
	y = x_hc[1]/x_hc[2]
	return (x,y)

def get_homogeneous_coordinates(X):
	return [*X, 1]

def get_output_space_dimensions(H, img):
	"""Map the corners of the image to the physical coordinates,
	to get the extreme coordinates in the physical coordiantes,
	Use these extreme points to adjust physical coordinates, 
	otherwise everything will be mapped to within the grid, that was used for
	dimensions in the physical coordinates initially.

	Args:
		H: 3x3 matrix = Homography matrix
		img: ndarray
		
	Returns:
		physical_height
		physical_width
		offset_height
		offset_width
	"""
	img_height, img_width, _ = img.shape
	print(f"Input image dim: {img_height}x{img_width}")
	image_corners = [
		[0,0], [0, img_width], [img_height, img_width], [img_height, 0]
	]
	output_corners = []
	for point in image_corners:
		output_corners.append(transform_pixel_coordinate(H, point))
	output_corners = np.stack(output_corners, axis=0)

	bottom, right = np.max(output_corners, axis=0)
	top, left = np.min(output_corners, axis=0)
	output_height = bottom - top
	output_width = right - left

	print(f"Output image dim: {output_height}x{output_width}")
	print(f"Output coordinates offsets: {top}, {left}")
	return output_height, output_width, top, left

def is_A_positive_definite(H):
	"""Checks if sub-matrix A (within H) is positive definite of not"""
	return bool(H[0,0]>0 and H[0,0]*H[1,1] > H[0,1]*H[1,0])

def apply_homography(H, img, resize=True, output_dim=(1200, 800)):
	"""Takes in homography matrix 'H' and input image 'img'
	and returns output image obtained after application of homography.

	Args:
		vectorize: bool= Default=True, vectorize homography application.
	"""
	if resize:
		H, *_ = adjust_homography_matrix(H, img, output_dim)
	output_height, output_width, height_offset, width_offset = get_output_space_dimensions(H, img)
	img_height, img_width, _ = img.shape
	H_inv = np.linalg.pinv(H)
	output_height = int(output_height+0.5)
	output_width = int(output_width+0.5)
	print(f"creating image size: {output_height}x{output_width}")
	output_img = np.zeros((output_height, output_width, 3), dtype=img.dtype)
	pixel_y, pixel_x = np.meshgrid(range(output_height), range(output_width))
	hc_coordinates = np.stack([pixel_y.flatten(), pixel_x.flatten(), np.ones_like(pixel_x).flatten()])
	transformed_coordinates = np.matmul(H_inv, hc_coordinates)
	transformed_coordinates /= transformed_coordinates[-1]
	transformed_coordinates = transformed_coordinates.astype(int)
	# create masks for valid (within img range) coordinates...
	mask_coordinates = (transformed_coordinates[0] > 0) & (transformed_coordinates[0] < img_height) & \
			(transformed_coordinates[1] > 0) & (transformed_coordinates[1] < img_width)
	
	output_img[hc_coordinates[:,mask_coordinates][0], hc_coordinates[:,mask_coordinates][1]] = img[
		transformed_coordinates[:,mask_coordinates][0],
		transformed_coordinates[:,mask_coordinates][1]
		]
	return output_img



def adjust_homography_matrix(H, img, output_dim):
	output_height, output_width, height_offset, width_offset = get_output_space_dimensions(H, img)
	# img_height, img_width, _ = img.shape
	# H_inv takes us back from output space to input space...
		# translate coordinates...
	translation_vector = [-1*height_offset, -1*width_offset]
	H_translate = np.eye(3)
	H_translate[:2, 2] = translation_vector
	H = np.matmul(H_translate, H)
	H_resize = np.eye(3)
	H_resize[0,0] = output_dim[0]/output_height
	H_resize[1,1] = output_dim[1]/output_width
	output_height = output_dim[0]
	output_width = output_dim[1]
	H = np.matmul(H_resize, H)
	return H, output_height, output_width


def correct_distortion_using_point_to_point_correspondence(
		img, 
		physical_points,
		image_points,
		vectorize=True,
		resize=True,
		output_dim=(1200,1000),
		
	):
	"""Uses point to point correspondence to compute homography,
	and correct purely projective and affine distortions in the image.

	Args:
		img: input (distorted) image
		physical_points: points on the physical plane
		image_points: points on the image plane
		scale_factor: float = factor that maps physical measurements to pixles
	"""
	# physical_points = [(p1*scale_factor, p2*scale_factor) for p1,p2 in physical_points]
	point_pairs = [[p1, p2] for p1, p2 in zip(physical_points, image_points)]

	H = compute_general_homography(point_pairs)
	if not is_A_positive_definite(H):
		H = condition_matrix(H)
	#     raise RuntimeError("sub-matrix A is not positive definit. Try choosing different points")

	# H = condition_matrix(H)
	H_inv = np.linalg.pinv(H)
	print(f"H_p2p: \nH_inv")
	output_img = apply_homography(
		H_inv, img, vectorize=vectorize, resize=resize, output_dim=output_dim)

	return output_img, H_inv



def condition_matrix(H):
	"""Homography matrices can have high condition number,
	that can make the transform very sensitive, 
	This function improves condition number of H by adding
	small perturbations to the singular values.
	"""
	U, S, Vt = np.linalg.svd(H)
	epsilon = 1e-1
	# S[np.where(S<epsilon)] = epsilon
	# S_perturbed = S
	S_perturbed = S + epsilon*np.random.rand(S.size)
	# S_perturbed = S + np.random.rand(S.size)
	D = S_perturbed*np.eye(3)
	conditioned_H = U@D@Vt
	# epsilon = 1e-3  # Small scalar for noise magnitude
	# noise = epsilon * np.random.randn(3, 3)
	# conditioned_H = H + noise
	return conditioned_H


def compute_H_proj(points):
	"""Given points on the image, computes homography 
	that removed purely projective distortions, using the
	vanishing line (VL).
	"""
	P = get_homogeneous_coordinates(points[0])
	Q = get_homogeneous_coordinates(points[1])
	R = get_homogeneous_coordinates(points[2])
	S = get_homogeneous_coordinates(points[3])

	l1 = np.cross(P,S)
	l2 = np.cross(Q,R)
	vp1 = np.cross(l1, l2)

	l3 = np.cross(P,Q)
	l4 = np.cross(S,R)
	vp2 = np.cross(l3, l4)

	vl = np.cross(vp1, vp2)
	vl = vl.astype(np.float64) / vl[-1]

	H_proj = np.eye(3)
	H_proj[2,:] = vl
	return H_proj

def compute_H_aff(points):
	"""Given points on a rectangle, computes the homography
	that removes affine distortion. Uses the expression for 
	cos(angle) between lines that are expected to be orthogonal 
	in the undistorted image.
	"""
	P = get_homogeneous_coordinates(points[0])
	Q = get_homogeneous_coordinates(points[1])
	R = get_homogeneous_coordinates(points[2])
	S = get_homogeneous_coordinates(points[3])

	l1 = np.cross(P,S).astype(np.float64)
	m1 = np.cross(P,Q).astype(np.float64)
	l2 = np.cross(P,R).astype(np.float64)
	m2 = np.cross(Q,S).astype(np.float64)

	l1 /= l1[-1]
	m1 /= m1[-1]
	l2 /= l2[-1]
	m2 /= m2[-1]

	X = np.zeros((2,2))
	X[0,0] = l1[0]*m1[0]
	X[0,1] = l1[0]*m1[1] + l1[1]*m1[0]
	X[1,0] = l2[0]*m2[0]
	X[1,1] = l2[0]*m2[1] + l2[1]*m2[0]

	y = np.zeros((2,1))
	y[0] = -l1[1]*m1[1]
	y[1] = -l2[1]*m2[1]

	sol = np.linalg.lstsq(X, y)[0].squeeze()
	S = np.zeros((2,2))
	S[0,0] = sol[0]
	S[0,1] = sol[1]
	S[1,0] = sol[1]
	S[1,1] = 1

	e, V = np.linalg.eig(S)

	D = np.eye(2)*np.sqrt(e)
	A = V@D@V.T
	H_aff = np.eye(3)
	H_aff[:2, :2] = A
	return np.linalg.pinv(H_aff)
	

def correct_distortion_using_two_step_method(
		img, 
		image_points,
		vectorize=True,
		resize=False,
		output_dim=(1200,1000),
	):
	"""Uses Vanishing line (VL) to remove purely projective 
	distortion and then uses orthogonal lines to get the affine distortion.
	"""
	H_proj = compute_H_proj(image_points)
	print(f"H_proj: \n{H_proj}")
	H_aff = compute_H_aff(image_points)
	print(f"H_aff: \n{H_aff}")
	H_comb = H_proj@H_aff
	print(f"H_comb: \n{H_comb}")

	img_proj = apply_homography(H_proj, img, vectorize=vectorize, resize=resize, output_dim=output_dim)
	# H_aff_inv = np.linalg.pinv(H_aff)
	img_aff = apply_homography(H_aff, img_proj, vectorize=vectorize, resize=resize, output_dim=output_dim)
	return img_proj, img_aff, H_proj, H_aff, H_comb

def get_line_pairs(corner_points):
	"""Given the corner points of the rectangle,
	returns five line pairs
	"""
	epsilon = 1e-6
	hc_points = [get_homogeneous_coordinates(pt) for pt in corner_points]
	line_pairs = []
	num_points = len(hc_points)
	for i in range(num_points):
		p1 = (i-1)%num_points
		p2 = i
		p3 = (i+1)%num_points
		l1 = np.cross(hc_points[p1], hc_points[p2]).astype(np.float64)
		l1 /= l1[-1] + epsilon
		l2 = np.cross(hc_points[p2], hc_points[p3]).astype(np.float64)
		l2 /= l2[-1] + epsilon
		line_pairs.append([l1, l2])

	p1, p2 = 0, 1
	l1 = np.cross(hc_points[p1], hc_points[(p1+2)%num_points]).astype(np.float64)
	l1 /= l1[-1] + epsilon
	l2 = np.cross(hc_points[p2], hc_points[(p2+2)%num_points]).astype(np.float64)
	l2 /= l2[-1] + epsilon
	line_pairs.append([l1, l2])
	return line_pairs
	
def compute_H_one_step(corner_points):
	"""Given the five line pairs computes homography matrix,
	using one step method of using orthogonal lines."""

	# from the corner points, get the five pair of lines..
	line_pairs = get_line_pairs(corner_points)

	X = []
	y = []
	for l, m in line_pairs:
		row = [l[0]*m[0], (l[0]*m[1]+l[1]*m[0]), l[1]*m[1], (l[0]*m[2]+l[2]*m[0]), (l[1]*m[2]+l[2]*m[1])]
		X.append(row)
		y.append(-l[2]*m[2])
	X = np.array(X)
	y = np.array(y)[:, None]

	# least-squares solution to the system of equations.. 
	sol = np.linalg.lstsq(X, y)[0].squeeze()

	S = np.array([[sol[0], sol[1]],
				[sol[1], sol[2]]])

	e, V = np.linalg.eig(S)
	D = np.eye(2)*np.sqrt(e)
	A = V@D@V.T
	v = np.linalg.pinv(A)@np.array([sol[3], sol[4]])[:, None]
	H = np.eye(3)
	H[:2, :2] = A
	H[2, :2] = v.T
	return np.linalg.pinv(H)

	


if __name__ == '__main__':
	
	
	# # Task-1: board image
	physical_points = [
		(0, 0),
		(0, 8),
		(12, 8),
		(12, 0),
	]
	image_points = [
		(420, 65),
		(140, 1217),
		(1958, 1356),
		(1794, 421),
	]
	# board image: Point-to-point
	filename = 'board_1.jpeg'
	img = read_image('hw3', filename)
	img_p2p, H = correct_distortion_using_point_to_point_correspondence(
		img,
		physical_points=physical_points,
		image_points=image_points,
		vectorize=True,
		resize=True,
		output_dim=img.shape[:-1], 
		)

	method = 'point-to-point'
	save_image(img_p2p, 'hw3', f'{filename[:-5]}-{method}.png')

	# board image: two-step
	img_proj, img_aff, H_proj, H_aff, H_comb = correct_distortion_using_two_step_method(
		img, image_points, resize=True, output_dim=img.shape[:-1]
	)
	method = 'two-step-proj'
	save_image(img_proj, 'hw3', f'{filename[:-5]}-{method}.png')
	method = 'two-step-aff'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# board image: one-step
	H = compute_H_one_step(image_points)
	output_img = apply_homography(H, img, resize=True, output_dim=img.shape[:-1])
	method = 'one-step'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# Task2: corridor
	physical_points = [
		(0,0),
		(0,24),
		(8, 24),
		(8, 0),
	]
	image_points = [
		(541, 533),
		(249, 1313),
		(1340, 1294),
		(909, 525)
	]

	filename = 'corridor.jpeg'
	img = read_image('hw3', filename)

	# corridor: point-to-point
	img_p2p, H = correct_distortion_using_point_to_point_correspondence(
		img,
		physical_points=physical_points,
		image_points=image_points,
		vectorize=True,
		resize=True,
		output_dim=img.shape[:-1],
		)

	method = 'point-to-point'
	save_image(img_p2p, 'hw3', f'{filename[:-5]}-{method}.png')

	# corridor: two-step
	img_proj, img_aff, H_proj, H_aff, H_comb = correct_distortion_using_two_step_method(
		img, image_points, resize=True, output_dim=img.shape[:-1]
	)
	method = 'two-step-proj'
	save_image(img_proj, 'hw3', f'{filename[:-5]}-{method}.png')
	method = 'two-step-aff'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# corridor: one-step
	H = compute_H_one_step(image_points)

	output_img = apply_homography(H, img, resize=True, output_dim=img.shape[:-1])
	method = 'one-step'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# Task2: IPad
	# Ipad: point-to-point
	physical_points = [
		(0, 0),
		(0, 8.8),
		(6.4, 8.8),
		(6.4, 0),
	]
	image_points = [
		(67, 287),
		(325, 516),
		(524, 354),
		(260, 189),
	]

	filename = 'ipad.png'
	img = read_image('hw3', filename)

	img_p2p, H = correct_distortion_using_point_to_point_correspondence(
		img,
		physical_points=physical_points,
		image_points=image_points,
		vectorize=True,
		resize=True,
		output_dim=img.shape[:-1], 
		)

	method = 'point-to-point'
	save_image(img_p2p, 'hw3', f'{filename[:-5]}-{method}.png')

	# Ipad: Two-point method...
	img_proj, img_aff, H_proj, H_aff, H_comb = correct_distortion_using_two_step_method(
		img, image_points, resize=True, output_dim=img.shape[:-1]
	)
	method = 'two-step-proj'
	save_image(img_proj, 'hw3', f'{filename[:-5]}-{method}.png')
	method = 'two-step-aff'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# Ipad: One-point method...
	H = compute_H_one_step(image_points)
	output_img = apply_homography(H, img, resize=True, output_dim=img.shape[:-1])
	method = 'one-step'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# Task-2: frame:
	physical_points = [
		(0, 0),
		(0, 10.8),
		(13, 10.8),
		(13, 0),
	]
	# 13, 10.8
	image_points = [
		(143, 391),
		(220, 628),
		(460, 352),
		(276, 106),
	]

	filename = 'frame.png'
	img = read_image('hw3', filename)

	# frame: point-to-point
	img_p2p, H = correct_distortion_using_point_to_point_correspondence(
		img,
		physical_points=physical_points,
		image_points=image_points,
		vectorize=True,
		resize=True,
		output_dim=img.shape[:-1], 
		)

	method = 'point-to-point'
	save_image(img_p2p, 'hw3', f'{filename[:-5]}-{method}.png')

	# frame: two-point

	img_proj, img_aff, H_proj, H_aff, H_comb = correct_distortion_using_two_step_method(
		img, image_points, resize=True, output_dim=img.shape[:-1]
	)
	method = 'two-step-proj'
	save_image(img_proj, 'hw3', f'{filename[:-5]}-{method}.png')
	method = 'two-step-aff'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	# frame: one-point
	H = compute_H_one_step(image_points)
	output_img = apply_homography(H, img, resize=True, output_dim=img.shape[:-1])
	method = 'one-step'
	save_image(img_aff, 'hw3', f'{filename[:-5]}-{method}.png')

	
	
	
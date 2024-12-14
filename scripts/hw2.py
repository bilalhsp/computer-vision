import cv2
import pathlib
import skimage
import computer_vision

import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np

def compute_affine_homography(point_pairs):
	"""Takes 3 pairs of points in two images, where 
	each pair of points is a pixel location in one image and corresponding
	pixel location in the 2nd image. 
	Computes general planar projective transform that maps pixel locations
	in the image 1 to pixel locations in the image 2.
	
	Args:
		point_pairs: list(list(tuples)): Each pair of point is represented 
			as [(x,y), (x', y')]. Function expects 4 such pairs of points.
	"""
	num_points = len(point_pairs)
	assert (num_points) ==4, f"expects 4 pairs of points but got {num_points}"
	(x1, y1), (x_p1, y_p1) = point_pairs[0]
	(x2, y2), (x_p2, y_p2) = point_pairs[1]
	(x3, y3), (x_p3, y_p3) = point_pairs[2]
	(x4, y4), (x_p4, y_p4) = point_pairs[3]


	A = np.array([
		[x1, y1, 1, 0, 0, 0],
		[0, 0, 0, x1, y1, 1],  
		[x2, y2, 1, 0, 0, 0],
		[0, 0, 0, x2, y2, 1], 
		[x3, y3, 1, 0, 0, 0],
		[0, 0, 0, x3, y3, 1], 
        [x4, y4, 1, 0, 0, 0],
		[0, 0, 0, x4, y4, 1], 
	])
	# y needs to be column vector
	y = np.array([x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_p4, y_p4])[:, None]

	h = np.linalg.lstsq(A, y)[0]
	# homography matrix would be...
	H = h.reshape(2,3)
	H = np.concatenate([H, np.array([0, 0, 1])[None,:]], axis=0)
	return H


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
	assert (num_points) ==4, f"expects 4 pairs of points but got {num_points}"
	(x1, y1), (x_p1, y_p1) = point_pairs[0]
	(x2, y2), (x_p2, y_p2) = point_pairs[1]
	(x3, y3), (x_p3, y_p3) = point_pairs[2]
	(x4, y4), (x_p4, y_p4) = point_pairs[3]

	A = np.array([
		[x1, y1, 1, 0, 0, 0, -x1*x_p1, -y1*x_p1],
		[0, 0, 0, x1, y1, 1, -x1*y_p1, -y1*y_p1],  
		[x2, y2, 1, 0, 0, 0, -x2*x_p2, -y2*x_p2],
		[0, 0, 0, x2, y2, 1, -x2*y_p2, -y2*y_p2], 
		[x3, y3, 1, 0, 0, 0, -x3*x_p3, -y3*x_p3],
		[0, 0, 0, x3, y3, 1, -x3*y_p3, -y3*y_p3], 
		[x4, y4, 1, 0, 0, 0, -x4*x_p4, -y4*x_p4],
		[0, 0, 0, x4, y4, 1, -x4*y_p4, -y4*y_p4],      
	])
	# y needs to be column vector
	y = np.array([x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_p4, y_p4])[:, None]

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
		pixel_coordinates: shape (2,): either a tuple of a list.
	"""
	pixel_hc = np.array([*pixel_coordinates, 1])
	transformed_hc = np.dot(H, pixel_hc) 
	transformed_hc /= transformed_hc[-1]
	return transformed_hc[:-1]


def project_image_onto_frame(frame_image, image_of_interest, points_on_frame, points_on_image,
                             affine_homography=False):
    """Given 4 points on the frame image and 4 corresponding points on the image of interest,
    projects region of interest (ROI) from image of interest (based on coordinates provided),
    onto the frame.
    
    Args:
        frame_image: frame image
        image_of_interest: image of interest
        points_on_frame: shape [(x,y), (x,y), (x,y), (x,y)] = corner points PQRS of the frame
        points_on_image: shape [(x,y), (x,y), (x,y), (x,y)] = corner points defining ROI on the image.
    Returns:
        modified_frame_image:
    """
    pairs_of_points = []
    for p1, p2 in zip(points_on_frame, points_on_image):
        pairs_of_points.append([p1, p2])

    # compute homography that gives pixel coordinates on the image for 
    # given pixel coordinate on the frame.
    if affine_homography:
        # needs only 3 points for affine...
        H = compute_affine_homography(pairs_of_points)
    else:
        H = compute_general_homography(pairs_of_points)

    # read pixels of the frame as a polygon
    rows = [x for x,_ in points_on_frame]
    cols = [y for _,y in points_on_frame]
    rr, cc = skimage.draw.polygon(rows, cols)

    # update pixels within the polygon by the corresponding pixels 
    # from the image of interest...
    for r, c in zip(rr, cc):
        rt, ct = transform_pixel_coordinate(H, np.array([r,c]))
        if rt < image_of_interest.shape[0] and ct < image_of_interest.shape[1]:
            pixel_value = image_of_interest[int(rt), int(ct)]
            frame_image[r,c] = pixel_value

    return frame_image
    

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


# Task: 1.1: project 1d onto 1a
P = (830, 410)
Q = (850, 2550)
R = (2288, 2400)
S = (3170, 675)

P_roi = (0, 0)
Q_roi = (0, 780)
R_roi = (660, 780)
S_roi = (660, 0)


points_on_frame = [P, Q, R, S]
points_on_image = [P_roi, Q_roi, R_roi, S_roi]

frame_image = read_image(subdir='hw2', image_name='img1.jpg')
image_of_interest = read_image(subdir='hw2', image_name='alex_honnold.jpg')

updated_frame_image = project_image_onto_frame(
    frame_image=frame_image,
    image_of_interest=image_of_interest,
    points_on_frame=points_on_frame,
    points_on_image=points_on_image
    )

save_image(updated_frame_image, 'hw2', 'projected-image-1a.jpg')

# Task: 1.1: project 1d onto 1b
# corners of frame
P = (1425, 518)
Q = (802, 1861)
R = (2820, 1946)
S = (2690, 456)

# ROI on the image of interest
P_roi = (0, 0)
Q_roi = (0, 780)
R_roi = (660, 780)
S_roi = (660, 0)


points_on_frame = [P, Q, R, S]
points_on_image = [P_roi, Q_roi, R_roi, S_roi]

frame_image = read_image(subdir='hw2', image_name='img2.jpg')
image_of_interest = read_image(subdir='hw2', image_name='alex_honnold.jpg')

updated_frame_image = project_image_onto_frame(
    frame_image=frame_image,
    image_of_interest=image_of_interest,
    points_on_frame=points_on_frame,
    points_on_image=points_on_image
    )

save_image(updated_frame_image, 'hw2', 'projected-image-1b.jpg')

# Task: 1.1: project 1d onto 1c
# corners of frame
P = (520, 1201)
Q = (2295, 2980)
R = (3190, 1794)
S = (1792, 242)

# ROI on the image of interest
P_roi = (0, 0)
Q_roi = (0, 780)
R_roi = (660, 780)
S_roi = (660, 0)


points_on_frame = [P, Q, R, S]
points_on_image = [P_roi, Q_roi, R_roi, S_roi]

frame_image = read_image(subdir='hw2', image_name='img3.jpg')
image_of_interest = read_image(subdir='hw2', image_name='alex_honnold.jpg')

updated_frame_image = project_image_onto_frame(
    frame_image=frame_image,
    image_of_interest=image_of_interest,
    points_on_frame=points_on_frame,
    points_on_image=points_on_image
    )

save_image(updated_frame_image, 'hw2', 'projected-image-1c.jpg')

# Task: 1.2: combining homographies
# coordinates on img1
P = (830, 410)
Q = (850, 2550)
R = (2288, 2400)
S = (3170, 675)
points_on_1a = [P, Q, R, S]

# coordinates on img2
# corners of frame
P = (1425, 518)
Q = (802, 1861)
R = (2820, 1946)
S = (2690, 456)
points_on_1b = [P, Q, R, S]

# coordinates on img3
# corners of frame
P = (520, 1201)
Q = (2295, 2980)
R = (3190, 1794)
S = (1792, 242)

points_on_1c = [P, Q, R, S]

# compute homography that takes from 1b to 1a
point_pairs = [[p1, p2] for p1, p2 in zip(points_on_1b, points_on_1a)]
H1 = compute_general_homography(point_pairs)

# compute homography that takes from 1c to 1b
point_pairs = [[p1, p2] for p1, p2 in zip(points_on_1c, points_on_1b)]
H2 = compute_general_homography(point_pairs)

# product of 2 homographies..
H = np.matmul(H1, H2)

# get coordinates of target frame
rows = [x for x,_ in points_on_1c]
cols = [y for _,y in points_on_1c]
rr, cc = skimage.draw.polygon(rows, cols)

frame_image = read_image(subdir='hw2', image_name='img3.jpg')
image_of_interest = read_image(subdir='hw2', image_name='img1.jpg')

# update pixels within the polygon by the corresponding pixels 
# from the image of interest...
for r, c in zip(rr, cc):
    rt, ct = transform_pixel_coordinate(H, np.array([r,c]))
    pixel_value = image_of_interest[int(rt), int(ct)]
    frame_image[r,c] = pixel_value

# plt.imshow(updated_frame_image)
save_image(frame_image, 'hw2', '1a-transformed-to-1c.jpg')


# Task 1.3: Affine Homography: 1a
P = (830, 410)
Q = (850, 2550)
R = (2288, 2400)
S = (3170, 675)

P_roi = (0, 0)
Q_roi = (0, 780)
R_roi = (660, 780)
S_roi = (660, 0)

points_on_frame = [P, Q, R, S]
points_on_image = [P_roi, Q_roi, R_roi, S_roi]

frame_image = read_image(subdir='hw2', image_name='img1.jpg')
image_of_interest = read_image(subdir='hw2', image_name='alex_honnold.jpg')

updated_frame_image = project_image_onto_frame(
    frame_image=frame_image,
    image_of_interest=image_of_interest,
    points_on_frame=points_on_frame,
    points_on_image=points_on_image,
    affine_homography=True
    )

save_image(updated_frame_image, 'hw2', 'projected-image-1a-using-affine.jpg')

# Task 1.3: Affine Homography: 1b
# corners of frame
P = (1425, 518)
Q = (802, 1861)
R = (2820, 1946)
S = (2690, 456)

P_roi = (0, 0)
Q_roi = (0, 780)
R_roi = (660, 780)
S_roi = (660, 0)


points_on_frame = [P, Q, R, S]
points_on_image = [P_roi, Q_roi, R_roi, S_roi]

frame_image = read_image(subdir='hw2', image_name='img2.jpg')
image_of_interest = read_image(subdir='hw2', image_name='alex_honnold.jpg')

updated_frame_image = project_image_onto_frame(
    frame_image=frame_image,
    image_of_interest=image_of_interest,
    points_on_frame=points_on_frame,
    points_on_image=points_on_image,
    affine_homography=True
    )

save_image(updated_frame_image, 'hw2', 'projected-image-1b-using-affine.jpg')


# Task 1.3: Affine Homography: 1c
# corners of frame
P = (520, 1201)
Q = (2295, 2980)
R = (3190, 1794)
S = (1792, 242)

P_roi = (0, 0)
Q_roi = (0, 780)
R_roi = (660, 780)
S_roi = (660, 0)


points_on_frame = [P, Q, R, S]
points_on_image = [P_roi, Q_roi, R_roi, S_roi]

frame_image = read_image(subdir='hw2', image_name='img3.jpg')
image_of_interest = read_image(subdir='hw2', image_name='alex_honnold.jpg')

updated_frame_image = project_image_onto_frame(
    frame_image=frame_image,
    image_of_interest=image_of_interest,
    points_on_frame=points_on_frame,
    points_on_image=points_on_image,
    affine_homography=True
    )

save_image(updated_frame_image, 'hw2', 'projected-image-1c-using-affine.jpg')

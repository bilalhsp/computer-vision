import os
import math
import numpy as np
import computer_vision
from pathlib import Path
import matplotlib.pyplot as plt
import computer_vision.utils as utils
from computer_vision.utils import Plotter


def get_physical_coordinates(x_hc):
	"""Returns physical coordinates from homogeneous coordinates."""
	x = x_hc[0]/x_hc[2]
	y = x_hc[1]/x_hc[2]
	return (x,y)

def laser_aim_check(vertices, alpha):
	"""Checks if the laser is going to intersect the triangle.
	
	Args:
		alpha: float = laser aiming angle in degrees.
		vertices: (v1, v2, v3): where v1, v2, v3 are tuples themselves,
			specifying physical (x,y) coordinates vertices of triangle.
			
	Return:
		bool: True when laser will interesect triangle, false otherwise
	"""
	alpha_rad = math.radians(alpha)

	origin = [0, 0, 1]
	point_on_aiming_line = [1, math.tan(alpha_rad), 1]
	aiming_line = np.cross(origin, point_on_aiming_line)
	print(f"aiming_line: {aiming_line}")
	v1, v2, v3 = vertices

	# lines intersecting vertices of triangle.
	l12 = np.cross([*v1, 1], [*v2, 1])
	print(f"l12: {l12}")
	l23 = np.cross([*v2, 1], [*v3, 1])
	print(f"l23: {l23}")
	l13 = np.cross([*v1, 1], [*v3, 1])
	print(f"l13: {l13}")
	# points of intersection of aiming line with lines connecting vertices.
	x12 = np.cross(l12, aiming_line)
	x23 = np.cross(l23, aiming_line)
	x13 = np.cross(l13, aiming_line)
	print(f"x12: {x12}")
	print(f"x23: {x23}")
	print(f"x13: {x13}")
	
	x12_physical_x, *_ = get_physical_coordinates(x12)
	x23_physical_x, *_ = get_physical_coordinates(x23)
	x13_physical_x, *_ = get_physical_coordinates(x13)
	
	# test if aiming line intersects the triangle or not...
	if (x12_physical_x > v1[0] and x12_physical_x < v2[0]) or \
			(x23_physical_x > v3[0] and x23_physical_x < v2[0]) or \
			(x13_physical_x > v3[0] and x13_physical_x < v1[0]):
		return True
	else:
		return False
	


def laser_aim_game():
	"""For randomly generated triangles and aiming angles, the function 
	checks if the laser is going to intersect the triangle and sets its color to green 
	otherwise red.
	"""
	path = Path(computer_vision.__file__)
	images_dir = path.parents[1] / 'images' / 'hw1'

	triangle_config = {
		'area': 4,
		'x_min': -8,
		'x_max': 8,
		'y_min': 7,
		'y_max': 15,
	}

	arrow_config = {
		'start_point': (0, 0),  # Starting point of the arrow
		'length': 5,
		'angle': 45,
		
	}
	arena_config = {
		'circle_radius': 2,
		'x_lim': (-10, 10),
		'y_lim': (-2, 16),
	}
	
	# randomly places triangle
	vertices = utils.create_random_triangle_2d(**triangle_config)
	# setting the laser aim randomly
	alpha = 20 * np.random.randn() + 90
	
	ax = Plotter.plot_arena(**arena_config)
	# check if laser aim is good enough to hit the enemy triangle
	hit = laser_aim_check(vertices=vertices, alpha=alpha)
	if hit:
		tri_color = 'tab:green'
	else:
		tri_color = 'tab:red'
	arrow_config['angle'] = alpha
	ax = Plotter.plot_arrow(**arrow_config, ax=ax)
	ax = Plotter.plot_triangle(vertices=vertices, ax=ax, color=tri_color)

	plt.savefig(os.path.join(images_dir, 'image6.jpg'))
		

if __name__ == '__main__':
	# laser_aim_game()
	vertices = (
		(5,3),
		(7,5),
		(3,5)
	)
	alpha = 45
	hit = laser_aim_check(vertices=vertices, alpha=alpha)
	if hit:
		print(f"It's is hit")
	else:
		print(f"It's a miss.")
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle


def create_random_triangle_2d(
		area=2, x_min=-10, x_max=10, y_min=-10, y_max=10):
	"""Creates a random triangle of specified area 
	in 2D space, and returns the vertices of triangle.
	The 2D space is bounded between (-x_max:x_max, -y_max:y_max)
	Args:
		area: float = area of triangle to be created.
		x_max: float = max magnitude along x-axis
		y_max: float = max magnitude along y-axis
	Returns:
		(v1, v2, v3): tuple of vertices of triangle.
	"""
	
	# randomly pick the center of triangle
	center_x = np.random.randint(x_min, x_max)
	center_y = np.random.randint(y_min, y_max)
	
	# area = b*h/2 = h**2/2 (for b=2h)
	# base = 4
	# height = 2
	height = math.sqrt(area)
	base = 2*height
	# side_length = math.sqrt(area*4/math.sqrt(3))
	
	# half_length = side_length/2
	v1 = (center_x, center_y-height/2)
	v2 = (center_x+base/2, center_y+height/2)
	v3 = (center_x-base/2, center_y+height/2)
	
	return (v1, v2, v3)





def get_physical_coordinates(x_hc):
	x = x_hc[0]/x_hc[2]
	y = x_hc[1]/x_hc[2]
	return (x,y)

def get_homogeneous_coordinates(X):
	return [*X, 1]


class Plotter:

	@staticmethod
	def plot_arena(circle_radius=2, x_lim=(-10, 10), y_lim=(-2, 30)):
		"""Plots the arena of the 'Laser aim game'"""

		# Create the plot
		fig, ax = plt.subplots()

		# Draw the x and y axes with dotted lines
		ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
		ax.axvline(0, color='black', linestyle='--', linewidth=0.7)

		# Add a red circle centered at the origin
		circle = Circle(
				(0, 0),
				circle_radius,
				color='red',
				fill=False,
				linewidth=2,
				linestyle='dashed',
				)
		ax.add_patch(circle)

		# Set plot limits
		ax.set_xlim(*x_lim)
		ax.set_ylim(*y_lim)

		ax.set_xticks([])
		ax.set_yticks([])

		# Set aspect ratio to be equal to maintain circle shape
		ax.set_aspect('equal')
		# Remove spines
		for spine in ax.spines.values():
			spine.set_visible(False)

		# Add labels and title
		plt.title('Laser aim game')
		plt.xlabel('x-axis')
		plt.ylabel('y-axis')

		return ax
	

	@staticmethod
	def plot_arrow(start_point=(0,0), length=2, angle=45, ax=None):
		"""Plots an arrow, starting at start point, and pointing at an angle specified.
		Make sure to externally set the xlim and ylim of the plot.
		
		Args:
			start_point: (x,y): coordinates of starting point of arrow
			length: float: length of arrow
			angle: float: angle of arrow
		
		"""
		# Convert angle to radians
		angle_rad = np.radians(angle)

		# Calculate the end point of the arrow using trigonometry
		end_point = (
			start_point[0] + length * np.cos(angle_rad),
			start_point[1] + length * np.sin(angle_rad)
		)

		if ax is None:
			# Create the plot
			fig, ax = plt.subplots()

		# Create a FancyArrowPatch object for the arrow
		arrow = FancyArrowPatch(
			start_point, 
			end_point, 
			# connectionstyle="arc3,rad=.1",  # Adjust rad for curvature
			arrowstyle="->",  # Arrow head style
			mutation_scale=10,  # Size of the arrowhead
			color="purple"
		)

		# Add the arrow to the plot
		ax.add_patch(arrow)
		# x_pos = 1
		# y_pos = math.tan(angle_rad)
		# if y_pos < 0:
		# 	y_pos *= -1
		# 	x_pos = -1
		# ax.text(
		# 	x_pos, y_pos , f'{angle:.2f}°',
		# 	fontsize=5, rotation=angle,
		# 	rotation_mode='anchor',
		# 	color='k'
		# 	)
		return ax

	@staticmethod
	def plot_triangle(vertices, font_size=5, ax=None, color=None):
		"""Given three vertices of triangle, plots the triangle on 2D plane.
		Make sure to adjust xlim and ylim outside this function
		"""
		if ax is None:
			fig, ax = plt.subplots()
		if color is None:
			color = 'tab:blue'
		vertices = np.array(vertices)
		
		ax.scatter(vertices[:,0], vertices[:,1], color='k', linewidths=1, s=5, edgecolors='r')
		ax.text(
			vertices[0,0], vertices[0,1]-1, f'({int(vertices[0,0])},{int(vertices[0,1])})',
		   	fontsize=font_size, ha='center', color='k'
			)

		for i in range(1,3):
			ax.text(
				vertices[i,0], vertices[i,1]+1, f'({int(vertices[i,0])},{int(vertices[i,1])})',
				fontsize=font_size, ha='center', color='k'
				)
		
		triangle = Polygon(vertices, closed=True, fill=True, color=color)

		ax.add_patch(triangle)

		return ax



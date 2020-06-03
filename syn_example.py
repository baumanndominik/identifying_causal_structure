import numpy as np 

'''
A synthetic, linear example to test the causal structure identification
'''

class synthetic_example:

	def __init__(self):
		# System dynamics
		self.A = np.array([[0.9, -0.75, 1.2], [0, 0.9, -1.1], [0, 0, 0.7]])
		self.B = np.array([[0.03, 0, 0], [0, 0.06, 0], [0.07, 0, 0.05]])
		self.noise_stddev = 1e-4
		# System state
		self.state = np.zeros((3, 1))
		# State boundaries
		self.high_obs = np.array([10, 10, 10])
		# Input dimension
		self.inp_dim = 3

	def step(self, action):
		self.state = np.dot(self.A, self.state) + np.dot(self.B, action) + np.random.normal(0, self.noise_stddev, (3,1))
		return self.state
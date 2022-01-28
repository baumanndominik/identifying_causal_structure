import numpy as np
from util import rk4 

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

	def reset(self):
		self.state = np.zeros((3, 1))

'''
A single tank system
'''

class tank_system:

	def __init__(self,inp_dim=1,x1=15,x2=0,A=50,a=0.2,connect=0):
		self.x1,self.x2,self.A,self.a = x1,x2,A,a
		self.state = np.vstack((self.x1,self.x2))
		self.dt = 0.1
		self.g = 981
		self.inp_bound = np.array([[0],[60]])
		self.inp_dim = inp_dim
		for i in range(inp_dim - 1):
			self.inp_bound = np.hstack(self.inp_bound,np.array([[0],[60]]))
		self.connect = connect
		self.u = 0
		self.var = np.array([[0.0],[0.1]])

	def dynamics(self,u):
		self.x1, self.x2 = self.state[:,0]
		self.u = u
		self.x1, self.x2 = rk4(self.state.flatten(),self.cont_dynamics,self.dt).flatten()
		if self.x1 < 0:
			self.x1 = 0
		self.state = np.vstack((self.x1,self.x2)) + np.random.normal(np.zeros((2,1)),self.var)
		return self.state

	def cont_dynamics(self,x):
		x = x.flatten()
		if x[0] < 0:
			x[0] = 0
			if x[1] < 0:
				x[1] = 0
		ret1 = x[1] 
		ret2 = - (self.a/self.A)*np.sqrt(2*self.g*x[0]) + (0.5/self.A)*(np.sum(self.u))
		try:
			for tank in self.connect:
				ret2 += (tank.a/self.A)*np.sqrt(2*self.g*tank.x1)
		except:
			pass
		return np.array([[ret1,ret2]], dtype=float)

'''
System with multiple tanks
'''

class mult_tank_system:

	def __init__(self,num_tanks,connections,inp_dim,inp_mapping):
		self.num_tanks,self.connections,self.inp_mapping,self.inp_dim = num_tanks,connections,inp_mapping,inp_dim
		self.inp_bound = np.array([[0],[60]])
		for i in range(inp_dim - 1):
			self.inp_bound = np.hstack((self.inp_bound,np.array([[0],[60]])))
		self.state = np.zeros((2*num_tanks,1))
		self.tanks = []
		self.u = np.zeros(inp_dim)
		self.dt = 0.1
		self.var = np.zeros(2*num_tanks)
		self.var[1::2] = [x+0.001 for x in self.var[1::2]]
		self.var = self.var.reshape(-1,1)
		self.construct_mult_tank_system()

	def construct_mult_tank_system(self):
		for i in range(self.num_tanks):
			self.tanks = np.append(self.tanks,tank_system(inp_dim=len(self.inp_mapping[i])))
		for idx,el in enumerate(self.connections):
			self.tanks[idx].connect = self.tanks[el] 
			if len(el) > 0:
				self.tanks[idx].a = 0.242
			else:
				self.tanks[idx].a = 0.127
		for idx,tank in enumerate(self.tanks):
			self.state[2*idx:2*idx+2,0] = tank.state[:,0]

	def dynamics(self,u):
		self.u = u 
		for idx,tank in enumerate(self.tanks):
			tank.u = self.u[self.inp_mapping[idx]]
		self.state = rk4(self.state.flatten(),self.cont_dynamics,self.dt).T
		self.state += np.random.normal(np.zeros((2*self.num_tanks,1)),self.var)
		for idx,tank in enumerate(self.tanks):
			if self.state[2*idx,0] < 0:
				self.state[2*idx,0] = 0
			tank.state[:,0] = self.state[2*idx:2*idx+2,0]
			tank.x1, tank.x2 = tank.state 
		return self.state 

	def cont_dynamics(self,x):
		ret = np.zeros((1,2*self.num_tanks))
		for idx,tank in enumerate(self.tanks):
			x_tmp = self.state[2*idx:2*idx+2,0]
			ret[0,2*idx:2*idx+2] = tank.cont_dynamics(x_tmp)[0,:]
		return ret

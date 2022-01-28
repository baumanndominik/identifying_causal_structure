from util import mmd2
from examples import mult_tank_system
import numpy as np
import GPy
import copy
from multiprocessing import Pool, cpu_count
from itertools import repeat

# Excite the system to get data for system identification
def sys_id_data(sys,T=5000):
	u = np.random.uniform(sys.inp_bound[0,:],sys.inp_bound[1,:],(T,sys.inp_dim))
	x_st = np.zeros((len(sys.state),T))
	for idx,inp in enumerate(u):
		state = sys.dynamics(inp)
		x_st[:,idx] = state[:,0]
	return [x_st,u.T]

# Learn a GP model for the dynamical system
class GPmodel:
	def __init__(self,sys,num_out,id_data,indep_var,T=1000,model=0):
		self.sys = sys
		self.indep_var = indep_var
		self.num_out = num_out
		self.min_val = 0
		self.max_val = 0
		self.id_data = id_data
		if model:
			self.model = model
		else:
			self.model = self.ident_GP_model(T)
		self.min_val = np.min(data[0][num_out,:])
		self.max_val = np.max(data[0][num_out,:])

	def ident_GP_model(self,T):
		input_dim = len(self.sys.state) + self.sys.inp_dim 
		# Reduce input dimension in case of independent variables
		try:
			for el in self.indep_var:
				input_dim -= 1
		except:
			pass
		# unconstrained RBF kernel
		ker = GPy.kern.RBF(input_dim=input_dim,ARD=True)
		ker.unconstrain()
		x_st = self.id_data[0]
		u = self.id_data[1]
		Y = x_st[self.num_out,1::] - x_st[self.num_out,0:-1]
		X = np.vstack((x_st[:,0:-1],u[:,0:-1]))
		# Delete independent variables from input
		try:
			for el in self.indep_var:
				X = np.delete(X,el,0)
		except:
			pass
		# Fit and optimize GP model
		model = GPy.core.gp.GP(X.T,Y.reshape(-1,1),ker,GPy.likelihoods.Gaussian(),inference_method=GPy.inference.latent_function_inference.ExactGaussianInference(),normalizer=True)
		model.optimize()
		return model 	

# Predict MMD based on the GP model
def predict_mmd(GPs,sys,test_infl_of,test_infl_on,init_cond1,inp_traj1,init_cond2=0,inp_traj2=0,mont=0,num_exp=0,num_var=0):
	sys1 = sys
	sys2 = copy.deepcopy(sys)
	sys1.state[:,0] = init_cond1[:]
	# Check whether we have separate initial conditions for the two systems
	try:
		sys2.state[:,0] = init_cond2[:]
	except:
		sys2.state = copy.deepcopy(sys1.state)
		init_cond2 = init_cond1
	# Check whether we have separate input trajectories
	try:
		len(inp_traj2)
	except:
		inp_traj2 = inp_traj1
	# Create arrays depending on whether or not we want to estimate the variance
	if num_var:
		x_st = np.zeros((num_var*num_exp,len(sys1.state)*inp_traj1.shape[1]+len(sys1.state)))
		y_st = np.zeros((num_var*num_exp,len(sys1.state)*inp_traj1.shape[1]+len(sys1.state)))
		x_st[:,0:len(sys1.state)] = init_cond1[:] 
		y_st[:,0:len(sys1.state)] = init_cond2[:]  
	else:
		x_st = np.zeros((1,len(sys1.state)*inp_traj1.shape[1]+len(sys1.state)))
		y_st = np.zeros((1,len(sys1.state)*inp_traj1.shape[1]+len(sys1.state)))
		x_st[:,0:len(sys1.state)] = init_cond1[:]
		y_st[:,0:len(sys1.state)] = init_cond2[:]
	# Simulate system and store results
	for i in range(inp_traj1.shape[1]):
		if num_var:
			X = np.hstack((x_st[:,i*len(sys1.state):i*len(sys1.state)+len(sys1.state)],np.matlib.repmat(inp_traj1[:,i],num_var*num_exp,1)))
			Y = np.hstack((y_st[:,i*len(sys1.state):i*len(sys1.state)+len(sys1.state)],np.matlib.repmat(inp_traj2[:,i],num_var*num_exp,1)))
		else:
			X = np.hstack((x_st[:,i*len(sys1.state):i*len(sys1.state)+len(sys1.state)],inp_traj1[:,i].reshape(1, -1)))
			Y = np.hstack((y_st[:,i*len(sys1.state):i*len(sys1.state)+len(sys1.state)],inp_traj2[:,i].reshape(1, -1)))
		for idx,GP in enumerate(GPs):
			X_tmp = np.delete(X,GP.indep_var,1)
			Y_tmp = np.delete(Y,GP.indep_var,1)
			if num_var:
				x_st[:,(i+1)*len(sys1.state)+idx] = x_st[:,i*len(sys1.state)+idx] + np.random.normal(GP.model.predict(X_tmp)[0],(GP.model.predict(X_tmp)[1]))[:,0]
				y_st[:,(i+1)*len(sys1.state)+idx] = y_st[:,i*len(sys1.state)+idx] + np.random.normal(GP.model.predict(Y_tmp)[0],(GP.model.predict(Y_tmp)[1]))[:,0]
			else:
				x_st[:,(i+1)*len(sys1.state)+idx] = x_st[:,i*len(sys1.state)+idx] + GP.model.predict(X_tmp)[0]
				y_st[:,(i+1)*len(sys1.state)+idx] = y_st[:,i*len(sys1.state)+idx] + GP.model.predict(Y_tmp)[0]
	# Calculate MMD
	if num_var:
		mmd = np.zeros((num_var,len(test_infl_on)))
		for idx, i in enumerate(test_infl_on):
			if i == test_infl_of:
				x_st[:,i::len(sys1.state)] -= init_cond1[i]
				y_st[:,i::len(sys1.state)] -= init_cond2[i]
			for j in range(num_var):
				mmd[j,idx] = mmd2(x_st[num_exp*j:num_exp*j+num_exp,i::len(sys1.state)].flatten(),y_st[num_exp*j:num_exp*j+num_exp,i::len(sys1.state)].flatten())
		mmd = np.std(mmd,axis=0)
	else:
		mmd = np.zeros(len(test_infl_on))
		for idx, i in enumerate(test_infl_on):
			if i == test_infl_of:
				x_st[:,i::len(sys1.state)] -= init_cond1[i]
				y_st[:,i::len(sys1.state)] -= init_cond2[i]
			mmd[idx] = mmd2(x_st[:,i::len(sys1.state)].flatten(),y_st[:,i::len(sys1.state)].flatten())
	return mmd, x_st, y_st

# Get actual MMD
def real_mmd(sys,test_infl_of,test_infl_on,init_cond1,inp_traj1,init_cond2=0,inp_traj2=0,num_exp=1):
	sys1 = sys
	sys2 = copy.deepcopy(sys)
	for _ in range(num_exp):
		sys1.state = init_cond1.reshape(-1,1)
		try:
			sys2.state = init_cond2.reshape(-1,1)
		except:
			sys2.state = copy.deepcopy(sys1.state)
			init_cond2 = init_cond1
		try:
			len(inp_traj2)
		except:
			inp_traj2 = inp_traj1
		x_st = np.zeros((len(sys1.state),inp_traj1.shape[1]+1))
		y_st = np.zeros((len(sys1.state),inp_traj1.shape[1]+1))
		x_st[:,0] = init_cond1[:]
		y_st[:,0] = init_cond2[:]
		for i in range(inp_traj1.shape[1]):
			X = np.vstack((sys1.state,inp_traj1[:,i].reshape(-1,1)))
			Y = np.vstack((sys2.state,inp_traj2[:,i].reshape(-1,1)))
			sys1.dynamics(inp_traj1[:,i])
			sys2.dynamics(inp_traj2[:,i])
			x_st[:,i+1] = sys1.state[:,0]
			y_st[:,i+1] = sys2.state[:,0]
		try:
			x_mult = np.hstack((x_mult,x_st))
			y_mult = np.hstack((y_mult,y_st))
		except:
			x_mult = copy.deepcopy(x_st)
			y_mult = copy.deepcopy(y_st)
	mmd = np.zeros(len(test_infl_on))
	for idx, i in enumerate(test_infl_on):
		if i == test_infl_of:
			x_mult[i,:] -= init_cond1[i]
			y_mult[i,:] -= init_cond2[i]
		mmd[idx] = mmd2(x_mult[i,:],y_mult[i,:])
	return mmd, x_st, y_st

# Get the causal structure of the system
def check_struct(GPs,sys1,sys2,id_data,test_infl_of,test_infl_on,nu=10):
	init_cond1 = []
	init_cond2 = []
	GPs_test = GPs
	num_gps = list(np.linspace(0,len(sys1.state)-1,len(sys1.state)))
	num_gps = [int(x) for x in num_gps]
	# Estimate GPs with parallel computing
	with Pool(cpu_count()) as p:
	 		GPs_test = p.starmap(GPmodel, zip(repeat(sys1),num_gps,repeat(id_data),repeat([test_infl_of])))
	for i in range(len(sys1.state)):
		if i != test_infl_of:
			init_cond1 = np.append(init_cond1,np.random.normal(np.mean([GPs[i].min_val,GPs[i].max_val]), 1e-2))
			init_cond2 = np.append(init_cond2,np.random.normal(np.mean([GPs[i].min_val,GPs[i].max_val]), 1e-2))
		else:
			init_cond1 = np.append(init_cond1,np.random.normal(GPs[i].min_val, 1e-2))
			init_cond2 = np.append(init_cond2,np.random.normal(GPs[i].max_val, 1e-2))
	T = 100
	if test_infl_of < len(sys1.state):
		inp_traj = np.random.uniform(sys1.inp_bound[0,:],sys1.inp_bound[1,:],(T,sys1.inp_dim))
	else:
		test_inp = test_infl_of - len(sys1.state)
		inp_traj = np.random.uniform(sys1.inp_bound[0,:],sys1.inp_bound[1,:],(T,sys1.inp_dim))
		inp_traj2 = copy.deepcopy(inp_traj)
		inp_traj[:,test_inp] = sys1.inp_bound[0,test_inp]
		inp_traj2[:,test_inp] = sys1.inp_bound[1,test_inp]
	num_var = 50
	num_exp = 10
	pred_mmd_st = np.zeros((len(sys1.state),num_var))
	if test_infl_of < len(sys1.state):
		std_mmd, _, _ = predict_mmd(GPs_test,sys1,test_infl_of,test_infl_on,init_cond1,inp_traj.T,init_cond2=init_cond2,num_exp=num_exp,num_var=num_var)
		e_mmd, _, _ = predict_mmd(GPs_test,sys1,test_infl_of,test_infl_on,init_cond1,inp_traj.T,init_cond2=init_cond2)
		e_mmd_init, _, _ = predict_mmd(GPs,sys1,test_infl_of,test_infl_on,init_cond1,inp_traj.T,init_cond2=init_cond2)
		exp_mmd, real_x, real_y = real_mmd(sys2,test_infl_of,test_infl_on,init_cond1,inp_traj.T,init_cond2=init_cond2,num_exp=num_exp)
	else:
		std_mmd, _, _ = predict_mmd(GPs_test,sys1,test_infl_of,test_infl_on,init_cond1,inp_traj.T,inp_traj2=inp_traj2.T,num_exp=num_exp,num_var=num_var)
		e_mmd, _, _ = predict_mmd(GPs_test,sys1,test_infl_of,test_infl_on,init_cond1,inp_traj.T,inp_traj2=inp_traj2.T)
		e_mmd_init, _, _ = predict_mmd(GPs,sys1,test_infl_of,test_infl_on,init_cond1,inp_traj.T,inp_traj2=inp_traj2.T)
		exp_mmd, real_x, real_y = real_mmd(sys2,test_infl_of,test_infl_on,init_cond1,inp_traj.T,inp_traj2=inp_traj2.T,num_exp=num_exp)
	return exp_mmd > e_mmd + nu*std_mmd

def create_multi_tank_system():
	# Create quadruple tank system (4 tanks, 2 inputs)
	num_tanks = 4
	connections = np.array([[2],[3],[],[]],dtype=object)
	inp_dim = 2
	inp_mapping = np.array([[0],[1],[1],[0]])
	multi_tank = mult_tank_system(num_tanks,connections,inp_dim,inp_mapping)
	return multi_tank

if __name__ == '__main__':
	sys1 = create_multi_tank_system()
	sys2 = create_multi_tank_system()
	data = sys_id_data(sys1)
	num_gps = list(np.linspace(0,len(sys1.state)-1,len(sys1.state)))
	num_gps = [int(x) for x in num_gps]
	with Pool(len(sys1.state)) as p:
		GPs = p.starmap(GPmodel, zip(repeat(sys1),num_gps,repeat(data),repeat([])))
	model_arr = []
	for i in range(len(sys1.state)):
		model_arr = np.append(model_arr,GPs[i].model) 
	num_tests = len(sys1.state) + sys1.inp_dim
	test_infl_of = [0,2,4,6,8,9]
	test_infl_on = [0,2,4,6]
	caus = np.zeros((len(test_infl_on), len(test_infl_of)))
	for idx1, el in enumerate(test_infl_of):
		indep_el = check_struct(GPs,sys1,sys2,data,el,test_infl_on)
		for idx2, ind in enumerate(indep_el):
			caus[idx2, idx1] = ind
		print("new causality matrix:")
		print(caus)

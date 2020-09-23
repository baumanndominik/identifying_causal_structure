import numpy as np 
import numpy.matlib as matlib
from scipy.signal import chirp
import copy
from util import mmd2, set_point_ctrl
from examples import synthetic_example 

# Excite the system to get data for system identification
def create_sys_id_data(sys, num_samples=100000):
	# Initialize input trajectory
	inp_traj = np.zeros((sys.inp_dim, num_samples))
	# Use chirp signal as input trajectory
	t = np.linspace(0,num_samples,num_samples)
	for i in range(sys.inp_dim):
		inp_traj[i,:] = chirp(t, 0.5*i, t[-1], 1*i)
	# Array to store state trajectory
	state_st = np.zeros((3, num_samples + 1))
	# Run experiment
	for i in range(num_samples):
		state_st[:, i + 1] = sys.step(inp_traj[:, i].reshape(-1, 1)).flatten()
	sys.state = np.zeros((3, 1))
	return state_st, inp_traj

# Identify the system dynamics (standard system identification via least squares)
def sys_id(state, inp_traj):
	data = np.vstack((state[:, 0:-2],inp_traj[:, 0:-1]))
	res = np.dot(state[:, 1:-1],np.linalg.pinv(data))
	A = res[0:len(state),0:len(state)]
	B = res[0:len(state),len(state)::]
	noise_stddev = np.zeros((len(A[:,0]),1))
	for i in range (len(A[:,0])):
		noise_stddev[i,0] = np.sqrt(np.sum((state[i,1:-1]-np.dot(res[i,:],data))**2)/(len(state[0,:])-1))
	return [A, B, noise_stddev]

# Identify local system dynamics under non-causality assumption
def sys_id_loc(state, inp_traj, test_infl_of):
	data = np.vstack((state[:, 0:-2],inp_traj[:, 0:-1]))
	# We assume that variable 'test_infl_of' does not cause x_i
	# Thus, we delete this data before estimating the model
	data_tmp = np.delete(data,test_infl_of,0)
	res = np.dot(state[:,1:-1],np.linalg.pinv(data_tmp))
	res = np.insert(res,test_infl_of,0,axis=1)
	A = res[0:len(state),0:len(state)]
	B = res[0:len(state),len(state)::]
	noise_stddev = np.zeros((len(A[:,0]),1))
	for i in range (len(A[:,0])):
		noise_stddev[i,0] = np.sqrt(np.sum((state[i, 1:-1]-np.dot(res[i, :],data))**2)/(len(state[0, :])-1))
	return [A, B, noise_stddev]

# Monte Carlo simulation of a given system with given initial state and input trajectories
def simulate_system(model, x0, u, num_exp=1000):
	A, B, noise = model 
	sys_dim = len(A[:,0])
	noise_arr = np.random.normal(0, matlib.repmat(noise, num_exp, 1), (sys_dim*num_exp, len(u[0,:])))
	# One simulation with zero noise to get the mean
	noise_arr[0:sys_dim,:] = 0
	x_st = np.zeros((sys_dim*num_exp, len(u[0, :]) + 1))
	x_st[:,0] = matlib.repmat(x0, num_exp, 1).flatten()
	# Extend matrices for parallel computing
	A_ext = np.kron(np.eye(num_exp), A)
	B_ext = np.kron(np.eye(num_exp), B)
	u_ext = matlib.repmat(u, num_exp, 1)
	for i in range(len(u[0, :])):
		x_st[:, i + 1] = np.dot(A_ext, x_st[:, i]) + np.dot(B_ext, u_ext[:, i]) + noise_arr[:, i]
	return x_st 

# Get the test statistic of a system with initial conditions and input trajectories
def get_test_statistic(model, x0_I, x0_II, u_I, u_II, num_exp=1000, nu=1):
	sys_dim = len(model[0][:,0])
	data_I = simulate_system(model, x0_I, u_I, num_exp=num_exp)
	data_II = simulate_system(model, x0_II, u_II, num_exp=num_exp)
	mmd_st = np.zeros((sys_dim, num_exp))
	mmd = np.zeros(sys_dim)
	for test_infl_on in range(sys_dim):
		for i in range(num_exp):
			mmd_st[test_infl_on, i] = mmd2(data_I[sys_dim*i + test_infl_on, 1::], data_II[sys_dim*i + test_infl_on, 1::])
		mmd[test_infl_on] = mmd_st[test_infl_on, 0] + nu*np.std(mmd_st[test_infl_on, 1::])
	return mmd

# Steer the system to the initial conditions for the next experiment
def go_to_init(model, sys, ctrl, x_init, tolerance=1e-2):
	int_state = np.zeros((len(sys.high_obs), 1))
	F, Mx, Mu = ctrl
	while True:
		action = np.dot(Mu - np.dot(F, Mx), x_init) + np.dot(F, sys.state)
		sys.step(action).flatten()
		if np.linalg.norm(sys.state - x_init) < tolerance:
			print("reached initial position for experiment")
			return

# Execute a causality testing experiment
def caus_exp(model, sys, x0_I, x0_II, u_I, u_II, ctrl):
	num_samples = len(u_I[0, :])
	xI_st = np.zeros((len(sys.high_obs), num_samples + 1))
	xII_st = copy.deepcopy(xI_st)
	# Go to initial position for first experiment
	go_to_init(model, sys, ctrl, x0_I)
	xI_st[:, 0] = sys.state.flatten() 
	# Start first experiment
	print("start first experiment")
	for i in range(num_samples):
		xI_st[:, i + 1] = sys.step(u_I[:,i].reshape(-1, 1)).flatten()
	# Go to initial position for second experiment
	go_to_init(model, sys, ctrl, x0_II)
	xII_st[:, 0] = sys.state.flatten() 
	# Start second experiment
	print("start second experiment")
	for i in range(num_samples):
		xII_st[:, i + 1] = sys.step(u_II[:,i].reshape(-1, 1)).flatten()
	return xI_st, xII_st

def caus_id():
	sys = synthetic_example()
	sys_dim = len(sys.high_obs)
	inp_dim = sys.inp_dim
	# Null hypothesis: no state/input has a causal influence on any state
	caus = np.zeros((sys_dim, sys_dim + inp_dim))
	# Start with standard system identification
	sys_id_state, sys_id_inp = create_sys_id_data(sys)
	init_model = sys_id(sys_id_state, sys_id_inp)
	# Get a controller based on the initial model
	ctrl = set_point_ctrl(init_model[0], init_model[1], np.diag([1,1,1]), np.diag([0.01, 0.01, 0.01]))
	# Start causal identification
	for test_infl_of in range(sys_dim + inp_dim):
		print("new causality test")
		if test_infl_of < sys_dim:
			print("testing influence of state ", test_infl_of)
			# Choose initial conditions as far apart as possible
			x0_I = np.zeros((sys_dim, 1))
			x0_I[test_infl_of, 0] = sys.high_obs[test_infl_of]
			x0_II = np.zeros((sys_dim, 1))
			x0_II[test_infl_of, 0] = -sys.high_obs[test_infl_of]
			# Choose input trajectory that excites the system
			u_I = np.random.uniform(-1, 1, (inp_dim, 100))
			u_II = u_I 
		else:
			test_inp = test_infl_of - sys_dim
			print("testing influence of input ", test_inp)
			# Choose initial position in 0
			x0_I = np.zeros((sys_dim, 1))
			x0_II = x0_I 
			# Choose different input trajectories that excite the system
			u_I = 10*np.random.uniform(-1, 1, (inp_dim, 100))
			u_II = copy.deepcopy(u_I)
			u_I[test_inp, :] = 100*np.random.uniform(-1, 1, 100)
			u_II[test_inp, :] = np.zeros(100)
		# Do causality experiment
		exp_data_I, exp_data_II = caus_exp(init_model, sys, x0_I, x0_II, u_I, u_II, ctrl)
		# Get model assuming variables are non-causal
		caus_model = sys_id_loc(sys_id_state, sys_id_inp, test_infl_of)
		# Get test statistic
		test_stat = get_test_statistic(caus_model, exp_data_I[:, 0].reshape(-1, 1), exp_data_II[:, 0].reshape(-1, 1), u_I, u_II, nu=3)
		print("Obtained test statistic")
		# Compute MMD and compare with test statistic
		for test_infl_on in range(sys_dim):
			if test_infl_of == test_infl_on:
				exp_data_I[test_infl_on, :] -= x0_I[test_infl_on, 0]
				exp_data_II[test_infl_on, :] -= x0_II[test_infl_on, 0]
			mmd_exp = mmd2(exp_data_I[test_infl_on, :], exp_data_II[test_infl_on, :])
			if mmd_exp > test_stat[test_infl_on]:
				caus[test_infl_on, test_infl_of] = 1
		print("new causality matrix:")
		print(caus)
		sys.state = np.zeros((3,1))

if __name__ == '__main__':
	caus_id()

import numpy as np 
import numpy.matlib as matlib
import scipy

# The radial basis function (RBF) kernel
def rbf(x,y,sigma=1.0):
	G = np.multiply(x,x)
	H = np.multiply(y,y)
	Q = matlib.repmat(G,len(y),1)
	R = matlib.repmat(H,len(x),1)
	H = Q + R.T - 2*x*y.reshape(-1, 1)
	return np.exp(-H/(2*sigma**2))

# Calculate the maximum mean discrepancy
def mmd2(x,y):
	max_val = np.max(np.append(np.abs(x),np.abs(y)))
	if max_val > 0:
		x = x/max_val
		y = y/max_val
	num_samples = float(np.min([len(x),len(y)]))
	K = rbf(x,x)
	L = rbf(y,y)
	KL = rbf(x,y)
	np.fill_diagonal(K,0)
	np.fill_diagonal(L,0)
	np.fill_diagonal(KL,0)
	return (1/(num_samples*(num_samples-1)))*(np.sum(K+L-KL-KL.T))

# Standard discrete time linear quadratic regulator (LQR)
def dlqr(A,B,Q,R):
	#first, try to solve the ricatti equation
	X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
	   
	#compute the LQR gain
	F = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
		 
	return -F

# Set-point controller
def set_point_ctrl(A, B, Q, R):
	F = dlqr(A, B, Q, R)
	sys_mat = np.block([[A - np.eye(A.shape[0]), B], [np.eye(A.shape[0]), np.zeros_like(B)]])
	res = np.dot(np.linalg.pinv(sys_mat), np.block([[np.zeros_like(A)], [np.eye(A.shape[0])]]))
	Mx = res[0:A.shape[0], :]
	Mu = res[A.shape[0]::, :]
	return [F, Mx, Mu]
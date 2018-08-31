import numpy as np

def least_squares_loss(W, data, cov, d, n):
	return (1/(2*n))*(np.linalg.norm(data - np.matmul(data, W), ord='fro')**2)

def least_squares_loss_grad(W, data, cov, d, n):
	return (-1.0/n)*np.matmul(data.T, data - np.matmul(data, W))

def least_squares_loss_cov(W, data, cov, d, n):
	return (1.0/2*n)*np.trace(np.matmul(np.matmul((np.identity(d) - W).T, cov.T), np.identity(d) - W))

def least_squares_loss_cov_grad(W, data, cov, d, n):
	return (-1.0/2*n)*(np.matmul(cov + cov.T, np.identity(d) - W))
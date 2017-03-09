import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  p = np.zeros_like(X)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  D = W.shape[0]
  A = np.exp(X.dot(W))
  Y = np.zeros((N, C))
  loss = 0.0
  for i in xrange(N):
  	Y[i] = A[i] / A[i].sum()
  	loss -= np.log(Y[i][y[i]])
  loss /= N
  loss += 0.5 * reg * (W ** 2).sum()
  for i in xrange(D):
  	for j in xrange(C):
  		for k in xrange(N):
  			if (y[k] == j):
  				dW[i][j] += (Y[k][j] - 1) * X[k][i]
  			else:
  				dW[i][j] += Y[k][j] * X[k][i]
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  A = np.exp(X.dot(W))
  Y = np.zeros((N, C))
  loss = 0.0
  Y = A / A.sum(axis = 1).reshape(-1, 1)
  loss -= np.log(Y[range(N), y]).sum()
  loss /= N
  loss += 0.5 * reg * (W ** 2).sum()
  Y[range(N), y] -= 1
  dW = np.transpose(X).dot(Y) / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
 

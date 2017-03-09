import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def conv_bn_relu_conv_bn_relu_pool_forward(x, w1, b1, w2, b2, conv_param, pool_param, gamma1, beta1, gamma2, beta2, bn_param1, bn_param2):
  out, conv_cache1 = conv_forward_fast(x, w1, b1, conv_param)
  out, bn_cache1 = spatial_batchnorm_forward(out, gamma1, beta1, bn_param1)
  out, relu_cache1 = relu_forward(out)
  out, conv_cache2 = conv_forward_fast(out, w2, b2, conv_param)
  out, bn_cache2 = spatial_batchnorm_forward(out, gamma2, beta2, bn_param2)
  out, relu_cache2 = relu_forward(out)
  out, pool_cache = max_pool_forward_fast(out, pool_param)
  cache = (conv_cache1, bn_cache1, relu_cache1, conv_cache2, bn_cache2, relu_cache2, pool_cache)
  return out, cache


def conv_bn_relu_conv_bn_relu_pool_backward(dout, cache):
  conv_cache1, bn_cache1, relu_cache1, conv_cache2, bn_cache2, relu_cache2, pool_cache = cache
  dx = max_pool_backward_fast(dout, pool_cache)
  dx = relu_backward(dx, relu_cache2)
  dx, dgamma2, dbeta2 = spatial_batchnorm_backward(dx, bn_cache2)
  dx, dw2, db2 = conv_backward_fast(dx, conv_cache2)
  dx = relu_backward(dx, relu_cache1)
  dx, dgamma1, dbeta1 = spatial_batchnorm_backward(dx, bn_cache1)
  dx, dw1, db1 = conv_backward_fast(dx, conv_cache1)
  return dx, dw1, db1, dw2, db2, dgamma1, dbeta1, dgamma2, dbeta2

def affine_bn_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
  out, fc_cache = affine_forward(x, w, b)
  out, bn_cache =  batchnorm_forward(out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)
  out, dropout_cache = dropout_forward(out, dropout_param)
  cache = (fc_cache, bn_cache, relu_cache, dropout_cache)
  return out, cache


def affine_bn_relu_dropout_backward(dout, cache):
  fc_cache, bn_cache, relu_cache, dropout_cache = cache
  dx = dropout_backward(dout, dropout_cache)
  dx = relu_backward(dx, relu_cache)
  dx, dgamma, dbeta = batchnorm_backward_alt(dx, bn_cache)
  dx, dw, db = affine_backward(dx, fc_cache)
  return dx, dw, db, dgamma, dbeta

class MyConvNet(object):
  """
  [conv-relu-pool] * N - conv - relu - [affine] * M - [softmax or SVM]
  """
  
  def __init__(self, num_filters, affine_dims, filter_size, input_dim=(3, 32, 32), num_classes=10,
               weight_scale=1e-3, reg=0.0, dtype=np.float32, dropout=0):
    self.params = {}
    self.bn_params = {}
    self.reg = reg
    self.dtype = dtype
    
    self.L1 = len(num_filters)
    self.L2 = len(affine_dims)
    L1, L2 = self.L1, self.L2
    channel = input_dim[0]
    drop = 1
    for i in xrange(L1):
      self.params['W1_1_' + str(i)] = weight_scale * np.random.randn(num_filters[i], channel, filter_size[i], filter_size[i])
      self.params['b1_1_' + str(i)] = np.zeros(num_filters[i])
      self.params['gamma1_1_' + str(i)] = np.ones(num_filters[i])
      self.params['beta1_1_' + str(i)] = np.zeros(num_filters[i])
      channel = num_filters[i]
      self.params['W1_2_' + str(i)] = weight_scale * np.random.randn(num_filters[i], channel, filter_size[i], filter_size[i])
      self.params['b1_2_' + str(i)] = np.zeros(num_filters[i])
      self.params['gamma1_2_' + str(i)] = np.ones(num_filters[i])
      self.params['beta1_2_' + str(i)] = np.zeros(num_filters[i])
      channel = num_filters[i]
      drop *= 2
    dim = (input_dim[1] / drop) * (input_dim[2] / drop) * num_filters[L1 - 1]
    for i in xrange(L2):
      self.params['W2_' + str(i)] = weight_scale * np.random.randn(dim, affine_dims[i])
      self.params['b2_' + str(i)] = np.zeros(affine_dims[i])
      self.params['gamma2_' + str(i)] = np.ones(affine_dims[i])
      self.params['beta2_' + str(i)] = np.zeros(affine_dims[i])
      dim = affine_dims[i]
    self.params['W3'] = weight_scale * np.random.randn(affine_dims[L2 - 1], num_classes)
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    for i in xrange(L1):
      self.bn_params['1_1_' + str(i)] = {'mode': 'train'}
      self.bn_params['1_2_' + str(i)] = {'mode': 'train'}
    for i in xrange(L2):
      self.bn_params['2_' + str(i)] = {'mode': 'train'}
    
    self.dropout_param = {'mode': 'train', 'p': dropout}

  def loss(self, X, y=None):
  
    L1 = self.L1
    L2 = self.L2
    mode = 'test' if y is None else 'train'
    for i in xrange(L1):
      self.bn_params['1_1_' + str(i)]['mode'] = mode
      self.bn_params['1_2_' + str(i)]['mode'] = mode
    for i in xrange(L2):
      self.bn_params['2_' + str(i)]['mode'] = mode
    self.dropout_param['mode'] = mode
    
    # forward pass
    
    out = X
    cache1 = [0] * L1
    reg = self.reg
    
    for i in xrange(L1):
      W1, b1 = self.params['W1_1_' + str(i)], self.params['b1_1_' + str(i)]
      gamma1, beta1 = self.params['gamma1_1_' + str(i)], self.params['beta1_1_' + str(i)]
      W2, b2 = self.params['W1_2_' + str(i)], self.params['b1_2_' + str(i)]
      gamma2, beta2 = self.params['gamma1_2_' + str(i)], self.params['beta1_2_' + str(i)]
      filter_size = W1.shape[2]
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
      out, cache1[i] = conv_bn_relu_conv_bn_relu_pool_forward(out, W1, b1, W2, b2, conv_param, pool_param, gamma1, beta1, gamma2, beta2, self.bn_params['1_1_' + str(i)], self.bn_params['1_2_' + str(i)])

    cache2 = [0] * L2
    
    for i in xrange(L2):
      W, b = self.params['W2_' + str(i)], self.params['b2_' + str(i)]
      gamma, beta = self.params['gamma2_' + str(i)], self.params['beta2_' + str(i)]
      out, cache2[i] = affine_bn_relu_dropout_forward(out, W, b, gamma, beta, self.bn_params['2_' + str(i)], self.dropout_param)
    
    W, b = self.params['W3'], self.params['b3']
    scores, cache3 = affine_forward(out, W, b)
    
    if y is None:
      return scores
    
    # backward pass
    
    grads = {}
    loss, dout = softmax_loss(scores, y)
    for i in xrange(L1):
      loss += 0.5 * reg * (self.params['W1_1_' + str(i)] ** 2).sum()
      loss += 0.5 * reg * (self.params['W1_2_' + str(i)] ** 2).sum()
    for i in xrange(L2):
      loss += 0.5 * reg * (self.params['W2_' + str(i)] ** 2).sum()
    loss += 0.5 * reg * (self.params['W3'] ** 2).sum()
    
    dout, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
    
    for i in xrange(L2 - 1, -1, -1):
      dout, grads['W2_' + str(i)], grads['b2_' + str(i)], grads['gamma2_' + str(i)], grads['beta2_' + str(i)] = affine_bn_relu_dropout_backward(dout, cache2[i])
      grads['W2_' + str(i)] += reg * self.params['W2_' + str(i)]
    for i in xrange(L1 - 1, -1, -1):
      dout, grads['W1_1_' + str(i)], grads['b1_1_' + str(i)], grads['W1_2_' + str(i)], grads['b1_2_' + str(i)], grads['gamma1_1_' + str(i)], grads['beta1_1_' + str(i)] , grads['gamma1_2_' + str(i)], grads['beta1_2_' + str(i)] = conv_bn_relu_conv_bn_relu_pool_backward(dout, cache1[i])
      grads['W1_1_' + str(i)] += reg * self.params['W1_1_' + str(i)]
      grads['W1_2_' + str(i)] += reg * self.params['W1_2_' + str(i)]
    
    return loss, grads

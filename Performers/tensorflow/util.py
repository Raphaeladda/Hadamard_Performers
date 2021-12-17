# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based einsum layer.
Copied from
https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/dense_einsum.py.
"""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf

_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]


@tf.keras.utils.register_keras_serializable(package="Text")
class DenseEinsum(tf.keras.layers.Layer):
  """A densely connected layer that uses tf.einsum as the backing computation.
  This layer can perform einsum calculations of arbitrary dimensionality.
  Arguments:
    output_shape: Positive integer or tuple, dimensionality of the output space.
    num_summed_dimensions: The number of dimensions to sum over. Standard 2D
      matmul should use 1, 3D matmul should use 2, and so forth.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common
      situation would be a 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D
      input with shape `(batch_size, input_dim)`, the output would have shape
      `(batch_size, units)`.
  """

  def __init__(self,
               output_shape,
               num_summed_dimensions=1,
               activation=None,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DenseEinsum, self).__init__(**kwargs)
    self._output_shape = output_shape if isinstance(
        output_shape, (list, tuple)) else (output_shape,)
    self._activation = tf.keras.activations.get(activation)
    self._use_bias = use_bias
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._num_summed_dimensions = num_summed_dimensions
    self._einsum_string = None

  def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
    input_str = ""
    kernel_str = ""
    output_str = ""
    letter_offset = 0
    for i in range(free_input_dims):
      char = _CHR_IDX[i + letter_offset]
      input_str += char
      output_str += char

    letter_offset += free_input_dims
    for i in range(bound_dims):
      char = _CHR_IDX[i + letter_offset]
      input_str += char
      kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
      char = _CHR_IDX[i + letter_offset]
      kernel_str += char
      output_str += char

    return input_str + "," + kernel_str + "->" + output_str

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_rank = input_shape.rank
    free_input_dims = input_rank - self._num_summed_dimensions
    output_dims = len(self._output_shape)

    self._einsum_string = self._build_einsum_string(free_input_dims,
                                                    self._num_summed_dimensions,
                                                    output_dims)

    # This is only saved for testing purposes.
    self._kernel_shape = (
        input_shape[free_input_dims:].concatenate(self._output_shape))

    self._kernel = self.add_weight(
        "kernel",
        shape=self._kernel_shape,
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        constraint=self._kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self._use_bias:
      self._bias = self.add_weight(
          "bias",
          shape=self._output_shape,
          initializer=self._bias_initializer,
          regularizer=self._bias_regularizer,
          constraint=self._bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self._bias = None
    super(DenseEinsum, self).build(input_shape)

  def get_config(self):
    config = {
        "output_shape":
            self._output_shape,
        "num_summed_dimensions":
            self._num_summed_dimensions,
        "activation":
            tf.keras.activations.serialize(self._activation),
        "use_bias":
            self._use_bias,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint)
    }
    base_config = super(DenseEinsum, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    ret = tf.einsum(self._einsum_string, inputs, self._kernel)
    if self._use_bias:
      ret += self._bias
    if self._activation is not None:
      ret = self._activation(ret)
    return ret





def isPowerOfTwo(x):
 
    # First x in the below expression
    # is for the case when x is 0
    return (x and (not(x & (x - 1))) )

def isOdd(integer):
    #assert isinstance(integer, int)
    return integer % 2 == 1

def isEven(integer):
    #assert isinstance(integer, int)
    return integer % 2 == 0

def _list_to_string(li):
    return ''.join(map(str, li))

class GrayCode(object):
    def __init__(self, nbits):
        self._nbits = nbits
        self._grayCode = []
        self.__generate()

    def __getitem__(self, i):
        return self._grayCode[i]

    def __str__(self):
        return str(self._grayCode)

    __repr__ = __str__

    def __iter__(self):            
        return self._grayCode.__iter__()

    def __generate(self):
        li = [0 for i in range(self._nbits)]
        self._grayCode.append(_list_to_string(li))

        for term in range(2, (1<<self._nbits)+1):
            if isOdd(term):                
                for i in range(-1,-(self._nbits),-1):
                    if li[i]==1:                        
                        li[i-1]=li[i-1]^1                        
                        break
                    
            if isEven(term):
                li[-1]=li[-1]^1

            self._grayCode.append(_list_to_string(li))

class GrayCodeIterator(object):
    def __init__(self, nbits):
        self._nbits = nbits

    def __iter__(self):
        li = [0 for i in range(self._nbits)]
        yield _list_to_string(li)

        for term in range(2, (1<<self._nbits)+1):
            if isOdd(term):                
                for i in range(-1,-(self._nbits),-1):
                    if li[i]==1:                        
                        li[i-1]=li[i-1]^1                        
                        break
                    
            if isEven(term):
                li[-1]=li[-1]^1

            yield _list_to_string(li)

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:21:43 2015
Fast Walsh-Hadamard Transform with Sequency Order
Author: Ding Luo@Fraunhofer IOSB
"""
from math import log
import numpy as np

def get_sequency_list(inputArray):
    """ Sort input 1D array into sequency order
    Utilizes gray code generation from a Python recipe from Internet.
    """
    length = inputArray.size
    bitlength = int(log(length,2))
    # Gray Code
    graycodes=GrayCode(bitlength)
    # Bitreverse of gray code
    bitreverse = [int(graycodes[i][::-1],2) for i in range(length)]
    
    outputArray = inputArray.copy()
    outputArray[bitreverse] = inputArray[:]

    return outputArray

def FWHT_vector(x):
    """ Fast Walsh-Hadamard Transform
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3 algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications of Walsh and Related Functions.
    """
    x = x.squeeze()
    N = x.size
    G = N//2 # Number of Groups
    M = 2 # Number of Members in Each Group

    # First stage
    y = np.zeros((N//2,2))
    y[:,0] = x[0::2] + x[1::2]
    y[:,1] = x[0::2] - x[1::2]
    x = y.copy()
    # Second and further stage
    for nStage in range(2,int(log(N,2))+1):
        y = np.zeros((G//2,M*2))
        y[0:G//2,0:M*2:4] = x[0:G:2,0:M:2] + x[1:G:2,0:M:2]
        y[0:G//2,1:M*2:4] = x[0:G:2,0:M:2] - x[1:G:2,0:M:2]
        y[0:G//2,2:M*2:4] = x[0:G:2,1:M:2] - x[1:G:2,1:M:2]
        y[0:G//2,3:M*2:4] = x[0:G:2,1:M:2] + x[1:G:2,1:M:2]
        x = y.copy()
        G = G//2
        M = M*2
    x = y[0,:]
    x = x.reshape((x.size,1))
    return x/np.sqrt(N)

def FWHT_matrix(M):

  assert M.shape[0] == M.shape[1], "M is a square matrix"
  assert isPowerOfTwo(M.shape[0]), "Matrix dimension must be a power of two"

  n = M.shape[0]
  FWHT = np.zeros((n,n))
  for k in range(n):
    L = M.T[k]
    FWHT.T[k] = FWHT_vector(L).flatten()
  return(FWHT)

def FWHT_matrix_bis(M):

  assert M.shape[0] == M.shape[1], "M is a square matrix"
  assert isPowerOfTwo(M.shape[0]), "Matrix dimension must be a power of two"
  FWHT = np.apply_along_axis(FWHT_vector, 1, M)
  return(FWHT.reshape((FWHT.shape[:-1])))


def Orthogonal_Hadamard(m,d,k=3):

  def Rademacher(n):
    return(np.diag(np.random.choice([-1,1],n)))
    
  def Orthogonal_Hadamard_square(d,k=3):
    SD = np.zeros((d,d))
    for i in range(k):
      D = Rademacher(d)
      SD += FWHT_matrix(D)
    return(SD)
  num_squares = int(m / d)
  blocks = [Orthogonal_Hadamard_square(d,k) for _ in range(num_squares)]

  remainder = m - d * num_squares
  if remainder:
      blocks.append(Orthogonal_Hadamard_square(d,k)[:remainder])

  matrix = np.vstack(blocks)
  matrix /= np.sqrt(num_squares + remainder / d)
  # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix
  return tf.convert_to_tensor(matrix, dtype=tf.float32) 
  
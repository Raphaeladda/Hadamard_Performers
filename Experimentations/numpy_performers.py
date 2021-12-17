import numpy as np
import matplotlib.pyplot as plt
from Fast_Walsh_Hadamard_Transform import *

"""
Implementation of regular attention and FAVOR+ with numpy
Inspired from:
Author: Teddy Koker
Source: https://teddykoker.com/2020/11/performers/#choromanski2020rethinking
"""

# Vanila Transformer attention implementation
def att(q, k, v, normalize=True):
    l, d = q.shape
    normalizer = 1 / (d ** 0.5) if normalize else 1
    a = np.exp(q @ k.T * normalizer)
    d_inv = np.diag(1 / (a @ np.ones(l)))
    return d_inv @ a @ v


# Perfomer attention implementation using some random feature map phi
def att_hat(q, k, v, phi, normalize=True):
    l, d = q.shape
    normalizer = 1 / (d ** 0.25)
    q_prime = phi(q * normalizer)
    k_prime = phi(k * normalizer)
    d_inv = np.diag(1 / (q_prime @ (k_prime.T @ np.ones(l))))
    return d_inv @ (q_prime @ (k_prime.T @ v))


# random feature map
def phi(h, fs, random_feats,m):
    return lambda x: (
        h(x)
        / np.sqrt(m)
        * np.concatenate(
            [f(np.einsum("...d,md->...m", x, random_feats)) for f in fs],
            axis=-1,
        )
    )

# Performer "sin/cos" attention
def sincos_att_hat(q, k, v, random_feats,m, normalize=True):
    def h(x):
        return np.exp(np.square(x).sum(axis=-1, keepdims=True) / 2)

    sin = lambda x: np.sin(2 * np.pi * x)
    cos = lambda x: np.cos(2 * np.pi * x)

    kernel = phi(h, [sin, cos], random_feats,m)
    return att_hat(q, k, v, kernel, normalize)


# Performer "positive" attention
def positive_att_hat(q, k, v, random_feats, m, normalize=True):
    def h(x):
        return np.exp(-np.square(x).sum(axis=-1, keepdims=True) / 2)

    kernel = phi(h, [np.exp], random_feats,m)
    return att_hat(q, k, v, kernel, normalize)


# generate IID Gaussian random features
def iid_gaussian(m, d):
    return np.random.normal(size=(m, d))

# generate IID Rademacher random features
def Rademacher(n):
  return(np.diag(np.random.choice([-1,1],n)))


def orthogonal_gaussian(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = np.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = np.vstack(blocks)
    matrix /= np.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix
    return(matrix)


def Orthogonal_Hadamard(m,d,k=3):
  def Orthogonal_Hadamard_square(d,k=3):
      # create random Hadamard orthogonal square matrix using Fast Walsh Hadamard transform process
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
  return matrix 

# function to find the next larger power of two
def next_larger_power_of_two(d):
  if isPowerOfTwo(d):
    return(d)
  return(2**(int(np.log2(int(d)) + 1)))
next_larger_power_of_two(16)

# mean squared error
def mse(a, b):
    return np.square(a - b).mean()

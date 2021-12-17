# Fast Walh Hadamard Transform

"""
Fast Walsh-Hadamard Transform with Sequency Order
Inspired from:
Author: Ding Luo@Fraunhofer IOSB
Source: https://github.com/dingluo/fwht/blob/master/FWHT.py
"""
from math import log
import numpy as np


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
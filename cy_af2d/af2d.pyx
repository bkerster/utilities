#Cython code to increase performance for larger planes

#translation of the 2d allan factor from matlab

#from the original matlab, courtesy Theo Rhodes
# AF2D performs Allan factor analysis on a 2D point process expressed as a sequence of counts.
#
# Example: [myafv,myafb] = af2d(myData) 
#
# Input: 
#    inData - m x n matrix containing count values at x and y coordinates
#             (can use the imread function to use images, for example)
#
# Output:
#    allanFactor - Allan factor variance for increasing box sizes
#    boxSeries   - box size series
# 
# To calculate alpha, plot log(boxSeries .^ 2) vs. log(allanFactor) and estimate slope.
# Currently box sizes have sides of length 1,2,4,9 ... 2^N with a max side length of the
# smallest inData dimension divided by four.
#
# Theo Rhodes (trhodes3@ucmerced.edu)
# 8/14/2011

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.int
DTYPEF = np.float
ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPEF_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
def sum2d(int[:, :] arr):    
    cdef int N = arr.shape[0]
    cdef int K = arr.shape[1]
    cdef int x = 0
    cdef int i, j
    for i in xrange(0,N):
        for j in xrange(0,K):
            x += arr[i, j]
    return x

@cython.boundscheck(False) # turn of bounds-checking for entire function    
def sum(int[:] y):    
    cdef int N = y.shape[0]
    cdef int x = y[0]
    cdef int i
    for i in xrange(1,N):
        x += y[i]
    return x

@cython.boundscheck(False) # turn of bounds-checking for entire function    
def min(int[:] x):
    cdef int N = x.shape[0]
    cdef int low = x[0]
    cdef int i
    for i in xrange(0, N):
        if x[i] < low:
            low = x[i]
    return low

@cython.boundscheck(False) # turn of bounds-checking for entire function
def af2d(np.ndarray[DTYPE_t, ndim=2] inData):
    """ 
        (af_variance, af_boxes) = af2d(data)
        Input: m x n matrix containing count values at x and y cooridinates 
        Output:
            allanFactor - Allan factor variance for increasing box sizes
            boxSeries   - box size series

        To calculate alpha, plot log(boxSeries .^ 2) vs. log(allanFactor) and estimate slope.
        Currently box sizes have sides of length 1,2,4,9 ... 2^N with a max side length of the
        smallest inData dimension divided by four.
    """
    cdef int xSize = inData.shape[0]
    cdef int ySize = inData.shape[1]
    cdef int maxBoxSize = min(np.array([xSize, ySize])) // 4
    cdef int numBoxes = int(np.log2(maxBoxSize)) + 1
    
    cdef np.ndarray[DTYPE_t, ndim=1] xTotal = np.zeros(numBoxes, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yTotal = np.zeros(numBoxes, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] allTotal = np.zeros(numBoxes, dtype=DTYPE)
    
    cdef np.ndarray[DTYPEF_t, ndim=1] xEx = np.zeros(numBoxes, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] yEx = np.zeros(numBoxes, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] allEx = np.zeros(numBoxes, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] allanFactor = np.zeros(numBoxes, dtype=DTYPEF)
    cdef np.ndarray[DTYPE_t, ndim=1] boxSeries = np.zeros(numBoxes, dtype=DTYPE)
    
    cdef int xBox, yBox, xOnset, yOnset, xnOnset, currBox, currnBox, boxSize, numXBoxes, numYBoxes
    
    cdef int boxNum
    for boxNum in range(numBoxes):
        boxSize = 2**(boxNum)
        boxSeries[boxNum] = boxSize
        
        numXBoxes = xSize // boxSize
        numYBoxes = ySize // boxSize
        
        #do X striping
        
        for xBox in range(int(numXBoxes-1)):
            for yBox in range(int(numYBoxes)):
                xOnset = int((xBox) * (xSize / numXBoxes))
                yOnset = int((yBox) * (ySize / numYBoxes))
                xnOnset = int((xBox + 1) * (xSize / numXBoxes)) 
                if boxSize > 1:
                    currBox = sum2d(inData[xOnset:xOnset+boxSize, yOnset:yOnset+boxSize])
                    currnBox = sum2d(inData[xnOnset:xnOnset+boxSize, yOnset:yOnset+boxSize])
                else:
                    currBox = inData[xOnset, yOnset]
                    currnBox = inData[xnOnset, yOnset]                                 
                xTotal[boxNum] = xTotal[boxNum] + (currBox - currnBox) ** 2
        xEx[boxNum] = xTotal[boxNum] / ((numXBoxes - 1) * (numYBoxes - 1))
   
        #do Y striping
        for xBox in range(int(numXBoxes)):
            for yBox in range(int(numYBoxes)):
                xOnset = int((xBox) * (xSize / numXBoxes) )
                yOnset = int((yBox) * (ySize / numYBoxes) )
                
                if boxSize > 1:
                    currBox = sum2d(inData[xOnset:xOnset+boxSize, yOnset:yOnset+boxSize])
                else:
                    currBox = inData[xOnset, yOnset]
                
                if yBox != numYBoxes - 1:
                    ynOnset = int((yBox + 1) * (ySize / numYBoxes))
                    if boxSize > 1:
                        currnBox = sum2d(inData[xOnset:xOnset+boxSize, ynOnset:ynOnset+boxSize])
                    else:
                        currnBox = inData[xOnset, ynOnset]

                    yTotal[boxNum] = yTotal[boxNum] + (currBox - currnBox) ** 2
                # sum2d all boxes
                allTotal[boxNum] = allTotal[boxNum] + currBox
        
        yEx[boxNum] = yTotal[boxNum] / ((numXBoxes - 1) * (numYBoxes - 1))        
        allEx[boxNum] = allTotal[boxNum] / (numXBoxes * numYBoxes)       
        allanFactor[boxNum] = ((xEx[boxNum] + yEx[boxNum]) / 2) / (2 * allEx[boxNum])
    return (allanFactor, boxSeries)
    

            
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

import numpy as np

def af2d(inData):
    """ 
        (af_variance, af_boxes) = af2d(data)
        Input: mx n matrix containing count values at x and y cooridinates 
        Output:
            allanFactor - Allan factor variance for increasing box sizes
            boxSeries   - box size series

        To calculate alpha, plot log(boxSeries .^ 2) vs. log(allanFactor) and estimate slope.
        Currently box sizes have sides of length 1,2,4,9 ... 2^N with a max side length of the
        smallest inData dimension divided by four.
    """
        
    (xSize, ySize) = inData.shape
    xSize = float(xSize)
    ySize = float(ySize)
    maxBoxSize = np.floor(np.amin([xSize, ySize]) / 4)
    numBoxes = np.floor(np.log2(maxBoxSize)) + 1
    
    xTotal = np.zeros(numBoxes)
    yTotal = np.zeros(numBoxes)
    allTotal = np.zeros(numBoxes)
    
    xEx = np.zeros(numBoxes)
    yEx = np.zeros(numBoxes)
    allEx = np.zeros(numBoxes)
    allanFactor = np.zeros(numBoxes)
    boxSeries = np.zeros(numBoxes)
    
    for boxNum in range(int(numBoxes)):
        boxSize = 2**(boxNum)
        boxSeries[boxNum] = boxSize
        
        numXBoxes = np.floor(xSize / boxSize)
        numYBoxes = np.floor(ySize / boxSize)
        
        #do X striping
        for xBox in range(int(numXBoxes-1)):
            for yBox in range(int(numYBoxes)):
                xOnset = np.floor((xBox) * (xSize / numXBoxes))
                yOnset = np.floor((yBox) * (ySize / numYBoxes))
                xnOnset = np.floor((xBox + 1) * (xSize / numXBoxes)) 
                if boxSize > 1:
                    currBox = np.sum(inData[xOnset:xOnset+boxSize, yOnset:yOnset+boxSize])
                    currnBox = np.sum(inData[xnOnset:xnOnset+boxSize, yOnset:yOnset+boxSize])
                else:
                    currBox = inData[xOnset, yOnset]
                    currnBox = inData[xnOnset, yOnset]                                 
                xTotal[boxNum] = xTotal[boxNum] + (currBox - currnBox) ** 2
        xEx[boxNum] = xTotal[boxNum] / ((numXBoxes - 1) * (numYBoxes - 1))
   
        #do Y striping
        for xBox in range(int(numXBoxes)):
            for yBox in range(int(numYBoxes)):
                xOnset = np.floor((xBox) * (xSize / numXBoxes) )
                yOnset = np.floor((yBox) * (ySize / numYBoxes) )
                
                if boxSize > 1:
                    currBox = np.sum(inData[xOnset:xOnset+boxSize, yOnset:yOnset+boxSize])
                else:
                    currBox = inData[xOnset, yOnset]
                
                if yBox != numYBoxes - 1:
                    ynOnset = np.floor((yBox + 1) * (ySize / numYBoxes))
                    if boxSize > 1:
                        currnBox = np.sum(inData[xOnset:xOnset+boxSize, ynOnset:ynOnset+boxSize])
                    else:
                        currnBox = inData[xOnset, ynOnset]

                    yTotal[boxNum] = yTotal[boxNum] + (currBox - currnBox) ** 2
                # sum all boxes
                allTotal[boxNum] = allTotal[boxNum] + currBox
        
        yEx[boxNum] = yTotal[boxNum] / ((numXBoxes - 1) * (numYBoxes - 1))        
        allEx[boxNum] = allTotal[boxNum] / (numXBoxes * numYBoxes)       
        allanFactor[boxNum] = ((xEx[boxNum] + yEx[boxNum]) / 2) / (2 * allEx[boxNum])
    return (allanFactor, boxSeries)
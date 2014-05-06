import numpy as np

#x must be a np array
def lnbin(x, BinNum):
    """ 
    Logarithmically bins a numpy array, returns (midpoints, Freq)

    This function take the input of a data vector x, which is to be binned;
    it also takes in the amount bins one would like the data binned into. The
    output is two vectors, one containing the normalised frequency of each bin 
    (Freq), the other, the midpoint of each bin (midpts).
    Added and error to the binned frequency: eFreq (As of June 30 2010). If this
    option is not required, just call the function without including the third out
    put; i.e.: [midpts Freq]=lnbin(x,BinNum).
    
    Updated 2/6/14 to change the min to scale automatically
    """
    if type(x) != np.ndarray:
        try:
            x = np.array(x)
            
        except:
            print 'Improper input format!'
            raise
    
    x = np.sort(x)
    i = 0
    while x[i] <= 0:
        i += 1
        
    percent_binned = float((x.size-(i+1))) / x.size*100
    #print 'Percentage of input vec binned {}'.format(percent_binned)
    
    FPT = x[i:]
    LFPT = np.log(FPT)
    max1 = np.log( np.ceil(np.amax(FPT)))
    #min1 = 1
    min1 = np.log(np.floor(np.min(FPT)))
    
    LFreq = np.zeros((BinNum, 1))
    LTime = np.zeros((BinNum, 1))
    Lends = np.zeros((BinNum, 2))
    
    step = (max1-min1) / BinNum
    
    #LOG Binning Data ###########################
    for i in range(FPT.size):
        for k in range(BinNum):
            if( k*step+min1 <= LFPT[i] and LFPT[i] < (k+1)*step+min1):
                LFreq[k] += 1 #check LFreq on the first bin
            LTime[k] = (k+1)*step-(0.5*step)+min1
            Lends[k, 0] = k*step+min1
            Lends[k, 1] = (k+1)*step+min1
    
    ends = np.exp(Lends)    
    
    widths = ends[:,1] - ends[:,0]
    Freq = LFreq.T / widths / x.size
    eFreq = 1.0 / np.sqrt(LFreq) * Freq
    midpts = np.exp(LTime)
    
    return (midpts[:,0], Freq.T[:,0])
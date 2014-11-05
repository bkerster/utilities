# DFA script for python. Based upon code by Guan Wenye. This code is under the same license as the original,
# and the original license is copied below.

#Copyright (c) 2009, Guan Wenye
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

    # * Redistributions of source code must retain the above copyright
      # notice, this list of conditions and the following disclaimer.
    # * Redistributions in binary form must reproduce the above copyright
      # notice, this list of conditions and the following disclaimer in
      # the documentation and/or other materials provided with the distribution

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


#The code can be run by calling DFA_main(data) on a 1d numpy array of time series data.

def DFA(data, win_length, order):
    ''' 
    Base DFA function 
    Args: 
        data: a 1d numpy array of time series data
        win_length: the size of the window to use
        order: the order of function to be used to make fits
    '''
    
    N = len(data)
    n = int(np.floor(N/win_length))
    N1 = n * win_length
    y = np.zeros(N1)
    Yn = np.zeros(N1)
    fitcoef = np.zeros((n, order+1))
    mean1 = np.mean(data[:N1])
    for i in range(N1):
        y[i] = np.sum(data[:i] - mean1)
    #y = y.T
    for j in range(n):
        
        try:
            fitcoef[j,:] = np.polyfit(range(win_length), y[j*win_length : (j+1)*win_length], order)
        except:
            import pdb; pdb.set_trace()
    
    for j in range(n):
        Yn[j*win_length:(j+1)*win_length] = np.polyval(fitcoef[j,:], range(win_length))
    
    sum1 = np.sum( (y - Yn)**2) / N1
    sum1 = np.sqrt(sum1)
    return sum1
    
def DFA_main(data):
    ''' Main driver function
    Args:
        data: a 1d numpy array of time series data
    Returns: 
        D: dimension of the data
        Alpha1: The alpha(Estimated scaling exponent) of the data
    '''
    #data should be a time series with a length greater than 2000
    n = np.arange(100,1001,100) #inclusive of 1000
    N1 = len(n)
    F_n = np.zeros(N1)
    for i in range(N1):
        F_n[i] = DFA(data, n[i], 1)
    
    #plt.plot(np.log(n), np.log(F_n))
    #plt.xlabel('n')
    #plt.ylabel('F(n)')
    A = np.polyfit(np.log(n), np.log(F_n), 1)
    Alpha1 = A[0]
    D = 3-A[0]
    return D, Alpha1
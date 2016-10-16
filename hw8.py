# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 13:30:28 2014
Implement EM to train an HMM for whichever dataset you used for assignment 7.
The observation probs should be as in assignment 7: either gaussian, or two 
discrete distributions conditionally independent given the hidden state.

Does the HMM model the data better than the original non-sequence model?
What is the best number of states?
@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""

import numpy as np
import matplotlib.pyplot as matlab
import matplotlib.mlab as mlab

# Note: X and mu are assumed to be column vector
def normPDF(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0/(np.math.pow((2*np.pi), float(size)/2) * np.math.pow(det, 1.0/2))
        x_mu = np.matrix(x - mu).T
        inv_ = np.linalg.inv(sigma)
        result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
        return -1

def initForwardBackward(X,K,d,N):
    # Initialize the state transition matrix, A. A is a KxK matrix where
    # element A_{jk} = p(Z_n = k | Z_{n-1} = j)
    # Therefore, the matrix will be row-wise normalized. IOW, Sum(Row) = 1  
    # State transition probability is time independent.
    A = np.ones((K,K))
    A = A/np.sum(A,1)[None].T
    
    # Initialize the marginal probability for the first hidden variable
    # It is a Kx1 vector
    PI = np.ones((K,1))/K
    
    # Initialize Emission Probability. We assume Gaussian distribution
    # for emission. So we just need to keep the mean and covariance. These 
    # parameters are different for different states.
    # Mu is dxK where kth column represents mu for kth state
    # SIGMA is a list of K matrices of shape dxd. Each element represent
    # covariance matrix for the corresponding state.
    # Given the current latent variable state, emission probability is
    # independent of time
    MU = np.random.rand(d,K)
    SIGMA = [np.eye(d) for i in xrange(K)]
    
    return A, PI, MU, SIGMA

def buildAlpha(X,PI,A,MU,SIGMA):
    # We build up Alpha here using dynamic programming. It is a KxN matrix
    # where the element ALPHA_{ij} represents the forward probability
    # for jth timestep (j = 1...N) and ith state. The columns of ALPHA are
    # normalized for preventing underflow problem as discussed in secion
    # 13.2.4 in Bishop's PRML book. So,sum(column) = 1
    # c_t is the normalizing costant
    N = np.size(X,1)
    K = np.size(PI,0)
    Alpha = np.zeros((K,N))
    c = np.zeros(N)
    
    # Base case: build the first column of ALPHA
    for i in xrange(K):
        Alpha[i,0] = PI[i]*normPDF(X[:,0],MU[:,i],SIGMA[i])
    c[0] = np.sum(Alpha[:,0])
    Alpha[:,0] = Alpha[:,0]/c[0]

    # Build up the subsequent columns
    for t in xrange(1,N):
        for i in xrange(K):
            for j in xrange(K):
                Alpha[i,t] += Alpha[j,t-1]*A[j,i] # sum part of recursion
            Alpha[i,t] *= normPDF(X[:,t],MU[:,i],SIGMA[i]) # product with emission prob
        c[t] = np.sum(Alpha[:,t])
        Alpha[:,t] = Alpha[:,t]/c[t]   # for scaling factors
    return Alpha, c
    
def buildBeta(X,c,PI,A,MU,SIGMA):
    # Beta is KxN matrix where Beta_{ij} represents the backward probability
    # for jth timestamp and ith state. Columns of Beta are normalized using
    # the element of vector c.
    
    N = np.size(X,1)
    K = np.size(PI,0)
    Beta = np.zeros((K,N))
    
    # Base case: build the last column of Beta
    for i in xrange(K):
        Beta[i,N-1]=1.
        
    # Build up the matrix backwards
    for t in xrange(N-2,-1,-1):
        for i in xrange(K):
            for j in xrange(K):
                Beta[i,t] += Beta[j,t+1]*A[i,j]*normPDF(X[:,t+1],MU[:,j],SIGMA[j])
        Beta[:,t] /= c[t+1]
    return Beta

def Estep(trainSet, PI,A,MU,SIGMA):
    # The goal of E step is to evaluate Gamma(Z_{n}) and Xi(Z_{n-1},Z_{n})
    # First, create the forward and backward probability matrices
    Alpha, c = buildAlpha(trainSet, PI,A,MU,SIGMA)
    Beta = buildBeta(trainSet,c,PI,A,MU,SIGMA)
    
    # Dimension of Gamma is equal to Alpha and Beta where nth column represents
    # posterior density of nth latent variable. Each row represents a state
    # value of all the latent variables. IOW, (i,j)th element represents
    # p(Z_j = i | X,MU,SIGMA) 
    Gamma = Alpha*Beta
    
    # Xi is a KxKx(N-1) matrix (N is the length of data seq)
    # Xi(:,:,t) = Xi(Z_{t-1},Z_{t})
    N = np.size(trainSet,1)
    K = np.size(PI,0)    
    Xi = np.zeros((K,K,N))
    for t in xrange(1,N):
        Xi[:,:,t] = (1/c[t])*Alpha[:,t-1][None].T.dot(Beta[:,t][None])*A
        # Now columnwise multiply the emission prob
        for col in xrange(K):
            Xi[:,col,t] *= normPDF(trainSet[:,t],MU[:,col],SIGMA[col])
    
    return Gamma, Xi, c

def Mstep(X, Gamma, Xi):
    # Goal of M step is to calculate PI, A, MU, and SIGMA while treating
    # Gamma and Xi as constant
    K = np.size(Gamma,0)
    d = np.size(X,0)

    PI = (Gamma[:,0]/np.sum(Gamma[:,0]))[None].T
    tempSum = np.sum(Xi[:,:,1:],axis=2)
    A = tempSum/np.sum(tempSum,axis=1)[None].T

    MU = np.zeros((d,K))
    GamSUM = np.sum(Gamma,axis=1)[None].T
    SIGMA = []
    for k in xrange(K):  
        MU[:,k] = np.sum(Gamma[k,:]*X,axis=1)/GamSUM[k]
        X_MU = X - MU[:,k][None].T
        SIGMA.append(X_MU.dot(((X_MU*(Gamma[k,:][None])).T))/GamSUM[k])
    return PI,A,MU,SIGMA

def main():
    # Reading the data file
    input_file = open('points.dat')
    lines = input_file.readlines()
    allData = np.array([line.strip().split() for line in lines]).astype(np.float)
    (m, n) = np.shape(allData)

    # Separating out dev and train set
    devSet = allData[np.math.ceil(m*0.9):, 0:].T
    trainSet = allData[:np.math.floor(m*0.9), 0:].T
    
    # Setting up total number of clusters which will be fixed
    K = 3

    # Initialization: Build a state transition matrix with uniform probability
    A, PI, MU, SIGMA = initForwardBackward(trainSet,K,n,m)

    # Temporary variables. X, Y mesh for plotting
    nx = np.arange(-4.0, 4.0, 0.1)
    ny = np.arange(-4.0, 4.0, 0.1)
    ax, ay = np.meshgrid(nx, ny)
    
    iter = 0
    prevll = -999999
    while(True):
        iter = iter + 1
        # E-Step
        Gamma, Xi, c = Estep(trainSet,PI,A,MU,SIGMA)
        
        # M-Step
        PI,A,MU,SIGMA = Mstep(trainSet, Gamma, Xi)
        
        # Calculate log likelihood. We use the c vector for log likelihood because
        # it already gives us p(X_1^N)
        ll_train = np.sum(np.log(c))
        Gamma_dev,Xi_dev,c_dev = Estep(devSet,PI,A,MU,SIGMA)
        ll_dev = np.sum(np.log(c_dev))
        
        # For first window
        matlab.figure(1)
        # Plot the log-likelihood of the training data
        matlab.subplot(211)
        matlab.scatter(iter,ll_train,c='b')
        matlab.hold(True)
        matlab.xlabel('Iteration')
        matlab.ylabel('Log Likelihood of Training Data')

        # Plot the log likelihood of Development Data
        matlab.subplot(212)        
        matlab.scatter(iter,ll_dev,c='r')        
        matlab.hold(True)
        matlab.xlabel('Iteration')
        matlab.ylabel('Log Likelihood of Development Data')
        
        # Render these        
        matlab.draw()
        matlab.pause(0.01)
        
        # Plot the scatter plots and clusters
        matlab.figure(2)
        # Plot scatter plot of training data and corresponding clusters
        matlab.subplot(211)
        matlab.scatter(trainSet[0,0:],trainSet[1,0:])
        matlab.hold(True)
        for k in range(0, K):
            az = mlab.bivariate_normal(ax, ay, SIGMA[k][0, 0], SIGMA[k][1, \
                1], MU[0,k], MU[1,k], SIGMA[k][1, 0])
            try:
                matlab.contour(ax, ay, az)
            except:
                continue
        matlab.hold(False)
        
        # Render these
        matlab.draw()
        matlab.pause(0.01)
                
        matlab.subplot(212)
        matlab.scatter(devSet[0,0:],devSet[1,0:])
        matlab.hold(True)
        for k in range(0, K):
            az = mlab.bivariate_normal(ax, ay, SIGMA[k][0, 0], SIGMA[k][1, \
                1], MU[0,k], MU[1,k], SIGMA[k][1, 0])
            try:
                matlab.contour(ax, ay, az)
            except:
                continue
        matlab.hold(False)

        # Render these
        matlab.draw()
        matlab.pause(0.01)
        
        if(iter>50 or (ll_train - prevll)< 0.05):
            break
        print abs(ll_train - prevll)
        prevll = ll_train

if __name__ == '__main__':
    main()
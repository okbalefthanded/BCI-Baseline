from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EPFL(BaseEstimator, TransformerMixin):

    def __init__(self, p=0.1, decimation_factor=12):
        # self.w = None
        self.l = []
        self.h = []
        self.p = p
        self.decimation_factor = decimation_factor

    def fit(self, X, y=None):
        # X : ndarray : samples, channels, trials 
        if X.ndim == 4:
            samples, channels, epochs, trials = X.shape
            X = X.reshape((samples, channels, epochs*trials), order='F')       
        X = X[::self.decimation_factor,:,:]
        samples, channels, trials = X.shape
        self.l = np.zeros((channels))
        self.h = np.zeros((channels))
        samples, channels, trials = X.shape
        clip = int(np.round(samples*trials*self.p / 2))
        X = X.transpose((1,0,2))
        X = X.reshape((channels, samples*trials))
        for ch in range(0, channels):
            tmp = np.sort(X[ch,:])
            self.l[ch] = tmp[clip]
            self.h[ch] = tmp[-1-clip]
        
        return self

    def transform(self, X):
        # X : ndarray : samples, channels, trials
        if X.ndim == 4:
            samples, channels, epochs, trials = X.shape
            X = X.reshape((samples, channels, epochs*trials), order='F')  
            
        X = X[::self.decimation_factor,:,:]       
        samples, channels, trials = X.shape
        X = X.transpose((1,0,2))
        X = X.reshape((channels, samples*trials))
        ll = np.tile(self.l, (samples*trials,1)).transpose((1,0))
        lh = np.tile(self.h, (samples*trials,1)).transpose((1,0))
        i_l = X < ll
        i_h = X > lh
        X[i_l] = ll[i_l]
        X[i_h] = lh[i_h]
        X = X.reshape((channels,samples,trials)).transpose((1,0,2))
        X = X.reshape((samples*channels, trials), order='F').transpose((1,0))
        return X
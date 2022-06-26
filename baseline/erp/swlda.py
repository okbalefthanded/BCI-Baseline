# Author: Okba Bekhelifi

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import softmax
from .stepwise.stepwise import stepwisefit
from scipy.special import expit
from copy import deepcopy
import numpy as np

class SWLDA(BaseEstimator, ClassifierMixin): 
	
	# Step-wise LDA for binary classification.
	# uses stepwisefit.py by Collin RM Stocks
	
    def __init__(self):
        self.w = None
        self.b = None
        self.inmodel = None
    
    def fit(self, X, y=None, penter=0.1, premove=0.15):
        yy = np.copy(y)
        yy[yy==0.] = -1
        b, se, pval, inmodel, stats, nextstep, history = stepwisefit(X, yy, penter=0.1, premove=0.15)
        x1 = X[yy==1,:]
        mu1 = x1.mean(axis=0)
        x2 = X[yy==-1,:] 
        mu2 = x2.mean(axis=0)
        mu_both = (mu1 + mu2) / 2 
        self.w = np.zeros(b.shape)
        self.w[inmodel] = b[inmodel]
        self.b = np.dot(-self.w.transpose(), mu_both)        
        self.inmodel = inmodel
        return self
    
    def decision_function(self,X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X, y=None):
        pred = np.sign(self.decision_function(X))
        pred[pred==-1.] = 0.
        return pred    
    
    def score(self,X, y=None):
        return accuracy_score(y, self.predict(X))
    
    def predict_proba(self,X):
        # return softmax(self.decision_function(X))
        # return self.decision_function(X)
        decision = self.decision_function(X)
        return proba = expit(decision)
            
        

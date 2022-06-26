from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit
from scipy.linalg import eigh
import numpy as np


class BLDA(BaseEstimator, ClassifierMixin):
    
    def __init__(self, verbose=False):
        self.evidence = 0
        self.alpha = 0
        self.beta = 0
        self.w = []
        self.p = []
        self.verbose = verbose
    
    def fit(self, X, y=None):
        '''
        '''
        # compute regression targets from class labels 
        # (to do lda via regression)      
        n_instances = y.shape[0]

        y = y.astype(np.float32)
        y = y.T
        n_posexamples = np.sum(y==1)
        n_negexamples = np.sum(y==-1)
        n_examples = n_posexamples + n_negexamples
        y[y==1] = n_examples / n_posexamples
        y[y==-1] = -n_examples / n_negexamples
        # add feature that is constantly one (bias term)
        if X.shape[0] == n_instances:
            X = X.T
        X = np.vstack ((X, np.ones((1,X.shape[1]))))
        
        # initialize variables for fast iterative estimation of
        # alpha and beta
        n_features = X.shape[0]               # dimension of feature vectors 
        d_beta = np.inf                       # (initial) diff. between new and old beta  
        d_alpha = np.inf                      # (initial) diff. between new and old alpha 
        alpha    = 25                         # (initial) inverse variance of prior distribution
        biasalpha = 0.00000001                # (initial) inverse variance of prior for bias term
        beta     = 1                          # (initial) inverse variance around targets
        stopeps  = 0.0001                     # desired precision for alpha and beta
        i        = 1                          # keeps track of number of iterations
        maxit    = 500                        # maximal number of iterations 
        d, v = eigh(np.dot(X,X.T))            # needed for fast estimation of alpha and beta 
        sort_perm =d.argsort()
        d.sort()
        v = v[:, sort_perm]
        d = d.reshape( (d.shape[0],1) )
        vxy = np.linalg.multi_dot([v.T, X, y.T])                 # dito
        vxy = vxy.reshape((vxy.shape[0], 1))                     # dito
        e = np.ones( (n_features-1,1))     # dito
        
        # estimate alpha and beta iteratively
        while ((d_alpha > stopeps) or (d_beta > stopeps)) and (i < maxit):
            alphaold = alpha
            betaold  = beta        
            m = beta * np.dot(v , np.multiply( np.power(beta*d+np.vstack((alpha*e, biasalpha)), -1), vxy) )            
            err = np.sum( np.power( y-np.dot(m.T,X), 2 ) )            
            gamma = np.sum(  np.true_divide(beta*d, beta*d+np.vstack((alpha*e, biasalpha))) )
            alpha = gamma / np.asscalar(np.dot(m.T,m))
            beta  = (n_examples - gamma) / err
            if self.verbose:
                print('Iteration %d : alpha = %f, beta = %f\n' % (i,alpha,beta) )            
            d_alpha = np.abs(alpha-alphaold)
            d_beta  = np.abs(beta-betaold)
            i = i + 1
        
        # process results of estimation 
        if (i < maxit):    
            # compute the log evidence
            # this can be used for simple model selection tasks
            # (see MacKays paper)        
            self.evidence = (n_features/2)*np.log(alpha) + (n_examples/2)*np.log(beta) - (beta/2)*err - \
            (alpha/2)*np.dot(m.T, m) - 0.5*np.sum(np.log( beta*d+np.vstack((alpha*e, biasalpha)))) -\
            (n_examples/2)*np.log(2*np.pi)

            # store alpha, beta, the posterior mean and the posterrior precision-
            # matrix in class attributes
            self.alpha = alpha
            self.beta  = beta
            self.w     = m
            self.p     = v* np.diag( np.power( beta*d+np.vstack((alpha*e, biasalpha)), -1) )* v.T
 
            if self.verbose:
                print('Optimization of alpha and beta successfull')
                print('The logevidence is ', self.evidence)
        else:
            print('Optimization of alpha and beta did not converge after %d iterations', maxit)
            print('Giving up.')           
                   
        return self
         
    def decision_function(self,X):
        X = X.T
        X = np.vstack( (X, np.ones((1,X.shape[1]))) )
        return np.dot(X.T, self.w) 
    
    def predict(self, X, y=None):
        return self.decision_function(X)            
    
    def score(self,X, y=None):
        pass
    
    def predict_proba(self,X):
        # return softmax(self.decision_function(X))
        # return self.decision_function(X)
        decision = self.decision_function(X)
        return proba = expit(decision)


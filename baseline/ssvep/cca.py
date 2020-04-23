from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.linalg import eig
from scipy import sqrt
import numpy as np


class CCA(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_harmonics=2, frequencies=[], references=None, length=4):
        self.n_harmonics = n_harmonics
        self.frequencies = frequencies
        self.references = references
        self.length = length

    def fit(self, X, y=None):
        # construct frequency template
        samples = X.shape[1]
        t = np.linspace(0.0, float(self.length), samples)
        refs = [ [np.cos(2*np.pi*f*t*i),np.sin(2*np.pi*f*t*i)] for f in self.frequencies for i in range(1, self.n_harmonics+1)]
        self.references =  np.array(refs).reshape(len(self.frequencies), 2*self.n_harmonics, samples) 
        return self

    def decision_function(self, X):
        return self._apply_cca(X)

    def predict(self, X, y=None):
        return np.argmax(self.decision_function(X)) 

    def predict_proba(self, X):
        pass

    def _apply_cca(self, X):
        coefs = []
        for i in range(self.references.shape[0]):
            coefs.append(self._cca_coef(X,self.references[i,:,:]))
        coefs = np.array(coefs).transpose()
        return coefs

    def _cca_coef(self, X,Y):
        if X.shape[1] != Y.shape[1]:
            raise Exception('unable to apply CCA, X and Y have different dimensions')
        z = np.vstack((X,Y))
        C = np.cov(z)
        sx = X.shape[0]
        sy = Y.shape[0]
        Cxx = C[0:sx, 0:sx] + 10**(-8)*np.eye(sx)
        Cxy = C[0:sx, sx:sx+sy]
        Cyx = Cxy.transpose()
        Cyy = C[sx:sx+sy, sx:sx+sy] + 10**(-8)*np.eye(sy)
        invCyy = np.linalg.pinv(Cyy)
        invCxx = np.linalg.pinv(Cxx)
        r, Wx = eig(invCxx.dot(Cxy).dot(invCyy).dot(Cyx))
        r = sqrt(np.real(r))
        r = np.sort(np.real(r),  axis=None)
        r = np.flipud(r)
        return r 

class ITCCA(CCA):
    
    def fit(self, X, y=None):
      '''
      X : samples x channels x trials?
      '''
      stimuli_count = len(list(set(y)))
      refs = []
      for i in range(stimuli_count): 
          refs.append(np.mean(X[:,:,np.where(y==i+1)].squeeze(), axis=2))
      self.references = np.array(refs).transpose((0,2,1))
      # references : 
      return self
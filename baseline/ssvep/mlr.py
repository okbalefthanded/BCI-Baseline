from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from scipy.linalg import eigh
from numpy.linalg import matrix_rank
from numpy.linalg import svd
import numpy as np


class MLR(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pca_w = None
        self.w = None
        self.mean = None        


    def fit(self, X, Y):
        data = np.array(X)
        data = data.transpose((2,1,0))
        lb = LabelBinarizer()
        samples, channels, epochs = data.shape
        trainRaw = np.array(data)
        trainRaw = trainRaw.reshape((samples*channels, epochs), order='F') 
        self.mean = trainRaw.mean(axis=1)  
        trainRaw -= self.mean[:, None]        
        y = lb.fit_transform(Y).T
        # 
        PCA_W = self._pca_func(trainRaw)
        train_Data = np.dot(PCA_W.T, trainRaw)
        train_Data = np.vstack((np.ones((1,epochs)),train_Data))
        # 
        w = self._MultiLR(train_Data, y)
        return self

    
    def transform(self, X):
        data = np.array(X)
        if data.ndim == 3:
            data = data.transpose((2,1,0))
            samples, channels, epochs = data.shape
        elif data.ndim == 2:
            samples, channels = data.shape
            epochs = 1
        #data = np.array(X)
        data = data.reshape((samples*channels, epochs), order='F')
        data -= self.mean[:, None]
        proj_data = np.dot(self.pca_w.T, data)
        proj_data = np.vstack((np.ones((1,epochs)), proj_data))
        return np.dot(proj_data.T, self.w)


    def _pca_func(self, X):
        # data = trainRaw
        meanData = X.mean(axis=1)
        X -= meanData[:,None]
        S, V = eigh(np.dot(X.T, X))
        sorted_rank = S[::-1].argsort()
        S[::-1].sort()
        S = np.diag(S)
        V = V[:, sorted_rank]
        r = matrix_rank(S)
        S1 = S[0:r, 0:r]
        S2 = np.diag(S1)
        S3 = np.power(S2, -0.5)
        V1 = V[:,0:r]
        U = np.linalg.multi_dot([X, V1, np.diag(S3)])   
        All_energy = np.sum(S2)
        for j in range(r):
            if (np.sum(S2[0:j])/All_energy ) > 0.99:
                break
        self.pca_w = U[:,0:j]
        return self.pca_w

    def _MultiLR(self, X, Y):
        U, S, V = svd(X, full_matrices=False)
        r = matrix_rank(np.diag(S))
        U1 = U[:,0:r]
        V1 = V[0:r,:].T
        S_r = np.diag(S[0:r])
        self.w = np.linalg.multi_dot([U1, np.diag(np.divide(1,np.diag(S_r))) ,V1.T, Y.T])
        return self.w
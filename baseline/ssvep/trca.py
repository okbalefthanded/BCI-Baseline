from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.signal import cheb1ord, cheby1, filtfilt
from scipy.sparse.linalg import eigs
import numpy as np


class TRCA(BaseEstimator, ClassifierMixin):
    """
    Implements Task-related component analysis for SSVEP frequencey classification [1]

    References :

    [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
    "Enhancing detection of SSVEPs for a high-speed brain speller using
    task-related component analysis", IEEE Trans. Biomed. Eng, 65(1): 104-112, 2018.
    http://ieeexplore.ieee.org/document/7904641/
    """
    def __init__(self, n_fbs=5, ensemble=True, fs=512):
        '''
        '''
        self.fs = fs
        self.n_fbs = n_fbs
        self.ensemble = ensemble
        self.trains = []
        self.w = []
        self.num_targs = 0

    def fit(self, X, y=None):
        '''
        '''
        # Transform eeg data
        samples, channels, epochs = X.shape
        idx = np.argsort(y)
        targets_count = y.max().astype(int)
        X = X[:, :, idx]
        X = X.transpose((2, 1, 0))

        if isinstance(epochs/targets_count, int):
            X = X.reshape((epochs/targets_count, targets_count,
                           channels, samples), order='F')
        else:
            tr = np.floor(epochs/targets_count).astype(int)
            X = X[0:tr*targets_count, :,
                  :].reshape((tr, targets_count, channels, samples), order='F')

        X = X.transpose((1, 2, 3, 0))
        #
        num_targs, num_chans, num_smpls, _ = X.shape
        self.num_targs = num_targs
        self.trains = np.zeros((num_targs, self.n_fbs, num_chans, num_smpls))
        self.w = np.zeros((self.n_fbs, num_targs, num_chans))

        for targ_i in range(num_targs):
            eeg_tmp = X[targ_i, :, :, :]
            for fb_i in range(self.n_fbs):
                eeg_tmp = self._filterbank(eeg_tmp, fb_i)
                self.trains[targ_i, fb_i, :, :] = np.mean(eeg_tmp, axis=-1)
                w_tmp = self._trca(eeg_tmp)
                self.w[fb_i, targ_i, :] = w_tmp[:, 0]

        return self

    def decision_fucntion(self):
        '''
        '''
        pass

    def predict(self, X, y=None):
        '''
        '''
        epochs = X.shape[-1]
        # idx = np.argsort(y)
        # X = X[:, :, idx]
        X = X.transpose((2, 1, 0))
        #
        fb_coefs = np.arange(1, 6)**(-1.25) + 0.25
        # r = np.zeros((self.n_fbs, epochs))
		r = np.zeros((self.n_fbs, self.num_targs))
        results = []
        for targ_i in range(epochs):
            test_tmp = X[targ_i, :, :]
            for fb_i in range(self.n_fbs):
                testdata = self._filterbank(test_tmp, fb_i)
                for class_i in range(self.num_targs):
                    traindata = self.trains[class_i, fb_i, :, :]
                    if self.ensemble:
                        w = self.w[fb_i, :, :].T
                    else:
                        w = self.w[fb_i, class_i, :]
                    r_tmp = np.corrcoef(
                        np.matmul(testdata.T, w), np.matmul(traindata.T, w))
                    r[fb_i, class_i] = r_tmp[0, 1]

            rho = np.matmul(fb_coefs, r)
            tau = np.argmax(rho)
            results.append(tau)

        results = np.array(results)
        # results[idx] = results
        return results

    def predict_proba(self, X):
        '''
        '''
        pass

    def _trca(self, eeg):
        '''
        '''
        num_chans, num_smpls, num_trials = eeg.shape

        S = np.zeros((num_chans, num_chans))

        for trial_i in range(num_trials-1):
            x1 = eeg[:, :, trial_i]
            x1 = x1 - np.mean(x1, axis=1)[:, None]
            for trial_j in range(trial_i+1, num_trials):
                x2 = eeg[:, :, trial_j]
                x2 = x2 - np.mean(x2, axis=1)[:, None]
                S = S + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)

        UX = eeg.reshape((num_chans, num_smpls*num_trials))
        UX = UX - np.mean(UX, axis=1)[:, None]
        Q = np.matmul(UX, UX.T)
        _, W = eigs(S, M=Q)
        #_, W = np.linalg.eig(np.dot(np.linalg.inv(Q), S))
        return np.real(W)

    def _filterbank(self, eeg, idx_fbi):
        '''
        '''
        if eeg.ndim == 2:
            num_chans = eeg.shape[0]
            num_trials = 1
        else:
            num_chans, _, num_trials = eeg.shape
        fs = self.fs / 2
        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        Wp = [passband[idx_fbi]/fs, 90/fs]
        Ws = [stopband[idx_fbi]/fs, 100/fs]
        [N, Wn] = cheb1ord(Wp, Ws, 3, 40)
        [B, A] = cheby1(N, 0.5, Wn, 'bp')
        yy = np.zeros_like(eeg)
        if num_trials == 1:
            yy = filtfilt(B, A, eeg, axis=1)
        else:
            for trial_i in range(num_trials):
                for ch_i in range(num_chans):
                    yy[ch_i, :, trial_i] = filtfilt(
                        B, A, eeg[ch_i, :, trial_i])
        return yy

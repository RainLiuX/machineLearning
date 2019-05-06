import numpy as np
import math

def pca(X, k, method="EVD") :
    # Centralize
    X = np.array(X)
    Xmean = np.mean(X, axis=1)
    X = X - np.array([Xmean]).T[:,np.newaxis]

    n = X.shape[1]
    d = X.shape[0]
    if method == 'EVD' :
        C = 1/(m-1) * np.matmul(X, X.T)    #covariance matrix
        e, U = np.linalg.eig(C)    #eigen value/vectors of C
        idx = e.argsort()[::-1]
        U = U[:,idx]    #sort eigen vectors by eig values
        Y = np.matmul(U[:,:k].T, X)    #compress by Y = U_dxk^T X
        Xhat = np.matmul(U[:,:k], Y)    #remap by Xhat = U_dxk Y
        return Y, Xhat, U[:, :k]
    elif method == 'SVD' :
        D = 1/(d-1) * np.matmul(X.T, X)    #dot product matrix
        er, V = np.linalg.eig(D)
        idxr = er.argsort()[::-1]
        V = V[:,idxr].real; er = er[idxr]    #right eigen matrix
        S = np.diag([math.sqrt(abs(e)) for e in er])    #square root of eig values
        Y = np.matmul(S[:k,:k], -V[:,:k].T)    #compress by Y = S_kxk V_nxk^T
        #C = 1/(m-1) * np.matmul(X, X.T)
        #el, U = np.linalg.eig(C)
        #idxl = el.argsort()[::-1]
        #U = U[:,idx]
        #Xhat = np.matmul(U[:,:D], Y)
        return Y, None, None
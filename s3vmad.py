import numpy as np
import sklearn
import cvxopt
import sklearn.metrics.pairwise
from functools import reduce

class s3vmad(object):
    """Semisupervised Outlier Detection
    
    Parameters
    ----------
    kernel : {'rbf','if'}, default='rbf'
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'rbf', 'if'.
         If none is given, 'rbf' will be used.
    gamma : rbf kernel parameter, default=1.
        Tolerance for stopping criterion.
    nu1 : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors for labelled data. Should be in the interval (0, 1]. By default 0.5
        will be taken.
    nu2 : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors for unlabelled data. Should be in the interval (0, 1]. By default 0.5
        will be taken.
    Attributes
    ----------
    alpha : Lagrange multipliers to define the decision function.
    sv: Support vectors.
    ind_sv : Index of X corresponding to support vectors.
    Y_sv : Labels (+/-1) of support vectors (+1 if corresponding to unlabelled data).
    alpha_trunc : Lagrange multipliers truncated to ensure the sparsity of the support vectors.
    sv_trunc : Corresponding truncated support vectors.
    ind_sv_trunc : Ind of X corresponding to truncated support vectors.
    Y_sv_trunc : Labels (+/-1) of truncated support vectors (+1 if corresponding to unlabeled data).
    rho1 : Margin for labeled vectors.
    rho2 : Margin for unlabeled vectors.
    b : Decision function bias.
    """
    def __init__(self, kernel='rbf', gamma=1, C0=0.5, C1=0.5, C2=0.5):
        
        self.kernel = 'rbf'
        self.gamma = gamma
        self.C0 = C0
        self.C1 = C1
        self.C2 = C2

    def fit(self, X, y, tol = 1e-5, verbose = True):
        """Detects the soft boundary of the set of samples X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples_labelled + n_samples_unlabelled, n_features)
            Set of samples, where n_samples_labelled is the number of labelled samples, n_samples_unlabelled
            is the number of unlabelled samples, and n_features is the number of features.
        y : {array-like} of shape (n_samples_labelled). Set of labels for labelled vectors.
        Returns
        tol : Tolerance to trunc support vectors.
        verbose : Boolean. If True, displays optimization steps.
        """

        if np.any(np.abs(y)!=1):
            raise Exception("The vectors of labels should only contains +1 (nominal) or -1 (anomaly)")

        r = len(y)
        [n,d] = X.shape
        
        # Check kernel
        if(self.kernel == 'rbf'):
            G = sklearn.metrics.pairwise.rbf_kernel(X, gamma = self.gamma)
        elif(self.kernel == 'if'):
            G = if_map
        else:
            raise Exception("Unknown kernel")
        
        # Compute Optimization problem matrices and constraints
        if r == 0:
            Y = np.eye(n)
        elif r == n:
            Y = np.diag(y)
        else:
            Y = np.diag(np.append(y,np.ones(n-r)))
        P = Y.dot(G).dot(Y)
        q = np.r_[np.ones([r,1]),np.zeros([n-r,1])]
        
        #Gc = np.vstack((-np.eye(n), np.eye(n), np.append(-np.ones([1,r]),np.zeros([1,n-r]))))
        #hc = np.vstack((np.zeros([n,1]), 1/(nu1*r)*np.ones([r,1]), 1/(nu2*(n-r))*np.ones([n-r,1]), -1))

        #Ac = np.vstack((np.append(y.reshape([1,r]), np.zeros([1,n-r])),np.append(np.zeros([1,r]),np.ones([1,n-r]))))
        #bc = np.array([0.,1.]).reshape([2,1])
        
        C0star = (n-r)/(self.C0*(n-r+1))
        C1star = 1/(self.C1*(n-r+1))
        C2star = 1/(self.C2*(r+1))
        
        if r == 0:
            Gc = np.r_[-np.eye(n),np.eye(n)]
            hc = np.r_[-np.zeros(n),C1star*np.ones(n),-n].reshape([2*n+1,1])
            Ac = np.ones([1,n])
        elif r == n:
            Gc = np.r_[-np.eye(n),np.eye(n)]
            hc = np.r_[-np.zeros(n),C2star*np.ones(n),-n].reshape([2*n+1,1])
            Ac = y.transpose()
        else:
            Gc = np.r_[-np.eye(n),np.eye(n)]
            hc = np.r_[-np.zeros(n),C2star*np.ones(r),C1star*np.ones(n-r)].reshape([2*n,1])
            Ac = np.c_[y.transpose(),np.ones([1,n-r])]
        
        bc = np.reshape(float(C0star),[1,1])

        alpha = cvxopt_solve_qp(P = P, q = q, G = Gc, h = hc, A = Ac, b = bc, verbose = verbose) # solve the dual
        
        # Get support vectors
        sv = np.where(alpha>0)[0]
        sv_trunc = np.where(alpha>tol)[0]
        
        # Truncate alphas value
        alpha_trunc=np.zeros(n)
        alpha_trunc[sv_trunc] = alpha[sv_trunc]
        alpha_trunc[np.where(abs(alpha-1)<tol)] = 1
        
        # Compute score inside the constrains
        score_training_data = G.dot(Y.dot(alpha))
        score_training_data_labeled=score_training_data[0:r]
        score_training_data_unlabeled=score_training_data[r:]

        I_b = np.intersect1d(np.where(alpha>tol),np.where(abs(alpha-hc[n:])>tol))
        b = (Ac.ravel() - score_training_data)[I_b].mean()
            
        # Save results
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.ind_sv = sv
        self.alpha_trunc = alpha_trunc[sv_trunc]
        self.ind_sv_trunc = sv_trunc
        self.sv_trunc = X[sv_trunc]
        self.Y_sv_trunc = Y[sv_trunc,:][:,sv_trunc]
        self.Y_sv = Y[sv,:][:,sv]
        self.b = b
        
        return self
    
    def compute_scores(self, X2):
        """Computes the scores of test vectors in X2 using full support vectors.
        Parameters
        ----------
        X2 : {array-like, sparse matrix} of shape (n_samples_X2, n_features)
            Set of samples, where n_samples_X2 is the number of samples and
            n_features is the number of features.
        Returns
        -------
        scores_X2 : {array-like, sparse matrix} of shape (n_samples_X2, 1)
            Scores of each samples in X2.
        """
        scores_X2 = sklearn.metrics.pairwise.rbf_kernel(X2, self.sv, gamma = self.gamma).dot(self.Y_sv.dot(self.alpha)) + self.b
        
        return scores_X2
    
    def compute_scores_trunc(self, X2):
        """Computes the scores of the test vectors in X2 using the truncated support vectors.
        Parameters
        ----------
        X2 : {array-like, sparse matrix} of shape (n_samples_X2, n_features)
            Set of samples, where n_samples_X2 is the number of samples and
            n_features is the number of features.
        Returns
        -------
        scores_X2 : {array-like, sparse matrix} of shape (n_samples_X2, 1)
            Scores of each samples in X2.
        """
        scores_X2 = sklearn.metrics.pairwise.rbf_kernel(X2, self.sv_trunc, gamma = \
                   self.gamma).dot(self.Y_sv_trunc.dot(self.alpha_trunc)) + self.b
        
        return scores_X2
    
    def predict_trunc(self, X2):
        """Predict the label of data in X2 using truncated support vectors.
        Parameters
        ----------
        X2 : {array-like, sparse matrix} of shape (n_samples, n_features)
            Set of samples to get prediction from.
        Returns
        pred_X2 : Prediction of labels.
        """
        labels_X2 = np.sign(self.compute_scores_trunc(X2))
        
        return labels_X2
       
    def predict(self, X2):
        """Predict the label of data in X2 using full support vectors.
        Parameters
        ----------
        X2 : {array-like, sparse matrix} of shape (n_samples, n_features)
            Set of samples to get prediction from.
        Returns
        pred_X2 : Prediction of labels.
        """
        labels_X2 = np.sign(self.compute_scores(X2))
        
        return labels_X2
    
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, verbose = True):
    """Solves quadratic constrained optimization problem using Quadratic Programming.
    Paramaters
    ----------
    Solves argmin_x (1/2)*x^T*P*x +q^T*x s.t. G*x<=h and A*x=b
    Returns
    -------
    alpha : Solution of optimization problem.
    """
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options['show_progress'] = verbose
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

import numpy as np
import sklearn
import cvxopt
import sklearn.metrics.pairwise
from functools import reduce

class nuSSVM(object):
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
    def __init__(self, kernel='rbf', gamma=1, nu1=0.5, nu2=0.5):

        if nu1<0 or nu1>1:
            raise Exception("nu1 should be between 0 and 1")
        if nu2<0 or nu2>1:
            raise Exception("nu2 should be between 0 and 1")
        
        self.kernel = kernel
        self.gamma = gamma
        self.nu1 = nu1
        self.nu2 = nu2

    def fit(self, X, y, tol = 1e-5, verbose = True):
        """Detects the soft boundary of the set of samples X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples_labelled + n_samples_unlabelled, n_features)
            Set of samples, where n_samples_labelled is the number of labelled samples, n_samples_unlabelled
            is the number of unlabelled samples, and n_features is the number of features. The first n_samples_labelled
            data must correspond to the labels in y
        y : {array-like} of shape (n_samples_labelled). Set of labels for labelled vectors.
        Returns
        tol : Tolerance to trunc support vectors.
        verbose : Boolean. If True, displays optimization steps.
        """
        nu1 = self.nu1
        nu2 = self.nu2

        if np.any(np.abs(y)!=1):
            raise Exception("The vectors of labels should only contains +1 (nominal) or -1 (anomaly)")

        r = len(y)
        [n,d] = X.shape
        if r>0:
            #Check nu1 value
            cp1 = sum(y == 1)
            cm1 = sum(y == -1)
            nu1max = (min(cp1,cm1)+cm1)/r
            if(nu1 > nu1max):
                nu1 = 0.999*nu1max
                #print("Warning: nu1 was superior the the limit and was thus set to",nu1)

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
        q = np.zeros(n)
        
        #Gc = np.vstack((-np.eye(n), np.eye(n), np.append(-np.ones([1,r]),np.zeros([1,n-r]))))
        #hc = np.vstack((np.zeros([n,1]), 1/(nu1*r)*np.ones([r,1]), 1/(nu2*(n-r))*np.ones([n-r,1]), -1))

        #Ac = np.vstack((np.append(y.reshape([1,r]), np.zeros([1,n-r])),np.append(np.zeros([1,r]),np.ones([1,n-r]))))
        #bc = np.array([0.,1.]).reshape([2,1])
                
        if r == 0:
            Gc = np.r_[-np.eye(n),np.eye(n),-np.ones([1,n])]
            hc = np.r_[-np.zeros(n),(1/nu2)*np.ones(n),-n].reshape([2*n+1,1])
            Ac = np.ones([1,n])
        elif r == n:
            Gc = np.r_[-np.eye(n),np.eye(n),-np.ones([1,n])]
            hc = np.r_[-np.zeros(n),(1/nu1)*np.ones(n),-n].reshape([2*n+1,1])
            Ac = y.transpose()
        else:
            Gc = np.r_[-np.eye(n),np.eye(n),-np.c_[np.ones([1,r]),np.zeros([1,n-r])],-np.c_[np.zeros([1,r]),np.ones([1,n-r])]]
            hc = np.r_[-np.zeros(n),(1/nu1)*np.ones(r),(1/nu2)*np.ones(n-r),-r,-(n-r)].reshape([2*n+2,1])
            Ac = np.c_[y.transpose(),np.ones([1,n-r])]
        
        bc = np.reshape(float(n-r),[1,1])

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
        
        if r == 0:
            rho1 = 0
            rho2 = 0
            I_rho2_m_b = np.intersect1d(np.where(alpha[r:]>tol),np.where(abs(alpha[r:]-1/nu1)>tol))
            b = - np.mean(score_training_data_unlabeled[I_rho2_m_b])
        else:  
            if abs(sum(alpha[:r])-r)>tol:
                rho1 = 0

                ind_sv_unbounded = np.intersect1d(np.where(abs(alpha[:r]-1/nu1) >tol), np.where(abs(alpha[:r]) >tol))

                if ind_sv_unbounded.shape[0] == 0:
                    ind_sup = np.intersect1d(np.where(abs(alpha[:r]-1/nu1) < tol), np.where(y.ravel() == 1))
                    ind_inf = np.intersect1d(np.where(abs(alpha[:r]-1/nu1) <tol), np.where(y.ravel() == -1))
                    if ind_sup.shape[0] == 0:
                        b = np.max(-score_training_data_labeled[ind_inf])
                    elif ind_inf.shape[0] == 0:
                        b = np.min(-score_training_data_labeled[ind_sup])
                    else:
                        sup_b = np.min(-score_training_data_labeled[ind_sup])
                        inf_b = np.max(-score_training_data_labeled[ind_inf])
                        b = (sup_b+inf_b)/2        
                else:
                    b = -score_training_data[ind_sv_unbounded].mean()
               
            else:      
                I_rho1_m_b = reduce(np.intersect1d, (np.where(alpha[:r]>tol),np.where(abs(alpha[:r]-1/nu1)>tol), np.where(y.ravel() == 1)))
                I_rho1_p_b = reduce(np.intersect1d, (np.where(alpha[:r]>tol),np.where(abs(alpha[:r]-1/nu1)>tol), np.where(y.ravel() == -1)))

                if I_rho1_p_b.shape[0] == 0:
                    ind_sup_rho1_p_b = np.intersect1d(np.where(abs(alpha[:r]) < tol), np.where(y.ravel() == -1))
                    ind_inf_rho1_p_b = np.intersect1d(np.where(abs(alpha[:r]-1/nu1) < tol), np.where(y.ravel() == -1))

                    if ind_sup_rho1_p_b.shape[0] == 0:
                        rho1_p_b = np.max(-score_training_data_labeled)
                    elif ind_inf_rho1_p_b.shape[0] == 0:
                        rho1_p_b = np.min(-score_training_data_labeled)
                    else:       
                        sup_rho1_p_b = np.min(-score_training_data_labeled)
                        inf_rho1_p_b = np.max(-score_training_data_labeled)        
                        rho1_p_b = (sup_rho1_p_b + inf_rho1_p_b)/2
                else:
                    rho1_p_b = np.mean(-score_training_data_labeled[I_rho1_p_b])

                if I_rho1_m_b.shape[0] == 0:
                    ind_sup_b_m_rho1 = np.intersect1d(np.where(abs(alpha[:r]-1/nu1) < tol), np.where(y.ravel() == 1))   
                    ind_inf_b_m_rho1 = np.intersect1d(np.where(abs(alpha[:r]) < tol), np.where(y.ravel() == 1))

                    if ind_sup_b_m_rho1.shape[0] == 0 :  
                        rho1_m_b = -np.max(-score_training_data_labeled)
                    elif ind_inf_b_m_rho1.shape[0] == 0 : 
                        rho1_m_b = -np.min(-score_training_data_labeled)
                    else:
                        sup_b_m_rho1 = np.min(-score_training_data_labeled)
                        inf_b_m_rho1 = np.max(-score_training_data_labeled)
                        rho1_m_b = -(sup_b_m_rho1 + inf_b_m_rho1)/2
                else:    
                    rho1_m_b = np.mean(score_training_data_labeled[I_rho1_m_b])

                # Compute margin and bias
                b = (rho1_p_b-rho1_m_b)/2
                rho1 = (rho1_p_b+rho1_m_b)/2
            if r == n:
                rho2 = 0
            else:          
                if abs(sum(alpha[r:])-nu2*(n-r))>tol:
                    rho2 = 0
                else:       
                    I_rho2_m_b = np.intersect1d(np.where(alpha[r:]>tol),np.where(abs(alpha[r:]-1/nu2)>tol))
                    rho2 = b + np.mean(score_training_data_unlabeled[I_rho2_m_b])
            
        # Save results
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.ind_sv = sv
        self.alpha_trunc = alpha_trunc[sv_trunc]
        self.ind_sv_trunc = sv_trunc
        self.sv_trunc = X[sv_trunc]
        self.Y_sv_trunc = Y[sv_trunc,:][:,sv_trunc]
        self.Y_sv = Y[sv,:][:,sv]
        self.rho1 = rho1
        self.rho2 = rho2
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
    #if 'optimal' not in sol['status']:
     #   return None
    return np.array(sol['x']).reshape((P.shape[1],))

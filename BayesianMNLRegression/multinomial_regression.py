import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky, inv
from scipy.optimize import minimize_scalar
from copy import copy

'''
Implementation of the paper below:

Leonhard Held. Chris C. Holmes. "Bayesian auxiliary variable models for binary
    and multinomial regression." Bayesian Anal. 1 (1) 145 - 168, March 2006.
    https://doi.org/10.1214/06-BA105
'''

class MNLRegression:

    def __init__(self, num_categories,
                 dims,
                 prior_mean,
                 prior_var):

        self.num_categories = num_categories
        self.dims = dims
        self.X = None
        self.y = None
        self.n = None

        if len(prior_mean) != self.dims or len(prior_var) != self.dims:
            raise Exception("Prior means and vars must have dimension = dims")

        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.lmbda = None
        self.Z = None
        self.beta = None

        self.lmbda_hist = []
        self.Z_hist = []
        self.beta_hist = []

    '''
    Fits the data to the regression model. Every time the function is called,
    old data is overwritten.

    param X: A 2d matrix containing vectors to regress on, 2d array-like
    param y: A list of integers corresponding to the proper categorizations
                of X. If one_hot is True, then input should be 2d matrix
                containing one-hot encodings, 1d array-like or 2d array-like
    param one_hot: If True, passes one-hot encoding of y, boolean
    '''

    def fit(self, X, y, one_hot = False):
        if len(X) != len(y):
            raise Exception("X and y must have same length")
        if len(X[0]) != self.dims:
            raise Exception("dim(Xi) must equal dims")

        self.X = np.array(X)
        if not(one_hot):
            A = np.eye(num_categories)
            self.y = np.array([A[i] for i in y])
        else:
            self.y = np.array(y)

        self.n = len(y)
        self.lmbda = np.zeros((self.num_categories,self.n,self.n))
        self.Z = np.zeros((self.num_categories,self.n))
        self.beta = np.zeros((self.num_categories,self.dims))

    def predict(self, X):
        betas = self.posterior_sample(size = len(X))
        preds = []

        for i in range(len(X)):
            probs = self.__softmax(np.matmul(betas[i],X[i]))
            preds.append(np.argmax(probs))

        return np.array(preds)


    #Joint posterior sampling of the model parameters
    def posterior_sample(self, size = 1, burn = 750, full_output = False):
        self.__init_params()
        self.test1 = []
        self.test2 = []

        for t in range(size + burn):
            print("Iterate " + str(t+1))
            for q in range(self.num_categories):
                self.__calc_beta(q)
                self.__calc_latent(q)

            self.beta_hist.append(copy(self.beta))
            self.lmbda_hist.append(copy(self.lmbda))
            self.Z_hist.append(copy(self.Z))

        if full_output:
            return np.array(self.beta_hist)
        else:
            return np.array(self.beta_hist[burn:])


    def __init_params(self):
        for i in range(self.num_categories):
            self.lmbda[i] = np.eye(self.n)
            self.Z[i] = self.__truncated_logistic(np.zeros((self.n)),self.y[:,i].flatten())

        self.V = np.zeros((self.num_categories, self.dims,self.dims))
        self.L = np.zeros((self.num_categories, self.dims,self.dims))
        self.B = np.zeros((self.num_categories,self.dims))

        self.beta_hist = []
        self.lmbda_hist = []
        self.Z_hist = []



    #Gibbs sampling for regression parameters
    def __calc_beta(self, q):
        self.V[q] = inv(self.__multiply_list(self.X.T, inv(self.lmbda[q]),self.X)
                                             + inv(np.diag(self.prior_var)))
        self.L[q] = cholesky(self.V[q])
        self.B[q] = np.matmul(self.__multiply_list(self.V[q],self.X.T,inv(self.lmbda[q])),
                              self.Z[q])
        #self.B[q] += np.outer((1/self.prior_var).T,self.prior_mean)
        T = np.random.multivariate_normal(mean = np.zeros((self.dims)),
                                          cov = np.eye(self.dims))
        self.beta[q] = self.B[q] + np.matmul(self.L[q],T)



    #Gibbs sampling for latent variables Z and lmbda
    def __calc_latent(self, q):
        m = np.matmul(self.X,self.beta[q])
        C = np.zeros((len(self.X)))

        for j in range(len(self.X)):
            indices = [k for k in range(len(self.beta)) if k!=q]
            C[j] = np.exp(np.matmul(self.beta[indices],self.X[j])).sum()


        #Updating Z and lmbda
        self.Z[q] = self.__truncated_logistic(m - np.log(C), self.y[:,q].flatten())
        R2 = (self.Z[q] - m)**2

        lmbda_vec = np.array([self.__sample_lmbda(val) for val in R2]).flatten()
        self.lmbda[q] = np.diag(lmbda_vec)


    #Custom truncated logistic sampling
    def __truncated_logistic(self, means, Y):
        vec = []
        for t in range(len(Y)):
            if Y[t] == 1:
                U = np.random.uniform(low=self.__inv_logit(-means[t]), high = 1)
            else:
                U = np.random.uniform(low = 0, high=self.__inv_logit(-means[t]))

            vec.append(self.__logit(U) + means[t])

        return np.array(vec)



    #Assuming unimodality
    def __sample_lmbda(self, r2, size = 1, num_points = 1000,
                       radius = 5, epsilon_tol = 1e-4):

        #Creating the sampling grid around thae mode of the conditional distribution
        mode = minimize_scalar(fun = lambda x : -self.__conditional_lmbda_pdf(x, r2),
                               bounds = (0,1000), method = 'bounded').x
        grid = np.linspace(max(mode - radius, epsilon_tol), mode + radius, num_points)

        #Sampling from the grid
        probs = self.__conditional_lmbda_pdf(grid, r2)
        probs /= probs.sum()

        return np.random.choice(grid, p = probs, size = size)


    def __conditional_lmbda_pdf(self, lmbda, r2):
        #pi(lmbda)
        acc = np.log(self.__ks_density(.5*np.sqrt(lmbda))) -.5*np.log(lmbda)
        #l(lmbda)
        acc -= (.5*r2/lmbda + .5*np.log(lmbda))

        return np.exp(acc)

    #Unnormalized Kolmogorov-Smirnov Density
    def __ks_density(self,x, tol = 15):
        acc = 0
        for k in range(1,tol+1):
            acc += -8*k**2*(-1)**k*x*np.exp(-2*k**2*x**2)

        return np.max(x,0)

    #Multiplies a list of matrices in order, left to right
    def __multiply_list(self,*tup):
        acc = tup[0]
        for i in range(1,len(tup)):
            acc = np.matmul(acc,tup[i])
        return acc

    #Basic softmax function
    def __softmax(self,vec):
        acc = np.exp(vec)
        acc /= acc.sum()
        return acc

    #Basic logit function
    def __logit(self,x):
        return np.log(x/(1-x))

    #Basic inverse logit function
    def __inv_logit(self, x):
        return 1/(1+ np.exp(-x))

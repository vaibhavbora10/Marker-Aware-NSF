#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Factorization base class

@author: townesf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from os import path
from math import ceil
from tensorflow import linalg as tfl
import tensorflow_probability.python.math.psd_kernels as tfk
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from models import likelihoods
from utils import misc, nnfu
from models.marker_kernel import MarkerAwareKernel
tfd = tfp.distributions
tfb = tfp.bijectors
tv = tfp.util.TransformedVariable
tfk = tfp.math.psd_kernels

dtp = "float32"
rng = np.random.default_rng()

class SpatialFactorization(tf.Module):
  def __init__(self, J, L, Z, lik="poi", psd_kernel=tfk.MaternThreeHalves,
               nugget=1e-5, length_scale=0.1, disp="default", marker_dim=None,
               nonneg=False, isotropic=True, feature_means=None, ml_weight=1.0, **kwargs):
    """
    Non-negative process factorization

    Parameters
    ----------
    J : integer scalar
        Number of features in the multivariate outcome.
    L : integer scalar
        Number of desired latent Gaussian processes.
    T : integer scalar
        Number of desired non-spatial latent factors
    Z : 2D numpy array
        Coordinates of inducing point locations in input space.
        First dimension: 'M' is number of inducing points.
        Second dimension: 'D' is dimensionality of input to GP.
        More inducing points= slower but more accurate inference.
    lik: likelihood (Poisson or Gaussian)
    disp: overdispersion parameter initialization.
    --for Gaussian likelihood, the scale (stdev) parameter
    --for negative binomial likelihood, the parameter phi such that
      var=mean+phi*mean^2, i.e. phi->0 corresponds to Poisson, phi=1 to geometric
    --for Poisson likelihood, this parameter is ignored and set to None
    psd_kernel : an object of class PositiveSemidefiniteKernel, must accept a
        length_scale parameter.
    feature_means : if input data is centered (for lik='gau', nonneg=False)
    """
    super().__init__(**kwargs)
    self.ml_weight = ml_weight
    self.lik = lik
    self.isotropic=isotropic
    M,D = Z.shape
    self.Z = tf.Variable(Z, trainable=False, dtype=dtp, name="inducing_points")
    self.nonneg = tf.Variable(nonneg, trainable=False, name="is_non_negative")
    with tf.name_scope("variational"):
      self.delta = tf.Variable(rng.normal(size=(L,M)), dtype=dtp, name="mean")
      _Omega_tril = self._init_Omega_tril(L,M,nugget=nugget)
      self.Omega_tril=tv(_Omega_tril, tfb.FillScaleTriL(), dtype=dtp, name="covar_tril")
      if tf.rank(self.Omega_tril) == 4:
        self.Omega_tril = tf.squeeze(self.Omega_tril, axis=0)
      tf.print("self.Omega_tril shape:", tf.shape(self.Omega_tril))
    with tf.name_scope("gp_mean"):
      if self.nonneg:
        prior_mu, prior_sigma = misc.lnormal_approx_dirichlet(max(L,1.1))
        self.beta0 = tf.Variable(prior_mu*tf.ones((L,1)), dtype=dtp,
                                 name="intercepts")
      else:
        self.beta0 = tf.Variable(tf.zeros((L,1)), dtype=dtp, name="intercepts")
      self.beta = tf.Variable(tf.zeros((L,D)), dtype=dtp, name="slopes")
    with tf.name_scope("gp_kernel"):
      self.nugget = tf.Variable(nugget, dtype=dtp, trainable=False, name="nugget")
      self.amplitude = tv(np.tile(1.0,[L]), tfb.Softplus(), dtype=dtp, name="amplitude")
      self._ls0 = length_scale
      if isinstance(psd_kernel, MarkerAwareKernel):
        if marker_dim is None:
            raise ValueError("marker_dim must be provided for MarkerAwareKernel.")
        spatial_dims = D - marker_dim
        self.kernel = psd_kernel
      elif psd_kernel == MarkerAwareKernel:
          if marker_dim is None:
              raise ValueError("marker_dim must be provided for MarkerAwareKernel.")
          spatial_dims = D - marker_dim
          base_kernel = MarkerAwareKernel(spatial_dims=spatial_dims,
                                          marker_dims=marker_dim,
                                          amplitude=self.amplitude,
                                          length_scale=1.0)
          self.kernel = tfk.FeatureScaled(base_kernel, self.scale_diag)


      if self.isotropic:
          self.length_scale = tv(np.tile(self._ls0, [L]), tfb.Softplus(), dtype=dtp, name="length_scale")
      else:
          if marker_dim is None:
              raise ValueError("marker_dim must be set when using anisotropic kernel with marker features.")
          spatial_scales = np.ones(2) * self._ls0
          marker_scales = np.ones(marker_dim) * self._ls0
          self.scale_diag = tv(
              np.tile(np.concatenate([spatial_scales, marker_scales]), [L, 1]),
              tfb.Softplus(), dtype=dtp, name="scale_diag"
          )
          self.kernel = tfk.FeatureScaled(psd_kernel(amplitude=self.amplitude), self.scale_diag)

    #Loadings weights
    if self.nonneg:
      self.W = tf.Variable(rng.exponential(size=(J,L)), dtype=dtp,
                           constraint=misc.make_nonneg, name="loadings")
    else:
      self.W = tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings")
    self.psd_kernel = psd_kernel #this is a class, not yet an object
    #likelihood parameters, set defaults
    self._disp0 = disp
    self._init_misc()
    self.Kuu_chol = tf.Variable(self.eval_Kuu_chol(self.get_kernel()), dtype=dtp, trainable=False)
    if self.lik=="gau" and not self.nonneg:
      self.feature_means = feature_means
    else:
      self.feature_means = None
    self.ml_weight = tf.constant(ml_weight, dtype=tf.float32)  # Default value; make tunable if needed
    self.ml_ind1 = None
    self.ml_ind2 = None
    print("Z shape:", Z.shape)  # Inducing points
  def set_marker_constraints(self, ml_ind1, ml_ind2):
    """
    Store must-link index arrays as constant tensors in the model.
    ml_ind1, ml_ind2: 1D arrays or lists of indices (must-link pairs).
    """
    M = self.Z.shape[0]
    ml_pairs = [(int(i), int(j)) for i, j in zip(ml_ind1, ml_ind2) if i < M and j < M]
    if ml_pairs:
        inds1_Z, inds2_Z = zip(*ml_pairs)
    else:
        inds1_Z, inds2_Z = [], []
    idx_pairs = tf.concat([
        tf.stack([inds1_Z, inds2_Z], axis=1),
        tf.stack([inds2_Z, inds1_Z], axis=1)
    ], axis=0)
    mask_vals = tf.ones((tf.shape(idx_pairs)[0],), dtype=tf.bool)
    full_mask_Z = tf.scatter_nd(idx_pairs, mask_vals, shape=(M, M))
    if isinstance(self.kernel, MarkerAwareKernel):
      self.kernel.full_must_link_mask = full_mask_Z
  @staticmethod
  def _init_Omega_tril(L, M, nugget=None):
    """
    convenience function for initializing the batch of lower triangular
    cholesky factors of the variational covariance matrices.
    L: number of latent dimensions (factors)
    M: number of inducing points
    """
    Omega_sqt = 0.01*rng.normal(size=(L,M,M))
    Omega = [Omega_sqt[l,:,:]@ Omega_sqt[l,:,:].T for l in range(L)]
    res = np.stack([np.linalg.cholesky(Omega[l]) for l in range(L)], axis=0)
    return res.astype(dtp)

  def _init_misc(self):
    """
    misc initialization shared between __init__ and reset
    """
    J = self.W.shape[0]
    self.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)
    self.trvars_kernel = tuple(i for i in self.trainable_variables if i.name[:10]=="gp_kernel/")
    self.trvars_nonkernel = tuple(i for i in self.trainable_variables if i.name[:10]!="gp_kernel/")
    if self.isotropic:
        self.kernel = self.psd_kernel(amplitude=self.amplitude, length_scale=self.length_scale)
    else:
      self.kernel = tfk.FeatureScaled(self.psd_kernel(amplitude=self.amplitude), self.scale_diag)

  def get_dims(self):
    return self.W.shape[1]

  def get_loadings(self):
    return self.W.numpy()

  def set_loadings(self,Wnew):
    self.W.assign(Wnew,read_value=False)

  def init_loadings(self,Y,X=None,sz=1,**kwargs):
    """
    Use either PCA or NMF to initialize the loadings matrix from data Y
    """
    if self.nonneg:
      init_nsf_with_nmf(self,Y,X=X,sz=sz,**kwargs)
    else: #real-valued factors
      if self.lik in ("poi","nb"):
        pass #use GLM-PCA?
      elif self.lik=="gau":
        L = self.W.shape[1]
        fit = TruncatedSVD(L).fit(Y)
        self.set_loadings(fit.components_.T)
      else:
        raise likelihoods.InvalidLikelihoodError

  def generate_pickle_path(self,sz,base=None):
    """
    sz : str
      Indicate what type of size factors are used (eg 'none' or 'scanpy').
    base : str, optional
      Parent directory for saving pickle files. The default is cwd.
    """
    pars = {"L":self.W.shape[1], "lik":self.lik, "sz":sz,
            "model":"NSF" if self.nonneg else "RSF",
            "kernel":self.psd_kernel.__name__,
            "M":self.Z.shape[0]
            }
    pth = misc.params2key(pars)
    if base: pth = path.join(base,pth)
    return pth

  def get_kernel(self):
    return self.kernel

  def eval_Kuu_chol(self, kernel=None):
    if kernel is None:
      kernel = self.get_kernel()
    M,D = self.Z.shape
    base_kernel = kernel.kernel if isinstance(kernel, tfk.FeatureScaled) else kernel
    if isinstance(base_kernel, MarkerAwareKernel):
      base_kernel.set_mask_for_indices(tf.range(M))
    if hasattr(kernel, "set_mask_for_indices"):
        M = tf.shape(self.Z)[0]
        kernel.set_mask_for_indices(tf.range(M), tf.range(M))
    Kuu = kernel.matrix(self.Z, self.Z) + self.nugget*tf.eye(M)
    return tfl.cholesky(Kuu)

  def get_Kuu_chol(self, kernel=None, from_cache=False):
    if not from_cache:
      Kuu_chol = self.eval_Kuu_chol(kernel=kernel)
      self.Kuu_chol.assign(Kuu_chol)
      return Kuu_chol
    else:
      return self.Kuu_chol

  def get_mu_z(self):
    return self.beta0+tfl.matmul(self.beta, self.Z, transpose_b=True) #LxM

  def sample_latent_GP_funcs(self, X, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
    """
    Draw random samples of the latent variational GP function values "F"
    based on spatial coordinates X.
    The sampling comes from the variational approximation to the posterior.
    This function is needed to compute the expected log-likelihood term of the
    ELBO.
    X is a numpy array with shape NxD
    N=number of observations
    D=number of spatial dimensions
    The first dimension can be observations from a minibatch.
    S is the number of random samples to draw from latent GPs
    logscale: if False the function vals are exponentiated before returning
    i.e. they are positive valued random functions.
    If logscale=True (default), the functions are real-valued
    """
    if kernel is None:
      kernel = self.get_kernel()
    if mu_z is None:
      mu_z = self.get_mu_z()
    if Kuu_chol is None:
      Kuu_chol = self.get_Kuu_chol(kernel=kernel, from_cache=(not chol))
      tf.print("Kuu_chol shape:", tf.shape(Kuu_chol))
    if tf.rank(Kuu_chol) == 4:
      Kuu_chol = tf.squeeze(Kuu_chol, axis=0)
      tf.print("Kuu_chol shape:", tf.shape(Kuu_chol))
    N = tf.shape(X)[0]
    L = self.W.shape[1]
    mu_x = self.beta0+tfl.matmul(self.beta, X, transpose_b=True)
    if hasattr(kernel, "set_mask_for_indices"):
      N = tf.shape(X)[0]
      idx = tf.range(N)
      kernel.set_mask_for_indices(idx, idx)
    Kuf = kernel.matrix(self.Z, X)
    if tf.rank(Kuf) == 4:
        Kuf = tf.squeeze(Kuf, axis=0)
    tf.print("Kuf shape:", tf.shape(Kuf))
    Kff = kernel.matrix(X, X)
    Kff = tf.cond(
        tf.equal(tf.rank(Kff), 4),
        lambda: tf.squeeze(Kff, axis=0),
        lambda: Kff
    )
    Kff_diag = tf.linalg.diag_part(Kff)
    alpha_x = tfl.cholesky_solve(Kuu_chol, Kuf)
    if tf.rank(alpha_x) == 4:
        alpha_x = tf.squeeze(alpha_x, axis=0)
    tf.print("alpha_x shape:", tf.shape(alpha_x))
    mu_tilde = mu_x + tfl.matvec(alpha_x, self.delta-mu_z, transpose_a=True)
    aKa = tf.reduce_sum(tf.square(tf.linalg.matmul(Kuu_chol, alpha_x)), axis=1)
    aOmega_a = tf.reduce_sum(tf.square(tf.linalg.matmul(self.Omega_tril, alpha_x)), axis=1)
    tf.print("Kff_diag shape:", tf.shape(Kff_diag))
    tf.print("aKa shape:", tf.shape(aKa))
    tf.print("aOmega_a shape:", tf.shape(aOmega_a))
    Sigma_tilde = Kff_diag - aKa + aOmega_a
    eps = tf.random.normal((S,L,N))
    return mu_tilde + tf.math.sqrt(Sigma_tilde)*eps

  def sample_predictive_mean(self, X, sz=1, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
    """
    See sample_latent_variational_GP_funcs for X,S definitions
    sz is a tensor of shape (N,1) of size factors.
    Typically sz would be the rowSums or rowMeans of the outcome matrix Y.
    """
    F = self.sample_latent_GP_funcs(X, S=S, kernel=kernel, mu_z=mu_z,
                                    Kuu_chol=Kuu_chol, chol=chol)
    if self.nonneg:
      Lam = tfl.matrix_transpose(tfl.matmul(self.W, tf.exp(F)))
      if self.lik=="gau":
        return Lam
      else:
        sz = tf.convert_to_tensor(sz, dtype=Lam.dtype)
        sz = tf.reshape(sz, [1, -1, 1])
        return sz * Lam
    else:
      Lam = tfl.matrix_transpose(tfl.matmul(self.W, F))
      if self.lik=="gau":
        return Lam
      else:
        sz = tf.convert_to_tensor(sz, dtype=Lam.dtype)
        sz = tf.reshape(sz, [1, -1, 1])
        return tf.exp(tf.math.log(sz) + Lam)


  def eval_kl_term(self, mu_z, Kuu_chol):
    """
    KL divergence from the prior distribution to the variational distribution.
    This is one component of the ELBO:
    ELBO=expected log-likelihood - sum(kl_terms)
    qpars: a tuple containing (mu_z,Kuu_chol)
    qpars can be obtained by calling self.get_variational_params()
    mu_z is the GP mean function at all inducing points (dimension: LxM)
    Kuu_chol is the cholesky lower triangular of the kernel matrix of all inducing points.
    Its dimension is LxMxM
    where L = number of latent dimensions and M = number of inducing points.
    """
    qu = tfd.MultivariateNormalTriL(loc=self.delta, scale_tril=self.Omega_tril)
    pu = tfd.MultivariateNormalTriL(loc=mu_z, scale_tril=Kuu_chol)
    return qu.kl_divergence(pu) #L-vector

  # @tf.function
  def elbo_avg(self, X, Y, sz=1, S=1, Ntot=None, chol=True, ml_ind1=None, ml_ind2=None):

    """
    Parameters
    ----------
    X : numpy array of spatial coordinates (NxD)
        **OR** a tuple of spatial coordinates, multivariate outcomes,
        and size factors (convenient for minibatches from tensor slices)
    Y : numpy array of multivariate outcomes (NxJ)
        If Y is None then X must be a tuple of length three
    sz : size factors, optional
        vector of length N, typically the rowSums or rowMeans of Y.
        If X is a tuple then this is ignored as sz is expected in the third
        element of the X tuple.
    S : integer, optional
        Number of random GP function evaluations to use. The default is 1.
        Larger S=more accurate approximation to true ELBO but slower
    Ntot : total number of observations in full dataset
        This is needed when X,Y,sz are a minibatch from the full data
        If Ntot is None, we assume X,Y,sz provided are the full data NOT a minibatch.

    Returns
    -------
    The numeric evidence lower bound value, divided by Ntot.
    """
    batch_size, J = Y.shape
    print("X shape:", X.shape)
    if Ntot is None: Ntot = batch_size
    ker = self.get_kernel()
    mu_z = self.get_mu_z()
    Kuu_chol = self.get_Kuu_chol(kernel=ker,from_cache=(not chol))
    kl_term = tf.reduce_sum(self.eval_kl_term(mu_z, Kuu_chol))
    Mu = self.sample_predictive_mean(X, sz=sz, S=S, kernel=ker, mu_z=mu_z, Kuu_chol=Kuu_chol)
    eloglik = likelihoods.lik_to_distr(self.lik, Mu, self.disp).log_prob(Y)
    ml_loss = tf.constant(0.0, dtype=eloglik.dtype)
    if ml_ind1 is not None and ml_ind2 is not None:
        F = self.sample_latent_GP_funcs(X, S=1, kernel=ker, mu_z=mu_z, Kuu_chol=Kuu_chol)
        F = tf.reduce_mean(F, axis=0)
        F = tf.transpose(F)
        F1 = tf.gather(F, ml_ind1, axis=0)
        F2 = tf.gather(F, ml_ind2, axis=0)
        ml_loss = tf.reduce_mean(tf.reduce_sum(tf.square(F1 - F2), axis=1))
        eloglik -= self.ml_weight * ml_loss
    return J*tf.reduce_mean(eloglik) - kl_term/Ntot - self.ml_weight * ml_loss

  def train_step(self, D, optimizer, optimizer_k, S=1, Ntot=None, chol=True):
    """
    Executes one training step and returns the loss.
    D is training data: a tensorflow dataset (from slices) of (X,Y,sz)
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape(persistent=True) as tape:
      loss = -self.elbo_avg(D["X"], D["Y"], sz=D["sz"], S=S, Ntot=Ntot, chol=chol)
    try:
      gradients = tape.gradient(loss, self.trvars_nonkernel)
      if chol:
        gradients_k = tape.gradient(loss, self.trvars_kernel)
        optimizer_k.apply_gradients(zip(gradients_k, self.trvars_kernel))
      optimizer.apply_gradients(zip(gradients, self.trvars_nonkernel))
    finally:
      del tape
    return loss

  def validation_step(self, D, S=1, chol=False):
    """
    Compute the validation loss on held-out data D
    D is a tensorflow dataset (from slices) of (X,Y,sz)
    """
    return -self.elbo_avg(D["X"], D["Y"], sz=D["sz"], S=S, chol=chol)

  def predict(self, Dtr, Dval=None, S=10):
    """
    Here Dtr,Dval should be raw counts (not normalized or log-transformed)

    returns the predicted training data mean and validation data mean
    on the original count scale
    """
    Mu_tr = misc.t2np(self.sample_predictive_mean(Dtr["X"], sz=Dtr["sz"], S=S))
    if self.lik=="gau":
      sz_tr = Dtr["Y"].sum(axis=1)
      misc.reverse_normalization(Mu_tr, feature_means=self.feature_means,
                            transform=np.expm1, sz=sz_tr, inplace=True)
    if Dval:
      Mu_val = misc.t2np(self.sample_predictive_mean(Dval["X"], sz=Dval["sz"], S=S))
      if self.lik=="gau":
        sz_val = Dval["Y"].sum(axis=1)
        misc.reverse_normalization(Mu_val, feature_means=self.feature_means,
                              transform=np.expm1, sz=sz_val, inplace=True)
    else:
      Mu_val = None
    return Mu_tr,Mu_val

def smooth_spatial_factors(F,Z,X=None):
  """
  F: real-valued factors (ie on the log scale for NSF)
  Z: inducing point locations
  X: spatial coordinates
  """
  M = Z.shape[0]
  if X is None:
    beta0 = F.mean(axis=0)
    U = np.tile(beta0,[M,1])
    beta = None
  else:
    lr = LinearRegression().fit(X,F)
    beta0 = lr.intercept_
    beta = lr.coef_
    nn = max(2, ceil(X.shape[0]/M))
    knn = KNeighborsRegressor(n_neighbors=nn).fit(X,F)
    U = knn.predict(Z)
  return U,beta0,beta

def init_nsf_with_nmf(fit, Y, X=None, sz=1, pseudocount=1e-2, factors=None,
                      loadings=None, shrinkage=0.2):
  L = fit.W.shape[1]
  kw = likelihoods.choose_nmf_pars(fit.lik)
  F,W = nnfu.regularized_nmf(Y, L, sz=sz, pseudocount=pseudocount,
                             factors=factors, loadings=loadings,
                             shrinkage=shrinkage, **kw)
  # eF = factors
  # W = loadings
  # if eF is None or W is None:
  #   kw = likelihoods.choose_nmf_pars(fit.lik)
  #   nmf = NMF(L,**kw)
  #   eF = nmf.fit_transform(Y)#/sz
  #   W = nmf.components_.T
  # W = postprocess.shrink_loadings(W, shrinkage=shrinkage)
  # wsum = W.sum(axis=0)
  # eF = postprocess.shrink_factors(eF*wsum, shrinkage=shrinkage)
  # F = np.log(pseudocount+eF)-np.log(sz)
  # beta0 = fit.beta0.numpy().flatten()
  # wt_to_W = F.mean(axis=0)- beta0
  # F-= wt_to_W
  # W*= np.exp(wt_to_W-np.log(wsum))
  # W,wsum = normalize_cols(W)
  # eF *= wsum
  # eFm2 = eF.mean()/2
  # eF/=eFm2
  # W*=eFm2
  # F = np.log(pseudocount+eF)
  fit.set_loadings(W)
  U,beta0,beta = smooth_spatial_factors(F,fit.Z.numpy(),X=X)
  # if X is None: #no spatial coordinates, just use the mean
  #   beta0 = F.mean(axis=0)
  #   U = np.tile(beta0,[M,1])
  # else: #spatial coordinates
  #   lr = LinearRegression().fit(X,F)
  #   beta0 = lr.intercept_
  #   fit.beta.assign(lr.coef_,read_value=False)
  #   nn = max(2, ceil(X.shape[0]/M))
  #   knn = KNeighborsRegressor(n_neighbors=nn).fit(X,F)
  #   U = knn.predict(fit.Z.numpy())
  fit.beta0.assign(beta0[:,None],read_value=False)
  fit.delta.assign(U.T,read_value=False)
  if beta is not None: fit.beta.assign(beta,read_value=False)
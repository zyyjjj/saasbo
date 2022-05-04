import time, os, itertools, math
from tkinter import YES
import warnings
from copy import deepcopy

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import value_and_grad
from jax.scipy.stats import norm
from numpyro.util import enable_x64
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import qmc
import matplotlib
import matplotlib.pyplot as plt
import pdb
import torch

from hartmann import hartmann6_50, hartmann6_1000
from saasgp import SAASGP


# pass in a sequence of dimensions to perturb
Hartmann6_1000_perturb_sequence = [
    [1], [7], [11], [23], [47], [33],
    list(set(range(1000)) - set([1, 7, 11, 23, 47, 33]))
]
LB, UB = np.zeros(1000), np.ones(1000)

def fit_saasgp_stepwise(f, lb, ub, 
    perturb_dims_sequence, 
    true_important_dims, 
    perturb_direction = 'ub_only'):

    input_dim = len(lb)
    X = np.expand_dims((lb + ub)/2, 0)
    Y = np.array([f(x) for x in X])
    
    ell_median_history = []
    ell_rank_history = []

    # ell_median, ell_rank = fit_one_gp(X, Y)
    # ell_median_history.append(ell_median)
    # ell_rank_history.append(ell_rank)

    # print('ranking of median lengthscales of important dimensions', ell_rank[true_important_dims])

    for perturb_dims in perturb_dims_sequence:
        print('perturb dims ', perturb_dims)

        x = (lb + ub)/2
        x[jnp.array(perturb_dims)] = deepcopy(ub[jnp.array(perturb_dims)])
        y = f(x)
        print('output value ', y)
        pdb.set_trace()
        
        X = np.vstack((X, jnp.expand_dims(x, 0)))
        Y = np.hstack((Y, y))

        ell_median, ell_rank = fit_one_gp(X, Y)
        ell_median_history.append(ell_median)
        ell_rank_history.append(ell_rank)

        print('ranking of median lengthscales of important dimensions', 
            ell_rank[jnp.array(true_important_dims)])


def fit_one_gp(X, Y,
    alpha=0.1,
    num_warmup=512,
    num_samples=512,
    thinning=16,
    kernel="rbf",
    ):

    train_Y = (Y - Y.mean()) / Y.std()
    gp = SAASGP(
            alpha=alpha,
            num_warmup=num_warmup,
            num_samples=num_samples,
            max_tree_depth=6,
            num_chains=1,
            thinning=thinning,
            verbose=False,
            observation_variance=1e-6,
            kernel=kernel,
        )
    
    # fit SAAS GP to training data
    gp = gp.fit(X, train_Y)
    # get the NUTS samples of inverse squared length scales for each dimension, thinned out
    ell = 1.0 / jnp.sqrt(gp.flat_samples["kernel_inv_length_sq"][::gp.thinning])
    # record important statistics: median, average rank of each dimension?
    ell_median = jnp.median(ell, 0)
    ell_rank = jnp.argsort(ell).mean(0)

    return ell_median, ell_rank


if __name__ == '__main__':
    fit_saasgp_stepwise(hartmann6_1000, lb = LB, ub = UB, 
    perturb_dims_sequence = Hartmann6_1000_perturb_sequence, 
    true_important_dims =  [1, 7, 11, 23, 47, 33] )
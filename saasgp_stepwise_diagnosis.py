import time, os, itertools, math, sys
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
from branin import branin100
from saasgp import SAASGP


# pass in a sequence of dimensions to perturb
Hartmann6_1000_perturb_sequence = [
    [1], [7], [11], [23], [47], [33],
    list(set(range(1000)) - set([1, 7, 11, 23, 47, 33]))
]

Hartmann6_50_perturb_sequence = [
    [1], [7], [11], [23], [47], [33],
    list(set(range(50)) - set([1, 7, 11, 23, 47, 33]))
]

Branin_100_perturb_sequence = [
    [17], [87], list(set(range(100)) - set([17, 87]))
]

LB_1000, UB_1000 = np.zeros(1000), np.ones(1000)
LB_50, UB_50 = np.zeros(50), np.ones(50)

Branin_LB_100 = np.hstack((-5 * np.ones(50), 0 * np.ones(50)))   # lower bounds for input domain
Branin_UB_100 = np.hstack((10 * np.ones(50), 15 * np.ones(50)))  # upper bounds for input domain
  

def fit_saasgp_stepwise(f, lb, ub, 
    true_important_dims, 
    save_folder,
    dim,
    perturb_dims_sequence, 
    is_sobol = False,
    n_sobol_samples = None):

    if dim < 1000:
        num_warmup = 512
        num_samples = 256
    else:
        num_warmup = 1024
        num_samples = 4096

    def print_and_save_results(f, out, is_sobol, save_folder, top_ells, ell_rank, ell_median, ell_sd, true_important_dims, perturb_dims):
        
        print('top inferred important dimensions',
            top_ells[:len(true_important_dims)])
        print('ranking of median lengthscales of important dimensions', 
            ell_rank[jnp.array(true_important_dims)])
        print('median lengthscales of important dimensions',
            ell_median[jnp.array(true_important_dims)])
        print('standard deviation of lengthscales of important dimensions',
            ell_sd[jnp.array(true_important_dims)])

        out.append({'eval_idx': i, 'perturb_dims': perturb_dims, 
            'top': top_ells[:len(true_important_dims)],
            'ranking of median lthscale of imptt dims': ell_rank[jnp.array(true_important_dims)],
            'median lthscale of imptt dims': ell_median[jnp.array(true_important_dims)],
            'sd of lthscale of imptt dims': ell_sd[jnp.array(true_important_dims)]
            })

        if is_sobol:
            torch.save(out, save_folder + 'test_sobol_' + f.__name__)
        else:
            torch.save(out, save_folder + 'test_' + f.__name__)

    out = []
    X = None

    if is_sobol:
        for i in range(n_sobol_samples):
            x = qmc.Sobol(len(lb), scramble=True).random(1)
            x = qmc.scale(x, lb, ub)
            # pdb.set_trace()
            if X is None:
                X = x 
            else:
                X = np.vstack((X, x))
            Y = np.array([f(x) for x in X])
            if i > 1:
                top_ells, ell_median, ell_rank, ell_sd = fit_one_gp(X, Y, 
                                num_warmup = num_warmup, num_samples = num_samples)
                
                print_and_save_results(f, out, is_sobol, save_folder, top_ells, ell_rank, ell_median, ell_sd, true_important_dims, perturb_dims = None)

    else:
        X = np.expand_dims((lb + ub)/2, 0)
        Y = np.array([f(x) for x in X])

        for i in range(len(perturb_dims_sequence)):

            perturb_dims = perturb_dims_sequence[i]
            print('perturb dims ', perturb_dims)

            x = (lb + ub)/2
            if (i // (len(true_important_dims)+1)) % 2 == 0:
                x[jnp.array(perturb_dims)] = deepcopy(ub[jnp.array(perturb_dims)])
            else:
                x[jnp.array(perturb_dims)] = deepcopy(lb[jnp.array(perturb_dims)])
            
            y = f(x)
            print('output value ', y)
            
            X = np.vstack((X, jnp.expand_dims(x, 0)))
            Y = np.hstack((Y, y))

            top_ells, ell_median, ell_rank, ell_sd = fit_one_gp(X, Y, num_warmup = num_warmup, num_samples = num_samples)
            
            print_and_save_results(f, out, is_sobol, save_folder, top_ells, ell_rank, ell_median, ell_sd, true_important_dims, perturb_dims = perturb_dims)


        


def fit_one_gp(X, Y,
    alpha=0.1,
    num_warmup=256,
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
    # pdb.set_trace()
    
    ell_median = jnp.median(ell, 0)
    ell_sd = jnp.std(ell, 0)
    ell_rank = jnp.argsort(jnp.argsort(ell_median))
    top_ells = jnp.argsort(ell_median)

    return top_ells, ell_median, ell_rank, ell_sd


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/hartmann6_1000/designed_sequence/"

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)

    PROBLEMS = ['hartmann_50', 'hartmann_1000', 'branin_100']
    problem_id = 2
    problem = PROBLEMS[problem_id]
    
    # sobol_args = {'is_sobol': True, 'n_sobol_samples': 50}
    sobol_args = {'is_sobol': False, 'n_sobol_samples' :None}

    if problem == 'hartmann_1000':

        fit_saasgp_stepwise(hartmann6_1000, lb = LB_1000, ub = UB_1000, 
        save_folder = results_folder,
        dim = 1000,
        perturb_dims_sequence = Hartmann6_1000_perturb_sequence * 5, 
        true_important_dims =  [1, 7, 11, 23, 47, 33])
    
    elif problem == 'hartmann_50':
        fit_saasgp_stepwise(hartmann6_50, lb = LB_50, ub = UB_50, 
        save_folder = results_folder,
        dim = 50,
        perturb_dims_sequence = Hartmann6_50_perturb_sequence * 5, 
        true_important_dims =  [1, 7, 11, 23, 47, 33] )
    
    elif problem == 'branin_100':
        fit_saasgp_stepwise(branin100, lb = Branin_LB_100, ub = Branin_UB_100, 
        save_folder = results_folder,
        dim = 100,
        perturb_dims_sequence = Branin_100_perturb_sequence * 10, 
        true_important_dims =  [17, 87],
        **sobol_args )
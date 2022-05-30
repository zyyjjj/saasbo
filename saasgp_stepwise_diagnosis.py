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
from itertools import chain

from hartmann import hartmann6_50, hartmann6_1000
from branin import branin100
from synthetic_linear import linear2_100
from saasgp import SAASGP

# TODO: enable setting to points other than middle and boundary -- is there a good way to unit-test this idea 
# with minimal efforts? -- could just generate training data and plug into botorch?


# pass in a sequence of dimensions to perturb
Hartmann6_1000_perturb_sequence = [
    [1], [7], [11], [23], [47], [33],
    # list(set(range(1000)) - set([1, 7, 11, 23, 47, 33]))
]

Hartmann6_50_perturb_sequence = [
    # list(set(range(50)) - set([1, 7, 11, 23, 47, 33])),
    [1], [7], [11], [23], [47], [33],
]

Branin_100_perturb_sequence = [
    # list(set(range(100)) - set([17, 87])),
    [17], [87], 
]


LB_1000, UB_1000 = np.zeros(1000), np.ones(1000)
LB_100, UB_100 = np.zeros(100), np.ones(100)
LB_50, UB_50 = np.zeros(50), np.ones(50)

# Branin_LB_100 = np.hstack((-5 * np.ones(50), 0 * np.ones(50)))   # lower bounds for input domain
# Branin_UB_100 = np.hstack((10 * np.ones(50), 15 * np.ones(50)))  # upper bounds for input domain
  

def fit_saasgp_stepwise(f, lb, ub, 
    true_important_dims, 
    save_folder,
    dim,
    perturb_dims_sequence, 
    perturb_strategy, # one of {'ub', 'ub_lb', 'sobol'}
    is_fully_sobol = False,
    n_sobol_samples = None,
    perturb_batch_size = 1):

    if dim < 1000:
        num_warmup = 512
        num_samples = 1024
    else:
        num_warmup = 1024
        num_samples = 4096

    def print_and_save_results(f, out, is_fully_sobol, save_folder, top_ells, ell_rank, ell_median, ell_sd, true_important_dims, perturb_dims):
        
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

        if is_fully_sobol:
            torch.save(out, save_folder + 'test_sobol_' + f.__name__)
        else:
            torch.save(out, save_folder + 'test_' + f.__name__)

    out = []
    X = None

    if is_fully_sobol:
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
                
                print_and_save_results(f, out, is_fully_sobol, save_folder, top_ells, ell_rank, ell_median, ell_sd, true_important_dims, perturb_dims = None)

    else:
        X = np.expand_dims((lb + ub)/2, 0)
        Y = np.array([f(x) for x in X])

        for i in range(len(perturb_dims_sequence)):

            perturb_dims = perturb_dims_sequence[i]
            print('perturb dims ', perturb_dims)

            # x = (lb + ub)/2 # replicate for perturb_batch_size number of times
            x = np.repeat(jnp.expand_dims((lb + ub)/2,0), perturb_batch_size, axis = 0)

            # a few options here for perturbation:
            if perturb_strategy == 'ub':
                # 1. set to upper bound or lower bound of that dimension of the domain
                x[jnp.array(perturb_dims)] = deepcopy(ub[jnp.array(perturb_dims)])
            
            elif perturb_strategy == 'ub_lb':
                # 2. alternatively set to lower bound and upper bound of the domain
                if (i // (len(true_important_dims)+1)) % 2 == 0:
                    x[jnp.array(perturb_dims)] = deepcopy(ub[jnp.array(perturb_dims)])
                else:
                    x[jnp.array(perturb_dims)] = deepcopy(lb[jnp.array(perturb_dims)])
            
            elif perturb_strategy == 'sobol':
                # 3. set to a random point (or more points) of the domain
                
                x_perturb = qmc.Sobol(len(perturb_dims), scramble=True).random(perturb_batch_size)
                x_perturb = qmc.scale(x_perturb, lb[jnp.array(perturb_dims)], ub[jnp.array(perturb_dims)])

                # x.at[np.repeat(jnp.array(perturb_dims), perturb_batch_size, axis=0)].set(x_perturb)
                # for j in range(perturb_batch_size):
                    # x.at[j, perturb_dims].set(x_perturb[1].item()) 
                    # this should work for changing the value at one index, but not multiple
                    # how to change multiple indices of a jax array? 
                    # either dig, or write my own tedious help function
                    
                # create indices for perturbing
                idx_list_0 = [k for _ in range(len(perturb_dims)) for k in range(perturb_batch_size)]
                idx_list_1 = perturb_dims * perturb_batch_size
                # flatten the values to set to the indices
                x = x.at[idx_list_0, idx_list_1].set(x_perturb.flatten())

            pdb.set_trace()
            y = f(x)
            print('output value ', y)
            
            # X = np.vstack((X, jnp.expand_dims(x, 0)))
            X = np.vstack((X, x))
            Y = np.hstack((Y, y))

            top_ells, ell_median, ell_rank, ell_sd = fit_one_gp(X, Y, num_warmup = num_warmup, num_samples = num_samples)
            
            print_and_save_results(f, out, is_fully_sobol, save_folder, top_ells, ell_rank, ell_median, ell_sd, true_important_dims, perturb_dims = perturb_dims)


def fit_one_gp(X, Y,
    alpha=0.01,
    num_warmup=256,
    num_samples=512,
    thinning=16,
    kernel="rbf",
    ):

    # train_Y = (Y - Y.mean()) / Y.std()

    train_Y = Y

    gp = SAASGP(
            alpha=alpha,
            num_warmup=num_warmup,
            num_samples=num_samples,
            max_tree_depth=6,
            num_chains=1,
            thinning=thinning,
            verbose=True,
            observation_variance=1e-6,
            kernel=kernel,
        )
    
    # fit SAAS GP to training data
    gp = gp.fit(X, train_Y)
    
    # get the NUTS samples of inverse squared length scales for each dimension, thinned out
    ell = 1.0 / jnp.sqrt(gp.flat_samples["kernel_inv_length_sq"][::gp.thinning])
    potential_energy = gp.flat_extra_fields["potential_energy"][::gp.thinning]
    print('mean potential energy across NUTS samples: ', jnp.mean(potential_energy))
    
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

    PROBLEMS = ['hartmann_50', 'hartmann_1000', 'branin_100', 'linear2_100']
    problem_id = 2
    problem = PROBLEMS[problem_id]
    
    # sampling_args = {'is_fully_sobol': True, 'n_sobol_samples': 50}
    sampling_args = {
        'is_fully_sobol': False, 
        'n_sobol_samples': None, 
        'perturb_strategy': 'sobol',
        'perturb_batch_size': 2
        }

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
        perturb_dims_sequence = Hartmann6_50_perturb_sequence, 
        true_important_dims =  [1, 7, 11, 23, 47, 33],
        **sampling_args )
    
    elif problem == 'branin_100':
        fit_saasgp_stepwise(branin100, lb = LB_100, ub = UB_100, 
        save_folder = results_folder,
        dim = 100,
        perturb_dims_sequence = Branin_100_perturb_sequence, 
        true_important_dims =  [17, 87],
        **sampling_args )
    
    elif problem == 'linear2_100':
        fit_saasgp_stepwise(linear2_100, 
        lb = np.zeros(100), ub = np.ones(100), 
        save_folder = results_folder,
        dim = 100,
        perturb_dims_sequence = Branin_100_perturb_sequence, 
        true_important_dims =  [17, 87],
        **sampling_args )
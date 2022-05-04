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

from saasgp import SAASGP


def ei(x, y_target, gp, xi=0.0):
    # Expected Improvement (EI)
    mu, var = gp.posterior(x)
    # print('computing EI, mean and var at test point is ', mu, var)
    std = jnp.maximum(jnp.sqrt(var), 1e-6)
    improve = y_target - xi - mu
    scaled = improve / std
    cdf, pdf = norm.cdf(scaled), norm.pdf(scaled)
    exploit = improve * cdf
    explore = std * pdf
    values = jnp.nan_to_num(exploit + explore, nan=0.0)
    return values.mean(axis=0)


def ei_grad(x, y_target, gp, xi=0.0):
    # Gradient of EI
    return ei(x, y_target, gp, xi).sum()


def optimize_ei(gp, y_target, bounds, xi=0.0, num_restarts_ei=5, num_init=5000):
    # Helper function for optimizing EI
    def negative_ei_and_grad(x, y_target, gp, xi):
        # Compute EI and its gradient and then flip the signs since L-BFGS-B minimizes
        x = jnp.array(x.copy())[None, :]
        ei_val, ei_val_grad = value_and_grad(ei_grad)(x, y_target, gp, xi)
        return -1 * ei_val.item(), -1 * np.array(ei_val_grad)

    dim = gp.X_train.shape[-1]
    with warnings.catch_warnings(record=True):  # Suppress qmc.Sobol UserWarning
        X_rand = qmc.Sobol(dim, scramble=True).random(num_init)

    # Make sure x_best is in the set of candidate EI maximizers
    x_best = gp.X_train[gp.Y_train.argmin(), :]
    X_rand[0, :] = np.clip(x_best + 0.001 * np.random.randn(1, dim), a_min=0.0, a_max=1.0)
    X_rand = jnp.array(X_rand)

    ei_rand = ei(X_rand, y_target, gp)
    _, top_inds = lax.top_k(ei_rand, num_restarts_ei)
    X_init = X_rand[top_inds, :]

    x_best, y_best = None, -float("inf")
    for x0 in X_init:
        x, fx, _ = fmin_l_bfgs_b(
            func=negative_ei_and_grad,
            x0=x0,
            fprime=None,
            bounds=bounds,
            args=(y_target, gp, 0.0),
            maxfun=100,  # this limits computational cost
        )
        fx = -1 * fx  # Back to maximization

        if fx > y_best:
            x_best, y_best = x.copy(), fx

    return x_best


def generate_initial_evaluations(f, 
        lb, ub, 
        seed, 
        perturb_dims_protocol, # 'random' or 'dyadic'
        perturb_direction, # 'ub_only' or 'ub_and_lb'
        frac_perturb_dims,
        n_perturb_samples = None, n_sobol_samples = None, 
        true_important_dims = None
        ):

    input_dim = len(lb)
    num_important_dims_perturbed = []
    X = None

    np.random.seed(seed)

    if n_perturb_samples > 0:
    
        # first, evaluate the function at the midpoint
        X = np.expand_dims((lb + ub)/2, 0)
        
        # if perturbation protocol is random, perturb the specified number of samples
        if perturb_dims_protocol == 'random':
            for i in range(n_perturb_samples-1):
                x = (lb + ub)/2

                dims_to_perturb = np.random.choice(
                    np.arange(input_dim), 
                    size = int(frac_perturb_dims * input_dim), 
                    replace = False)
        
                x = np.expand_dims(perturb_dims(x, lb, ub, dims_to_perturb, perturb_direction), 0)
                X = np.vstack((X, x))
                num_important_dims_perturbed.append(len(set(dims_to_perturb).intersection(set(true_important_dims))))

        # if perturbation policy is dyadic, 
        elif perturb_dims_protocol == 'dyadic':
            dims_to_perturb_list = get_dyadic_dims(input_dim)
            for dims_to_perturb in dims_to_perturb_list:
                x = (lb + ub)/2
                x = np.expand_dims(perturb_dims(x, lb, ub, dims_to_perturb, perturb_direction), 0)
                X = np.vstack((X, x))
                num_important_dims_perturbed.append(len(set(dims_to_perturb).intersection(set(true_important_dims))))
        

    # generate the remaining samples from Sobol sequence
    with warnings.catch_warnings(record=True):  # suppress annoying qmc.Sobol UserWarning
        X_sobol = qmc.Sobol(len(lb), scramble=True, seed=seed).random(n_sobol_samples)

    if X is None:
        X = X_sobol
    else:
        X = np.vstack((X, X_sobol))
    # Y = np.array([f(lb + (ub - lb) * x) for x in X])
    Y = np.array([f(x) for x in X])

    return X, Y, num_important_dims_perturbed


def perturb_dims(x, lb, ub, dims_to_perturb, perturb_direction):

    if perturb_direction == 'ub_and_lb':
        for dim_to_perturb in dims_to_perturb:
            if np.random.randint(0,2) == 0:
                x[dim_to_perturb] = lb[dim_to_perturb]
            else:
                x[dim_to_perturb] = ub[dim_to_perturb]
    elif perturb_direction == 'ub_only':
        x[dims_to_perturb] = ub[dims_to_perturb]
    
    return x


def run_saasbo(
    f,
    lb,
    ub,
    max_evals,
    num_init_evals,
    results_folder,
    true_important_dims, 
    perturb_dims_protocol,
    perturb_direction,
    seed=None,
    alpha=0.1,
    num_warmup=512,
    num_samples=256,
    thinning=16,
    num_restarts_ei=5,
    kernel="rbf",
    device="cpu",
    frac_perturb = None, 
    frac_perturb_dims = None
    ):
    """
    Run SAASBO and approximately minimize f.

    Arguments:
    f: function to minimize. should accept a D-dimensional np.array as argument. the input domain of f
        is assumed to be the D-dimensional rectangular box bounded by lower and upper bounds lb and ub.
    lb: D-dimensional vector of lower bounds (np.array)
    ub: D-dimensional vector of upper bounds (np.array)
    max_evals: The total evaluation budget
    num_init_evals: The initial num_init_evals query points are chosen at random from the input
        domain using a Sobol sequence. must satisfy num_init_evals < max_evals.
    seed: Random number seed (int or None); defaults to None
    alpha: Positive float that controls the level of sparsity (smaller alpha => more sparsity).
        defaults to alpha = 0.1.
    num_warmup: The number of warmup samples to use in HMC inference. defaults to 512.
    num_samples: The number of post-warmup samples to use in HMC inference. defaults to 256.
    thinning: Positive integer that controls the fraction of posterior hyperparameter samples
        that are used to compute the expected improvement. for example thinning==2 will use every
        other sample. defaults to no thinning (thinning==1).
    num_restarts_ei: The number of restarts for L-BFGS-B when optimizing EI.
    kernel: By default saasbo uses rbf, but matern is also supported.
    device: Whether to use cpu or gpu. defaults to "cpu".

    Returns:
        X: np.array containing all query points (of which there are max_evals many)
        Y: np.array containing all observed function evaluations (of which there are max_evals many)
    """

    if max_evals < num_init_evals:
        raise ValueError("Must choose max_evals >= num_init_evals.")
    if lb.shape != ub.shape or lb.ndim != 1:
        raise ValueError("The lower/upper bounds lb and ub must have the same shape and be D-dimensional vectors.")
    if alpha <= 0.0:
        raise ValueError("The hyperparameter alpha must be positive.")
    if device not in ["cpu", "gpu"]:
        raise ValueError("The device must be cpu or gpu.")

    numpyro.set_platform(device)
    enable_x64()
    numpyro.set_host_device_count(1)

    max_exceptions = 3
    num_exceptions = 0

    ell_median_history = []
    ell_rank_history = []

    bounds = [(lb[i], ub[i]) for i in range(len(lb))]

    # generate initial queries
    if perturb_dims_protocol == 'random':
        n_perturb_samples = int(num_init_evals * frac_perturb)
        n_sobol_samples = num_init_evals - int(num_init_evals * frac_perturb)
    elif perturb_dims_protocol == 'dyadic':
        n_perturb_samples = int(math.log2(len(lb))) + 2
        n_sobol_samples = 0
        # TODO: look into this later -- number of initial samples for dyadic policy can also be tuned

    X, Y, num_important_dims_perturbed = generate_initial_evaluations(
        f = f, 
        lb = lb, 
        ub = ub, 
        seed = seed,
        perturb_dims_protocol = perturb_dims_protocol, 
        perturb_direction = perturb_direction,
        frac_perturb_dims = frac_perturb_dims, 
        n_perturb_samples = n_perturb_samples,
        n_sobol_samples = n_sobol_samples,
        true_important_dims = true_important_dims
        )

    print(f"First {len(Y)} queries drawn at random, {n_perturb_samples} perturbed using {perturb_dims_protocol} policy in direction {perturb_direction}. Best minimum thus far: {Y.min().item():.3f} at index {np.argmin(Y)}",\
        '\n generated Y values: ', Y,
        '\n number of important dimensions perturbed: ', num_important_dims_perturbed
        )


    num_evals = len(Y)

    while num_evals <= max_evals:
        # print(f"=== Number of function evaluations performed: {num_evals} ===", flush=True)
        # standardize training data
        train_Y = (Y - Y.mean()) / Y.std()
        y_target = train_Y.min().item()
    
        start = time.time()
        # define GP with SAAS prior
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
        print(f"GP fitting took {time.time() - start:.2f} seconds")

        if num_evals < max_evals:
            # use EI to generate a new point to evaluate
            try:
                start = time.time()
                # do EI optimization using LBFGS
                x_next = optimize_ei(gp=gp, y_target=y_target, bounds=bounds, xi=0.0, num_restarts_ei=num_restarts_ei, num_init=5000)
                print(f"Optimizing EI took {time.time() - start:.2f} seconds")

            # If for whatever reason we fail to return a query point above we choose one at random from the domain
            except Exception:
                num_exceptions += 1
                if num_exceptions <= max_exceptions:
                    print("WARNING: Exception was raised, using a random point.")
                    x_next = np.random.rand(len(lb))
                else:
                    raise RuntimeException("ERROR: Maximum number of exceptions raised!")

            # transform to original coordinates
            y_next = f(lb + (ub - lb) * x_next)

            X = np.vstack((X, deepcopy(x_next[None, :])))
            Y = np.hstack((Y, deepcopy(y_next)))
            print(f"Observed function value: {y_next:.3f}, Best function value seen thus far: {Y.min():.3f}")


        # get the NUTS samples of inverse squared length scales for each dimension, thinned out
        ell = 1.0 / jnp.sqrt(gp.flat_samples["kernel_inv_length_sq"][::gp.thinning])
        # record important statistics: median, average rank of each dimension?
        ell_median = jnp.median(ell, 0)
        ell_rank = jnp.argsort(ell).mean(0)

        print('ell_median shape', ell_median.shape)
        print('ell_rank shape', ell_rank.shape)

        ell_median_history.append(ell_median)
        ell_rank_history.append(ell_rank)

        # create directory for saving results
        if frac_perturb is not None:
            frac_perturb_str = str(frac_perturb).replace('.', '')
            frac_perturb_dims_str = str(frac_perturb_dims).replace('.', '')
            save_folder = results_folder + perturb_dims_protocol + '_' + \
                'initevals=%d_frac_initevals_perturbed=%s_frac_dims_perturbed=%s/' \
                    % (num_init_evals, frac_perturb_str, frac_perturb_dims_str)
        else:
            save_folder = results_folder + perturb_dims_protocol + '_' + \
                'initevals=%d/' % (num_init_evals)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # save results in a dictionary 'out'
        out = dict()
        out['X'] = X
        out['Y'] = Y
        out['median_lengthscales'] = ell_median_history
        out['lengthscale_rankings'] = ell_rank_history
        out['num_important_dims_perturbed'] = num_important_dims_perturbed

        torch.save(out, save_folder + perturb_direction + str(seed))
        
        trace_important_dims(gp, thinning)

        del gp  # Free memory

        num_evals += 1

    return lb + (ub - lb) * X, Y

def trace_important_dims(gp, thinning):

    ell = 1.0 / jnp.sqrt(gp.flat_samples["kernel_inv_length_sq"][::gp.thinning])
    ell_median = jnp.median(ell, 0)

    for i in ell_median.argsort()[:10]:
        print(f"Parameter {i:2} Median lengthscale = {ell_median[i]:.2e}")

def get_dyadic_dims(n_dims):
    """
    INPUT: n_dim, the number of total dimensions of the problem (e.g., 50 or 100)
    OUTPUT: a nested list of dimensions to perturb at each round; 
            the length of the list is equal to the length of the binary representation of n_dim
    """

    l = int(math.log2(n_dims)) + 1

    # generate list of length (l-1) strings with all possible combinations of 0's and 1's in each digit
    base_strings = list(''.join(comb) for comb in itertools.product('01', repeat = l-1))
    
    dyadic_dims = []
    for loc in range(l):
        # insert '1' into location loc in each string in base_digits
        # then convert to int
        
        single_set = []
        
        for base_string in base_strings:
            dim_after_insertion = int(''.join((base_string[:loc], '1', base_string[loc:])), 2)
            if dim_after_insertion < n_dims:
                single_set.append(dim_after_insertion)
        
        dyadic_dims.append(single_set)
    
    return dyadic_dims


# if __name__ == '__main__':
#     print(get_dyadic_dims(50))
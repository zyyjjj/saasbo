import time
from tkinter import YES
import warnings
from copy import deepcopy
import os

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


def optimize_ei(gp, y_target, xi=0.0, num_restarts_ei=5, num_init=5000):
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
            bounds=[(0.0, 1.0) for _ in range(dim)],
            args=(y_target, gp, 0.0),
            maxfun=100,  # this limits computational cost
        )
        fx = -1 * fx  # Back to maximization

        if fx > y_best:
            x_best, y_best = x.copy(), fx

    return x_best


def generate_initial_evaluations(f, 
        lb, ub, 
        n_perturb_samples, n_sobol_samples, 
        seed, 
        frac_perturb_dims,
        perturb_dims_protocol, # 'random' or 'dyadic'
        perturb_direction, # 'ub_only' or 'ub_and_lb'
        true_important_dims = None
        ):
    
    # TODO: eventually modify this function to allow dyadic policy too

    input_dim = len(lb)

    avg_num_important_dims_perturbed = 0

    X = None

    if n_perturb_samples > 0:
    
        # first, evaluate the function at the midpoint
        X = np.expand_dims((lb + ub)/2, 0)

        # then, perturb the specified number of samples
        for i in range(n_perturb_samples-1):
            x = (lb + ub)/2

            if perturb_dims_protocol == 'random':
                dims_to_perturb = np.random.choice(
                    np.arange(input_dim), 
                    size = int(frac_perturb_dims * input_dim), 
                    replace = False)
                avg_num_important_dims_perturbed += len(set(dims_to_perturb).intersection(set(true_important_dims)))
            
                if perturb_direction == 'ub_and_lb':
                    for dim_to_perturb in dims_to_perturb:
                        if np.random.randint(0,2) == 0:
                            x[dim_to_perturb] = lb[dim_to_perturb]
                        else:
                            x[dim_to_perturb] = ub[dim_to_perturb]
                elif perturb_direction == 'ub_only':
                    x[dims_to_perturb] = ub[dims_to_perturb]
            
            x = np.expand_dims(x, 0)
            X = np.vstack((X, x))
        
        if n_perturb_samples > 1:
            avg_num_important_dims_perturbed /= n_perturb_samples - 1

    # generate the remaining samples from Sobol sequence
    with warnings.catch_warnings(record=True):  # suppress annoying qmc.Sobol UserWarning
        X_sobol = qmc.Sobol(len(lb), scramble=True, seed=seed).random(n_sobol_samples)

    if X is None:
        X = X_sobol
    else:
        X = np.vstack((X, X_sobol))
    Y = np.array([f(lb + (ub - lb) * x) for x in X])

    return X, Y, avg_num_important_dims_perturbed


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
    frac_perturb = 0, 
    frac_perturb_dims = 0, 
    plot_title = ''
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
    if max_evals <= num_init_evals:
        raise ValueError("Must choose max_evals > num_init_evals.")
    if lb.shape != ub.shape or lb.ndim != 1:
        raise ValueError("The lower/upper bounds lb and ub must have the same shape and be D-dimensional vectors.")
    if alpha <= 0.0:
        raise ValueError("The hyperparameter alpha must be positive.")
    if device not in ["cpu", "gpu"]:
        raise ValueError("The device must be cpu or gpu.")

    plot_title += '\n fraction of initial samples perturbed: {}; \n fraction of input dimensions perturbed: {}'.format(frac_perturb, frac_perturb_dims)

    numpyro.set_platform(device)
    enable_x64()
    numpyro.set_host_device_count(1)

    max_exceptions = 3
    num_exceptions = 0

    ell_median_history = []
    ell_rank_history = []

    # Initial queries are drawn from a Sobol sequence
    # with warnings.catch_warnings(record=True):  # suppress annoying qmc.Sobol UserWarning
    #     X = qmc.Sobol(len(lb), scramble=True, seed=seed).random(num_init_evals)

    # Y = np.array([f(lb + (ub - lb) * x) for x in X])

    # generate initial queries
    X, Y, avg_num_important_dims_perturbed = generate_initial_evaluations(
        f = f, 
        lb = lb, 
        ub = ub, 
        n_perturb_samples = int(num_init_evals * frac_perturb),
        n_sobol_samples = num_init_evals - int(num_init_evals * frac_perturb),
        seed = seed,
        frac_perturb_dims = frac_perturb_dims, 
        perturb_dims_protocol = perturb_dims_protocol, 
        perturb_direction = perturb_direction,
        true_important_dims = true_important_dims
        )

    print("Starting SAASBO optimization run.")
    print(f"First {num_init_evals} queries drawn at random. Best minimum thus far: {Y.min().item():.3f}")
    print(f"{int(num_init_evals * frac_perturb)} perturbed, average number of important dimensions perturbed is {avg_num_important_dims_perturbed}")
    print('protocol for selecting which dimensions to perturb: ', perturb_dims_protocol, ', perturb direction: ', perturb_direction)


    while len(Y) < max_evals:
        print(f"=== Iteration {len(Y)} ===", flush=True)
        # standardize training data
        train_Y = (Y - Y.mean()) / Y.std()
        y_target = train_Y.min().item()

        # If for whatever reason we fail to return a query point above we choose one at random from the domain
        try:
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

            start = time.time()
            # do EI optimization using LBFGS
            x_next = optimize_ei(gp=gp, y_target=y_target, xi=0.0, num_restarts_ei=num_restarts_ei, num_init=5000)
            print(f"Optimizing EI took {time.time() - start:.2f} seconds")
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


        # get the NUTS samples of inverse squared length scales for each dimension, thinned out
        ell = 1.0 / jnp.sqrt(gp.flat_samples["kernel_inv_length_sq"][::gp.thinning])
        # record important statistics: median, average rank of each dimension?
        # pdb.set_trace()
        ell_median = jnp.median(ell, 0)
        ell_rank = jnp.argsort(ell).mean(0)

        ell_median_history.append(ell_median)
        ell_rank_history.append(ell_rank)

        # TODO: save intermediate kernel lengthscales 
        save_outputs_and_plot(X, Y, ell_median_history, ell_rank_history, avg_num_important_dims_perturbed,
            results_folder, seed, 
            frac_perturb, frac_perturb_dims,
            plot_title = plot_title)
        
        trace_important_dims(gp, thinning)

        print(f"Observed function value: {y_next:.3f}, Best function value seen thus far: {Y.min():.3f}")

        del gp  # Free memory

    return lb + (ub - lb) * X, Y


def save_outputs_and_plot(X, Y, ell_median_history, ell_rank_history, avg_num_important_dims_perturbed,
        results_folder, seed, frac_perturb, frac_perturb_dims, 
        to_plot=False, plot_title=None):
    
    frac_perturb_str = str(frac_perturb).replace('.', '')
    frac_perturb_dims_str = str(frac_perturb_dims).replace('.', '')

    if not os.path.exists(results_folder + frac_perturb_str + '_' + frac_perturb_dims_str + "/"):
        os.makedirs(results_folder + frac_perturb_str + '_' + frac_perturb_dims_str + "/")
    save_folder = results_folder + frac_perturb_str + '_' + frac_perturb_dims_str + "/" + str(seed) + "/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    np.save(save_folder + 'X_' + frac_perturb_str + '_' + frac_perturb_dims_str \
        + '_' + str(seed) + '.npy', X)
    np.save(save_folder + 'output_at_X_' + frac_perturb_str + '_' + frac_perturb_dims_str \
        + '_' + str(seed)  + '.npy', Y)
    np.save(save_folder + 'median_lengthscales_' + frac_perturb_str + '_' + frac_perturb_dims_str \
        + '_' + str(seed)  + '.npy', ell_median_history)
    np.save(save_folder + 'lengthscale_rankings_' + frac_perturb_str + '_' + frac_perturb_dims_str \
        + '_' + str(seed)  + '.npy', ell_rank_history)
    np.save(save_folder + 'avg_num_important_dims_perturbed_' + frac_perturb_str + '_' + frac_perturb_dims_str \
        + '_' + str(seed)  + '.npy', avg_num_important_dims_perturbed)


    if to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.minimum.accumulate(Y), color="b", label="SAASBO")
        ax.plot([0, 50], [-3.322, -3.322], "--", c="g", lw=3, label="Optimal value")
        ax.grid(True)
        if plot_title is not None:
            ax.set_title(plot_title, fontsize=20)
        ax.set_xlabel("Number of evaluations", fontsize=20)
        ax.set_xlim([0, 100])
        ax.set_ylabel("Best value found", fontsize=20)
        ax.set_ylim([-3.5, -0.5])
        ax.legend(fontsize=18)
        fig.savefig(results_folder + 'visualization_' + str(seed)+ '_' + frac_perturb_str + '_' + frac_perturb_dims_str)


# TODO: write a function to trace the identification of important dimensions (over iters)
# where "important" means median length scale is below some cutoff
# need to log the median length scale



def trace_important_dims(gp, thinning):

    ell = 1.0 / jnp.sqrt(gp.flat_samples["kernel_inv_length_sq"][::gp.thinning])
    ell_median = jnp.median(ell, 0)

    for i in ell_median.argsort()[:10]:
        print(f"Parameter {i:2} Median lengthscale = {ell_median[i]:.2e}")
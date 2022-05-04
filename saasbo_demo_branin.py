import argparse

import numpy as np
import numpyro
from numpyro.util import enable_x64
import sys, os
from types import SimpleNamespace
from multiprocessing import Pool

from branin import branin_1000
from saasbo import run_saasbo

def wrapper(num_init_evals, max_evals, perturb_dims_protocol, perturb_direction, seed, device, frac_perturb_samples, frac_perturb_dims):
    args_dict = locals()
    print(args_dict)
    args = SimpleNamespace(**args_dict)

    main(args)


# demonstrate how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions
def main(args):

    lb = np.hstack((-5 * np.ones(500), 0 * np.ones(500)))   # lower bounds for input domain
    ub = np.hstack((10 * np.ones(500), 15 * np.ones(500)))  # upper bounds for input domain
    
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/branin_1000/random_vs_dyadic/"

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    
    run_saasbo(
        branin_1000,
        lb,
        ub,
        args.max_evals,
        args.num_init_evals,
        results_folder = results_folder,
        true_important_dims = [17, 876],
        perturb_dims_protocol = args.perturb_dims_protocol, 
        perturb_direction = args.perturb_direction,
        seed=args.seed,
        alpha=0.01,
        num_warmup=256,
        num_samples=256,
        thinning=32,
        device=args.device,
        frac_perturb = args.frac_perturb_samples, 
        frac_perturb_dims = args.frac_perturb_dims
    )


if __name__ == "__main__":
    # assert numpyro.__version__.startswith("0.7")

    parser = argparse.ArgumentParser(description="We demonstrate how to run SAASBO.")
    parser.add_argument("--seed_start", type=int)
    parser.add_argument("--seed_end", type = int)
    parser.add_argument("--max-evals", default=30, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--num_init_evals", default = [10, 20, 30], type = int, nargs = '+',
        help = 'number of initial function evaluations')
    parser.add_argument("--frac_perturb_samples", default = [0, 0.2, 0.5, 0.8, 1], type = float, nargs = '+', 
        help = 'fraction of initial samples to generate using perturbation-from-baseline approach')
    parser.add_argument("--frac_perturb_dims", default = [0.1, 0.2, 0.5], type = float, nargs = '+',
        help = 'fraction of input dimensions to perturb at every function evaluation')
    parser.add_argument("--perturb_dims_protocol", default = 'random', type = str,
        help = 'protocol for choosing which dimensions to perturb')
    parser.add_argument("--perturb_directions", default = ['ub_only', 'ub_and_lb'], type = str, nargs='+', 
        help = 'directions in which to change the dimensions to perturb, can be ub_only or ub_and_lb')

    pargs = parser.parse_args()
    if pargs.perturb_dims_protocol == 'dyadic':
        pargs.frac_perturb_dims = [None]
        pargs.frac_perturb_samples = [None]
    
    seeds = list(range(pargs.seed_start, pargs.seed_end + 1)) 
        
    # numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(1)

    args_iter = ((num_init_evals, num_init_evals, pargs.perturb_dims_protocol, perturb_direction, seed, pargs.device, f_p_samples, f_p_dims)
                    for num_init_evals in pargs.num_init_evals
                    for perturb_direction in pargs.perturb_directions
                    for seed in seeds
                    for f_p_samples in pargs.frac_perturb_samples
                    for f_p_dims in pargs.frac_perturb_dims)
    
    pool = Pool()
    pool.starmap(wrapper, args_iter)
    pool.close()
    pool.join
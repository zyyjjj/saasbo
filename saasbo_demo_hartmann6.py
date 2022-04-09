import argparse

import numpy as np
import numpyro
from numpyro.util import enable_x64
import sys, os
from types import SimpleNamespace
from multiprocessing import Pool

from hartmann import hartmann6_50
from saasbo import run_saasbo

def wrapper(max_evals, perturb_dims_protocol, perturb_direction, seed, device, frac_perturb_samples, frac_perturb_dims):
    args_dict = locals()
    print(args_dict)
    args = SimpleNamespace(**args_dict)

    main(args)


# demonstrate how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions
def main(args):
    lb = np.zeros(50)
    ub = np.ones(50)
    num_init_evals = 20

    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/hartmann6_50/"

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)

    run_saasbo(
        hartmann6_50,
        lb,
        ub,
        args.max_evals,
        num_init_evals,
        results_folder = results_folder,
        true_important_dims = [1, 7, 11, 23, 47, 33],
        perturb_dims_protocol = args.perturb_dims_protocol, 
        perturb_direction = args.perturb_direction,
        seed=args.seed,
        alpha=0.01,
        num_warmup=256,
        num_samples=256,
        thinning=32,
        device=args.device,
        frac_perturb = args.frac_perturb_samples, 
        frac_perturb_dims = args.frac_perturb_dims, 
        plot_title = 'Hartmann 6, Dim = 50'
    )


if __name__ == "__main__":
    # assert numpyro.__version__.startswith("0.7")

    parser = argparse.ArgumentParser(description="We demonstrate how to run SAASBO.")
    parser.add_argument("--seeds", default=[1,2,3,4,5], type=int, nargs = '+')
    parser.add_argument("--max-evals", default=30, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--frac_perturb_samples", default = [0, 0.2, 0.5, 0.8, 1], type = float, nargs = '+', 
        help = 'fraction of initial samples to generate using perturbation-from-baseline approach')
    parser.add_argument("--frac_perturb_dims", default = [0.1, 0.2, 0.5], type = float, nargs = '+',
        help = 'fraction of input dimensions to perturb at every function evaluation')
    parser.add_argument("--perturb_dims_protocol", default = 'random', type = str,
        help = 'protocol for choosing which dimensions to perturb')
    parser.add_argument("--perturb_direction", default = 'ub_only', type = str,
        help = 'direction in which to change the dimensions to perturb, can be ub_only or ub_and_lb')

    pargs = parser.parse_args()

    # numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(1)

    pool = Pool()

    args_iter = ((pargs.max_evals, pargs.perturb_dims_protocol, pargs.perturb_direction, seed, pargs.device, f_p_samples, f_p_dims)
                    for seed in pargs.seeds
                    for f_p_samples in pargs.frac_perturb_samples
                    for f_p_dims in pargs.frac_perturb_dims)
                
    pool.starmap(wrapper, args_iter)
    pool.close()
    pool.join

import argparse

import numpy as np
import numpyro
from numpyro.util import enable_x64
import sys, os

from hartmann import hartmann6_50
from saasbo import run_saasbo




# demonstrate how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions
def main(args):
    lb = np.zeros(50)
    ub = np.ones(50)
    num_init_evals = 20

    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/hartmann6_50/"

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    if not os.path.exists(results_folder + "X/"):
        os.makedirs(results_folder + "X/")
    if not os.path.exists(results_folder + "output_at_X/"):
        os.makedirs(results_folder + "output_at_X/")
    if not os.path.exists(results_folder + "median_lengthscales/"):
        os.makedirs(results_folder + "median_lengthscales/")


    run_saasbo(
        hartmann6_50,
        lb,
        ub,
        args.max_evals,
        num_init_evals,
        results_folder = results_folder,
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

# TODO: run multiple frac_perturb values, multiple frac_perturb_dims values
# for each configuration, run 3 or 5 trials, 
# then generate plots of iteration-wise min and standard deviation


if __name__ == "__main__":
    # assert numpyro.__version__.startswith("0.7")

    parser = argparse.ArgumentParser(description="We demonstrate how to run SAASBO.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-evals", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--frac_perturb_samples", default = 0.5, type = float, 
        help = 'fraction of initial samples to generate using perturbation-from-baseline approach')
    parser.add_argument("--frac_perturb_dims", default = 0.5, type = float, 
        help = 'fraction of input dimensions to perturb at every function evaluation')

    args = parser.parse_args()

    numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(1)

    main(args)

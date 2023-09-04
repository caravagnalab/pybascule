# from matplotlib import interactive
from rich.console import Console
from rich.table import Table
from rich import box
import time
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn, RenderableColumn
from rich.live import Live
from rich.table import Table
from sys import maxsize

from pybasilica.svi import PyBasilica

def single_run(seed_list, save_runs_seed, kwargs):
    minBic = maxsize
    bestRun = None
    runs_seed = dict()

    scores = dict()

    for seed in seed_list:
        print("Running model with " + str(kwargs["k_denovo"]) + \
              " signatures, " + str(kwargs["cluster"]) + " groups and " + str(seed) + " seed\n")

        obj = PyBasilica(seed=seed, **kwargs)
        obj._fit()

        scores["seed_"+str(seed)] = {"bic":obj.bic, "aic":obj.aic, "icl":obj.icl, "llik":obj.likelihood, "reg_llik":obj.reg_likelihood}
        # scores["seed_"+str(seed)] = {"bic":obj.bic, "llik":obj.likelihood}

        if bestRun is None or obj.bic < minBic:
            minBic = obj.bic
            bestRun = obj

        if save_runs_seed:
            runs_seed["seed_"+str(seed)] = obj

    bestRun.runs_scores = scores

    if save_runs_seed:
        bestRun.runs_seed = runs_seed

    return bestRun


def fit(x, k_list=[0,1,2,3,4,5], lr=0.05, n_steps=500, enumer="parallel", cluster=None, groups=None, beta_fixed=None, hyperparameters=None,
        dirichlet_prior = True, compile_model = False, CUDA = False, enforce_sparsity = False, regularizer = "cosine", reg_weight = 0.,
        store_parameters=False, verbose=True, stage = "", regul_compare = None, seed = 10, initializ_seed = False, save_all_fits=False,
        save_runs_seed = False, initializ_pars_fit = False, regul_denovo = True, regul_fixed=True, nonparametric=False, do_initial_fit=False):

    if isinstance(seed, int):
        seed = [seed]

    if isinstance(cluster, int):
        cluster = [cluster]

    if isinstance(k_list, list):
        if len(k_list) > 0: pass
        else: raise Exception("k_list is an empty list!")
    elif isinstance(k_list, int):
        k_list = [k_list]
    else: raise Exception("invalid k_list datatype")

    kwargs = {
        "x":x,
        "lr":lr,
        "n_steps":n_steps,
        "enumer":enumer,
        "groups":groups,
        "dirichlet_prior":dirichlet_prior,
        "beta_fixed":beta_fixed,
        "hyperparameters":hyperparameters,
        "compile_model":compile_model,
        "CUDA":CUDA,
        "enforce_sparsity":enforce_sparsity,
        "regularizer":regularizer,
        "reg_weight":reg_weight,
        "store_parameters":store_parameters,
        "stage":stage,
        "regul_compare":regul_compare,
        "initializ_seed":initializ_seed,
        "initializ_pars_fit":initializ_pars_fit,
        "regul_denovo":regul_denovo,
        "regul_fixed":regul_fixed,
        "nonparam":nonparametric,
        }

    has_clusters = True
    if cluster is None: 
        cluster = [1]
        has_clusters = False
    
    elif nonparametric and isinstance(cluster, list):
        cluster = [max(cluster)]

    if verbose:
    # Verbose run

        console = Console()
        if beta_fixed is None:
            betaFixed = "No fixed signatures"
        elif len(list(beta_fixed.index.values)) > 10:
            betaFixed = f'{len(list(beta_fixed.index.values))} signatures, Too many to fit here'
        else:
            betaFixed = ', '.join(list(beta_fixed.index.values))

        table = Table(title="Information", show_header=False, box=box.ASCII, show_lines=False)
        table.add_column("Variable", style="cyan")
        table.add_column("Values", style="magenta")
        table.add_row("No. of samples", str(int(x.shape[0])))
        table.add_row("learning rate", str(lr))
        table.add_row("k denovo list", ', '.join(map(str, k_list)))
        table.add_row("fixed signatures", betaFixed)
        table.add_row("Max inference steps", str(n_steps))
        console.print('\n', table)

        myProgress = Progress(
            TextColumn('{task.description} [bold blue] inference {task.completed}/{task.total} done'), 
            BarColumn(), 
            TaskProgressColumn(), 
            TimeRemainingColumn(), 
            SpinnerColumn(), 
            RenderableColumn())

        with myProgress as progress:

            task = progress.add_task("[red]running...", total=len(k_list))

            minBic = maxsize
            secondMinBic = maxsize
            bestRun, secondBest = None, None

            scores_k, all_fits_stored = dict(), dict()
            for k in k_list:
                kwargs["k_denovo"] = k

                if has_clusters and do_initial_fit:
                    kwargs_init = {key: value for key, value in kwargs.items()}
                    kwargs_init["cluster"], kwargs_init["hyperparameters"], kwargs_init["enforce_sparsity"] = None, None, False
                    obj_init = single_run(seed_list=seed, save_runs_seed=False, kwargs=kwargs_init)
                    kwargs["initial_fit"] = obj_init

                for cl in list(cluster):
                    kwargs["cluster"] = cl if has_clusters else None

                    try:
                        obj = single_run(seed_list=seed, save_runs_seed=save_runs_seed, kwargs=kwargs)

                        if obj.bic < minBic:
                            minBic = obj.bic
                            bestRun = obj
                        if minBic == secondMinBic or (obj.bic > minBic and obj.bic < secondMinBic):
                            secondMinBic = obj.bic
                            secondBest = obj
                    except:
                        raise Exception("Failed to run for k_denovo:{k}!")
                
                    # scores_k["K_"+str(k)] = {"bic":obj.bic, "llik":obj.likelihood}
                    scores_k["K_"+str(k)+".G_"+str(cl)] = obj.runs_scores
                    if save_all_fits:
                        # obj.convert_to_dataframe(x, beta_fixed)
                        all_fits_stored["K_"+str(k)+".G_"+str(cl)] = obj
                
                progress.console.print(f"Running on k_denovo={k} | BIC={obj.bic}")
                progress.update(task, advance=1)

            if bestRun is not None:
                bestRun.convert_to_dataframe(x, beta_fixed)
            
            if secondBest is not None:
                secondBest.convert_to_dataframe(x, beta_fixed)
            
        from uniplot import plot
        console.print('\n-------------------------------------------------------\n\n[bold red]Best Model:')
        console.print(f"k_denovo: {bestRun.k_denovo}\nBIC: {bestRun.bic}\nStopped at {len(bestRun.losses)}th step\n")
        plot(
            [bestRun.losses, bestRun.likelihoods], 
            title="Loss & Log-Likelihood vs SVI steps", 
            width=40, height=10, color=True, legend_labels=['loss', 'log-likelihood'], interactive=False, 
            x_gridlines=[0,50,100,150,200,250,300,350,400,450,500], 
            y_gridlines=[max(bestRun.losses)/2, min(bestRun.likelihoods)/2])
        console.print('\n')

    else:
    # Non-verbose run

        minBic = maxsize
        secondMinBic = maxsize
        bestRun, secondBest = None, None

        scores_k, all_fits_stored = dict(), dict()
        for k in k_list:
            kwargs["k_denovo"] = k

            for cl in list(cluster):
                kwargs["cluster"] = cl if has_clusters else None

                if has_clusters and do_initial_fit:
                    kwargs_init = {key: value for key, value in kwargs.items()}
                    kwargs_init["cluster"], kwargs_init["hyperparameters"], kwargs_init["enforce_sparsity"] = None, None, False
                    obj_init = single_run(seed_list=seed, save_runs_seed=False, kwargs=kwargs_init)
                    kwargs["initial_fit"] = obj_init

                try:
                    obj = single_run(seed_list=seed, save_runs_seed=save_runs_seed, kwargs=kwargs)

                    if obj.bic < minBic:
                        minBic = obj.bic
                        bestRun = obj
                    if minBic == secondMinBic or (obj.bic > minBic and obj.bic < secondMinBic):
                        secondMinBic = obj.bic
                        secondBest = obj

                except:
                    raise Exception("Failed to run for k_denovo:{k}!")

                # scores_k["K_"+str(k)] = {"bic":obj.bic, "llik":obj.likelihood}
                scores_k["K_"+str(k)+".G_"+str(cl)] = obj.runs_scores
                if save_all_fits:
                    # obj.convert_to_dataframe(x, beta_fixed)
                    all_fits_stored["K_"+str(k)+".G_"+str(cl)] = obj

        if bestRun is not None:
            bestRun.convert_to_dataframe(x, beta_fixed)
        
        if secondBest is not None:
            secondBest.convert_to_dataframe(x, beta_fixed)

    bestRun.scores_K = scores_k
    bestRun.all_fits = all_fits_stored

    return bestRun, secondBest



'''
#import utilities

import torch
import pyro
import pyro.distributions as dist

from pybasilica import svi
from pybasilica import utilities



#------------------------------------------------------------------------------------------------
# run model with single k value
#------------------------------------------------------------------------------------------------
def single_k_run(params):
    #params = {
    #    "M" :               torch.Tensor
    #    "beta_fixed" :      torch.Tensor | None
    #    "k_denovo" :        int
    #    "lr" :              int
    #    "steps_per_iter" :  int
    #}
    #"alpha" :           torch.Tensor    added inside the single_k_run function
    #"beta" :            torch.Tensor    added inside the single_k_run function
    #"alpha_init" :      torch.Tensor    added inside the single_k_run function
    #"beta_init" :       torch.Tensor    added inside the single_k_run function

    # if No. of inferred signatures and input signatures are zero raise error
    #if params["beta_fixed"] is None and params["k_denovo"]==0:
    #    raise Exception("Error: both denovo and fixed signatures are zero")


    #-----------------------------------------------------
    #M = params["M"]
    num_samples = params["M"].size()[0]

    if params["beta_fixed"] is None:
        k_fixed = 0
    else:
        k_fixed = params["beta_fixed"].size()[0]
    
    k_denovo = params["k_denovo"]

    if k_fixed + k_denovo == 0:
        raise Exception("Error: both denovo and fixed signatures are zero")
    #-----------------------------------------------------

    
    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    if k_denovo > 0:
        params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    if k_denovo > 0:
        params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    if k_denovo > 0:
        params["beta"] = pyro.param("beta").clone().detach()

    #----- outputs ----------------------------------------------------------------------OK
    alpha_tensor, beta_tensor = utilities.get_alpha_beta(params)  # dtype: torch.Tensor (beta_tensor==0 if k_denovo==0)
    #lh = utilities.log_likelihood(params)           # log-likelihood
    bic = utilities.compute_bic(params)                     # BIC
    #M_R = utilities.Reconstruct_M(params)           # dtype: tensor
    
    return bic, alpha_tensor, beta_tensor


#------------------------------------------------------------------------------------------------
# run model with list of k value
#------------------------------------------------------------------------------------------------
def multi_k_run(params, k_list):
    
    #params = {
    #    "M" :               torch.Tensor
    #    "beta_fixed" :      torch.Tensor
    #    "lr" :              int
    #    "steps_per_iter" :  int
    #}
    #"k_denovo" : int    added inside the multi_k_run function
    

    bic_best = 10000000000
    k_best = -1

    for k_denovo in k_list:
        try:
            params["k_denovo"] = int(k_denovo)
            bic, alpha, beta = single_k_run(params)
            if bic <= bic_best:
                bic_best = bic
                k_best = k_denovo
                alpha_best = alpha
                beta_best = beta

        except Exception:
            continue
    
    return k_best, alpha_best, beta_best

'''


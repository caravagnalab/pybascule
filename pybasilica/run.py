# from matplotlib import interactive
from rich.console import Console
from rich.table import Table
from rich import box
import pandas as pd
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn, RenderableColumn
from rich.live import Live
from rich.table import Table
from sys import maxsize

from pybasilica.svi import PyBasilica
from pybasilica.svi_mixture import PyBasilica_mixture

def single_run(seed_list, kwargs, kwargs_mixture):
    minBic = maxsize
    bestRun = None
    runs_seed = dict()

    scores = dict()

    for seed in seed_list:
        obj = PyBasilica(seed=seed, **kwargs)
        obj._fit()

        if kwargs_mixture["cluster"] is not None:
            kwargs_mixture["alpha"] = obj.params["alpha"]
            obj_mixt = PyBasilica_mixture(seed=seed, **kwargs_mixture)
            obj_mixt._fit()

            obj.gradient_norms = {**obj.gradient_norms, **obj_mixt.gradient_norms}
            if kwargs["store_parameters"]:
                obj.train_params = [{**obj.train_params[i], **obj_mixt.train_params[i]} for i in range(len(obj.train_params))]
            obj.losses_dmm = obj_mixt.losses
            obj.likelihoods_dmm = obj_mixt.likelihoods
            obj.regs_dmm = obj_mixt.regs
            obj.groups, obj.n_groups = obj_mixt.groups, obj_mixt.n_groups
            obj.params = {**obj.params, **obj_mixt.params}
            obj.init_params = {**obj.init_params, **obj_mixt.init_params}

        scores["seed_"+str(seed)] = {"bic":obj.bic, "aic":obj.aic, "icl":obj.icl, "llik":obj.likelihood, "reg_llik":obj.reg_likelihood}

        if bestRun is None or obj.bic < minBic:
            minBic = obj.bic
            bestRun = obj

        runs_seed["seed_"+str(seed)] = obj

    bestRun.runs_scores = scores
    bestRun.runs_seed = runs_seed

    return bestRun


def fit(x, k_list=[0,1,2,3,4,5], lr = 0.005, optim_gamma = 0.1, n_steps = 500, enumer = "parallel", cluster = None, beta_fixed = None, 
        hyperparameters = None, dirichlet_prior = True, compile_model = False, CUDA = False, enforce_sparsity = True, nonparametric = False, 
        regularizer = "cosine", reg_weight = 0., regul_compare = None, regul_denovo = True, regul_fixed = True, stage = "", 
        seed = 10, store_parameters = False, save_all_fits=False):

    if isinstance(seed, int): seed = [seed]
    if isinstance(cluster, int) and cluster < 1: cluster = None
    elif isinstance(cluster, int): cluster = [cluster]
    if isinstance(k_list, int): k_list = [k_list]

    kwargs = {
        "x":x,
        "cluster":None,
        "lr":lr,
        "optim_gamma":optim_gamma,
        "n_steps":n_steps,
        "enumer":enumer,
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
        "regul_denovo":regul_denovo,
        "regul_fixed":regul_fixed
        }

    kwargs_mixture = {
        "lr":lr,
        "optim_gamma":optim_gamma,
        "n_steps":n_steps,
        "enumer":enumer,
        "hyperparameters":hyperparameters,
        "compile_model":compile_model,
        "CUDA":CUDA,
        "store_parameters":store_parameters,
        "nonparam":nonparametric
    }

    if nonparametric and isinstance(cluster, list): cluster = [max(cluster)]
    has_clusters = True
    if cluster is None: has_clusters, cluster = False, [1]

    minBic = maxsize
    secondMinBic = maxsize
    bestRun, secondBest = None, None

    scores_k, all_fits_stored = dict(), dict()
    for k in k_list:
        kwargs["k_denovo"] = k

        for cl in list(cluster):
            kwargs_mixture["cluster"] = cl if has_clusters else None

            obj = single_run(seed_list=seed, kwargs=kwargs, kwargs_mixture=kwargs_mixture)

            if obj.bic < minBic:
                minBic, bestRun = obj.bic, obj
            if minBic == secondMinBic or (obj.bic > minBic and obj.bic < secondMinBic):
                secondMinBic, secondBest = obj.bic, obj

            scores_k["K_"+str(k)+".G_"+str(cl)] = obj.runs_scores
            if save_all_fits: all_fits_stored["K_"+str(k)+".G_"+str(cl)] = obj

    if bestRun is not None: bestRun.convert_to_dataframe(x)
    if secondBest is not None: secondBest.convert_to_dataframe(x)


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


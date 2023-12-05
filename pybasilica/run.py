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
from copy import deepcopy

from pybasilica.svi import PyBasilica
from pybasilica.svi_mixture import PyBasilica_mixture


def fit(x=None, alpha=None, k_list=[0,1,2,3,4,5], lr = 0.005, optim_gamma = 0.1, n_steps = 500, enumer = "parallel", 
        cluster = None, beta_fixed = None, hyperparameters = None, dirichlet_prior = True, 
        compile_model = False, CUDA = False, nonparametric = False, stage = "",  seed_list = [10], 
        store_parameters = False, store_fits=False):

    if isinstance(seed_list, int): seed_list = [seed_list]
    if isinstance(cluster, int) and cluster < 1: cluster = None
    elif isinstance(cluster, int): cluster = [cluster]
    if isinstance(k_list, int): k_list = [k_list]

    if x is None and alpha is None: raise "Both count and exposure matrices are None."

    kwargs = {
        "x":x,
        # "cluster":None,
        "lr":lr,
        "optim_gamma":optim_gamma,
        "n_steps":n_steps,
        "enumer":enumer,
        "dirichlet_prior":dirichlet_prior,
        "beta_fixed":beta_fixed,
        "hyperparameters":hyperparameters,
        "compile_model":compile_model,
        "CUDA":CUDA,
        # "enforce_sparsity":enforce_sparsity,
        # "regularizer":regularizer,
        # "reg_weight":reg_weight,
        "store_parameters":store_parameters,
        "stage":stage,
        # "regul_compare":regul_compare,
        # "regul_denovo":regul_denovo,
        # "regul_fixed":regul_fixed
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

    bestK, scoresK, fitsK, fits_alpha = None, None, dict(), dict()
    if x is not None:
        bestK, scoresK, fitsK = run_fit(seed_list=seed_list, kwargs=kwargs, parname="k_denovo",
                                        parlist=k_list, score_name="bic", store_fits=store_fits, cls=PyBasilica)
        bestK.scores = scoresK
        bestK.fits = fitsK

    if cluster is not None:
        if bestK is not None: alpha = bestK.params["alpha"]

        kwargs_mixture["alpha"] = alpha
        bestCL, scoresCL, fitsCL = run_fit(seed_list=seed_list, kwargs=kwargs_mixture, parname="cluster",
                                           parlist=list(cluster), score_name="icl", store_fits=store_fits,
                                           cls=PyBasilica_mixture)
        bestCL.scores = scoresCL
        bestCL.fits = fitsCL

        bestK = merge_k_cl(obj=bestK, obj_mixt=bestCL, store_parameters=store_parameters) if bestK is not None else bestCL

    if bestK is not None: bestK.convert_to_dataframe(x) if x is not None else bestK.convert_to_dataframe(alpha)

    return bestK


def single_run(seed_list, kwargs, cls, score_name, idd):
    best_score, best_run = maxsize, None
    fits, scores = dict(), dict()

    for seed in seed_list:
        idd_s = "seed:"+str(seed)
        obj = cls(seed=seed, **kwargs)
        obj._fit()
        obj.idd = idd + "." + idd_s

        sc_s = obj.__dict__[score_name]

        if best_run is None or sc_s < best_score:
            best_score = sc_s
            best_run = deepcopy(obj)

        scores[idd_s] = {"bic":obj.bic, "aic":obj.aic, "icl":obj.icl, "llik":obj.likelihood}
        fits[idd_s] = obj

    return best_run, scores, fits


def run_fit(seed_list, kwargs, parname, parlist, score_name, store_fits, cls):
    '''
    `seed_list` -> list of seeds to test \\
    `kwargs` -> dict of arguments for the fit \\
    `parname` -> name of the parameter in the for loop \\
    `parlist` -> list of values to iterate through \\
    `score_name` -> name of the score to minimize (either `bic`, `icl` or `aic`) \\
    `store_fits` -> if True, all fits will be stores \\
    `cls` -> class to be used for the fit \\
    '''
    best_score, best_run = maxsize, None
    fits, scores = dict(), dict()
    for i in parlist:
        idd_i = parname + ":" + str(i)

        # fits_i contains a dict with the fits for all seeds in seed_list
        kwargs[parname] = i
        best_i, scores_i, fits_i = single_run(seed_list=seed_list, kwargs=kwargs, cls=cls, score_name=score_name, idd=idd_i)

        sc_i = best_i.__dict__[score_name]

        if best_run is None or sc_i < best_score:
            best_score = sc_i
            best_run = deepcopy(best_i)
            best_idd = idd_i

        scores[idd_i] = scores_i
        if store_fits: 
            input_val = kwargs["x"] if "x" in kwargs.keys() else kwargs["alpha"]
            for k, v in fits_i.items():
                v.convert_to_dataframe(input_val)
            fits[idd_i] = fits_i

    return best_run, scores, fits


def merge_k_cl(obj, obj_mixt, store_parameters):
    obj.__dict__["fits"] = {"NMF":obj.fits, "CL":obj_mixt.fits}
    obj.__dict__["scores"] = {"NMF":obj.scores, "CL":obj_mixt.scores}
    obj.__dict__["idd"] = obj.idd + ";" + obj_mixt.idd

    obj.gradient_norms = {**obj.gradient_norms, **obj_mixt.gradient_norms}
    if store_parameters:
        obj.train_params = [{**obj.train_params[i], **obj_mixt.train_params[i]} for i in range(len(obj.train_params))]
    obj.losses_dmm = obj_mixt.losses
    obj.likelihoods_dmm = obj_mixt.likelihoods
    obj.regs_dmm = obj_mixt.regs
    obj.groups, obj.n_groups = obj_mixt.groups, obj_mixt.n_groups
    obj.params = {**obj.params, **obj_mixt.params}
    obj.init_params = {**obj.init_params, **obj_mixt.init_params}
    obj.hyperparameters = {**obj.hyperparameters, **obj_mixt.hyperparameters}
    return obj


# def select_best(parlist, parname, seed, kwargs, classname, save_all_fits, score):
#     bestScore, secondScore = maxsize, maxsize
#     best_idd, secondBest = None, None

#     scores_k, all_fits = dict(), dict()
#     for k in parlist:
#         idd = parname + ":" + str(k)
#         kwargs[parname] = k

#         obj, fits_scores, fits_seed = single_run(seed_list=seed, kwargs=kwargs, classname=classname, score=score)

#         if score == "bic": score_k = obj.bic
#         elif score == "icl": score_k = obj.icl

#         if score_k < bestScore:
#             bestScore, best_idd = score_k, 
#         if bestScore == secondScore or (score_k > bestScore and score_k < secondScore):
#             secondScore, secondBest = score_k, deepcopy(obj)

#         bestRun.fits_seed = fits_seed
#         bestRun.idd = idd
#         scores_k[idd] = fits_scores
#         if save_all_fits: 
#             obj.fits_seed = fits_seed
#             all_fits[idd] = obj

#     return bestRun, secondBest, scores_k, all_fits



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


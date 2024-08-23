import numpy as np

from sys import maxsize
from copy import deepcopy

from pybascule.svi import PyBascule
from pybascule.svi_mixture import PyBascule_mixture


def fit(x=None, alpha=None, k_list=[0,1,2,3,4,5], lr = 0.005, optim_gamma = 0.1, n_steps = 500, enumer = "parallel", 
        cluster = None, beta_fixed = None, hyperparameters = None, dirichlet_prior = True, autoguide = False,
        compile_model = False, CUDA = False, nonparametric = False, stage = "",  seed_list = [10], 
        store_parameters = False, store_fits=False):

    if isinstance(seed_list, int): seed_list = [seed_list]
    if isinstance(cluster, int) and cluster < 1: cluster = None
    elif isinstance(cluster, int): cluster = [cluster]
    if isinstance(k_list, int): k_list = [k_list]

    if x is None and alpha is None: raise "Both count and exposure matrices are None."

    kwargs = {
        "x":x,
        "lr":lr,
        "optim_gamma":optim_gamma,
        "n_steps":n_steps,
        "enumer":enumer,
        "dirichlet_prior":dirichlet_prior,
        "beta_fixed":beta_fixed,
        "hyperparameters":hyperparameters,
        "compile_model":compile_model,
        "CUDA":CUDA,
        "store_parameters":store_parameters,
        "stage":stage,
        }

    kwargs_mixture = {
        "lr":lr,
        "optim_gamma":optim_gamma,
        "n_steps":n_steps,
        "enumer":enumer,
        "hyperparameters":hyperparameters,
        "autoguide":autoguide,
        "compile_model":compile_model,
        "CUDA":CUDA,
        "store_parameters":store_parameters,
        "nonparam":nonparametric
    }

    if nonparametric and isinstance(cluster, list): cluster = [max(cluster)]

    bestK, scoresK, fitsK, fits_alpha = None, None, dict(), dict()
    if x is not None:
        bestK, scoresK, fitsK = run_fit(seed_list=seed_list, kwargs=kwargs, parname="k_denovo",
                                        parlist=k_list, score_name="bic", store_fits=store_fits, cls=PyBascule)
        bestK.scores = scoresK
        bestK.fits = fitsK

    if cluster is not None:
        if bestK is not None: alpha = bestK.params["alpha"]

        kwargs_mixture["alpha"] = alpha
        bestCL, scoresCL, fitsCL = run_fit(seed_list=seed_list, kwargs=kwargs_mixture, parname="cluster",
                                           parlist=list(cluster), score_name="icl", store_fits=store_fits,
                                           cls=PyBascule_mixture)
        bestCL.scores = scoresCL
        bestCL.fits = fitsCL

        bestK = merge_k_cl(obj=bestK, obj_mixt=bestCL, store_parameters=store_parameters, store_fits=store_fits) if bestK is not None else bestCL

    if bestK is not None: bestK.convert_to_dataframe(x) if x is not None else bestK.convert_to_dataframe(alpha)
    if store_fits: convert_fits(bestK)

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

        if parname == "cluster": idd_i = parname + ":" + str(i) + "_" + str(len(np.unique(best_i.groups)))

        sc_i = best_i.__dict__[score_name]

        if best_run is None or sc_i < best_score:
            best_score = sc_i
            best_run = deepcopy(best_i)

        scores[idd_i] = scores_i
        if store_fits: 
            fits[idd_i] = fits_i

    return best_run, scores, fits


def merge_k_cl(obj, obj_mixt, store_parameters, store_fits):
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


def convert_fits(obj):
    try: _convert_fits_aux(obj.fits, input=obj.x)
    except: pass
    try: _convert_fits_aux(obj.fits, input=obj.alpha)
    except: pass
    try: _convert_fits_aux(obj.fits["NMF"], input=obj.x)
    except: pass
    try: _convert_fits_aux(obj.fits["CL"], input=obj.params["alpha"])
    except: pass


def _convert_fits_aux(fits, input):
    for _, v_1 in fits.items(): 
        for _, v_2 in v_1.items():
            v_2.convert_to_dataframe(input)


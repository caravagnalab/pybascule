from matplotlib import interactive
from rich.console import Console
from rich.table import Table
from rich import box
import time
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn, RenderableColumn
from rich.live import Live
from rich.table import Table

from pybasilica.svi import PyBasilica
#from svi import PyBasilica

def single_run(x, k_denovo, lr=0.05, n_steps=500, enumer=False, cluster=None, groups=None, beta_fixed=None, compile_model = False, CUDA = False, enforce_sparsity = False, regularizer = "cosine", store_parameters=False):
    
    obj = PyBasilica(x, k_denovo, lr, n_steps, enumer=enumer, cluster=cluster, groups=groups, beta_fixed=beta_fixed, compile_model = compile_model, CUDA = CUDA, enforce_sparsity = enforce_sparsity, regularizer = regularizer, store_parameters=store_parameters)
    obj._fit()
    minBic = obj.bic
    bestRun = obj

    #for i in track(range(2), description="Processing..."):
    for i in range(2):

        obj = PyBasilica(x, k_denovo, lr, n_steps, enumer=enumer, cluster=cluster, groups=groups, beta_fixed=beta_fixed, compile_model = compile_model, CUDA = CUDA, enforce_sparsity = enforce_sparsity, regularizer = regularizer, store_parameters=store_parameters)
        obj._fit()

        if obj.bic < minBic:
            minBic = obj.bic
            bestRun = obj

    return bestRun



def fit(x, k_list=[0,1,2,3,4,5], lr=0.05, n_steps=500, enumer="sequential", cluster=None, groups=None, beta_fixed=None, compile_model = False, CUDA = False, enforce_sparsity = False, regularizer = "cosine", store_parameters=False, verbose=True):

    if isinstance(k_list, list):
        if len(k_list) > 0:
            pass
        else:
            raise Exception("k_list is an empty list!")
    elif isinstance(k_list, int):
        k_list = [k_list]
    else:
        raise Exception("invalid k_list datatype")


    #===============================================================
    # verbose run ==================================================
    #===============================================================
    if verbose:
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

        #print(', '.join(names))

        myProgress = Progress(
            TextColumn('{task.description} [bold blue] inference {task.completed}/{task.total} done'), 
            BarColumn(), 
            TaskProgressColumn(), 
            TimeRemainingColumn(), 
            SpinnerColumn(), 
            RenderableColumn())

        with myProgress as progress:

            task = progress.add_task("[red]running...", total=len(k_list))

            obj = single_run(x=x, k_denovo=k_list[0], lr=lr, n_steps=n_steps, enumer=enumer, cluster=cluster, groups=groups, 
            beta_fixed=beta_fixed, compile_model = compile_model, 
            CUDA=CUDA, enforce_sparsity = enforce_sparsity, regularizer = regularizer, store_parameters=store_parameters)
            minBic = obj.bic
            bestRun = obj
            progress.console.print(f"Running on k_denovo={k_list[0]} | BIC={obj.bic}")
            progress.update(task, advance=1)

            for k in k_list[1:]:
            
                try:

                    obj = single_run(x=x, k_denovo=k, lr=lr, n_steps=n_steps, enumer=enumer, cluster=cluster,
                    groups=groups, beta_fixed=beta_fixed, compile_model=compile_model, CUDA = CUDA, 
                    enforce_sparsity = enforce_sparsity, regularizer = regularizer, store_parameters=store_parameters)

                    if obj.bic < minBic:
                        minBic = obj.bic
                        bestRun = obj
                except:
                    continue
                
                progress.console.print(f"Running on k_denovo={k} | BIC={obj.bic}")
                progress.update(task, advance=1)

            try:
                bestRun._convert_to_dataframe(x, beta_fixed)
            except:
                raise Exception("Not able to convert tensors to dataframe!")

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

        return bestRun


    else:
    #===============================================================
    # Non-verbose run ==============================================
    #===============================================================

        obj = single_run(x=x, k_denovo=k_list[0], lr=lr, n_steps=n_steps, enumer=enumer, cluster=cluster, groups=groups, beta_fixed=beta_fixed,
        compile_model = compile_model, CUDA = CUDA, enforce_sparsity = enforce_sparsity, regularizer = regularizer, store_parameters=store_parameters)
        minBic = obj.bic
        bestRun = obj
        
        for k in k_list[1:]:
            try:

                obj = single_run(x=x, k_denovo=k, lr=lr, n_steps=n_steps, enumer=enumer, cluster=cluster, groups=groups,
                beta_fixed=beta_fixed, compile_model = compile_model, CUDA = CUDA,
                enforce_sparsity = enforce_sparsity, regularizer = regularizer, store_parameters=store_parameters)
                if obj.bic < minBic:
                    minBic = obj.bic
                    bestRun = obj
            except:
                raise Exception("Failed to run for k_denovo:{k}!")

        try:
            bestRun._convert_to_dataframe(x, beta_fixed)
        except:
            raise Exception("Not able to convert tensors to dataframe!")
        
        return bestRun
            
        #try:
        #    bestRun._convert_to_dataframe(x, beta_fixed)
    #minBic = 10000000
    #bestRun = None

    '''
    obj = single_run(x=x, k_denovo=k_list[0], lr=lr, n_steps=n_steps, groups=groups, beta_fixed=beta_fixed, CUDA = CUDA, compile_model = compile_model)
    minBic = obj.bic
    bestRun = obj

    for k in k_list[1:]:
        try:
            obj = single_run(x=x, k_denovo=k, lr=lr, n_steps=n_steps, groups=groups, beta_fixed=beta_fixed, CUDA = CUDA, compile_model = compile_model)

            if obj.bic < minBic:
                minBic = obj.bic
                bestRun = obj
        except:
            raise Exception("No run, please take care of inputs, probably k_list!")

        return bestRun

        '''



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


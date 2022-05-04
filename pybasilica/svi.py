import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import torch
from pybasilica import utilities


#------------------------------------------------------------------------------------------------
# model
#------------------------------------------------------------------------------------------------

def model(params):
    '''
    params = {
        "M" :           torch.Tensor
        "beta_fixed" :  torch.Tensor
        "k_denovo" :    int
        "alpha" :       torch.Tensor
        "beta" :        torch.Tensor
    }
    '''

    num_samples = params["M"].size()[0]

    if type(params["beta_fixed"]) is int:
    #if params["beta_fixed"]==0:
        beta_fixed = 0
        k_fixed = 0
    else:
        beta_fixed = params["beta_fixed"]
        k_fixed = beta_fixed.size()[0]
    
    k_denovo = params["k_denovo"]
    
    # alpha is relative exposure (normalized or percentages of signature activity)
    # theta encodes the total number of mutations in each branch
    # parametarize the activity matrix as : theta * alpha

    num_groups = max(params["groups"]) + 1

    alpha_tissues = pyro.sample("alpha_tissues", dist.Normal(torch.zeros(num_groups, k_fixed + k_denovo), 1))

    # sample from the alpha prior
    with pyro.plate("K", k_fixed + k_denovo):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(alpha_tissues[params["groups"], :], 1))
    
    alpha = torch.exp(alpha)                                # enforce non negativity
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))     # normalize
    

    #-------------------------- if beta denovo is NULL ----------------------------
    if k_denovo==0:
        beta_denovo=0
    
    #-------------------------- if beta denovo is present -------------------------
    else:
        # sample from the beta prior
        with pyro.plate("contexts", 96):            # columns
            with pyro.plate("k_denovo", k_denovo):  # rows
                beta_denovo = pyro.sample("denovo_signatures", dist.Normal(params["beta"], 1))

        beta_denovo = torch.exp(beta_denovo)                                    # enforce non negativity
        beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))   # normalize


    # compute the custom likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.factor("obs", utilities.custom_likelihood(params["M"], alpha, beta_fixed, beta_denovo))



#------------------------------------------------------------------------------------------------
# guide
#------------------------------------------------------------------------------------------------

def guide(params):
    '''
    params = {
        "M" :           torch.Tensor
        "beta_fixed" :  torch.Tensor
        "k_denovo" :    int
        "alpha_init" :  torch.Tensor
        "beta_init" :   torch.Tensor
    }
    '''
    num_samples = params["M"].size()[0]
    k_denovo = params["k_denovo"]

    if type(params["beta_fixed"]) is int:
    #if params["beta_fixed"]==0:
        k_fixed = 0
    else:
        k_fixed = params["beta_fixed"].size()[0]


    with pyro.plate("K", k_fixed + k_denovo):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", params["alpha_init"])
            pyro.sample("activities", dist.Delta(alpha))

    if k_denovo!=0:
        with pyro.plate("contexts", 96):
            with pyro.plate("k_denovo", k_denovo):
                beta = pyro.param("beta", params["beta_init"])
                pyro.sample("denovo_signatures", dist.Delta(beta))



#------------------------------------------------------------------------------------------------
# inference
#------------------------------------------------------------------------------------------------

def inference(params):
    '''
    params = {
        "M" :               torch.Tensor
        "beta_fixed" :      torch.Tensor
        "k_denovo" :        int
        "alpha" :           torch.Tensor
        "beta" :            torch.Tensor
        "alpha_init" :      torch.Tensor
        "beta_init" :       torch.Tensor
        "lr" :              int
        "steps_per_iter" :  int
        }
    '''
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": params["lr"]}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model, guide, optimizer, loss=elbo)

#   inference - do gradient steps
    for step in range(params["steps_per_iter"]):
        loss = svi.step(params)




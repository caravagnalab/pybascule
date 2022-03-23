import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import torch
import utilities


#------------------------------------------------------------------------------------------------
# model
#------------------------------------------------------------------------------------------------

def model(params):
    
    num_samples = params["M"].size()[0]
    beta_fixed = params["beta_fixed"]
    K_fixed = beta_fixed.size()[0]
    K_denovo = params["k_denovo"]
    theta = torch.sum(params["M"], axis=1)

    # parametrize the activity matrix as theta*alpha
    # theta encodes the total number of mutations of the branches
    # alpha is relative exposure (percentages of signature activity)

    # sample from the alpha prior
    with pyro.plate("K", K_denovo + K_fixed):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    # sample from the beta prior
    with pyro.plate("contexts", 96):            # columns
        with pyro.plate("K_denovo", K_denovo):  # rows
            beta_denovo = pyro.sample("extra_signatures", dist.Normal(params["beta"], 1))

    # enforce non negativity
    alpha = torch.exp(alpha)
    beta_denovo = torch.exp(beta_denovo)

    # normalize
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    # build full signature profile (beta) matrix
    beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    # compute the likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.sample("obs", 
                        dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)), 
                        obs=params["M"])

    # compute the custom likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.factor("obs", utilities.custom_likelihood(alpha, beta, params["M"], params["cosmic_path"]))


#------------------------------------------------------------------------------------------------
# guide
#------------------------------------------------------------------------------------------------

def guide(params):

    num_samples = params["M"].size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]

    with pyro.plate("K", K_denovo + K_fixed):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", params["alpha_init"])
            pyro.sample("activities", dist.Delta(alpha))

    with pyro.plate("contexts", 96):
        with pyro.plate("K_denovo", K_denovo):
            beta = pyro.param("beta", params["beta_init"])
            pyro.sample("extra_signatures", dist.Delta(beta))


#------------------------------------------------------------------------------------------------
# inference
#------------------------------------------------------------------------------------------------

def inference(params):
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": params["lr"]}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    #-------TEST----------------
    #cosmic_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    #beta = params["beta"]
    #_, _, cosmic = utilities.beta_read_csv(cosmic_path)
    # compute loss
    #loss_fn = Trace_ELBO().differentiable_loss
    #elbo = loss_fn(model, guide, params) + utilities.regularizer(beta, cosmic)
    #elbo.backward()
    #optimizer.step()
    #optimizer.zero_grad()
    #-------TEST----------------

    svi = SVI(model, guide, optimizer, loss=elbo)

#   inference - do gradient steps
    for step in range(params["steps_per_iter"]):
        loss = svi.step(params)


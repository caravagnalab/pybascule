import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import model
import transfer

def inference_single_run(M, params, lr=0.05, num_steps=200):
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": lr}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

#   inference

#   do gradient steps

    for step in range(num_steps):

        loss = svi.step(M, params)


def full_inference(M, params, lr=0.05, steps_per_iteration=200, num_iterations=10):
    # first indipendent run

    num_samples = M.size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]

    alphas = []
    betas = []

    # step 0 : indipendent inference

    print("iteration ", 0)

    params["alpha"] = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    inference_single_run(M, params, lr=lr,num_steps=steps_per_iteration)

    alphas.append(pyro.param("alpha").clone().detach())
    betas.append(pyro.param("beta").clone().detach())

    # do iterations transferring alpha's

    alphas = []
    betas = []

    for i in range(num_iterations):

        print("iteration ", i + 1)
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        # calculate transfer coeff
        transfer_coeff = transfer.calculate_transfer_coeff(M, params)

        # update alpha prior with transfer coeff
        old_alpha = params["alpha"]
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        # do inference with updates alpha_prior and beta_prior
        inference_single_run(M, params, lr=lr, num_steps=steps_per_iteration)

        alphas.append(pyro.param("alpha").clone().detach())
        betas.append(pyro.param("beta").clone().detach())

        loss_alpha = torch.sum((old_alpha - pyro.param("alpha").clone().detach()) ** 2)
        loss_beta = torch.sum((params["beta"] - pyro.param("beta").clone().detach()) ** 2)

        print("loss alpha =", loss_alpha)
        #print("loss beta =", loss_beta)

    # save final inference
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    return params, alphas, betas

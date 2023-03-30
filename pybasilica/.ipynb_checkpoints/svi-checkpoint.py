import numpy as np
import pandas as pd
import torch
import pyro
from pyro.infer import SVI,Trace_ELBO,JitTrace_ELBO
from pyro.optim import Adam
import pyro.distributions.constraints as constraints
import pyro.distributions as dist
import torch.nn.functional as F
from tqdm import trange

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


class PyBasilica():

    def __init__(
        self, 
        x, 
        k_denovo, 
        lr, 
        n_steps, 
        groups=None, 
        beta_fixed=None, 
        compile_model = True, 
        CUDA = False, 
        enforce_sparsity = False
        ):
        
        self._set_data_catalogue(x)
        self._set_beta_fixed(beta_fixed)
        #self.k_denovo = int(k_denovo)
        self._set_k_denovo(k_denovo)
        self.lr = lr
        self.n_steps = int(n_steps)
        self.compile_model = compile_model  
        self.CUDA = CUDA
        self.enforce_sparsity = enforce_sparsity
        self._set_groups(groups)
        self._check_args()
    

    def _set_data_catalogue(self, x):
        try:
            self.x = torch.tensor(x.values).float()
            self.n_samples = x.shape[0]
        except:
            raise Exception("Invalid mutations catalogue, expected Dataframe!")
    
    
    def _set_beta_fixed(self, beta_fixed):
        try:
            self.beta_fixed = torch.tensor(beta_fixed.values).float()
            self.k_fixed = beta_fixed.shape[0]
        except:
            if beta_fixed is None:
                self.beta_fixed = None
                self.k_fixed = 0
            else:
                raise Exception("Invalid fixed signatures catalogue, expected DataFrame!")
    
    def _set_k_denovo(self, k_denovo):
        if isinstance(k_denovo, int):
            self.k_denovo = k_denovo
        else:
            raise Exception("Invalid k_denovo value, expected integer!")


    def _set_groups(self, groups):
        if groups is None:
            self.groups = None
        else:
            if isinstance(groups, list) and len(groups)==self.n_samples:
                self.groups = groups
            else:
                raise Exception("invalid groups argument, expected 'None' or a list with {} elements!".format(self.n_samples))


    def _check_args(self):
        if self.k_denovo==0 and self.k_fixed==0:
            raise Exception("No. of denovo and fixed signatures could NOT be zero at the same time!")
    
    
    def model(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        groups = self.groups

        #----------------------------- [ALPHA] -------------------------------------
        if groups != None:

            #num_groups = max(params["groups"]) + 1
            n_groups = len(set(groups))
            alpha_tissues = dist.Normal(torch.zeros(n_groups, k_fixed + k_denovo), 1).sample()
            
            alpha = pyro.sample("latent_exposure", dist.Normal(alpha_tissues[groups, :], 1))
            
            print("INSIDE IF",alpha)

            # sample from the alpha prior
            # with pyro.plate("k", k_fixed + k_denovo):   # columns
            #     with pyro.plate("n", n_samples):        # rows
            #         alpha = pyro.sample("latent_exposure", dist.Normal(alpha_tissues[groups, :], 1))
        else:
            alpha_mean = dist.Normal(torch.zeros(n_samples, k_fixed + k_denovo), 1).sample()

            with pyro.plate("k", k_fixed + k_denovo):   # columns
                with pyro.plate("n", n_samples):        # rows
                    if self.enforce_sparsity:
                        alpha = pyro.sample("latent_exposure", dist.Exponential(3))
                    else:
                        alpha = pyro.sample("latent_exposure", dist.HalfNormal(1))
        
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))     # normalize
        alpha = torch.clamp(alpha, 0,1)

        #----------------------------- [BETA] -------------------------------------
        if k_denovo==0:
            beta_denovo = None
        else:
            #beta_mean = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()
            with pyro.plate("contexts", 96):            # columns
                with pyro.plate("k_denovo", k_denovo):  # rows
                    beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(1))
            beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))   # normalize
            beta_denovo = torch.clamp(beta_denovo, 0,1)

        #----------------------------- [LIKELIHOOD] -------------------------------------
        if self.beta_fixed is None:
            beta = beta_denovo
            reg = 0
        elif beta_denovo is None:
            beta = self.beta_fixed
            reg = 0
        else:
            beta = torch.cat((self.beta_fixed, beta_denovo), axis=0)
            reg = self._regularizer(self.beta_fixed, beta_denovo)
        
        with pyro.plate("contexts2", 96):
            with pyro.plate("n2", n_samples):
                lk =  dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)).log_prob(self.x)
                pyro.factor("loss", lk - reg)
    

    def guide(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        groups = self.groups
        
        # Alpha ---------------------------------------------------------------
        if groups != None:
            n_groups = len(set(groups))
            alpha_tissues = dist.Normal(torch.zeros(n_groups, k_fixed + k_denovo), 1).sample()
            
            with pyro.plate("k", k_fixed + k_denovo):   # columns
                with pyro.plate("n", n_samples):        # rows
                    alpha = pyro.param("alpha", alpha_tissues[groups, :], constraint=constraints.greater_than_eq(0))
                    pyro.sample("latent_exposure", dist.Delta(alpha))
        
        else:
            alpha_mean = dist.HalfNormal(torch.ones(n_samples, k_fixed + k_denovo)).sample()
    
            with pyro.plate("k", k_fixed + k_denovo):
                with pyro.plate("n", n_samples):
                    alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than_eq(0))
                    pyro.sample("latent_exposure", dist.Delta(alpha))

        # Beta ----------------------------------------------------------------
        if k_denovo != 0:
            beta_mean = dist.HalfNormal(torch.ones(k_denovo, 96)).sample()
            with pyro.plate("contexts", 96):
                with pyro.plate("k_denovo", k_denovo):
                    beta = pyro.param("beta_denovo", beta_mean, constraint=constraints.greater_than_eq(0))
                    pyro.sample("latent_signatures", dist.Delta(beta))

    
    def _regularizer(self, beta_fixed, beta_denovo):
        '''
        if beta_denovo == None:
            dd = 0
        else:
            dd = 0
            c1 = 0
            for denovo1 in beta_denovo:
                c1 += 1
                c2 = 0
                for denovo2 in beta_denovo:
                    c2 += 1
                    if c1!=c2:
                        dd += F.kl_div(denovo1, denovo2, reduction="batchmean").item()
        '''
        loss = 0
        for fixed in beta_fixed:
            for denovo in beta_denovo:
                loss += F.kl_div(torch.log(fixed), torch.log(denovo), log_target = True, reduction="batchmean")
                #loss += cosi(fixed, denovo).item()
        #print("loss:", loss)
        return loss
    
    
    def _likelihood(self, M, alpha, beta_fixed, beta_denovo):
        
        if beta_fixed is None:
            beta = beta_denovo
        elif beta_denovo is None:
            beta = beta_fixed
        else:
            beta = torch.cat((beta_fixed, beta_denovo), axis=0)

        _log_like_matrix = dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(M, axis=1)), alpha), beta)).log_prob(M)
        _log_like_sum = torch.sum(_log_like_matrix)
        _log_like = float("{:.3f}".format(_log_like_sum.item()))
        #print("loglike:",_log_like)

        return _log_like
    
    
    
    def _fit(self):
        
        pyro.clear_param_store()  # always clear the store before the inference
        if self.CUDA and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.x = self.x.cuda()
            if self.beta_fixed is not None:
                self.beta_fixed = self.beta_fixed.cuda()
        else:
            torch.set_default_tensor_type(t=torch.FloatTensor)
        
        
        if self.compile_model and not self.CUDA:
            elbo = JitTrace_ELBO()
        else:
            elbo = Trace_ELBO()


        # learning global parameters
        adam_params = {"lr": self.lr}
        optimizer = Adam(adam_params)

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        losses = []
        likelihoods = []
        for _ in range(self.n_steps):   # inference - do gradient steps
            loss = svi.step()
            losses.append(loss)

            # create likelihoods -------------------------------------------------------------
            alpha = pyro.param("alpha").clone().detach()
            #alpha = torch.exp(alpha)
            alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

            if self.k_denovo == 0:
                beta_denovo = None
            else:
                beta_denovo = pyro.param("beta_denovo").clone().detach()
                #beta_denovo = torch.exp(beta_denovo)
                beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

            likelihoods.append(self._likelihood(self.x, alpha, self.beta_fixed, beta_denovo))
            # --------------------------------------------------------------------------------

            
            # convergence test ---------------------------------------------------------------
            r = 50
            if len(losses) >= r:
                if len(losses)%r==0:
                    #print(convergence(x=losses[-r:], alpha=0.05))
                    if convergence(x=losses[-r:], alpha=0.05):
                        break
            # --------------------------------------------------------------------------------
        
        '''
        t = trange(self.n_steps, desc='Bar desc', leave = True)
        for step in t:   # inference - do gradient steps
            loss = svi.step()
            losses.append(loss)
            t.set_description('ELBO: {:.5f}  '.format(loss))
            t.refresh()
        ''' 
        if self.CUDA and torch.cuda.is_available():
          self.x = self.x.cpu()
          if self.beta_fixed is not None:
            self.beta_fixed = self.beta_fixed.cpu()
          

        self.losses = losses
        self.likelihoods = likelihoods
        self._set_alpha()
        self._set_beta_denovo()
        self._set_bic()
        #self.likelihood = self._likelihood(self.x, self.alpha, self.beta_fixed, self.beta_denovo)
        #self.regularization = self._regularizer(self.beta_fixed, self.beta_denovo)


    def _set_alpha(self):
        # exposure matrix
        alpha = pyro.param("alpha")
        if self.CUDA and torch.cuda.is_available():
            alpha = alpha.cpu()
        alpha = alpha.clone().detach()
        #alpha = torch.exp(alpha)
        self.alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    
    def _set_beta_denovo(self):
        # signature matrix
        if self.k_denovo == 0:
            self.beta_denovo = None
        else:
            beta_denovo = pyro.param("beta_denovo")
            if self.CUDA and torch.cuda.is_available():
                beta_denovo = beta_denovo.cpu()
            beta_denovo = beta_denovo.clone().detach()
            #beta_denovo = torch.exp(beta_denovo)
            self.beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))
    

    def _set_bic(self):

        M = self.x
        alpha = self.alpha

        _log_like = self._likelihood(M, alpha, self.beta_fixed, self.beta_denovo)

        k = (alpha.shape[0] * (alpha.shape[1])) + ((self.k_denovo + self.k_fixed) * M.shape[1])
        n = M.shape[0] * M.shape[1]
        bic = k * torch.log(torch.tensor(n)) - (2 * _log_like)

        self.bic = bic.item()


    
    def _convert_to_dataframe(self, x, beta_fixed):

        # mutations catalogue
        self.x = x
        sample_names = list(x.index)
        mutation_features = list(x.columns)

        # fixed signatures
        fixed_names = []
        if self.beta_fixed is not None:
            fixed_names = list(beta_fixed.index)
            self.beta_fixed = beta_fixed

        # denovo signatures
        denovo_names = []
        if self.beta_denovo is not None:
            for d in range(self.k_denovo):
                denovo_names.append("D"+str(d+1))
            self.beta_denovo = pd.DataFrame(np.array(self.beta_denovo), index=denovo_names, columns=mutation_features)

        # alpha
        self.alpha = pd.DataFrame(np.array(self.alpha), index=sample_names , columns= fixed_names + denovo_names)
        
    def _mv_to_gpu(self,*cpu_tens):
        [print(tens) for tens in cpu_tens]
        [tens.cuda() for tens in cpu_tens]
      
    def _mv_to_cpu(self,*gpu_tens):
        [tens.cpu() for tens in gpu_tens]





'''
Augmented Dicky-Fuller (ADF) test
* Null hypothesis (H0) — Time series is not stationary.
* Alternative hypothesis (H1) — Time series is stationary.

Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
* Null hypothesis (H0) — Time series is stationary.
* Alternative hypothesis (H1) — Time series is not stationary.

both return tuples where 2nd value is P-value
'''

	
import warnings
warnings.filterwarnings('ignore')

def is_stationary(data: pd.Series, alpha: float = 0.05):
  
    # Test to see if the time series is already stationary
    if kpss(data, regression='c', nlags="auto")[1] > alpha:
    #if adfuller(data)[1] < alpha:
        # stationary - stop inference
        return True
    else:
        # non-stationary - continue inference
        return False

def convergence(x, alpha: float = 0.05):
    ### !!! REMEMBER TO CHECK !!! ###
    #return False
    if isinstance(x, list):
        data = pd.Series(x)
    else:
        raise Exception("input list is not valid type!, expected list.")

    return is_stationary(data, alpha=alpha)



    # def model(self):

    #     n_samples = self.n_samples
    #     k_fixed = self.k_fixed
    #     k_denovo = self.k_denovo
    #     groups = self.groups

    #     #----------------------------- [ALPHA] -------------------------------------
    #     if groups != None:

    #         #num_groups = max(params["groups"]) + 1
    #         n_groups = len(set(groups))
    #         alpha_tissues = dist.Normal(torch.zeros(n_groups, k_fixed + k_denovo), 1).sample()

    #         # sample from the alpha prior
    #         with pyro.plate("k", k_fixed + k_denovo):   # columns
    #             with pyro.plate("n", n_samples):        # rows
    #                 alpha = pyro.sample("latent_exposure", dist.Normal(alpha_tissues[groups, :], 1))
    #     else:
    #         alpha_mean = dist.Normal(torch.zeros(n_samples, k_fixed + k_denovo), 1).sample()

    #         with pyro.plate("k", k_fixed + k_denovo):   # columns
    #             with pyro.plate("n", n_samples):        # rows
    #                 if self.enforce_sparsity:
    #                     alpha = pyro.sample("latent_exposure", dist.Exponential(3))
    #                 else:
    #                     alpha = pyro.sample("latent_exposure", dist.HalfNormal(1))
        
    #     alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))     # normalize
    #     alpha = torch.clamp(alpha, 0,1)

    #     #----------------------------- [BETA] -------------------------------------
    #     if k_denovo==0:
    #         beta_denovo = None
    #     else:
    #         #beta_mean = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()
    #         with pyro.plate("contexts", 96):            # columns
    #             with pyro.plate("k_denovo", k_denovo):  # rows
    #                 beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(1))
    #         beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))   # normalize
    #         beta_denovo = torch.clamp(beta_denovo, 0,1)

    #     #----------------------------- [LIKELIHOOD] -------------------------------------
    #     if self.beta_fixed is None:
    #         beta = beta_denovo
    #         reg = 0
    #     elif beta_denovo is None:
    #         beta = self.beta_fixed
    #         reg = 0
    #     else:
    #         beta = torch.cat((self.beta_fixed, beta_denovo), axis=0)
    #         reg = self._regularizer(self.beta_fixed, beta_denovo)
        
    #     with pyro.plate("contexts2", 96):
    #         with pyro.plate("n2", n_samples):
    #             lk =  dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)).log_prob(self.x)
    #             pyro.factor("loss", lk - reg)
    

    # def guide(self):

    #     n_samples = self.n_samples
    #     k_fixed = self.k_fixed
    #     k_denovo = self.k_denovo
    #     #groups = self.groups

    #     alpha_mean = dist.HalfNormal(torch.ones(n_samples, k_fixed + k_denovo)).sample()

    #     with pyro.plate("k", k_fixed + k_denovo):
    #         with pyro.plate("n", n_samples):
    #             alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than_eq(0))
    #             pyro.sample("latent_exposure", dist.Delta(alpha))

    #     if k_denovo != 0:
    #         beta_mean = dist.HalfNormal(torch.ones(k_denovo, 96)).sample()
    #         with pyro.plate("contexts", 96):
    #             with pyro.plate("k_denovo", k_denovo):
    #                 beta = pyro.param("beta_denovo", beta_mean, constraint=constraints.greater_than_eq(0))
    #                 pyro.sample("latent_signatures", dist.Delta(beta))
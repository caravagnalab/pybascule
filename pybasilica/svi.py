import numpy as np
import pandas as pd
import torch
import pyro
from pyro.infer import SVI,Trace_ELBO, JitTrace_ELBO, TraceEnum_ELBO
from pyro.ops.indexing import Vindex
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
        enumer=False, # if True, will enumerate the Z
        cluster=None,
        alpha_var=1,
        groups=None, 
        beta_fixed=None, 
        compile_model = True, 
        CUDA = False, 
        enforce_sparsity = False,
        store_parameters = False
        regularizer = "cosine"
        ):
        
        self._set_data_catalogue(x)
        self._set_beta_fixed(beta_fixed)
        #self.k_denovo = int(k_denovo)
        self._set_k_denovo(k_denovo)
        self.enumer = enumer
        self.cluster = cluster
        self.alpha_var = alpha_var
        self.lr = lr
        self.n_steps = int(n_steps)
        self.compile_model = compile_model  
        self.CUDA = CUDA
        self.enforce_sparsity = enforce_sparsity
        self.regularizer = regularizer
        self._set_groups(groups)
        self._check_args()

        self.store_parameters = store_parameters

        if not enumer and cluster != None:
            self.z_prior = torch.multinomial(torch.ones(cluster), self.n_samples, replacement=True).float()


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
        cluster = self.cluster  # number of clusters or None
        enumer = self.enumer
        alpha_var = self.alpha_var

        #----------------------------- [ALPHA] -------------------------------------
        if cluster != None:
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster) / cluster))
            # print("MODEL pi", pi)
            with pyro.plate("k1", k_fixed+k_denovo):
                with pyro.plate("g", cluster):
                    alpha_tissues = pyro.sample("alpha_t", dist.HalfNormal(alpha_var))

            with pyro.plate("n",n_samples):
                if enumer != False:
                    z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":enumer})
                else:
                    z = pyro.sample("latent_class", dist.Categorical(pi)).long()
                alpha = pyro.sample("latent_exposure", dist.MultivariateNormal(alpha_tissues[z], torch.eye(k_fixed+k_denovo) * torch.tensor(alpha_var)))

        elif groups != None:

            n_groups = len(set(groups))
            # alpha_tissues = dist.HalfNormal(torch.ones(n_groups, k_fixed + k_denovo)).sample()

            with pyro.plate("k1", k_fixed+k_denovo):
                with pyro.plate("g", n_groups):
                    alpha_tissues = pyro.sample("alpha_t", dist.HalfNormal(alpha_var))

            # sample from the alpha prior
            with pyro.plate("k", k_fixed + k_denovo):   # columns
                with pyro.plate("n", n_samples):        # rows
                    alpha = pyro.sample("latent_exposure", dist.Normal(alpha_tissues[groups,:], alpha_var))
                    # alpha = pyro.sample("latent_exposure", dist.Normal(alpha_t_resh, 1))

        else:
            # alpha_mean = dist.Normal(torch.zeros(n_samples, k_fixed + k_denovo), alpha_var).sample()

            with pyro.plate("k", k_fixed + k_denovo):   # columns
                with pyro.plate("n", n_samples):        # rows
                    if self.enforce_sparsity:
                        alpha = pyro.sample("latent_exposure", dist.Exponential(3))
                    else:
                        alpha = pyro.sample("latent_exposure", dist.HalfNormal(alpha_var))
        
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
            reg = self._regularizer(self.beta_fixed, beta_denovo, self.regularizer)
        
        with pyro.plate("contexts2", 96):
            with pyro.plate("n2", n_samples):
                ## TODO might try to insert the alpha here

                lk =  dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)).log_prob(self.x)
                pyro.factor("loss", lk.sum() + (reg * self.x.shape[0] * self.x.shape[1]))
    

    def guide(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        groups = self.groups
        cluster = self.cluster
        enumer = self.enumer
        alpha_var = self.alpha_var

        # Alpha ---------------------------------------------------------------
        if cluster != None:
            pi_param = pyro.param("pi_param", torch.ones(cluster), constraint=constraints.simplex)
            pi = pyro.sample("pi", dist.Delta(pi_param).to_event(1))
            # print("GUIDE pi", pi)

            alpha_tissues = pyro.param("alpha_t_param", dist.HalfNormal(torch.ones(cluster, k_fixed + k_denovo) * torch.tensor(alpha_var)).sample(),
                                       constraint=constraints.greater_than_eq(0))
            
            with pyro.plate("k1", k_fixed+k_denovo):
                with pyro.plate("g", cluster):
                    pyro.sample("alpha_t", dist.Delta(alpha_tissues))
            
            if enumer == False:
                z_par = pyro.param("latent_class_p", lambda: self.z_prior)

            with pyro.plate("n",n_samples):  # + (n_samples) BATCH
                if enumer != False:
                    z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":enumer})
                else:
                    z = pyro.sample("latent_class", dist.Delta(z_par)).long()
                    # Delta shape -> batch=(n_samples) + event=()
                    # adding plate dims -> BATCH : (k_denovo+k_fixed) (n_samples)
                alpha_p = pyro.param("alpha", lambda: alpha_tissues[z, :], constraint=constraints.greater_than_eq(0))
                alpha = pyro.sample("latent_exposure", dist.Delta(alpha_p).to_event(1))

        elif groups != None:
            n_groups = len(set(groups))
            # alpha_tissues = dist.HalfNormal(torch.ones(n_groups, k_fixed + k_denovo)).sample()
            alpha_tissues = pyro.param("alpha_t_param", dist.HalfNormal(torch.ones(n_groups, k_fixed + k_denovo)).sample(),
                                       constraint=constraints.greater_than_eq(0))
            
            with pyro.plate("k1", k_fixed+k_denovo):
                with pyro.plate("g", n_groups):
                    pyro.sample("alpha_t", dist.Delta(alpha_tissues))

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
        # CHECK
        # with pyro.plate("k", k_fixed + k_denovo):
        #     with pyro.plate("n", n_samples):
        #         alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than_eq(0))
        #         alpha = torch.clamp(alpha, 0,1)
        #         pyro.sample("latent_exposure", dist.Delta(alpha))

        # Beta ----------------------------------------------------------------
        if k_denovo != 0:
            beta_mean = dist.HalfNormal(torch.ones(k_denovo, 96)).sample()
            with pyro.plate("contexts", 96):
                with pyro.plate("k_denovo", k_denovo):
                    beta = pyro.param("beta_denovo", beta_mean, constraint=constraints.greater_than_eq(0))
                    beta = torch.clamp(beta, 0,1)
                    pyro.sample("latent_signatures", dist.Delta(beta))

    
    def _regularizer(self, beta_fixed, beta_denovo, reg_type = "cosine"):
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

        if reg_type == "cosine":
          for fixed in beta_fixed:
            for denovo in beta_denovo:
              loss +=  torch.log((1 - F.cosine_similarity(fixed, denovo, dim = -1)))
        else:
          for fixed in beta_fixed:
            for denovo in beta_denovo:
              loss += torch.log(F.kl_div(torch.log(fixed), torch.log(denovo), log_target = True, reduction="batchmean"))
        #print("loss:", loss)
        return loss
    

    def _get_unique_beta(self, beta_fixed, beta_denovo):
        if beta_fixed is None:
            beta = beta_denovo
        elif beta_denovo is None:
            beta = beta_fixed
        else:
            beta = torch.cat((beta_fixed, beta_denovo), axis=0)

        return beta
    
    
    def _likelihood(self, M, alpha, beta_fixed, beta_denovo):
        
        # if beta_fixed is None:
        #     beta = beta_denovo
        # elif beta_denovo is None:
        #     beta = beta_fixed
        # else:
        #     beta = torch.cat((beta_fixed, beta_denovo), axis=0)

        beta = self._get_unique_beta(beta_fixed, beta_denovo)

        _log_like_matrix = dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(M, axis=1)), alpha), beta)).log_prob(M)
        _log_like_sum = torch.sum(_log_like_matrix)
        _log_like = float("{:.3f}".format(_log_like_sum.item()))

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
        
        if self.cluster != None and self.enumer != False:
            elbo = TraceEnum_ELBO()
        elif self.compile_model and not self.CUDA:
            elbo = JitTrace_ELBO()
        else:
            elbo = Trace_ELBO()

        train_params = []
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
            # alpha = torch.exp(alpha)
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

            if self.store_parameters:
                train_params.append(self.get_param_dict())
        
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

        self.train_params = train_params
        self.losses = losses
        self.likelihoods = likelihoods
        self._set_alpha()
        self._set_beta_denovo()
        self._set_clusters()
        self._set_bic()
        self.likelihood = self._likelihood(self.x, self.alpha, self.beta_fixed, self.beta_denovo)
        # self.regularization = self._regularizer(self.beta_fixed, self.beta_denovo)

    def _get_param(self, param_name, normalize=False):

        try:
            par = pyro.param(param_name)

            if self.CUDA and torch.cuda.is_available():
                par = par.cpu()

            try:
                par = par.clone().detach()
            finally:
                if normalize: 
                    par = par / (torch.sum(par, 1).unsqueeze(-1))
        except:
            return None

        return par


    def _set_alpha(self):
        self.alpha = self._get_param("alpha", normalize=True)
        self.alpha_unn = self._get_param("alpha", normalize=False)
        try: 
            self.alpha_prior = self._get_param("alpha_t_param", normalize=True)
            self.alpha_prior_unn = self._get_param("alpha_t_param", normalize=False)
        except: 
            self.alpha_prior = None
            self.alpha_prior_unn = None
        # exposure matrix
        # alpha = pyro.param("alpha")
        # try: alpha_prior = pyro.param("alpha_t_param") 
        # except: alpha_prior = None

        # if self.CUDA and torch.cuda.is_available():
        #     alpha = alpha.cpu()
        #     try: alpha_prior = alpha_prior.cpu()
        #     except: alpha_prior = None

        # alpha = alpha.clone().detach()
        # #alpha = torch.exp(alpha)
        # self.alpha = (alpha / (torch.sum(alpha, 1).unsqueeze(-1)))
        # self.alpha_unn = alpha

        # try:
        #     alpha_prior = alpha_prior.clone().detach()
        #     self.alpha_prior = alpha_prior / (torch.sum(alpha_prior, 1).unsqueeze(-1))
        #     self.alpha_prior_unn = alpha_prior
        # except:
        #     self.alpha_prior = None


    def _set_beta_denovo(self):
        # signature matrix
        if self.k_denovo == 0:
            self.beta_denovo = None
        else:
            self.beta_denovo = self._get_param("beta_denovo", normalize=True)
            # beta_denovo = pyro.param("beta_denovo")
            # if self.CUDA and torch.cuda.is_available():
            #     beta_denovo = beta_denovo.cpu()
            # beta_denovo = beta_denovo.clone().detach()
            # #beta_denovo = torch.exp(beta_denovo)
            # self.beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))


    def _set_clusters(self):
        if self.cluster is None:
            return
        
        # pi = pyro.param("pi_param")
        # print(pi)

        # if self.CUDA and torch.cuda.is_available():
        #     pi = pi.cpu()
        # self.pi = pi.clone().detach()

        self.pi = self._get_param("pi", normalize=False)

        if self.enumer == False:
            self.z = pyro.param("latent_class_p")
            self.z = self._get_param("latent_class_p", normalize=False)
        else:
            self.z = self._compute_posterior_probs()


    def _logsumexp(self, weighted_lp) -> torch.Tensor:
        '''
        Returns `m + log( sum( exp( weighted_lp - m ) ) )`
        - `m` is the the maximum value of weighted_lp for each observation among the K values
        - `torch.exp(weighted_lp - m)` to perform some sort of normalization
        In this way the `exp` for the maximum value will be exp(0)=1, while for the 
        others will be lower than 1, thus the sum across the K components will sum up to 1.
        '''
        m = torch.amax(weighted_lp, dim=0)  # the maximum value for each observation among the K values
        summed_lk = m + torch.log(torch.sum(torch.exp(weighted_lp - m), axis=0))
        return summed_lk 


    def get_param_dict(self):
        params = dict()
        params["alpha"] = self._get_param("alpha", normalize=True)
        params["alpha_prior"] = self._get_param("alpha_t_param", normalize=True)

        params["beta_d"] =  self._get_param("beta_denovo", normalize=True)
        params["beta_f"] = self.beta_fixed

        params["pi"] = self._get_param("pi", normalize=False)

        return params


    def _compute_posterior_probs(self):
        params = self.get_param_dict()
        M = torch.tensor(self.x)
        cluster = self.cluster
        n_samples = self.n_samples

        try:
            beta = torch.cat((torch.tensor(params["beta_f"]), torch.tensor(params["beta_d"])), axis=0) 
        except:
            beta = torch.tensor(params["beta_d"])
        
        z = torch.zeros(n_samples)

        for n in range(n_samples):
            m_n = M[n,:].unsqueeze(0)
            ll_nk = torch.zeros((cluster, M.shape[1]))

            for k in range(cluster):
                muts_n = torch.sum(m_n, axis=1).float()  # muts for patient n
                rate = torch.matmul( \
                    torch.matmul( torch.diag(muts_n), params["alpha_prior"][k,:].unsqueeze(0) ), \
                    beta.float() )

                # compute weighted log probability
                ll_nk[k,:] = torch.log(params["pi"][k]) + pyro.distributions.Poisson( rate ).log_prob(m_n) 

            ll_nk_sum = ll_nk.sum(axis=1)  # sum over the contexts -> reaches a tensor of shape (n_clusters)

            ll = self._logsumexp(ll_nk_sum)
            probs = torch.exp(ll_nk_sum - ll)

            best_cl = torch.argmax(probs)
            z[n] = best_cl

        return z


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
        self.alpha = pd.DataFrame(np.array(self.alpha), index=sample_names , columns=fixed_names + denovo_names)


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
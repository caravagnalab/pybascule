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
from logging import warning

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
        enumer = False, # if True, will enumerate the Z
        cluster = None,
        hyperparameters = {"alpha_var":1, "alpha_prior_var":1, "exp_rate":10, "beta_var":1, "eps_var":10},
        groups = None,
        beta_fixed = None,
        compile_model = True,
        CUDA = False,
        enforce_sparsity = False,
        store_parameters = False,
        regularizer = "cosine",
        reg_weight = 1.,
        reg_bic = True, 
        stage = "random_noise", 
        regul_compare = None,
        seed = 10,
        initializ_seed = True,
        initializ_pars_fit = False
        ):

        self._hyperpars_default = {"alpha_var":1, "alpha_prior_var":1, "exp_rate":10, "beta_var":1, "eps_var":10}

        self._set_data_catalogue(x)
        self._set_fit_settings(enforce_sparsity, lr, n_steps, compile_model, CUDA, regularizer, reg_weight, reg_bic, \
                               store_parameters, stage, initializ_seed, initializ_pars_fit, seed)

        self._set_beta_fixed(beta_fixed)
        self._set_k_denovo(k_denovo)

        self._set_hyperparams(enumer, cluster, groups, hyperparameters)

        self._fix_zero_denovo_null_reference()
        self._set_external_catalogue(regul_compare)


    def _set_fit_settings(self, enforce_sparsity, lr, n_steps, compile_model, CUDA, \
                          regularizer, reg_weight, reg_bic, store_parameters, stage,
                          initializ_seed, initializ_pars_fit, seed):
        self.enforce_sparsity = enforce_sparsity
        self.lr = lr
        self.n_steps = int(n_steps)
        self.compile_model = compile_model
        self.CUDA = CUDA
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.reg_bic = reg_bic

        self.store_parameters = store_parameters
        self.stage = stage

        self.initializ_seed = initializ_seed

        self._initializ_params_with_fit = initializ_pars_fit
        self.seed = seed

        if self.initializ_seed is True and self._initializ_params_with_fit is True:
            warning("\n\t`initializ_seed` and `initializ_pars_fit` can't be both `True`.\n\tSetting the initialization of the seed to `False` and running with seed " +
                    str(seed))
            self.initializ_seed = False


    def _set_hyperparams(self, enumer, cluster, groups, hyperparameters):
        self.enumer = enumer
        self.cluster = cluster
        self._set_groups(groups)

        self.init_params = None

        if hyperparameters is None:
            self.hyperparameters = self._hyperpars_default
        else:
            self.hyperparameters = dict()
            for parname in self._hyperpars_default.keys():
                self.hyperparameters[parname] = hyperparameters.get(parname, self._hyperpars_default[parname])

        if not enumer and cluster != None:
            self.hyperparameters["z_prior"] = torch.multinomial(torch.ones(cluster), self.n_samples, replacement=True).float()


    def _fix_zero_denovo_null_reference(self):
        if self.k_denovo == 0 and self.k_fixed == 0:
            self.stage = "random_noise"
            self.beta_fixed = torch.zeros(1, self.contexts, dtype=torch.float64)
            self.k_fixed = 1
            self._noise_only = True
        else:
            self._noise_only = False


    def _set_data_catalogue(self, x):
        try:
            self.x = torch.tensor(x.values, dtype=torch.float64)
            self.n_samples = x.shape[0]
            self.contexts = x.shape[1]
        except:
            raise Exception("Invalid mutations catalogue, expected Dataframe!")


    def _set_beta_fixed(self, beta_fixed):
        try:
            self.beta_fixed = torch.tensor(beta_fixed.values, dtype=torch.float64)
            if len(self.beta_fixed.shape)==1:
                self.beta_fixed = self.beta_fixed.reshape(1, self.beta_fixed.shape[0])

            self.k_fixed = beta_fixed.shape[0]

        except:
            if beta_fixed is None:
                self.beta_fixed = None
                self.k_fixed = 0
            else:
                raise Exception("Invalid fixed signatures catalogue, expected DataFrame!")

        if self.k_fixed > 0:
            self._fix_zero_contexts()


    def _fix_zero_contexts(self):
        colsums = torch.sum(self.beta_fixed, axis=0)
        zero_contexts = torch.where(colsums==0)[0]
        if torch.any(colsums == 0):
            random_sig = [0] if self.k_fixed == 1 else torch.randperm(self.beta_fixed.shape[0])[:torch.numel(zero_contexts)]

            for rr in random_sig:
                self.beta_fixed[rr, zero_contexts] = 1e-07

            self.beta_fixed = self.beta_fixed / (torch.sum(self.beta_fixed, 1).unsqueeze(-1))


    def _set_external_catalogue(self, regul_compare):
        try:
            self.regul_compare = torch.tensor(regul_compare.values, dtype=torch.float64)
            if self.CUDA and torch.cuda.is_available():
                self.regul_compare = self.regul_compare.cuda()
        except:
            if regul_compare is None:
                self.regul_compare = None
            else:
                raise Exception("Invalid external signatures catalogue, expected DataFrame!")


    def _set_k_denovo(self, k_denovo):
        if isinstance(k_denovo, int):
            self.k_denovo = k_denovo
        else:
            raise Exception("Invalid k_denovo value, expected integer!")


    def _set_groups(self, groups):
        if groups is None:
            self.groups = None
            self.n_groups = None
        else:
            if isinstance(groups, list) and len(groups)==self.n_samples:
                self.groups = torch.tensor(groups).long()
                # n_groups = len(set(groups)) # WRONG!!!!! not working since groups is a tensor
                self.n_groups = torch.tensor(groups).unique().numel()

            else:
                raise Exception("invalid groups argument, expected 'None' or a list with {} elements!".format(self.n_samples))


    def _check_args(self):
        pass
        # if self.k_denovo==0 and self.k_fixed==0:
        #     raise Exception("No. of denovo and fixed signatures could NOT be zero at the same time!")
        #if self.stage=="two":
        #    if self.alpha0 == None:
        #        raise Exception("stage two needs first stage exposure!")


    def model(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        groups = self.groups
        cluster = self.cluster  # number of clusters or None
        enumer = self.enumer
        n_groups = self.n_groups

        alpha_var = self.hyperparameters["alpha_var"]
        alpha_prior_var = self.hyperparameters["alpha_prior_var"]
        exp_rate = self.hyperparameters["exp_rate"]
        beta_var = self.hyperparameters["beta_var"]
        eps_var = self.hyperparameters["eps_var"]

        # Alpha
        if groups is not None:
            if self._noise_only:
                alpha = torch.zeros(n_samples, 1, dtype=torch.float64)
            else:
                with pyro.plate("k1", k_fixed+k_denovo):
                    with pyro.plate("g", n_groups):
                        # G x K matrix
                        alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(alpha_prior_var))

                # sample from the alpha prior
                with pyro.plate("k", k_fixed + k_denovo):  # columns
                    with pyro.plate("n", n_samples):  # rows
                        alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[groups,:], alpha_var))

                alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
                alpha = torch.clamp(alpha, 0, 1)

        elif cluster is not None:
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster) / cluster))
            with pyro.plate("k1", k_fixed + k_denovo):
                with pyro.plate("g", cluster):
                    # G x K matrix
                    alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(alpha_prior_var))

        else:
            if self._noise_only:
                alpha = torch.zeros(n_samples, 1, dtype=torch.float64)
            else:
                with pyro.plate("k", k_fixed + k_denovo):  # columns
                    with pyro.plate("n", n_samples):  # rows
                        if self.enforce_sparsity:
                            alpha = pyro.sample("latent_exposure", dist.Exponential(exp_rate))
                        else:
                            alpha = pyro.sample("latent_exposure", dist.HalfNormal(alpha_var))

                alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
                alpha = torch.clamp(alpha, 0, 1)

        # Epsilon
        if self.stage == "random_noise":
            with pyro.plate("contexts3", self.contexts):  # columns
                    with pyro.plate("n3", n_samples):  # rows
                        epsilon = pyro.sample("latent_m", dist.HalfNormal(eps_var))
        else:
            epsilon = None

        # Beta
        if k_denovo == 0:
            beta_denovo = None
        else:
            with pyro.plate("contexts", self.contexts):  # columns
                with pyro.plate("k_denovo", k_denovo):  # rows
                    beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(beta_var))

            beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))  # normalize
            beta_denovo = torch.clamp(beta_denovo, 0, 1)

        beta = self._get_unique_beta(self.beta_fixed, beta_denovo)
        reg = self._regularizer(self.beta_fixed, beta_denovo, self.regularizer)
        self.reg = reg

        # Observations
        with pyro.plate("n2", n_samples):
            if cluster is not None:
                z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":enumer})
                alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[z], alpha_var).expand([1, k_fixed+k_denovo]))

            a = torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)

            if self.stage == "random_noise":
                xx = a + epsilon
                lk =  dist.Poisson(xx).log_prob(self.x)
            else:
                lk =  dist.Poisson(a).log_prob(self.x)
            # pyro.factor("loss", lk + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1]))
            pyro.factor("loss", lk.sum() + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1]))


    def guide(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        groups = self.groups
        cluster = self.cluster
        enumer = self.enumer
        n_groups = self.n_groups

        init_params = self._initialize_params()

        # Alpha
        if groups is not None:
            if not self._noise_only:
                alpha_prior = pyro.param("alpha_t_param", init_params["alpha_t_param"], constraint=constraints.greater_than_eq(0))

                with pyro.plate("k1", k_fixed+k_denovo):
                    with pyro.plate("g", n_groups):
                        pyro.sample("alpha_t", dist.Delta(alpha_prior))

                with pyro.plate("k", k_fixed + k_denovo):  # columns
                    with pyro.plate("n", n_samples):  # rows
                        alpha = pyro.param("alpha", alpha_prior[groups, :], constraint=constraints.greater_than_eq(0))
                        pyro.sample("latent_exposure", dist.Delta(alpha))

        elif cluster is not None:
            if not self._noise_only:
                pi_param = pyro.param("pi_param", init_params["pi_param"], constraint=constraints.simplex)
                pi = pyro.sample("pi", dist.Delta(pi_param).to_event(1))

                alpha_prior = pyro.param("alpha_t_param", init_params["alpha_t_param"], constraint=constraints.greater_than_eq(0))

                if enumer == False:
                    z_par = pyro.param("latent_class_p", lambda: self.hyperparameters["z_prior"])
                    z = pyro.sample("latent_class", dist.Delta(z_par))

                with pyro.plate("k1", k_fixed+k_denovo):
                    with pyro.plate("g", cluster):
                        pyro.sample("alpha_t", dist.Delta(alpha_prior))

                with pyro.plate("n2", n_samples):
                    if enumer != False:
                        z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":enumer})

                    alpha = pyro.param("alpha", lambda: alpha_prior[z.long()], constraint=constraints.greater_than_eq(0))
                    pyro.sample("latent_exposure", dist.Delta(alpha).expand([1, k_fixed+k_denovo]))

        # No groups
        else:
            if not self._noise_only:
                alpha_mean = init_params["alpha_mean"]

                with pyro.plate("k", k_fixed + k_denovo):
                    with pyro.plate("n", n_samples):
                        if self.enforce_sparsity:
                            alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than(0.0))
                        else:
                            alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than_eq(0.0))
                        pyro.sample("latent_exposure", dist.Delta(alpha))

        # Epsilon 
        if self.stage == "random_noise":
            eps_var = pyro.param("lambda_epsilon", init_params["epsilon_var"], constraint=constraints.positive)

            with pyro.plate("contexts3", self.contexts):
                with pyro.plate("n3", n_samples):
                    pyro.sample("latent_m", dist.HalfNormal(eps_var))

        # Beta
        if k_denovo > 0:
            beta_par = init_params["beta_dn_param"]
            with pyro.plate("contexts", self.contexts):
                with pyro.plate("k_denovo", k_denovo):
                    beta = pyro.param("beta_denovo", beta_par, constraint=constraints.greater_than_eq(0.0))
                    pyro.sample("latent_signatures", dist.Delta(beta))


    def _initialize_params_hier(self):
        groups_true = torch.tensor(self.groups)
        steps_true = self.n_steps
        hyperpars_true = self.hyperparameters
        
        self.groups = None
        self.n_steps = 50
        self.initializ_seed = False
        self.hyperparameters = self._hyperpars_default

        params = dict()

        print("Initializing parameters with non-hierarchical fit")
        self._fit(set_attributes=False)

        if self.cluster is not None:
            raise NotImplementedError

        alpha = self._get_param("alpha", normalize=True, to_cpu=False)
        alpha_t_param = torch.zeros((groups_true.unique().numel(), self.k_fixed+self.k_denovo), dtype=torch.float64)

        for gid in groups_true.unique().tolist(): alpha_t_param[gid] = torch.mean(alpha[groups_true == gid, :], dim=0)
        params["alpha_t_param"] = alpha_t_param

        if self.k_denovo > 0: params["beta_dn_param"] = self._get_param("beta_denovo", to_cpu=False, normalize=True)

        params["epsilon_var"] = torch.ones(self.n_samples, self.contexts, dtype=torch.float64)

        pyro.get_param_store().clear()

        self.groups = groups_true
        self.n_steps = steps_true
        self.hyperparameters = hyperpars_true

        return params


    def _initialize_params_nonhier(self):
        params = dict()

        alpha_var = self.hyperparameters["alpha_var"]
        alpha_prior_var = self.hyperparameters["alpha_prior_var"]
        exp_rate = self.hyperparameters["exp_rate"]
        beta_var = self.hyperparameters["beta_var"]
        eps_var = self.hyperparameters["eps_var"]

        if self.cluster is not None:
            params["pi_param"] = torch.ones(self.cluster, dtype=torch.float64)
            params["alpha_t_param"] = dist.HalfNormal(torch.ones(self.cluster, self.k_fixed + self.k_denovo, dtype=torch.float64) * \
                                                      torch.tensor(alpha_prior_var)).sample()
        elif self.groups is not None:
            params["alpha_t_param"] = dist.HalfNormal(torch.ones(self.n_groups, self.k_fixed + self.k_denovo, dtype=torch.float64) * \
                                                      torch.tensor(alpha_prior_var)).sample()
        else:
            if self.enforce_sparsity:
                params["alpha_mean"] = dist.Exponential(torch.ones(self.n_samples, self.k_fixed + self.k_denovo, dtype=torch.float64) * exp_rate).sample()
            else:
                params["alpha_mean"] = dist.HalfNormal(torch.ones(self.n_samples, self.k_fixed + self.k_denovo, dtype=torch.float64) * alpha_var).sample()

        params["epsilon_var"] = torch.ones(self.n_samples, self.contexts, dtype=torch.float64) * eps_var

        if self.k_denovo > 0:
            params["beta_dn_param"] = dist.HalfNormal(torch.ones(self.k_denovo, self.contexts, dtype=torch.float64) * beta_var).sample()

        return params


    def _initialize_params(self):
        if self.init_params is None:

            if self.groups is not None and self._initializ_params_with_fit:
                self.init_params = self._initialize_params_hier()
            else:
                self.init_params = self._initialize_params_nonhier()

        return self.init_params


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

        if self.regul_compare is not None:
            beta_fixed = self.regul_compare

        if beta_fixed is None or beta_denovo is None or self._noise_only:
            return loss
        
        beta_fixed[beta_fixed==0] = 1e-07

        if reg_type == "cosine":
            for fixed in beta_fixed:
                for denovo in beta_denovo:
                    loss += torch.log((1 - F.cosine_similarity(fixed, denovo, dim = -1)))
        elif reg_type == "KL":
            for fixed in beta_fixed:
                for denovo in beta_denovo:
                    loss += torch.log(F.kl_div(torch.log(fixed), torch.log(denovo), log_target = True, reduction="batchmean"))
        else:
            raise("The regularization admits either 'cosine' or 'KL'")

        return loss


    def _get_unique_beta(self, beta_fixed, beta_denovo):
        if beta_fixed is None: 
            return beta_denovo

        if beta_denovo is None or self._noise_only:
            return beta_fixed
        
        return torch.cat((beta_fixed, beta_denovo), axis=0)


    def _likelihood(self, M, alpha, beta_fixed, beta_denovo, eps_var=None):
        beta = self._get_unique_beta(beta_fixed, beta_denovo)

        ssum = torch.sum(M, axis=1)
        ddiag = torch.diag(ssum)
        mmult1 = torch.matmul(ddiag, alpha)

        a = torch.matmul(mmult1, beta)

        if eps_var == None:
            _log_like_matrix = dist.Poisson(a).log_prob(M)
        else:
            xx = a + dist.HalfNormal(eps_var).sample()
            _log_like_matrix = dist.Poisson(xx).log_prob(M)

        _log_like_sum = torch.sum(_log_like_matrix)
        _log_like = float("{:.3f}".format(_log_like_sum.item()))

        return _log_like


    def _initialize_seed(self, optim, elbo, seed):
        '''
        Auxiliary function to optimize the random initialization of the SVI object.
        '''
        pyro.set_rng_seed(seed)
        pyro.get_param_store().clear()

        svi = SVI(self.model, self.guide, optim, elbo)
        loss = svi.step()
        self.init_params = None

        return np.round(loss, 3), seed


    def _fit(self, set_attributes=True):
        pyro.clear_param_store()  # always clear the store before the inference

        if self.CUDA and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.x = self.x.cuda()
            if self.beta_fixed is not None:
                self.beta_fixed = self.beta_fixed.cuda()
            if self.regul_compare is not None:
                self.regul_compare = self.regul_compare.cuda()
        else:
            torch.set_default_tensor_type(t=torch.FloatTensor)

        if self.cluster != None and self.enumer != False:
            elbo = TraceEnum_ELBO()
        elif self.compile_model and not self.CUDA:
            elbo = JitTrace_ELBO()
        else:
            elbo = Trace_ELBO()

        min_steps = 50

        train_params = []
        # learning global parameters
        adam_params = {"lr": self.lr}
        optimizer = Adam(adam_params)

        if self.initializ_seed:
            _, self.seed = min([self._initialize_seed(optimizer, elbo, seed) for seed in range(50)], key = lambda x: x[0])

        print("Running with seed {}\n".format(self.seed))
        pyro.set_rng_seed(self.seed)
        pyro.get_param_store().clear()

        self._initialize_params()

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        losses = []
        regs = []
        likelihoods = []
        for _ in range(self.n_steps):   # inference - do gradient steps

            loss = svi.step()
            losses.append(loss)
            regs.append(self.reg)

            # create likelihoods 
            alpha = self._get_param("alpha", normalize=True, to_cpu=False)
            eps_var = self._get_param("eps_var", normalize=False, to_cpu=False)
            beta_denovo = self._get_param("beta_denovo", normalize=True, to_cpu=False)
            likelihoods.append(self._likelihood(self.x, alpha, self.beta_fixed, beta_denovo, eps_var))

            if self.store_parameters:
                train_params.append(self.get_param_dict())

            # convergence test 
            if len(losses) >= min_steps and len(losses) % min_steps == 0 and convergence(x=losses[-min_steps:], alpha=0.05):
                break

        if set_attributes is False:
            return

        if self.CUDA and torch.cuda.is_available():
          self.x = self.x.cpu()
          if self.beta_fixed is not None:
            self.beta_fixed = self.beta_fixed.cpu()
          if self.regul_compare is not None:
                self.regul_compare = self.regul_compare.cpu()

        self.train_params = train_params
        self.losses = losses
        self.likelihoods = likelihoods
        self.regs = regs
        self._set_params()
        self._set_bic()
        self.likelihood = self._likelihood(self.x, self.alpha, self.beta_fixed, self.beta_denovo, self.eps_var)


    def _set_params(self):
        self._set_alpha()
        self._set_beta_denovo()
        self._set_epsilon()
        self._set_clusters()

        if isinstance(self.groups, torch.Tensor):
            self.groups = self.groups.tolist()


    def _get_param(self, param_name, normalize=False, to_cpu=True):
        try:
            if param_name == "beta_fixed": par = self.beta_fixed
            else: par = pyro.param(param_name)

            if to_cpu and self.CUDA and torch.cuda.is_available(): par = par.cpu()

            try:
                par = par.clone().detach()
            finally:
                if normalize:
                    par = par / (torch.sum(par, 1).unsqueeze(-1))
        except:
            if self._noise_only and param_name=="alpha":
                # TODO check when running in GPU 
                return torch.zeros(self.n_samples, 1, dtype=torch.float64)
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


    def _set_beta_denovo(self):
        if self.k_denovo == 0:
            self.beta_denovo = None
        else:
            self.beta_denovo = self._get_param("beta_denovo", normalize=True)


    def _set_epsilon(self):
        if self.stage=="random_noise":
            self.eps_var = self._get_param("lambda_epsilon", normalize=False)
        else:
            self.eps_var = None


    def _set_clusters(self):
        if self.cluster is None:
            return

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
        params["beta_f"] = self._get_param("beta_fixed")

        params["pi"] = self._get_param("pi", normalize=False)

        if self.stage=="random_noise":
            params["lambda_epsilon"] = self._get_param("lambda_epsilon", normalize=False)
        else:
            params["lambda_epsilon"] = None

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

        _log_like = self._likelihood(M, alpha, self.beta_fixed, self.beta_denovo, self.eps_var)

        # adding regularizer
        if self.reg_weight != 0 and self.reg_bic:
            reg = self._regularizer(self.beta_fixed, self.beta_denovo, reg_type = self.regularizer)
            _log_like += self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1])

        k = self._number_of_params() 
        n = M.shape[0] * M.shape[1]
        bic = k * torch.log(torch.tensor(n, dtype=torch.float64)) - (2 * _log_like)

        self.bic = bic.item()


    def _number_of_params(self):
        if self.k_denovo == 0 and torch.sum(self.beta_fixed) == 0:
            k = 0
        else:
            k = (self.n_samples * (self.k_denovo + self.k_fixed)) + ((self.k_denovo) * self.contexts)
        
        if self.eps_var is not None:
            k = k + self.eps_var.shape[0] * self.eps_var.shape[1]
        
        return k


    def convert_to_dataframe(self, x, beta_fixed):

        if isinstance(self.beta_fixed, pd.DataFrame):
            self.beta_fixed = torch.tensor(self.beta_fixed.values, dtype=torch.float64)

        # mutations catalogue
        self.x = x
        sample_names = list(x.index)
        mutation_features = list(x.columns)

        # fixed signatures
        fixed_names = []
        if self.beta_fixed is not None and torch.sum(self.beta_fixed) > 0:
            fixed_names = list(beta_fixed.index)
            self.beta_fixed = beta_fixed

        # denovo signatures
        denovo_names = []
        if self.beta_denovo is not None:
            for d in range(self.k_denovo):
                denovo_names.append("D"+str(d+1))
            self.beta_denovo = pd.DataFrame(np.array(self.beta_denovo), index=denovo_names, columns=mutation_features)

        # alpha
        if len(fixed_names+denovo_names) > 0:
            self.alpha = pd.DataFrame(np.array(self.alpha), index=sample_names , columns=fixed_names + denovo_names)

        # epsilon variance
        if self.stage=="random_noise":
            self.eps_var = pd.DataFrame(np.array(self.eps_var), index=sample_names , columns=mutation_features)
        else:
            self.eps_var = None


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
    #         #beta_mean = dist.Normal(torch.zeros(k_denovo, self.contexts), 1).sample()
    #         with pyro.plate("contexts", self.contexts):            # columns
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

    #     with pyro.plate("contexts2", self.contexts):
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
    #         beta_mean = dist.HalfNormal(torch.ones(k_denovo, self.contexts)).sample()
    #         with pyro.plate("contexts", self.contexts):
    #             with pyro.plate("k_denovo", k_denovo):
    #                 beta = pyro.param("beta_denovo", beta_mean, constraint=constraints.greater_than_eq(0))
    #                 pyro.sample("latent_signatures", dist.Delta(beta))

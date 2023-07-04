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
        enumer = "parallel", # if True, will enumerate the Z
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
        initializ_pars_fit = False,
        new_hier = False,
        regul_denovo = True
        ):

        self._hyperpars_default = {"alpha_var":1, "alpha_prior_var":1, "exp_rate":10, "beta_var":1, "eps_var":10, "alpha_noise_var":0.1}
        self.regul_denovo = regul_denovo
        self.new_hier = new_hier

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
        if groups is None and cluster is None:
            self.new_hier = False
        self.enumer = enumer

        self.cluster = cluster
        if self.cluster is not None: self.cluster = int(self.cluster)
        self._set_groups(groups)

        self.init_params = None

        if hyperparameters is None:
            self.hyperparameters = self._hyperpars_default
        else:
            self.hyperparameters = dict()
            for parname in self._hyperpars_default.keys():
                self.hyperparameters[parname] = hyperparameters.get(parname, self._hyperpars_default[parname])


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
            self.regul_compare = self._to_gpu(self.regul_compare)
            # if self.CUDA and torch.cuda.is_available():
            #     self.regul_compare = self.regul_compare.cuda()
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


    # def _sample_alpha_hier(self):
    #     if self._noise_only:
    #             alpha = torch.zeros(self.n_samples, 1, dtype=torch.float64)
    #     else:
    #         with pyro.plate("k1", self.k_fixed + self.k_denovo):
    #             with pyro.plate("g", self.n_groups):
    #                 # G x K matrix
    #                 alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(self.hyperparameters["alpha_prior_var"]))

    #         # sample from the alpha prior
    #         # alpha_noise = torch.zeros((self.n_samples, self.k_fixed + self.k_denovo))
    #         with pyro.plate("k", self.k_fixed + self.k_denovo):  # columns
    #             with pyro.plate("n", self.n_samples):  # rows
    #                 if self.new_hier: 
    #                     alpha_noise = pyro.sample("alpha_noise", dist.HalfNormal(self.hyperparameters["alpha_noise_var"]))
    #                     alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[self.groups,:], alpha_noise))
    #                 else:
    #                     alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[self.groups,:], self.hyperparameters["alpha_var"]))

    #         alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    #         alpha = torch.clamp(alpha, 0, 1)
        
    #     return alpha


    # def _sample_alpha_flat(self):
    #     if self._noise_only:
    #         alpha = torch.zeros(self.n_samples, 1, dtype=torch.float64)
    #     else:
    #         with pyro.plate("k", self.k_fixed + self.k_denovo):  # columns
    #             with pyro.plate("n", self.n_samples):  # rows
    #                 if self.enforce_sparsity:
    #                     alpha = pyro.sample("latent_exposure", dist.Exponential(self.hyperparameters["exp_rate"]))
    #                 else:
    #                     alpha = pyro.sample("latent_exposure", dist.HalfNormal(self.hyperparameters["alpha_var"]))

    #         alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    #         alpha = torch.clamp(alpha, 0, 1)

    #     return alpha

    # def _sample_epsilon(self):
    #     # Epsilon
    #     if self.stage == "random_noise":
    #         with pyro.plate("contexts3", self.contexts):  # columns
    #                 with pyro.plate("n3", self.n_samples):  # rows
    #                     epsilon = pyro.sample("latent_m", dist.HalfNormal(self.hyperparameters["eps_var"]))
    #     else:
    #         epsilon = None

    #     return epsilon


    # def _sample_beta_dn(self):
    #     if self.k_denovo == 0:
    #         beta_denovo = None
    #     else:
    #         with pyro.plate("contexts", self.contexts):  # columns
    #             with pyro.plate("k_denovo", self.k_denovo):  # rows
    #                 beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(self.hyperparameters["beta_var"]))

    #         beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))  # normalize
    #         beta_denovo = torch.clamp(beta_denovo, 0, 1)

    #     return beta_denovo


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
        alpha_noise_var = self.hyperparameters["alpha_noise_var"]

        # Alpha
        if groups is not None:
            if self._noise_only:
                alpha = torch.zeros(self.n_samples, 1, dtype=torch.float64)
            else:
                with pyro.plate("k1", self.k_fixed + self.k_denovo):
                    with pyro.plate("g", self.n_groups):
                        # G x K matrix
                        alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(self.hyperparameters["alpha_prior_var"]))

                # sample from the alpha prior
                # alpha_noise = torch.zeros((self.n_samples, self.k_fixed + self.k_denovo))
                with pyro.plate("k", self.k_fixed + self.k_denovo):  # columns
                    with pyro.plate("n", self.n_samples):  # rows
                        if self.new_hier: 
                            alpha_noise = pyro.sample("alpha_noise", dist.HalfNormal(self.hyperparameters["alpha_noise_var"]))
                            alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[self.groups,:], alpha_noise))
                        else:
                            alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[self.groups,:], self.hyperparameters["alpha_var"]))

                alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
                alpha = torch.clamp(alpha, 0, 1)

        elif cluster is not None:
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster) / cluster))
            with pyro.plate("k1", k_fixed + k_denovo):
                with pyro.plate("g", cluster):  # G x K matrix
                    alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(alpha_prior_var))

        else:
            if self._noise_only:
                alpha = torch.zeros(self.n_samples, 1, dtype=torch.float64)
            else:
                with pyro.plate("k", self.k_fixed + self.k_denovo):  # columns
                    with pyro.plate("n", self.n_samples):  # rows
                        if self.enforce_sparsity:
                            alpha = pyro.sample("latent_exposure", dist.Exponential(self.hyperparameters["exp_rate"]))
                        else:
                            alpha = pyro.sample("latent_exposure", dist.HalfNormal(self.hyperparameters["alpha_var"]))

                alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
                alpha = torch.clamp(alpha, 0, 1)

        if self.stage == "random_noise":
            with pyro.plate("contexts3", self.contexts):  # columns
                    with pyro.plate("n3", n_samples):  # rows
                        epsilon = pyro.sample("latent_m", dist.HalfNormal(self.hyperparameters["eps_var"]))
        else:
            epsilon = None

        # Beta
        if self.k_denovo == 0:
            beta_denovo = None
        else:
            with pyro.plate("contexts", self.contexts):  # columns
                with pyro.plate("k_denovo", self.k_denovo):  # rows
                    beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(self.hyperparameters["beta_var"]))

            beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))  # normalize
            beta_denovo = torch.clamp(beta_denovo, 0, 1)

        beta = self._get_unique_beta(self.beta_fixed, beta_denovo)
        reg = self._regularizer(self.beta_fixed, beta_denovo, self.regularizer)
        self.reg = reg

        # Observations
        with pyro.plate("n2", n_samples):
            if cluster is not None:
                z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":enumer})
                if self.new_hier: 
                    alpha_noise = pyro.sample("alpha_noise", dist.HalfNormal(torch.ones(k_fixed+k_denovo)*alpha_noise_var).to_event(1))
                    alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[z], alpha_noise).to_event(1))
                else:
                    alpha  = pyro.sample("latent_exposure", dist.Normal(alpha_prior[z], alpha_var).to_event(1))

            a = torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)

            if self.stage == "random_noise":
                xx = a + epsilon
                lk =  dist.Poisson(xx).log_prob(self.x)
            else:
                lk =  dist.Poisson(a).log_prob(self.x)

            pyro.factor("loss", lk.sum() + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1]))


    def guide(self):

        n_samples = self.n_samples
        k_denovo = self.k_denovo
        groups = self.groups
        cluster = self.cluster
        init_params = self._initialize_params()

        # Alpha
        if groups is not None:
            if not self._noise_only:
                alpha_prior_param = pyro.param("alpha_t_param", init_params["alpha_t_param"], constraint=constraints.greater_than_eq(0))

                with pyro.plate("k1", self.k_fixed+self.k_denovo):
                    with pyro.plate("g", self.n_groups):
                        pyro.sample("alpha_t", dist.Delta(alpha_prior_param))

                    with pyro.plate("n", self.n_samples):  # rows
                        if self.new_hier:
                            alpha_noise_param = pyro.param("alpha_noise_param", init_params["alpha_noise"], constraint=constraints.greater_than(0.))
                            alpha_noise = pyro.sample("alpha_noise", dist.Delta(alpha_noise_param))
                            pyro.sample("latent_exposure", dist.Delta(alpha_prior_param[self.groups, :] + alpha_noise))
                        else:
                            alpha = pyro.param("alpha", alpha_prior_param[self.groups, :], constraint=constraints.greater_than_eq(0))
                            pyro.sample("latent_exposure", dist.Delta(alpha))

        elif cluster is not None:
            if not self._noise_only:
                pi_param = pyro.param("pi_param", init_params["pi_param"], constraint=constraints.simplex)
                pi = pyro.sample("pi", dist.Delta(pi_param).to_event(1))

                alpha_prior_param = pyro.param("alpha_t_param", init_params["alpha_t_param"], constraint=constraints.greater_than_eq(0.))
                alpha_noise_param = pyro.param("alpha_noise_param", init_params["alpha_noise"], constraint=constraints.greater_than(0.))

                with pyro.plate("k1", self.k_fixed+self.k_denovo):
                    with pyro.plate("g", self.cluster):
                        pyro.sample("alpha_t", dist.Delta(alpha_prior_param))

                with pyro.plate("n2", self.n_samples):
                    z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})

                    if self.new_hier:
                        alpha_noise = pyro.sample("alpha_noise", dist.Delta(alpha_noise_param).to_event(1))
                        pyro.sample("latent_exposure", dist.Delta(alpha_prior_param[z.long()] + alpha_noise).to_event(1))
                    else:
                        alpha = pyro.param("alpha", lambda: alpha_prior_param[z.long()], constraint=constraints.greater_than_eq(0))
                        pyro.sample("latent_exposure", dist.Delta(alpha).to_event(1))

        else:
            if not self._noise_only:
                alpha_mean = init_params["alpha_mean"]

                with pyro.plate("k", self.k_fixed + k_denovo):
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


    # def _guide_alpha_hier(self, init_params):
    #     if not self._noise_only:
    #         alpha_prior_param = pyro.param("alpha_t_param", init_params["alpha_t_param"], constraint=constraints.greater_than_eq(0))

    #         with pyro.plate("k1", self.k_fixed+self.k_denovo):
    #             with pyro.plate("g", self.n_groups):
    #                 pyro.sample("alpha_t", dist.Delta(alpha_prior_param))

    #             with pyro.plate("n", self.n_samples):  # rows
    #                 if self.new_hier:
    #                     alpha_noise_param = pyro.param("alpha_noise_param", init_params["alpha_noise"], constraint=constraints.greater_than(0.))
    #                     alpha_noise = pyro.sample("alpha_noise", dist.Delta(alpha_noise_param))
    #                     pyro.sample("latent_exposure", dist.Delta(alpha_prior_param[self.groups, :] + alpha_noise))
    #                 else:
    #                     alpha = pyro.param("alpha", alpha_prior_param[self.groups, :], constraint=constraints.greater_than_eq(0))
    #                     pyro.sample("latent_exposure", dist.Delta(alpha))


    # def _guide_alpha_clustering(self, init_params):
    #     if not self._noise_only:
    #         pi_param = pyro.param("pi_param", init_params["pi_param"], constraint=constraints.simplex)
    #         pi = pyro.sample("pi", dist.Delta(pi_param).to_event(1))

    #         alpha_prior_param = pyro.param("alpha_t_param", init_params["alpha_t_param"], constraint=constraints.greater_than_eq(0.))
    #         alpha_noise_param = pyro.param("alpha_noise_param", init_params["alpha_noise"], constraint=constraints.greater_than(0.))

    #         with pyro.plate("k1", self.k_fixed+self.k_denovo):
    #             with pyro.plate("g", self.cluster):
    #                 pyro.sample("alpha_t", dist.Delta(alpha_prior_param))

    #         with pyro.plate("n2", self.n_samples):
    #             z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})

    #             if self.new_hier:
    #                 alpha_noise = pyro.sample("alpha_noise", dist.Delta(alpha_noise_param).to_event(1))
    #                 pyro.sample("latent_exposure", dist.Delta(alpha_prior_param[z.long()] + alpha_noise).to_event(1))
    #             else:
    #                 alpha = pyro.param("alpha", lambda: alpha_prior_param[z.long()], constraint=constraints.greater_than_eq(0))
    #                 pyro.sample("latent_exposure", dist.Delta(alpha).to_event(1))


    # def _guide_alpha_flat(self, init_params):
    #     if not self._noise_only:
    #         alpha_mean = init_params["alpha_mean"]

    #         with pyro.plate("k", self.k_fixed+self.k_denovo):
    #             with pyro.plate("n", self.n_samples):
    #                 if self.enforce_sparsity:
    #                     alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than(0.0))
    #                 else:
    #                     alpha = pyro.param("alpha", alpha_mean, constraint=constraints.greater_than_eq(0.0))
    #                 pyro.sample("latent_exposure", dist.Delta(alpha))


    def _initialize_params_hier(self):
        groups_true = torch.tensor(self.groups)
        steps_true = self.n_steps
        hyperpars_true = self.hyperparameters
        new_hier_true = self.new_hier

        self.groups = None
        self.n_steps = 50
        self.initializ_seed = False
        self.hyperparameters = self._hyperpars_default
        self.new_hier = False

        params = dict()

        # print("Initializing parameters with non-hierarchical fit")
        self._fit(set_attributes=False)

        if self.cluster is not None: raise NotImplementedError

        alpha = self._get_param("alpha", normalize=True, to_cpu=False)
        alpha_t_param = torch.zeros((groups_true.unique().numel(), self.k_fixed+self.k_denovo), dtype=torch.float64)

        for gid in groups_true.unique().tolist(): alpha_t_param[gid] = torch.mean(alpha[groups_true == gid, :], dim=0)
        params["alpha_t_param"] = alpha_t_param

        params["alpha_noise"] = torch.ones(self.n_samples, self.k_denovo+self.k_fixed) * \
            self.hyperparameters["alpha_noise_var"]

        if self.k_denovo > 0: params["beta_dn_param"] = self._get_param("beta_denovo", to_cpu=False, normalize=True)

        params["epsilon_var"] = torch.ones(self.n_samples, self.contexts, dtype=torch.float64)

        pyro.get_param_store().clear()

        self.groups = groups_true
        self.n_steps = steps_true
        self.hyperparameters = hyperpars_true
        self.new_hier = new_hier_true

        return params


    def _initialize_params_nonhier(self):
        params = dict()

        alpha_var = self.hyperparameters["alpha_var"]
        alpha_prior_var = self.hyperparameters["alpha_prior_var"]
        exp_rate = self.hyperparameters["exp_rate"]
        beta_var = self.hyperparameters["beta_var"]
        eps_var = self.hyperparameters["eps_var"]

        params["alpha_noise"] = torch.ones(self.n_samples, self.k_denovo+self.k_fixed, dtype=torch.float64) * \
            self.hyperparameters["alpha_noise_var"]

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
        loss = 0

        if self.reg_weight == 0:
            return loss

        if self.regul_compare is not None:
            beta_fixed = self.regul_compare

        if beta_fixed is None or beta_denovo is None or self._noise_only:
            return loss

        beta_fixed[beta_fixed==0] = 1e-07

        if reg_type == "cosine":
            for fixed in beta_fixed:
                for denovo in beta_denovo:
                    loss += torch.log((1 - F.cosine_similarity(fixed, denovo, dim = -1)))

            if self.regul_denovo and self.k_denovo > 1:
                for dn1 in range(self.k_denovo):
                    for dn2 in range(dn1, self.k_denovo):
                        if dn1 == dn2: continue
                        loss += torch.log((1 - F.cosine_similarity(beta_denovo[dn1,], beta_denovo[dn2,], dim = -1)))

        elif reg_type == "KL":
            for fixed in beta_fixed:
                for denovo in beta_denovo:
                    loss += torch.log(F.kl_div(torch.log(fixed), torch.log(denovo), log_target = True, reduction="batchmean"))

            if self.regul_denovo and self.k_denovo > 1:
                for dn1 in range(self.k_denovo):
                    for dn2 in range(dn1, self.k_denovo):
                        if dn1 == dn2: continue
                        loss += torch.log(F.kl_div(torch.log(beta_denovo[dn1,]), torch.log(beta_denovo[dn2,]), log_target = True, reduction="batchmean"))

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
        pyro.set_rng_seed(seed)
        pyro.get_param_store().clear()

        svi = SVI(self.model, self.guide, optim, elbo)
        loss = svi.step()
        self.init_params = None

        return np.round(loss, 3), seed


    def _fit(self, set_attributes=True):
        pyro.clear_param_store()  # always clear the store before the inference

        self.x = self._to_gpu(self.x)
        self.beta_fixed = self._to_gpu(self.beta_fixed)
        self.regul_compare = self._to_gpu(self.regul_compare)

        if self.CUDA and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type(t=torch.FloatTensor)

        if self.cluster is not None:
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

        # print("Running with seed {}\n".format(self.seed))
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

            if alpha is None:
                print(_, self.cluster)

            likelihoods.append(self._likelihood(self.x, alpha, self.beta_fixed, beta_denovo, eps_var))

            if self.store_parameters:
                train_params.append(self.get_param_dict())

            # convergence test 
            if len(losses) >= min_steps and len(losses) % min_steps == 0 and convergence(x=losses[-min_steps:], alpha=0.05):
                break

        if set_attributes is False:
            return

        self.x = self._to_cpu(self.x)
        self.beta_fixed = self._to_cpu(self.beta_fixed)
        self.regul_compare = self._to_cpu(self.regul_compare)

        self.train_params = train_params
        self.losses = losses
        self.likelihoods = likelihoods
        self.regs = regs
        self._set_params()
        self._set_bic()
        self.likelihood = self._likelihood(self.x, self.alpha, self.beta_fixed, self.beta_denovo, self.eps_var)

        reg = self._regularizer(self.beta_fixed, self.beta_denovo, reg_type=self.regularizer)
        self.reg_likelihood = self.likelihood + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1])
        try:
            self.reg_likelihood = self.reg_likelihood.item()
        except: return


    def _set_params(self):
        self._set_alpha()
        self._set_beta_denovo()
        self._set_epsilon()
        self._set_clusters()
        self.params = self.get_param_dict()

        if isinstance(self.groups, torch.Tensor):
            self.groups = self.groups.tolist()


    def _to_cpu(self, param, move=True):
        if param is None: return None
        if move and self.CUDA and torch.cuda.is_available():
            return param.cpu()
        return param


    def _to_gpu(self, param, move=True):
        if param is None: return None
        if move and self.CUDA and torch.cuda.is_available():
            return param.cuda()
        return param


    def _get_alpha_hier(self, grps):
        alpha_prior = pyro.param("alpha_t_param")
        alpha = torch.zeros((self.n_samples, self.k_denovo + self.k_fixed))
        # for gid in self.groups.unique().tolist(): alpha_prior[gid] = torch.mean(, dim=0)
        alpha = alpha_prior[grps]
        try:
            alpha_noise = self._get_param("alpha_noise_param", normalize=False, to_cpu=False)
        except:
            alpha_noise = torch.zeros_like(alpha)
        # for gid in self.groups.unique().tolist():
        #     alpha[self.groups.unique() == gid, :]
        return alpha + alpha_noise


    def _get_param(self, param_name, normalize=False, to_cpu=True):
        try:
            if param_name == "beta_fixed": 
                par = self.beta_fixed

            elif param_name == "alpha":
                if self._noise_only:
                    return self._to_gpu(torch.zeros(self.n_samples, 1, dtype=torch.float64), move=not to_cpu)
                if self.new_hier:
                    if self.groups is not None:
                        par = self._get_alpha_hier(grps=self.groups)
                    elif self.cluster is not None:
                        par = self._get_alpha_hier(grps=self._compute_posterior_probs(to_cpu=to_cpu))
                else:
                    par = pyro.param("alpha")

            else:
                par = pyro.param(param_name)

            # if to_cpu and self.CUDA and torch.cuda.is_available(): par = par.cpu()
            par = self._to_cpu(par, move=to_cpu)

            try:
                par = par.clone().detach()
            finally:
                if normalize:
                    par = par / (torch.sum(par, 1).unsqueeze(-1))
            return par
        except:
            return None


    def _set_alpha(self):
        self.alpha = self._get_param("alpha", normalize=True)
        self.alpha_unn = self._get_param("alpha", normalize=False)
        try:
            self.alpha_prior = self._get_param("alpha_t_param", normalize=True)
            self.alpha_prior_unn = self._get_param("alpha_t_param", normalize=False)
        except:
            self.alpha_prior = None
            self.alpha_prior_unn = None

        try: self.alpha_noise = self._get_param("alpha_noise_param")
        except: self.alpha_noise = None


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


    def _set_clusters(self, to_cpu=True):
        if self.cluster is None:
            self.pi = None
            return

        self.pi = self._get_param("pi_param", normalize=False, to_cpu=to_cpu)

        self.groups = self._compute_posterior_probs(to_cpu=to_cpu)


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
        params["alpha_prior"] = self._get_param("alpha_t_param", normalize=False)
        params["alpha_noise"] = self._get_param("alpha_noise_param", normalize=False)

        params["beta_d"] =  self._get_param("beta_denovo", normalize=True)
        params["beta_f"] = self._get_param("beta_fixed")

        params["pi"] = self._get_param("pi_param", normalize=False)

        if self.stage=="random_noise":
            params["lambda_epsilon"] = self._get_param("lambda_epsilon", normalize=False)
        else:
            params["lambda_epsilon"] = None

        return params


    def _compute_posterior_probs(self, to_cpu=True):
        # params = self.get_param_dict()
        pi = self._get_param("pi_param", to_cpu=to_cpu)
        alpha_prior = self._get_param("alpha_t_param", to_cpu=to_cpu, normalize=False)  # G x K
        if self.new_hier:
            alpha_noise = self._get_param("alpha_noise_param", to_cpu=to_cpu, normalize=False) 
        else:
            alpha_noise = self._to_cpu(torch.zeros(self.n_samples, self.k_denovo + self.k_fixed), move=to_cpu)
        beta_denovo = self._get_param("beta_denovo", to_cpu=to_cpu)
        M = torch.tensor(self.x, dtype=torch.double)
        cluster = self.cluster
        n_samples = self.n_samples

        beta = self._get_unique_beta(self.beta_fixed, beta_denovo)

        z = torch.zeros(n_samples)
        n_muts = torch.sum(M, axis=1).unsqueeze(1)
        ll_k = torch.zeros((cluster, self.n_samples))  # N x C
        for k in range(cluster):
            alpha_k = alpha_prior[k,:] + alpha_noise
            alpha_k = alpha_k / (torch.sum(alpha_k, 1).unsqueeze(-1))
            rate = torch.matmul( alpha_k * n_muts, beta )
            ll_k[k,:] = torch.log(pi[k]) + pyro.distributions.Poisson( rate ).log_prob(M).sum(axis=1)  # N dim vector, summed over contexts

        ll = self._logsumexp(ll_k)

        probs = torch.exp(ll_k - ll)
        z = torch.argmax(probs, dim=0)
        return z.long()


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
            k = self.k_denovo * self.contexts # beta denovo
            print("line 882", k)

        if self.cluster is not None:
            k += self.params["pi"].numel()  # mixing proportions
            print("line 886", k)

        if self.eps_var is not None:
            k += self.eps_var.shape[0] * self.eps_var.shape[1]  # random noise
            print("line 890", k)

        if self.new_hier:
            k += self.params["alpha_noise"].numel() + self.params["alpha_prior"].numel()  # alpha if noise is learned
            print("line 894", k)
        else:
            if self.params["alpha_prior"] is not None:
                k += self.params["alpha_prior"].numel()
                print("line 898", k)
            k += self.params["alpha"].numel()  # alpha if no noise is learned
            print("line 900", k)

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

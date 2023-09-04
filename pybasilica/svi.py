import numpy as np
import pandas as pd
import torch
import pyro
from pyro.infer import SVI,Trace_ELBO, JitTrace_ELBO, TraceEnum_ELBO
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from sklearn.cluster import KMeans
import pyro.distributions.constraints as constraints
import pyro.distributions as dist
import torch.nn.functional as F

from tqdm import trange
from logging import warning
from collections import defaultdict

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
        enumer = "parallel",
        cluster = None,
        hyperparameters = {"alpha_sigma":0.1, "alpha_p_sigma":1., "alpha_p_conc0":0.6, "alpha_p_conc1":0.6, "alpha_rate":5, 
                           "beta_d_sigma":1, "eps_sigma":10, "alpha_noise_sigma":0.01, "pi_conc0":0.6, "scale_factor":1000},
        groups = None,
        dirichlet_prior = False,
        beta_fixed = None,
        compile_model = True,
        CUDA = False,
        enforce_sparsity = False,
        store_parameters = False,
        regularizer = "cosine",
        reg_weight = 0.,
        reg_bic = True, 
        stage = "", 
        regul_compare = None,
        seed = 10,
        initializ_seed = False,
        initializ_pars_fit = False,
        new_hier = False,
        regul_denovo = True,
        regul_fixed = True,
        nonparam = False,
        initial_fit = None
        ):

        self._hyperpars_default = {"alpha_sigma":1., "alpha_p_sigma":1., "alpha_p_conc0":0.6, "alpha_p_conc1":0.6, "alpha_rate":5, 
                                   "beta_d_sigma":1, "eps_sigma":10, "alpha_noise_sigma":0.01, "pi_conc0":0.6, "scale_factor":1000}
        self.regul_denovo = regul_denovo
        self.regul_fixed = regul_fixed
        self.new_hier = new_hier
        self.initial_fit = initial_fit
        self.dirichlet_prior = dirichlet_prior

        self._set_data_catalogue(x)
        self._set_fit_settings(enforce_sparsity, lr, n_steps, compile_model, CUDA, regularizer, reg_weight, reg_bic, \
                               store_parameters, stage, initializ_seed, initializ_pars_fit, seed, nonparam)

        self._set_beta_fixed(beta_fixed)
        self._set_k_denovo(k_denovo)

        self._set_hyperparams(enumer, cluster, groups, hyperparameters)

        self._fix_zero_denovo_null_reference()
        self._set_external_catalogue(regul_compare)

        self._init_seed = None
        self.K = self.k_denovo + self.k_fixed


    def _set_fit_settings(self, enforce_sparsity, lr, n_steps, compile_model, CUDA, \
                          regularizer, reg_weight, reg_bic, store_parameters, stage,
                          initializ_seed, initializ_pars_fit, seed, nonparam):
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
        self.nonparam = nonparam

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

            self.beta_fixed = self._norm_and_clamp(self.beta_fixed)


    def _set_external_catalogue(self, regul_compare):
        try:
            self.regul_compare = torch.tensor(regul_compare.values, dtype=torch.float64)
            self.regul_compare = self._to_gpu(self.regul_compare)
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
        
        if self.cluster is not None:
            self.n_groups = self.cluster


    def _mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


    def model(self):

        n_samples, k_fixed, k_denovo = self.n_samples, self.k_fixed, self.k_denovo
        groups, cluster = self.groups, self.cluster  # number of clusters or None

        alpha_sigma, alpha_rate = self.hyperparameters["alpha_sigma"], self.hyperparameters["alpha_rate"]
        alpha_p_sigma, alpha_noise_sigma = self.hyperparameters["alpha_p_sigma"], self.hyperparameters["alpha_noise_sigma"]
        alpha_conc0, alpha_conc1 = self.hyperparameters["alpha_p_conc0"], self.hyperparameters["alpha_p_conc1"]
        beta_d_sigma = self.hyperparameters["beta_d_sigma"]
        pi_conc0 = self.hyperparameters["pi_conc0"]

        # Alpha
        if self._noise_only: alpha = torch.zeros(self.n_samples, 1, dtype=torch.float64)
        if groups is not None:
            if not self._noise_only:
                with pyro.plate("k1", self.K):
                    with pyro.plate("g", self.n_groups):
                        # G x K matrix
                        if self.enforce_sparsity:
                            alpha_prior = pyro.sample("alpha_t", dist.Exponential(alpha_p_sigma))
                        else:
                            alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(alpha_p_sigma))

                # sample from the alpha prior
                r_noise = dist.HalfNormal(torch.ones(n_samples, k_fixed + k_denovo) * alpha_noise_sigma).sample()
                with pyro.plate("k", self.K):  # columns
                    with pyro.plate("n", self.n_samples):  # rows
                        if self.new_hier: 
                            alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[self.groups,:], r_noise))
                        else:
                            alpha = pyro.sample("latent_exposure", dist.Normal(alpha_prior[self.groups,:], alpha_sigma))

                alpha = self._norm_and_clamp(alpha)

        elif cluster is not None:
            if self.nonparam:
                with pyro.plate("beta_plate", cluster-1):
                    pi_beta = pyro.sample("beta", dist.Beta(1, pi_conc0))
                    pi = self._mix_weights(pi_beta)
            else:
                pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster, dtype=torch.float64)))

            # if not self.spike_n_slab:
            with pyro.plate("k1", self.K):
                with pyro.plate("g", cluster):  # G x K matrix
                    if self.enforce_sparsity:
                        # alpha_prior = pyro.sample("alpha_t", dist.Exponential(alpha_prior_var))
                        alpha_prior = pyro.sample("alpha_t", dist.Beta(alpha_conc1, alpha_conc0))
                    else:
                        alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(torch.tensor(alpha_p_sigma, dtype=torch.float64)))

            if self.dirichlet_prior: alpha_prior = alpha_prior * self.hyperparameters["scale_factor"]  # Dirichlet
            else: 
                alpha_prior = self._norm_and_clamp(alpha_prior)  # Normal or Cauchy

                q05 = alpha_prior - alpha_sigma # * alpha_prior
                q95 = alpha_prior + alpha_sigma # * alpha_prior

                self.alpha_sigma_corr = (q95 - q05) / ( dist.Normal(alpha_prior, 1).icdf(torch.tensor(1-0.05/2)) -\
                                                       dist.Normal(alpha_prior, 1).icdf(torch.tensor(0.05/2)) )  # good clustering

        else:
            if not self._noise_only:
                with pyro.plate("k", self.K):  # columns
                    with pyro.plate("n", self.n_samples):  # rows
                        if self.enforce_sparsity:
                            alpha = pyro.sample("latent_exposure", dist.Exponential(alpha_rate))
                        else:
                            alpha = pyro.sample("latent_exposure", dist.HalfNormal(torch.tensor(alpha_sigma, dtype=torch.float64)))
                            # alpha = pyro.sample("latent_exposure", dist.HalfCauchy(torch.tensor(alpha_sigma, dtype=torch.float64)))

                alpha = self._norm_and_clamp(alpha)

        epsilon = None
        if self.stage == "random_noise":
            with pyro.plate("contexts3", self.contexts):  # columns
                    with pyro.plate("n3", n_samples):  # rows
                        epsilon = pyro.sample("latent_m", dist.HalfNormal(self.hyperparameters["eps_sigma"]))

        # Beta
        beta_denovo = None
        if self.k_denovo > 0:
            if cluster is not None and self.new_hier:
                beta_denovo = self._get_param("beta_denovo")
            else:
                with pyro.plate("contexts", self.contexts):  # columns
                    with pyro.plate("k_denovo", self.k_denovo):  # rows
                        beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(beta_d_sigma))

            beta_denovo = self._norm_and_clamp(beta_denovo)

        beta = self._get_unique_beta(self.beta_fixed, beta_denovo)
        reg = self._regularizer(self.beta_fixed, beta_denovo, self.regularizer)
        self.reg = reg

        # Observations
        r_noise = dist.HalfNormal(torch.ones(n_samples, k_fixed + k_denovo) * alpha_noise_sigma).sample()
        with pyro.plate("n2", n_samples):
            if cluster is not None:
                z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})

                if self.new_hier: 
                    alpha = self._get_param("alpha")
                else:
                    if self.dirichlet_prior: alpha = pyro.sample("latent_exposure", dist.Dirichlet(alpha_prior[z]))
                    else: alpha  = pyro.sample("latent_exposure", dist.Normal(alpha_prior[z], self.alpha_sigma_corr[z]).to_event(1))

                alpha = self._norm_and_clamp(alpha)

            a = torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)
            if self.stage == "random_noise": a = a + epsilon

            pyro.sample("obs", dist.Poisson(a).to_event(1), obs=self.x)

            if self.reg_weight > 0:
                # lk =  dist.Poisson(a).log_prob(self.x)
                # lk_sum = lk.sum()
                # pyro.factor("loss", lk_sum + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1]))
                pyro.factor("loss", self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1]))


    def guide(self):

        n_samples, k_denovo = self.n_samples, self.k_denovo
        groups, cluster = self.groups, self.cluster
        init_params = self._initialize_params()

        # Alpha
        if groups is not None:
            if not self._noise_only:
                alpha_prior_param = pyro.param("alpha_prior_param", init_params["alpha_prior_param"], constraint=constraints.greater_than_eq(0))

                with pyro.plate("k1", self.K):
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
                pi_param = pyro.param("pi_param", lambda: init_params["pi_param"], constraint=constraints.simplex)

                if self.nonparam:
                    pi_conc0 = pyro.param("pi_conc0", lambda: dist.Uniform(0, 2).sample([cluster-1]), constraint=constraints.greater_than_eq(torch.finfo().tiny))
                    with pyro.plate("beta_plate", cluster-1):
                        pyro.sample("beta", dist.Beta(torch.ones(cluster-1, dtype=torch.float64), pi_conc0))

                else:
                    pyro.sample("pi", dist.Delta(pi_param).to_event(1))

                if self.enforce_sparsity:
                    alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.interval(0., 1.))
                else:
                    alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.greater_than_eq(0.))

                with pyro.plate("k1", self.K):
                    with pyro.plate("g", self.cluster):
                        pyro.sample("alpha_t", dist.Delta(alpha_prior_param))

                if self.dirichlet_prior: alpha_prior_param = alpha_prior_param * self.hyperparameters["scale_factor"]  # Dirichlet
                else: alpha_prior_param = self._norm_and_clamp(alpha_prior_param)  # Normal or Cauchy

                with pyro.plate("n2", n_samples):
                    z = pyro.sample("latent_class", dist.Categorical(pi_param), infer={"enumerate":self.enumer})

                    if not self.new_hier:
                        if self.dirichlet_prior: alpha = pyro.param("alpha", lambda: alpha_prior_param[z.long()], constraint=constraints.simplex)  # Dirichlet
                        else: alpha = pyro.param("alpha", lambda: alpha_prior_param[z.long()], constraint=constraints.greater_than_eq(0))  # Normal or Cauchy

                        pyro.sample("latent_exposure", dist.Delta(alpha).to_event(1))

        else:
            if not self._noise_only:
                alpha_param = init_params["alpha"]

                with pyro.plate("k", self.k_fixed + k_denovo):
                    with pyro.plate("n", n_samples):
                        if self.enforce_sparsity:
                            alpha = pyro.param("alpha", alpha_param, constraint=constraints.greater_than(0.0))
                        else:
                            alpha = pyro.param("alpha", alpha_param, constraint=constraints.greater_than_eq(0.0))
                        pyro.sample("latent_exposure", dist.Delta(alpha))

        # Epsilon 
        if self.stage == "random_noise":
            eps_sigma = pyro.param("lambda_epsilon", init_params["epsilon_var"], constraint=constraints.positive)

            with pyro.plate("contexts3", self.contexts):
                with pyro.plate("n3", n_samples):
                    pyro.sample("latent_m", dist.HalfNormal(eps_sigma))

        # Beta
        if k_denovo > 0:
            if cluster is None or not self.new_hier:
                beta_param = pyro.param("beta_denovo", lambda: init_params["beta_dn_param"], constraint=constraints.greater_than_eq(0.0))
                with pyro.plate("contexts", self.contexts):
                    with pyro.plate("k_denovo", k_denovo):
                        pyro.sample("latent_signatures", dist.Delta(beta_param))


    def check_input_kmeans(self, counts):
        '''
        Function to check the inputs of the Kmeans. There might be a problem when multiple observations 
        are equal since the Kmeans will keep only a unique copy of each and the others will not be initialized.
        '''

        tmp, indexes, count = np.unique(counts, axis=0, return_counts=True, return_index=True)
        repeated_groups = tmp[count > 1].tolist()

        unq = np.array([counts[index] for index in sorted(indexes)])

        removed_idx = {}
        for i, repeated_group in enumerate(repeated_groups):
            rpt_idxs = np.argwhere(np.all(counts == repeated_group, axis=1)).flatten()
            removed = rpt_idxs[1:]
            for rm in removed:
                removed_idx[rm] = rpt_idxs[0]

        return removed_idx, unq


    def run_kmeans(self, X, G, seed):
        X = self._to_cpu(X, move=True)
        try:
            km = KMeans(n_clusters=G, random_state=seed).fit(X.numpy())
        
        except:
            removed_idx, data_unq = self.check_input_kmeans(X.numpy())

            km = KMeans(n_clusters=G, random_state=seed).fit(data_unq)
            assert km.n_iter_ < km.max_iter

            clusters = km.labels_
            for rm in sorted(removed_idx.keys()):
                # insert 0 elements to restore the original number of obs
                clusters = np.insert(clusters, rm, 0, 0)

            for rm in removed_idx.keys():
                # insert in the repeated elements the correct cluster
                rpt = removed_idx[rm]  # the index of the kept row
                clusters[rm] = clusters[rpt]

        return km


    def _initialize_weights(self, X, G):
        '''
        Function to run KMeans on the counts.
        Returns the vector of mixing proportions and the clustering assignments.
        '''
        km = self.run_kmeans(X=X, G=G, seed=15)
        self._init_km = km

        return km


    def _run_initial_fit(self):

        cluster_true = self.cluster
        enforce_sparsity_true = self.enforce_sparsity
        hyperpars_true = self.hyperparameters
        new_hier_true = self.new_hier
        x_true, n_samples_true = self.x, self.n_samples

        self.cluster = None
        self.initializ_seed = False
        self.enforce_sparsity = False
        self.hyperparameters = self._hyperpars_default
        self.new_hier = False

        pyro.get_param_store().clear()
        self._fit(set_attributes=False)
        
        alpha = self._get_param("alpha", normalize=False, to_cpu=False)
        beta_dn = self._get_param("beta_denovo", normalize=False, to_cpu=False) 
        pyro.get_param_store().clear()

        self.x, self.n_samples = x_true, n_samples_true
        self.cluster = cluster_true
        self.enforce_sparsity = enforce_sparsity_true
        self.hyperparameters = hyperpars_true
        self.new_hier = new_hier_true

        return alpha, beta_dn


    def _initialize_params_clustering(self):
        pi = alpha_noise = alpha_prior = epsilon = beta_dn = None
        eps_sigma, beta_d_sigma = self.hyperparameters["eps_sigma"], self.hyperparameters["beta_d_sigma"]

        if self.initial_fit is None:
            alpha, beta_dn = self._run_initial_fit()
        else:
            alpha = self._to_gpu(self.initial_fit.alpha_unn, move=True)
            if self.new_hier: beta_dn = self._to_gpu(self.initial_fit.beta_denovo, move=True)

        if not self.new_hier: beta_dn = dist.HalfNormal(torch.ones(self.k_denovo, self.contexts, dtype=torch.float64) * beta_d_sigma).sample()

        km = self._initialize_weights(X=alpha.clone(), G=self.cluster)
        pi_km = torch.tensor([(np.where(km.labels_ == k)[0].shape[0]) / self.n_samples for k in range(km.n_clusters)])
        groups_kmeans = torch.tensor(km.labels_)
        alpha_prior_km = torch.tensor(km.cluster_centers_)

        pi = self._to_gpu(pi_km, move=True)
        alpha_prior = self._to_gpu(alpha_prior_km, move=True)

        if self.stage == "random_noise": epsilon = torch.ones(self.n_samples, self.contexts, dtype=torch.float64) * eps_sigma

        params = self._create_init_params_dict(pi=torch.tensor(pi, dtype=torch.float64), alpha_noise=alpha_noise, 
                                               alpha_prior=torch.tensor(alpha_prior, dtype=torch.float64), 
                                               alpha=alpha, epsilon=epsilon, beta_dn=beta_dn)
        params["init_clusters"] = groups_kmeans
        return params


    def _initialize_params_nonhier(self):
        pi = alpha_noise = alpha_prior = alpha = epsilon = beta_dn = None

        alpha_sigma, alpha_rate = self.hyperparameters["alpha_sigma"], self.hyperparameters["alpha_rate"]
        alpha_p_sigma, alpha_noise_sigma = self.hyperparameters["alpha_p_sigma"], self.hyperparameters["alpha_noise_sigma"]
        alpha_p_conc0, alpha_p_conc1 = self.hyperparameters["alpha_p_conc0"], self.hyperparameters["alpha_p_conc1"]
        beta_d_sigma = self.hyperparameters["beta_d_sigma"]
        eps_sigma = self.hyperparameters["eps_sigma"]

        alpha_noise = dist.Normal(0, torch.ones(self.n_samples, self.K, dtype=torch.float64) * alpha_noise_sigma).sample()
        if self.cluster is not None:
            ones_tmp = torch.ones(self.cluster, self.K, dtype=torch.float64)
            pi = torch.ones(self.cluster, dtype=torch.float64)
            # pi = dist.Dirichlet(1 / self.cluster * torch.ones(self.cluster, dtype=torch.float64)).sample()
        else:
            ones_tmp = torch.ones(self.n_samples, self.K, dtype=torch.float64)

        if self.enforce_sparsity:
            # alpha_prior = dist.Exponential(ones_tmp * alpha_p_sigma).sample()
            # alpha_prior = dist.Beta(ones_tmp * alpha_p_conc1, ones_tmp * alpha_p_conc0).sample()
            # alpha = dist.Exponential(ones_tmp * alpha_rate).sample()
            alpha = dist.Beta(ones_tmp * alpha_p_conc1, ones_tmp * alpha_p_conc0).sample()
        else:
            # alpha_prior = dist.HalfNormal(ones_tmp * alpha_p_sigma).sample()
            alpha = dist.HalfNormal(ones_tmp * alpha_sigma).sample()

        if self.stage == "random_noise":
            epsilon = dist.HalfNormal(torch.ones(self.n_samples, self.contexts, dtype=torch.float64) * eps_sigma).sample()

        if self.k_denovo > 0:
            beta_dn = dist.HalfNormal(torch.ones(self.k_denovo, self.contexts, dtype=torch.float64) * beta_d_sigma).sample()

        params = self._create_init_params_dict(pi=pi, alpha_noise=alpha_noise, alpha_prior=alpha_prior, \
                                               alpha=alpha, epsilon=epsilon, beta_dn=beta_dn)

        return params


    def _create_init_params_dict(self, pi=None, alpha_noise=None, alpha_prior=None, alpha=None, epsilon=None, beta_dn=None):
        params = dict()

        params["pi_param"] = pi
        params["alpha_noise_param"] = alpha_noise
        params["alpha_prior_param"] = alpha_prior
        params["alpha"] = alpha
        params["epsilon_var"] = epsilon
        params["beta_dn_param"] = beta_dn

        return params


    def _initialize_params(self):
        if self.init_params is None:
            # if self.groups is not None and self._initializ_params_with_fit: 
            #     self.init_params = self._initialize_params_hier()

            if self.cluster is not None:
                self.init_params = self._initialize_params_clustering()
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
            if self.regul_fixed:
                for fixed in beta_fixed:
                    for denovo in beta_denovo:
                        loss += torch.log((1 - F.cosine_similarity(fixed, denovo, dim = -1)))

            if self.regul_denovo and self.k_denovo > 1:
                for dn1 in range(self.k_denovo):
                    for dn2 in range(dn1, self.k_denovo):
                        if dn1 == dn2: continue
                        loss += torch.log((1 - F.cosine_similarity(beta_denovo[dn1,], beta_denovo[dn2,], dim = -1)))

        elif reg_type == "KL":
            if self.regul_fixed:
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


    def _get_param(self, param_name, normalize=False, to_cpu=True, convert=False):
        try:
            if param_name == "beta_fixed": 
                par = self.beta_fixed

            elif param_name == "beta_denovo" and (self.cluster is not None and self.new_hier):
                par = self.init_params["beta_dn_param"]

            elif param_name == "alpha":
                if self._noise_only: return self._to_gpu(torch.zeros(self.n_samples, 1, dtype=torch.float64), move=not to_cpu)
                elif self.new_hier: 
                    # par = self._get_alpha_hier(grps=self._get_groups(to_cpu=to_cpu), normalize=normalize)
                    par = self.init_params["alpha"]
                else: par = pyro.param("alpha")

            else:
                par = pyro.param(param_name)

            par = self._to_cpu(par, move=to_cpu)

            if isinstance(par, torch.Tensor):
                par = par.clone().detach()

            if normalize: 
                par = self._norm_and_clamp(par)

            if par is not None and convert:
                par = self._to_cpu(par, move=True)
                par = np.array(par)

            return par

        except Exception as e:
            return None


    def _get_groups(self, to_cpu):
        if self.groups is not None: 
            return self.groups

        if self.cluster is not None: 
            return self._compute_posterior_probs(to_cpu=to_cpu)[0]

        return None


    def _get_alpha_hier(self, grps, normalize):
        alpha_prior = self._get_param("alpha_prior_param", normalize=False, to_cpu=False)
        alpha_noise = self._get_param("alpha_noise_param", normalize=False, to_cpu=False)

        if alpha_noise is None:
            alpha_noise = self._to_cpu(dist.Normal(0, torch.ones(self.K, dtype=torch.float64) * \
                                                   self.hyperparameters["alpha_noise_sigma"]).sample())

        alpha = alpha_prior[grps] + alpha_noise
        if normalize: alpha = self._norm_and_clamp(alpha)

        return alpha


    def get_param_dict(self, convert=False):
        params = dict()
        params["alpha"] = self._get_param("alpha", normalize=True, convert=convert)
        params["alpha_prior"] = self._get_param("alpha_prior_param", normalize=False, convert=convert)
        params["alpha_noise"] = self._get_param("alpha_noise_param", normalize=False, convert=convert)

        params["beta_d"] =  self._get_param("beta_denovo", normalize=True, convert=convert)
        params["beta_f"] = self._get_param("beta_fixed", convert=convert)

        params["pi"] = self._get_param("pi_param", normalize=False, convert=convert)

        params["pi_conc0"] = self._get_param("pi_conc0", normalize=False, convert=convert)

        params["lambda_epsilon"] = self._get_param("lambda_epsilon", normalize=False, convert=convert)

        return params


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

        if self.cluster is not None: elbo = TraceEnum_ELBO()
        elif self.compile_model and not self.CUDA: elbo = JitTrace_ELBO()
        else: elbo = Trace_ELBO()

        train_params = []
        # learning global parameters
        adam_params = {"lr": self.lr}
        optimizer = Adam(adam_params)

        if self.initializ_seed:
            _, self.seed = min([self._initialize_seed(optimizer, elbo, seed) for seed in range(50)], key = lambda x: x[0])

        pyro.set_rng_seed(self.seed)
        pyro.get_param_store().clear()

        self._initialize_params()

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)
        loss = svi.step()

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        losses = []
        regs = []
        likelihoods = []
        for i in range(self.n_steps):   # inference - do gradient steps
            # self._step = i

            loss = svi.step()
            losses.append(loss)
            regs.append(self.reg)

            alpha = self._get_param("alpha", normalize=True, to_cpu=False)
            eps_sigma = self._get_param("eps_sigma", normalize=False, to_cpu=False)
            beta_denovo = self._get_param("beta_denovo", normalize=True, to_cpu=False)

            if alpha is None: print("Alpha is None at step", i, self.cluster)
            # if beta_denovo is None: print("Beta_denovo is None at step", i, self.cluster)

            likelihoods.append(self._likelihood(self.x, alpha, self.beta_fixed, beta_denovo, eps_sigma))

            if self.store_parameters: train_params.append(self.get_param_dict(convert=True))

            # convergence test 
            # if len(losses) >= 100 and len(losses) % 100 == 0 and convergence(x=losses[-100:], alpha=0.05):
            #     break

        if set_attributes is False: return

        self.x = self._to_cpu(self.x)
        self.beta_fixed = self._to_cpu(self.beta_fixed)
        self.regul_compare = self._to_cpu(self.regul_compare)

        self.train_params = train_params
        self.losses = losses
        self.likelihoods = likelihoods
        self.regs = regs
        self.gradient_norms = dict(gradient_norms) if gradient_norms is not None else None
        self._set_params()
        self.likelihood = self._likelihood(self.x, self.alpha, self.beta_fixed, self.beta_denovo, self.eps_sigma)
        self.set_scores()

        reg = self._regularizer(self.beta_fixed, self.beta_denovo, reg_type=self.regularizer)
        self.reg_likelihood = self.likelihood + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1])
        try: self.reg_likelihood = self.reg_likelihood.item()
        except: return


    def _likelihood(self, M, alpha, beta_fixed, beta_denovo, eps_sigma=None):
        beta = self._get_unique_beta(beta_fixed, beta_denovo)

        ssum = torch.sum(M, axis=1)
        ddiag = torch.diag(ssum)
        mmult1 = torch.matmul(ddiag, alpha)

        a = torch.matmul(mmult1, beta)

        if eps_sigma == None: 
            _log_like_matrix = dist.Poisson(a).log_prob(M)
        else:
            xx = a + dist.HalfNormal(eps_sigma).sample()
            _log_like_matrix = dist.Poisson(xx).log_prob(M)

        _log_like_sum = torch.sum(_log_like_matrix)
        _log_like = float("{:.3f}".format(_log_like_sum.item()))

        return _log_like


    def _compute_posterior_probs(self, to_cpu=True, compute_exp=True):
        scale = self.hyperparameters["scale_factor"] if self.dirichlet_prior else 1

        pi = self._get_param("pi_param", to_cpu=to_cpu, normalize=False)
        alpha_prior = self._get_param("alpha_prior_param", to_cpu=to_cpu, normalize=True) * scale  # G x K

        M = torch.tensor(self.x, dtype=torch.double)
        beta_denovo = self._get_param("beta_denovo", to_cpu=to_cpu, normalize=True)
        beta = self._get_unique_beta(self.beta_fixed, beta_denovo)
        alpha = self._get_param("alpha", normalize=True)
        if not self.dirichlet_prior: alpha_sigma = self.alpha_sigma_corr.detach().clone()

        z = torch.zeros(self.n_samples)
        n_muts = torch.sum(M, axis=1).unsqueeze(1)
        ll_k = torch.zeros((self.cluster, self.n_samples))  # K x N x C
        for k in range(self.cluster):
            alpha_k = alpha_prior[k,:]
            rate = torch.matmul( alpha_k / scale * n_muts, beta )

            logprob_alpha = dist.Dirichlet(alpha_k).log_prob(alpha) if self.dirichlet_prior else dist.Normal(alpha_k, alpha_sigma[k,:]).log_prob(alpha).sum(axis=1)  # N dim vector
            logprob_lik = dist.Poisson( rate ).log_prob(M).sum(axis=1)  # N dim vector, summed over contexts
            ll_k[k,:] = logprob_lik + torch.log(pi[k]) + logprob_alpha

        ll = self._logsumexp(ll_k)

        probs = torch.exp(ll_k - ll) if compute_exp else ll_k - ll
        z = torch.argmax(probs, dim=0)
        return self._to_cpu(z.long(), move=to_cpu), self._to_cpu(probs, move=to_cpu)


    def _norm_and_clamp(self, par):
        mmin = 0
        if torch.any(par < 0): mmin = torch.min(par, dim=-1)[0].unsqueeze(-1)

        nnum = par - mmin
        par = nnum / torch.sum(nnum, dim=-1).unsqueeze(-1)

        return par


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


    def _set_params(self):
        self._set_alpha()
        self._set_beta_denovo()
        self._set_epsilon()
        self._set_clusters()
        self.params = self.get_param_dict(convert=True)

        if isinstance(self.groups, torch.Tensor): self.groups = self.groups.tolist()


    def _set_init_params(self):
        # return
        for k, v_tmp in self.init_params.items():
            v = self._to_cpu(v_tmp, move=True)
            if v is None: continue

            if k == "alpha_noise_param" or k == "alpha":
                self.init_params[k] = pd.DataFrame(np.array(v), index=self.alpha.index, columns=self.alpha.columns)
            elif k == "beta_dn_param":
                self.init_params[k] = pd.DataFrame(np.array(v), index=self.beta_denovo.index, columns=self.beta_denovo.columns) if self.beta_denovo is not None else np.array(v)
            elif k == "alpha_prior_param":
                self.init_params[k] = pd.DataFrame(np.array(v), index=range(self.n_groups), columns=self.alpha.columns)
            elif k == "epsilon_var":
                self.init_params[k] = pd.DataFrame(np.array(v), index=self.eps_sigma.index, columns=self.eps_sigma.columns)
            else:
                self.init_params[k] = np.array(v)


    def set_scores(self):
        self._set_bic()
        self._set_aic()
        self._set_icl()


    def _set_alpha(self):
        self.alpha = self._get_param("alpha", normalize=True)
        self.alpha_unn = self._get_param("alpha", normalize=False)

        self.alpha_prior = self._get_param("alpha_prior_param", normalize=True)
        self.alpha_prior_unn = self._get_param("alpha_prior_param", normalize=False)

        self.alpha_noise = self._get_param("alpha_noise_param", normalize=False)


    def _set_beta_denovo(self):
        self.beta_denovo = self._get_param("beta_denovo", normalize=True)
        self.beta_denovo_unn = self._get_param("beta_denovo", normalize=False)


    def _set_epsilon(self):
        self.eps_sigma = self._get_param("lambda_epsilon", normalize=False)


    def _set_clusters(self, to_cpu=True):
        if self.cluster is None:
            self.pi = None
            self.post_probs = None
            return

        self.pi = self._get_param("pi_param", normalize=False, to_cpu=to_cpu)

        self.groups, self.post_probs = self._compute_posterior_probs(to_cpu=to_cpu)


    def _set_bic(self):
        _log_like = self.likelihood
        # adding regularizer
        if self.reg_weight != 0 and self.reg_bic:
            reg = self._regularizer(self.beta_fixed, self.beta_denovo, reg_type = self.regularizer)
            _log_like += self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1])

        k = self._number_of_params() 
        n = self.n_samples
        bic = k * torch.log(torch.tensor(n, dtype=torch.float64)) - (2 * _log_like)

        self.bic = bic.item()

    def _set_aic(self):
        _log_like = self.likelihood
        # adding regularizer
        if self.reg_weight != 0 and self.reg_bic:
            reg = self._regularizer(self.beta_fixed, self.beta_denovo, reg_type = self.regularizer)
            _log_like += self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1])

        k = self._number_of_params() 
        aic = 2*k - 2*_log_like

        if (isinstance(aic, torch.Tensor)):
            self.aic = aic.item()
        else:
            self.aic = aic


    def _set_icl(self):
        self.icl = None
        if self.cluster is not None:
            icl = self.bic + self._compute_entropy()
            self.icl = icl.item()


    def _compute_entropy(self, params=None) -> np.array:
        '''
        `entropy(z) = - sum^K( sum^N( z_probs_nk * log(z_probs_nk) ) )`
        `entropy(z) = - sum^K( sum^N( exp(log(z_probs_nk)) * log(z_probs_nk) ) )`
        '''
        if params is None:
            params = self.params

        logprobs = self._compute_posterior_probs(to_cpu=True, compute_exp=False)[1]
        entr = 0
        for n in range(self.n_samples):
            for k in range(self.cluster):
                entr += torch.exp(logprobs[k,n]) * logprobs[k,n]
        return -entr.detach()


    def _number_of_params(self):
        if self.groups is not None:
            n_grps = len(np.unique(np.array(self._to_cpu(self.groups, move=True))))
        k = 0
        if self.k_denovo == 0 and torch.sum(self.beta_fixed) == 0:
            k = 0
        else:
            if self.initial_fit is None and not self.new_hier:
                k += self.k_denovo * self.contexts # beta denovo

        if self.cluster is not None:
            k += n_grps  # mixing proportions
            # k += self.params["pi"].numel()  # mixing proportions

        if self.eps_sigma is not None:
            k += self.eps_sigma.shape[0] * self.eps_sigma.shape[1]  # random noise

        if self.new_hier:
            k += self.params["alpha_prior"].shape[1] * n_grps
            # k += self.params["alpha_prior"].numel()
            # if self.params["alpha_noise"] is not None:
            #     k += self.params["alpha_noise"].numel() # alpha if noise is learned
        else:
            if self.params["alpha_prior"] is not None:
                k += self.params["alpha_prior"].shape[1] * n_grps
                # k += self.params["alpha_prior"].numel()
            if not self.new_hier:
                k += self.n_samples * self.K  # alpha if no noise is learned

        print("N parameters", k)
        return k


    def _to_cpu(self, param, move=True):
        if param is None: return None
        if move and self.CUDA and torch.cuda.is_available() and isinstance(param, torch.Tensor):
            return param.cpu()
        return param


    def _to_gpu(self, param, move=True):
        if param is None: return None
        if move and self.CUDA and torch.cuda.is_available() and isinstance(param, torch.Tensor):
            return param.cuda()
        return param


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
            self.beta_denovo = pd.DataFrame(np.array(self._to_cpu(self.beta_denovo, move=True)), index=denovo_names, columns=mutation_features)

        # alpha
        if len(fixed_names+denovo_names) > 0:
            self.alpha = pd.DataFrame(np.array(self._to_cpu(self.alpha, move=True)), index=sample_names , columns=fixed_names + denovo_names)

        # epsilon variance
        if self.stage=="random_noise":
            self.eps_sigma = pd.DataFrame(np.array(self._to_cpu(self.eps_sigma, move=True)), index=sample_names , columns=mutation_features)
        else:
            self.eps_sigma = None

        if isinstance(self.pi, torch.Tensor): 
            self.pi = self.pi.tolist()
        if isinstance(self.post_probs, torch.Tensor): 
            self.post_probs = pd.DataFrame(np.array(torch.transpose(self._to_cpu(self.post_probs, move=True), dim0=1, dim1=0)), index=sample_names , columns=range(self.cluster))

        for parname, par in self.params.items():
            par = self._to_cpu(par, move=True)
            if parname == "alpha": self.params["alpha"] = self.alpha
            elif parname == "beta_d": self.params["beta_d"] = self.beta_denovo
            elif parname == "beta_f": self.params["beta_f"] = self.beta_fixed
            elif parname == "pi": self.params["pi"] = self.pi
            elif parname == "pi_conc0": self.params["pi_conc0"] = self.params["pi_conc0"] if self.params["pi_conc0"] is not None else None
            elif parname == "lambda_epsilon": self.params["lambda_epsilon"] = self.eps_sigma
            elif parname == "alpha_prior" and par is not None: 
                self.params["alpha_prior"] = pd.DataFrame(np.array(par), index=range(self.n_groups), columns=fixed_names + denovo_names)
            elif parname == "alpha_noise" and par is not None:
                self.params["alpha_noise"] = pd.DataFrame(np.array(par), index=sample_names, columns=fixed_names + denovo_names)

        self._set_init_params()


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


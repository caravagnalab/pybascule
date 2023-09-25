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
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


class PyBasilica():

    def __init__(
        self,
        x,
        k_denovo,
        n_steps,
        lr = 0.001,
        optim_gamma = 0.1,
        enumer = "parallel",
        cluster = None,
        hyperparameters = {"alpha_sigma":0.1, "alpha_p_sigma":1., "alpha_p_conc0":0.6, "alpha_p_conc1":0.6, "alpha_rate":5, "alpha_conc":1.,
                           "beta_d_sigma":1, "eps_sigma":10, "pi_conc0":0.6, "scale_factor_centroid":1000, "scale_factor_alpha":1000},
        dirichlet_prior = False,
        beta_fixed = None,
        compile_model = True,
        CUDA = False,
        enforce_sparsity = True,
        store_parameters = False,

        reg_weight = 0.,
        regularizer = "cosine",
        regul_denovo = True,
        regul_fixed = True,
        regul_compare = None,
        reg_bic = True, 

        stage = "", 
        seed = 10,

        nonparam = False,
        initial_fit = None
        ):

        self._hyperpars_default = {"alpha_sigma":1., "alpha_p_sigma":1., "alpha_p_conc0":0.6, "alpha_p_conc1":0.6, "alpha_rate":5, "alpha_conc":1.,
                                   "beta_d_sigma":1, "eps_sigma":10, "pi_conc0":0.6, "scale_factor_centroid":1000, "scale_factor_alpha":1000}
        self.regul_denovo = regul_denovo
        self.regul_fixed = regul_fixed
        self.initial_fit = initial_fit
        self.dirichlet_prior = dirichlet_prior
        self.two_steps = two_steps

        self._set_data_catalogue(x)
        self._set_fit_settings(enforce_sparsity=enforce_sparsity, lr=lr, optim_gamma=optim_gamma, n_steps=n_steps, \
                               compile_model=compile_model, CUDA=CUDA, regularizer=regularizer, reg_weight=reg_weight, \
                               reg_bic=reg_bic, store_parameters=store_parameters, stage=stage, seed=seed, nonparam=nonparam)

        self._set_hyperparams(enumer=enumer, cluster=cluster, hyperparameters=hyperparameters)
        self._set_k_denovo(k_denovo)
        self._set_beta_fixed(beta_fixed)

        self._fix_zero_denovo_null_reference()
        self._set_external_catalogue(regul_compare)

        self._init_seed = None
        self.K = self.k_denovo + self.k_fixed


    def _set_fit_settings(self, enforce_sparsity, lr, optim_gamma, n_steps, compile_model, CUDA, \
                          regularizer, reg_weight, reg_bic, store_parameters, stage, seed, nonparam):
        self.enforce_sparsity = enforce_sparsity
        self.lr = lr
        self.optim_gamma = optim_gamma
        self.n_steps = int(n_steps)
        self.compile_model = compile_model
        self.CUDA = CUDA
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.reg_bic = reg_bic

        self.store_parameters = store_parameters
        self.stage = stage

        self.seed = seed
        self.nonparam = nonparam


    def _set_hyperparams(self, enumer, cluster, hyperparameters):
        self.enumer = enumer

        self.cluster = cluster
        if self.cluster is not None: 
            self.cluster = int(self.cluster)
            self.n_groups = self.cluster

        self.init_params = None

        if hyperparameters is None:
            self.hyperparameters = self._hyperpars_default
        else:
            self.hyperparameters = dict()
            for parname in self._hyperpars_default.keys():
                self.hyperparameters[parname] = hyperparameters.get(parname, self._hyperpars_default[parname])


    def _fix_zero_denovo_null_reference(self):
        '''
        If there are zero denovo (k=0) and the reference is empty the model will only fit a random noise.
        '''
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
        if beta_fixed is None:
            self.beta_fixed, self.k_fixed = None, 0
            return

        if isinstance(beta_fixed, pd.DataFrame):
            self.fixed_names = list(beta_fixed.index)
        else:
            self.fixed_names = ["F"+str(f+1) for f in range(beta_fixed.shape[0])]

        if isinstance(beta_fixed, pd.DataFrame):
            first = [i for i in self.fixed_names if i in ["SBS1","SBS5"]]
            others = [i for i in self.fixed_names if i not in ["SBS1","SBS5"]]
            beta_fixed = beta_fixed.loc[first + others]
            self.hyperparameters["alpha_conc"] = self._to_gpu(torch.concat((torch.ones(len(first))*self.hyperparameters["alpha_conc"], 
                                                              torch.ones(len(others) + self.k_denovo))))
            beta_fixed = beta_fixed.values

        self.beta_fixed = torch.tensor(beta_fixed, dtype=torch.float64)
        if len(self.beta_fixed.shape) == 1: self.beta_fixed = self.beta_fixed.reshape(1, self.beta_fixed.shape[0])

        self.k_fixed = beta_fixed.shape[0]
        if self.k_fixed > 0: self._fix_zero_contexts()

        if isinstance(self.hyperparameters["alpha_conc"], int):
            self.hyperparameters["alpha_conc"] = torch.ones(self.k_denovo + self.k_fixed, dtype=torch.float64)


    def _fix_zero_contexts(self):
        '''
        If a context has a density of 0 in all signatures but X contains mutations in that context, an error will raise.
        A value of 1e-07 is set to one random signature in such contexts.
        '''
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


    def _mix_weights(self, beta):
        '''
        Function used for the stick-breaking process.
        '''
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


    def model(self):

        n_samples = self.n_samples
        cluster = self.cluster  # number of clusters or None

        alpha_sigma, alpha_rate = self.hyperparameters["alpha_sigma"], self.hyperparameters["alpha_rate"]
        alpha_p_sigma = self.hyperparameters["alpha_p_sigma"]
        # alpha_conc0, alpha_conc1 = self.hyperparameters["alpha_p_conc0"], self.hyperparameters["alpha_p_conc1"]
        beta_d_sigma = self.hyperparameters["beta_d_sigma"]
        pi_conc0 = self.hyperparameters["pi_conc0"]
        scale_factor_centroid, scale_factor_alpha = self.hyperparameters["scale_factor_centroid"], self.hyperparameters["scale_factor_alpha"]

        if self._noise_only: alpha = torch.zeros(self.n_samples, 1, dtype=torch.float64)

        # Clustering model
        if cluster is not None:
            if self.nonparam:
                with pyro.plate("beta_plate", cluster-1):
                    pi_beta = pyro.sample("beta", dist.Beta(1, pi_conc0))
                    pi = self._mix_weights(pi_beta)
            else:
                pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster, dtype=torch.float64)))

            if self.enforce_sparsity:
                scale_factor_centroid = pyro.sample("scale_factor_centroid", dist.Normal(scale_factor_centroid, 100))

                with pyro.plate("g", cluster):
                    alpha_prior = pyro.sample("alpha_t", dist.Dirichlet(self.init_params["alpha_prior_param"] * scale_factor_centroid))
                # with pyro.plate("k1", self.K):
                #     with pyro.plate("g", cluster):  # G x K matrix
                #         alpha_prior = pyro.sample("alpha_t", dist.Beta(alpha_conc1, alpha_conc0))
                # alpha_prior = self._norm_and_clamp(alpha_prior)
            else:
                with pyro.plate("k1", self.K):
                    with pyro.plate("g", cluster):  # G x K matrix
                        alpha_prior = pyro.sample("alpha_t", dist.HalfNormal(torch.tensor(alpha_p_sigma, dtype=torch.float64)))
                alpha_prior = self._norm_and_clamp(alpha_prior) 

            # if self.dirichlet_prior: 
            #     alpha_prior = alpha_prior * scale_factor_alpha  # Dirichlet
            if not self.dirichlet_prior: # Normal or Cauchy
                q05 = alpha_prior - alpha_sigma
                q95 = alpha_prior + alpha_sigma


                self.alpha_sigma_corr = (q95 - q05) / ( dist.Normal(alpha_prior, 1).icdf(torch.tensor(1-0.05/2)) -\
                                                       dist.Normal(alpha_prior, 1).icdf(torch.tensor(0.05/2)) )

        # Flat model
        else:
            if not self._noise_only:
                if self.enforce_sparsity:
                    with pyro.plate("n", self.n_samples):
                        # alpha = pyro.sample("latent_exposure", dist.Dirichlet(torch.ones(self.K, dtype=torch.float64)))
                        alpha = pyro.sample("latent_exposure", dist.Dirichlet(self.hyperparameters["alpha_conc"]))

                else:
                    with pyro.plate("k", self.K):  # columns
                        with pyro.plate("n", self.n_samples):  # rows
                            if self.enforce_sparsity:
                                alpha = pyro.sample("latent_exposure", dist.Exponential(alpha_rate))
                            else:
                                alpha = pyro.sample("latent_exposure", dist.HalfNormal(torch.tensor(alpha_sigma, dtype=torch.float64)))

                    alpha = self._norm_and_clamp(alpha)

        epsilon = None
        if self.stage == "random_noise":
            with pyro.plate("contexts3", self.contexts):  # columns
                    with pyro.plate("n3", n_samples):  # rows
                        epsilon = pyro.sample("latent_m", dist.HalfNormal(self.hyperparameters["eps_sigma"]))

        # Beta
        beta_denovo = None
        if self.k_denovo > 0:
            # with pyro.plate("k_denovo", self.k_denovo):
            #     beta_denovo = pyro.sample("latent_signatures", dist.Dirichlet(torch.ones(self.contexts, dtype=torch.float64)))

            with pyro.plate("contexts", self.contexts):  # columns
                with pyro.plate("k_denovo", self.k_denovo):  # rows
                    beta_denovo = pyro.sample("latent_signatures", dist.HalfNormal(beta_d_sigma))

            beta_denovo = self._norm_and_clamp(beta_denovo)

        beta = self._get_unique_beta(self.beta_fixed, beta_denovo)  # put together fixed and denovo
        reg = self._regularizer(self.beta_fixed, beta_denovo, self.regularizer)  # compute the regularization if needed
        self.reg = reg

        if self.cluster is not None and self.dirichlet_prior: 
            scale_factor_alpha = pyro.sample("scale_factor_alpha", dist.Normal(scale_factor_alpha, 100))

        # Observations
        with pyro.plate("n2", n_samples):
            if cluster is not None:
                z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})

                if self.dirichlet_prior: 
                    alpha = pyro.sample("latent_exposure", dist.Dirichlet(alpha_prior[z] * scale_factor_alpha))
                else: 
                    alpha  = pyro.sample("latent_exposure", dist.Normal(alpha_prior[z], self.alpha_sigma_corr[z]).to_event(1))
                    alpha = self._norm_and_clamp(alpha)

            a = torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)
            if self.stage == "random_noise": a = a + epsilon

            pyro.sample("obs", dist.Poisson(a).to_event(1), obs=self.x)

            if cluster is not None:
                # !!! Not sure about this !!!
                if self.dirichlet_prior: 
                    pyro.factor("loss", torch.log(pi[z]) + dist.Dirichlet(alpha_prior[z]).log_prob(alpha)) 
                else: 
                    pyro.factor("loss", torch.log(pi[z]) + dist.Normal(alpha_prior[z], self.alpha_sigma_corr[z]).to_event(1).log_prob(alpha))

            if self.reg_weight > 0:
                pyro.factor("loss", self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1]))


    def guide(self):
        n_samples, k_denovo = self.n_samples, self.k_denovo
        cluster = self.cluster
        init_params = self._initialize_params()
        scale_factor_centroid, scale_factor_alpha = self.hyperparameters["scale_factor_centroid"], self.hyperparameters["scale_factor_alpha"]

        if cluster is not None:
            if not self._noise_only:
                pi_param = pyro.param("pi_param", lambda: init_params["pi_param"], constraint=constraints.simplex)

                if self.nonparam:
                    pi_conc0 = pyro.param("pi_conc0", lambda: dist.Uniform(0, 2).sample([cluster-1]), constraint=constraints.greater_than_eq(torch.finfo().tiny))
                    with pyro.plate("beta_plate", cluster-1):
                        pyro.sample("beta", dist.Beta(torch.ones(cluster-1, dtype=torch.float64), pi_conc0))

                else:
                    pyro.sample("pi", dist.Delta(pi_param).to_event(1))

                if self.enforce_sparsity:
                    scale_factor_centroid = pyro.param("scale_factor_centr_param", torch.tensor(scale_factor_centroid), constraint=constraints.positive)
                    pyro.sample("scale_factor_centroid", dist.Delta(scale_factor_centroid))


                    alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.simplex)
                    with pyro.plate("g", cluster):
                        pyro.sample("alpha_t", dist.Delta(alpha_prior_param).to_event(1))
                    # alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.greater_than_eq(torch.finfo().tiny))
                    # with pyro.plate("k1", self.K):
                    #     with pyro.plate("g", self.cluster):
                    #         pyro.sample("alpha_t", dist.Delta(alpha_prior_param))
                else:
                    alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.greater_than_eq(0.))
                    with pyro.plate("k1", self.K):
                        with pyro.plate("g", self.cluster):
                            pyro.sample("alpha_t", dist.Delta(alpha_prior_param))

                alpha_prior_param = self._norm_and_clamp(alpha_prior_param)
                if self.dirichlet_prior: 
                    scale_factor_alpha = pyro.param("scale_factor_alpha_param", torch.tensor(scale_factor_alpha), constraint=constraints.positive)
                    pyro.sample("scale_factor_alpha", dist.Delta(scale_factor_alpha))

                with pyro.plate("n2", n_samples):
                    z = pyro.sample("latent_class", dist.Categorical(pi_param), infer={"enumerate":self.enumer})

                    if self.dirichlet_prior: 

                        alpha = pyro.param("alpha", lambda: alpha_prior_param[z.long()] * scale_factor_alpha, constraint=constraints.simplex)  # Dirichlet
                    else: 
                        alpha = pyro.param("alpha", lambda: alpha_prior_param[z.long()], constraint=constraints.greater_than_eq(0))  # Normal or Cauchy

                    pyro.sample("latent_exposure", dist.Delta(alpha).to_event(1))

        else:
            if not self._noise_only:
                alpha_param = init_params["alpha"]
                if self.enforce_sparsity:
                    with pyro.plate("n", n_samples):
                        alpha = pyro.param("alpha", init_params["alpha"], constraint=constraints.simplex)
                        pyro.sample("latent_exposure", dist.Delta(alpha).to_event(1))

                else:
                    with pyro.plate("k", self.K):
                        with pyro.plate("n", n_samples):
                            if self.enforce_sparsity:
                                alpha = pyro.param("alpha", lambda: alpha_param, constraint=constraints.greater_than(0.0))
                            else:
                                alpha = pyro.param("alpha", lambda: alpha_param, constraint=constraints.greater_than_eq(0.0))
                            pyro.sample("latent_exposure", dist.Delta(alpha))

        # Epsilon 
        if self.stage == "random_noise":
            eps_sigma = pyro.param("lambda_epsilon", init_params["epsilon_var"], constraint=constraints.positive)

            with pyro.plate("contexts3", self.contexts):
                with pyro.plate("n3", n_samples):
                    pyro.sample("latent_m", dist.HalfNormal(eps_sigma))

        # Beta
        if k_denovo > 0:
            # beta_param = pyro.param("beta_denovo", lambda: init_params["beta_dn_param"], constraint=constraints.simplex)
            # with pyro.plate("k_denovo", k_denovo):
            #     pyro.sample("latent_signatures", dist.Delta(beta_param).to_event(1))

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
            removed_idx, data_unq = self.check_input_kmeans(X.numpy())  # if multiple points are equal the function will throw an error
            km = KMeans(n_clusters=G, random_state=seed).fit(data_unq)

            clusters = km.labels_
            for rm in sorted(removed_idx.keys()):
                clusters = np.insert(clusters, rm, 0, 0)  # insert 0 elements to restore the original number of obs 

            for rm in removed_idx.keys():
                rpt = removed_idx[rm]  # the index of the kept row
                clusters[rm] = clusters[rpt]  # insert in the repeated elements the correct cluster

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
        hyperpars_true = self.hyperparameters
        x_true, n_samples_true = self.x, self.n_samples

        self.cluster, self.hyperparameters = None, self._hyperpars_default

        pyro.get_param_store().clear()
        self._fit(set_attributes=False)

        alpha = self._get_param("alpha", normalize=False, to_cpu=False)
        beta_dn = self._get_param("beta_denovo", normalize=False, to_cpu=False) 
        pyro.get_param_store().clear()

        self.x, self.n_samples = x_true, n_samples_true
        self.cluster = cluster_true
        self.hyperparameters = hyperpars_true

        return alpha, beta_dn


    def _initialize_params_clustering(self):
        pi = alpha_prior = epsilon = beta_dn = None

        if self.initial_fit is None:
            alpha, beta_dn = self._run_initial_fit()
        else:
            alpha = self._to_gpu(self.initial_fit.alpha_unn, move=True)
            beta_dn = self._to_gpu(self.initial_fit.beta_denovo_unn, move=True)

        km = self._initialize_weights(X=alpha.clone(), G=self.cluster)  # run the Kmeans
        pi_km = torch.tensor([(np.where(km.labels_ == k)[0].shape[0]) / self.n_samples for k in range(km.n_clusters)])
        groups_kmeans = torch.tensor(km.labels_)
        alpha_prior_km = self._norm_and_clamp(torch.tensor(km.cluster_centers_))

        pi = self._to_gpu(pi_km, move=True)
        alpha_prior = self._to_gpu(alpha_prior_km, move=True)

        if self.stage == "random_noise": epsilon = torch.ones(self.n_samples, self.contexts, dtype=torch.float64) * self.hyperparameters["eps_sigma"]

        params = self._create_init_params_dict(pi=torch.tensor(pi, dtype=torch.float64), 
                                               alpha_prior=torch.tensor(alpha_prior, dtype=torch.float64), 
                                               alpha=alpha, epsilon=epsilon, beta_dn=beta_dn)
        params["init_clusters"] = groups_kmeans
        return params


    def _initialize_params_nonhier(self):
        pi = alpha_prior = alpha = epsilon = beta_dn = None

        alpha_sigma = self.hyperparameters["alpha_sigma"]
        alpha_p_conc0, alpha_p_conc1 = self.hyperparameters["alpha_p_conc0"], self.hyperparameters["alpha_p_conc1"]
        beta_d_sigma = self.hyperparameters["beta_d_sigma"]
        eps_sigma = self.hyperparameters["eps_sigma"]

        if self.cluster is not None:
            ones_tmp = torch.ones(self.cluster, self.K, dtype=torch.float64)
            pi = torch.ones(self.cluster, dtype=torch.float64)
        else:
            ones_tmp = torch.ones(self.n_samples, self.K, dtype=torch.float64)

        if self.enforce_sparsity:
            # alpha = dist.Beta(ones_tmp * alpha_p_conc1, ones_tmp * alpha_p_conc0).sample()
            alpha = dist.Dirichlet(torch.ones(self.K, dtype=torch.float64)).sample((self.n_samples,))
        else:
            alpha = dist.HalfNormal(ones_tmp * alpha_sigma).sample()

        if self.stage == "random_noise":
            epsilon = dist.HalfNormal(torch.ones(self.n_samples, self.contexts, dtype=torch.float64) * eps_sigma).sample()

        if self.k_denovo > 0:
            beta_dn = dist.HalfNormal(torch.ones(self.k_denovo, self.contexts, dtype=torch.float64) * beta_d_sigma).sample()

        params = self._create_init_params_dict(pi=pi, alpha_prior=alpha_prior, alpha=alpha, epsilon=epsilon, beta_dn=beta_dn)

        return params


    def _create_init_params_dict(self, pi=None, alpha_prior=None, alpha=None, epsilon=None, beta_dn=None):
        params = dict()

        if pi is not None: params["pi_param"] = pi
        if alpha_prior is not None: params["alpha_prior_param"] = alpha_prior
        if alpha is not None: params["alpha"] = alpha
        if epsilon is not None: params["epsilon_var"] = epsilon
        if beta_dn is not None: params["beta_dn_param"] = beta_dn

        return params


    def _initialize_params(self):
        if self.init_params is None:
            if self.cluster is not None: self.init_params = self._initialize_params_clustering()
            else: self.init_params = self._initialize_params_nonhier()

        return self.init_params


    def _regularizer(self, beta_fixed, beta_denovo, reg_type = "cosine"):
        loss = 0

        if self.reg_weight == 0:
            return loss
        
        if beta_fixed is not None and not isinstance(beta_fixed, torch.Tensor): beta_fixed = torch.tensor(beta_fixed)
        if beta_denovo is not None and not isinstance(beta_denovo, torch.Tensor): beta_denovo = torch.tensor(beta_denovo)

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
        if beta_fixed is None: return beta_denovo
        if beta_denovo is None or self._noise_only: return beta_fixed

        return torch.cat((beta_fixed, beta_denovo), axis=0)


    def _get_param(self, param_name, normalize=False, to_cpu=True, convert=False):
        try:
            if param_name == "beta_fixed": 
                par = self.beta_fixed
            elif param_name == "alpha" and self._noise_only:
                return self._to_gpu(torch.zeros(self.n_samples, 1, dtype=torch.float64), move=not to_cpu)
            else:
                par = pyro.param(param_name)

            par = self._to_cpu(par, move=to_cpu)
            if isinstance(par, torch.Tensor): par = par.clone().detach()
            if normalize: par = self._norm_and_clamp(par)

            if par is not None and convert:
                par = self._to_cpu(par, move=True)
                par = np.array(par)

            return par

        except Exception as e: return None


    def get_param_dict(self, convert=False, to_cpu=True, all=False):
        params = dict()
        params["alpha"] = self._get_param("alpha", normalize=True, convert=convert, to_cpu=to_cpu)
        params["beta_d"] =  self._get_param("beta_denovo", normalize=True, convert=convert, to_cpu=to_cpu)
        params["beta_f"] = self._get_param("beta_fixed", convert=convert, to_cpu=to_cpu)

        if self.cluster is not None:
            params["alpha_prior"] = self._get_param("alpha_prior_param", normalize=True, convert=convert, to_cpu=to_cpu)
            params["pi"] = self._get_param("pi_param", normalize=False, convert=convert, to_cpu=to_cpu)
            params["pi_conc0"] = self._get_param("pi_conc0", normalize=False, convert=convert, to_cpu=to_cpu)
            params["alpha_prior_unn"] = self._get_param("alpha_prior_param", normalize=False, convert=convert, to_cpu=to_cpu)

        if self.stage == "random_noise":
            params["lambda_epsilon"] = self._get_param("lambda_epsilon", normalize=False, convert=convert, to_cpu=to_cpu)


        return params


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

        lrd = self.optim_gamma ** (1 / self.n_steps)
        optimizer = pyro.optim.ClippedAdam({"lr": self.lr, "lrd":lrd, "clip_norm":1e10})

        pyro.set_rng_seed(self.seed)
        pyro.get_param_store().clear()

        self._initialize_params()

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)
        loss = svi.step()

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        self.losses = list()
        self.regs = list()
        self.likelihoods = list()
        self.train_params = list()
        for i in range(self.n_steps):   # inference - do gradient steps
            loss = svi.step()
            self.losses.append(loss)
            self.regs.append(self.reg)

            self.likelihoods.append(self._likelihood(to_cpu=False))

            if self.store_parameters and i%50==0: 
                self.train_params.append(self.get_param_dict(convert=True, all=False))

            # convergence test 
            # if len(losses) >= 100 and len(losses) % 100 == 0 and convergence(x=losses[-100:], alpha=0.05):
            #     break

        if set_attributes is False: return

        self.x = self._to_cpu(self.x)
        self.beta_fixed = self._to_cpu(self.beta_fixed)
        self.regul_compare = self._to_cpu(self.regul_compare)

        self.gradient_norms = dict(gradient_norms) if gradient_norms is not None else None
        self._set_params()
        self.likelihood = self._likelihood(to_cpu=True)

        reg = np.array(self._regularizer(self.beta_fixed, self.params["beta_d"], reg_type=self.regularizer))
        self.reg_likelihood = self.likelihood + self.reg_weight * (reg * self.x.shape[0] * self.x.shape[1])

        self.set_scores()


    def _likelihood(self, to_cpu=False):
        if self.cluster is None: llik = self._likelihood_flat(to_cpu=to_cpu)
        if self.cluster is not None: llik = self._likelihood_mixture(to_cpu=to_cpu)
        return llik.sum().item()


    def _likelihood_flat(self, to_cpu=False):
        params = self.get_param_dict(convert=False, to_cpu=to_cpu)
        beta = self._get_unique_beta(beta_fixed=params["beta_f"], beta_denovo=params["beta_d"])
        M = self._to_cpu(self.x, move=to_cpu)

        ddiag = torch.diag(torch.sum(M, axis=1))
        mmult1 = torch.matmul(ddiag, params["alpha"])
        rate = torch.matmul(mmult1, beta)

        if self.stage == "random_noise": rate += dist.HalfNormal(params["lambda_epsilon"]).sample()
        llik = dist.Poisson(rate).log_prob(M)

        return llik


    def _likelihood_mixture(self, to_cpu=False):
        params = self.get_param_dict(convert=False, to_cpu=to_cpu)
        beta = self._get_unique_beta(beta_fixed=params["beta_f"], beta_denovo=params["beta_d"])
        M = self._to_cpu(self.x, move=to_cpu)

        ddiag = torch.diag(torch.sum(M, axis=1))
        mmult = torch.matmul(ddiag, params["alpha"])
        rate = torch.matmul(mmult, beta)
        if params["lambda_epsilon"] is not None: rate += dist.HalfNormal(params["lambda_epsilon"]).sample()

        lprob_pois = dist.Poisson(rate).log_prob(M).sum(axis=1)

        llik = torch.zeros(self.cluster, self.n_samples)
        for g in range(self.cluster):
            alpha_prior_g = params["alpha_prior"][g]

            if self.dirichlet_prior:
                alpha_prior_g = self._norm_and_clamp(alpha_prior_g) * scale_factor_alpha
                lprob_alpha = dist.Dirichlet(alpha_prior_g).log_prob(params["alpha"])
            else:
                sigma = self.alpha_sigma_corr.clone().detach()
                lprob_alpha = dist.Normal(alpha_prior_g, sigma[g]).log_prob(params["alpha"]).sum(axis=1)

            llik[g, :] = torch.log(params["pi"][g]) + lprob_pois + lprob_alpha

        return llik


    def _compute_posterior_probs(self, to_cpu=True, compute_exp=True):
        ll_k = self._likelihood_mixture(to_cpu=to_cpu)
        ll = self._logsumexp(ll_k)

        probs = torch.exp(ll_k - ll) if compute_exp else ll_k - ll
        z = torch.argmax(probs, dim=0)
        return self._to_cpu(z.long(), move=to_cpu), self._to_cpu(probs, move=to_cpu)


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


    def _norm_and_clamp(self, par):
        mmin = 0
        if torch.any(par < 0): mmin = torch.min(par, dim=-1)[0].unsqueeze(-1)

        nnum = par - mmin
        par = nnum / torch.sum(nnum, dim=-1).unsqueeze(-1)

        return par


    def _set_params(self):
        self.params = self.get_param_dict(convert=True, all=True)
        self._set_clusters()
        if isinstance(self.groups, torch.Tensor): self.groups = self.groups.tolist()


    def set_scores(self):
        self._set_bic()
        self._set_aic()
        self._set_icl()


    def _set_clusters(self, to_cpu=True):
        if self.cluster is None:
            self.groups = None

        else:
            self.params["pi"] = self._get_param("pi_param", normalize=False, to_cpu=to_cpu)
            self.groups, self.params["post_probs"] = self._compute_posterior_probs(to_cpu=to_cpu)


    def _set_bic(self):
        _log_like = self.reg_likelihood

        k = self._number_of_params() 
        n = self.n_samples
        bic = k * torch.log(torch.tensor(n, dtype=torch.float64)) - (2 * _log_like)

        self.bic = bic.item()


    def _set_aic(self):
        _log_like = self.reg_likelihood

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
        if self.cluster is not None: n_grps = len(np.unique(np.array(self._to_cpu(self.groups, move=True))))
        k = 0
        if self.k_denovo == 0 and torch.sum(self.beta_fixed) == 0: k = 0
        elif self.initial_fit is None: k += self.k_denovo * self.contexts # beta denovo

        if self.cluster is not None: k += n_grps  # mixing proportions

        if self.stage == "random_noise":

            k += self.params["lambda_epsilon"].shape[0] * self.params["lambda_epsilon"].shape[1]  # random noise

        if self.cluster is not None:
            k += self.params["alpha_prior"].shape[1] * n_grps
        if not self._noise_only: k += self.n_samples * self.K  # alpha if no noise is learned

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
    


    def convert_to_dataframe(self, x):
        # mutations catalogue
        self.x = x
        sample_names, contexts = list(x.index), list(x.columns)
        fixed_names = self.fixed_names
        denovo_names = ["D"+str(d+1) for d in range(self.k_denovo)] if self.k_denovo>0 else []

        if self.groups is not None: self.groups = np.array(self.groups)

        if self.beta_fixed is not None and isinstance(self.beta_fixed, torch.Tensor) and torch.sum(self.beta_fixed) > 0:
            self.beta_fixed = pd.DataFrame(self.beta_fixed, index=fixed_names, columns=contexts)

        for parname, par in self.params.items():
            if par is None: continue
            par = self._to_cpu(par, move=True)
            if parname == "alpha": 
                self.params[parname] = pd.DataFrame(np.array(par), index=sample_names , columns=fixed_names + denovo_names)
            elif parname == "beta_d": 
                self.params[parname] = pd.DataFrame(np.array(par), index=denovo_names, columns=contexts)
            elif parname == "beta_f": 
                self.params[parname] = self.beta_fixed
            elif parname == "pi": 
                self.params[parname] = par.tolist() if isinstance(par, torch.Tensor) else par
            elif parname == "lambda_epsilon":
                self.params[parname] = pd.DataFrame(np.array(par), index=sample_names, columns=contexts)
            elif (parname == "alpha_prior" or parname == "alpha_prior_unn"): 
                self.params[parname] = pd.DataFrame(np.array(par), index=range(self.n_groups), columns=fixed_names + denovo_names)
            elif parname == "post_probs" and isinstance(par, torch.Tensor):
                self.params[parname] = pd.DataFrame(np.array(torch.transpose(par, dim0=1, dim1=0)), index=sample_names , columns=range(self.n_groups))

        self._set_init_params(sample_names=sample_names, fixed_names=fixed_names, 
                              denovo_names=denovo_names, contexts=contexts)


    def _set_init_params(self, sample_names, fixed_names, denovo_names, contexts):
        # return
        for k, v_tmp in self.init_params.items():
            v = self._to_cpu(v_tmp, move=True)
            if v is None: continue

            if k == "alpha":
                self.init_params[k] = pd.DataFrame(np.array(v), index=sample_names, columns=fixed_names + denovo_names)
            elif k == "beta_dn_param":
                self.init_params[k] = pd.DataFrame(np.array(v), index=denovo_names, columns=contexts)
            elif k == "alpha_prior_param":
                self.init_params[k] = pd.DataFrame(np.array(v), index=range(self.n_groups), columns=fixed_names + denovo_names)
            elif k == "epsilon_var":
                self.init_params[k] = pd.DataFrame(np.array(v), index=sample_names, columns=contexts)
            else:
                self.init_params[k] = np.array(v)





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


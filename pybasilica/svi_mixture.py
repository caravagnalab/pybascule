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


class PyBasilica_mixture():

    def __init__(
        self,
        alpha,
        n_steps,
        lr = 0.001,
        optim_gamma = 0.1,
        enumer = "parallel",
        cluster = 5,
        hyperparameters = {"pi_conc0":0.6, "alpha_p_conc0":0.6, "alpha_p_conc1":0.6,
                           "scale_factor_alpha":1000, "scale_factor_centroid":1000,
                           "scale_tau":10},
        compile_model = True,
        CUDA = False,
        store_parameters = False,

        seed = 10,

        nonparam = True,
        ):

        self._hyperpars_default = {"pi_conc0":0.6, "scale_factor_alpha":1000, "scale_factor_centroid":1000, 
                                   "alpha_p_conc0":0.6, "alpha_p_conc1":0.6, "scale_tau":10}
        self._set_data_catalogue(alpha)
        self._set_fit_settings(lr=lr, optim_gamma=optim_gamma, n_steps=n_steps, \
                               compile_model=compile_model, CUDA=CUDA, store_parameters=store_parameters, 
                               seed=seed, nonparam=nonparam)

        self._set_hyperparams(enumer=enumer, cluster=cluster, hyperparameters=hyperparameters)

        self._init_seed = None


    def _set_fit_settings(self, lr, optim_gamma, n_steps, compile_model, CUDA, \
                          store_parameters, seed, nonparam):
        self.lr = lr
        self.optim_gamma = optim_gamma
        self.n_steps = int(n_steps)
        self.compile_model = compile_model
        self.CUDA = CUDA

        self.store_parameters = store_parameters

        self.seed = seed
        self.nonparam = nonparam


    def _set_hyperparams(self, enumer, cluster, hyperparameters):
        self.enumer = enumer

        self.cluster = int(cluster)
        self.n_groups = self.cluster

        self.init_params = None

        if hyperparameters is None:
            self.hyperparameters = self._hyperpars_default
        else:
            self.hyperparameters = dict()
            for parname in self._hyperpars_default.keys():
                self.hyperparameters[parname] = hyperparameters.get(parname, self._hyperpars_default[parname])


    def _set_data_catalogue(self, alpha):
        if isinstance(alpha, pd.DataFrame):
            alpha = torch.tensor(alpha.values, dtype=torch.float64)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float64)

        self.alpha = alpha
        self.n_samples = alpha.shape[0]
        self.K = alpha.shape[1]
        self.alpha = self._norm_and_clamp(self.alpha)


    def _mix_weights(self, beta):
        '''
        Function used for the stick-breaking process.
        '''
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


    def model_mixture(self):
        cluster, n_samples = self.cluster, self.n_samples
        pi_conc0 = self.hyperparameters["pi_conc0"]
        scale_factor_alpha, scale_factor_centroid = self.hyperparameters["scale_factor_alpha"], self.hyperparameters["scale_factor_centroid"]

        tau = self.hyperparameters["scale_tau"]
        scale_factor_centroid = torch.max(torch.tensor(1), torch.floor(torch.tensor((self._curr_step+1) / (self.n_steps / tau)))) * (scale_factor_centroid / tau)
        scale_factor_alpha = torch.max(torch.tensor(1), torch.floor(torch.tensor((self._curr_step+1) / (self.n_steps / tau)))) * (scale_factor_alpha / tau)

        if self.nonparam:
            with pyro.plate("beta_plate", cluster-1):
                pi_beta = pyro.sample("beta", dist.Beta(1, pi_conc0))
                pi = self._mix_weights(pi_beta)
        else:
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster, dtype=torch.float64)))

        scale_factor_centroid = pyro.sample("scale_factor_centroid", dist.Normal(scale_factor_centroid, 50))
        with pyro.plate("g", cluster):
            alpha_prior = pyro.sample("alpha_t", dist.Dirichlet(self.init_params["alpha_prior_param"] * scale_factor_centroid))

        scale_factor_alpha = pyro.sample("scale_factor_alpha", dist.Normal(scale_factor_alpha, 50))
        with pyro.plate("n2", n_samples):
            z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})

            pyro.sample("obs", dist.Dirichlet(alpha_prior[z] * scale_factor_alpha), obs=self.alpha)


    def guide_mixture(self):
        cluster, n_samples = self.cluster, self.n_samples
        init_params = self._initialize_params()

        scale_factor_alpha, scale_factor_centroid = self.hyperparameters["scale_factor_alpha"], self.hyperparameters["scale_factor_centroid"]
        tau = self.hyperparameters["scale_tau"]

        pi_param = pyro.param("pi_param", lambda: init_params["pi_param"], constraint=constraints.simplex)
        if self.nonparam:
            pi_conc0 = pyro.param("pi_conc0_param", lambda: dist.Uniform(0, 2).sample([cluster-1]), 
                                  constraint=constraints.greater_than_eq(torch.finfo().tiny))
            with pyro.plate("beta_plate", cluster-1):
                pyro.sample("beta", dist.Beta(torch.ones(cluster-1, dtype=torch.float64), pi_conc0))
        else:
            pyro.sample("pi", dist.Delta(pi_param).to_event(1))

        scale_factor_centroid = pyro.param("scale_factor_centroid_param", torch.tensor(scale_factor_centroid / tau), constraint=constraints.positive)
        pyro.sample("scale_factor_centroid", dist.Delta(scale_factor_centroid))

        alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.simplex)
        with pyro.plate("g", cluster):
            pyro.sample("alpha_t", dist.Delta(alpha_prior_param).to_event(1))

        scale_factor_alpha = pyro.param("scale_factor_alpha_param", torch.tensor(scale_factor_alpha / tau), constraint=constraints.positive)
        pyro.sample("scale_factor_alpha", dist.Delta(scale_factor_alpha))

        with pyro.plate("n2", n_samples):
            pyro.sample("latent_class", dist.Categorical(pi_param), infer={"enumerate":self.enumer})


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


    def _initialize_weights(self, X, G):
        '''
        Function to run KMeans on the counts.
        Returns the vector of mixing proportions and the clustering assignments.
        '''
        km = self.run_kmeans(X=X, G=G, seed=15)
        self._init_km = km

        return km


    def _initialize_params_clustering(self):
        pi = alpha_prior = None

        km = self._initialize_weights(X=self.alpha.clone(), G=self.cluster)  # run the Kmeans
        pi_km = torch.tensor([(np.where(km.labels_ == k)[0].shape[0]) / self.n_samples for k in range(km.n_clusters)])
        groups_kmeans = torch.tensor(km.labels_)
        alpha_prior_km = self._norm_and_clamp(torch.tensor(km.cluster_centers_))

        pi = self._to_gpu(pi_km, move=True)
        alpha_prior = self._to_gpu(alpha_prior_km, move=True)

        params = dict()
        params["pi_param"] = torch.tensor(pi, dtype=torch.float64)
        params["alpha_prior_param"] = torch.tensor(alpha_prior, dtype=torch.float64)
        params["init_clusters"] = groups_kmeans
        return params
    

    def _initialize_params(self):
        if self.init_params is None:
            self.init_params = self._initialize_params_clustering()
        return self.init_params


    def _fit(self):
        pyro.clear_param_store()  # always clear the store before the inference

        self.alpha = self._to_gpu(self.alpha)

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

        self._curr_step = 0
        svi = SVI(self.model_mixture, self.guide_mixture, optimizer, loss=elbo)
        loss = svi.step()

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        self.losses = list()
        self.regs = list()
        self.likelihoods = list()
        self.train_params = list()
        for i in range(self.n_steps):   # inference - do gradient steps
            self._curr_step = i
            loss = svi.step()
            self.losses.append(loss)

            self.likelihoods.append(self._likelihood_mixture(to_cpu=False).sum())

            if self.store_parameters and i%50==0: 
                self.train_params.append(self.get_param_dict(convert=True, to_cpu=False))

        self.alpha = self._to_cpu(self.alpha)

        self.gradient_norms = dict(gradient_norms) if gradient_norms is not None else {}
        self.params = {**self._set_clusters(to_cpu=True), **self.get_param_dict(convert=True, to_cpu=True)}
        self.bic = self.aic = self.icl = self.reg_likelihood = None
        self.likelihood = self._likelihood_mixture(to_cpu=True).sum()
        self.set_scores()


    def get_param_dict(self, convert=False, to_cpu=True):
        params = dict()
        params["alpha_prior"] = self._get_param("alpha_prior_param", normalize=True, convert=convert, to_cpu=to_cpu)
        params["pi"] = self._get_param("pi_param", normalize=False, convert=convert, to_cpu=to_cpu)
        params["scale_factor_alpha"] = self._get_param("scale_factor_alpha_param", convert=convert, to_cpu=to_cpu)
        params["scale_factor_centroid"] = self._get_param("scale_factor_centroid_param", convert=convert, to_cpu=to_cpu)
        params["pi_conc0"] = self._get_param("pi_conc0_param", convert=convert, to_cpu=to_cpu)
        return params


    def set_scores(self):
        self._set_bic()
        self._set_aic()
        self._set_icl()


    def _get_param(self, param_name, normalize=False, to_cpu=True, convert=False):
        try:
            par = pyro.param(param_name)
            par = self._to_cpu(par, move=to_cpu)
            if isinstance(par, torch.Tensor): par = par.clone().detach()
            if normalize: par = self._norm_and_clamp(par)

            if par is not None and convert:
                par = self._to_cpu(par, move=True)
                par = np.array(par)

            return par

        except Exception as e: return None


    def _likelihood_mixture(self, to_cpu=False):
        alpha = self._to_cpu(self.alpha, move=to_cpu)
        alpha_centroid = self._get_param("alpha_prior_param", normalize=True, to_cpu=to_cpu)
        pi = self._get_param("pi_param", to_cpu=to_cpu)
        scale_factor_alpha = self._get_param("scale_factor_alpha_param", normalize=False, to_cpu=to_cpu)
        if scale_factor_alpha is None: scale_factor_alpha = self.hyperparameters["scale_factor_alpha"]

        llik = torch.zeros(self.cluster, self.n_samples)
        for g in range(self.cluster):
            alpha_prior_g = alpha_centroid[g]

            lprob_alpha = dist.Dirichlet(alpha_prior_g * scale_factor_alpha).log_prob(alpha)

            llik[g, :] = torch.log(pi[g]) + lprob_alpha

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


    def _set_clusters(self, to_cpu=True):
        params = dict()
        # self.alpha_prior = self._get_param("alpha_prior_param", normalize=True, to_cpu=to_cpu)
        # params["pi"] = self._get_param("pi_param", normalize=False, to_cpu=to_cpu)
        self.groups, params["post_probs"] = self._compute_posterior_probs(to_cpu=to_cpu)
        return params


    def _set_bic(self):
        _log_like = self.likelihood
        
        k = self._number_of_params() 
        n = self.n_samples
        bic = k * torch.log(torch.tensor(n, dtype=torch.float64)) - (2 * _log_like)

        self.bic = bic.item()


    def _set_aic(self):
        _log_like = self.likelihood

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


    def _compute_entropy(self) -> np.array:
        '''
        `entropy(z) = - sum^K( sum^N( z_probs_nk * log(z_probs_nk) ) )`
        `entropy(z) = - sum^K( sum^N( exp(log(z_probs_nk)) * log(z_probs_nk) ) )`
        '''

        logprobs = self._compute_posterior_probs(to_cpu=True, compute_exp=False)[1]
        entr = 0
        for n in range(self.n_samples):
            for k in range(self.cluster):
                entr += torch.exp(logprobs[k,n]) * logprobs[k,n]
        return -entr.detach()


    def _number_of_params(self):
        n_grps = len(np.unique(np.array(self._to_cpu(self.groups, move=True))))
        k = n_grps + self.params["alpha_prior"].shape[1] * n_grps
        return k


    # def convert_to_dataframe(self, alpha):
    #     self.alpha = alpha
    #     sample_names = list(alpha.index)
    #     signatures = list(alpha.columns)

    #     if isinstance(self.pi, torch.Tensor): 
    #         self.pi = self.pi.tolist()
    #     if isinstance(self.post_probs, torch.Tensor): 
    #         self.post_probs = pd.DataFrame(np.array(torch.transpose(self._to_cpu(self.post_probs, move=True), dim0=1, dim1=0)), index=sample_names , columns=range(self.cluster))
    #     if isinstance(self.alpha_prior, torch.Tensor):
    #         self.alpha_prior = pd.DataFrame(np.array(self.alpha_prior), index=range(self.n_groups), columns=signatures)

    #     self._set_init_params()


    # def _set_init_params(self):
    #     # return
    #     for k, v_tmp in self.init_params.items():
    #         v = self._to_cpu(v_tmp, move=True)
    #         if v is None: continue

    #         if k == "alpha_prior_param":
    #             self.init_params[k] = pd.DataFrame(np.array(v), index=range(self.n_groups), columns=self.alpha.columns)
    #         else:
    #             self.init_params[k] = np.array(v)


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



import torch
import pyro
import pyro.distributions.constraints as constraints
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pyro.infer import SVI,Trace_ELBO, JitTrace_ELBO, TraceEnum_ELBO
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from collections import defaultdict
from copy import deepcopy



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
                           "scale_tau":0},

        compile_model = True,
        CUDA = False,
        store_parameters = False,

        seed = 10,

        nonparam = True
        ):

        ## if alpha is a tensor -> more than one variant type ??

        self._hyperpars_default = {"pi_conc0":0.6, "scale_factor_alpha":1000, "scale_factor_centroid":1000, 
                                   "alpha_p_conc0":0.6, "alpha_p_conc1":0.6, "scale_tau":0}
        self._set_data_catalogue(alpha)
        self._set_fit_settings(lr=lr, optim_gamma=optim_gamma, n_steps=n_steps, \
                               compile_model=compile_model, CUDA=CUDA, store_parameters=store_parameters, 
                               seed=seed, nonparam=nonparam)

        self._set_hyperparams(enumer=enumer, cluster=cluster, hyperparameters=hyperparameters)

        self._init_seed = 15


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


    def _pad_tensor(self, param):
        max_shape = max([i.shape[1] for i in param])
        for i in range(len(param)):
            if param[i].shape[1] < max_shape:
                pad_dims = max_shape - param[i].shape[1]
                param[i] = F.pad(input=param[i], pad=(0, pad_dims, 0,0), mode="constant", value=torch.finfo().tiny)
        return param


    def _set_data_catalogue(self, alpha):
        if not isinstance(alpha, list):
            alpha = [alpha]

        self.alpha = deepcopy(alpha)
        for i in range(len(self.alpha)):
            if isinstance(self.alpha[i], pd.DataFrame):
                self.alpha[i] = torch.tensor(self.alpha[i].values, dtype=torch.float64)
            if not isinstance(self.alpha[i], torch.Tensor):
                self.alpha[i] = torch.tensor(self.alpha[i], dtype=torch.float64)

        self.alpha = torch.stack(self._pad_tensor(self.alpha))  # tensor of tensors

        self.n_samples = self.alpha.shape[-2]
        self.K = self.alpha.shape[-1]
        self.n_variants = self.alpha.shape[0]
        self.alpha = self._norm_and_clamp(self.alpha)


    def _mix_weights(self, beta):
        '''
        Function used for the stick-breaking process.
        '''
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


    def model_mixture(self):
        # alpha = torch.permute(self.alpha.clone(), dims=(1,0,2))
        alpha = self.alpha
        cluster, n_samples, n_variants = self.cluster, self.n_samples, self.n_variants
        pi_conc0 = self.hyperparameters["pi_conc0"]
        scale_factor_alpha, scale_factor_centroid = self.hyperparameters["scale_factor_alpha"], self.hyperparameters["scale_factor_centroid"]

        tau = self.hyperparameters["scale_tau"]
        if tau > 0:
            scale_factor_centroid = torch.max(torch.tensor(1), torch.floor(torch.tensor((self._curr_step+1) / (self.n_steps / tau)))) * (scale_factor_centroid / tau)
            scale_factor_alpha = torch.max(torch.tensor(1), torch.floor(torch.tensor((self._curr_step+1) / (self.n_steps / tau)))) * (scale_factor_alpha / tau)

        else:
            scale_factor_centroid = pyro.sample("scale_factor_centroid", dist.Normal(scale_factor_centroid, 50))
            scale_factor_alpha = pyro.sample("scale_factor_alpha", dist.Normal(scale_factor_alpha, 50))

        '''
        pi_beta = tensor([0.3724, 0.5093, 0.1001, 0.5359, 0.8083])
        beta = tensor([0.3724, 0.5093, 0.1001, 0.5359, 0.8083])
        beta1m_cumprod = tensor([0.6276, 0.3080, 0.2771, 0.1286, 0.0247])
        res1 = tensor([0.3724, 0.5093, 0.1001, 0.5359, 0.8083, 1.0000])
        res2 = tensor([1.0000, 0.6276, 0.3080, 0.2771, 0.1286, 0.0247])
        tensor([0.3724, 0.3197, 0.0308, 0.1485, 0.1040, 0.0247])
        '''
        if self.nonparam:
            with pyro.plate("beta_plate", cluster-1):
                pi_beta = pyro.sample("beta", dist.Beta(1, pi_conc0))
                pi = self._mix_weights(pi_beta)
        else:
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster, dtype=torch.float64)))

        with pyro.plate("g", cluster):
            with pyro.plate("n_vars1", n_variants):
                alpha_prior = pyro.sample("alpha_t", dist.Dirichlet(self.init_params["alpha_prior_param"] * scale_factor_centroid))

        with pyro.plate("n2", n_samples):
            z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})
            with pyro.plate("n_vars2", n_variants):
                pyro.sample("obs", dist.Dirichlet(alpha_prior[:,z,:] * scale_factor_alpha), obs=self.alpha)


    def guide_mixture(self):
        cluster, n_samples, n_variants = self.cluster, self.n_samples, self.n_variants
        init_params = self._initialize_params()

        scale_factor_alpha, scale_factor_centroid = self.hyperparameters["scale_factor_alpha"], self.hyperparameters["scale_factor_centroid"]
        tau = self.hyperparameters["scale_tau"]

        if tau == 0:
            scale_factor_centroid = pyro.param("scale_factor_centroid_param", torch.tensor(scale_factor_centroid), constraint=constraints.positive)
            pyro.sample("scale_factor_centroid", dist.Delta(scale_factor_centroid))
            scale_factor_alpha = pyro.param("scale_factor_alpha_param", torch.tensor(scale_factor_alpha), constraint=constraints.positive)
            pyro.sample("scale_factor_alpha", dist.Delta(scale_factor_alpha))

        pi_param = pyro.param("pi_param", lambda: init_params["pi_param"], constraint=constraints.simplex)
        if self.nonparam:
            # pi_conc0 = pyro.param("pi_conc0_param", lambda: dist.Uniform(0, 2).sample([cluster-1]), 
            pi_conc0 = pyro.param("pi_conc0_param", lambda: dist.Gamma(0.01, 0.01).sample([cluster-1]), 
                                  constraint=constraints.greater_than_eq(torch.finfo().tiny))
            with pyro.plate("beta_plate", cluster-1):
                pyro.sample("beta", dist.Beta(torch.ones(cluster-1, dtype=torch.float64), pi_conc0))
        else:
            pyro.sample("pi", dist.Delta(pi_param).to_event(1))

        alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_params["alpha_prior_param"], constraint=constraints.simplex)
        with pyro.plate("g", cluster):
            with pyro.plate("n_vars1", n_variants):
                pyro.sample("alpha_t", dist.Delta(alpha_prior_param).to_event(1))

        with pyro.plate("n2", n_samples):
            pyro.sample("latent_class", dist.Categorical(pi_param), infer={"enumerate":self.enumer})


    def _check_kmeans(self, X):
        if X.unique(dim=0).shape[0] < 2:
            return False
        return True


    def _find_best_G(self, X, g_interval, seed, index_fn=calinski_harabasz_score):
        g_min = min(max(g_interval[0], 2), X.unique(dim=0).shape[0])
        g_max = min(g_min, X.unique(dim=0).shape[0])

        if g_min > g_max: g_max = g_min + 1
        if g_min == g_max: return g_min

        k_interval = (g_min, g_max)

        scores = torch.zeros(k_interval[1])
        for g in range(k_interval[0], k_interval[1]):
            km = self.run_kmeans(X=X, G=g, seed=seed)
            labels = km.labels_
            real_g = len(np.unique(labels))
            scores[real_g] = max(scores[real_g], index_fn(X, labels))

        best_k = scores.argmax()  # best k is the one maximing the calinski score
        return best_k


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


    def kmeans_optim(self, X, G):
        '''
        Function to run KMeans on the counts.
        Returns the vector of mixing proportions and the clustering assignments.
        '''
        best_G = self._find_best_G(X=X, g_interval=[_ for _ in range(G-5, G+1)], seed=self._init_seed)

        if best_G < 2: return

        km = self.run_kmeans(X=X, G=best_G, seed=self._init_seed)
        self._init_km = km

        return km, best_G


    def _initialize_params_clustering(self):
        pi = alpha_prior = None

        alpha_km = torch.cat(tuple(self.alpha.clone()), dim=1)

        pi_km = torch.ones((self.cluster,)) * 10e-3
        alpha_prior_km = torch.ones((self.cluster, self.K * self.n_variants)) * 10e-3

        km, best_G = self.kmeans_optim(X=alpha_km, G=self.cluster)  # run the Kmeans

        ## setup mixing proportion vector
        pi_km[:best_G] = torch.tensor([(np.where(km.labels_ == g)[0].shape[0]) / self.n_samples for g in range(km.n_clusters)])

        alpha_prior_km[:best_G,:] = self._norm_and_clamp(torch.tensor(km.cluster_centers_))
        alpha_prior_km[alpha_prior_km < torch.finfo().tiny] = torch.finfo().tiny

        pi_km = dist.Dirichlet(pi_km * 30).sample()

        last, alpha_prior = 0, list()
        for i in range(self.n_variants):
            alpha_p_tmp = dist.Dirichlet(alpha_prior_km[:,last : last + self.K] * 30).sample()
            alpha_p_tmp[alpha_p_tmp < torch.finfo().tiny] = torch.finfo().tiny
            alpha_p_tmp = self._norm_and_clamp(alpha_p_tmp)

            alpha_prior.append(alpha_p_tmp)
            last = last + self.K - 1

        pi = self._to_gpu(pi_km, move=True)
        alpha_prior = self._to_gpu(torch.stack(alpha_prior), move=True)

        params = dict()
        params["pi_param"] = pi.clone().detach().double()
        params["alpha_prior_param"] = alpha_prior.clone().detach().double()
        params["init_clusters"] = torch.tensor(km.labels_)
        return params


    def _initialize_params_random(self):
        pi = dist.Dirichlet(torch.ones(self.cluster, dtype=torch.float64)).sample()
        alpha_prior = dist.Dirichlet(torch.ones(self.n_variants, self.cluster, self.K, dtype=torch.float64)).sample()

        params = {"pi_param":pi, "alpha_prior_param":alpha_prior}
        return params


    def _initialize_params(self):
        if self.init_params is None:
            if self._check_kmeans(torch.cat(tuple(self.alpha.clone()), dim=1)):
                self.init_params = self._initialize_params_clustering()
            else:
                self.init_params = self._initialize_params_random()
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
        loss = float(svi.step())

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        self.losses = list()
        self.regs = list()
        self.likelihoods = list()
        self.train_params = list()
        for i in range(self.n_steps):   # inference - do gradient steps
            self._curr_step = i
            loss = float(svi.step())
            self.losses.append(loss)

            # self.likelihoods.append(self._likelihood_mixture(to_cpu=False).sum())

            if self.store_parameters and i%50==0: 
                self.train_params.append(self.get_param_dict(convert=True, to_cpu=False))

        self.alpha = self._to_cpu(self.alpha)

        self.gradient_norms = dict(gradient_norms) if gradient_norms is not None else {}
        self.params = {**self._set_clusters(to_cpu=True), **self.get_param_dict(convert=True, to_cpu=True)}
        self.init_params = self._get_init_param_dict(convert=True, to_cpu=True)
        self.bic = self.aic = self.icl = self.reg_likelihood = None
        self.likelihood = self._likelihood_mixture(to_cpu=True).sum().item()
        self.set_scores()


    def get_param_dict(self, convert=False, to_cpu=True):
        params = dict()
        params["alpha_prior"] = self._get_param("alpha_prior_param", normalize=True, convert=convert, to_cpu=to_cpu)
        params["pi"] = self._get_param("pi_param", normalize=False, convert=convert, to_cpu=to_cpu)
        params["scale_factor_alpha"] = self._get_param("scale_factor_alpha_param", convert=convert, to_cpu=to_cpu)
        params["scale_factor_centroid"] = self._get_param("scale_factor_centroid_param", convert=convert, to_cpu=to_cpu)
        params["pi_conc0"] = self._get_param("pi_conc0_param", convert=convert, to_cpu=to_cpu)
        return params


    def _get_init_param_dict(self, convert=True, to_cpu=True):
        params = dict()
        for k, v in self.init_params.items():
            if not convert: params[k] = v
            else:
                params[k] = self._to_cpu(v, move=to_cpu)
                if len(v.shape) > 2: params[k] = self._concat_tensors(params[k], dim=1)
                params[k] = np.array(params[k])
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
                if len(par.shape) > 2: par = self._concat_tensors(par, dim=1)
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
            lprob_alpha = 0
            for v in range(self.n_variants):
                alpha_prior_g = alpha_centroid[v, g]

                lprob_alpha += dist.Dirichlet(alpha_prior_g * scale_factor_alpha).log_prob(alpha[v])

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


    def _concat_tensors(self, param, dim=1):
        if len(param.shape) <= 2: return param
        if isinstance(param, torch.Tensor):
            return torch.cat(tuple(param), dim=dim)
        return np.concatenate(tuple(param), axis=dim)


    def _pad_dataframes(self, param):
        new_param = []
        max_shape = max([i.shape[1] for i in param])
        for i in range(len(param)):
            if param[i].shape[1] < max_shape:
                pad_dims = max_shape - param[i].shape[1]
                columns = list(str(i)+"_"+param[i].columns) + [str(i)+"_P"+str(j) for j in range(pad_dims)]
                index = param[i].index
                values = F.pad(input=torch.tensor(param[i].values), pad=(0, pad_dims, 0,0), mode="constant", value=torch.finfo().tiny)
                new_param.append(pd.DataFrame(values.numpy(), index=index, columns=columns))
            else: 
                new_param.append(param[i].copy(deep=True))
                new_param[i].columns = list(str(i)+"_"+param[i].columns)
        return new_param


    def convert_to_dataframe(self, alpha):
        # mutations catalogue
        if not isinstance(alpha, list): alpha = [alpha]

        self.alpha = pd.concat(self._pad_dataframes(alpha), axis=1)
        sample_names, sigs = list(self.alpha.index), list(self.alpha.columns)

        if self.groups is not None: self.groups = np.array(self.groups)

        for parname, par in self.params.items():
            if par is None: continue
            par = self._to_cpu(par, move=True)
            if parname == "pi": 
                self.params[parname] = par.tolist() if isinstance(par, torch.Tensor) else par
            elif (parname == "alpha_prior" or parname == "alpha_prior_unn"): 
                self.params[parname] = pd.DataFrame(np.array(self._concat_tensors(par, dim=1)), index=range(self.n_groups), columns=sigs)
            elif parname == "post_probs" and isinstance(par, torch.Tensor):
                self.params[parname] = pd.DataFrame(np.array(torch.transpose(par, dim0=1, dim1=0)), index=sample_names , columns=range(self.n_groups))

        for k, v in self.hyperparameters.items():
            if v is None: continue
            v = self._to_cpu(v, move=True)
            if isinstance(v, torch.Tensor): 
                if len(v.shape) == 0: self.hyperparameters[k] = int(v)
                else: self.hyperparameters[k] = v.numpy()

        self._set_init_params(sigs)


    def _set_init_params(self, sigs):
        # return
        for k, v_tmp in self.init_params.items():
            v = self._to_cpu(v_tmp, move=True)
            if v is None: continue

            if k == "alpha_prior_param":
                self.init_params[k] = pd.DataFrame(np.array(self._concat_tensors(v, dim=1)), index=range(self.n_groups), columns=sigs)
            else:
                self.init_params[k] = np.array(v)


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



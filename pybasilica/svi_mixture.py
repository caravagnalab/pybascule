import torch
import pyro
import pyro.distributions.constraints as constraints
import pyro.distributions as dist
import pyro.poutine as poutine
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO, config_enumerate
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from collections import defaultdict
from copy import deepcopy
from tqdm import trange


class PyBasilica_mixture():
    def __init__(
        self,
        alpha,
        n_steps,
        lr = 0.005,
        optim_gamma = 0.1,
        enumer = "parallel",
        cluster = 5,
        hyperparameters = None,

        compile_model = False,
        CUDA = False,
        store_parameters = False,

        seed = 10,
        nonparam = True
        ):

        self._hyperpars_default = {"pi_conc0":0.6, "scale_factor_alpha":1,
                                   "scale_factor_centroid":1, "scale_tau":0}
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
            self.hyperparameters = {}
            for parname in self._hyperpars_default.keys():
                self.hyperparameters[parname] = hyperparameters.get(parname, self._hyperpars_default[parname])


    def _pad_tensor(self, param):
        '''
        `param`: list of tensors for which the rightmost dimension will be padded \\
        output: list of matrices of same dimension N x K
        '''
        max_shape = max([i.shape[-1] for i in param])
        for i, v in enumerate(param):
            if v.shape[-1] < max_shape:
                pad_dims = max_shape - v.shape[-1]
                v = F.pad(input=v, pad=(0, pad_dims, 0, 0), mode="constant", value=torch.finfo().tiny)
                param[i] = self._norm_and_clamp(v)
        return param


    def _set_data_catalogue(self, alpha) -> torch.Tensor:
        '''
        `alpha`: either a list (len=V) of N x K matrices or a single N x K matrix (if V=1) \\
        each matrix will be padded if necessary (adding missing K=0) and converted to tensor \\
        output: a tensor of shape N x V x K
        '''
        if not isinstance(alpha, list): alpha = [alpha]

        alpha_stacked = deepcopy(alpha)
        for i, v in enumerate(alpha_stacked):  # for each alpha matrix
            if isinstance(v, pd.DataFrame):
                v = torch.tensor(v.values, dtype=torch.float64)
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float64)
            alpha_stacked[i] = v

        # alpha_stacked -> V x N x K, self.alpha -> N x V x K
        self.alpha = torch.permute(torch.stack(self._pad_tensor(alpha_stacked)), (1, 0, 2))

        self.n_samples = self.alpha.shape[0]
        self.K = self.alpha.shape[-1]
        self.n_variants = self.alpha.shape[-2]


    def _mix_weights(self, beta):
        '''
        Function used for the stick-breaking process.
        '''
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


# alpha -> N x V x K
# alpha_prior -> G x V x K
# pi -> [G]


    def model_mixture(self):
        cluster, n_samples, n_variants = self.cluster, self.n_samples, self.n_variants
        pi_conc0 = self.hyperparameters["pi_conc0"]

        if self.nonparam:
            with pyro.plate("beta_plate", cluster-1):
                pi_beta = pyro.sample("beta_pi", dist.Beta(1, pi_conc0))
            pi = self._mix_weights(pi_beta)
        else:
            # pi = pyro.sample("pi", dist.Dirichlet(torch.ones(cluster, dtype=torch.float64)))
            pi = pyro.sample("pi", dist.Dirichlet(self.init_params["pi"]))

        # alpha_prior = V x G x K 
        alpha_centr = torch.permute(self.init_params["alpha_t"], (1,0,2))
        with pyro.plate("g1", cluster):
            with pyro.plate("n_vars1", n_variants):
                # alpha_prior = pyro.sample("alpha_t", dist.Dirichlet(torch.ones(self.K)))
                alpha_prior = pyro.sample("alpha_t", dist.Dirichlet(alpha_centr))

        # # Usual enumeration with "enumerate":"parallel"
        # with pyro.plate("n", n_samples):
        #     z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})
        #     alpha_c = Vindex(alpha_prior)[z]
        #     pyro.sample("obs", dist.Dirichlet(alpha_c).to_event(1), obs=self.alpha)

        # Try with relaxedOneHotCategorical -> continuous Categorical
        # alpha = N x V x K
        with pyro.plate("n", n_samples):
            z_onehot = pyro.sample("latent_class", dist.RelaxedOneHotCategorical(temperature=torch.tensor(0.1), 
                                                                                logits=torch.log(pi)))
            # first find argmax over variant types (dim=-2), then over clusters (dim=-1)
            z = z_onehot.argmax(dim=-1).long()

            # z = pyro.sample("latent_class", dist.Categorical(pi), infer={"enumerate":self.enumer})

            for v in pyro.plate("n_vars2", n_variants):
                pyro.sample(f"obs_{v}", dist.Dirichlet(Vindex(alpha_prior)[v,z,:]), 
                            obs=self.alpha[:,v,:])


    def guide_mixture(self):
        cluster, n_samples, n_variants = self.cluster, self.n_samples, self.n_variants
        init_params = self._initialize_params()

        init_pi = init_params["pi"]
        pi_param = pyro.param("pi_param", lambda: init_pi, constraint=constraints.simplex)
        if self.nonparam:
            pi_conc0 = pyro.param("pi_conc0_param", lambda: (init_pi[:-1])*10,
                                  constraint=constraints.greater_than_eq(torch.finfo().tiny))

            with pyro.plate("beta_plate", cluster-1):
                pyro.sample("beta_pi", dist.Beta(1, pi_conc0))
        else:
            pyro.sample("pi", dist.Delta(pi_param).to_event(1))

        # print("GUIDE pi", pi_param)

        init_alpha = torch.permute(init_params["alpha_t"], (1,0,2))
        alpha_prior_param = pyro.param("alpha_prior_param", lambda: init_alpha, constraint=constraints.simplex)
        with pyro.plate("g1", cluster):
            with pyro.plate("n_vars1", n_variants):
                pyro.sample("alpha_t", dist.Delta(alpha_prior_param).to_event(1))

        latent_class = pyro.param("latent_class_param", lambda: init_params["latent_class"], constraint=constraints.simplex)
        with pyro.plate("n", n_samples):
            pyro.sample("latent_class", dist.Delta(latent_class).to_event(1))
            
            # pyro.sample("latent_class", dist.Categorical(pi_param), infer={"enumerate":self.enumer})


    # def _llik_inference(self, params):
    #     G, N, V, K = self.cluster, self.n_samples, self.n_variants, self.K
    #     pi = params["pi"].unsqueeze(1)
    #     alpha_prior = params["alpha_prior"].permute(1,0,2).unsqueeze(1)
    #     alpha = self.alpha.unsqueeze(0)  # add empty first dimension

    #     assert pi.shape == (G,1)
    #     assert alpha_prior.shape == (G,1,V,K)
    #     assert alpha.shape == (1,N,V,K)

    #     lprob = dist.Dirichlet(alpha_prior).log_prob(alpha).sum(dim=-1)
    #     lk = torch.log(pi) + lprob

    #     assert lk.shape == (G, N)

    #     # logsumexp trick to sum the values of weighted log likelihood
    #     m = torch.max(lk, dim=-2).values.unsqueeze(0)
    #     expp = torch.exp(lk - m).sum(dim=0)
    #     wlk = m + torch.log(expp)

    #     return(wlk.sum())


    def _check_kmeans(self, X):
        '''
        If there are less than 2 identical samples in the data the kmeans cannot be run.
        '''
        if X.unique(dim=0).shape[0] < 2:
            return False
        return True


    def _find_best_G(self, X, g_interval, seed, index_fn=calinski_harabasz_score):
        ## g_interval is 0:G
        g_min = min(max(g_interval[0], 2), X.unique(dim=0).shape[0])
        g_max = max(g_interval[-1], g_min)

        if g_min > g_max: g_max = g_min + 1
        if g_min == g_max: return g_min

        k_interval = (int(g_min), int(g_max))

        scores = torch.zeros(k_interval[1])
        for g in range(k_interval[0], k_interval[1]):
            km = self.run_kmeans(X=X, G=g, seed=seed)
            labels = km.labels_
            real_g = len(np.unique(labels))
            scores[real_g] = max(scores[real_g], index_fn(X, labels))

        best_k = scores.argmax()  # best k is the one maximing the calinski score
        return int(best_k)


    def run_kmeans(self, X, G, seed):
        X = self._to_cpu(X, move=True)
        try:
            km = KMeans(n_clusters=G, random_state=seed, n_init=10).fit(X.numpy())

        except:
            removed_idx, data_unq = self.check_input_kmeans(X.numpy())  # if multiple points are equal the function will throw an error
            km = KMeans(n_clusters=G, random_state=seed, n_init=10).fit(data_unq)

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
        # best_G = self._find_best_G(X=X, g_interval=[_ for _ in range(G+1)], seed=self._init_seed)
        best_G = G

        if best_G < 2: return 2

        km = self.run_kmeans(X=X, G=best_G, seed=self._init_seed)
        self._init_km = km

        return km, best_G


    def _initialize_params_clustering(self):
        '''
        Function to initialize the values for each parameter.
        The pars are initialized through a Kmeans, run on the optimal number of clusters G.
        `alpha` is permuted and concatenated to have shape N x K*V
        `alpha_prior` is estimated as G x K*V
        '''
        pi = alpha_prior = None

        alpha_perm = torch.permute(self.alpha.clone(), (1,0,2))
        alpha_km = torch.cat(tuple(alpha_perm), dim=1)

        pi_km = torch.ones((self.cluster,)) * 10e-4
        alpha_prior_km = torch.ones((self.cluster, self.K * self.n_variants)) / self.K

        km, best_G = self.kmeans_optim(X=alpha_km, G=self.cluster)  # run the Kmeans

        ## setup mixing proportion vector
        pi_km[:best_G] = torch.tensor([(np.where(km.labels_ == g)[0].shape[0]) / self.n_samples for g in range(km.n_clusters)])
        pi_km = dist.Dirichlet(pi_km * 10).sample()

        alpha_prior_km[:best_G,:] = torch.tensor(km.cluster_centers_)
        last, alpha_prior = 0, list()
        for _ in range(self.n_variants):
            alpha_p_tmp = dist.Dirichlet(alpha_prior_km[:, last:(last + self.K)] * 50).sample()
            alpha_p_tmp[alpha_p_tmp < torch.finfo().tiny] = torch.finfo().tiny
            alpha_p_tmp = self._norm_and_clamp(alpha_p_tmp)

            # alpha_p_tmp = alpha_prior_km[:, last:(last + self.K)]
            alpha_prior.append(alpha_p_tmp)
            last = last + self.K

        latent_class =  self._to_gpu(dist.RelaxedOneHotCategorical(temperature=torch.tensor(0.1), 
                                                                   logits=torch.log(pi_km)).sample((self.n_samples,)))
        pi = self._to_gpu(pi_km, move=True)
        alpha_prior = self._to_gpu(torch.stack(alpha_prior), move=True)

        params = dict()
        params["pi"] = pi.clone().detach().double()
        params["alpha_t"] = torch.permute(alpha_prior.clone().detach().double(), (1,0,2))
        params["init_clusters"] = torch.tensor(km.labels_)
        params["latent_class"] = latent_class.clone().detach().double()
        return params


    def _initialize_params_random(self):
        pi = dist.Dirichlet(torch.ones(self.cluster, dtype=torch.float64)).sample()
        alpha_prior = dist.Dirichlet(torch.ones(self.cluster, self.n_variants, self.K, dtype=torch.float64)).sample()

        params = {"pi_param":pi, "alpha_prior_param":alpha_prior}
        return params


    def _initialize_params(self):
        if self.init_params is None:
            if self._check_kmeans(torch.cat(tuple(self.alpha.clone()), dim=1)) and self.cluster > 1:
                self.init_params = self._initialize_params_clustering()
            else:
                self.init_params = self._initialize_params_random()
            # self.init_params = self._initialize_params_random()
        return self.init_params


    def _fit(self):
        pyro.clear_param_store()  # always clear the store before the inference

        self.alpha = self._to_gpu(self.alpha)

        if self.CUDA and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type(t=torch.FloatTensor)

        pyro.set_rng_seed(self.seed)
        pyro.get_param_store().clear()

        self._initialize_params()

        # elbo = TraceEnum_ELBO()
        elbo = Trace_ELBO()
        optimizer = pyro.optim.Adam({"lr": self.lr})
        self._curr_step = 0

        svi = SVI(self.model_mixture, self.guide_mixture, optimizer, loss=elbo)
        loss = float(svi.step())

        train_params_each = 2 if self.n_steps <= 100 else int(self.n_steps / 100)
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        self.losses, self.regs, self.likelihoods, self.train_params = list(), list(), list(), list()
        t = trange(self.n_steps, desc="Bar desc", leave=True)
        for i in t:   # inference - do gradient steps
            self._curr_step = i
            loss = float(svi.step())
            self.losses.append(loss)

            self.likelihoods.append(self._likelihood_mixture(to_cpu=False, detach=False).clone().detach().sum())

            if self.store_parameters and i%train_params_each==0:
                self.train_params.append(self.get_param_dict(convert=True, 
                                                             to_cpu=False,
                                                             detach=True))

            if i%5 == 0:
                t.set_description("ELBO %f" % loss)
                t.refresh()

        self.alpha = self._to_cpu(self.alpha)

        self.gradient_norms = dict(gradient_norms) if gradient_norms is not None else {}
        self.params = {**self._set_clusters(to_cpu=True), 
                       **self.get_param_dict(convert=True, to_cpu=True, 
                                             permute=False, detach=True)}

        self.init_params = self._get_init_param_dict(convert=True, to_cpu=True)
        self.bic = self.aic = self.icl = self.reg_likelihood = None
        self.likelihood = self._likelihood_mixture(to_cpu=True, detach=True).sum().item()
        self.set_scores()


    def get_param_dict(self, convert=False, permute=False, to_cpu=True, detach=True):
        params = dict()
        params["alpha_prior"] = self._get_param("alpha_prior_param", convert=convert, 
        # params["alpha_prior"] = self._get_param("AutoDelta.alpha_t", convert=convert, 
                                                permute=permute, to_cpu=to_cpu,
                                                detach=detach)
        params["pi"] = self._get_param("pi_param", convert=convert, to_cpu=to_cpu,
        # params["pi"] = self._get_param("AutoDelta.pi", convert=convert, to_cpu=to_cpu,
                                       detach=detach)
        # params["scale_factor_alpha"] = self._get_param("scale_factor_alpha_param", convert=convert, to_cpu=to_cpu)
        # params["scale_factor_centroid"] = self._get_param("scale_factor_centroid_param", convert=convert, to_cpu=to_cpu)

        # params["pi_conc0"] = self._get_param("pi_conc0_param", convert=convert, to_cpu=to_cpu)
        return params


    def _get_init_param_dict(self, convert=True, to_cpu=True):
        params = dict()
        for k, v in self.init_params.items():
            if not convert: params[k] = v
            else:
                params[k] = self._to_cpu(v, move=to_cpu)
                if len(v.shape) > 2:
                    params[k] = torch.permute(params[k], (1,0,2))
                    params[k] = self._concat_tensors(params[k], dim=1)
                params[k] = np.array(params[k])
        return params


    def set_scores(self):
        self._set_bic()
        self._set_aic()
        self._set_icl()


    def _get_param(self, param_name, to_cpu=True, permute=False, convert=False, detach=True):
        try:
            # for k, v in pyro.get_param_store().items(): print(k, v)
            par = pyro.param(param_name)
            par = self._to_cpu(par, move=to_cpu)
            if isinstance(par, torch.Tensor) and detach: par = par.clone().detach()
            if permute: par = torch.permute(par, (1,0,2))

            if par is not None and convert:
                par = self._to_cpu(par, move=True)
                if len(par.shape) > 2: par = self._concat_tensors(par, dim=1)
                par = np.array(par)

            return par

        except Exception as e: return None


    def _convert(self, par, convert=True):
        if not convert: return par
        if len(par.shape) > 2: par = self._concat_tensors(par, dim=1)
        par = np.array(par)
        return par


    def _likelihood_mixture(self, to_cpu=False, params=None, detach=True):
        alpha = self._to_cpu(self.alpha, move=to_cpu)
        if params is None: params = self.get_param_dict(convert=False, permute=True, 
                                                        to_cpu=to_cpu, detach=detach)

        assert params["alpha_prior"].shape == (self.cluster, self.n_variants, self.K)
        assert alpha.shape == (self.n_samples, self.n_variants, self.K)

        llik = torch.zeros(self.cluster, self.n_samples)
        for g in range(self.cluster):
            lprob_alpha = torch.zeros((self.n_variants, self.n_samples))
            for v in range(self.n_variants):
                alpha_prior_g = params["alpha_prior"][g, v]
                lprob_alpha[v] = dist.Dirichlet(alpha_prior_g).log_prob(alpha[:,v,:])

            assert lprob_alpha.shape == (self.n_variants, self.n_samples)

            # sum over independent variants and add log(pi)
            llik[g, :] = torch.sum(lprob_alpha, dim=0) + torch.log(params["pi"][g])

        assert llik.shape == (self.cluster, self.n_samples)

        return llik


    def _compute_posterior_probs(self, to_cpu=True, compute_exp=True):
        ll_k = self._likelihood_mixture(to_cpu=to_cpu, detach=True)
        ll = self._logsumexp(ll_k).unsqueeze(0)

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
        m = torch.amax(weighted_lp, dim=0).unsqueeze(0)  # the maximum value for each observation among the K values
        summed_lk = m[-1] + torch.log(torch.sum(torch.exp(weighted_lp - m), axis=0))
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
            par_i = param[i]
            if par_i.shape[1] < max_shape:
                pad_dims = max_shape - par_i.shape[1]
                columns = list(str(i)+"_"+par_i.columns) + [str(i)+"_P"+str(j) for j in range(pad_dims)]
                index = par_i.index
                values = F.pad(input=self._to_cpu(torch.tensor(par_i.values)), pad=(0, pad_dims, 0,0), mode="constant", value=torch.finfo().tiny)
                new_param.append(pd.DataFrame(values.numpy(), index=index, columns=columns))
            else:
                new_param.append(par_i.copy(deep=True))
                new_param[i].columns = list(str(i)+"_"+par_i.columns)
        return new_param


    def convert_to_dataframe(self, alpha):
        # mutations catalogue
        if isinstance(alpha, list): alpha = [self._to_cpu(i, move=True) for i in alpha]
        if not isinstance(alpha, list): alpha = [self._to_cpu(alpha, move=True)]

        self.alpha = pd.concat(self._pad_dataframes(alpha), axis=1)
        sample_names, sigs = list(self.alpha.index), list(self.alpha.columns)

        if self.groups is not None: self.groups = np.array(self.groups)

        self.params = self._convert_pars(param_dict=self.params, sigs=sigs, sample_names=sample_names)
        if len(self.train_params) > 0:
            for i, v in enumerate(self.train_params):
                self.train_params[i] = self._convert_pars(param_dict=v, sigs=sigs,
                                                          sample_names=sample_names)

        for k, v in self.hyperparameters.items():
            if v is None: continue
            v = self._to_cpu(v, move=True)
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 0: self.hyperparameters[k] = int(v)
                else: self.hyperparameters[k] = v.numpy()

        self._set_init_params(sigs)


    def _convert_pars(self, param_dict, sigs, sample_names):
        for parname, par in param_dict.items():
            if par is None: continue
            par = self._to_cpu(par, move=True)
            if parname == "pi":
                param_dict[parname] = par.tolist() if isinstance(par, torch.Tensor) else par
            elif (parname == "alpha_prior" or parname == "alpha_prior_unn"):
                param_dict[parname] = pd.DataFrame(np.array(self._concat_tensors(par, dim=1)), index=range(self.n_groups), columns=sigs)
            elif parname == "post_probs" and isinstance(par, torch.Tensor):
                param_dict[parname] = pd.DataFrame(np.array(torch.transpose(par, dim0=1, dim1=0)), index=sample_names , columns=range(self.n_groups))
        return param_dict



    def _set_init_params(self, sigs):
        # return
        for k, v_tmp in self.init_params.items():
            v = self._to_cpu(v_tmp, move=True)
            if v is None: continue

            if k == "alpha_t":
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



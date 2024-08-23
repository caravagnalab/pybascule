import torch
import pyro
import numpy as np
import pandas as pd
import pyro.distributions.constraints as constraints
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from collections import defaultdict
from tqdm import trange


MIN_POS_N = torch.finfo(torch.float32).tiny


class PyBascule():
    def __init__(
        self,
        x,
        k_denovo,
        n_steps,
        lr = 0.001,
        optim_gamma = 0.1,
        enumer = "parallel",
        hyperparameters = None,
        dirichlet_prior = True,
        beta_fixed = None,
        compile_model = True,
        CUDA = False,
        enforce_sparsity = True,
        store_parameters = False,

        stage = "", 
        seed = 10
        ):

        self._set_data_catalogue(x)
        self._hyperpars_default = {"alpha_sigma":0.1, "alpha_p_conc0":0.6, "alpha_conc":1., 
                                   "omega_conc":10., "eps_sigma":10, "pi_conc0":0.6, 
                                   "penalty_scale":0}
        self._set_fit_settings(enforce_sparsity=enforce_sparsity, lr=lr, optim_gamma=optim_gamma, n_steps=n_steps,
                               compile_model=compile_model, CUDA=CUDA, store_parameters=store_parameters, 
                               stage=stage, seed=seed)

        self._set_k_denovo(k_denovo)  # set self.k_denovo
        self._set_beta_fixed(beta_fixed)  # set self.beta_fixed and self.k_fixed
        self._fix_zero_denovo_null_reference()
        self.K = self.k_denovo + self.k_fixed
        self.K_alpha = self.K

        self._set_hyperparams(enumer=enumer, hyperparameters=hyperparameters)

        self._init_seed = None


    def _set_fit_settings(self, enforce_sparsity, lr, optim_gamma, n_steps, 
                          compile_model, CUDA, store_parameters, stage, seed):
        self.enforce_sparsity = enforce_sparsity
        self.lr = lr
        self.optim_gamma = optim_gamma
        self.n_steps = int(n_steps)
        self.compile_model = compile_model
        self.CUDA = CUDA

        self.store_parameters = store_parameters
        self.stage = stage

        self.seed = seed


    def _set_hyperparams(self, enumer, hyperparameters):
        self.enumer = enumer
        self.init_params = None

        if hyperparameters is None:
            self.hyperparameters = self._hyperpars_default
        else:
            self.hyperparameters = dict()
            for parname, value in self._hyperpars_default.items():
                self.hyperparameters[parname] = hyperparameters.get(parname, value)

        self.hyperparameters["omega_conc"] = torch.cat((torch.ones(self.k_denovo, self.k_fixed) * self.hyperparameters["omega_conc"], 
                                                        torch.ones(self.k_denovo, 1)), dim=1)

        alpha_conc = self.hyperparameters["alpha_conc"]
        if self.k_fixed > 0:
            if (isinstance(alpha_conc, list)):
                if len(alpha_conc) != self.k_fixed:
                    raise "The length of `alpha_conc` list must be the same as the number of fixed signatures."
                
                self.hyperparameters["alpha_conc"] = torch.cat((torch.tensor(alpha_conc, dtype=torch.float64), 
                                                                torch.ones(self.k_denovo, dtype=torch.float64)))

            elif isinstance(alpha_conc, int) or isinstance(alpha_conc, float):
                # if given as input, the fixed signatures will have a user-defined concentration
                self.hyperparameters["alpha_conc"] = torch.cat((alpha_conc * torch.ones(self.k_fixed, dtype=torch.float64),
                                                                torch.ones(self.k_denovo, dtype=torch.float64)))
            else:
                raise "`alpha_conc` must be either a list or a number."
        
        else:
            self.hyperparameters["alpha_conc"] = self._hyperpars_default["alpha_conc"] * torch.ones(self.k_denovo, dtype=torch.float64)

        for k, v in self.hyperparameters.items():
            self.hyperparameters[k] = self._to_gpu(v)


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
            self.beta_fixed, self.k_fixed, self.fixed_names = None, 0, []
            return

        self.alpha_conc = None
        self.fixed_names = list(beta_fixed.index) if isinstance(beta_fixed, pd.DataFrame) else ["F"+str(f+1) for f in range(beta_fixed.shape[0])]

        if isinstance(beta_fixed, pd.DataFrame):
            beta_fixed = beta_fixed.values

        self.beta_fixed = torch.tensor(beta_fixed, dtype=torch.float64)
        if len(self.beta_fixed.shape) == 1: self.beta_fixed = self.beta_fixed.reshape(1, self.beta_fixed.shape[0])

        self.k_fixed = beta_fixed.shape[0]
        if self.k_fixed > 0: self._fix_zero_contexts()

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


    def _set_k_denovo(self, k_denovo):
        if isinstance(k_denovo, int):
            self.k_denovo = k_denovo
        else:
            raise Exception("Invalid k_denovo value, expected integer!")


    def model(self):
        n_samples, n_contexts = self.n_samples, self.contexts

        if self._noise_only: alpha = self._to_gpu(torch.zeros(self.n_samples, 1, dtype=torch.float64))
        # Flat model
        if not self._noise_only:
            with pyro.plate("n", self.n_samples):
                alpha = pyro.sample("latent_exposure", dist.Dirichlet(self.hyperparameters["alpha_conc"]))

        epsilon = None
        if self.stage == "random_noise":
            with pyro.plate("contexts3", self.contexts):  # columns
                with pyro.plate("n3", n_samples):  # rows
                    epsilon = pyro.sample("latent_m", dist.HalfNormal(self.hyperparameters["eps_sigma"]))

        # Beta
        beta_denovo = None
        if self.k_denovo > 0:
            ## directly learn the beta_denovo
            beta_weight = None
            beta_conc = self._get_beta_centroid(eps=1e-15, power=1)
            with pyro.plate("k_denovo", self.k_denovo):  # rows
                beta_denovo = pyro.sample("latent_signatures", dist.Dirichlet(beta_conc))

            beta = self._get_unique_beta(beta_fixed=self.beta_fixed, 
                                         beta_denovo=beta_denovo, 
                                         beta_weights=beta_weight)

        else: 
            beta = self._get_unique_beta(beta_fixed=self.beta_fixed,
                                         beta_denovo=beta_denovo)

        a = torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)
        if self.stage == "random_noise": a = a + epsilon
        with pyro.plate("contexts", n_contexts):
            with pyro.plate("n2", n_samples):
                pyro.sample("obs", dist.Poisson(a), obs=self.x)

        if self.k_denovo > 0:
            alpha_recomputed = self._get_alpha_stick_breaking(alpha_star=alpha, beta_weights=beta_weight)
            beta_fixed_cum = self._compute_cum_beta_fixed(self.beta_fixed)
            penalty = self._compute_penalty(alpha=alpha_recomputed, beta_fixed_cum=beta_fixed_cum, beta_denovo=beta_denovo)
            pyro.factor("penalty", penalty)


    def _get_beta_centroid(self, eps=1e-10, power=2):
        beta_fixed_cum = self._compute_cum_beta_fixed(self.beta_fixed)
        max_val = beta_fixed_cum.max()
        centroid = (max_val - beta_fixed_cum + eps) ** power
        return centroid * 100


    def _compute_cum_beta_fixed(self, beta_fixed):
        if self.beta_fixed is not None:
            return torch.sum(beta_fixed, dim=0)
        return dist.Dirichlet(torch.ones(self.contexts).double()).sample()


    def _compute_penalty(self, alpha, beta_fixed_cum, beta_denovo):
        alpha_fixed = torch.sum(alpha[:,:self.k_fixed], dim=0).unsqueeze(1)
        alpha_denovo = torch.sum(alpha[:,self.k_fixed:], dim=0).unsqueeze(1)

        w_fixed = torch.sum(beta_fixed_cum * alpha_fixed, dim=0)
        w_denovo = torch.sum(beta_denovo * alpha_denovo, dim=0)

        w_difference = torch.abs(w_fixed - w_denovo)
        res = torch.sum(w_difference)

        return res * self.hyperparameters["penalty_scale"]


    def guide(self):
        n_samples, k_denovo = self.n_samples, self.k_denovo
        init_params = self._initialize_params()

        if not self._noise_only:
            with pyro.plate("n", n_samples):
                alpha = pyro.param("alpha", init_params["alpha"], constraint=constraints.simplex)
                pyro.sample("latent_exposure", dist.Delta(alpha).to_event(1))

        # Epsilon 
        if self.stage == "random_noise":
            eps_sigma = pyro.param("lambda_epsilon", init_params["epsilon_var"], constraint=constraints.greater_than(MIN_POS_N))

            with pyro.plate("contexts3", self.contexts):
                with pyro.plate("n3", n_samples):
                    pyro.sample("latent_m", dist.HalfNormal(eps_sigma))

        # Beta
        if k_denovo > 0:
            beta_param = pyro.param("beta_denovo", lambda: init_params["beta_dn_param"], constraint=constraints.simplex)
            with pyro.plate("k_denovo", k_denovo):
                pyro.sample("latent_signatures", dist.Delta(beta_param).to_event(1))


    def _initialize_params_nonhier(self):
        pi = alpha_prior = alpha = epsilon = beta_dn = beta_weights = pi_conc0 = None

        eps_sigma = self.hyperparameters["eps_sigma"]
        pi_conc0 = dist.Gamma(0.001, 0.001).sample([self.k_fixed])
        alpha = dist.Dirichlet(self.hyperparameters["alpha_conc"]).sample((self.n_samples,))

        if self.stage == "random_noise":
            epsilon = dist.HalfNormal(torch.ones(self.n_samples, self.contexts, dtype=torch.float64) * eps_sigma).sample()

        if self.k_denovo > 0:
            beta_conc = self._get_beta_centroid(eps=1e-3, power=1)
            beta_dn = dist.Dirichlet(beta_conc).sample((self.k_denovo,))
            omega_conc = self.hyperparameters["omega_conc"][0]
            # beta_weights = dist.Dirichlet(omega_conc).sample((self.k_denovo,))

        params = self._create_init_params_dict(pi=pi, alpha_prior=alpha_prior, alpha=alpha, epsilon=epsilon, 
                                               beta_dn=beta_dn, beta_weights=beta_weights, pi_conc0=pi_conc0)
        return params


    def _create_init_params_dict(self, pi=None, alpha_prior=None, alpha=None, epsilon=None, 
                                 beta_dn=None, beta_weights=None, pi_conc0=None):
        params = dict()

        if pi is not None: params["pi_param"] = pi
        if alpha_prior is not None: params["alpha_prior_param"] = alpha_prior
        if alpha is not None: params["alpha"] = alpha
        if epsilon is not None: params["epsilon_var"] = epsilon
        if beta_dn is not None: params["beta_dn_param"] = beta_dn
        if beta_weights is not None: params["beta_weights"] = beta_weights
        if pi_conc0 is not None: params["pi_conc0_param"] = pi_conc0

        return params


    def _initialize_params(self):
        if self.init_params is None:
            self.init_params = self._initialize_params_nonhier()

        return self.init_params


    def _get_unique_beta(self, beta_fixed, beta_denovo, beta_weights=None, convert=False):
        if beta_weights is not None:
            return self._get_unique_beta_stick_breaking(beta_fixed, beta_denovo, beta_weights, convert=convert)
        if beta_fixed is None: return beta_denovo
        if beta_denovo is None or self._noise_only: return beta_fixed

        return torch.cat((beta_fixed, beta_denovo), axis=0)


    def _get_unique_beta_stick_breaking(self, beta_fixed, beta_denovo, beta_weights, convert=False):
        beta = torch.zeros(beta_weights.shape[0], self.contexts, dtype=torch.float64)

        for j in range(beta_weights.shape[0]):
            for r in range(self.k_fixed):
                beta[j] += beta_weights[j,r] * beta_fixed[r]
                beta[j] += beta_denovo[j]

        if not convert: return beta
        return np.array(beta)


    def _get_alpha_stick_breaking(self, alpha_star, beta_weights, convert=False):
        if beta_weights is None: 
            if not convert: return alpha_star
            return np.array(alpha_star)
        
        alpha = torch.zeros((self.n_samples, self.K))

        if not isinstance(alpha_star, torch.Tensor): alpha_star = torch.tensor(alpha_star)
        if not isinstance(beta_weights, torch.Tensor): beta_weights = torch.tensor(beta_weights)

        for n in range(self.n_samples):
            alpha_n = alpha_star[n,:]
            for j in range(beta_weights.shape[1]-1):
                beta_w_j = beta_weights[:,j]
                alpha[n,j] = torch.sum(alpha_n * beta_w_j)
            
            alpha[n,j+1:] = beta_weights[:,-1] * alpha_n
        
        if not convert: return self._norm_and_clamp(alpha)
        return np.array(self._norm_and_clamp(alpha))


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
        params["beta_w"] = self._get_param("beta_weights", convert=convert, to_cpu=to_cpu, normalize=False)
        params["pi_conc0"] = self._get_param("pi_conc0", normalize=False, convert=convert, to_cpu=to_cpu)

        if self.stage == "random_noise":
            params["lambda_epsilon"] = self._get_param("lambda_epsilon", normalize=False, convert=convert, to_cpu=to_cpu)

        return params


    def _fit(self, set_attributes=True):
        pyro.clear_param_store()  # always clear the store before the inference

        self.x = self._to_gpu(self.x)
        self.beta_fixed = self._to_gpu(self.beta_fixed)

        if self.CUDA and torch.cuda.is_available(): 
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type(t=torch.FloatTensor)

        if self.compile_model and not self.CUDA: elbo = JitTrace_ELBO()
        else: elbo = Trace_ELBO()

        lrd = self.optim_gamma ** (1 / self.n_steps)
        optimizer = pyro.optim.ClippedAdam({"lr": self.lr, "lrd":lrd, "clip_norm":1e10})

        pyro.set_rng_seed(self.seed)
        pyro.get_param_store().clear()

        self._initialize_params()

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)
        loss = float(svi.step())

        train_params_each = 2 if self.n_steps <= 100 else int(self.n_steps / 100)
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        self.losses, self.regs, self.likelihoods, self.train_params = list(), list(), list(), list()
        t = trange(self.n_steps, desc="Bar desc", leave=True)
        for i in t:  # inference - do gradient steps
            loss = float(svi.step())
            self.losses.append(loss)

            self.regs.append(self.compute_regularization(beta_denovo=self._get_param("beta_denovo", to_cpu=False), to_cpu=False))
            self.likelihoods.append(self._likelihood(to_cpu=False))

            if self.store_parameters and i%train_params_each==0: 
                self.train_params.append(self.get_param_dict(to_cpu=True, convert=True, all=False))

            if i%5 == 0:
                t.set_description("ELBO %f" % loss)
                t.refresh()

        if set_attributes is False: return

        self.x = self._to_cpu(self.x)
        self.beta_fixed = self._to_cpu(self.beta_fixed)

        self.gradient_norms = dict(gradient_norms) if gradient_norms is not None else None
        self._set_params()
        self.likelihood = self._likelihood(to_cpu=True)
        self.likelihood_reg = self._likelihood(to_cpu=True) + self.compute_regularization(self._get_param("beta_denovo", to_cpu=True), to_cpu=True)

        self.set_scores()


    def _likelihood(self, to_cpu=False):
        llik = self._likelihood_flat(to_cpu=to_cpu)
        return llik.sum().item()


    def _likelihood_flat(self, to_cpu=False):
        params = self.get_param_dict(convert=False, to_cpu=to_cpu)
        beta = self._get_unique_beta(beta_fixed=params["beta_f"], beta_denovo=params["beta_d"],
                                     beta_weights=params["beta_w"])
        M = self._to_cpu(self.x, move=to_cpu)

        alpha_hat = torch.sum(M, axis=1).unsqueeze(1) * params["alpha"]
        rate = torch.matmul(alpha_hat, beta)

        if self.stage == "random_noise": rate += dist.HalfNormal(params["lambda_epsilon"]).sample()
        llik = dist.Poisson(rate).log_prob(M)

        return llik


    def _norm_and_clamp(self, par):
        mmin = 0
        if torch.any(par < 0): mmin = torch.min(par, dim=-1)[0].unsqueeze(-1)

        nnum = par - mmin
        par = nnum / torch.sum(nnum, dim=-1).unsqueeze(-1)

        return par


    def _set_params(self):
        self.params = self.get_param_dict(convert=True, all=True)


    def set_scores(self):
        self._set_bic()
        self._set_aic()
        self.icl = None


    def compute_regularization(self, beta_denovo, to_cpu=False):
        if beta_denovo is None: return 0
        if not isinstance(beta_denovo, torch.Tensor): beta_denovo = torch.tensor(beta_denovo)
        beta_fixed_cum = self._compute_cum_beta_fixed(self.beta_fixed)
        alpha_star = self._get_param("alpha", normalize=True, to_cpu=to_cpu)
        beta_weights = self._get_param("beta_weights", to_cpu=to_cpu)
        alpha = self._get_alpha_stick_breaking(alpha_star=alpha_star, beta_weights=beta_weights)
        return self._compute_penalty(alpha=alpha, beta_fixed_cum=beta_fixed_cum, beta_denovo=beta_denovo).item()


    def _set_bic(self):
        _log_like = self.likelihood_reg
        k = self._number_of_params() 
        bic = k * torch.log(torch.tensor(self.n_samples, dtype=torch.float64)) - (2 * _log_like)
        self.bic = bic.item()


    def _set_aic(self):
        _log_like = self.likelihood_reg
        k = self._number_of_params() 
        aic = 2*k - 2*_log_like

        if (isinstance(aic, torch.Tensor)): self.aic = aic.item()
        else: self.aic = aic


    def _compute_entropy(self) -> np.array:
        '''
        `entropy(z) = - sum^K( sum^N( z_probs_nk * log(z_probs_nk) ) )`
        `entropy(z) = - sum^K( sum^N( exp(log(z_probs_nk)) * log(z_probs_nk) ) )`
        '''

        matrix = self._compute_denovo_cosine()
        entr = 0
        for n in range(self.n_samples):
            for k in range(self.cluster):
                entr += torch.exp(matrix[k,n]) * matrix[k,n]
        return -entr.detach()


    def _number_of_params(self):
        k = 0
        if self.k_denovo == 0 and self._noise_only: k = 0
        else: k += self.k_denovo * self.contexts # beta denovo

        if self.stage == "random_noise":
            k += self.params["lambda_epsilon"].shape[0] * self.params["lambda_epsilon"].shape[1]  # random noise

        if not self._noise_only: k += self.n_samples * self.K  # alpha if no noise is learned

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

        alpha_columns = fixed_names + denovo_names

        # if self.K_alpha == self.k_denovo: alpha_columns = denovo_names
        # else: alpha_columns = fixed_names + denovo_names

        if self.beta_fixed is not None and isinstance(self.beta_fixed, torch.Tensor) and torch.sum(self.beta_fixed) > 0:
            self.beta_fixed = pd.DataFrame(self.beta_fixed, index=fixed_names, columns=contexts)

        self.params = self._convert_pars(param_dict=self.params, sample_names=sample_names, 
                                         fixed_names=fixed_names, denovo_names=denovo_names, 
                                         contexts=contexts)

        if len(self.train_params) > 0:
            for i,v in enumerate(self.train_params):
                self.train_params[i] = self._convert_pars(param_dict=v, sample_names=sample_names, 
                                                          fixed_names=fixed_names, denovo_names=denovo_names, 
                                                          contexts=contexts)

        for k, v in self.hyperparameters.items():
            if v is None: continue
            v = self._to_cpu(v, move=True)
            if isinstance(v, torch.Tensor): 
                if len(v.shape) == 0: self.hyperparameters[k] = int(v)
                else: self.hyperparameters[k] = v.numpy()

        self._set_init_params(sample_names=sample_names, fixed_names=fixed_names, 
                              denovo_names=denovo_names, contexts=contexts, 
                              alpha_columns=alpha_columns)


    def _convert_pars(self, param_dict, sample_names, fixed_names, denovo_names, contexts):
        for parname, par in param_dict.items():
            if par is None: continue
            par = self._to_cpu(par, move=True)
            if parname == "alpha": 
                param_dict[parname] = pd.DataFrame(np.array(par), index=sample_names, columns=fixed_names+denovo_names)
            elif parname == "beta_d" or parname == "beta_star": 
                param_dict[parname] = pd.DataFrame(np.array(par), index=denovo_names, columns=contexts)
            elif parname == "beta_f": 
                param_dict[parname] = self.beta_fixed
            elif parname == "beta_w":
                param_dict[parname] = pd.DataFrame(np.array(par), index=denovo_names, columns=fixed_names+["DN"])
            elif parname == "pi": 
                param_dict[parname] = par.tolist() if isinstance(par, torch.Tensor) else par
            elif parname == "lambda_epsilon":
                param_dict[parname] = pd.DataFrame(np.array(par), index=sample_names, columns=contexts)
            elif (parname == "alpha_prior" or parname == "alpha_prior_unn"): 
                param_dict[parname] = pd.DataFrame(np.array(par), index=range(self.n_groups), columns=fixed_names+denovo_names)
            elif parname == "alpha_star": 
                param_dict[parname] = pd.DataFrame(np.array(par), index=sample_names, columns=denovo_names)
            elif parname == "post_probs" and isinstance(par, torch.Tensor):
                param_dict[parname] = pd.DataFrame(np.array(torch.transpose(par, dim0=1, dim1=0)), index=sample_names , columns=range(self.n_groups))
        return param_dict


    def _set_init_params(self, sample_names, fixed_names, denovo_names, contexts, alpha_columns):
        # return
        for k, v_tmp in self.init_params.items():
            v = self._to_cpu(v_tmp, move=True)
            if v is None: continue

            if k == "alpha":
                self.init_params[k] = pd.DataFrame(np.array(v), index=sample_names, columns=alpha_columns)
            elif k == "beta_dn_param":
                self.init_params[k] = pd.DataFrame(np.array(v), index=denovo_names, columns=contexts)
            elif k == "alpha_prior_param":
                self.init_params[k] = pd.DataFrame(np.array(v), index=range(self.n_groups), columns=fixed_names+denovo_names)
            elif k == "beta_weights":
                self.init_params[k] = pd.DataFrame(np.array(v), index=denovo_names, columns=fixed_names+["DN"])
            elif k == "epsilon_var":
                self.init_params[k] = pd.DataFrame(np.array(v), index=sample_names, columns=contexts)
            else:
                self.init_params[k] = np.array(v)


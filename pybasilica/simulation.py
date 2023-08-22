import random
import pyro
import pyro.distributions as dist
import numpy as np
import pandas as pd
import torch

def norm_and_clamp(par):
    mmin = 0
    if torch.any(par < 0): mmin = torch.min(par, dim=-1)[0].unsqueeze(-1)

    nnum = par - mmin
    par = nnum / torch.sum(nnum, dim=-1).unsqueeze(-1)

    return par


def to_cpu(param, move=True):
    if param is None: return None
    if move and torch.cuda.is_available() and isinstance(param, torch.Tensor):
        return param.cpu()
    return param


def generate_model(alpha_prior, beta, n_muts, N, alpha_sigma=0.05, seed=15, use_normal=True):
    input_beta = beta
    if isinstance(beta, pd.DataFrame):
        beta = torch.tensor(beta.values)
    alpha_prior = torch.tensor(alpha_prior, dtype=torch.float64)
    beta = torch.tensor(beta, dtype=torch.float64)
    n_muts = torch.tensor(n_muts, dtype=torch.float64)

    pyro.set_rng_seed(seed)

    alpha_prior = norm_and_clamp(alpha_prior)
    beta = norm_and_clamp(beta)

    q_01 = alpha_prior - alpha_sigma # * alpha_prior
    q_99 = alpha_prior + alpha_sigma # * alpha_prior
    alpha_sigma_corr = (q_99 - q_01) / (2 * dist.Normal(alpha_prior, 1).icdf(torch.tensor(0.99)))  # good clustering

    # Observations
    with pyro.plate("n2", N):
        if use_normal:
            alpha  = pyro.sample("latent_exposure", dist.Normal(alpha_prior, alpha_sigma_corr).to_event(1))  # good clustering
        else:
            alpha  = pyro.sample("latent_exposure", dist.Cauchy(alpha_prior, alpha_sigma_corr).to_event(1))  # good clustering

        alpha = norm_and_clamp(alpha)

        a = torch.matmul(torch.matmul(torch.diag(n_muts), alpha), beta)
        data = pyro.sample("obs", dist.Poisson(a).to_event(1))

    return {"data":pd.DataFrame(to_cpu(data, move=True), columns=input_beta.columns), 
            "alpha":pd.DataFrame(to_cpu(alpha, move=True), columns=input_beta.index), 
            "alpha_sigma":to_cpu(alpha_sigma_corr, move=True), 
            "beta":input_beta}




# #-----------------------------------------------------------------[<QC-PASSED>]
# # divide cosmic signatures into two parts:
# # 1st part: considered as cosmic signatures
# # 2nd part: considered as denovo signatures

# def cosmic_denovo(full_cosmic_df):

#     #random.seed(a=seed)

#     # cosmic_path --- <class 'str'>
#     #full_cosmic_df = pd.read_csv(cosmic_path, index_col=0)
#     full_cosmic_list = list(full_cosmic_df.index)

#     cosmic_list = random.sample(full_cosmic_list, k=50)
#     denovo_list = list(set(full_cosmic_list) - set(cosmic_list))

#     cosmic_df = full_cosmic_df.loc[cosmic_list] # <class 'pandas.core.frame.DataFrame'>
#     denovo_df = full_cosmic_df.loc[denovo_list] # <class 'pandas.core.frame.DataFrame'>

#     return cosmic_df, denovo_df
#     # cosmic_df --- dtype: DataFrame
#     # denovo_df --- dtype: DataFrame

# #-----------------------------------------------------------------[QC:PASSED]

# def target_generator(cosmic_df, denovo_df, target_complexity, num_samples):
#     # profile ------- {"A", "B", "C"}
#     # cosmic_df ----- <class 'pandas.core.frame.DataFrame'>
#     # denovo_df ----- <class 'pandas.core.frame.DataFrame'>

#     #random.seed(a=seed)

#     # num_samples = random.randint(15, 25)
#     num_samples = int(num_samples)

#     # error handling for profile argument
#     valid = ["low", "medium", "high"]
#     if target_complexity not in valid:
#         raise ValueError("profile must be one of %s." % valid)

#     if target_complexity=="low":
#         fixed_num = random.randint(3, 5)    # <class 'int'>
#         denovo_num = random.randint(0, 2)
#     elif target_complexity=="medium":
#         fixed_num = random.randint(0, 2)
#         denovo_num = random.randint(3, 5)
#     elif target_complexity=="high":
#         fixed_num = random.randint(3, 5)
#         denovo_num = random.randint(3, 5)

#     cosmic_list = list(cosmic_df.index)
#     denovo_list = list(denovo_df.index)
#     mutation_features = list(cosmic_df.columns)

#     # beta fixed ------------------------------------------------------
#     if fixed_num > 0:
#         fixed_list = random.sample(cosmic_list, k=fixed_num)
#         beta_fixed_df = cosmic_df.loc[fixed_list]
#     else:
#         beta_fixed_df = pd.DataFrame(columns=mutation_features)
#         #beta_fixed_df = None
    
#     # beta denovo -----------------------------------------------------
#     if denovo_num > 0:
#         denovo_list = random.sample(denovo_list, k=denovo_num)
#         beta_denovo_df = denovo_df.loc[denovo_list]

#         # add "_D" to the end of denovo signatures to distinguish them from fixed signatures
#         denovo_labels = []
#         for i in range(len(denovo_list)):
#             denovo_labels.append(denovo_list[i] + '_D')
#         beta_denovo_df.index = denovo_labels
#     else:
#         beta_denovo_df = pd.DataFrame(columns=mutation_features)
#         #beta_denovo_df = None

#     if beta_denovo_df.empty:
#         beta_df = beta_fixed_df
#     elif beta_fixed_df.empty:
#         beta_df = beta_denovo_df
#     else:
#         beta_df = pd.concat([beta_fixed_df, beta_denovo_df], axis=0)

#     signatures = list(beta_df.index)
#     beta_tensor = torch.tensor(beta_df.values).float()

#     #------- alpha ----------------------------------------------
#     matrix = np.random.rand(num_samples, len(signatures))
#     alpha_tensor = torch.tensor(matrix / matrix.sum(axis=1)[:, None]).float()
#     alpha_np = np.array(alpha_tensor)
#     alpha_df = pd.DataFrame(alpha_np, columns=signatures)

#     #------- theta ----------------------------------------------
#     theta = random.sample(range(1000, 4000), k=num_samples) # dtype:list
    
#     #------- check dimensions -----------------------------------
#     m_alpha = alpha_tensor.size()[0]    # no. of branches (alpha)
#     m_theta = len(theta)                # no. of branches (theta)
#     k_alpha = alpha_tensor.size()[1]    # no. of signatures (alpha)
#     k_beta = beta_tensor.size()[0]      # no. of signatures (beta)
#     if not(m_alpha == m_theta and k_alpha == k_beta):
#         raise ValueError("wrong dimensions!")
    
#     M_tensor = torch.zeros([num_samples, 96])   # initialize mutational catalogue with zeros

#     for i in range(num_samples):
#         p = alpha_tensor[i]         # selecting branch i
#         for k in range(theta[i]):   # iterate for number of the mutations in branch i

#             # sample signature profile index from categorical data
#             b = beta_tensor[dist.Categorical(p).sample().item()]

#             # sample mutation feature index for corresponding signature from categorical data
#             j = dist.Categorical(b).sample().item()

#             # add +1 to the mutation feature in position j in branch i
#             M_tensor[i, j] += 1

#     # phylogeny
#     M_np = np.array(M_tensor)
#     M_df = pd.DataFrame(M_np, columns=mutation_features)
#     M_df = M_df.astype(int)

#     # return all in dataframe format
#     return M_df, alpha_df, beta_fixed_df, beta_denovo_df
#     # M_df ------------- dtype: dataframe
#     # alpha_df --------- dtype: dataframe
#     # beta_fixed_df ---- dtype: dataframe, could be empty (if empty --> beta_denovo_df != empty)
#     # beta_denovo_df --- dtype: dataframe, could be empty (if empty --> beta_fixed_df != empty)

# #-----------------------------------------------------------------[QC:PASSED]
# def input_catalogue_generator(cosmic_df, beta_fixed_df, beta_denovo_df, input_complexity):
#     # profile ---------- {"X", "Y", "Z"}
#     # beta_fixed_df ---- dtype: dataframe
#     # beta_denovo_df --- dtype: dataframe
#     # cosmic_df -------- dtype: dataframe

#     #random.seed(a=seed)

#     # error handling for profile argument
#     valid = ["low", "medium", "high"]
#     if input_complexity not in valid:
#         raise ValueError("profile must be one of %s." % valid)

#     # TARGET DATA -----------------------------------------------------------------------------
#     beta_fixed_names = list(beta_fixed_df.index)    # target beta signatures names (dtype: list)
#     k_fixed_target = len(beta_fixed_names)
#     beta_denovo_names = list(beta_denovo_df.index)  # target beta signatures names (dtype: list)
#     k_denovo_target = len(beta_denovo_names)

#     # common fixed signatures (target intersect test)
#     # different fixed signatures (test minus target)
#     if input_complexity=="low":
#         if k_fixed_target > 0:
#             k_overlap = random.randint(1, k_fixed_target)
#             k_extra = 0
#         else:
#             k_overlap = 0
#             k_extra = random.randint(1, k_denovo_target)

#     elif input_complexity=="medium":
#         if k_fixed_target > 0:
#             k_overlap = random.randint(1, k_fixed_target)
#             k_extra = random.randint(1, k_fixed_target)
#         else:
#             k_overlap = 0
#             k_extra = random.randint(1, k_denovo_target)

#     elif input_complexity=="high":
#         k_overlap = 0
#         k_extra = random.randint(1, k_fixed_target + k_denovo_target)
    
    
#     cosmic_names = list(cosmic_df.index)    # cosmic signatures names (dtype: list)

#     # exclude beta target fixed signatures from cosmic
#     for signature in beta_fixed_names:
#         cosmic_names.remove(signature)
    
#     # common fixed signatures list
#     if k_overlap > 0:
#         overlap_sigs = random.sample(beta_fixed_names, k=k_overlap)
#     else:
#         overlap_sigs = []
    
#     # different fixed signatures list
#     if k_extra > 0:
#         extra_sigs = random.sample(cosmic_names, k=k_extra)
#     else:
#         extra_sigs = []
    
#     beta_input = cosmic_df.loc[overlap_sigs + extra_sigs]

#     return beta_input   # dtype: dataframe


# #-----------------------------------------------------------------[PASSED]
# def input_generator(full_cosmic_df, target_complexity, input_complexity, num_samples):

#     cosmic_df, denovo_df = cosmic_denovo(full_cosmic_df)
    
#     M_df, alpha_df, beta_fixed_df, beta_denovo_df = target_generator(cosmic_df, denovo_df, target_complexity, num_samples)
    
#     beta_input_df = input_catalogue_generator(cosmic_df, beta_fixed_df, beta_denovo_df, input_complexity)
    
#     #beta_df = pd.concat([beta_fixed_df, beta_denovo_df], axis=0)

#     if beta_fixed_df.empty:
#         beta_fixed_df = None

#     if beta_denovo_df.empty:
#         beta_denovo_df = None

#     if beta_input_df.empty:
#         beta_input_df = None

#     data = {
#         "M" : M_df,                         # dataframe
#         "alpha" : alpha_df,                 # dataframe
#         "beta_fixed" : beta_fixed_df,       # dataframe
#         "beta_denovo" : beta_denovo_df,     # dataframe
#         "beta_input" : beta_input_df,       # dataframe
#         "cosmic_df" : cosmic_df             # dataframe
#     }

#     return data



from random import random
import torch
import numpy as np
import pandas as pd
import random


from pybasilica.utilities import fixedFilter
from pybasilica.utilities import denovoFilter
from pybasilica.utilities import stopRun
from pybasilica.utilities import initialize_params
from pybasilica.run import multi_k_run

'''
from utilities import fixedFilter
from utilities import denovoFilter
from utilities import stopRun
from utilities import initialize_params
from run import multi_k_run
'''




def pyfit(M, groups, input_catalogue, reference_catalogue, k, lr, steps, phi, delta):
    # M --------------------- dataframe
    # groups ---------------- list
    # input_catalogue ------- dataframe
    # reference_catalogue --- dataframe
    # k --------------------- list
    # lr -------------------- float
    # steps ----------------- integer
    # phi ------------------- float
    # delta ----------------- float

    theta = np.sum(M.values, axis=1)

    params = initialize_params(M, groups, input_catalogue, lr, steps)

    counter = 1
    while True:

        #print("iteration:", counter)

        # k ------- dtype: list
        k_inf, A_inf, B_inf = multi_k_run(params, k)
        # k_inf --- dtype: int
        # A_inf --- dtype: torch.Tensor
        # B_inf --- dtype: torch.Tensor

        # A_inf ----- dtype: torch.Tensor 
        # B_input --- dtype: data.frame
        # theta ----- dtype: numpy
        input_catalogue_sub = fixedFilter(A_inf, input_catalogue, theta, phi)
        # B_input_sub ---- dtype: list
        
        # B_inf -------- dtype: torch.Tensor
        # cosmic_df ---- dtype: dataframe
        input_catalogue_new = denovoFilter(B_inf, reference_catalogue, delta)
        # B_input_new --- dtype: list

        if input_catalogue is None:
            input_catalogue_list = []
        else:
            input_catalogue_list = list(input_catalogue.index)

        if stopRun(input_catalogue_sub, input_catalogue_list, input_catalogue_new):
            signatures_inf = []
            for p in range(k_inf):
                p = int(p)
                signatures_inf.append("D"+str(p+1))
            signatures = input_catalogue_list + signatures_inf
            mutation_features = list(M.columns)

            # alpha
            A_inf_np = np.array(A_inf)
            A_inf_df = pd.DataFrame(A_inf_np, columns=signatures)   # dataframe

            # beta
            if B_inf is None:
                #B_inf_denovo_df = pd.DataFrame(columns=mutation_features)
                B_inf_denovo_df = None
            else:
                B_inf_denovo_np = np.array(B_inf)
                B_inf_denovo_df = pd.DataFrame(B_inf_denovo_np, index=signatures_inf, columns=mutation_features)
            
            B_inf_fixed_df = input_catalogue    # dataframe | None

            return A_inf_df, B_inf_fixed_df, B_inf_denovo_df
            # A_inf_df ---------- dtype: dataframe
            # B_inf_fixed_df ---- dtype: dataframe
            # B_inf_denovo_df --- dtype: dataframe

        #print("input_catalogue_list", input_catalogue_list, "\n", "input_catalogue_sub", input_catalogue_sub, "\n", "input_catalogue_new", input_catalogue_new, "\n")

        if len(input_catalogue_sub + input_catalogue_new)==0:
            input_catalogue = None
            params["beta_fixed"] = None
        else:
            input_catalogue = reference_catalogue.loc[input_catalogue_sub + input_catalogue_new]  # dtype: dataframe
            params["beta_fixed"] = torch.tensor(input_catalogue.values).float()
        
        counter += 1


# import pybasilica
from pybasilica import run
from pybasilica import svi
import pandas as pd

file = "pybasilica/data/degasperi/counts_all.tsv"
data = pd.read_csv(file, index_col=0, sep='\t')

#filter data
data = data.iloc[0:10]
#make categories for groups
data.organ = pd.Categorical(data.organ)
data['groups'] = data.organ.cat.codes
groups = data.groups.to_list()
groups = [0,0,0,1,1,1,0,0,2,2]
#mutation values
values = data.drop(['cohort', 'organ', 'groups'], axis=1)

# #without groups
# r = pybasilica.fit(values, k_list=10)
# r.beta_denovo

#with groups
obj = svi.PyBasilica(values, k_denovo=2, groups=groups, lr=0.05, n_steps=500)
obj.model()
obj.guide()
r2 = run.fit(values, k_list=2, groups=groups)
# print(r2.groups)
# print("AFTER RUN")
# r2.beta_denovo



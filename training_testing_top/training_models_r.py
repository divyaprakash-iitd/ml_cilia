from ast import Num
from numpy import dtype, matlib
from dataclasses import dataclass
from locale import nl_langinfo
import numpy as np
import pandas as pd
from pyparsing import nested_expr
from regex import W
from sklearn.metrics import nan_euclidean_distances
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torch import dropout, nn, norm, normal
import lstm_models as lm

device = 'cpu'
csvpath = 'train_data/'
# Training parameters
params = lm.hparams(n_sequence       = 3,
                    num_hidden_units = 60,
                    batch_size       = 2,
                    learning_rate    = 1e-4,
                    n_epochs         = 1000,
                    weight_decay     = 0.0000)

# Read the x and y positions of the cilia as pytorch tensors
cx       = torch.from_numpy(pd.read_csv(csvpath+'cx.csv', header=None).to_numpy(dtype=np.float32))
cy       = torch.from_numpy(pd.read_csv(csvpath+'cy.csv', header=None).to_numpy(dtype=np.float32))
xloc     = torch.from_numpy(pd.read_csv(csvpath+'px.csv', header=None).to_numpy(dtype=np.float32))
arvals   = torch.from_numpy(pd.read_csv(csvpath+'r.csv', header=None).to_numpy(dtype=np.float32))
avals    = torch.from_numpy(pd.read_csv(csvpath+'a.csv', header=None).to_numpy(dtype=np.float32))
areavals = np.pi*avals**2 / arvals


# The following changes for each simulation
# Define features and target
features    = torch.cat((cx,cy),1).to(device)
target      = arvals.to(device)
fname       = "ar_"
lm.run_training(features,target,xloc,params,fname)

#features    = cy
#target      = arvals
#fname       = "ar_y_"
#lm.run_training(features,target,xloc,params,fname)

#features    = cx
#target      = avals*1e4
#fname       = "area_x_"
#lm.run_training(features,target,xloc,params,fname)
#
#features    = cy
#target      = areavals
#fname       = "area_y_"
#lm.run_training(features,target,xloc,params,fname)

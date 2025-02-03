import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import lstm_models_working as lm
from sklearn.metrics import r2_score

plt.rcParams['font.size'] = 24
H = 1.4e-3
#csvpath='ml_unseen_data/data_top/'
#impath='images/ml_unseen/top_unseen_'
csvpath='testing_data_top/'
impath='images/top_uniform_'

# Read in data files
cx       = torch.from_numpy(pd.read_csv(csvpath+'ucx.csv', header=None).to_numpy(dtype=np.float32))
cy       = torch.from_numpy(pd.read_csv(csvpath+'ucy.csv', header=None).to_numpy(dtype=np.float32))
xloc     = torch.from_numpy(pd.read_csv(csvpath+'upx.csv', header=None).to_numpy(dtype=np.float32))
arvals   = torch.from_numpy(pd.read_csv(csvpath+'ur.csv', header=None).to_numpy(dtype=np.float32))
avals    = torch.from_numpy(pd.read_csv(csvpath+'ua.csv', header=None).to_numpy(dtype=np.float32))

theta = np.arctan2(cy,cx)

# Predict particle's aspect ratio
#fname       = 'ar_x_'
fname       = 'a_'
features    = lm.normalize_data(torch.cat((cx,cy),1))
#features    = lm.normalize_data(theta)
#target      = arvals
target      = avals*1e4

# Load the saved model
model   = torch.load(fname+'lstm')
model.eval()

# Make datasets of the unseen data
n_sequence      = 3 # This n_sequence has to be same as the model that was trained
batch_size      = 2
total_dataset   = lm.CiliaDataset(features, target, xloc, n_sequence=n_sequence)
val_dataset     = torch.utils.data.Subset(total_dataset, range(len(total_dataset)))
eval_loader     = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# eval_loader = torch.load(fname+'eval_loader')
preds, act, locx  = lm.predict(eval_loader, model)

err = abs((preds - act)/abs(act))
mape = 1.0/len(err)*err.sum()
mape = mape*100
r2 = r2_score(act,preds)
fig, ax = plt.subplots()
# ax.plot(act,preds,'.')

# Conversion to 2a/H values
act = 2*act/(H*1e4)
preds = 2*preds/(H*1e4)

p1vals = np.unique(act.round(decimals=3))
boxvals = [preds[act.round(decimals=3)==p1vals[i]] for i in range(p1vals.shape[0])]

# plt.boxplot(boxvals,positions=p1vals,widths=0.1,showfliers=False)
plt.boxplot(boxvals,positions=p1vals,widths=0.028,showfliers=False)
# plt.plot(act,preds,'.')
plt.plot([p1vals[0],p1vals[-1]],[p1vals[0],p1vals[-1]])
# plt.title(f"Semi-Major Axis Prediction: \n MAPE = {mape.item():.2f}%")
plt.title(f"MAPE = {mape.item():.2f}%, $R^2$ = {r2:.2f}")

plt.xticks(p1vals)
plt.yticks(p1vals)
# plt.xlim([0.9,3.1])
# plt.ylim([0.9,3.1])
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()

plt.xlim(ax.get_ylim())

#ct = np.round(np.linspace(ymin, ymax, 5), 1)
#plt.gca().set_yticks(ct)
#plt.gca().set_yticklabels(ct)
#plt.gca().set_xticks(ct)
#plt.gca().set_xticklabels(ct)
# plt.xlabel('Ground Truth')
# plt.ylabel('Prediction')
plt.xlabel(r'Ground Truth, $2a/H$')
plt.ylabel('Prediction')
plt.gcf().set_size_inches(9,6)
#plt.savefig('ar_prediction.png',dpi=300)
plt.tight_layout()

plt.savefig(impath+'a_prediction_2ah.pdf')

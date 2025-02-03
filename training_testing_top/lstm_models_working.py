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
from matplotlib.ticker import ScalarFormatter

device = 'cpu'

class hparams: 
    def __init__(self,n_sequence,num_hidden_units,batch_size,learning_rate,n_epochs,weight_decay):
        self.n_sequence         = n_sequence
        self.num_hidden_units   = num_hidden_units
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.n_epochs           = n_epochs
        self.weight_decay       = weight_decay

# Normalize the dataset
def normalize_data(data):
    mean, std = data.mean(), data.std()
    return (data-mean)/std

def inverse_transform(data, mean, std):
    return data*std + mean

# Implement a custom dataset
class CiliaDataset(Dataset):
    def __init__(self, cdata, target, loc, n_sequence=5):
        # n_sequence > 0
        self.target     = target.flatten()
        self.cdata      = cdata
        self.loc        = loc
        self.n_sequence = n_sequence
        self.cdata     = torch.split(self.cdata,self.n_sequence)
        self.target    = torch.split(self.target,self.n_sequence)
        self.loc      = torch.split(self.loc,self.n_sequence)
        
    def __len__(self):
        #return len(self.cdata) - self.n_sequence - 1
        return len(self.cdata)
    
    def __getitem__(self, idx):
        X       = self.cdata[idx]
        y       = self.target[idx].mean()
        z       = self.loc[idx].mean()
        #X = self.cdata[idx:idx+self.n_sequence,:]
        #y = self.target[idx:idx+self.n_sequence].mean()
        #z = self.loc[idx:idx+self.n_sequence].mean()
        return X, y, z

# Define the custom LSTM model
class LSTMModel(nn.Module):
    def __init__(self, n_cilia, hidden_units):
        super().__init__()
        self.n_cilia = n_cilia
        self.hidden_units = hidden_units
        self.num_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=n_cilia,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout = 0.3
        )
        
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[self.num_layers-1]).flatten()
        
        return out
        

# Define function to train the model
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y, _ in data_loader:
        output = model(X)
        loss = loss_function(output, y.flatten())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss
    
# Define function to test the model
def test_model(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for X, y, _ in data_loader:
            output = model(X)
            total_loss += loss_function(output, y.flatten()).item()
            
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss
   
# Define function to make predictions from the model
def predict(data_loader, model):
    output = torch.tensor([])
    actual = torch.tensor([])
    la     = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, y, locx in data_loader:
            y_star = model(X)
            # Transform back to original
            #y_star = inverse_transform(y_star,mean,std)
            #y = inverse_transform(y,mean,std)
            output = torch.cat((output, y_star),0)
            actual = torch.cat((actual, y),0)
            la     = torch.cat((la, locx),0)
    return output, actual, la 

# Define a function to partition the data and return dataloaders
def create_dataloaders(features,target,xloc,batch_size,n_sequence): 
    total_dataset = CiliaDataset(features, target, xloc, n_sequence=n_sequence)

    # Partition the data into training and test set
    train_size  = int(0.6 * len(total_dataset))
    val_size    = int(0.3 * len(total_dataset))
    test_size   = len(total_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [train_size, val_size, test_size])

    train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_loader     = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, eval_loader, train_dataset, val_dataset, test_dataset


def train_lstm(features, target, xloc, params):
    # Pre-processing the data
    n_cil           = features.shape[1]

    # Normalize the data
    features = normalize_data(features)
    #target   = normalize_data(target)

    # Generate data loaders for training, testing and evaluation(prediction)
    train_loader, test_loader, eval_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(features, target, xloc, params.batch_size, params.n_sequence)

    # Create the model and select the loss function and optimizer
    model           = LSTMModel(n_cilia=n_cil, hidden_units=params.num_hidden_units).to(device)
    loss_function   = nn.MSELoss()
    optimizer       = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)        

    # Define early stopping parameters
    patience = 30  # Number of epochs with no improvement to wait
    best_val_loss = float('inf')
    counter = 0

    # Train the model
    t_loss = []
    v_loss = []
    for ix_epoch in range(params.n_epochs):
        print(f"Epoch {ix_epoch}\n----------")
        avg_loss = train_model(train_loader, model, loss_function, optimizer=optimizer)
        t_loss.append(avg_loss)
        avg_loss = test_model(test_loader, model, loss_function)
        v_loss.append(avg_loss)
        # Check for early stopping
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss # Here avg_loss corresponds to validation loss (lin # 178)
            counter = 0
        else:
            counter += 1
        # Early stopping condition
        if counter >= patience:
            print("Early stopping!")
            break
        print()

    # eval_loader = DataLoader(val_dataset, batch_size=1,shuffle=False)
    return model, eval_loader, val_dataset, t_loss, v_loss

def plot_graphs(act,preds,pval,boxval,fname):
    #plt.axline([act[0],act[0]],[act[-1],act[-1]])
    plt.axline([act.min().item(),act.min().item()], [act.max().item(), act.max().item()])
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.grid()
    plt.savefig('plot_func.png')
    plt.tight_layout()

    err = abs((preds - act)/abs(act))
    mape = 1.0/len(err)*err.sum()
    print(f"MAPE = {mape*100} %")

    # ar = np.sort(bval[0:-1:750*5].numpy().squeeze())
    dx = pval.mean()/len(pval)
    # print(dx)
    pp = preds.numpy()
    # boxval = [pp[np.isclose(act.numpy(),pval[i])] for i in range(5)]
    plt.boxplot(boxval,positions=pval,flierprops={'markeredgecolor': 'blue'}, showfliers=False, widths=dx)
    # plt.boxplot(boxval,positions=ar,flierprops={'markeredgecolor': 'blue'}, showfliers=False, widths=5e-8)
    plt.gca().set_xticks(pval)
    plt.gca().set_yticks(pval)
    plt.xlim([pval[0]-dx, pval[-1]+dx])
    plt.ylim([pval[0]-dx, pval[-1]+dx])
   
    # Define formatter for tick labels 
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Adjust this to control when scientific notation is used
    
    
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title(fname.strip('_')+'\n ' + f" (MAPE = {mape*100:.2f}%)")
    plt.gca().grid(b=True)
    plt.tight_layout()
    plt.savefig('images/'+fname+'plot.png')
    plt.close()
    
def run_training(features,target,xloc,params,fname):
    # call the training function
    model, eval_loader, eval_dataset, t_loss, v_loss = train_lstm(features, target, xloc, params)

    # Save model and data
    torch.save(model,fname+'lstm')
    torch.save(eval_loader,fname+'eval_loader')
    torch.save(eval_dataset,fname+'eval_dataset')
    np.savetxt(fname+"training_loss.txt", np.array(t_loss))
    np.savetxt(fname+"validation_loss.txt", np.array(v_loss))

    # Generate plots
    plt.semilogy(t_loss)
    plt.semilogy(v_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('both')
    plt.savefig(fname+'loss.png')
    plt.close()

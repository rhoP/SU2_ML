#! /usr/bin/env python3


import configparser
import ast
from decimal import Decimal
import torch
import torch.nn as nn
import torch.nn.functional as Fu
import torch.optim as optim
# data handling, transforms and models
from matplotlib import pyplot as plt

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from matplotlib import cm


# Base settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Import all functions from the auxiliary files
# The main function is defined here. It takes the inputs from the function call arguments.
# This will make it easier to track changes rather than change everything each time.
# So we can just make one script and duplicate work directories as necessary.
#
# INPUTS
# batch size
# optimizer
# model
# learning rate
# train, test split
# maximum iterations
# loss
# input features
#
# DATA LOADING and PRE PROCESSING
# Important inputs are features, split, bs;
# DataHandling.py will contain all necessary functions to carry out these jobs.
# Calls from main to this function typically returns data-loaders with some batch size.
#
# MODEL SELECTION
# inputs used are: optimizer, model, learning rate, loss
# If we have several models or optimizers, we deploy the ones selected with the featuers
# requested. Also contains auxiliary functions such as early stopping functions etc., which
# need not be available as options. Calls from the main function will return the model, with
# optimizer and loss function but it also contains the fuctions for calculating parameter updates etc.
#
# FIT FUNCTION
# Calls data handlers to get data; then calls the model selector to get model and opt.
# Then calls the main loop which runs the iterations to the specified number.
# Has data structures to save information of import.
#
#
#



def train_loop(model, loss_func, opt, train_dl, valid_dl, device, max_iter=200):
    iterations_counter = 0
    loss_val = []
    loss_train = []
    # liveloss = PlotLosses()

    gl = early_stop_up(loss_val, )
    try:
        while not gl and iterations_counter < max_iter:
            # logs = {}
            model.train()
            tr_losses = []
            tr_nums = []
            for xb, yb in train_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                # xb = xb.view(xb.shape[0], -1)
                tr_losses, tr_nums = loss_batch(model, loss_func, xb, yb, opt)
            # sch.step(val_loss)

            avg_train_loss = np.sum(np.multiply(tr_losses, tr_nums)) / np.sum(tr_nums)
            loss_train.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                for xb, yb in valid_dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    # xb = xb.view(xb.shape[0], -1)
                    losses, nums = zip(*[loss_batch(model, loss_func, xb, yb)])

            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            loss_val.append(val_loss)
            # sch.step(val_loss)
            gl = early_stop_up(loss_val)

            if loss_train.index(min(loss_train)) == iterations_counter:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                }, './checkpoint_tr.pt')

            if loss_val.index(min(loss_val)) == iterations_counter:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                }, './checkpoint_va.pt')

            if iterations_counter % 20 == 1:
                print('\nEpoch: {}, Avg. loss: {:.2E}, Min. loss: {:.2E}\n'.format(
                    iterations_counter, Decimal(val_loss), Decimal(min(loss_val))))
            iterations_counter += 1

        # logs['training loss'] = avg_train_loss
        # logs['validation loss'] = val_loss

        # liveloss.update(logs)
        # liveloss.send()
    except KeyboardInterrupt:
        pass

    if iterations_counter == max_iter:
        print('reached maximum iterations limit.')
        print('Minimum Error: ', min(loss_val))

    return loss_train, loss_val



# Define the model class

class Cnn(nn.Module):
    def __init__(self, num_features, kernel, forward_width):
        super(Cnn, self).__init__()
        self.channels = num_features
        self.fwd_width = forward_width
        self._kernel = kernel
        self._act = nn.ReLU()

        self.con1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=self._kernel)
        self.pool1 = nn.MaxPool2d(2)
        self. BN1 = nn.BatchNorm2d(self.channels*2, affine=False)
        self.con2 = nn.Conv2d(in_channels=2*self.channels, out_channels=self.channels, kernel_size=self._kernel)
        self.pool2 = nn.MaxPool2d(2)
        self. BN2 = nn.BatchNorm2d(self.channels, affine=False)
        # self.con3 = nn.Conv2d(in_channels=self.channels, out_channels=int(self.channels/2), kernel_size=self._kernel)
        self.fc1 = nn.Linear(self.channels * self._kernel * self._kernel, self.fwd_width)
        self.fc2 = nn.Linear(self.fwd_width, self.fwd_width)
        self.fc3 = nn.Linear(self.fwd_width, self.fwd_width)
        self.fc4 = nn.Linear(self.fwd_width, 1, bias=False)

    def forward(self, x_in):
        x_in = x_in.float()
        x = self.BN1(self.pool1(self._act(self.con1(x_in))))
        x = self.BN2(self.pool2(self._act(self.con2(x))))
        # x = self.pool2(self._act(self.con3(x)))
        x = x.view(-1, self.channels * self._kernel * self._kernel)
        x = self._act(self.fc1(x))
        x = self._act(self.fc2(x))
        x = self._act(self.fc3(x))
        x = self.fc4(x)
        return x


def get_model(mdl, opt, loss, nf, kernel, forward_width, lr=0.001, alpha=0, wd=0):
    # if mdl == 'Cnn':
    model = Cnn(num_features=nf, kernel=kernel, forward_width=forward_width)

    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                         betas=(0.9, 0.999), eps=1e-08,
                         weight_decay=wd, amsgrad=True
                         )
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)

    loss_func = nn.MSELoss()

    return model, optimizer, loss_func


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb.view(-1, 1))
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def early_stop_gl(loss_list):
    if len(loss_list) < 2:
        return False
    else:
        e_opt = min(loss_list)
        if e_opt != 0.0:
            gl = 100.0 * (loss_list[-1] / e_opt - 1.0)
            if gl > 50.0:
                return True
            else:
                return True


def early_stop_up(loss_list, k=2):
    if not loss_list:
        return False
    else:
        e_opt = (loss_list[-1])
        if k > 4:
            e_opt_k = (loss_list[-k])
            e_opt_twok = (loss_list[-2 * k])
            if e_opt > e_opt_k > e_opt_twok:
                return True
            else:
                return False
        else:
            return False






# Load data into the respective containers
def load_data(spl):
    df = np.load('../../Data/pics.npy', allow_pickle=True).astype(np.single)
    dfb = np.load('../../Data/pics_beta.npy', allow_pickle=True).astype(np.single)
    # We select the features by "column" because the data is already a tensor
    df = df[:, spl]
    # Reshape the data
    data_shape = df.shape
    df = np.reshape(df, (data_shape[0], data_shape[1], 20, 20), order='F')
    return df, dfb


# Make the training validation split
def split_data(df, dfb, test_size, random_state=42):
    x_train, x_test, y_train, y_test = \
        train_test_split(df, dfb, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


# Map the data into torch types
# and return the data loader
def dfs_to_dls(x, y, batch_size, n_workers=24, p_m=True):
    x, y = map(torch.tensor, (x, y))
    # print(x.shape, x.dtype)
    tr = TensorDataset(x, y)
    tr = DataLoader(tr, batch_size=batch_size, num_workers=n_workers, pin_memory=p_m)
    return tr


# All the above in a neat little package
def preprocess(split, test_size, random_state, batch_size, num_workers,
               pin_memory):
    df, dfb = load_data(split)

    x_train, x_test, y_train, y_test = \
        split_data(df, dfb, test_size=test_size, random_state=random_state)

    train_dl = dfs_to_dls(x_train, y_train,
                          batch_size=batch_size, n_workers=num_workers, p_m=pin_memory)

    valid_dl = dfs_to_dls(x_test, y_test,
                          batch_size=batch_size, n_workers=num_workers, p_m=pin_memory)

    return train_dl, valid_dl


# Generate a picture from one instance of the data
def generate_picture(xc, yc, df):
    gx = np.linspace(xc - 0.025 / 2, xc + 0.025 / 2, num=20)
    gy = np.linspace(yc - 0.025 / 2, yc + 0.025 / 2, num=20)
    xp, yp = np.meshgrid(gx, gy, sparse=False, indexing='ij')

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(xp, yp, df, cmap=cm.jet)
    ax.set_rasterization_zorder(-10)
    fig.colorbar(cp)
    ax.set_title('Predicted Pic')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.set_size_inches(18.5, 10.5)
    plt.show()















def main():
    # Read the configuration file and evaluate the pertinent options
    config = configparser.ConfigParser()
    config.read('trial.conf')
    spl = ast.literal_eval(config['DATA']['feature_selection'])
    test_size = ast.literal_eval(config['DATA']['split_size'])
    random_state = ast.literal_eval(config['DATA']['random_state'])
    num_workers = ast.literal_eval(config['DATA']['num_workers'])
    pin_memory = ast.literal_eval(config['DATA']['pin_memory'])
    batch_size = ast.literal_eval(config['DATA']['batch_size'])
    learning_rate = ast.literal_eval(config['MODEL']['learning_rate'])
    model = config['MODEL']['model']
    kernel = ast.literal_eval(config['MODEL']['kernel'])
    forward_width = ast.literal_eval(config['MODEL']['forward_width'])
    optimization = config['MODEL']['optimization']
    loss = config['MODEL']['loss_function']

    max_iterations = ast.literal_eval(config['TRAINING']['max_iterations'])

    # Data loaders
    train_dl, valid_dl = preprocess(split=spl, test_size=test_size,
                                    random_state=random_state, batch_size=batch_size,
                                    num_workers=num_workers, pin_memory=pin_memory)

    # Load the model and optimization

    mdl, opt, loss_func = get_model(model, optimization, loss, len(spl), kernel, forward_width, learning_rate)

    mdl.to(device)

    lmse, lmsecv = train_loop(mdl, loss_func, opt,
                              train_dl, valid_dl, device, max_iter=max_iterations)

    np.savetxt('./lmse.txt', lmse)
    np.savetxt('./lmsecv.txt', lmsecv)


if __name__ == '__main__':
    main()

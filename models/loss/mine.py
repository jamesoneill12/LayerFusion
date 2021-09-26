import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import bokeh
import holoviews as hv
import matplotlib.pyplot as plt
hv.extension('bokeh')



class Net(nn.Module):

    def __init__(self, H):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(1, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2


def func(x):
    return x


def gen_x(data_size):
    return np.sign(np.random.normal(0.,1.,[data_size,1]))


def gen_y(x, var, data_size):
    return func(x)+np.random.normal(0.,np.sqrt(var),[data_size,1])


def mutual_info(var, data_size):
    x = gen_x(data_size)
    y = gen_y(x, var, data_size)
    p_y_x = np.exp(-(y - x) ** 2 / (2 * var))
    p_y_x_minus = np.exp(-(y + 1) ** 2 / (2 * var))
    p_y_x_plus = np.exp(-(y - 1) ** 2 / (2 * var))
    mi = np.average(np.log(p_y_x / (0.5 * p_y_x_minus + 0.5 * p_y_x_plus)))
    return mi


def run_mine(H = 10, n_epoch = 500, data_size = 20000):

    model = Net(H)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []

    for epoch in tqdm(range(n_epoch)):
        x_sample = gen_x(data_size)
        y_sample = gen_y(x_sample, var, data_size)
        y_shuffle = np.random.permutation(y_sample)

        x_sample = Variable(torch.from_numpy(x_sample).type(torch.FloatTensor), requires_grad=True)
        y_sample = Variable(torch.from_numpy(y_sample).type(torch.FloatTensor), requires_grad=True)
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad=True)

        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        plot_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return plot_loss


def plot_results(plot_loss):
    plot_x = np.arange(len(plot_loss))
    plot_y = np.array(plot_loss).reshape(-1, )
    hv.Curve((plot_x, -plot_y)) * hv.Curve((plot_x, mi))
    plt.plot(plot_x, -plot_y)
    plt.show()


if __name__ == "__main__":

    # data
    var = 0.2
    data_size = 1000000

    mi = mutual_info(var, data_size)
    loss = run_mine()
    print(loss)

    plot_results(loss)
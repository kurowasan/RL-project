import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def plot_training(l_a2b, l_b2a, output='', savefig=False):
    plt.plot(moving_average(np.mean(l_a2b, axis=0)), label='a->b')
    plt.plot(moving_average(np.mean(l_b2a, axis=0)), label='b->a')
    plt.title('Likelihood during training for each model')
    plt.xlabel('Nb episodes')
    plt.ylabel('Likelihood P(D| model)')
    plt.legend()
    if savefig:
        path = os.path.join(output, 'training.png')
        plt.savefig(path)
    else:
        plt.show()

def plot_adaptation(l_a2b, l_b2a, output='', savefig=False):
    plt.plot(moving_average(np.mean(l_a2b, axis=0)), label='a->b')
    plt.plot(moving_average(np.mean(l_b2a, axis=0)), label='b->a')
    plt.title('Likelihood change after adaptation')
    plt.xlabel('Nb episodes')
    plt.ylabel('Likelihood P(D| model)')
    plt.legend()
    if savefig:
        path = os.path.join(output, 'adaptation.png')
        plt.savefig(path)
    else:
        plt.show()


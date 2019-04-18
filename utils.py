import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def random_walk_from_every_start(env):
    nb_state_visited = np.zeros((env.state_dim, env.state_dim))

    for i in range(env.state_dim):
        for j in range(env.state_dim):
            observed_states = random_walk(env, i, j)
            nb_state_visited[i, j] = len(set(observed_states))

    return nb_state_visited

def random_walk(self, env, old_s1=None, old_s2=None):
    if old_s1 is None and old_s2 is None:
        old_s1, old_s2 = env.reset()
    observed_states = []
    done = False
    env.reset()
    while not done:
        a = env.sample_action_uniformly()
        (s1, s2), reward, done, _ = env.step(a, old_s1, old_s2)
        observed_states.append(s1 + s2*env.state_dim)
        old_s1, old_s2 = s1, s2
    return observed_states

def test_ergodicity(env, hparam):
    env.max_step = 1000
    state_visited = random_walk_from_every_start(env)
    print(f'nb states:{env.state_dim**2}')
    print(state_visited)
    plt.imshow(state_visited, cmap='hot', interpolation='nearest')
    plt.show()

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def plot_graph(l_a2b, l_b2a, output, fname, savefig=False, title='', std=False):
    curve_a2b = moving_average(np.mean(l_a2b, axis=0))
    curve_b2a = moving_average(np.mean(l_b2a, axis=0))

    plt.figure()
    plt.plot(curve_a2b, label='a->b', color='blue')
    plt.plot(curve_b2a, label='b->a', color='red')
    if std:
        std_a2b = np.std(l_a2b, axis=0)[:len(curve_a2b)]
        std_b2a = np.std(l_b2a, axis=0)[:len(curve_b2a)]
        x = np.arange(len(curve_b2a))
        plt.fill_between(x, curve_a2b+std_a2b, curve_a2b-std_a2b,
                         facecolor='blue', alpha=0.2)
        plt.fill_between(x, curve_b2a+std_b2a, curve_b2a-std_b2a,
                         facecolor='red', alpha=0.2)

    plt.title(title)
    plt.xlabel('Nb episodes')
    plt.ylabel('Likelihood P(D| model)')
    plt.legend()

    if savefig:
        img_path = os.path.join(output, f'{fname}.png')
        plt.savefig(img_path)
    else:
        plt.show()

def plot_graph_reward(r_a2b, r_b2a, output, fname, savefig=False, title='', std=False):
    curve_a2b = np.mean(r_a2b, axis=0)
    curve_b2a = np.mean(r_b2a, axis=0)

    plt.figure()
    plt.plot(curve_a2b, label='a->b', color='blue')
    plt.plot(curve_b2a, label='b->a', color='red')
    plt.title(title)
    plt.xlabel('Nb episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()

    if savefig:
        img_path = os.path.join(output, f'{fname}.png')
        plt.savefig(img_path)
    else:
        plt.show()


def save_raw_data(l_a2b, l_b2a, output, fname):
    txt_path = os.path.join(output, f'{fname}_a2b.txt')
    np.savetxt(txt_path, l_a2b)
    txt_path = os.path.join(output, f'{fname}_b2a.txt')
    np.savetxt(txt_path, l_b2a)

def plot_training(l_a2b, l_b2a, output='', savefig=False):
    title = 'Likelihood during training for each model'
    fname = 'training'
    plot_graph(l_a2b, l_b2a, output, fname, savefig, title)
    save_raw_data(l_a2b, l_b2a, output, fname)

def plot_adaptation(l_a2b, l_b2a, output='', savefig=False):
    title = 'Likelihood change after adaptation'
    fname = 'adaptation'
    plot_graph(l_a2b, l_b2a, output, fname, savefig, title)
    save_raw_data(l_a2b, l_b2a, output, fname)

def plot_reward(r_a2b, r_b2a, output='', savefig=False):
    title = 'Reward during training'
    fname = 'reward_training'
    plot_graph_reward(r_a2b, r_b2a, output, fname, savefig, title)
    save_raw_data(r_a2b, r_b2a, output, fname)

def plot_reward_adapt(r_a2b, r_b2a, output='', savefig=False):
    title = 'Reward change after adaptation'
    fname = 'reward_adapt'
    plot_graph_reward(r_a2b, r_b2a, output, fname, savefig, title)
    save_raw_data(r_a2b, r_b2a, output, fname)

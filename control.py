import numpy as np
import environment
import model
import utils
from data import DataManager
from random import shuffle
import copy

def train_with_buffer(env, likelihood, nb_episode):
    s = environment.State()
    nb_total_step = nb_episode * env.max_step
    buffer = []
    l_a2b = np.zeros(nb_total_step)
    l_b2a = np.zeros(nb_total_step)

    # fill buffer
    for episode in range(nb_episode):
        done = False
        old_s1, old_s2 = env.reset()
        while not done:
            a = env.sample_action_uniformly()
            (s1, s2), reward, done, _ = env.step(a, old_s1, old_s2)
            s.set_state(old_s1, old_s2, s1, s2, a)
            buffer.append(copy.copy(s))
            old_s1, old_s2 = s1, s2

    shuffle(buffer)

    # train
    for i in range(nb_total_step):
        s = buffer[i]
        l1, l2 = likelihood.update(s)
        real_likelihood = env.get_likelihood(s.old_s1, s.old_s2, s.s1, s.s2, s.a)
        l_a2b[i] = np.abs(real_likelihood - l1.detach().numpy())
        l_b2a[i] = np.abs(real_likelihood - l2.detach().numpy())

        # l_a2b[i] = l1.detach().numpy()
        # l_b2a[i] = l2.detach().numpy()

    return l_a2b, l_b2a

def train(env, likelihood, nb_episode):
    s = environment.State()
    l_a2b = np.zeros(nb_episode)
    l_b2a = np.zeros(nb_episode)

    for episode in range(nb_episode):
        done = False
        old_s1, old_s2 = env.reset()
        while not done:
            a = env.sample_action_uniformly()
            (s1, s2), reward, done, _ = env.step(a, old_s1, old_s2)
            s.set_state(old_s1, old_s2, s1, s2, a)
            l1, l2 = likelihood.update(s)
            real_likelihood = env.get_likelihood(old_s1, old_s2, s1, s2, a)
            l_a2b[episode] += np.abs(real_likelihood - l1.detach().numpy())
            l_b2a[episode] += np.abs(real_likelihood - l2.detach().numpy())
            old_s1, old_s2 = s1, s2

        l_a2b[episode] /= 100
        l_b2a[episode] /= 100
    return l_a2b, l_b2a

class DynaQ:
    def __init__(self, model, likelihood, env):
        self.model = model
        self.likelihood = likelihood
        self.env = env
        self.q_learning = TDLearning(env)
        self.reset_observation()

    def reset(self):
        self.q_learning = TDLearning(self.env)
        self.reset_observation()
        # self.model.reset()

    def update_observation(self, s):
        self.observation[s.s1, s.s2, s.a] = 1

    def reset_observation(self):
        self.observation = np.zeros((self.env.state_dim,
                                     self.env.state_dim,
                                     self.env.action_dim))

    def sample_observation(self):
        state_observed = np.sum(self.observation, axis=2)
        idx = np.nonzero(state_observed)
        choice = np.random.randint(len(idx[0]))
        s1 = idx[0][choice]
        s2 = idx[1][choice]
        idx = np.nonzero(self.observation[s1, s2])
        choice = np.random.randint(len(idx[0]))
        a = int(self.observation[s1, s2, idx[0][choice]])
        return s1, s2, a

    def train(self, nb_episode, nb_simulation):
        s = environment.State()
        self.reset_observation()
        l_list = np.zeros(nb_episode)
        reward_list = np.zeros(nb_episode)

        for episode in range(nb_episode):
            if episode % 10000 == 0 and episode > 0:
                print(episode)
                # __import__('ipdb').set_trace()
                print(self.env.s1[0,0,0,:])
                print(utils.softmax(self.likelihood.model.cause.w[:,0,0,0].detach().numpy()))
            done = False
            old_s1, old_s2 = self.env.reset()
            while not done:
                a = self.q_learning.sample_action(old_s1, old_s2)
                (s1, s2), reward, done, _ = self.env.step(a, old_s1, old_s2)
                s.set_state(old_s1, old_s2, s1, s2, a)
                self.q_learning.update(s, reward)

                l_list[episode] = self.likelihood.update(s)
                reward_list[episode] += reward

                self.update_observation(s)
                self.model.update(s, reward)
                old_s1, old_s2 = s1, s2

                for _ in range(nb_simulation):
                    old_s1_, old_s2_, a = self.sample_observation()
                    s1_, s2_, reward = self.model.simulate(self.likelihood, old_s1_, old_s2_, a)
                    s.set_state(old_s1_, old_s2_, s1_, s2_, a)
                    self.q_learning.update(s, reward)

        return l_list, reward_list


class TDLearning:
    def __init__(self, env, TEMPERATURE=1, GAMMA=0.9, STEP_SIZE=0.25):
        self.temperature = TEMPERATURE
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.env = env
        self.q = np.zeros((env.state_dim**2, env.action_dim))
        self.learning_function = self.q_learning

    def flatten_state(self, s1, s2):
        return s1 + s2 * self.env.state_dim

    def sample_action(self, s1, s2):
        s = self.flatten_state(s1, s2)
        pi = utils.softmax(self.q[s])
        return np.random.choice(self.q.shape[1], 1, p=pi)[0]

    def best_action(self, s):
        return np.argmax(self.q[s])

    def update(self, s, reward):
        action = s.a
        s_prime = self.flatten_state(s.s1, s.s2)
        s = self.flatten_state(s.old_s1, s.old_s2)
        self.learning_function(s, s_prime, action, reward)

    def q_learning(self, s, s_prime, a, reward):
        self.q[s, a] += self.step_size*(reward + \
                        self.gamma*(np.max(self.q[s_prime])) - self.q[s, a])

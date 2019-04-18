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
    def __init__(self, env, env_model=None):
        self.env = env
        self.model = env_model
        self.q_learning = model.TDLearning(env)

    def reset(self):
        self.q_learning = model.TDLearning(self.env)
        self.model.reset_observation()

    def train(self, nb_episode, nb_simulation, buffer_size=None):
        s = environment.State()
        if self.model is not None:
            self.model.reset_observation()

        if buffer_size is None:
            buffer_size = int(nb_episode/10)
        buffer = self._fill_buffer(buffer_size)

        likelihood_list = []
        reward_list = []

        for _ in range(10):
            for i in range(len(buffer)):
                s = buffer[i]
                new = np.random.randint(10)
                if new < 1:
                    old_s1, old_s2 = self.env.reset()
                else:
                    old_s1, old_s2 = s.old_s1, s.old_s2
                s, reward, done = self._learn(s.old_s1, s.old_s2)
                buffer[i] = s

                if self.model is not None:
                    for _ in range(nb_simulation):
                        self.simulate()

                    real_likelihood = self.env.get_likelihood(s.old_s1, s.old_s2, s.s1, s.s2, s.a)
                    l = self.model.likelihood.get_likelihood(s)
                    likelihood_list.append(np.abs(real_likelihood - l.detach().numpy()))
                reward_list.append(reward)
            shuffle(buffer)

        return likelihood_list, reward_list

    def simulate(self):
        s = environment.State()
        old_s1, old_s2, a = self.model.sample_observation()
        s1, s2, reward = self.model.simulate(old_s1, old_s2, a)
        s.set_state(old_s1, old_s2, s1, s2, a)
        self.q_learning.update(s, reward)

    def print_info(self, episode):
        print(episode)
        # __import__('ipdb').set_trace()
        print(self.env.s1[0,0,0,:])
        print(utils.softmax(self.likelihood.model.cause.w[:,0,0,0].detach().numpy()))

    def _fill_buffer(self, nb_episode):
        buffer = []

        for episode in range(nb_episode):
            done = False
            old_s1, old_s2 = self.env.reset()
            while not done:
                s, reward, done = self._learn(old_s1, old_s2)
                buffer.append(copy.copy(s))
                old_s1, old_s2 = s.s1, s.s2
        shuffle(buffer)
        return buffer

    def _learn(self, old_s1, old_s2):
        s = environment.State()
        a = self.q_learning.sample_action(old_s1, old_s2)
        (s1, s2), reward, done, _ = self.env.step(a, old_s1, old_s2)
        s.set_state(old_s1, old_s2, s1, s2, a)

        self.q_learning.update(s, reward)
        self.model.update(s, reward)
        return s, reward, done


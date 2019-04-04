import numpy as np
import environment
import model
import utils

def train(env, likelihood, nb_episode):
    s = environment.State()
    l_a2b = np.zeros(nb_episode)
    l_b2a = np.zeros(nb_episode)

    for episode in range(nb_episode):
        done = False
        old_s1, old_s2 = env.reset()
        while not done:
            a = env.sample_action()
            (s1, s2), reward, done, _ = env.step(a)
            s.set_state(old_s1, old_s2, s1, s2, a)
            l1, l2 = likelihood.update(s)
            l_a2b[episode] = l1
            l_b2a[episode] = l2
            old_s1, old_s2 = s1, s2
    return l_a2b, l_b2a


class TDLearning:
    def __init__(self, env, TEMPERATURE=1, GAMMA=0.9, STEP_SIZE=0.25):
        self.temperature = TEMPERATURE
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.env = env
        self.q = np.zeros((env.state_dim**2, env.action_dim))
        self.learning_function = self.q_learning

    def sample_action(self,s):
        pi = utils.softmax(self.q[s])
        return np.random.choice(self.q.shape[1], 1, p=pi)[0]

    def best_action(self,s):
        return np.argmax(self.q[s])

    def train_episode(self):
        done = False
        reward_episode = 0
        step = 0

        s = self.env.reset()
        a = self.sample_action(s)
        while not done:
            step += 1
            a, s, reward, done = self.train_one_step(s, a)
            reward_episode += reward
        return step, reward_episode

    def test_episode(self):
        done = False
        reward_episode = 0
        step = 0

        s = self.env.reset()
        while not done:
            step += 1
            a = self.best_action(s)
            s, reward, done, _ = self.env.step(a)
            reward_episode += reward
        return step, reward_episode

    def train_one_step(self,s,a):
        s_prime, reward, done, _ = self.env.step(a)
        a, s = self.learning_function(s, s_prime, a, reward)
        return a, s, reward, done

    def q_learning(self,s,s_prime,a,reward):
        self.q[s, a] += self.step_size*(reward + \
                        self.gamma*(np.max(self.q[s_prime])) - self.q[s, a])
        a = self.sample_action(s_prime)
        return a, s_prime


class DoinaQ:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.td_learning = TDLearning(env)

    def train(self):
        pass

import numpy as np

class CausalEnvironment:
    def __init__(self, state_dim=10, action_dim=4, n_step=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_step = n_step

        self.set_prob()
        self.set_reward_function()
        self.reset()

    def set_reward_function(self):
        # TODO
        pass

    def set_prob(self):
        self.action = self.create_rv(self.action_dim, 1)
        self.old_s1 = self.create_rv(self.state_dim, 1)
        self.old_s2 = self.create_rv(self.state_dim, self.state_dim)
        self.s1 = self.create_rv(self.state_dim, self.state_dim**2 * self.action_dim)
        self.s1 = self.s1.reshape((self.state_dim, self.state_dim, self.action_dim, self.state_dim))
        self.s2 = self.create_rv(self.state_dim, self.state_dim**3 * self.action_dim)
        self.s2 = self.s2.reshape((self.state_dim, self.state_dim,
                                   self.state_dim, self.action_dim,
                                   self.state_dim))

    def create_rv(self, dim, n, peak=10):
        # if peak is high (>1), the distribution
        # will have low entropy
        mass = np.ones(dim)
        mass[np.random.randint(dim)] = peak
        return np.random.dirichlet(mass, n)

    def adapt_a(self):
        self.s1 = self.create_rv(self.state_dim, self.state_dim**2 * self.action_dim)
        self.s1 = self.s1.reshape((self.state_dim, self.state_dim, self.action_dim, self.state_dim))

    def sample_action(self):
        return np.random.randint(self.action_dim)

    def sample_state_uniformly(self):
        return np.random.randint(self.state_dim)

    def sample_state(self, prob, n=1):
        return np.argmax(np.random.multinomial(1, prob, size=n))

    def sample(self, action, old_s1=None, old_s2=None):
        a = action
        # self.sample_state(self.action[0])
        if old_s1 is None:
            old_s1 = self.sample_state(self.old_s1[0])
            old_s2 = self.sample_state(self.old_s2[old_s1])
        s1 = self.sample_state(self.s1[old_s1, old_s2, a])
        s2 = self.sample_state(self.s2[old_s1, old_s2, s1, a])
        self.state_a = s1
        self.state_b = s2

    def reset(self):
        self.n_step = 0
        self.state_a = self.sample_state_uniformly()
        self.state_b = self.sample_state_uniformly()
        return self.state_a, self.state_b

    def step(self, action, old_s1=None, old_s2=None):
        self.sample(action, old_s1, old_s2)

        self.n_step += 1
        reward = 1
        if self.n_step >= self.max_step:
            done = True
        else:
            done = False
        return (self.state_a, self.state_b), reward, done, 0


class State:
    def __init__(self):
        self.set_state(None, None, None, None, None)

    def set_state(self, old_s1, old_s2, s1, s2, a):
        self.old_s1 = old_s1
        self.old_s2 = old_s2
        self.s1 = s1
        self.s2 = s2
        self.a = a

    def _choose_node(self, my_state):
        if my_state == 1:
            node, other_node = self.s1, self.s2
        elif my_state == 2:
            node, other_node = self.s2, self.s1
        else:
            raise ValueError(f'my_state should either be 1 or 2, not {my_state}')
        return node, other_node

    def __str__(self):
        return f'{self.old_s1}, {self.old_s2}, {self.s1}, {self.s2}, {self.a}'

    def get_cause(self, my_state):
        node, _ = self._choose_node(my_state)
        return node, self.old_s1, self.old_s2, self.a

    def get_effect(self, my_state):
        node, other_node = self._choose_node(my_state)
        return other_node, self.old_s1, self.old_s2, node, self.a

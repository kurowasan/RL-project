import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import environment

class EnvironmentModel:
    def __init__(self, state_dim, action_dim, determinist_reward=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.determinist_reward = determinist_reward

        if determinist_reward:
            self.reward_table = np.zeros((state_dim, state_dim, action_dim))
        else:
            self.mean = np.zeros((state_dim, state_dim, action_dim))
            self.count = np.zeros((state_dim, state_dim, action_dim))
            self.k = np.zeros((state_dim, state_dim, action_dim))
            self.sum = np.zeros((state_dim, state_dim, action_dim))
            self.sum_sq = np.zeros((state_dim, state_dim, action_dim))

    def update(self, s, reward):
        if self.determinist_reward:
            self.reward_table[s.s1, s.s2, s.a] = reward
        else:
            if self.count[s.s1, s.s2, s.a]:
                self.k[s.s1, s.s2, s.a] = reward
            self.mean[s.s1, s.s2, s.a] += reward
            self.count[s.s1, s.s2, s.a] += 1
            K = self.k[s.s1, s.s2, s.a]
            self.sum[s.s1, s.s2, s.a] += reward - K
            self.sum_sq[s.s1, s.s2, s.a] += reward * reward - K

    def sample_reward(self, s1, s2, a):
        if self.determinist_reward:
            reward = self.reward_table[s1, s2, a]
        else:
            n = self.count[s1, s2, a]
            if n == 0:
                return -10
            mean = self.mean[s1, s2, a]/n
            if n > 1:
                var = (self.sum[s1, s2, a] - self.sum_sq[s1, s2, a]**2/n)/(n-1)
                std = np.sqrt(var)
            else:
                std = 2
            reward = np.random.normal(mean, scale=std, size=1)
        return reward

    def simulate(self, likelihood, old_s1, old_s2, a):
        prob = np.zeros((self.state_dim*self.state_dim))
        state = environment.State()
        for s1 in range(self.state_dim):
            for s2 in range(self.state_dim):
                state.set_state(old_s1, old_s2, s1, s2, a)
                l = likelihood.get_likelihood(state).detach().numpy()
                prob[s1 + s2*self.state_dim] = np.exp(l)
        if np.sum(prob) != 1:
            prob[-1] += 1-np.sum(prob)
            # __import__('ipdb').set_trace()
        s = np.random.choice(np.arange(prob.shape[0]), p=prob)
        s1 = s % self.state_dim
        s2 = s // self.state_dim
        reward = self.sample_reward(s1, s2, a)
        return s1, s2, reward


class LikelihoodEstimators:
    def __init__(self, s, action_dim, lr):
        self.model_a2b = ModelInterface(s, action_dim, 1, lr)
        self.model_b2a = ModelInterface(s, action_dim, 2, lr)

    def update(self, state):
        l_a2b = self.model_a2b.get_likelihood(state)
        l_b2a = self.model_b2a.get_likelihood(state)
        nll_a2b = self.model_a2b.update(state)
        nll_b2a = self.model_b2a.update(state)
        return l_a2b, l_b2a

    def get_likelihood(self, state):
        return self.model_a2b.get_likelihood(state).item(), \
               self.model_b2a.get_likelihood(state).item()

    def reinitialize_optimizer(self, lr):
        self.model_a2b.reinitialize_optimizer(lr)
        self.model_b2a.reinitialize_optimizer(lr)


class ModelInterface:
    def __init__(self, s, action_dim, a2b, lr=1e-4):
        cause = Cause(s,s,s, action_dim, a2b)
        effect = Effect(s,s,s,s, action_dim, a2b)
        self.a2b = a2b
        self.model = CausalModel(cause, effect)
        self.optim = optim.RMSprop(self.model.parameters(), lr=lr)
        self.nb_step = 0

    def reinitialize_optimizer(self, lr):
        self.optim = optim.RMSprop(self.model.parameters(), lr=lr)

    def update(self, state, batch_size=10):
        if self.nb_step == 0:
            self.nll = torch.zeros(1)
        self.nb_step += 1
        self.model.train()
        self.nll += -self.model(state)
        if self.nb_step % batch_size == 0:
            self.nll.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.nll = torch.zeros(1)
        return -self.model(state).item()

    def get_likelihood(self, state):
        self.model.eval()
        return self.model(state)


class LikelihoodWithoutGradient:
    def __init__(self, s, action_dim, lr):
        self.count_total = np.zeros((s,s,s,s, action_dim))

    def update(self, s):
        likelihood = self.get_likelihood(s)
        self.count_total[s.s1, s.s2, s.old_s1, s.old_s2, s.a] += 1
        return likelihood

    def get_likelihood(self, s):
        event = self.count_total[s.s1, s.s2, s.old_s1, s.old_s2, s.a]
        total = np.sum(self.count_total)
        total_a = np.sum(self.count_total[s.s1])
        total_b = np.sum(self.count_total[:, s.s2])
        if event == 0:
            return -3, -3

        p_a = total_a/total
        p_b = total_b/total
        p_b_given_a = event/np.sum(self.count_total[s.s1,:,s.old_s1, s.old_s2, s.a])
        p_a_given_b = event/np.sum(self.count_total[:,s.s2,s.old_s1, s.old_s2, s.a])
        likelihood = np.log(p_a)+np.log(p_b_given_a), np.log(p_b)+np.log(p_a_given_b)
        return likelihood

    def reinitialize_optimizer(self, lr):
        pass


class CausalModel(nn.Module):
    def __init__(self, cause, effect):
        super().__init__()
        self.cause = cause
        self.effect = effect

    def forward(self, state):
        return self.cause(state) + self.effect(state)


class Cause(nn.Module):
    def __init__(self, s1, s2, s3, a, a2b):
        super().__init__()
        self.s1_size = s1
        self.s2_size = s2
        self.s3_size = s3
        self.a_size = a
        self.a2b = a2b
        self.w = nn.Parameter(torch.zeros((s1, s2, s3, a)))

    def forward(self, s):
        cste = torch.logsumexp(self.w[:, s.old_s1, s.old_s2, s.a], dim=0)
        return self.w[s.get_cause(self.a2b)] - cste


class Effect(nn.Module):
    def __init__(self, s1, s2, s3, s4, a, a2b):
        super().__init__()
        self.s1_size = s1
        self.s2_size = s2
        self.s3_size = s3
        self.s4_size = s4
        self.a_size = a
        self.a2b = a2b
        self.w = nn.Parameter(torch.zeros((s1, s2, s3, s4, a)))

    def forward(self, s):
        _, old_s1, old_s2, node, a = s.get_effect(self.a2b)
        cste = torch.logsumexp(self.w[:, old_s1, old_s2, node, a], dim=0)

        return self.w[s.get_effect(self.a2b)] - cste

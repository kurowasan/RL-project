import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

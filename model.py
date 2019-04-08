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


class ModelJoint(nn.Module):
    """This model simply models p and v jointly"""
    def __init__(self, num_h_layers, num_h_units):
        super(ModelJoint, self).__init__()
        self.num_h_layers = num_h_layers
        self.num_h_units = num_h_units

        self.layers = nn.ModuleList()

        for i in range(num_h_layers + 1):
            in_dim, out_dim = self.num_h_units, self.num_h_units
            if i == 0: in_dim = 5  # position, velocity, action 1, action 2, action 3 (one hot vector for action)
            if i == num_h_layers: out_dim = 2 # position, velocity

            self.layers.append(nn.Linear(in_dim, out_dim))

        self.reset_parameters()

    def forward(self, x):
        """x is (batch_size, 5)"""
        for i in range(self.num_h_layers + 1):
            x = self.layers[i](x)
            if i != self.num_h_layers: x = nn.functional.relu(x)

        return x

    def reset_parameters(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)


class ModelCausal(nn.Module):
    """This model models p and v with mechanisms, either with the correct or incorrect decomposition"""
    def __init__(self, num_h_layers, num_h_units, correct=True):
        super(ModelCausal, self).__init__()
        self.num_h_layers = num_h_layers
        self.num_h_units = num_h_units
        self.correct = correct  # is it correct factorization?

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()

        for i in range(num_h_layers + 1):
            in_dim, out_dim = self.num_h_units, self.num_h_units
            if i == 0: in_dim = 5  # position, velocity, action 1, action 2, action 3 (one hot vector for action)
            if i == num_h_layers: out_dim = 1  # if correct: new_velocity else: new_position
            self.layers1.append(nn.Linear(in_dim, out_dim))

        for i in range(num_h_layers + 1):
            in_dim, out_dim = self.num_h_units, self.num_h_units
            if i == 0: in_dim = 6  # position, velocity, action 1, action 2, action 3, {if correct: new_velocity else: new_position} (one hot vector for action)
            if i == num_h_layers: out_dim = 1  # if correct: new_position else: new_velocity
            self.layers2.append(nn.Linear(in_dim, out_dim))

        self.reset_parameters()

    def forward(self, x):
        """x is (batch_size, 5)"""
        tmp = x
        for i in range(self.num_h_layers + 1):
            tmp = self.layers1[i](tmp)
            if i != self.num_h_layers: tmp = nn.functional.relu(tmp)

        new_v_or_p = tmp
        x = torch.cat([x, new_v_or_p], 1)

        tmp = x
        for i in range(self.num_h_layers + 1):
            tmp = self.layers2[i](tmp)
            if i != self.num_h_layers: tmp = nn.functional.relu(tmp)

        new_p_or_v = tmp

        if self.correct:
            v = new_v_or_p
            p = new_p_or_v
        else:
            v = new_p_or_v
            p = new_v_or_p

        return torch.cat([p, v], 1)

    def reset_parameters(self):
        for layer in self.layers1:
            torch.nn.init.xavier_uniform_(layer.weight)

        for layer in self.layers2:
            torch.nn.init.xavier_uniform_(layer.weight)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import environment
from approximators import FastTileCoding
import utils

class EnvironmentModel:
    def __init__(self, state_dim, action_dim, a2b, batch_size, lr, determinist_reward=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a2b = a2b
        self.batch_size = batch_size
        self.lr = lr
        self.determinist_reward = determinist_reward
        self.likelihood = ModelInterface(state_dim, action_dim, a2b, batch_size, lr)
        self.reset_observation()
        self.simulation_total = 0

        if determinist_reward:
            self.reward_table = np.zeros((state_dim, state_dim, action_dim))
        else:
            self.mean = np.zeros((state_dim, state_dim, action_dim))
            self.count = np.zeros((state_dim, state_dim, action_dim))
            self.k = np.zeros((state_dim, state_dim, action_dim))
            self.sum = np.zeros((state_dim, state_dim, action_dim))
            self.sum_sq = np.zeros((state_dim, state_dim, action_dim))

    def update_observation(self, s):
        self.observation[s.s1, s.s2, s.a] = 1

    def reset_observation(self):
        self.observation = np.zeros((self.state_dim, self.state_dim,
                                     self.action_dim))

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

    def reinitialize_optimizer(self, lr):
        self.lr = lr
        self.likelihood.reinitialize_optimizer(lr)

    def update(self, s, reward):
        l = self.likelihood.update(s)
        self.update_observation(s)
        self.update_reward(s, reward)
        return l

    def update_reward(self, s, reward):
        if self.determinist_reward:
            self.reward_table[s.old_s1, s.old_s2, s.a] = reward
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

    def simulate(self, old_s1, old_s2, a, env):
        self.simulation_total += 1
        prob = np.zeros((self.state_dim*self.state_dim))
        state = environment.State()
        #TODO: keep values for same step...
        for s1 in range(self.state_dim):
            for s2 in range(self.state_dim):
                state.set_state(old_s1, old_s2, s1, s2, a)
                l = self.likelihood.get_likelihood(state).detach().numpy()
                prob[s1 + s2*self.state_dim] = np.exp(l)

        if np.sum(prob) != 1:
            prob = prob/np.sum(prob)
        s = np.random.choice(np.arange(prob.shape[0]), p=prob)
        s1 = s % self.state_dim
        s2 = s // self.state_dim
        confidence_level = np.amax(prob)
        if confidence_level > 0.5:
            print(f'Prob max:{np.amax(prob)}')
            print(f'simulation result: ({s1}, {s2})')
            print(env.step(a, old_s1, old_s2))
            # __import__('ipdb').set_trace()
        reward = self.sample_reward(s1, s2, a)
        return s1, s2, reward, confidence_level


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


class LikelihoodEstimators:
    def __init__(self, s, action_dim, batch_size, lr):
        self.model_a2b = ModelInterface(s, action_dim, 1, batch_size, lr)
        self.model_b2a = ModelInterface(s, action_dim, 2, batch_size, lr)

    def update(self, state):
        l_a2b = self.model_a2b.get_likelihood(state)
        l_b2a = self.model_b2a.get_likelihood(state)
        self.model_a2b.update(state)
        self.model_b2a.update(state)
        return l_a2b, l_b2a

    def get_likelihood(self, state):
        return self.model_a2b.get_likelihood(state).item(), \
               self.model_b2a.get_likelihood(state).item()

    def reinitialize_optimizer(self, lr):
        self.model_a2b.reinitialize_optimizer(lr)
        self.model_b2a.reinitialize_optimizer(lr)


class ModelInterface:
    def __init__(self, s, action_dim, a2b, batch_size, lr=1e-4):
        cause = Cause(s,s,s, action_dim, a2b)
        effect = Effect(s,s,s,s, action_dim, a2b)
        self.a2b = a2b
        self.model = CausalModel(cause, effect)
        self.optim = optim.RMSprop(self.model.parameters(), lr=lr)
        self.nb_step = 0
        self.batch_size = batch_size

    def reinitialize_optimizer(self, lr):
        self.optim = optim.RMSprop(self.model.parameters(), lr=lr)

    def update(self, state):
        if self.nb_step == 0:
            self.nll = torch.zeros(1)
        self.nb_step += 1
        self.model.train()
        self.nll += -self.model(state)
        if self.nb_step % self.batch_size == 0:
            self.nll.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.nll = torch.zeros(1)
        return self.model(state).item()

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
        self.a_size = a
        self.a2b = a2b
        self.w = nn.Parameter(torch.zeros((s1, s2, s3, a))) # zeros

    def forward(self, s):
        cste = torch.logsumexp(self.w[:, s.old_s1, s.old_s2, s.a], dim=0)
        return self.w[s.get_cause(self.a2b)] - cste


class Effect(nn.Module):
    def __init__(self, s1, s2, s3, s4, a, a2b):
        super().__init__()
        self.a_size = a
        self.a2b = a2b
        self.w = nn.Parameter(torch.zeros((s1, s2, s3, s4, a))) # zeros

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


class TileCoding_Joint(nn.Module):
    def __init__(self, num_actions, num_bins, num_tilings, limits):
        super(TileCoding_Joint, self).__init__()

        self.tile_codings_p = nn.ModuleList()
        self.tile_codings_v = nn.ModuleList()
        for _ in range(num_actions):
            self.tile_codings_p.append(TileCoding(num_bins, num_tilings, limits))
            self.tile_codings_v.append(TileCoding(num_bins, num_tilings, limits))

    def forward(self, state):
        bs = state.size(0)
        p_v = state[:, :2].detach().cpu().numpy()
        action_one_hot = state[:, 2:].byte()
        action = torch.masked_select(torch.arange(3), action_one_hot).detach().cpu().numpy()

        ps = []
        vs = []
        for i in range(bs):
            ps.append(self.tile_codings_p[action[i]](p_v[i]).unsqueeze(0))
            vs.append(self.tile_codings_v[action[i]](p_v[i]).unsqueeze(0))

        p_vs = [torch.cat(ps).unsqueeze(1), torch.cat(vs).unsqueeze(1)]
        return torch.cat(p_vs, 1)


class FastTileCoding_Joint(nn.Module):
    def __init__(self, num_actions, num_bins, num_tilings, limits, optimizer=None):
        super(FastTileCoding_Joint, self).__init__()
        self.num_actions = num_actions
        self.tile_codings_p = nn.ModuleList()
        self.tile_codings_v = nn.ModuleList()
        self.tile_codings_r = nn.ModuleList()

        for _ in range(num_actions):
            self.tile_codings_p.append(FastTileCoding(num_bins, num_tilings, limits))
            self.tile_codings_v.append(FastTileCoding(num_bins, num_tilings, limits))
            self.tile_codings_r.append(FastTileCoding(num_bins, num_tilings, limits))

        self.optimizer = optimizer

    def forward(self, state, action=None):
        p_min, p_max = self.tile_codings_p[0].limits[0, 0], self.tile_codings_p[0].limits[0, 1]
        v_min, v_max = self.tile_codings_p[0].limits[1, 0], self.tile_codings_p[0].limits[1, 1]
        if action is None:
            p_primes = []
            v_primes = []
            r_primes = []
            for a in range(self.num_actions):
                if state[a].size(0) > 0:
                    p_v = state[a][:, :2]
                    p = p_v[:, :1]
                    v = p_v[:, 1:]
                    p_delta = self.tile_codings_p[a](p_v)
                    v_delta = self.tile_codings_v[a](p_v)
                    r_prime = self.tile_codings_r[a](p_v)
                    p_primes.append(torch.clamp(p + p_delta, p_min, p_max))
                    v_primes.append(torch.clamp(v + v_delta, v_min, v_max))
                    r_primes.append(r_prime)
            p_primes = torch.cat(p_primes, 0)
            v_primes = torch.cat(v_primes, 0)
            r_primes = torch.cat(r_primes, 0)

            return torch.cat([p_primes, v_primes, r_primes], 1)
        else:
            p_v = state[:, :2]
            p, v = state[:,:1], state[:, 1:2]
            p_delta = self.tile_codings_p[action](p_v)
            v_delta = self.tile_codings_v[action](p_v)
            r_prime = self.tile_codings_r[action](p_v)
            p_prime = torch.clamp(p + p_delta, p_min, p_max)
            v_prime = torch.clamp(v + v_delta, v_min, v_max)

            return torch.cat([p_prime, v_prime, r_prime], 1)

class FastTileCoding_Causal(nn.Module):
    def __init__(self, num_actions, num_bins, num_tilings, limits, optimizer=None):
        super(FastTileCoding_Causal, self).__init__()
        self.num_actions = num_actions
        self.tile_codings_p = nn.ModuleList()
        self.tile_codings_v = nn.ModuleList()
        self.tile_codings_r = nn.ModuleList()
        for _ in range(num_actions):
            self.tile_codings_p.append(FastTileCoding(int(num_bins ** (2/3)), num_tilings, np.concatenate([limits, limits[1:]], 0)))
            self.tile_codings_v.append(FastTileCoding(num_bins, num_tilings, limits))
            self.tile_codings_r.append(FastTileCoding(num_bins, num_tilings, limits))

        self.optimizer = optimizer

    def forward(self, state, action=None):
        p_min, p_max = self.tile_codings_p[0].limits[0, 0], self.tile_codings_p[0].limits[0, 1]
        v_min, v_max = self.tile_codings_p[0].limits[1,0], self.tile_codings_p[0].limits[1,1]

        if action is None:  # states are group action wise
            vs = []
            v_primes = []
            r_primes = []
            for a in range(self.num_actions):
                if state[a].size(0) > 0:
                    p_v = state[a][:, :2]
                    v = state[a][:, 1:2]
                    #print(p_v.size())
                    v_delta = self.tile_codings_v[a](p_v)
                    v_prime = torch.clamp(v + v_delta, v_min, v_max)
                    vs.append(v)
                    v_primes.append(v_prime)
                    r_primes.append(self.tile_codings_r[a](p_v))
                else:
                    v_primes.append(None)

            p_primes = []
            for a in range(self.num_actions):
                if state[a].size(0) > 0:
                    p_v = state[a][:, :2]
                    p = state[a][:, 0:1]
                    p_v_v_prime = torch.cat([p_v, v_primes[a]], 1)
                    p_delta = self.tile_codings_p[a](p_v_v_prime)
                    p_prime = torch.clamp(p + p_delta, p_min, p_max)
                    p_primes.append(p_prime)

            v_primes = [v_prime for v_prime in v_primes if v_prime is not None]
            p_primes = torch.cat(p_primes, 0)
            v_primes = torch.cat(v_primes, 0)
            r_primes = torch.cat(r_primes, 0)

            return torch.cat([p_primes, v_primes, r_primes], 1)
        else:  # action is provided
            p_v = state[:, :2]
            v = state[:, 1:2]
            v_delta = self.tile_codings_v[action](p_v)
            v_prime = torch.clamp(v + v_delta, v_min, v_max)

            p = state[:, :1]
            p_v_v_prime = torch.cat([p_v, v_prime], 1)
            p_delta = self.tile_codings_p[action](p_v_v_prime)
            p_prime = torch.clamp(p + p_delta, p_min, p_max)

            r_prime = self.tile_codings_r[action](p_v)

            return torch.cat([p_prime, v_prime, r_prime], 1)


class FastTileCoding_AntiCausal(nn.Module):
    def __init__(self, num_actions, num_bins, num_tilings, limits, optimizer=None):
        super(FastTileCoding_AntiCausal, self).__init__()
        self.num_actions = num_actions
        self.tile_codings_p = nn.ModuleList()
        self.tile_codings_v = nn.ModuleList()
        self.tile_codings_r = nn.ModuleList()
        for _ in range(num_actions):
            self.tile_codings_p.append(FastTileCoding(num_bins, num_tilings, limits))
            self.tile_codings_v.append(FastTileCoding(int(num_bins ** (2/3)), num_tilings, np.concatenate([limits, limits[0:1]], 0)))
            self.tile_codings_r.append(FastTileCoding(num_bins, num_tilings, limits))

        self.optimizer = optimizer

    def forward(self, state, action=None):
        p_min, p_max = self.tile_codings_p[0].limits[0, 0], self.tile_codings_p[0].limits[0, 1]
        v_min, v_max = self.tile_codings_p[0].limits[1, 0], self.tile_codings_p[0].limits[1, 1]
        if action is None:  # states are group action wise
            ps = []
            p_primes = []
            r_primes = []
            for a in range(self.num_actions):
                if state[a].size(0) > 0:
                    p_v = state[a][:, :2]
                    p = state[a][:, :1]
                    p_delta = self.tile_codings_p[a](p_v)
                    p_prime = torch.clamp(p + p_delta, p_min, p_max)
                    ps.append(p)
                    p_primes.append(p_prime)
                    r_primes.append(self.tile_codings_r[a](p_v))
                else:
                    p_primes.append(None)

            v_primes = []
            for a in range(self.num_actions):
                if state[a].size(0) > 0:
                    p_v = state[a][:, :2]
                    v = state[a][:, 1:2]
                    p_v_p_prime = torch.cat([p_v, p_primes[a]], 1)
                    v_delta = self.tile_codings_v[a](p_v_p_prime)
                    v_prime = torch.clamp(v + v_delta, v_min, v_max)
                    v_primes.append(v_prime)

            p_primes = [p_prime for p_prime in p_primes if p_prime is not None]
            p_primes = torch.cat(p_primes, 0)
            v_primes = torch.cat(v_primes, 0)
            r_primes = torch.cat(r_primes, 0)

            return torch.cat([p_primes, v_primes, r_primes], 1)

        else:  # action is provided
            p_v = state[:, :2]
            p = state[:, :1]
            p_delta = self.tile_codings_p[action](p_v)
            p_prime = torch.clamp(p + p_delta, p_min, p_max)

            v = state[:, 1:2]
            p_v_p_prime = torch.cat([p_v, p_prime], 1)
            v_delta = self.tile_codings_v[action](p_v_p_prime)
            v_prime = torch.clamp(v + v_delta, v_min, v_max)

            r_prime = self.tile_codings_r[action](p_v)

            return torch.cat([p_prime, v_prime, r_prime], 1)


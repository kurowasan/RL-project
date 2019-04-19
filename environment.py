import numpy as np
import utils
from abstractgraph import SparseGraphEnvironment

class CausalEnvironment:
    def __init__(self, state_dim=10, action_dim=4, n_step=100, peak = 10,
                 deterministic_reward=True, graph_traversal=True, 
                 correct_path_proportion = 0.6, branching_prob = 0.75, output = './'):
        self.state_dim = state_dim

        self.action_dim = action_dim
        self.max_step = n_step
        self.peak = peak
        self.reward = np.zeros((state_dim, state_dim, action_dim))
        self.final_reward = 10
        self.action_reward = -1
        self.correct_path_reward = 1
        self.deterministic_reward = deterministic_reward

        self.set_prob()
        if graph_traversal:
            self.graph_env = SparseGraphEnvironment(nb_nodes=self.state_dim, correct_path_proportion = correct_path_proportion, 
                branching_prob = branching_prob, nb_actions = self.action_dim)
            np.save(output+'adjacency.npy',self.graph_env.adjacency)
            self.s1 = np.einsum('ijkl,il->ijkl',self.s1,self.graph_env.adjacency)
            self.s2 = np.einsum('ijklm,ik->ijklm',self.s2,self.graph_env.adjacency)
            self.normalize()

            self.set_reward_function(reward_type='graph_traversal') ##TODO

        else:
            self.set_reward_function()

        self.reset()

    def set_reward_function(self, reward_type='bern'):
        if reward_type == 'normal':
            self.reward = np.random.normal(0, 1, (self.reward.shape))
        elif reward_type == 'bern':
            self.reward = np.random.binomial(1, p=(0.1), size=(self.reward.shape))
            self.reward *= 10
            reward = np.random.binomial(1, p=(0.1), size=(self.reward.shape))
            reward *= -1000
            self.reward += reward
        elif reward_type == 'graph_traversal':
            self.reward+=(self.action_reward)

            for i in self.graph_env.correct_path[:-1]:
                for j in self.graph_env.correct_path[1:]:
                    action = np.argmax(self.graph_env.transition[i,j])
                    self.reward[i,j,action]+=self.correct_path_reward

            ### setting the end reward

            end_i, end_a = np.nonzero(self.graph_env.transition[:,self.action_dim-1,:])
            for i in end_i:
                for a in end_a:
                    self.reward[i,self.action_dim-1,a]+=self.final_reward

    def set_prob(self):
        self.s1 = self.create_rv(self.state_dim, self.state_dim**2 * self.action_dim)
        # old_s1, old_s2, a, s1
        self.s1 = self.s1.reshape((self.state_dim, self.state_dim, self.action_dim, self.state_dim))
        self.s2 = self.create_rv(self.state_dim, self.state_dim**3 * self.action_dim)
        # old_s1, old_s2, s1, a, s2
        self.s2 = self.s2.reshape((self.state_dim, self.state_dim,
                                   self.state_dim, self.action_dim,
                                   self.state_dim))

    def adapt_a(self):
        self.s1 = self.create_rv(self.state_dim, self.state_dim**2 * self.action_dim)
        self.s1 = self.s1.reshape((self.state_dim, self.state_dim, self.action_dim, self.state_dim))

    def create_rv(self, dim, n):
        # if peak is high (>1), the distribution
        # will have low entropy
        distr = np.zeros((n, dim))
        for i in range(n):
            mass = np.ones(dim)
            mass[np.random.randint(dim)] = self.peak
            distr[i] = np.random.dirichlet(mass, 1)
        return distr

    def normalize(self):
        for i in np.arange(self.s1.shape[0]):
            for j in np.arange(self.s1.shape[1]):
                for k in np.arange(self.s1.shape[2]):
                    if np.sum(self.s1[i,j,k])>0:
                        self.s1[i,j,k]/=np.sum(self.s1[i,j,k])


        for i in np.arange(self.s2.shape[0]):
            for j in np.arange(self.s2.shape[1]):
                for k in np.arange(self.s2.shape[2]):
                    for l in np.arange(self.s2.shape[3]):
                        if np.sum(self.s2[i,j,k,l])>0:
                            self.s2[i,j,k,l]/=np.sum(self.s2[i,j,k,l])



    def sample_action_uniformly(self):
        return np.random.randint(self.action_dim)

    def sample_state_uniformly(self):
        return np.random.randint(self.state_dim)

    def sample_state(self, prob, n=1):
        return np.argmax(np.random.multinomial(1, prob, size=n))

    def sample_reward(self, s1, s2, a):
        if self.deterministic_reward:
            reward = self.reward[s1, s2, a]
        else:
            r = self.reward[s1, s2, a]
            reward = np.random.normal(r, 0.1, 1)
        return reward

    def sample(self, a, old_s1, old_s2):
        self.state_a = self.sample_state(self.s1[old_s1, old_s2, a])
        self.state_b = self.sample_state(self.s2[old_s1, old_s2, self.state_a, a])
        return self.state_a, self.state_b

    def reset(self):
        self.n_step = 0
        self.state_a = self.sample_state_uniformly()
        self.state_b = self.sample_state_uniformly()
        return self.state_a, self.state_b

    def step(self, action, old_s1, old_s2):
        self.sample(action, old_s1, old_s2)

        self.n_step += 1
        reward = self.sample_reward(self.state_a, self.state_b, action)
        if self.n_step >= self.max_step:
            done = True
        else:
            done = False
        return (self.state_a, self.state_b), reward, done, 0

    def get_likelihood(self, old_s1, old_s2, s1, s2, a):
        p_a = self.s1[old_s1, old_s2, a, s1]
        p_b_given_a = self.s2[old_s1, old_s2, s1, a, s2]
        return np.log(p_a) + np.log(p_b_given_a)

    def compare_directions(self, likelihood):
        ab = []
        ba = []
        for i in range(100):
            self.set_prob()
            p_a = self.s1
            p_ba = self.s2
            joint = np.einsum('ijkl,ijlkm->ijlmk', p_a, p_ba)
            p_b = np.sum(joint, axis=2) /np.sum(joint, axis=(2,3))[None,:,:]
            dim = self.state_dim
            p_ab = np.zeros(joint.shape)
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for l in range(2):
                            p_ab[i,j,:,k,l] = joint[i,j,:,k,l] / np.sum(joint[i,j,:,k,l])
                            #np.sum(joint, axis=2)[None,:,:,:]
            joint2 = np.einsum('ijmk,ijlmk->ijlmk', p_b, p_ab)
            # __import__('ipdb').set_trace()

            p_a_tilde = np.ones(self.s1.shape) * 1/8
            p_ba_tilde = np.ones(self.s2.shape) * 1/8
            p_b_tilde = np.ones(p_b.shape) * 1/8
            p_ab_tilde = np.ones(p_ab.shape) * 1/8

            # __import__('ipdb').set_trace()
            diff_ab = np.linalg.norm(p_a - 1/dim)
            diff_p_ba = np.linalg.norm(p_ba - 1/dim)
            diff_ba = np.linalg.norm(p_b - 1/dim)
            diff_p_ab = np.linalg.norm(p_ab - 1/dim)

            ab.append(diff_ab + diff_p_ba)
            ba.append(diff_ba + diff_p_ab)
            print(f'cond:  a->b: {diff_p_ba}, b->a: {diff_p_ab}')
            print(f'marg: a->b: {diff_ab}, b->a: {diff_ba}')
        print(f'a->b: {sum(ab)}, b->a: {sum(ba)}')

    def compare_likelihood(self, likelihood, old_s1, old_s2, s1, s2, a):
        p_a = self.s1[old_s1, old_s2, a, :]
        p_ba = self.s2[old_s1, old_s2, s1, a, :]

        joint = np.einsum('ijkl,ijlkm->ijlmk', self.s1, self.s2)
        p_b_emp = np.sum(joint, axis=2) /np.sum(joint, axis=(2,3))[None,:,:]
        p_ab_emp = np.zeros(joint.shape)
        dim = self.state_dim
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(2):
                        p_ab_emp[i,j,:,k,l] = joint[i,j,:,k,l] / np.sum(joint[i,j,:,k,l])

        # self.get_empirical_prob(a, old_s1, old_s2)
        # p_a_emp, p_ba_emp, p_b_emp, p_ab_emp = self.get_cond_prob(s1, s2)

        p_a_model = likelihood.model_a2b.model.cause.w.detach().numpy()
        p_a_model = utils.softmax(p_a_model[:, old_s1, old_s2, a])
        p_ba_model = likelihood.model_a2b.model.effect.w.detach().numpy()
        p_ba_model = utils.softmax(p_ba_model[:, old_s1, old_s2, s1, a])

        p_b_model = likelihood.model_b2a.model.cause.w.detach().numpy()
        p_b_model = utils.softmax(p_b_model[:, old_s1, old_s2, a])
        p_ab_model = likelihood.model_b2a.model.effect.w.detach().numpy()
        p_ab_model = utils.softmax(p_ab_model[:, old_s1, old_s2, s2, a])

        print('-' * 80)
        print(f'old_s1:{old_s1}, old_s2:{old_s2}, s1:{s1}, s2:{s2}, a:{a}')

        print('P(A) ' + '-' * 10)
        print(f'p_a: {p_a}, \n p_a_model:{p_a_model}')
        print(f'p_ba: {p_ba}, \n p_ba_model:{p_ba_model}')

        print('P(B) ' + '-' * 10)
        print(f'p_b_emp: {p_b_emp[old_s1, old_s2, :, a]}, \n p_b_model:{p_b_model}')
        print(f'p_ab_emp: {p_ab_emp[old_s1, old_s2, :, s2, a]}, \n p_ab_model:{p_ab_model}')

        print(f'Real likelihood a->b: {p_a[s1] * p_ba[s2]}')
        # print(f'Empirical likelihood b->a: {p_b_emp[s2] * p_ab_emp[s1]}')
        print(f'Model likelihood a->b: {p_a_model[s1] * p_ba_model[s2]}')
        print(f'Model likelihood b->a: {p_b_model[s2] * p_ab_model[s1]}')

    def get_empirical_joint(self, a):
        self.joint = np.zeros((self.state_dim, self.state_dim,
                               self.state_dim, self.state_dim))
        for i in range(self.state_dim):
            for j in range(self.state_dim):
                for _ in range(1000):
                    s1, s2 = self.sample(a, i, j)
                    self.joint[s1, s2, i, j] += 1
        self.joint = self.joint/np.sum(self.joint)

    def get_marginal_conditional(self, s1, s2):
        p_a = np.sum(self.joint, axis=(1,2,3))
        p_b = np.sum(self.joint, axis=(0,2,3))
        p_ab = np.sum(self.joint, axis=(2,3))

        p_b_given_a = p_ab/p_a
        p_a_given_b = p_ab/p_b

        return p_a, p_b_given_a, p_b, p_a_given_b

    def get_empirical_prob(self, a, old_s1, old_s2):
        self.joint = np.zeros((self.state_dim, self.state_dim,))
        for _ in range(10000):
            s1, s2 = self.sample(a, old_s1, old_s2)
            self.joint[s1, s2] += 1
        self.joint = self.joint/np.sum(self.joint)

    def get_cond_prob(self, s1, s2):
        p_a = np.sum(self.joint, axis=(1))
        p_b = np.sum(self.joint, axis=(0))

        p_b_given_a = self.joint[s1,:]/p_a[s1]
        p_a_given_b = self.joint[:,s2]/p_b[s2]

        return p_a, p_b_given_a, p_b, p_a_given_b


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

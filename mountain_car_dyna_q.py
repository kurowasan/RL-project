import os
import argparse
import math
import pickle

from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.envs.classic_control.mountain_car
import torch

from model import ModelCausal, ModelJoint, FastTileCoding_Joint, FastTileCoding_Causal, FastTileCoding_AntiCausal, TileCoding_Joint
from approximators import FastTileCoding
from data import DataManager
from mountain_car import get_modified_MountainCarEnv, to_tensor, group_by_actions, get_evaluation_dataset, test_agent


class QTileCoding(torch.nn.Module):
    def __init__(self, num_actions, num_bins, num_tilings, limits):
        """For discrete actions and continuous state space"""
        super(QTileCoding, self).__init__()
        self.num_actions = num_actions
        self.tile_codings_q = torch.nn.ModuleList()
        for _ in range(num_actions):
            self.tile_codings_q.append(FastTileCoding(num_bins, num_tilings, limits))

    def forward(self, state):
        qs = []
        for a in range(self.num_actions):
            qs.append(self.tile_codings_q[a](state[a][:, :2]))
        qs = torch.cat(qs, 0)
        return qs

    def update(self, s, a, s_prime, r, alpha, gamma):
        """Does one stochastic gradient step.
        s: (1, 2)
        a: int
        s_prime: (1, 2)
        """
        with torch.no_grad():
            max_q_s_prime = torch.Tensor([-np.inf])
            for a_prime in range(self.num_actions):
                q = self.tile_codings_q[a_prime](s_prime)
                if q > max_q_s_prime:
                    max_q_s_prime = q

        q_s_a = self.tile_codings_q[a](s)
        self.tile_codings_q[a].zero_grad()
        q_s_a.backward()
        with torch.no_grad():
            target = r + gamma * max_q_s_prime
            self.tile_codings_q[a].weights += alpha * (target - q_s_a) * self.tile_codings_q[a].weights.grad


class Agent:
    def __init__(self, q, model, data_buffer):
        self.q = q
        self.model = model
        self.data_buffer = data_buffer

    def get_action(self, s, eps=0.):
        explore = np.random.uniform(0, 1) < eps
        if explore:
            arg_max = np.random.randint(0, 3)
        else:
            max = -np.inf
            for a in range(self.q.num_actions):
                q = self.q.tile_codings_q[a](s)
                if q > max:
                    arg_max, max = a, q
        return arg_max

    def train_model(self, s_a_eval, s_r_prime_eval, num_steps, bs, i_optim):
        model_loss_avg = 0
        last_batch = False
        i = 0
        while i < num_steps:
            x, new_state_r_th = self.data_buffer.get_batch(bs)
            x, new_state_r_th = group_by_actions(x, new_state_r_th)  # sorting is necessary since model_joint is

            new_state_r_model = self.model(x)

            # compute loss
            model_loss = torch.abs(new_state_r_th - new_state_r_model).sum(1).mean()
            model_loss_avg += model_loss.item()

            # do gradient step
            self.model.optimizer.zero_grad()
            model_loss.backward()
            self.model.optimizer.step()

            i_optim += 1
            i += 1

        if num_steps > 0:
            # compute model_loss_avg and model_loss_eval
            model_loss_avg /= i
            new_state_r_model_eval = self.model(s_a_eval)
            model_loss_eval = torch.abs(s_r_prime_eval - new_state_r_model_eval).sum(1).mean().item()

            return model_loss_avg, model_loss_eval, i_optim
        else:
            return 0, 0, i_optim

    def plan(self, num_steps, i_plan, alpha, gamma):
        last_batch = False
        i = 0
        while not last_batch and i < num_steps:
            x, _, last_batch = self.data_buffer.get_batch_without_replacement(1)  # TODO: why a model? why not using directly the saved experience?
            state = x[:, :2]
            a = torch.dot(x[0, 2:], torch.arange(self.model.num_actions).float()).long()

            new_state_r_model = self.model(x, action=a)
            new_r_model = new_state_r_model[:, -1:]
            new_state_model = new_state_r_model[:, :-1]

            self.q.update(state, a, new_state_model, new_r_model, alpha, gamma)

            i_plan += 1
            i += 1
        return i_plan

# TODO: at the moment, the evaluation dataset was generated using random policy, not really representative
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dyna-Q for mountain-car')
    parser.add_argument('--exp-path', type=str, default=None,
                        help='experiment path where we save stuff')
    parser.add_argument('--model', type=str, default='causal',
                        help='causal|anticausal|joint|none')
    parser.add_argument('--bs-model', type=int, default=64,
                        help='batch size for training model')
    parser.add_argument('--buffer-size', type=int, default=4000,
                        help='size of the data buffer')
    parser.add_argument('--num-episodes', type=int, default=300,  # 300
                        help='number of episode before shift')
    parser.add_argument('--num-episodes-shift', type=int, default=500,  # 500
                        help='number of episode after shift')
    parser.add_argument('--num-bins', type=int, default=20,
                        help='number of bins for tile coding approximators')
    parser.add_argument('--num-tiles', type=int, default=10,
                        help='number of tiles for tile coding approximators')
    parser.add_argument('--max-num-step', type=int, default=200,
                        help='max length of episodes')
    parser.add_argument('--num-train-steps', type=int, default=100,  # 100
                        help='number of model training steps between each episodes')
    parser.add_argument('--num-plan-steps', type=int, default=1000,  # 1000
                        help='number of planning steps between each episodes')
    parser.add_argument('--lr', type=float, default=0.05,  # did some hparam search
                        help='learning rate for learning the model')
    parser.add_argument('--alpha', type=float, default=1.1,  # did some hparam search to find it
                        help='learning rate for RL')
    parser.add_argument('--gamma', type=float, default=1.,  # 1. makes sense in an episodic task
                        help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.,  # did some hparam search to find it
                        help='amount of exploration')
    parser.add_argument('--no-comet', action="store_true", help="disable comet.ml")
    opt = parser.parse_args()

    lr = opt.lr / opt.num_tiles
    alpha = opt.alpha / opt.num_tiles

    if opt.model == 'joint':
        model = FastTileCoding_Joint(3, opt.num_bins, opt.num_tiles,
                                          np.array([[-1.2, 0.6], [-0.07, 0.07]], dtype="float32"))
    elif opt.model == 'causal':
        model = FastTileCoding_Causal(3, opt.num_bins, opt.num_tiles,
                                          np.array([[-1.2, 0.6], [-0.07, 0.07]], dtype="float32"))
    elif opt.model == 'anticausal':
        model = FastTileCoding_AntiCausal(3, opt.num_bins, opt.num_tiles,
                                          np.array([[-1.2, 0.6],[-0.07, 0.07]], dtype="float32"))
    elif opt.model == 'none':
        model = FastTileCoding_Joint(3, opt.num_bins, opt.num_tiles,
                                     np.array([[-1.2, 0.6], [-0.07, 0.07]], dtype="float32"))
        opt.num_plan_steps, opt.num_train_steps = 0, 0
    else:
        assert False, "{} is an unknown model".format(opt.model)

    # set up comet.ml
    comet_exp = Experiment(api_key="go2q9NSgpaDoutQIk5IFAWEOz", project_name='Causality-RL',
                           auto_param_logging=True,
                           auto_metric_logging=False, parse_args=True, auto_output_logging=None,
                           disabled=opt.no_comet)
    comet_exp.set_name("{}-nts{}nps{}al{}lr{}eps{}bs{}buf{}nb{}nt{}".format(opt.model, opt.num_train_steps,
                                                                       opt.num_plan_steps, opt.alpha, opt.lr,
                                                                       opt.epsilon, opt.bs_model, opt.buffer_size,
                                                                       opt.num_bins, opt.num_tiles))

    q = QTileCoding(3, opt.num_bins, opt.num_tiles, np.array([[-1.2, 0.6], [-0.07, 0.07]], dtype="float32"))

    agent = Agent(q, model, data_buffer=None)

    # init logging stuff
    rewards = []
    rewards_test = []
    rewards_std_test = []
    model_losses = []
    model_losses_eval = []
    i_optims = []
    i_plans = []
    i_envs = []

    i_optim = 0  # total num of gradient steps for models
    i_plan = 0  # total num of planning steps
    for i_episode in range(opt.num_episodes + opt.num_episodes_shift):
        # before shift
        if i_episode == 0:
            #test_agent(agent, num_episodes=5, eps=0., render=True)  # render behavior of the agent

            i_env = 0  # total num of real interactions (in steps)
            env = get_modified_MountainCarEnv(1)
            env._max_episode_steps = opt.max_num_step

            env.env.force *= 10
            # env.env.gravity *= 10

            # create dataset for evaluation
            s_a_eval, s_r_prime_eval, _ = get_evaluation_dataset(env, reward=True)

            agent.data_buffer = DataManager(opt.buffer_size, 5, 3, remove_random=False)

            model.optimizer = torch.optim.Adagrad(agent.model.parameters(), lr=lr)

        # after shift
        elif i_episode == opt.num_episodes:
            env.close()
            #test_agent(agent, num_episodes=5, eps=0., render=True)  # render behavior of the agent
            i_env = 0  # total num of real interactions (in steps)
            env = get_modified_MountainCarEnv(1)
            env._max_episode_steps = opt.max_num_step

            # create dataset for evaluation
            s_a_eval, s_r_prime_eval, _ = get_evaluation_dataset(env, reward=True)

            agent.data_buffer = DataManager(opt.buffer_size, 5, 3, remove_random=False)

            model.optimizer = torch.optim.Adagrad(agent.model.parameters(), lr=lr)

        total_reward = 0
        state = env.reset()
        for i_step in range(env._max_episode_steps):
            # env.render()
            state_th, _ = to_tensor(state)

            # take action
            action = agent.get_action(state_th, eps=opt.epsilon)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Save transition to the data buffer
            x, new_state_r_th = to_tensor(state, action, new_state, reward=reward)
            new_state_th = new_state_r_th[:, :-1]
            agent.data_buffer.add(x, new_state_r_th)

            # model-free RL
            agent.q.update(state_th, action, new_state_th, reward, alpha, opt.gamma)
            if np.isnan(agent.q.tile_codings_q[0].weights.detach().cpu().numpy()).any():
                import ipdb; ipdb.set_trace(context=5)

            if done: break
            i_env += 1
            state = new_state

        # model-based part
        if opt.model != 'none':
            # train model (once the buffer is fully refreshed)
            if agent.data_buffer.number_of_samples > 500:
                print("Training model")
                model_loss, model_loss_eval, i_optim = agent.train_model(s_a_eval, s_r_prime_eval, opt.num_train_steps,
                                                                         opt.bs_model, i_optim)
            else: model_loss, model_loss_eval = np.inf, np.inf
            # planning
            if model_loss_eval < 0.05:
                print("Planning")
                i_plan = agent.plan(opt.num_plan_steps, i_plan, alpha, opt.gamma)
        else:
            model_loss, model_loss_eval = 0, 0

        # test model performance
        if i_episode % 10 == 0:
            total_reward_test, total_reward_test_std = test_agent(agent, num_episodes=50, eps=0., render=False,
                                                                  after_shift=(i_episode >= opt.num_episodes))
            rewards_test.append([i_episode, total_reward_test])
            rewards_std_test.append([i_episode, total_reward_test_std])

        # logging
        print("Episode {} took {} steps".format(i_episode, i_step))
        model_losses.append(model_loss)
        model_losses_eval.append(model_loss_eval)
        rewards.append(total_reward)
        i_optims.append(i_optim)
        i_plans.append(i_plan)
        i_envs.append(i_env)
        comet_exp.log_metric('model-loss', model_losses[-1], step=i_episode)
        comet_exp.log_metric('model-loss-eval', model_losses_eval[-1], step=i_episode)
        comet_exp.log_metric('reward', rewards[-1], step=i_episode)
        if i_episode % 10 == 0: comet_exp.log_metric('reward-test', rewards_test[-1][1], step=i_episode)
        comet_exp.log_metric('i_optim', i_optims[-1], step=i_episode)
        comet_exp.log_metric('i_plan', i_plans[-1], step=i_episode)
        comet_exp.log_metric('i_episode', i_episode, step=i_episode)
        comet_exp.log_metric('i_env', i_envs[-1], step=i_episode)
        print('    model-loss', model_losses[-1])
        print('    model-loss-eval', model_losses_eval[-1])
        print('    reward', rewards[-1])
        if i_episode % 10 == 0: print('    reward-test', rewards_test[-1][1])
        print('    i_optim', i_optims[-1])
        print('    i_plan', i_plans[-1])

    # save
    if opt.exp_path is not None:
        f = open(os.path.join(opt.exp_path, "model_losses.pkl"), 'wb')
        pickle.dump(model_losses, f)
        f = open(os.path.join(opt.exp_path, "model_losses_eval.pkl"), 'wb')
        pickle.dump(model_losses_eval, f)
        f = open(os.path.join(opt.exp_path, "rewards.pkl"), 'wb')
        pickle.dump(rewards, f)
        f = open(os.path.join(opt.exp_path, "rewards_test.pkl"), 'wb')
        pickle.dump(rewards_test, f)
        f = open(os.path.join(opt.exp_path, "rewards_std_test.pkl"), 'wb')
        pickle.dump(rewards_std_test, f)
        f = open(os.path.join(opt.exp_path, "i_optims.pkl"), 'wb')
        pickle.dump(i_optims, f)
        f = open(os.path.join(opt.exp_path, "i_plans.pkl"), 'wb')
        pickle.dump(i_plans, f)
        f = open(os.path.join(opt.exp_path, "i_envs.pkl"), 'wb')
        pickle.dump(i_envs, f)

    env.close()
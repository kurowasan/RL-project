import math

from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.envs.classic_control.mountain_car
import torch

from model import ModelCausal, ModelJoint, FastTileCoding_Joint, FastTileCoding_Causal, FastTileCoding_AntiCausal, TileCoding_Joint
from approximators import FastTileCoding
from data import DataManager


def get_modified_MountainCarEnv(env=1):

    if env == 1:
        def _height(self, xs):
            return 0.05 / (xs + 1.2 + 1e-16) + np.sin(3 * xs) * .45 + .5

        def step(self, action):
            # only two lines are modified
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            position, velocity = self.state

            # 3rd modification
            reward = (position - 0.6) / 1.8

            # 1st modification
            velocity += (action - 1) * self.force + (-0.1 * (position + 1.2 + 1e-16)**(-2) + math.cos(3 * position)) * (-self.gravity)
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position += velocity
            position = np.clip(position, self.min_position, self.max_position)
            # 2nd modification
            #if (position == self.min_position and velocity < 0): velocity = 0

            done = bool(position >= self.goal_position)

            self.state = (position, velocity)
            return np.array(self.state), reward, done, {}

    elif env == 2:
        def _height(self, xs):
            return 0.05 / (xs + 1.2 + 1e-16) + np.sin(3 * xs + 1.7) * .45 + .5

        def step(self, action):
            # only two lines are modified
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            position, velocity = self.state
            # 1st modification
            velocity += (action - 1) * self.force + (-0.1 * (position + 1.2 + 1e-16)**(-2) + math.cos(3 * position + 1.7)) * (-self.gravity)
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position += velocity
            position = np.clip(position, self.min_position, self.max_position)
            # 2nd modification
            #if (position == self.min_position and velocity < 0): velocity = 0

            done = bool(position >= self.goal_position)
            reward = -1.0

            self.state = (position, velocity)
            return np.array(self.state), reward, done, {}
    else:
        assert False

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-1.1, high=0.6), 0.]) #self.np_random.uniform(low=-self.max_speed, high=self.max_speed)])
        return np.array(self.state)

    gym.envs.classic_control.mountain_car.MountainCarEnv._height = _height
    gym.envs.classic_control.mountain_car.MountainCarEnv.step = step
    #gym.envs.classic_control.mountain_car.MountainCarEnv.reset = reset

    return gym.make("MountainCar-v0")


def to_tensor(state, action=None, new_state=None, reward=None):
    if action is not None:
        action_one_hot = np.zeros(3)
        action_one_hot[action] = 1
        x = np.concatenate([state, action_one_hot], 0)
    else:
        x = state
    x_th = torch.as_tensor(x).unsqueeze(0).float()

    if new_state is not None:
        new_x_th = torch.as_tensor(new_state).unsqueeze(0).float()
    else:
        new_x_th = None

    if reward is not None:
        new_x_th = torch.cat([new_x_th, torch.Tensor([reward]).unsqueeze(0)], 1)
    return x_th, new_x_th


def group_by_actions(s_a, s_r_prime):
    num_actions = s_a.size(1) - s_r_prime.size(1) + 1
    action_one_hot = s_a[:, -num_actions:]
    one_hots = torch.eye(num_actions)
    s_as = []
    s_r_primes = []
    for a in range(num_actions):
        mask = torch.matmul(action_one_hot, one_hots[a]).byte()  # (bs,)
        s_as.append(torch.masked_select(s_a, mask.unsqueeze(1)).view(-1, s_a.size(1)))
        s_r_primes.append(torch.masked_select(s_r_prime, mask.unsqueeze(1)).view(-1, s_r_prime.size(1)))
    return s_as, torch.cat(s_r_primes, 0)


def get_evaluation_dataset(env, reward=False):
    s_a_list = []
    s_r_prime_list = []
    for i_episode in range(100):
        state = env.reset()
        for i_step in range(env._max_episode_steps):
            #env.render()
            action = env.action_space.sample()
            new_state, r, done, _ = env.step(action)
            # train on state, action, new_state
            x, new_state_r_th = to_tensor(state, action, new_state, reward=r)
            s_a_list.append(x)
            s_r_prime_list.append(new_state_r_th)

            if done: break
            state = new_state

    s_a_eval = torch.cat(s_a_list, 0)
    s_r_prime_eval = torch.cat(s_r_prime_list, 0)
    if not reward:
        benchmark_loss_eval = torch.abs(s_r_prime_eval - s_a_eval[:, :2]).sum(1).mean()

    s_a_eval, s_r_prime_eval = group_by_actions(s_a_eval, s_r_prime_eval)
    if reward:
        return s_a_eval, s_r_prime_eval, None
    else:
        return s_a_eval, s_r_prime_eval[:, :-1], benchmark_loss_eval

def test_agent(agent, num_episodes=5, after_shift=False, eps=0., render=False):
    rewards = []
    i_env = 0  # total num of real interactions
    # before shift
    if not after_shift:
        env = get_modified_MountainCarEnv(1)
        env._max_episode_steps = 200

        env.env.force *= 10
        # env.env.gravity *= 10
    # after shift
    else:
        env = get_modified_MountainCarEnv(1)
        env._max_episode_steps = 200

    for i_episode in range(num_episodes):
        total_reward = 0
        state = env.reset()
        for i_step in range(env._max_episode_steps):
            if render: env.render()
            state_th, _ = to_tensor(state)

            # take action
            action = agent.get_action(state_th, eps=eps)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done: break
            i_env += 1
            state = new_state

        rewards.append(total_reward)

    env.close()

    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    bs = 64
    buffer_size = int(1e4)
    maximal_eps_duration_eval = 200
    maximal_eps_duration_train = 200
    num_episodes = 500
    num_episodes_shift = 500
    # current best
    # num_bins = 20
    # num_tiles = 10
    # lr = 0.01 / num_tiles
    num_bins = 20
    num_tiles = 10
    lr = 0.01 / num_tiles

    # set up comet.ml
    comet_exp = Experiment(api_key="go2q9NSgpaDoutQIk5IFAWEOz", project_name='Causality-RL', auto_param_logging=True,
                           auto_metric_logging=False, parse_args=True, auto_output_logging=None)

    model_joint = FastTileCoding_Joint(3, num_bins, num_tiles, np.array([[-1.2, 0.6],[-0.07, 0.07]], dtype="float32"))
    model_correct = FastTileCoding_Causal(3, num_bins, num_tiles, np.array([[-1.2, 0.6], [-0.07, 0.07]], dtype="float32"))
    model_incorrect = FastTileCoding_AntiCausal(3, num_bins, num_tiles, np.array([[-1.2, 0.6], [-0.07, 0.07]], dtype="float32"))

    #import ipdb; ipdb.set_trace(context=5)
    i_optim = 0  #total num of gradient steps
    for i_episode in range(num_episodes + num_episodes_shift):
        if i_episode == 0:
            i = 0
            env = get_modified_MountainCarEnv(1)

            # create dataset for evaluation
            env._max_episode_steps = maximal_eps_duration_eval
            s_a_eval, s_prime_eval, benchmark_loss_eval = get_evaluation_dataset(env)
            env._max_episode_steps = maximal_eps_duration_train

            data_buffer = DataManager(buffer_size, 5, 2, remove_random=False)

            optimizer_joint = torch.optim.Adagrad(model_joint.parameters(), lr=lr)
            optimizer_correct = torch.optim.Adagrad(model_correct.parameters(), lr=lr)
            optimizer_incorrect = torch.optim.Adagrad(model_incorrect.parameters(), lr=lr)
        elif i_episode == num_episodes:
            env.close()
            i = 0  # total num of steps
            env = get_modified_MountainCarEnv(1)

            env.env.force *= 10
            #env.env.gravity *= 10

            # create dataset for evaluation
            env._max_episode_steps = maximal_eps_duration_eval
            s_a_eval, s_prime_eval, benchmark_loss_eval = get_evaluation_dataset(env)
            env._max_episode_steps = maximal_eps_duration_train

            data_buffer = DataManager(buffer_size, 5, 2, remove_random=False)

            optimizer_joint = torch.optim.Adagrad(model_joint.parameters(), lr=lr)
            optimizer_correct = torch.optim.Adagrad(model_correct.parameters(), lr=lr)
            optimizer_incorrect = torch.optim.Adagrad(model_incorrect.parameters(), lr=lr)

        state = env.reset()
        for i_step in range(env._max_episode_steps):
            #print(i_episode, i_step)
            #env.render()
            action = env.action_space.sample()
            new_state, _, done, _ = env.step(action)
            # Get next state prediction
            x, new_state_th = to_tensor(state, action, new_state)
            data_buffer.add(x, new_state_th)

            # once the buffer is fully refreshed, train model
            if i % buffer_size == buffer_size - 1:
                #import ipdb; ipdb.set_trace(context=5)
                print("Training model")
                #import ipdb; ipdb.set_trace(context=5)
                last_batch = False
                while not last_batch:
                    x, new_state_th, last_batch = data_buffer.get_batch_without_replacement(bs)
                    x, new_state_th = group_by_actions(x, new_state_th)  # sorting is necessary since model_joint is

                    new_state_joint = model_joint(x)
                    new_state_correct = model_correct(x)
                    new_state_incorrect = model_incorrect(x)

                    # compute loss
                    model_joint_loss = torch.abs(new_state_th - new_state_joint).sum(1).mean()
                    model_correct_loss = torch.abs(new_state_th - new_state_correct).sum(1).mean()
                    model_incorrect_loss = torch.abs(new_state_th - new_state_incorrect).sum(1).mean()
                    benchmark_loss = torch.abs(new_state_th - torch.cat(x, 0)[:, :2]).sum(1).mean()

                    if i_optim % 5 == 0:
                        with torch.no_grad():
                            # log on comet
                            metrics = {}
                            metrics['model-joint-loss'] = model_joint_loss.item()
                            metrics['model-correct-loss'] = model_correct_loss.item()
                            metrics['model-incorrect-loss'] = model_incorrect_loss.item()
                            metrics['benchmark-loss'] = benchmark_loss.item()
                            metrics['num-samples-in-buffer'] = data_buffer.number_of_samples

                            # eval
                            s_prime_joint = model_joint(s_a_eval)
                            s_prime_correct = model_correct(s_a_eval)
                            s_prime_incorrect = model_incorrect(s_a_eval)
                            model_joint_loss_eval = torch.abs(s_prime_eval - s_prime_joint).sum(1).mean()
                            model_correct_loss_eval = torch.abs(s_prime_eval - s_prime_correct).sum(1).mean()
                            model_incorrect_loss_eval = torch.abs(s_prime_eval - s_prime_incorrect).sum(1).mean()

                            metrics['model-joint-loss-eval'] = model_joint_loss_eval.item()
                            metrics['model-correct-loss-eval'] = model_correct_loss_eval.item()
                            metrics['model-incorrect-loss-eval'] = model_incorrect_loss_eval.item()
                            metrics['benchmark-loss-eval'] = benchmark_loss_eval.item()
                            print('model-joint-loss', model_joint_loss.item())
                            print('model-correct-loss', model_correct_loss.item())
                            print('model-incorrect-loss', model_incorrect_loss.item())
                            print('benchmark-loss', benchmark_loss.item())
                            print('num-samples-in-buffer', data_buffer.number_of_samples)

                            print('model-joint-loss-eval', model_joint_loss_eval.item())
                            print('model-correct-loss-eval', model_correct_loss_eval.item())
                            print('model-incorrect-loss-eval', model_incorrect_loss_eval.item())
                            print('benchmark-loss-eval', benchmark_loss_eval.item())
                            comet_exp.log_metrics(metrics, step=i_optim)

                    # do gradient step
                    optimizer_joint.zero_grad()
                    optimizer_correct.zero_grad()
                    optimizer_incorrect.zero_grad()
                    model_joint_loss.backward()
                    model_correct_loss.backward()
                    model_incorrect_loss.backward()
                    optimizer_joint.step()
                    optimizer_correct.step()
                    optimizer_incorrect.step()

                    i_optim += 1

            if done: break
            i += 1
            state = new_state

    env.close()


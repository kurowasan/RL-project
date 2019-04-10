import math

from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.envs.classic_control.mountain_car
import torch

from model import ModelCausal, ModelJoint
from data import DataManager

def get_modified_MountainCarEnv():

    def _height(self, xs):
        return 0.05 / (xs + 1.2 + 1e-16) + np.sin(3 * xs) * .45 + .5

    def step(self, action):
        # only two lines are modified
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        # 1st modification
        velocity += (action - 1) * self.force + (-0.1 * (position + 1.2)**(-2) + math.cos(3 * position)) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        # 2nd modification
        #if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-1.1, high=0.6), 0.]) #self.np_random.uniform(low=-self.max_speed, high=self.max_speed)])
        return np.array(self.state)

    gym.envs.classic_control.mountain_car.MountainCarEnv._height = _height
    gym.envs.classic_control.mountain_car.MountainCarEnv.step = step
    gym.envs.classic_control.mountain_car.MountainCarEnv.reset = reset

    return gym.make("MountainCar-v0")


def to_tensor(state, action, new_state):
    action_one_hot = np.zeros(3)
    action_one_hot[action] = 1
    x = np.concatenate([state, action_one_hot], 0)
    x_th = torch.as_tensor(x).unsqueeze(0).float()

    new_state_th = torch.as_tensor(new_state).unsqueeze(0).float()

    return x_th, new_state_th


def get_evaluation_dataset(env):
    s_a_list = []
    s_prime_list = []
    for i_episode in range(1000):
        state = env.reset()
        for i_step in range(env._max_episode_steps):
            action = env.action_space.sample()
            new_state, _, done, _ = env.step(action)
            # train on state, action, new_state
            x, new_state_th = to_tensor(state, action, new_state)
            s_a_list.append(x)
            s_prime_list.append(new_state_th)

            if done: break
            state = new_state

    s_a_eval = torch.cat(s_a_list, 0)
    s_prime_eval = torch.cat(s_prime_list, 0)
    benchmark_loss_eval = torch.abs(s_prime_eval - s_a_eval[:, :2]).sum(1).mean()

    return s_a_eval, s_prime_eval, benchmark_loss_eval


if __name__ == "__main__":
    bs = 32
    buffer_size = int(1e4)
    maximal_eps_duration_eval = 200
    maximal_eps_duration_train = 200
    num_episodes = 1000
    num_epidodes_shift = 1000

    # set up comet.ml
    comet_exp = Experiment(api_key="go2q9NSgpaDoutQIk5IFAWEOz", project_name='Causality-RL', auto_param_logging=True,
                           auto_metric_logging=False, parse_args=True, auto_output_logging=None)

    data_buffer = DataManager(buffer_size, 5, 2, remove_random=False)

    model_joint = ModelJoint(1, 20)
    model_correct = ModelCausal(1, 20, correct=True)
    model_incorrect = ModelCausal(1, 20, correct=False)

    optimizer_joint = torch.optim.RMSprop(model_joint.parameters(), lr=0.001)
    optimizer_correct = torch.optim.RMSprop(model_correct.parameters(), lr=0.001)
    optimizer_incorrect = torch.optim.RMSprop(model_incorrect.parameters(), lr=0.001)

    env = get_modified_MountainCarEnv()
    #import ipdb; ipdb.set_trace(context=5)

    # create dataset for evaluation
    env._max_episode_steps = maximal_eps_duration_eval
    s_a_eval, s_prime_eval, benchmark_loss_eval = get_evaluation_dataset(env)

    env._max_episode_steps = maximal_eps_duration_train
    i = 0  # total num of steps
    i_optim = 0  #total num of gradient steps
    for i_episode in range(num_episodes + num_epidodes_shift):
        if i_episode == num_episodes:
            import ipdb; ipdb.set_trace(context=5)
            env._max_episode_steps = maximal_eps_duration_eval
            # modify environment
            env.env.gravity *= 2
            s_a_eval, s_prime_eval, benchmark_loss_eval = get_evaluation_dataset(env)


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
                print("Training model")
                #import ipdb; ipdb.set_trace(context=5)
                last_batch = False
                while not last_batch:
                    x, new_state_th, last_batch = data_buffer.get_batch_without_replacement(bs)
                    #import ipdb; ipdb.set_trace(context=5)
                    new_state_joint = model_joint(x)
                    new_state_correct = model_correct(x)
                    new_state_incorrect = model_incorrect(x)

                    # compute loss
                    model_joint_loss = torch.abs(new_state_th - new_state_joint).sum(1).mean()
                    model_correct_loss = torch.abs(new_state_th - new_state_correct).sum(1).mean()
                    model_incorrect_loss = torch.abs(new_state_th - new_state_incorrect).sum(1).mean()
                    benchmark_loss = torch.abs(new_state_th - x[:, :2]).sum(1).mean()

                    if i_optim % 50 == 0:
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


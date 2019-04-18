import numpy as np
import utils
import environment
import model
import control
import argument_parser as arg
import matplotlib.pyplot as plt


if __name__ == '__main__':
    hparam = arg.parse()
    rl_mode = True

    nb_total_step = hparam['nb_episode'] * hparam['nb_step']
    likelihood_a2b = np.zeros((hparam['nb_run'], nb_total_step))
    likelihood_b2a = np.zeros((hparam['nb_run'], nb_total_step))
    reward_a2b = np.zeros((hparam['nb_run'], nb_total_step))
    reward_b2a = np.zeros((hparam['nb_run'], nb_total_step))

    nb_total_step = hparam['nb_episode_adapt'] * hparam['nb_step']
    likelihood_a2b_adapt = np.zeros((hparam['nb_run'], nb_total_step))
    likelihood_b2a_adapt = np.zeros((hparam['nb_run'], nb_total_step))
    reward_a2b_adapt = np.zeros((hparam['nb_run'], nb_total_step))
    reward_b2a_adapt = np.zeros((hparam['nb_run'], nb_total_step))

    for run in range(hparam['nb_run']):
        print(f'run #{run}')
        env = environment.CausalEnvironment(hparam['state_dim'],
                                            hparam['action_dim'],
                                            hparam['nb_step'],
                                            hparam['peak'])

        if rl_mode:
            model_a2b = model.EnvironmentModel(hparam['state_dim'], hparam['action_dim'], 1,
                                               hparam['batch_size'], hparam['lr'], True)
            model_b2a = model.EnvironmentModel(hparam['state_dim'], hparam['action_dim'], 2,
                                               hparam['batch_size'], hparam['lr'], True)

            dyna_a2b = control.DynaQ(env, model_a2b)
            dyna_b2a = control.DynaQ(env, model_b2a)

            l_a2b, r_a2b = dyna_a2b.train(hparam['nb_episode'], 0)
            print('a->b finished')
            l_b2a, r_b2a = dyna_b2a.train(hparam['nb_episode'], 0)
            print('b->a finished')
            likelihood_a2b[run, :] = np.array(l_a2b)
            likelihood_b2a[run, :] = np.array(l_b2a)
            reward_a2b[run, :] = np.array(r_a2b)
            reward_b2a[run, :] = np.array(r_b2a)
            print('Training finished')

            # reset all!
            env.adapt_a()
            dyna_a2b.reset()
            dyna_b2a.reset()
            model_a2b.reinitialize_optimizer(lr=1e-1)
            model_b2a.reinitialize_optimizer(lr=1e-1)

            l_a2b, r_a2b = dyna_a2b.train(hparam['nb_episode_adapt'], 1)
            print('a->b finished')
            l_b2a, r_b2a = dyna_b2a.train(hparam['nb_episode_adapt'], 1)
            print('b->a finished')
            likelihood_a2b_adapt[run, :] = np.array(l_a2b)
            likelihood_b2a_adapt[run, :] = np.array(l_b2a)
            reward_a2b_adapt[run, :] = np.array(r_a2b)
            reward_b2a_adapt[run, :] = np.array(r_b2a)

        else:
            likelihood = model.LikelihoodEstimators(hparam['state_dim'],
                                                    hparam['action_dim'],
                                                    hparam['batch_size'],
                                                    hparam['lr'])

            l_a2b, l_b2a = control.train_with_buffer(env, likelihood, hparam['nb_episode'])
            likelihood_a2b[run, :] = l_a2b
            likelihood_b2a[run, :] = l_b2a
            print('Training finished')

            test(env, likelihood)
            env.adapt_a()

            likelihood.reinitialize_optimizer(lr=1e-2)
            l_a2b, l_b2a = control.train_with_buffer(env, likelihood, hparam['nb_episode_adapt'])
            likelihood_a2b_adapt[run, :] = l_a2b
            likelihood_b2a_adapt[run, :] = l_b2a


    if rl_mode:
        reward_a2b = np.cumsum(reward_a2b, axis=1)
        reward_b2a = np.cumsum(reward_b2a, axis=1)
        reward_a2b_adapt = np.cumsum(reward_a2b_adapt, axis=1) + reward_a2b[:,-1]
        reward_b2a_adapt = np.cumsum(reward_b2a_adapt, axis=1) + reward_b2a[:,-1]
        utils.plot_reward(reward_a2b, reward_b2a, hparam['output'], True)
        utils.plot_reward_adapt(reward_a2b_adapt, reward_b2a_adapt, hparam['output'], True)

    utils.plot_training(likelihood_a2b, likelihood_b2a, hparam['output'], True)
    utils.plot_adaptation(likelihood_a2b_adapt, likelihood_b2a_adapt, hparam['output'], True)

def test(env, likelihood):
    env.compare_directions(likelihood)
    env.compare_likelihood(likelihood, 0, 0, 0, 0, 0)
    env.compare_likelihood(likelihood, 1, 0, 0, 0, 0)
    env.compare_likelihood(likelihood, 2, 0, 0, 0, 0)
    env.compare_likelihood(likelihood, 3, 0, 0, 0, 0)
    __import__('ipdb').set_trace()

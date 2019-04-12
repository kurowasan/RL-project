import numpy as np
import utils
import environment
import model
import control
import argument_parser as arg


if __name__ == '__main__':
    hparam = arg.parse()
    rl_mode = False
    save_every = 10

    likelihood_a2b = np.zeros((hparam['nb_run'], hparam['nb_episode']))
    likelihood_b2a = np.zeros((hparam['nb_run'], hparam['nb_episode']))
    reward_a2b = np.zeros((hparam['nb_run'], hparam['nb_episode']))
    reward_b2a = np.zeros((hparam['nb_run'], hparam['nb_episode']))

    likelihood_a2b_adapt = np.zeros((hparam['nb_run'], hparam['nb_episode_adapt']))
    likelihood_b2a_adapt = np.zeros((hparam['nb_run'], hparam['nb_episode_adapt']))
    reward_a2b_adapt = np.zeros((hparam['nb_run'], hparam['nb_episode_adapt']))
    reward_b2a_adapt = np.zeros((hparam['nb_run'], hparam['nb_episode_adapt']))

    for run in range(hparam['nb_run']):
        print(f'run #{run}')
        env = environment.CausalEnvironment(hparam['state_dim'],
                                            hparam['action_dim'],
                                            hparam['nb_step'])

        if rl_mode:
            likelihood_estimator_a2b = model.ModelInterface(hparam['state_dim'],
                                                          hparam['action_dim'],
                                                          True,
                                                          hparam['lr'])
            likelihood_estimator_b2a = model.ModelInterface(hparam['state_dim'],
                                                          hparam['action_dim'],
                                                          True,
                                                          hparam['lr'])
            env_model_a2b = model.EnvironmentModel(hparam['state_dim'],
                                               hparam['action_dim'],
                                               True)
            env_model_b2a = model.EnvironmentModel(hparam['state_dim'],
                                               hparam['action_dim'],
                                               True)
            dyna_a2b = control.DynaQ(env_model_a2b, likelihood_estimator_a2b, env)
            dyna_b2a = control.DynaQ(env_model_b2a, likelihood_estimator_b2a, env)

            l_a2b, r_a2b = dyna_a2b.train(hparam['nb_episode'], 1)
            print('a->b finished')
            l_b2a, r_b2a = dyna_b2a.train(hparam['nb_episode'], 1)
            likelihood_a2b[run, :] = l_a2b
            likelihood_b2a[run, :] = l_b2a
            # __import__('ipdb').set_trace()
            reward_a2b[run, :] = r_a2b
            reward_b2a[run, :] = r_b2a
            print('Training finished')

            # reset all!
            env.adapt_a()
            dyna_a2b.reset()
            dyna_b2a.reset()
            likelihood_estimator_a2b.reinitialize_optimizer(lr=1e-1)
            likelihood_estimator_b2a.reinitialize_optimizer(lr=1e-1)

            l_a2b, r_a2b = dyna_a2b.train(hparam['nb_episode_adapt'], 1)
            print('a->b finished')
            l_b2a, r_b2a = dyna_b2a.train(hparam['nb_episode_adapt'], 1)
            likelihood_a2b_adapt[run, :] = l_a2b
            likelihood_b2a_adapt[run, :] = l_b2a
            reward_a2b_adapt[run, :] = r_a2b
            reward_b2a_adapt[run, :] = r_b2a

        else:
            likelihood = model.LikelihoodEstimators(hparam['state_dim'],
                                                    hparam['action_dim'],
                                                    hparam['batch_size'],
                                                    hparam['lr'])

            l_a2b, l_b2a = control.train(env, likelihood, hparam['nb_episode'])
            likelihood_a2b[run, :] = l_a2b
            likelihood_b2a[run, :] = l_b2a
            print('Training finished')

            env.adapt_a()
            likelihood.reinitialize_optimizer(lr=1e-1)
            l_a2b, l_b2a = control.train(env, likelihood, hparam['nb_episode_adapt'])
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

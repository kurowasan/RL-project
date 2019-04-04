import numpy as np
import utils
import environment
import model
import control
import argument_parser as arg

if __name__ == '__main__':
    hparam = arg.parse()

    likelihood_a2b = np.zeros((hparam['nb_run'], hparam['nb_episode']))
    likelihood_b2a = np.zeros((hparam['nb_run'], hparam['nb_episode']))
    likelihood_a2b_adapt = np.zeros((hparam['nb_run'], hparam['nb_episode_adapt']))
    likelihood_b2a_adapt = np.zeros((hparam['nb_run'], hparam['nb_episode_adapt']))

    for run in range(hparam['nb_run']):
        print(f'run #{run}')
        env = environment.CausalEnvironment(hparam['state_dim'],
                                            hparam['action_dim'],
                                            hparam['nb_step'])

        likelihood = model.LikelihoodEstimators(hparam['state_dim'],
                                                hparam['action_dim'],
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

    utils.plot_training(likelihood_a2b, likelihood_b2a, hparam['output'])
    utils.plot_adaptation(likelihood_a2b_adapt, likelihood_b2a_adapt, hparam['output'])

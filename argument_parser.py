import os, sys
import json
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Causal RL experiments')
    parser.add_argument('--state-dim', type=int, default=20,
                        help='number of value each state can take')
    parser.add_argument('--action-dim', type=int, default=4,
                        help='number of possible actions')
    parser.add_argument('--nb-episode', type=int, default=1000,
                        help='number of episode used for training')
    parser.add_argument('--nb-episode-adapt', type=int, default=200,
                        help='number of episode used for adaptation')
    parser.add_argument('--nb-run', type=int, default=1,
                        help='number of independent run')
    parser.add_argument('--nb-step', type=int, default=50,
                        help='length of episodes')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='epsilon for the normalisation')

    parser.add_argument('--confidence-threshold', type=float, default=0.25,
                        help='confidence threshold for the DynaQ simulation')
    parser.add_argument('--nb-simulation', type=int, default=0,
                        help='number of simulations in DynaQ')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='size of batches for training')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--peak', type=int, default=1,
                        help='level of entropy for the environment distributions')
    parser.add_argument('--output', default='DYNA_Q/same_start/',
                        help='Relative path where the result will be logged')

    args = parser.parse_args()
    hparam = vars(args)
    save_config(hparam)

    return hparam

def save_config(hparam):
    if not os.path.exists(hparam['output']):
        os.makedirs(hparam['output'])
    with open(os.path.join(hparam['output'], 'config.json'), 'w') as f:
        json.dump(hparam, f, indent=4)

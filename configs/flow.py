"""Arguments for normalizing flows that is typically combined with generative args"""
import argparse
import os
import random
from macros import *
import sys


def get_flow_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flows')
    parser.add_argument( '--batch-size', type=int, default=100,
        help='input batch size for training (default: 100)')
    parser.add_argument( '--test-batch-size', type=int, default=1000,
        help='input batch size for testing (default: 1000)')
    parser.add_argument( '--epochs', type=int, default=1000,
        help='number of epochs to train (default: 1000)')
    parser.add_argument( '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument( '--flow', default='maf', help='flow to use: maf | realnvp | glow')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cond', action='store_true', default=False,
        help='train class conditional flow (only for MNIST)')
    parser.add_argument( '--num-blocks', type=int, default=5,
        help='number of invertible blocks (default: 5)')
    parser.add_argument( '--seed', type=int, default=1111, help='random seed (default: 1)')
    parser.add_argument( '--log-interval', type=int, default=1000,
        help='how many batches to wait before logging training status')

    # Data parameters
    parser.add_argument('--run_name', type=str, default='charptb_AF-AF')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--load_dir', type=str)
    parser.add_argument('--evaluate_only', action='store_true')
    # 'ptb'
    parser.add_argument('--dataset', type=str, default=PTB_ROOT)
    parser.add_argument('--dset',default='POWER',
        help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS')
    parser.add_argument('--nll_every', type=int, default=5)
    parser.add_argument('--indep_bernoulli', action='store_true')
    parser.add_argument('--noT_condition', action='store_true')

    # Optimization parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--B_train', type=int, default=15)
    parser.add_argument('--B_val', type=int, default=15)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--seed', type=int)

    # KL/LR schedule parameters
    parser.add_argument('--initial_kl_zero', type=int, default=4)
    parser.add_argument('--kl_rampup_time', type=int, default=10)
    parser.add_argument('--patience', type=int, default=1)

    # Sample parameters
    parser.add_argument('--ELBO_samples', type=int, default=10)
    parser.add_argument('--nll_samples', type=int, default=30)

    # General model parameters
    parser.add_argument('--model_type', type=str, default='discrete_flow')
    parser.add_argument('--inp_embedding_size', type=int, default=500)
    parser.add_argument('--zsize', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dlocs', nargs='*', default=['prior_rnn'])
    parser.add_argument('--notie_weights', action='store_true')

    # Inference network parameters
    parser.add_argument('--q_rnn_layers', type=int, default=2)

    # Generative network parameters
    ## Prior
    parser.add_argument('--prior_type', type=str, default='AF')
    parser.add_argument('--p_ff_layers', type=int, default=0)
    parser.add_argument('--p_rnn_layers', type=int, default=2)
    parser.add_argument('--p_rnn_units', type=int, default=500)
    parser.add_argument('--p_num_flow_layers', type=int, default=1)
    parser.add_argument('--transform_function', type=str, default='nlsq')

    ### Prior MADE Flow
    parser.add_argument('--nohiddenflow', action='store_true')
    parser.add_argument('--hiddenflow_layers', type=int, default=2)
    parser.add_argument('--hiddenflow_units', type=int, default=100)
    parser.add_argument('--hiddenflow_flow_layers', type=int, default=5)
    parser.add_argument('--hiddenflow_scf_layers', action='store_true')

    ## Likelihood parameters
    parser.add_argument('--gen_bilstm_layers', type=int, default=2)

    args = parser.parse_args()

    if args.dlocs is None:
        setattr(args, 'dlocs', [])

    setattr(args, 'savedir', args.output_dir+'/'+args.run_name+'/saves/')
    setattr(args, 'logdir', args.output_dir+'/'+args.run_name+'/logs/')

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    if args.seed is None: setattr(args, 'seed', random.randint(0, 1000000))
    return args
"""configuration settings for awd-lstm with fraternal dropout"""
import argparse
import time
from macros import *

def get_awd_lstm_args():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default=WIKITEXT2_ROOT,
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='custom',
                        help='FD|ELD|PM for prepared mods or anything else for custom settings')
    parser.add_argument('--emsize', type=int, default=655,
                        help='size of word embeddings')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to the RNN output (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=321,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='not use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str,  default='output/',
                        help='dir path to save the log and the final model')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--name', type=str,  default=randomhash,
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=0,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--kappa', type=float, default=0,
                        help='kappa penalty for hidden states discrepancy (kappa = 0 means no penalty)')
    parser.add_argument('--double_target', action='store_true',
                        help='use target for the auxiliary network as well')
    parser.add_argument('--eval_auxiliary', action='store_true',
                        help='forward auxiliary network in evaluation mode (without dropout)')
    parser.add_argument('--same_mask_e', action='store_true',
                        help='use the same dropout mask for removing words from embedding layer in both networks')
    parser.add_argument('--same_mask_i', action='store_true',
                        help='use the same dropout mask for input embedding layers in both networks')
    parser.add_argument('--same_mask_w', action='store_true',
                        help='use the same dropout mask for the RNN hidden to hidden matrix in both networks')
    parser.add_argument('--same_mask_o', action='store_true',
                        help='use the same dropout mask for the RNN output in both networks')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    args = parser.parse_args()
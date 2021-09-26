import argparse
import time

def get_analysis_args():
    parser = argparse.ArgumentParser(description='Test language model on perturbed inputs')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    # Consistent with Language Modeling code from Merity et al.
    parser.add_argument('--cuda', action='store_false',
                        help='Using this flag turns off CUDA, default value set to True')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--logdir', type=str, default='./',
                        help='location to write per token log loss')
    parser.add_argument('--use_test', action='store_true', default=False,
                        help='Run on test set')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size')
    parser.add_argument('--seq_len', type=int, default=300,
                        help='sequence length')
    parser.add_argument('--max_seq_len', type=int, default=1000,
                        help='max context size')
    parser.add_argument('--freq', type=int, default=108,
                        help='Frequent words cut-off, all words with corpus count > 800')
    parser.add_argument('--rare', type=int, default=1986,
                        help='Rare words cut-off, all words with corpus count < 50')
    parser.add_argument('--func', type=str,
                        help='random_drop_many, drop_pos, keep_pos, shuffle, shuffle_within_spans, reverse, reverse_within'
                             '_spans, replace_target, replace_target_with_nearby_token, drop_target, replace_with_rand_seq')
    parser.add_argument('--span', type=int, default=20,
                        help='For shuffle and reverse within spans')
    parser.add_argument('--drop_or_replace_target_window', type=int, default=300,
                        help='window for drop or replace target experiments')
    parser.add_argument('--n', type=float,
                        help='Fraction of tokens to drop, between 0 and 1')
    # Specify a list
    parser.add_argument('--pos', action='append', default=None,
                        help='Pos tags to drop. Sample usage is --pos NN --pos VB --pos JJ')
    parser.add_argument('--use_range', action='append', default=None,
                        help='Use these values for the boundary loop, but first convert to ints. '
                             'Sample usage is --use_range 5 --use_range 20 --use_range 100')
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--pretrained', type=bool, default=None,
                        help='whether to use pretrained embeddings or not')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for recurrent layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--tied', action='store_false',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')

    args = parser.parse_args()
    return args
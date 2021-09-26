import argparse


"""
vocab_size=None, dec_hsz=500, rnn_layers=3,
                          bsz=128, max_len=51, dropout=0.2, use_cuda=True,
                          data= None, path='data/', encoder = 'resnet'
"""

def get_actor_critic_args(vocab_size=None, dec_hsz=500, rnn_layers=3,
                          bsz=128, max_len=51, dropout=0.2, use_cuda=True,
                          data= 'flickr30k', encoder = 'resnet'):
    parser = argparse.ArgumentParser()

    root = 'D:/data/{}/pretrained_models/actor_critic/'.format(data)

    parser.add_argument('--root', type=str, default=root, help='root')
    parser.add_argument('--vocab_size', type=int, default=vocab_size,
                        help='vocab size if not already specified')
    parser.add_argument('--dec_hsz', type=int, default=dec_hsz)
    parser.add_argument('--encoder', type=str, default=encoder,
                        help='resnet or polar. If polar, we need to change the images'
                                ' to transform to shape that polar net expects')
    parser.add_argument('--rnn_layers', type=int, default=rnn_layers)
    parser.add_argument('--batch_size', type=int, default=bsz)
    parser.add_argument('--max_len', type=int, default=max_len)
    parser.add_argument('--dropout', type=float, default=dropout)
    parser.add_argument('--use_cuda', type=bool, default=use_cuda)
    parser.add_argument('--logdir', type=str, default=root+'tb_logdir/')
    parser.add_argument('--seed', type=int, default=1111)
    # parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--data', type=str, default=data)
    parser.add_argument('--save', type=str, default=root+'imgcapt_v2_{}.pt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--new_lr', type=float, default=5e-6)
    parser.add_argument('--actor_epochs', type=int, default=2)
    parser.add_argument('--critic_epochs', type=int, default=3)
    parser.add_argument('--critic_loss', type=str, default='mse',
                        help='could be mse or kl (cross entropy)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--critic_reward', type=str, default='rouge',
                        help='called critic_reward instead of reward because '
                             'the latter refers to reward used to evaluate the model')
    parser.add_argument('--reward_dist', type=str, default='cosine',
                        help='choose distance measure to use between pred and targ sentence reps'
                             'choices: cosine, manhattan, euclidean')
    args = parser.parse_args()
    return args
import argparse


def get_prune_args(data_path = 'D:/data/mnist/', model = 'lstm', thresh=0.5):

    # Hyper parameters
    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('--data_path', default=data_path, type=str)
    parser.add_argument('--model', default=model, type=str)
    parser.add_argument('--sequence_length', default=28, type=int)
    parser.add_argument('--input_size', default=28, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--threshold', default=thresh, type=float)

    # below for fisher compress from - https://github.com/BayesWatch/pytorch-prunes/blob/master/prune.py
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
    parser.add_argument('--save_file', default='wrn16_2_p', type=str, help='save file for checkpoints')
    parser.add_argument('--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=True, help='resume from checkpoint')
    parser.add_argument('--resume_ckpt', default='checkpoint', type=str,
                        help='save file for resumed checkpoint')
    parser.add_argument('--data_loc', default='/disk/scratch/datasets/cifar', type=str, help='where is the dataset')
    parser.add_argument('--map_loc', default='cpu', type=str, help='where checkpoint is')
    # Learning specific arguments
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', type=str, help='optimizer')
    parser.add_argument('-epochs', '--no_epochs', default=1300, type=int, metavar='epochs', help='no. epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=0.0005, type=float, metavar='W', help='weight decay')
    parser.add_argument('--prune_every', default=100, type=int, help='prune every X steps')
    parser.add_argument('--save_every', default=100, type=int, help='save model every X EPOCHS')
    parser.add_argument('--random', default=False, type=bool, help='Prune at random')
    parser.add_argument('--base_model', default='base_model', type=str, help='basemodel')
    parser.add_argument('--val_every', default=1, type=int, help='val model every X EPOCHS')
    parser.add_argument('--mask', default=1, type=int, help='Mask type')
    parser.add_argument('--l1_prune', default=False, type=bool, help='Prune via l1 norm')
    parser.add_argument('--net', default='dense', type=str, help='dense, res')
    parser.add_argument('--width', default=2.0, type=float, metavar='D')
    parser.add_argument('--depth', default=40, type=int, metavar='W')
    parser.add_argument('--growth', default=12, type=int, help='growth rate of densenet')
    parser.add_argument('--transition_rate', default=0.5, type=float, help='transition rate of densenet')

    args = parser.parse_args()
    return args
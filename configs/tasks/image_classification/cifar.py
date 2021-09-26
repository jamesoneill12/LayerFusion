import argparse


def get_cifar_config(model='resnet18', n_epochs = 100, bsize = 56, learning_rate = 0.01,
                     decay_step=30, lr_decay=0.1, retrain=False, start_from_pretained=True, nbits = 8,
                     layers_to_prune=['linear', 'conv'], fuse_method='mix', compress_method = None,
                     fuse_distance='cosine', compress_perc=0.0, num_compress_steps=10, w_thresh=1e-6):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=learning_rate, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default=model, choices=['resnet18', 'preact_resnet18',
                                                           'lenet', 'densenet', 'resnext',
                                                           'mobilenet', 'mobilenet_v2', 'dpn92', 'shufflenet_g2',
                                                           'senet', 'shufflenet_v2' 'efficientnet'])
    parser.add_argument('--checkpoint', default="D:/data/cifar10/pretrained_models/")
    parser.add_argument('--bsize', default=bsize, type=int)
    parser.add_argument('--epochs', default=n_epochs, type=int)
    parser.add_argument('--lr_decay_step', default=decay_step, type=int, help="reduces learning rate by a"
                                                                              " factor of 0.1 every decay step")
    parser.add_argument('--lr_decay', default=lr_decay, type=float)


    # compression args

    parser.add_argument("--start_from_pretrained", default=start_from_pretained)
    parser.add_argument("--merge_measure", default='cov', type=str, help="which similarity metric to use when merging layers")
    parser.add_argument("--prune_threshold", default=w_thresh, type=float,
                        help="instead of pruning a % weights, sets a threshold for weights to be over, else pruned")
    parser.add_argument("--num_bits", default=nbits, type=bool,
                        help="if compress_method=quantize, threshold chose according to percentage for all weights combined")
    parser.add_argument('--retrain', default=retrain, type=bool, help="training with compression")
    parser.add_argument('--train', default=retrain, type=bool, help="training with no compression")
    parser.add_argument('--layers_to_prune', default=layers_to_prune, type=list)
    parser.add_argument('--compress_method',
                        default=compress_method, type=str, choices=[None, 'global_prune',
                                                                                         'layer_prune',
                                                                                         'threshold_prune',
                                                                                         'svd',
                                                                                         'autoencode',
                                                                                         'fuse'
                                                                                         'quantize'])
    parser.add_argument('--fuse_method', default=fuse_method, type=str,
                        help="if fuse the compress_method, then choose fuse type", choices = ['average', 'mix', 'drop'])
    parser.add_argument('--fuse_distance', default=fuse_distance, type=str)
    parser.add_argument('--compress_step', default=num_compress_steps, type=int)
    parser.add_argument('--compress_perc', default=compress_perc/num_compress_steps , type=float,
                        help='spreads the amount of pruning evenely over prune steps')
    args = parser.parse_args()
    return args


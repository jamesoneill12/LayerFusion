import argparse


def get_alexnet_args(model_names):
    parser = argparse.ArgumentParser(
        description='PyTorch AlexNet Prune & Retraining')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: alexnet)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--prune', default=90., type=float,
                        help='Pruning percentage')
    parser.add_argument('--logfolder', default='', type=str,
                        help='Output Log and Chkpoint folder')
    parser.add_argument('--snapshot', default=1000, type=int,
                        help='Snapshot frequency')
    parser.add_argument('--iterative', action='store_true',
                        help='Resets epoch and perform compress')
    parser.add_argument('--validate', action='store_true',
                        help='Validate accuracy on resume')
    # i currently do not have imagenet downloaded, better to use pretrained models
    parser.add_argument('--train_dir', default='D:/data/imagenet/',
                        help='Validate accuracy on resume')

    args = parser.parse_args()
    return args
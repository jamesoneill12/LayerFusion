from models.networks.compress.prune.lenet import *
from torch.autograd import Variable


def iter_prune(args, data_func=get_loaders, model_class=None, **kwargs):

    trainloader, val_loader, test_loader = data_func(args)
    topk = tuple(sorted(args.topk)) # sort in ascending order

    epoch = 1
    best_loss = sys.maxsize

    if args.arch:
        model = models.__dict__[args.arch](args.pretrained)
        optimizer = optim.__dict__[args.optimizer](model.parameters(), lr=args.lr)
        # TODO: check if the optimizer needs momentum and weight decay
        if args.optimizer == 'SGD':
            optimizer.momentum = args.momentum
            optimizer.weight_decay = args.weight_decay
    else:
        assert model_class is not None
        model = model_class(**kwargs)

        optimizer = optim.__dict__[args.optimizer](model.parameters(), lr=args.lr)
        # TODO: check if the optimizer needs momentum and weight decay
        if args.optimizer == 'SGD':
            optimizer.momentum = args.momentum
            optimizer.weight_decay = args.weight_decay

        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch'] % args.max_epochs + 1
            best_loss = checkpoint['best_loss']
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                raise e

    print(model.__class__)

    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    compressor = Compressor(model, cuda=args.cuda)

    model.train()
    pct_pruned = 0.0
    scores = [AverageMeter() for _ in topk]

    val_scores = [0.0 for _ in topk]

    while True:
        if epoch == 1:
            if args.compress == 'prune': new_pct_pruned = compressor.prune()
            elif args.compress == 'fuse': new_pct_pruned = compressor.fuse()

            logging.info('Pruned %.3f %%' % (100 * new_pct_pruned))

            top_accs = validate(model, val_loader, topk, cuda=args.cuda)

            if new_pct_pruned - pct_pruned <= 0.001 and converged(val_scores, top_accs):
                break

            pct_pruned = new_pct_pruned
            val_scores = top_accs

        for e in range(epoch, args.max_epochs + 1):
            for i, (input, label) in enumerate(trainloader, 0):
                input, label = Variable(input), Variable(label)

                if args.cuda:
                    input, label = input.cuda(), label.cuda()

                optimizer.zero_grad()
                output = model(input)
                precisions = accuracy(output, label, topk)

                for i, s in enumerate(scores):
                    s.update(precisions[i][0], input.size(0))

                loss = criterion(output, label)
                loss.backward()

                if args.compress == 'prune':
                    compressor.set_grad()
                elif args.compress == 'fuse':
                    compressor.set_fuse_grad()

                optimizer.step()

            if e % args.interval == 0:
                checkpoint = {
                    'state_dict': model.module.state_dict()
                    if args.cuda else model.state_dict(),
                    'epoch': e,
                    'best_loss': max(best_loss, loss.item()),
                    'optimizer': optimizer.state_dict()
                }

                save_checkpoint(checkpoint, is_best=(loss.item() < best_loss))

            if e % 30 == 0:
                # TODO: currently manually adjusting learning rate, could be changed to user input
                lr = optimizer.lr * 0.1
                adjust_learning_rate(optimizer, lr, verbose=True)

        epoch = 1

    test_topk = validate(model, test_loader, topk, cuda=args.cuda)


def main(model='mlp', dataset='mnist', compress_type='prune'):

    args = parse_args()
    args.compress = compress_type

    color = True
    if dataset == 'mnist':
        loader = get_mnist_loaders
        color = False
    if dataset == 'fashion-mnist':
        loader = get_fashion_loaders
        color = False
    elif dataset == 'svhn':
        loader = get_svhn_loaders
    elif dataset == 'cifar':
        loader = get_cifar10_loaders

    if model=='lenet':
        from models.networks.convolutional.lenet import LeNet_300_100 as Model
        iter_prune(args, data_func=loader, model_class=Model)
    elif model=='alexnet':
        from models.networks.convolutional.alexnet import AlexNet as Model
        # remember to remove num_classes when using LeNet_300_100 since it doesn't use it as an init arg.
        iter_prune(args, data_func=loader, model_class=Model, num_classes=10, color=color)
    elif model == 'mlp':
        from models.networks.task_specific.image.classification.mnist.model import MLP as Model
        iter_prune(args, data_func=loader, model_class=Model,
                   input_dims=784, n_hiddens=[256, 256, 256], n_class=10)


if __name__ == '__main__':
    models = ['mlp', 'lenet', 'alexnet']
    datasets = ['mnist', 'fashion-mnist', 'svhn', 'cifar']
    compress_types = ['fuse', 'prune']
    for model in models:
        for dataset in datasets:
            for compress_type in compress_types:
                print(f"{model}-{dataset}-{compress_type}")
                main(model, dataset, compress_type=compress_type)
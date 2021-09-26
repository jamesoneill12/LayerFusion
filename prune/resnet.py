from configs.pruning import get_prune_args
from models.networks.convolutional.wide_resnet import *
from models.networks.compress.prune.fischer import *
from models.networks.convolutional.densenet import *
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import torchvision
import torch
import time
import os


args = get_prune_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

device = torch.device("cuda:%s" % '0' if torch.cuda.is_available() else "cpu")


if args.net == 'res':
    model = WideResNet(args.depth, args.width, mask=args.mask)
elif args.net =='dense':
    model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, mask=args.mask)

model.load_state_dict(torch.load('checkpoints/%s.t7' % args.base_model,
                                 map_location='cpu')['state_dict'], strict=True)


if args.resume:
    state = torch.load('checkpoints/%s.t7' % args.resume_ckpt, map_location=args.map_loc)
    model.load_state_dict(state, model_type=args.net)
    error_history = state['error_history']
    prune_history = state['prune_history']
    flop_history = state['flop_history']
    param_history = state['param_history']
    start_epoch = state['epoch']

else:
    error_history = []
    prune_history = []
    param_history = []
    start_epoch = 0

model.to(device)

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

print('==> Preparing data..')
num_classes = 10

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    normTransform

])

trainset = torchvision.datasets.CIFAR10(root=args.data_loc,
                                        train=True, download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root=args.data_loc,
                                      train=False, download=True, transform=transform_val)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=False)

prune_count = 0
pruner = Pruner()
pruner.prune_history = prune_history

NO_STEPS = args.prune_every


def finetune():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    dataiter = iter(trainloader)

    for i in range(0, NO_STEPS):

        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Prunepoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, NO_STEPS, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def prune():
    print('Pruning')
    if args.random is False:
        if args.l1_prune is False:
            print('fisher compress')
            pruner.fisher_prune(model, prune_every=args.prune_every)
        else:
            print('l1 compress')
            pruner.l1_prune(model, prune_every=args.prune_every)
    else:
        print('random compress')
        pruner.random_prune(model, )


def freeze_resnet(model, layer_to_freeze):
    """
    layer_to_freeze should either be a list of ints
    or a list of name of parameters
    """
    ct = 0
    for child in model.children():
        ct += 1
    if ct < 7:
        for name, param in child.named_parameters():
            if name in layer_to_freeze:
                param.requires_grad = False


def validate():
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(valloader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(valloader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # Record Top 1 for CIFAR
    error_history.append(top1.avg)


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.no_epochs):

        print('Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])

        # finetune for one epoch
        finetune()
        # # evaluate on validation set
        if epoch != 0 and ((epoch % args.val_every == 0) or (epoch + 1 == args.no_epochs)):  # Save at last epoch!
            validate()

            # Error history is recorded in validate(). Record params here
            no_params = pruner.get_cost(model) + model.fixed_params
            param_history.append(no_params)

        # Save before compress
        if epoch != 0 and ((epoch % args.save_every == 0) or (epoch + 1 == args.no_epochs)):  #
            filename = 'checkpoints/%s_%d_prunes.t7' % (args.save_file, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'error_history': error_history,
                'param_history': param_history,
                'prune_history': pruner.prune_history,
            }, filename=filename)

        ## Prune
        prune()
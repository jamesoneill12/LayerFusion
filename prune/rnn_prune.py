from time import time
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.networks.compress.prune.rnn import *
from models.networks.compress.prune.basic import thresh_prune
from configs.pruning import get_prune_args


def get_dataset(args):
    test_dataset = dsets.MNIST(root=args.data_path,
                               train=False,
                               transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    # MNIST Dataset
    train_dataset = dsets.MNIST(root=args.data_path,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    return (train_dataset, train_loader, test_dataset, test_loader)


def get_model(args):
    if args.model == 'recurrent':
        rnn = RNN(args.input_size, args.hidden_size,
                  args.num_layers, args.num_classes)
    elif args.model == 'lstm':
        rnn = LSTM(args.input_size, args.hidden_size,
                   args.num_layers, args.num_classes)
    elif args.model == 'gru':
        rnn = GRU(args.input_size, args.hidden_size,
                  args.num_layers, args.num_classes)
    if torch.cuda.is_available():
        rnn.cuda()

    return rnn


def train_net(args, train_loader):

    rnn = get_model(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(rnn.parameters(), lr=args.learning_rate)

    # Train the Model
    for epoch in range(args.num_epochs):
        t0 = time()
        for i, (images, labels) in enumerate(train_loader):
            images = to_var(images.view(-1, args.sequence_length, args.input_size))
            labels = to_var(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Time: %.2f'
                      % (epoch + 1, args.num_epochs, i + 1,
                         len(train_dataset) // args.batch_size, loss.item(), \
                         time() - t0))
    return (rnn, criterion, optimizer)


def test(args, rnn, test_loader):

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = to_var(images.view(-1, args.sequence_length, args.input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100. * float(correct) / total))
    # Find out entries with weights smaller than a certain threshold


def test_post_prune(test_loader, rnn, args):
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = to_var(images.view(-1, args.sequence_length, args.input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    acc = 100. * float(correct) / total
    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (acc))
    return acc


def retrain_pruned(args, train_dataset, train_loader, optimizer,
                   rnn, criterion, pruned_inds_by_layer):
    # Re-initialize a GRU with previous weights pruned

    # Re-train the network but don't update zero-weights (by setting the corresponding gradients to zero)
    for epoch in range(args.num_epochs):
        t0 = time()
        for i, (images, labels) in enumerate(train_loader):
            images = to_var(images.view(-1, args.sequence_length, args.input_size))
            labels = to_var(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for l, p in enumerate(rnn.parameters()):
                pruned_inds = pruned_inds_by_layer[l]
                if type(pruned_inds) is not str:
                    p.grad.data[pruned_inds] = 0.

            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Time: %.2f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_dataset) // args.batch_size, loss.item(), \
                         time() - t0))

    return (rnn, optimizer)


def test_retrained_pruned(test_loader, rnn, args):
    # Test the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = to_var(images.view(-1, args.sequence_length, args.input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    acc = 100. * float(correct) / total
    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (acc))
    return acc


def prune_test(rnn, pruned_inds_by_layer, threshold):
    weight_tensors2 = []
    for p in rnn.parameters():
        pruned_inds = 'None'
        if len(p.data.size()) == 2:
            pruned_inds = p.data.abs() < threshold
            weight_tensors2.append(p.clone())
        pruned_inds_by_layer.append(pruned_inds)
    param = list(rnn.parameters())
    return pruned_inds_by_layer, param, weight_tensors2


def plot_pruned_subp(wt,  bins, pruned_inds_by_layer, i):
    plt.figure(i)

    j = 1
    if type(pruned_inds_by_layer[0]) != str:
        plt.subplot(3, 1, j)
        print(wt[0][0])
        print(wt[0][1])
        print()
        print()
        print(pruned_inds_by_layer[0].flatten())
        plt.hist(wt[0][1 - pruned_inds_by_layer[0].flatten()], bins=bins)
    if type(pruned_inds_by_layer[1]) != str:
        j += 1
        plt.subplot(3, 1, j)
        plt.hist(wt[1][1 - pruned_inds_by_layer[1].flatten()], bins=bins)
    if type(pruned_inds_by_layer[3]) != str:
        j += 1
        plt.subplot(3, 1, j)
        plt.hist(wt[2][1 - pruned_inds_by_layer[3].flatten()], bins=bins)


def plot_pruned(pruned_inds_by_layer, wt0=None, wt1=None, wt2=None, bins = 100):

    i = 0
    if wt0[0].is_cuda: wt0 = [wts0.cpu().data.numpy() for wts0 in wt0]
    if wt1[0].is_cuda: wt1 = [wts1.cpu().data.numpy() for wts1 in wt1]
    if wt2[0].is_cuda: wt2 = [wts2.cpu().data.numpy() for wts2 in wt2]

    #for i in range(4):
    #    print(pruned_inds_by_layer[i])
    #    print(type(pruned_inds_by_layer[i]))

    pruned_inds_by_layer = [pruned_inds_by_layer[i].cpu().numpy() if type(pruned_inds_by_layer[i]) != str
                            else pruned_inds_by_layer[i] for i in range(4)]

    if wt0 is not None:
        plt.figure(i)
        plt.subplot(3, 1, 1)
        plt.hist(wt0[0].flatten(), bins=bins)
        plt.subplot(3, 1, 2)
        plt.hist(wt0[1].flatten(), bins=bins)
        plt.subplot(3, 1, 3)
        plt.hist(wt0[2].flatten(), bins=bins)
        i += 1

    if wt1 is not None:
        plot_pruned_subp(wt1,  bins, pruned_inds_by_layer, i)
        i += 1

    if wt2 is not None:
        plot_pruned_subp(wt2,  bins, pruned_inds_by_layer, i)

    plt.show()
    # Look at the weights distribution again


if __name__ == "__main__":

    args = get_prune_args()
    train_dataset, train_loader, test_dataset, test_loader = get_dataset(args)
    rnn, criterion, optimizer = train_net(args, train_loader)
    test(args, rnn, test_loader)
    pruned_inds_by_layer, weight_tensors0, weight_tensors1 = thresh_prune(rnn, args.threshold)
    prune_acc = test_post_prune(test_loader, rnn, args)

    rnn, optimizer = retrain_pruned(args, train_dataset, train_loader, optimizer,
                   rnn, criterion, pruned_inds_by_layer)

    prune_retrained_acc = test_retrained_pruned(test_loader, rnn, args)
    pruned_inds_by_layer, param, weight_tensors2 = prune_test(rnn, pruned_inds_by_layer, args.threshold)
    plot_pruned(pruned_inds_by_layer, wt0=weight_tensors0, wt1=weight_tensors1, wt2=weight_tensors2)

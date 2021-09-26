""" ResNet prune """

import torch
from utils.metrics.accuracy import calculate_accuracy
from models.networks.convolutional.convolution import get_pretrained_cnn
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import random
import numpy as np


def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy(fx, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_cifar10():

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=3),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transforms)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transforms)
    n_train_examples = int(len(train_data) * 0.9)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])
    return train_data, valid_data, test_data


def run(args):

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')

    train_data, valid_data, test_data = get_cifar10()
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.bsize)
    valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=args.bsize)
    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=args.bsize)

    model = get_pretrained_cnn(args.model)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet-cifar10.pt')

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(
            f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:05.2f}% |')

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(model, device, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')


if __name__ == "__main__":

    run()
"""primarily cityscapes dataset provided by torchvision"""
from util.metrics.accuracy import calculate_accuracy
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
from loaders.segmentation.cityscapes import CityscapesSegmentation
from loaders.segmentation.seg_utils import decode_segmap
from torchvision import models
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch


def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for sample in iterator:
        x = sample['image'].to(device)
        y = sample['label'].type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        fx = model(x)

        print(fx)

        fx = fx['out']

        print(fx.size())
        print(type(fx))
        print(y.size())
        print(type(y))

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


def run(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda') if args.cuda else 'cpu'

    train_data = CityscapesSegmentation(args, split='train') # transform = train_transform
    valid_data = CityscapesSegmentation(args, split='val')
    test_data = CityscapesSegmentation(args, split='test')
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.bsize)
    valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=args.bsize)
    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=args.bsize)

    model = models.segmentation.fcn_resnet101(pretrained=True).to(device)
    # model = mods.segmentation.fcn_resnet50(True).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    SAVE_DIR = 'D:/data/{}/mods/'.format(args.dataset, args.model)
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, '{}.pt'.format(args.model))

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
            f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} |'
            f' Train Acc: {train_acc * 100:05.2f}% | Val. Loss: {valid_loss:.3f} |'
            f' Val. Acc: {valid_acc * 100:05.2f}% |'
        )

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(model, device, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')


def plot_img_seg():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.bsize = 64

    dataloader = CityscapesSegmentation(args, split='train')

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


if __name__ == "__main__":
    from configs.tasks.segmentation_config import get_segmentation_config
    args = get_segmentation_config(bsize=1, cuda=False)
    run(args)
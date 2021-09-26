import torch
import torch.nn as nn
from models.regularizers.dropconnect import ConcreteDropConnect


class Linear_relu(nn.Module):

    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class SigDrop(nn.Module):

    def __init__(self, inp, args):
        super(SigDrop, self).__init__()

        self.model = nn.Linear(inp, args.nhidden_caps)
        self.prior_probs = torch.rand(args.nhidden_caps, 1)
        self.drop = nn.Dropout()

    def forward(self, x, prob):
        x = self.drop(x, self.prior_probs)

        return self.model(x)


class SiameseNetwork(nn.Module):
    def __init__(self, args):
        super(SiameseNetwork, self).__init__()

        dims = [4, 8, 100, 500,
                args.nhidden_caps]  # if args.dataset == 'att' else [16, 32, 32, 73728, args.nhidden_caps]

        channels = 3 if args.dataset == 'wild' else 1

        self.concrete_dropout = args.concrete_dropout
        self.norm_dist = args.norm_dist

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, dims[0], kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dims[0]),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dims[0], dims[1], kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dims[1]),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dims[1], dims[1], kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dims[1]),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dims[1] * dims[2] * dims[2], 500),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(500, dims[4]))

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = ConcreteDropConnect(Linear_relu(20, 1), input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)

        if self.concrete_dropout:
            output = self.cd(output)
        if self.norm_dist:
            # cannot use output/= output.pow(...) in python 2.7 not supported
            output = torch.div(output, output.pow(2).sum(1, keepdim=True).sqrt())

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseAlexNet(nn.Module):

    def __init__(self, args):
        super(SiameseAlexNet, self).__init__()

        # should change based on dataset, last is num. of classes
        dims = [16, 32, 32, 73728,
                args.nhidden_caps]  # if args.dataset == 'att' else [16, 32, 32, 73728, args.nhidden_caps]

        channels = 3 if args.dataset == 'wild' else 1

        self.norm_dist = args.norm_dist
        self.concrete_dropout = args.concrete_dropout

        self.cnn1 = nn.Sequential(
            nn.Conv2d(channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dims[3], 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, dims[4]),
        )

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = ConcreteDropConnect(Linear_relu(20, 1), input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def forward_once(self, x):
        x = self.cnn1(x)
        x = x.view(x.size(0), -1)
        # print(x.cpu().size())
        x = self.fc1(x)

        if self.norm_dist:
            x = torch.div(x, x.pow(2).sum(1, keepdim=True).sqrt())

        if self.concrete_dropout:
            x = self.cd(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

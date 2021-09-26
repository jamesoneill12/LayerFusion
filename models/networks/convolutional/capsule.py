import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from models.regularizers.dropconnect import ConcreteDropConnect


BATCH_SIZE = 100
NUM_CLASSES = 5
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()




class Linear_relu(nn.Module):

    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS, cuda=False, squash=False):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.cuda = cuda
        self.num_capsules = num_capsules
        self.squasher = squash

        if squash == False:
            self.logit = nn.Sigmoid()

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
            logits = Variable(torch.zeros(*priors.size()))

            if self.cuda:
                logits = logits.cuda()

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)

                if self.squasher:
                    outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                else:
                    outputs = self.logit((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNetwork(nn.Module):
    def __init__(self, args):
        super(CapsuleNetwork, self).__init__()

        self.nhidden_caps = args.nhidden_caps
        self.recon = args.recon
        self.linear = args.linear
        self.num_convs = args.num_convs
        self.batch_norm = args.batch_norm
        self.args = args.squash

        self.norm_dist = args.norm_dist
        self.concrete_dropout = args.concrete_dropout

        if args.dataset == 'att':
            dims = [1, 9, 12, 16]
            capsule_route = 12
        # if wild is transformed to 50 x 50
        elif args.dataset == 'wild':
            dims = [3, 9, 12, 16]
            capsule_route = 12
            # dims = [3, 6, 12, 16]
            # capsule_route = 5
        # youtube face datasets choose
        elif args.dataset == 'ytf':
            dims = [3, 6, 12, 16]
        else:
            dims = [1, 6]

        if args.batch_norm:
            if self.num_convs == 1:

                self.conv1 = nn.Sequential(
                    # nn.ReflectionPad2d(1),
                    nn.Conv2d(dims[0], 256, kernel_size=9, stride=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(256)
                )

            elif self.num_convs == 2:
                self.conv1 = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(dims[0], 16, kernel_size=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(16),

                    nn.ReflectionPad2d(1),
                    nn.Conv2d(16, 32, kernel_size=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),

                    nn.ReflectionPad2d(1),
                    nn.Conv2d(32, 32, kernel_size=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),
                )
                self.conv2 = nn.Conv2d(in_channels=32,
                                       out_channels=32, kernel_size=2, stride=2)

        else:
            if self.num_convs > 1:
                self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=256, kernel_size=3, stride=3)
                self.conv2 = nn.Conv2d(in_channels=dims[0], out_channels=256, kernel_size=9, stride=3)
            else:
                self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=256, kernel_size=9, stride=3)

        if args.batch_norm and self.num_convs > 1:
            capsule_route = 21
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=32, out_channels=16,
                                                 kernel_size=dims[1], stride=2, cuda=True, squash=args.squash)
            if args.intermediate_convs:
                self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            self.face_capsules = CapsuleLayer(num_capsules=args.nhidden_caps,
                                              num_route_nodes=16 * capsule_route * capsule_route, in_channels=8,
                                              out_channels=dims[3], cuda=True, squash=args.squash)

        else:
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=dims[1], stride=2, cuda=True, squash=args.squash)
            self.face_capsules = CapsuleLayer(num_capsules=args.nhidden_caps,
                                              num_route_nodes=32 * capsule_route * capsule_route, in_channels=8,
                                              out_channels=dims[3], cuda=True, squash=args.squash)

        # should connect face_capsules -> fc1
        self.fc1 = nn.Linear(dims[3] * args.nhidden_caps, args.nhidden_caps) if self.linear else None


        if self.recon:
            self.decoder = nn.Sequential(
                nn.Linear(dims[3] * args.nhidden_caps, 128),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2048),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 10304),
                nn.Sigmoid()
            )

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = ConcreteDropConnect(Linear_relu(20, 1),
                                          input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def reconstruct(self, x1, x2):
        recon_1 = self.decoder((x1 ).view(x1.size(0), -1))
        recon_2 = self.decoder((x2).view(x2.size(0), -1))
        reconstructions = F.pairwise_distance(recon_1, recon_2)
        return reconstructions

    def forward_once(self, x):

        # print()
        # print(x.cpu().size())
        x = F.relu(self.conv1(x), inplace=True)

        # print()
        # print(x1.cpu().size())

        if self.num_convs > 1 and self.batch_norm:
            x = self.conv2(x)
            # print(x1.cpu().size())

        print("Conv Layer {}".format(x.cpu().size()))

        x = self.primary_capsules(x)

        # print()
        # print(x.cpu().size())
        print("Primary Capsule output {}".format(x.cpu().size()))
        x = self.face_capsules(x).squeeze().transpose(0, 1)
        print("Face capsule input {}".format(x.cpu().size()))
        print(x.cpu().size())

        # 16x20x7056x16 when sigmoid squasher
        # original squasher
        # print(x1.cpu().size())
        # print(x2.cpu().size())

        if self.fc1:
            classes = self.fc1(x.contiguous().view(-1, x.contiguous().size(1) * x.size(2)))
        if self.recon:
            classes = (x ** 2).sum(dim=-1) ** 0.5
        if self.norm_dist:
            classes = torch.div(classes, classes.pow(2).sum(1, keepdim=True).sqrt())
        if self.concrete_dropout:
            classes = self.cd(classes)

        return classes

    def forward(self, x1, x2, y=None):

        classes1 = self.forward_once(x1)
        classes2 = self.forward_once(x2)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes1.max(dim=1)

            if self.cuda:
                y = Variable(torch.sparse.torch.eye(self.nhidden_caps)).cuda().index_select(dim=0,
                                                                                            index=max_length_indices.data)
            else:
                y = Variable(torch.sparse.torch.eye(self.nhidden_caps))
                y = y.index_select(dim=0, index=max_length_indices)  # apparently should be this .data)

        reconstructions = self.reconstruct(x1, x2) if self.recon else None
        return classes1, classes2, reconstructions

    def _name(self):
        return "Capsule"
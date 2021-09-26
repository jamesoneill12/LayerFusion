"""This loss is a regularizer that ensures that the weights are """

import torch.nn as nn
import re


def get_s_name(name):
    """At least works for Wide ResNet"""
    s_name = name
    mo = re.match('.+([0-9])[^0-9]*$', s_name)
    s_name = list(s_name)
    s_name[int(mo.start(1))] = str(int(mo.group(1)) + 1)
    s_name = ''.join(s_name)
    return s_name


def convert_param_name(name):
    dot_cnt = name.count(".")

    if dot_cnt > 1:
        layer_split_name = name.split('layer')
        s_name = layer_split_name[1].rsplit(".", -1)[1]

        if type(s_name) is list:
            s_name[1] = str(int(s_name[1]) - 1)
        else:
            s_name = str(int(s_name) - 1)
        print(name)
        print(layer_split_name)
        s_name = layer_split_name + ''.join(s_name)
    else:
        s_name = re.split('(\d+)', name)
        s_name[1] = str(int(s_name[1]) + 1)
        s_name = "".join(s_name)
    return s_name


class SimParamLoss(nn.Module):

    def __init__(self, num_layers, reg=1e-0, constraint='exp'):
        super(SimParamLoss, self).__init__()

        layers = torch.FloatTensor(list(range(num_layers)))
        self.reg_term = reg
        if constraint:
            self.gamma = torch.exp(-layers)
        else:
            self.gamma = layers

        self.w_names = ['dense', 'fcnn', 'core', 'conv']

    def _forward_adjacent(self, model):
        """only computes similarity between adjacent layers which may be quicker"""
        loss, cnt = 0., 0.
        params = list(model.named_parameters())

        # for name, _ in params: print(name)
        for i, (name, param) in enumerate(params):
            for w_name in self.w_names:
                if w_name in name and 'bias' not in name and '1' in name:
                    s_name = get_s_name(name)
                    if s_name in model.state_dict():
                        s_param = model.state_dict()[s_name]

                        if param.size(1) * 2 == s_param.size(1):
                            param = torch.cat([param, param], 1)

                        sim = F.cosine_similarity(param, s_param)
                        sim = sim.mean([1, 2]).mean()
                        loss += sim
                        cnt += 1.

        # we want adjacent layer to be as similar as possible
        return (loss / cnt) * self.reg_term if cnt != 0 else 0

    def forward(self, model):
        loss, cnt = 0., 0.
        for i, f_name, f_param in enumerate(model.named_parameters()):
            for j, (s_name, s_param) in enumerate(list(model.named_parameters())[i:len(model.parameters())]):
                for w_name in self.w_names:
                    if w_name in f_name and w_name in s_name:
                        loss += self.gamma[abs(j - i)] * F.cosine_similarity(f_param, s_param)
                        cnt += 1.
        return (loss / cnt) * self.reg_term if cnt != 0 else 0


if __name__ ==  "__main__":

    from models.networks.convolutional.wide_resnet import *
    from configs.pruning import get_prune_args

    args = get_prune_args()
    print(args.growth)
    print(args.depth)
    print(args.transition_rate)
    args.growth = 10
    args.transition_rate = 1
    model = WideResNet(args.depth, args.width, mask=args.mask)

    criterion = SimParamLoss(args.depth, )
    loss = criterion._forward_adjacent(model)

    print(loss)







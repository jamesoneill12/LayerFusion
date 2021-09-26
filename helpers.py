import torch

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_prune_n_params(model):
    zeros = 0
    all = 0.0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data[0]
            all += param.numel()
    return zeros/all


def get_n_layers(model):
    pp=0
    for p in list(model.parameters()):
        pp += 1
    return pp


def pop(self):
    '''Removes a layer instance on top of the layer stack.'''
    if not self.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False
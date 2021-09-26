from torch import nn
import torch


# old
def get_discounts_list(num_walks, gpu=True):
    z = torch.exp(-torch.FloatTensor(num_walks))
    vals = torch.exp(z) / torch.sum(torch.exp(z), dim=0)
    shape = torch.Size((len(num_walks)))
    indices = torch.LongTensor([list(range(len(num_walks)))])
    out = torch.sparse.FloatTensor(indices, vals)
    if gpu:
        out = out.cuda()
    return out


def softmax(z):
    z = torch.exp(z) / torch.sum(torch.exp(z), dim=0)
    return z


def normer(z):
    return (z - torch.min(z)) / (torch.max(z) - torch.min(z))


def get_discounts(num_walks, gpu=True, norm='softmax', sparse=False):
    if norm == 'softmax':
        z = torch.exp(- torch.FloatTensor(num_walks))
        vals = softmax(z)
    else:
        vals = normer(torch.FloatTensor(list(reversed(num_walks))))
    if sparse:
        shape = torch.Size((1, len(num_walks)))
        indices = torch.LongTensor([list(range(len(num_walks)))])
        # need the list -i because sparse tensor does not have neg implemented
        # vals = torch.Tensor([-i for i in num_walks])
        vals = torch.sparse.FloatTensor(indices, vals)
    if gpu:
        vals = vals.cuda()

    return (vals)


# use this one instead when looping through
def get_discount_list(num_walks):
    x = torch.exp(torch.FloatTensor([-i for i in range(num_walks)]))
    norm_discs = torch.exp(x) / torch.sum(torch.exp(x), dim=0)
    return (norm_discs)


# splits tensor along row based on walks
def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


class RandomWalkAE(nn.Module):
    """
    Description:
        This function (forward discount) takes the neighbour nodes and looks to predict the
        exponential average of the neighbours given the current node in an autoencoder
        which can be referred to as Continuous Walk of Words (CWOW)

    Arguments:
        walk_nodes: a list of tensors that contain random walk from x_node
        x_node: current node used as input to predict walk_nodes

    Notes: be careful here, might need to use contiguous()
    """

    def __init__(self, x_dim, emb_dim, exp_decay=False, bias=True, train_scheme='individual'):
        super(RandomWalkAE, self).__init__()

        self.exp_decay = exp_decay
        self.train_scheme = train_scheme
        self.fc1 = nn.Linear(x_dim, emb_dim, bias=bias)
        self.fc2 = nn.Linear(emb_dim, x_dim, bias=bias)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    # changed the output to sparse using to_sparse

    def forward_once(self, x, k=None):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x  # (to_sparse(x))

    # elif self.train_scheme == 'loop' and neighbour_positions==None:
    def forward_discount_loop(self, x_node, x_walk_nodes, neighbour_positions=None):

        discs = get_discounts_list(x_walk_nodes.size(0))
        xs = [dis * self.forward_once(x) for x, dis in zip(x_walk_nodes, discs)]
        # self.decayed_forward_once
        x = torch.stack(xs, 0)
        x = x.contiguous().mean(0)
        # not sure what labels should be, maybe exp average or instead on the recon loss
        return x  # (to_sparse(x))

    # if self.train_scheme == 'individual' and neighbour_positions!=None:
    def forward_discount(self, x_node, x_walk_nodes, neighbour_positions=None):

        discs = get_discounts(neighbour_positions)
        x = discs * self.forward_once(x_walk_nodes, discs)
        # shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.FloatTensor(indices, values, shape)
        return x.to_dense()  # (to_sparse(x))

    def forward(self, x):
        if self.train_scheme == 'individual':
            x = self.forward_once(x)
        elif self.train_scheme == 'loop':
            x = [self.forward_once(xs) for xs in x]
            x = torch.stack(x, 0)
        return (x)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.fc1) + ' -> ' + str(self.fc2) + ')'

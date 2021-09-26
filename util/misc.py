"""
Miscellaneous functions that might be useful for pytorch
https://github.com/rowanz/swagaf/blob/master/pytorch_misc.py
"""

import h5py
import numpy as np
import torch
from torch.autograd import Variable
import dill as pkl
from itertools import tee
from torch import nn
import time
import cv2
import os
import shutil
import pickle as pkl
import hashlib
from IPython import embed


def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_ranking(predictions, labels, num_guesses=5):
    """
    Given a matrix of predictions and labels for the correct ones, get the number of guesses
    required to get the prediction right per example.
    :param predictions: [batch_size, range_size] predictions
    :param labels: [batch_size] array of labels
    :param num_guesses: Number of guesses to return
    :return:
    """
    assert labels.size(0) == predictions.size(0)
    assert labels.dim() == 1
    assert predictions.dim() == 2

    values, full_guesses = predictions.topk(predictions.size(1), dim=1)
    _, ranking = full_guesses.topk(full_guesses.size(1), dim=1, largest=False)
    gt_ranks = torch.gather(ranking.data, 1, labels.data[:, None]).squeeze()

    guesses = full_guesses[:, :num_guesses]
    return gt_ranks, guesses


def cache(f):
    """
    Caches a computation
    """

    def cache_wrapper(fn, *args, **kwargs):
        if os.path.exists(fn):
            with open(fn, 'rb') as file:
                data = pkl.load(file)
        else:
            print("file {} not found, so rebuilding".format(fn))
            data = f(*args, **kwargs)
            with open(fn, 'wb') as file:
                pkl.dump(data, file)
        return data

    return cache_wrapper


class Flattener(nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def to_variable(f):
    """
    Decorator that pushes all the outputs to a variable
    :param f:
    :return:
    """

    def variable_wrapper(*args, **kwargs):
        rez = f(*args, **kwargs)
        if isinstance(rez, tuple):
            return tuple([Variable(x) for x in rez])
        return Variable(rez)

    return variable_wrapper


def arange(base_tensor, n=None):
    new_size = base_tensor.size(0) if n is None else n
    new_vec = base_tensor.new(new_size).long()
    torch.arange(0, new_size, out=new_vec)
    return new_vec


def to_onehot(vec, num_classes, on_fill=1, off_fill=0):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = on_fill
    and off_fill otherwise.


    :param vec: 1d torch LongTensor (not a variable)
    :param num_classes: int
    :param on_fill: what to fill the things that are on
    :param off_fill: fill things that are off
    :return:
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(off_fill)
    arange_inds = vec.new(vec.size(0))
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result[arange_inds, vec] = on_fill
    return onehot_result


def save_net(fname, net):
    h5f = h5py.File(fname, mode='w')
    for k, v in list(net.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    h5f = h5py.File(fname, mode='r')
    for k, v in list(net.state_dict().items()):
        param = torch.from_numpy(np.asarray(h5f[k]))

        if v.size() != param.size():
            print("On k={} desired size is {} but supplied {}".format(k, v.size(), param.size()))
        else:
            v.copy_(param)


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


def batch_map(f, a, batch_size):
    """
    Maps f over the array a in chunks of batch_size.
    :param f: function to be applied. Must take in a block of
            (batch_size, dim_a) and map it to (batch_size, something).
    :param a: Array to be applied over of shape (num_rows, dim_a).
    :param batch_size: size of each array
    :return: Array of size (num_rows, something).
    """
    rez = []
    for s, e in batch_index_iterator(a.size(0), batch_size, skip_end=False):
        print("Calling on {}".format(a[s:e].size()))
        rez.append(f(a[s:e]))

    return torch.cat(rez)


def const_row(fill, l, volatile=False):
    input_tok = Variable(torch.LongTensor([fill] * l), volatile=volatile)
    if torch.cuda.is_available():
        input_tok = input_tok.cuda()
    return input_tok


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        strings.append("{:<60s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def nonintersecting_2d_inds(x):
    """
    Returns np.array([(a,b) for a in range(x) for b in range(x) if a != b]) efficiently
    :param x: Size
    :return: a x*(x-1) array that is [(0,1), (0,2)... (0, x-1), (1,0), (1,2), ..., (x-1, x-2)]
    """
    rs = 1 - np.diag(np.ones(x, dtype=np.int32))
    relations = np.column_stack(np.where(rs))
    return relations


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def gather_nd(x, index):
    """

    :param x: n dimensional tensor [x0, x1, x2, ... x{n-1}, dim]
    :param index: [num, n-1] where each row contains the indices we'll use
    :return: [num, dim]
    """
    nd = x.dim() - 1
    assert nd > 0
    assert index.dim() == 2
    assert index.size(1) == nd
    dim = x.size(-1)

    sel_inds = index[:, nd - 1].clone()
    mult_factor = x.size(nd - 1)
    for col in range(nd - 2, -1, -1):  # [n-2, n-3, ..., 1, 0]
        sel_inds += index[:, col] * mult_factor
        mult_factor *= x.size(col)

    grouped = x.view(-1, dim)[sel_inds]
    return grouped


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)
    # num_im = im_inds[-1] + 1
    # # print("Num im is {}".format(num_im))
    # for i in range(num_im):
    #     # print("On i={}".format(i))
    #     inds_i = (im_inds == i).nonzero()
    #     if inds_i.dim() == 0:
    #         continue
    #     inds_i = inds_i.squeeze(1)
    #     s = inds_i[0]
    #     e = inds_i[-1] + 1
    #     # print("On i={} we have s={} e={}".format(i, s, e))
    #     yield i, s, e


def diagonal_inds(tensor):
    """
    Returns the indices required to go along first 2 dims of tensor in diag fashion
    :param tensor: thing
    :return:
    """
    assert tensor.dim() >= 2
    assert tensor.size(0) == tensor.size(1)
    size = tensor.size(0)
    arange_inds = tensor.new(size).long()
    torch.arange(0, tensor.size(0), out=arange_inds)
    return (size + 1) * arange_inds


def enumerate_imsize(im_sizes):
    s = 0
    for i, (h, w, scale, num_anchors) in enumerate(im_sizes):
        na = int(num_anchors)
        e = s + na
        yield i, s, e, h, w, scale, na

        s = e


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def unravel_index(index, dims):
    unraveled = []
    index_cp = index.clone()
    for d in dims[::-1]:
        unraveled.append(index_cp % d)
        index_cp /= d
    return torch.cat([x[:, None] for x in unraveled[::-1]], 1)


def de_chunkize(tensor, chunks):
    s = 0
    for c in chunks:
        yield tensor[s:(s + c)]
        s = s + c


def random_choose(tensor, num):
    "randomly choose indices"
    num_choose = min(tensor.size(0), num)
    if num_choose == tensor.size(0):
        return tensor

    # Gotta do this in numpy because of https://github.com/pytorch/pytorch/issues/1868
    rand_idx = np.random.choice(tensor.size(0), size=num, replace=False)
    rand_idx = torch.LongTensor(rand_idx).cuda(tensor.get_device())
    chosen = tensor[rand_idx].contiguous()

    # rand_values = tensor.new(tensor.size(0)).float().normal_()
    # _, idx = torch.sort(rand_values)
    #
    # chosen = tensor[idx[:num]].contiguous()
    return chosen


def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """

    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer + 1)].copy())
        cum_add[:(length_pointer + 1)] += 1
        new_lens.append(length_pointer + 1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def right_shift_packed_sequence_inds(lengths):
    """
    :param lengths: e.g. [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    :return: perm indices for the old stuff (TxB) to shift it right 1 slot so as to accomodate
             BOS toks

             visual example: of lengths = [4,3,1,1]
    before:

        a (0)  b (4)  c (7) d (8)
        a (1)  b (5)
        a (2)  b (6)
        a (3)

    after:

        bos a (0)  b (4)  c (7)
        bos a (1)
        bos a (2)
        bos
    """
    cur_ind = 0
    inds = []
    for (l1, l2) in zip(lengths[:-1], lengths[1:]):
        for i in range(l2):
            inds.append(cur_ind + i)
        cur_ind += l1
    return inds


def clip_grad_norm(named_parameters, max_norm, clip=False, verbose=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = tuple(p.size())

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        print('-------------------------------', flush=True)

    return total_norm


def time_batch(gen, reset_every=100):
    """
    Gets timing info for a batch
    :param gen:
    :param reset_every: How often we'll reset
    :return:
    """
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i+1-start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i


def update_lr(optimizer, lr=1e-4):
    print("------ Learning rate -> {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def all_upper_triangular_pairs(gen):
    """ Iterates over all pairs from a generator where x < y """
    hist = []
    for g in gen:
        for h in hist:
            yield (h, g)
        hist.append(g)


def pad_last_dim(tensor, new_size):
    """
    Pads an n-dimensional tensor by 0's in the last dimension
    :param tensor: N-dimensional tensor
    :param new_size: how much the last dim should be
    :return: padded tensor
    """
    assert tensor.size(-1) < new_size
    to_add = tensor.new(tensor.size()[:-1] + (new_size-tensor.size(-1),)).fill_(0)
    return torch.cat((tensor, to_add), -1)


def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    print(x)
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


def handle_inputs(inputs, use_cuda):
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result


def kld_coef(i):
    import math
    return (math.tanh((i - 3500)/1000) + 1)/2


class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info
def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


def load_lmdb(lmdb_file, n_records=None):
    import lmdb
    lmdb_file = expand_user(lmdb_file)
    if os.path.exists(lmdb_file):
        data = []
        env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
        with env.begin() as txn:
            cursor = txn.cursor()
            begin_st = time.time()
            print("Loading lmdb file {} into memory".format(lmdb_file))
            for key, value in cursor:
                _, target, _ = key.decode('ascii').split(':')
                target = int(target)
                img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
                data.append((img, target))
                if n_records is not None and len(data) >= n_records:
                    break
        env.close()
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
        return data
    else:
        print("Not found lmdb file".format(lmdb_file))


def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()

def eval_model(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))



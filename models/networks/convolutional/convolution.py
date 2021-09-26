import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



def get_pretrained_cnn(model):
    if model == 'resnet':
        # not 18
        if '18' in model:
            return models.resnet18(pretrained=True)
        elif '34' in models:
            return models.resnet34(pretrained=True)
        elif '101' in model:
            return models.resnet101(pretrained=True)
        else:
            return models.resnet152(pretrained=True)
    elif model == "alexnet":
        return models.alexnet(pretrained=True)
    elif model == "squeezenet":
        return models.squeezenet1_0(pretrained=True)
    elif model == "vgg":
        if '11' in model:
            return models.vgg11(pretrained=True)
        elif '13' in model:
            return models.vgg13(pretrained=True)
        elif '16' in model:
            return models.vgg16(pretrained=True)
        else:
            return models.vgg19(pretrained=True)
    elif model == "densenet":
        if '121' in model:
            return models.densenet121(pretrained=True)
        elif '161' in models:
            return models.densenet161(pretrained=True)
        elif '169' in model:
            return models.densenet169(pretrained=True)
        else:
            return models.densenet201(pretrained=True)
    elif model == "polar":
        return models.inception_v3(pretrained=True)


class CNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, nlayers=2, dropout_rate=0.2,
                 class_num=None, kernel_num=None, kernel_sizes=None,s2s=False):
        super(CNN, self).__init__()

        V = vocab_size
        D = hidden_size
        C = hidden_size if class_num is None else class_num
        Ci = 1
        Co = 300 if kernel_num is None else kernel_num
        Ks = (2, 3, 4) if kernel_sizes is None else kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout_rate)
        self.seq2seq = s2s
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        if self.seq2seq:
            return x
        logit = self.fc1(x)  # (N, C)
        return logit


class DeCNN(nn.Module):
    """ DeConvolutional NN that takes encoded source rep and
    concatenates it with encoded rep for on target and makes
    each word prediction
    """
    def __init__(self, vocab_size, hidden_size, nlayers=2, dropout_rate=0.2,
                 class_num=None, kernel_num=None, kernel_sizes=None, s2s=False):
        super(DeCNN, self).__init__()

        V = vocab_size
        D = hidden_size
        C = hidden_size if class_num is None else class_num
        Ci = 1
        Co = 400 if kernel_num is None else kernel_num
        Ks = (2, 3, 4) if kernel_sizes is None else kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.Conv1d(Ci, Co, D)
        self.dropout = nn.Dropout(dropout_rate)
        self.seq2seq = s2s
        # *2 because output of x is concatenated with encoded rep from source.
        self.fc1 = nn.Linear(Co * 3, C)
        self.fc2 = nn.Linear(C, vocab_size)

    def forward(self, x, enc_h):
        x = self.embed(x)  # (N, W, D)
        h = []
        for i in range(x.size(1)):
            if i == 0:
                x_t = x[:, i]
                x_tm1 = x[:, i]
            x_in = torch.cat([x_tm1, x_t, enc_h], 1)
            h_t = self.fc1(x_in)
            h_t = self.dropout(h_t)  # (N, len(Ks)*Co)
            h_t = self.fc2(h_t)
            h_t = self.dropout(h_t)  # (N, len(Ks)*Co)
            print(h_t.size())
            h.append(h_t.unsqueeze(0))
        out = torch.cat(h, 0)
        return out


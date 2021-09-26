"""https://github.com/aykutfirat/pyTorchTree"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RecursiveNN(nn.Module):
    def __init__(self, vocabSize, embedSize=100, numClasses=5):
        super(RecursiveNN, self).__init__()
        self.embedding = nn.Embedding(int(vocabSize), embedSize)
        self.W = nn.Linear(2*embedSize, embedSize, bias=True)
        self.projection = nn.Linear(embedSize, numClasses, bias=True)
        self.activation = F.relu
        self.nodeProbList = []
        self.labelList = []

    def traverse(self, node):
        if node.isLeaf(): currentNode = self.activation(self.embedding(Variable(torch.LongTensor([node.getLeafWord()])).cuda()))
        else: currentNode = self.activation(self.W(torch.cat((self.traverse(node.left()),self.traverse(node.right())),1)))
        self.nodeProbList.append(self.projection(currentNode))
        self.labelList.append(torch.LongTensor([node.label()]))
        return currentNode

    def forward(self, x):
        self.nodeProbList = []
        self.labelList = []
        self.traverse(x)
        self.labelList = Variable(torch.cat(self.labelList)).cuda()
        return torch.cat(self.nodeProbList)

    def getLoss(self, tree):
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions,loss

    def evaluate(self, trees):
        n = nAll = correctRoot = correctAll = 0.0
        for j, tree in enumerate(trees):
            predictions, loss = self.getLoss(tree)
            correct = (predictions.data == self.labelList.data)
            correctAll += correct.sum()
            nAll += correct.squeeze().size()[0]
            correctRoot += correct.squeeze()[-1]
            n += 1
        return correctRoot / n, correctAll / nAll


if __name__ == "__main__":

    vocabSize = 10000
    embedSize = 100
    num_classes = 10
    sent_len = 20
    rec_nn = RecursiveNN(vocabSize, embedSize=100, numClasses=num_classes)
    x = torch.randn((sent_len, embedSize))
    out = rec_nn(x)
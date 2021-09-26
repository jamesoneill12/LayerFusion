from torch import nn

class StackedGRUAutoencoder(nn.Module):

    def __init__(self, x_dim, emb_dim, exp_decay=False):
        super(StackedGRUAutoencoder, self).__init__()

        self.fc1 = nn.GRU(x_dim, 1000)
        self.fc2 = nn.GRU(1000, emb_dim)
        self.fc3 = nn.GRU(emb_dim, 1000)
        self.fc4 = nn.GRU(1000, x_dim)

        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyRelu(self.fc1(x))
        x = self.dropout(x)
        x = self.leakyRelu(self.fc2(x))
        x = self.dropout(x)
        x = self.leakyRelu(self.fc3(x))
        x = self.dropout(x)
        x = self.leakyRelu(self.fc4(x))
        return x
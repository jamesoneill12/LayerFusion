from torch import nn

class StackedAutoencoder(nn.Module):
    def __init__(self, x_dim, emb_dim, h_dim=1000 exp_decay=False):
        super(StackedAutoencoder, self).__init__()
				
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, emb_dim)
        self.fc3 = nn.Linear(emb_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.Relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x


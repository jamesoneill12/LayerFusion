from torch import nn
import torch


class EncoderMaxEnt(nn.Module):
    def __init__(self, vocab_size, hidden_size, nlayers=2):
        super(EncoderMaxEnt, self).__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size, num_layers=nlayers)

    def forward(self, input, hidden):
        input = self.embedding(input)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, bsz):
        return torch.zeros(self.nlayers, bsz, self.hidden_size, device='cuda')


if __name__ == "__main__":
    vocab_size = 10000
    hidden_size = 400
    batch_size = 20
    sent_len = 40
    nlayers = 2
    dropout_rate = 0.2
    rnn = True
    encode = True
    x = torch.randint(0, vocab_size, (batch_size, sent_len)).type(torch.LongTensor)

    rnnhway_net = EncoderMaxEnt(vocab_size=vocab_size, hidden_size=hidden_size,
                           dropout_rate=dropout_rate,  encoder=encode)
    y = rnnhway_net(x)



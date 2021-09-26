import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from models.networks.recurrent import get_rnn
from models.networks.convolutional.convolution import get_pretrained_cnn
from models.networks.transformer.getter import get_transformer


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, pretrained='resnet'):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = get_pretrained_cnn(pretrained)
        #mods.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.transformer = get_transformer(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers,
                 rnn_type='lstm', max_seq_length=20,
                 drop_rate=0.15, dropout_method='standard'):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn_type = rnn_type
        self.lstm = get_rnn(rnn_type, embed_size, hidden_size, num_layers, vocab=vocab_size,
                            dropout=drop_rate, drop_method=dropout_method, cuda=True)
        if 'transformer' in rnn_type: hidden_size = embed_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        print("vocab size {}".format(vocab_size))
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # print(embeddings.size())
        if 'transformer' in self.rnn_type:
            # inputs, encoder_output
            hiddens = self.lstm(embeddings, features)
            # print("hidden size {}".format(hiddens.size()))
            # (hidden_size, vocab_size) = (256, 10000)
            # not sure if I should be taking the first (0) here in the case of the transformer
            outputs = self.linear(hiddens[:, 0, :])
        elif self.rnn_type == 'qrnn' or self.rnn_type == 'highway':
            # not using pack_padded_sequence in qrnn input
            hiddens, _ = self.lstm(embeddings)
            outputs = self.linear(hiddens)
        else:
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm(packed)
            outputs = self.linear(hiddens[0])
        # print("linear size {}".format(outputs.size()))
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

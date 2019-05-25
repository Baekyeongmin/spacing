import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from vocab import Vocabulary
import json


class CharConv(nn.Module):

    def __init__(self,
                 in_feature,
                 filters,
                 features):
        super(CharConv, self).__init__()

        self.filters = filters
        self.features = features

        assert len(filters) == len(features)

        self.num_conv = len(filters)

        for idx, (filter, out_feature) in enumerate(zip(filters, features)):
            name = f"conv1d_{idx}"

            setattr(self, name, nn.Conv1d(in_channels=in_feature,
                                          out_channels=out_feature,
                                          kernel_size=filter,
                                          stride=1))

    def forward(self, x):

        result = []

        for idx in range(self.num_conv):
            name = f"conv1d_{idx}"

            padding = self.filters[idx] - 1
            padding_left = int(padding / 2)
            padding_right = padding - padding_left
            conv_input = F.pad(x, (padding_left, padding_right))

            result.append(getattr(self, name)(conv_input))

        return torch.cat(result, dim=1)


class Spacing(nn.Module):
    def __init__(self, vocab):
        super(Spacing, self).__init__()

        with open('model_config.json') as model_config_file:
            model_config = json.load(model_config_file)

        self.embedding_dim = model_config['embedding_dim']
        self.filters = model_config['char_filters']
        self.features = model_config['char_features']
        self.linear1_dim = model_config['linear1_dim']
        self.linear2_dim = model_config['linear2_dim']
        self.gru_hidden = model_config['gru_hidden']
        self.gru_bidirectional = model_config['gru_bidirectional']

        self.vocab = vocab

        num_features = sum(self.features)

        if self.gru_bidirectional:
            linear_3_in_dim = self.gru_hidden * 2
        else:
            linear_3_in_dim = self.gru_hidden

        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=self.embedding_dim)

        self.char_conv = CharConv(in_feature=self.embedding_dim,
                                  filters=self.filters,
                                  features=self.features)
        self.batch_norm = nn.BatchNorm1d(num_features=num_features)
        self.linear1 = nn.Linear(in_features=num_features,
                                 out_features=self.linear1_dim)
        self.linear2 = nn.Linear(in_features=self.linear1_dim,
                                 out_features=self.linear2_dim)
        self.gru = nn.GRU(input_size=self.linear2_dim,
                          hidden_size=self.gru_hidden,
                          num_layers=1,
                          bidirectional=self.gru_bidirectional)
        self.linear3 = nn.Linear(in_features=linear_3_in_dim,
                                 out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sents):
        if type(sents[0]) == list:
            lengths = [len(s) for s in sents]
        else:
            lengths = [len(sents)]
        x = self.vocab.to_input_tensor(sents, self.device())
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x = self.char_conv(x)
        x = self.batch_norm(x)
        x = torch.transpose(x, 1, 2)
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.transpose(x, 0, 1)
        x = pack_padded_sequence(x, lengths)
        x, _ = self.gru(x)
        x, length = pad_packed_sequence(x)
        x = torch.transpose(x, 1, 0)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return torch.squeeze(x), length

    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.linear1.weight.device


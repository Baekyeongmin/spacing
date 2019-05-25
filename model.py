import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from vocab import Vocabulary


class CharConv(nn.Module):

    def __init__(self,
                 in_feature,
                 filters=[1, 2, 3, 4, 5],
                 features=[128, 256, 128, 64, 32]):
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
    def __init__(self,
                 vocab,
                 num_vocab=11174,
                 embedding_dim=100,
                 filters=[1, 2, 3, 4, 5],
                 features=[128, 256, 128, 64, 32],
                 linear1_dim=300,
                 linear2_dim=150,
                 gru_hidden=50,
                 gru_bidirectional=True):
        super(Spacing, self).__init__()

        self.vocab = vocab

        num_features = sum(features)

        if gru_bidirectional:
            linear_3_in_dim = gru_hidden * 2
        else:
            linear_3_in_dim = gru_hidden

        #embedding : 표현가능한 음절 (11172) + padding (1) + unknown (1) = 11174
        self.embedding = nn.Embedding(num_embeddings=num_vocab,
                                      embedding_dim=embedding_dim)

        self.char_conv = CharConv(in_feature=embedding_dim,
                                  filters=filters,
                                  features=features)
        self.batch_norm = nn.BatchNorm1d(num_features=num_features)
        self.linear1 = nn.Linear(in_features=num_features,
                                 out_features=linear1_dim)
        self.linear2 = nn.Linear(in_features=linear1_dim,
                                 out_features=linear2_dim)
        self.gru = nn.GRU(input_size=linear2_dim,
                          hidden_size=gru_hidden,
                          num_layers=1,
                          bidirectional=gru_bidirectional)
        self.linear3 = nn.Linear(in_features=linear_3_in_dim,
                                 out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sents):
        if type(sents[0]) == list:
            lengths = [len(s) for s in sents]
        else:
            lengths = [len(sents)]
        #x shape: [b_s, max_sequnece_length]
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


if __name__ == "__main__":
    print('-' * 10 + 'Character Convolution Test' + '-' * 10)
    # CharConv test
    char_conv = CharConv(100)
    # test_input [bs, seq_len, embedding_dim]
    input_tensor = torch.randn(30, 100, 20)
    print(char_conv.forward(input_tensor).size())

    print('-' * 10 + 'Spacing Model Test' + '-' * 10)
    vocab = Vocabulary('./test_vocab.txt')
    spacing_model = Spacing(vocab=vocab)

    input_sent = ['안 녕 하 세 요 백 영 민 입 니 다 .'.split(), '네 안 녕 하 세 요'.split()]
    print(spacing_model.forward(input_sent)[0].size())

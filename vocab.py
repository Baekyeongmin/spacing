import os
from utils import pad_sents
import torch


class Vocabulary(object):

    def __init__(self, vocab_path):
        '''
        :param vocab_path: (string) vocab file path, vocab have to contain one word per one line
        '''
        assert os.path.isfile(vocab_path), 'No vocab file!'

        self.word2id = dict()
        self.word2id['[PAD]'] = 0
        self.word2id['[UNK]'] = 1

        vocab_idx = 2
        with open(vocab_path, 'r') as f:
            words = [line.rstrip('\n') for line in f]
            for word in words:
                self.word2id[word] = vocab_idx
                vocab_idx += 1

        self.unk_id = self.word2id['[UNK]']
        self.pad_id = self.word2id['[PAD]']

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents, device='cpu'):
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self.pad_id)
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return sents_var



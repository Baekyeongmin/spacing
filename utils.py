import os
import math
import numpy as np


def pad_sents(sents, pad_token):

    sents_padded = []
    if type(sents[0]) == list:
        max_sentence_length = max([len(sent) for sent in sents])
    else:
        max_sentence_length = len(sents)
        sents = [sents]

    for sent in sents:
        pad_len = max(0, max_sentence_length - len(sent))
        sents_padded.append(sent + pad_len * [pad_token])
    return sents_padded


def load_corpus(file_path, make_vocab=False):
    input_data = []
    input_label = []

    assert os.path.isfile(file_path), 'No input file!'

    with open(file_path, 'r') as f:
        sents = [line.rstrip('\n') for line in f]
        for sent in sents:
            words = sent.split()

            data = []
            label = []

            for word in words:
                chars = [char for char in word]
                data.extend(chars)
                label.extend([0] * (len(chars) - 1) + [1])

            input_data.append(data)
            input_label.append(label)

    vocab = None
    if make_vocab:
        vocab = list(set([x for sub_list in input_data for x in sub_list]))

    return input_data, input_label, vocab


def batch_iter(input_data, input_label, batch_size, shuffle=False):

    assert len(input_data) == len(input_label)

    batch_num = math.ceil(len(input_data) / batch_size)
    index_array = list(range(len(input_data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]

        batch_data = [input_data[idx] for idx in indices]
        batch_label = [input_label[idx] for idx in indices]

        batch_data = sorted(batch_data, key=lambda e: len(e), reverse=True)
        batch_label = sorted(batch_label, key=lambda e: len(e), reverse=True)

        yield batch_data, batch_label


if __name__ == "__main__":
    input_data, input_label, vocab = load_corpus('./test_corpus.txt')
    for data, label in batch_iter(input_data, input_label, 2, True):
        print(data)
        print(label)


from utils import load_corpus, batch_iter, pad_sents
from vocab import Vocabulary
from model import Spacing
from loss import BCELossWithLength
import torch
import torch.optim as optim
import numpy as np
import os


def train(config):
    train_data_path = './train_corpus.txt'
    test_data_path = './test_corpus.txt'
    vocab_path = './vocab.txt'

    train_input_data, train_input_label, vocab = load_corpus(file_path=train_data_path, make_vocab=True)
    test_input_data, test_input_label, _ = load_corpus(file_path=test_data_path, make_vocab=False)

    #write_vocab
    if vocab:
        with open(vocab_path, 'w') as fw:
            for word in vocab:
                fw.write(word + '\n')

    trainer = Trainer(vocab=vocab_path,
                      train_data=train_input_data,
                      train_label=train_input_label,
                      val_data=test_input_data,
                      val_label=test_input_label,
                      config=None)

    trainer.train(total_epoch=10, validation_epoch=1)


class Trainer(object):

    def __init__(self, vocab_path, train_data=None, train_label=None, val_data=None, val_label=None, config=None):
        #training_config
        self.batch_size = 32
        self.learnig_rate = 0.001

        #model_config
        self.vocab = Vocabulary(vocab_path)

        #data
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label

        #save_and_load
        self.model_save_path = './weights/'

        #training
        self.model = Spacing(self.vocab, len(vocab))
        self.loss = BCELossWithLength()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learnig_rate)

    def accuracy(self, output, labels, length):
        total = 0
        correct = 0

        output = torch.round(output)
        correct_metrix = (output == labels)

        for idx, l in enumerate(length):
            correct += torch.sum(correct_metrix[idx, :l]).item()
            total += l.item()

        return correct, total

    def train_one_epoch(self, epoch, validataion=False):

        epoch_loss = []

        self.model.train()

        for batch_data, batch_label in batch_iter(self.train_data,
                                                  self.train_label,
                                                  self.batch_size,
                                                  shuffle=True):

            batch_label = pad_sents(batch_label, 0)
            batch_label = torch.tensor(batch_label, dtype=torch.float)

            self.optimizer.zero_grad()
            output, length = self.model.forward(batch_data)
            loss = self.loss(output, batch_label, length)
            loss.backward()
            self.optimizer.step()

            epoch_loss.append(loss.item())

        print(f'[Epoch]: {epoch} \t [Loss]: {np.mean(epoch_loss)}')

        val_accuracy = None
        if validataion:
            val_accuracy = self.validation()

        return np.mean(epoch_loss), val_accuracy

    def validation(self):

        self.model.eval()

        correct_sum = 0
        total_sum = 0

        for batch_data, batch_label in batch_iter(self.val_data,
                                                  self.val_label,
                                                  self.batch_size,
                                                  shuffle=False):
            batch_label = pad_sents(batch_label, 0)
            batch_label = torch.tensor(batch_label, dtype=torch.float)
            output, length = self.model.forward(batch_data)
            loss = self.loss(output, batch_label, length)

            correct, total = self.accuracy(output, batch_label, length)
            correct_sum += correct
            total_sum += total

        accuracy = correct_sum / total_sum
        print(f'[Validation]: accuracy {accuracy} \t loss {loss}')
        return accuracy

    def train(self, total_epoch, validation_epoch=10):

        losses = []

        max_val_accuracy = 0

        for epoch in range(total_epoch):
            validation = False

            if (epoch + 1) % validation_epoch == 0:
                validation = True

            epoch_loss, val_accuracy = self.train_one_epoch(epoch, validation)
            losses.append(epoch_loss)
            if val_accuracy:
                if max_val_accuracy < val_accuracy:
                    max_val_accuracy = val_accuracy
                    self.save(epoch, epoch_loss)

    def save(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        save_path = os.path.join(self.model_save_path,
                                 'model_checkpoint_epoch_{}.pth'.format(epoch))
        torch.save(checkpoint, save_path)

    def load(self, epoch):
        load_path = os.path.join(self.model_save_path,
                                 'model_checkpoint_epoch_{}.pth'.format(epoch))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    inference('전 이미 누워서 항공권 찾아보고 있었음', 8)

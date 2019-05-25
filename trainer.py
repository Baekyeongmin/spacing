import torch
import torch.optim as optim
import numpy as np
import os

from utils import pad_sents, batch_iter, load_corpus
from loss import BCELossWithLength


class Trainer(object):

    def __init__(self,
                 train_data=None,
                 train_label=None,
                 val_data=None,
                 val_label=None,
                 model = None,
                 config=None):
        #training_config
        self.batch_size = config['batch_size']
        self.learnig_rate = config['learning_rate']

        #save_and_load
        self.model_save_path = './weights/'

        #data
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label

        #training
        self.model = model
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

        with open(os.path.join(self.model_save_path, 'checkpoint.txt'), 'w') as fw:
            fw.write(f'best checkpoint:{epoch}')

        print(f'Best Model is saved at {save_path}')


    def load(self, epoch):
        load_path = os.path.join(self.model_save_path,
                                 'model_checkpoint_epoch_{}.pth'.format(epoch))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

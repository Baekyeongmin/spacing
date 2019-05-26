import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import math

from utils import pad_sents, batch_iter, load_corpus


class Trainer(object):

    def __init__(self,
                 train_data=None,
                 train_label=None,
                 val_data=None,
                 val_label=None,
                 vocab=None,
                 model=None,
                 loss=None,
                 config=None):
        #training_config
        self.batch_size = config['batch_size']
        self.learnig_rate = config['learning_rate']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #save_and_load
        self.model_save_path = './weights/'

        #data
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label

        #training
        self.vocab = vocab
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learnig_rate)

    def make_input_tensor(self, batch_data, batch_label):

        if type(batch_data[0]) == list:
            lengths = [len(s) for s in batch_data]
        else:
            lengths = [len(batch_data)]

        batch_data = self.vocab.words2indices(batch_data)
        batch_data = pad_sents(batch_data, 0)
        batch_data = torch.tensor(batch_data, dtype=torch.long, device=self.device)

        if batch_label:
            batch_label = pad_sents(batch_label, 0)
            batch_label = torch.tensor(batch_label, dtype=torch.float, device=self.device)

        return batch_data, batch_label, lengths

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

        total = math.ceil(len(self.train_data) / self.batch_size)

        for batch_data, batch_label in tqdm(batch_iter(self.train_data,
                                                  self.train_label,
                                                  self.batch_size,
                                                  shuffle=True),  total=total):

            batch_data, batch_label, lengths = self.make_input_tensor(batch_data, batch_label)

            self.optimizer.zero_grad()
            output, length = self.model.forward(batch_data, lengths)
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

            batch_data, batch_label, lengths = self.make_input_tensor(batch_data, batch_label)

            output, length = self.model.forward(batch_data, lengths)
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

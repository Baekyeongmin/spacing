import torch
import torch.optim as optim
import numpy as np
import os

from utils import pad_sents, batch_iter


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
        self.scheduler = config['scheduler']
        self.scheduler_step = config['scheduler_step']
        self.scheduler_decay = config['scheduler_decay']
        self.model_save_path = config['model_save_path']
        self.gradient_clamping = config['gradient_clamping']
        self.gradient_clamping_threshold = config['clamping_threshold']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        if self.scheduler:
            assert type(self.scheduler_step) is int
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                       step_size=self.scheduler_step,
                                                       gamma=0.5)
        else:
            self.scheduler = None

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

    def metric(self, output, labels, length):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        total = 0

        zero_tensor = torch.zeros(labels.size()).type(torch.long).to(self.device)
        one_tensor = torch.ones(labels.size()).type(torch.long).to(self.device)

        output = torch.round(output).type(torch.long)
        labels = labels.type(torch.long)

        tp_metrix = torch.where((output == 1) & (labels == 1),
                                one_tensor,
                                zero_tensor)
        fp_metrix = torch.where((output == 1) & (labels == 0),
                                one_tensor,
                                zero_tensor)
        fn_metrix = torch.where((output == 0) & (labels == 1),
                                one_tensor,
                                zero_tensor)
        tn_metrix = torch.where((output == 0) & (labels == 0),
                                one_tensor,
                                zero_tensor)

        for idx, l in enumerate(length):
            tp += torch.sum(tp_metrix[idx, :l]).item()
            fp += torch.sum(fp_metrix[idx, :l]).item()
            fn += torch.sum(fn_metrix[idx, :l]).item()
            tn += torch.sum(tn_metrix[idx, :l]).item()
            total += l.item()
        assert total == (tp + fp + tn + fn)

        return tp, fp, tn, fn

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_one_epoch(self, epoch, validataion=False):

        epoch_loss = []

        self.model.train()

        logging_step = 100

        for idx, (batch_data, batch_label) in enumerate(batch_iter(self.train_data,
                                                                   self.train_label,
                                                                   self.batch_size,
                                                                   shuffle=True)):

            batch_data, batch_label, lengths = self.make_input_tensor(batch_data, batch_label)

            self.optimizer.zero_grad()
            output, length = self.model.forward(batch_data, lengths)
            loss = self.loss(output, batch_label, length)
            loss.backward()

            if self.gradient_clamping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.gradient_clamping_threshold)

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss.append(loss.item())

            if (idx + 1) % logging_step == 0:
                cur_learning_rate = self.get_lr()
                print(f'[Epoch]: {epoch} [Step]: {idx + 1}'
                      f' \t [Loss]: {np.mean(epoch_loss)} '
                      f'\t [learning_rate] {cur_learning_rate}')


        val_accuracy = None
        if validataion:
            val_accuracy = self.validation()

        return np.mean(epoch_loss), val_accuracy

    def validation(self):
        self.model.eval()

        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0

        for batch_data, batch_label in batch_iter(self.val_data,
                                                  self.val_label,
                                                  self.batch_size,
                                                  shuffle=False):

            batch_data, batch_label, lengths = self.make_input_tensor(batch_data, batch_label)

            output, length = self.model.forward(batch_data, lengths)
            loss = self.loss(output, batch_label, length)

            tp, fp, tn, fn = self.metric(output, batch_label, length)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f'[Validation]: accuracy {accuracy} \t precision {precision} \t recall {recall} \t f1 {f1} \t loss {loss}')
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

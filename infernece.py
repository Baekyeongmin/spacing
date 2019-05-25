from train import Trainer
from vocab import Vocabulary
import torch

def inference(text, epoch):
    vocab_path = './vocab.txt'
    vocab = Vocabulary(vocab_path)

    words = text.split()

    data = []

    for word in words:
        chars = [char for char in word]
        data.extend(chars)

    trainer = Trainer(vocab=vocab)
    trainer.load(epoch)

    output, _ = trainer.model.forward(data)
    output = torch.round(output)

    result = ''
    for idx, o in enumerate(output):
        if o == 1:
            result += (data[idx] + ' ')
        else:
            result += data[idx]

    print(result)
from train import Trainer
from vocab import Vocabulary
from model import Spacing
import torch
import json
import os


def inference():
    with open('train_config.json') as train_config_file:
        train_config = json.load(train_config_file)

    vocab_path = train_config['vocab_path']
    model_save_path = train_config['model_save_path']

    epoch = None
    with open(os.path.join(model_save_path, 'checkpoint.txt')) as f:
        epoch = f.readlines()[0].split(':')[1]
        print(f'Weight is loaded from best checkpoint epoch {epoch}')

    vocab = Vocabulary(vocab_path)

    model = Spacing(vocab=vocab)
    trainer = Trainer(model=model, config=train_config)
    trainer.load(epoch)

    while True:
        text = input('Enter input text : ')
        words = text.split()
        data = []

        for word in words:
            chars = [char for char in word]
            data.extend(chars)

        output, _ = trainer.model.forward(data)
        output = torch.round(output)

        result = ''
        for idx, o in enumerate(output):
            if o == 1:
                result += (data[idx] + ' ')
            else:
                result += data[idx]

        print(result)


if __name__ == "__main__":
    inference()

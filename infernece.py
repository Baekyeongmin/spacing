from train import Trainer
from vocab import Vocabulary
from model import Spacing
import torch
import json


def inference(text, epoch):
    with open('train_config.json') as train_config_file:
        train_config = json.load(train_config_file)

    vocab_path = train_config['vocab_path']
    vocab = Vocabulary(vocab_path)

    words = text.split()

    data = []

    for word in words:
        chars = [char for char in word]
        data.extend(chars)

    model = Spacing(vocab=vocab)
    trainer = Trainer(model=model, config=train_config)
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


if __name__ == "__main__":
    text = input('Enter input text : ')
    inference(text, 8)

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

    model = Spacing(vocab_len=len(vocab)).eval()

    trainer = Trainer(model=model,
                      vocab=vocab,
                      config=train_config)
    trainer.load(epoch)

    while True:
        text = input('Enter input text : ')
        words = text.split()
        data = []

        for word in words:
            chars = [char for char in word]
            data.append(chars)
        sorted_data = sorted(data, key=lambda e: len(e), reverse=True)
        idx = sorted(range(len(data)), key=lambda e: len(data[e]), reverse=True)
        batch_data, batch_label, lengths = trainer.make_input_tensor(sorted_data, None)

        outputs, _ = trainer.model.forward(batch_data, lengths)
        outputs = torch.round(outputs)

        results = []
        for output, data in zip(outputs, sorted_data):
            result = ''
            for output_char, char in zip(output, data):
                if output_char == 1:
                    result += (char + ' ')
                else:
                    result += char
            results.append(result)

        sorted_result = ''
        for i in range(len(idx)):
            sorted_result += results[idx.index(i)]

        print(sorted_result)


if __name__ == "__main__":
    inference()

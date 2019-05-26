from utils import load_corpus
from vocab import Vocabulary
from model import Spacing
from trainer import Trainer
import json


def train():
    with open('train_config.json') as train_config_file:
        train_config = json.load(train_config_file)
    train_data_path = train_config['train_data_path']
    test_data_path = train_config['test_data_path']
    vocab_path = train_config['vocab_path']

    train_input_data, train_input_label = load_corpus(file_path=train_data_path,
                                                      make_vocab=True,
                                                      vocab_path=vocab_path)
    val_input_data, val_input_label = load_corpus(file_path=test_data_path,
                                                  make_vocab=False)

    vocab = Vocabulary(vocab_path)

    model = Spacing(vocab_len=len(vocab))

    print(model)

    trainer = Trainer(model=model,
                      vocab=vocab,
                      train_data=train_input_data,
                      train_label=train_input_label,
                      val_data=val_input_data,
                      val_label=val_input_label,
                      config=train_config)
    trainer.train(total_epoch=10, validation_epoch=1)


if __name__ == "__main__":
    train()

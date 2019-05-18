from preprocess import load_train_val_test_subsets, load_embedding_weights, load_vocabulary, \
    load_word_mappings, load_sequences
from models import SentDetect
from data_generators import sent_detect_data_generator
import time


def train_sent_detect(r1, r2, padding_size, embedding_size, learning_rate, batch_size, num_epochs, steps_per_epoch):
    train_ids, val_ids, _ = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, r1, r2)
    model = SentDetect()
    model.build(padding_size, len(vocab), embedding_size, embedding_weights, 2)
    model.compile(learning_rate)
    model.summary()
    w_to_i, _ = load_word_mappings(vocab)
    train_sequences, train_labels = load_sequences(r1, r2, train_ids, w_to_i, padding_size)
    val_sequences, val_labels = load_sequences(r1, r2, val_ids, w_to_i, padding_size)
    train_data_generator = sent_detect_data_generator(train_sequences, train_labels, batch_size)
    val_data_generator = sent_detect_data_generator(val_sequences, val_labels, batch_size)
    print(f'Training SentDetect model with reviews in range {r1}-{r2} ...')
    start_time = time.time()
    model.train(num_epochs, steps_per_epoch, train_data_generator, val_data_generator)
    end_time = time.time()
    print(f'Training took {end_time - start_time} seconds')


if __name__ == '__main__':
    train_sent_detect(r1=50, r2=500, padding_size=15, embedding_size=50, learning_rate=0.001, batch_size=32,
                      num_epochs=5, steps_per_epoch=500)

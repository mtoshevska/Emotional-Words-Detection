from preprocess import load_train_val_test_subsets, load_embedding_weights, load_vocabulary, \
    load_word_mappings, load_sequences
from models import SentDetect, StarDetect
from data_generators import sent_detect_data_generator, star_detect_data_generator
import time
import numpy as np


def balance_dataset(sequences, labels, model_type):
    assert model_type in ['sentiment', 'star']
    star_1 = labels.count(1)
    star_2 = labels.count(2)
    star_3 = labels.count(3)
    star_4 = labels.count(4)
    star_5 = labels.count(5)
    new_sequences = list()
    new_labels = list()
    if model_type == 'sentiment':
        num = np.min([star_1 + star_2 + star_3, star_4 + star_5])
        num_p = 0
        num_n = 0
        for sequence, label in zip(sequences, labels):
            if label > 3 and num_p < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_p += 1
            elif label < 4 and num_n < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_n += 1
    else:
        num = np.min([star_1, star_2, star_3, star_4, star_5])
        num_1 = 0
        num_2 = 0
        num_3 = 0
        num_4 = 0
        num_5 = 0
        for sequence, label in zip(sequences, labels):
            if label == 1 and num_1 < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_1 += 1
            elif label == 2 and num_2 < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_2 += 1
            elif label == 3 and num_3 < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_3 += 1
            elif label == 4 and num_4 < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_4 += 1
            elif label == 5 and num_5 < num:
                new_sequences.append(sequence)
                new_labels.append(label)
                num_5 += 1
    return np.array(new_sequences), new_labels


def train_sent_detect(r1, r2, padding_size, embedding_size, embedding_source, learning_rate, batch_size, num_epochs):
    train_ids, val_ids, _ = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, embedding_source, r1, r2)
    model = SentDetect(r1, r2, embedding_source[0])
    model.build(padding_size, len(vocab), embedding_size, embedding_weights)
    model.compile(learning_rate)
    model.summary()
    w_to_i, _ = load_word_mappings(vocab, r1, r2)
    train_sequences, train_labels = load_sequences(r1, r2, train_ids, w_to_i, padding_size)
    train_sequences, train_labels = balance_dataset(train_sequences, train_labels, 'sentiment')
    val_sequences, val_labels = load_sequences(r1, r2, val_ids, w_to_i, padding_size)
    val_sequences, val_labels = balance_dataset(val_sequences, val_labels, 'sentiment')
    train_data_generator = sent_detect_data_generator(train_sequences, train_labels, batch_size)
    val_data_generator = sent_detect_data_generator(val_sequences, val_labels, batch_size)
    print(f'Training SentDetect model with reviews in range {r1}-{r2} with {embedding_source} embedding vectors ...')
    start_time = time.time()
    steps_per_epoch = len(train_sequences) // batch_size
    model.train(num_epochs, steps_per_epoch, train_data_generator, val_data_generator)
    end_time = time.time()
    print(f'Training took {end_time - start_time} seconds')


def train_star_detect(r1, r2, padding_size, embedding_size, embedding_source, learning_rate, batch_size, num_epochs):
    train_ids, val_ids, _ = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, embedding_source, r1, r2)
    model = StarDetect(r1, r2, embedding_source[0])
    model.build(padding_size, len(vocab), embedding_size, embedding_weights)
    model.compile(learning_rate)
    model.summary()
    w_to_i, _ = load_word_mappings(vocab, r1, r2)
    train_sequences, train_labels = load_sequences(r1, r2, train_ids, w_to_i, padding_size)
    train_sequences, train_labels = balance_dataset(train_sequences, train_labels, 'star')
    val_sequences, val_labels = load_sequences(r1, r2, val_ids, w_to_i, padding_size)
    val_sequences, val_labels = balance_dataset(val_sequences, val_labels, 'star')
    train_data_generator = star_detect_data_generator(train_sequences, train_labels, batch_size)
    val_data_generator = star_detect_data_generator(val_sequences, val_labels, batch_size)
    print(f'Training StarDetect model with reviews in range {r1}-{r2} with {embedding_source} embedding vectors ...')
    start_time = time.time()
    steps_per_epoch = len(train_sequences) // batch_size
    model.train(num_epochs, steps_per_epoch, train_data_generator, val_data_generator)
    end_time = time.time()
    print(f'Training took {end_time - start_time} seconds')


if __name__ == '__main__':
    train_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    train_star_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=100)
    train_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=50)
    train_star_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=100)
    train_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=25)
    train_star_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                      learning_rate=0.0001, batch_size=256, num_epochs=100)
    train_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=25)
    train_star_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                      learning_rate=0.0001, batch_size=256, num_epochs=100)

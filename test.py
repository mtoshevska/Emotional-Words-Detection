from preprocess import load_train_val_test_subsets, load_vocabulary, load_embedding_weights, \
    load_word_mappings, load_sequences
from models import SentDetect, StarDetect
from train import balance_dataset
from sklearn.metrics import accuracy_score  # , precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import _pickle as pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def evaluate(gt_labels, predicted_labels):
    print(f'Accuracy: {accuracy_score(gt_labels, predicted_labels)}')
    # print(f'Precision: {precision_score(gt_labels, predicted_labels)}')
    # print(f'Recall: {recall_score(gt_labels, predicted_labels)}')
    # print(f'F1 score: {f1_score(gt_labels, predicted_labels)}')
    print()


def generate_predictions(model, sequences, predictions_file):
    if not os.path.exists(predictions_file):
        predictions = list()
        for _, sequence in zip(tqdm(list(range(len(sequences)))), sequences):
            predictions.append(model.predict(sequence))
        with open(predictions_file, 'wb') as doc_w:
            pickle.dump(predictions, doc_w)
    else:
        with open(predictions_file, 'rb') as doc_r:
            predictions = pickle.load(doc_r)
    return predictions


def evaluate_sent_detect(r1, r2, padding_size, embedding_size, embedding_source, learning_rate, epoch):
    _, _, test_ids = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, embedding_source, r1, r2)
    model = SentDetect(r1, r2, embedding_source[0])
    model.build(padding_size, len(vocab), embedding_size, embedding_weights)
    model.compile(learning_rate)
    model.load_weights(f'SentDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}')
    # model.summary()
    w_to_i, _ = load_word_mappings(vocab, r1, r2)
    test_sequences, test_labels = load_sequences(r1, r2, test_ids, w_to_i, padding_size)
    test_sequences, test_labels, test_ids = balance_dataset(test_sequences, test_labels, 'sentiment', test_ids)
    print(f'Evaluating SentDetect model with reviews in range {r1}-{r2} with {embedding_source} embedding vectors ...')
    test_labels = [1 if t < 4 else 0 for t in test_labels]
    predictions_file = f'data/SentDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}_predictions.pkl'
    predicted_labels = generate_predictions(model, test_sequences, predictions_file)
    evaluate(test_labels, predicted_labels)


def evaluate_star_detect(r1, r2, padding_size, embedding_size, embedding_source, learning_rate, epoch):
    _, _, test_ids = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, embedding_source, r1, r2)
    model = StarDetect(r1, r2, embedding_source[0])
    model.build(padding_size, len(vocab), embedding_size, embedding_weights)
    model.compile(learning_rate)
    model.load_weights(f'StarDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}')
    # model.summary()
    w_to_i, _ = load_word_mappings(vocab, r1, r2)
    test_sequences, test_labels = load_sequences(r1, r2, test_ids, w_to_i, padding_size)
    test_sequences, test_labels, test_ids = balance_dataset(test_sequences, test_labels, 'star', test_ids)
    print(f'Evaluating StarDetect model with reviews in range {r1}-{r2} with {embedding_source} embedding vectors ...')
    test_labels = [int(t - 1) for t in test_labels]
    predictions_file = f'data/StarDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}_predictions.pkl'
    predicted_labels = generate_predictions(model, test_sequences, predictions_file)
    evaluate(test_labels, predicted_labels)


def map_word_with_weights(weights, sequences, seq_ids, id_to_word):
    word_weights = dict()
    for weight, sequence, seq_id in zip(weights, sequences, seq_ids):
        weight_reverse = np.flip(weight)
        sequence_reverse = np.flip(sequence)
        w = dict()
        for i in range(np.min([weight_reverse.shape[0], sequence_reverse.shape[0]])):
            word = id_to_word[sequence_reverse[i]] if sequence_reverse[i] != 0 else '<UNK>'
            w[word] = weight_reverse[i]
        word_weights[seq_id] = w
    return word_weights


def generate_sent_detect_att_weights(r1, r2, padding_size, embedding_size, embedding_source, learning_rate, epoch):
    _, _, test_ids = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, embedding_source, r1, r2)
    model = SentDetect(r1, r2, embedding_source[0])
    model.build(padding_size, len(vocab), embedding_size, embedding_weights, True)
    model.compile(learning_rate)
    model.load_weights(f'SentDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}')
    # model.summary()
    w_to_i, i_to_w = load_word_mappings(vocab, r1, r2)
    test_sequences, test_labels = load_sequences(r1, r2, test_ids, w_to_i, padding_size, False)
    test_sequences, test_labels, test_ids = balance_dataset(test_sequences, test_labels, 'sentiment', test_ids)
    print(f'Generating SentDetect model attention weights with reviews in range {r1}-{r2} with {embedding_source} '
          f'embedding vectors ...')
    predictions_file = f'data/SentDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}_att_predictions.pkl'
    predicted_weights = generate_predictions(model, pad_sequences(test_sequences, padding_size), predictions_file)
    word_weights = map_word_with_weights(predicted_weights, test_sequences, test_ids, i_to_w)
    with open(f'data/SentDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}_word_weights.pkl', 'wb') as doc_w:
        pickle.dump(word_weights, doc_w)


def generate_star_detect_att_weights(r1, r2, padding_size, embedding_size, embedding_source, learning_rate, epoch):
    _, _, test_ids = load_train_val_test_subsets(r1, r2)
    vocab = load_vocabulary(r1, r2)
    embedding_weights = load_embedding_weights(vocab, embedding_size, embedding_source, r1, r2)
    model = StarDetect(r1, r2, embedding_source[0])
    model.build(padding_size, len(vocab), embedding_size, embedding_weights, True)
    model.compile(learning_rate)
    model.load_weights(f'StarDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}')
    # model.summary()
    w_to_i, i_to_w = load_word_mappings(vocab, r1, r2)
    test_sequences, test_labels = load_sequences(r1, r2, test_ids, w_to_i, padding_size, False)
    test_sequences, test_labels, test_ids = balance_dataset(test_sequences, test_labels, 'star', test_ids)
    print(f'Generating StarDetect model attention weights with reviews in range {r1}-{r2} with {embedding_source} '
          f'embedding vectors ...')
    predictions_file = f'data/StarDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}_att_predictions.pkl'
    predicted_weights = generate_predictions(model, pad_sequences(test_sequences, padding_size), predictions_file)
    word_weights = map_word_with_weights(predicted_weights, test_sequences, test_ids, i_to_w)
    with open(f'data/StarDetect_{r1}_{r2}_{embedding_source[0]}-{epoch}_word_weights.pkl', 'wb') as doc_w:
        pickle.dump(word_weights, doc_w)


if __name__ == '__main__':
    evaluate_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    evaluate_star_detect(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    evaluate_sent_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    evaluate_star_detect(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    evaluate_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    evaluate_star_detect(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                         learning_rate=0.0001, epoch=50)
    evaluate_sent_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    evaluate_star_detect(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                         learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=50, r2=500, padding_size=45, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=300, embedding_source='wikipedia',
                                     learning_rate=0.0001, epoch=50)
    generate_sent_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)
    generate_star_detect_att_weights(r1=10, r2=500, padding_size=42, embedding_size=200, embedding_source='twitter',
                                     learning_rate=0.0001, epoch=50)

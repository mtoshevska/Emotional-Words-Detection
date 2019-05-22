import numpy as np


def sent_detect_data_generator(sequences, labels, batch_size, shuffle=False):
    """
    Generates data for SentDetect model.
    :param sequences: review sequences
    :type sequences: numpy.array
    :param labels: review label (stars)
    :type labels: numpy.array
    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: shuffle the data if true
    :type shuffle: bool
    """
    b = 0
    sequence_index = -1
    sequence_id = 0
    sequence_ids = np.array([i for i in range(len(sequences))])
    batch_review_sequences, batch_review_labels = None, None
    while True:
        try:
            sequence_index = (sequence_index + 1) % len(sequences)
            if shuffle and sequence_index == 0:
                np.random.shuffle(sequences)
            sequence_id = sequence_ids[sequence_index]
            sequence_input = sequences[sequence_id]
            label_output = [1, 0] if labels[sequence_id] > 3 else [0, 1]
            label_output = np.array(label_output)
            if b == 0:
                batch_review_sequences = np.zeros((batch_size,) + sequence_input.shape, dtype=sequence_input.dtype)
                batch_review_labels = np.zeros((batch_size,) + label_output.shape, dtype=label_output.dtype)
            batch_review_sequences[b] = sequence_input
            batch_review_labels[b] = label_output
            b += 1
            if b >= batch_size:
                yield batch_review_sequences, batch_review_labels
                b = 0
        except:
            raise Exception('An error occurred while processing sequence ' + str(sequence_id))


def star_detect_data_generator(sequences, labels, batch_size, shuffle=False):
    """
    Generates data for StarDetect model.
    :param sequences: review sequences
    :type sequences: numpy.array
    :param labels: review label (stars)
    :type labels: numpy.array
    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: shuffle the data if true
    :type shuffle: bool
    """
    b = 0
    sequence_index = -1
    sequence_id = 0
    sequence_ids = np.array([i for i in range(len(sequences))])
    batch_review_sequences, batch_review_labels = None, None
    while True:
        try:
            sequence_index = (sequence_index + 1) % len(sequences)
            if shuffle and sequence_index == 0:
                np.random.shuffle(sequences)
            sequence_id = sequence_ids[sequence_index]
            sequence_input = sequences[sequence_id]
            if labels[sequence_id] == 1:
                label_output = [1, 0, 0, 0, 0]
            elif labels[sequence_id] == 2:
                label_output = [0, 1, 0, 0, 0]
            elif labels[sequence_id] == 3:
                label_output = [0, 0, 1, 0, 0]
            elif labels[sequence_id] == 4:
                label_output = [0, 0, 0, 1, 0]
            else:
                label_output = [0, 0, 0, 0, 1]
            label_output = np.array(label_output)
            if b == 0:
                batch_review_sequences = np.zeros((batch_size,) + sequence_input.shape, dtype=sequence_input.dtype)
                batch_review_labels = np.zeros((batch_size,) + label_output.shape, dtype=label_output.dtype)
            batch_review_sequences[b] = sequence_input
            batch_review_labels[b] = label_output
            b += 1
            if b >= batch_size:
                yield batch_review_sequences, batch_review_labels
                b = 0
        except:
            raise Exception('An error occurred while processing sequence ' + str(sequence_id))

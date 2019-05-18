from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten, Activation, Permute, Multiply, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np


class SentDetect:
    def __init__(self):
        """
        Initializes class for SentDetect model.
        """
        self.name = 'SentDetect'
        self.model = None
        self.model_filepath = 'models/SentDetect-{epoch:02d}.h5'
        self.logs_filepath = 'logs/SentDetect.log'

    def build(self, padding_size, vocabulary_size, embedding_size, weights, output_shape):
        """
        Builds SentDetect model.
        :param padding_size: padding size
        :type padding_size: int
        :param vocabulary_size: vocabulary size
        :type vocabulary_size: int
        :param embedding_size: embedding size
        :type embedding_size: int
        :param weights: embedding weights
        :type weights: numpy.array
        :param output_shape: output shape of the model
        :type output_shape: int
        """
        input_layer = Input(shape=(padding_size,), name='SentDetect_input')
        word_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size,
                                   weights=[weights], name='SentDetect_embedding')(input_layer)
        rnn = Bidirectional(LSTM(1024, return_sequences=True), name='SentDetect_BiLSTM')(word_embedding)
        attention = Permute([2, 1], name='SentDetect_attention_permute1')(rnn)
        attention = Activation('tanh', name='SentDetect_attention_tanh')(attention)
        attention = Dense(15, activation='softmax', name='SentDetect_attention_dense')(attention)
        attention_weights = Permute((2, 1), name='SentDetect_attention_permute2')(attention)
        attention_weighted_sum = Multiply(name='SentDetect_attention_weights')([rnn, attention_weights])
        intermediate_layer = Flatten(name='SentDetect_flatten')(attention_weighted_sum)
        intermediate_layer = Dense(64, activation='tanh', name='SentDetect_dense1')(intermediate_layer)
        intermediate_layer = Dropout(0.5, name='SentDetect_dropout1')(intermediate_layer)
        intermediate_layer = Dense(64, activation='tanh', name='SentDetect_dense2')(intermediate_layer)
        intermediate_layer = Dropout(0.5, name='SentDetect_dropout2')(intermediate_layer)
        output_layer = Dense(output_shape, activation='softmax', name='SentDetect_dense3')(intermediate_layer)
        self.model = Model(input_layer, output_layer)

    def compile(self, learning_rate):
        """
        Compiles SentDetect model.
        :param learning_rate: learning rate
        :type learning_rate: float
        """
        opt = Adam(amsgrad=True, lr=learning_rate)
        self.model.compile(optimizer=opt, loss=categorical_crossentropy)

    def summary(self):
        """
        Prints summary of the SentDetect model.
        """
        self.model.summary()

    def train(self, num_epochs, steps_per_epoch, train_data_generator, val_data_generator):
        """
        Trains the SentDetect model.
        :param num_epochs: number of epochs
        :type num_epochs: int
        :param steps_per_epoch: steps per epoch
        :type steps_per_epoch: int
        :param train_data_generator: train data generator
        :type train_data_generator: generator
        :param val_data_generator: val data generator
        :type val_data_generator: generator
        """
        checkpoint = ModelCheckpoint(self.model_filepath, verbose=1, save_weights_only=True, mode='min')
        logger = CSVLogger(self.logs_filepath)
        self.model.fit_generator(train_data_generator, epochs=num_epochs, steps_per_epoch=steps_per_epoch,
                                 callbacks=[checkpoint, logger], validation_data=next(val_data_generator), verbose=1)

    def load_weights(self, model_name):
        """
        Load weights from the model with the given name.
        :param model_name: name of the model
        :type model_name: str
        """
        self.model.load_weights(f'models/{model_name}.h5', by_name=True)

    def predict(self, features):
        """
        Generates prediction with the given image features.
        :return: predicted caption length
        :rtype: int
        """
        result = self.model.predict(np.array([features]))
        return np.argmax(result[0])


if __name__ == '__main__':
    model = SentDetect()
    model.build(15, 5000, 300, None, 2)
    print()

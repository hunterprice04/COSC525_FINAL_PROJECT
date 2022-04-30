import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense, Dropout, Softmax
from keras.layers import LSTM as LSTM_layer
from keras.models import Sequential

from src.models.ModelUtils import ModelUtils


class LSTM:

    @staticmethod
    def get_LSTM(config, training=True):
        TR, VOCAB_SZ = config.TRAINING, len(config.EMBED)
        BATCH_SZ = TR.BATCH_SIZE if training else 1

        name = ModelUtils.create_model_name(config, 'LSTM')
        model = Sequential(name=name)

        model.add(LSTM_layer(units=TR.HIDDEN_STATE_SIZE[0],
                             batch_input_shape=(BATCH_SZ, None, VOCAB_SZ),
                             return_sequences=True, stateful=True))
        model.add(Dropout(TR.DROPOUT))
        for HIDDEN_STATE_SIZE in TR.HIDDEN_STATE_SIZE[1:]:
            model.add(LSTM_layer(units=HIDDEN_STATE_SIZE,
                                 return_sequences=True, stateful=True))
            model.add(Dropout(TR.DROPOUT))

        model.add(Dense(VOCAB_SZ))
        if training:
            model.add(Softmax())

        opt = keras.optimizers.Adam(learning_rate=TR.LR)
        loss = tf.losses.CategoricalCrossentropy()
        model.compile(loss=loss,
                      optimizer=opt, metrics=['acc'])
        return model

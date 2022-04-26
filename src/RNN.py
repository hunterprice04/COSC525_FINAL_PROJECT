import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential

from src.Utils import Utils


class RNN:

    @staticmethod
    def get_RNN(config, training=True):
        TR, VOCAB_SZ = config.TRAINING, len(config.EMBED)
        BATCH_SZ = TR.BATCH_SIZE if training else 1

        name = Utils.create_model_name(config, 'RNN')
        model = Sequential(name=name)

        model.add(SimpleRNN(units=TR.HIDDEN_STATE_SIZE[0],
                            batch_input_shape=(BATCH_SZ, None, VOCAB_SZ),
                            return_sequences=True, stateful=True))
        model.add(Dropout(TR.DROPOUT))
        for HIDDEN_STATE_SIZE in TR.HIDDEN_STATE_SIZE[1:]:
            model.add(SimpleRNN(units=HIDDEN_STATE_SIZE,
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

    @staticmethod
    def model_test(config, *args, **kwargs):
        VOCAB_SZ = len(config.EMBED)
        model = Sequential()
        model.add(SimpleRNN(100,
                            stateful=True,
                            return_sequences=True,
                            batch_size=1,
                            input_shape=(None, VOCAB_SZ)))
        model.add(Dense(VOCAB_SZ, activation='softmax'))
        opt = keras.optimizers.Adam(learning_rate=0.005)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
        return model

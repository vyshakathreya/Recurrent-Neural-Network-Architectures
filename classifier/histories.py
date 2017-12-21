
from keras.callbacks import Callback

class ErrorHistory(Callback):
    def on_train_begin(self, logs={}):
        self.error = []

    def on_batch_end(self, batch, logs={}):
        self.error.append(100 - logs.get('acc'))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

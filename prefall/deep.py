"""Step 2b: deep sequence models from slide 12 (Keras 3 / TensorFlow).

Both stay shallow on purpose (slide 12) so step 3 can compress them under 50 KB.
Input: raw 50 x 6 window (1 s at 50 Hz), scaled by dataset.SCALE.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import keras
from keras import layers


def cnn_bilstm(lstm_units=32):
    """1D-CNN (local features) -> BiLSTM (forward + backward context) -> Dense + Dropout -> softmax."""
    inp = keras.Input((50, 6))
    x = layers.Conv1D(16, 5, padding="same", activation="relu")(inp)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(2, activation="softmax")(x), name="cnn_bilstm")


def convlstm(filters=16):
    """ConvLSTM variant: convolution inside the recurrent cell. The 50-sample window is reshaped
    to 5 time steps x 10 samples x 6 channels, the layout ConvLSTM1D expects."""
    inp = keras.Input((50, 6))
    x = layers.Reshape((5, 10, 6))(inp)
    x = layers.ConvLSTM1D(filters, 3, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(2, activation="softmax")(x), name="convlstm")


def train(model, Xtr, ytr, Xval, yval, epochs=20, batch=256, seed=0, verbose=2, lr=1e-3, callbacks=()):
    keras.utils.set_random_seed(seed)
    n = len(ytr)
    cw = {0: n / (2 * (ytr == 0).sum()), 1: n / (2 * (ytr == 1).sum())}   # same as sklearn 'balanced'
    model.compile(keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    return model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs, batch_size=batch,
                     class_weight=cw, callbacks=[stop, *callbacks], verbose=verbose)


def predict(model, X, thr=0.5):
    return (model.predict(X, batch_size=1024, verbose=0)[:, 1] > thr).astype(int)


def size_kb(model):
    """(parameters, FP32 KB, INT8 KB). INT8 is the ideal 1 byte per weight; real TFLite adds overhead."""
    n = model.count_params()
    return n, n * 4 / 1024, n / 1024

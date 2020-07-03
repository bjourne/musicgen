# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.utils import SP
from os import environ
from tensorflow.data import Dataset
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices,
                               list_physical_devices)
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import *
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import *
from tensorflow.nn import softmax
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np
import tensorflow as tf

# Transformer epsilon. Unsure if this needs to be configurable.
TRANSF_EPS = 1e-6

def select_strategy():
    SP.header('SELECTING STRATEGY')
    gpus = list_physical_devices('GPU')
    SP.header('%d GPU(s)' % len(gpus))
    for gpu in gpus:
        SP.print(gpu)
    SP.leave()
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        dev = '/GPU:0' if gpus else '/CPU:0'
        SP.print('No TPU, using %s instead.' % dev)
        SP.leave()
        return OneDeviceStrategy(device = dev)
    SP.print('TPU address: %s' % tpu_addr)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    tpus = list_logical_devices('TPU')
    SP.header('%d TPU(s)' % len(tpus))
    for tpu in tpus:
        SP.print(tpu)
    SP.leave()
    strategy = TPUStrategy(resolver)
    SP.print('%d synced replicas.' % strategy.num_replicas_in_sync)
    SP.leave()
    return strategy

def sequence_to_samples(seq, length):
    stride = length - 1
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(length + 1, drop_remainder = True)
    source = tf.constant(seq, dtype = tf.int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(length + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000)

def generate_sequences(model, temps, seed, length, excluded):
    batch_size = len(temps)

    # Make temps into a row vector
    temps = np.array(temps)[:,None]

    # Priming the model
    for i in range(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])

    preds = [seed[:, -1:]]
    eps = np.finfo('float').eps
    for _ in range(length):
        last_word = preds[-1]
        Ps = model.predict(last_word)[:, 0, :]

        # Assign a very low probability to tokens to be avoided.
        Ps[:,excluded] = eps

        # Weigh probs according to temps
        Ps = np.exp(np.log(Ps) / temps)

        # Normalize
        Ps = (Ps.T / Ps.sum(axis = 1)).T

        next_idx = [np.random.choice(len(P), p = P) for P in Ps]
        preds.append(np.asarray(next_idx, dtype = np.int32))

    SP.leave()
    return [[int(preds[j][i]) for j in range(length)]
            for i in range(batch_size)]

def compute_and_apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training = True)
        loss = model.compiled_loss(y, y_hat,
                                   regularization_losses = model.losses)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    grads, _ = tf.clip_by_global_norm(grads, 15)
    model.optimizer.apply_gradients(zip(grads, vars))
    return y_hat

class MyModel(Model):
    def train_step(self, data):
        x, y = data
        y_hat = compute_and_apply_gradients(self, x, y)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

def lstm_model(vocab_size, emb_size,
               lstm1_units, lstm2_units,
               dropout, rec_dropout,
               stateful, batch_size):
    inp = Input(
        shape = (None,),
        batch_size = batch_size,
        dtype = tf.int32)
    embedding = Embedding(
        input_dim = vocab_size,
        output_dim = emb_size)
    lstm1 = LSTM(
        lstm1_units,
        stateful = stateful,
        return_sequences = True,
        dropout = dropout,
        recurrent_dropout = rec_dropout)
    lstm2 = LSTM(
        lstm2_units,
        stateful = stateful,
        return_sequences = True,
        dropout = dropout,
        recurrent_dropout = rec_dropout)
    time_dist = TimeDistributed(Dense(vocab_size))
    out = time_dist(lstm2(lstm1(embedding(inp))))
    return MyModel(inputs = [inp], outputs = [out])

def split_heads(inp, n_heads, depth, batch_size):
    inp = tf.reshape(inp, (batch_size, -1, n_heads, depth))
    return tf.transpose(inp, perm = [0, 2, 1, 3])

def transformer_model(vocab_size, d_model, ffn_units, dropout,
                      n_layers, n_heads):

    # Variables.
    depth = d_model // n_heads
    d_model_f32 = tf.cast(d_model, tf.float32)
    depth_f32 = tf.cast(depth, tf.float32)

    # Setup the position encoding
    pos = tf.range(5000, dtype = tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype = tf.float32)[tf.newaxis, :]
    angle_rads = pos / tf.pow(10_000, (2 * (i // 2)) / d_model_f32)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis = -1)
    pos_encoding = tf.expand_dims(pos_encoding, 0)

    # Input and look-ahead mask.
    inp = Input(shape = (None,))

    random_uniform = RandomUniform(-0.1, 0.1)
    x = Embedding(vocab_size, d_model,
                  embeddings_initializer = random_uniform)(inp)
    x *= tf.math.sqrt(d_model_f32)

    # Important variables.
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]

    # Hopefully this is only calculated once?
    x += pos_encoding[:, :seq_len, :]
    x = Dropout(dropout)(x)

    # Look-ahead mask
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    for _ in range(n_layers):
        # Multihead attention part
        wq = Dense(d_model)(x)
        wk = Dense(d_model)(x)
        wv = Dense(d_model)(x)

        q = split_heads(wq, n_heads, depth, batch_size)
        k = split_heads(wk, n_heads, depth, batch_size)
        v = split_heads(wv, n_heads, depth, batch_size)

        # Scaled dot product attention
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        logits = matmul_qk / tf.math.sqrt(depth_f32) + mask * -1e9
        weights = softmax(logits, axis = -1)
        attn = tf.matmul(weights, v)

        # Pass through dense layer and normalize.
        attn = tf.transpose(attn, perm = [0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, -1, d_model))
        attn = Dense(d_model)(attn)
        attn = Dropout(dropout)(attn)
        x = LayerNormalization(epsilon = TRANSF_EPS)(x + attn)

        # Point-wise feed-forward
        ffn = Dense(ffn_units, activation = 'relu')(x)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(d_model)(ffn)
        ffn = Dropout(dropout)(ffn)
        x = LayerNormalization(epsilon = TRANSF_EPS)(x + ffn)

    x = Dense(vocab_size, kernel_initializer = random_uniform)(x)
    return Model(inputs = inp, outputs = x)

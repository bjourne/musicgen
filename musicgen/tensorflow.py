# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen import gpt2
from musicgen.utils import SP
from os import environ
from tensorflow.config import *
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute import TPUStrategy
from tensorflow.keras import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from tensorflow.tpu.experimental import initialize_tpu_system
from tqdm import trange

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

def compute_and_apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training = True)
        loss = model.compiled_loss(y, y_hat,
                                   regularization_losses = model.losses)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    # Not sure what the proper gradient clipping should be.
    grads, _ = tf.clip_by_global_norm(grads, 5)
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
    dtype = tf.uint16)
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

def transformer(vocab_size, d_model, ffn_units, dropout,
                n_layers, n_heads, seq_len, is_training):
    # Input and look-ahead mask
    inp = Input(shape = (None,))

    # Variables
    depth = d_model // n_heads
    depth_f32 = tf.cast(depth, tf.float32)
    d_model_f32 = tf.cast(d_model, tf.float32)

    # Setup pos encoding
    pos = tf.range(5000, dtype = tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype = tf.float32)[tf.newaxis, :]
    rads = pos / tf.pow(10_000, (2 * (i // 2)) / d_model_f32)
    sines = tf.math.sin(rads[:, 0::2])
    cosines = tf.math.cos(rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis = -1)
    pos_encoding = tf.expand_dims(pos_encoding, 0)

    random_uniform = RandomUniform(-0.1, 0.1)
    x = Embedding(vocab_size, d_model,
                  embeddings_initializer = random_uniform)(inp)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    # Shapes
    batch_size = tf.shape(x)[0]

    # Idk why this doesn't work.
    if not is_training:
        seq_len = tf.shape(x)[1]

    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    # Hopefully this is only calculated once?
    x = x + pos_encoding[:, :seq_len, :]
    x = Dropout(dropout)(x)

    # For head splitting/merging
    split_pat = (batch_size, -1, n_heads, depth)
    transp_pat = (0, 2, 1, 3)

    for _ in range(n_layers):
        # Multihead attention part
        wq = Dense(d_model)(x)
        wk = Dense(d_model)(x)
        wv = Dense(d_model)(x)

        # Split heads
        q = tf.transpose(tf.reshape(wq, split_pat), transp_pat)
        k = tf.transpose(tf.reshape(wk, split_pat), transp_pat)
        v = tf.transpose(tf.reshape(wv, split_pat), transp_pat)

        # Scaled dot product attention
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        logits = matmul_qk / tf.math.sqrt(depth_f32) + mask * -1e9
        weights = softmax(logits, axis = -1)
        attn = tf.matmul(weights, v)

        # Merge heads
        attn = tf.transpose(attn, transp_pat)
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

def load_training_model(g, vocab_size):
    with select_strategy().scope():
        if g['network-type'] == 'transformer':
            model = transformer(vocab_size, 128, 2048, 0.2, 8, 16,
                                g['sequence-length'], True)
            opt = RMSprop(learning_rate = g['learning-rate'])
        elif g['network-type'] == 'lstm':
            model = lstm_model(vocab_size,
                               g['embedding-size'],
                               g['lstm1-units'],
                               g['lstm2-units'],
                               g['dropout'],
                               g['recurrent-dropout'],
                               False, g['batch-size'])
            opt = RMSprop(learning_rate = g['learning-rate'])
        elif g['network-type'] == 'gpt2':
            gpt2.VOCAB_SIZE = vocab_size
            gpt2.HIDDEN_SIZE = g['hidden-size']
            model = gpt2.GPT2()
            opt = Adam(learning_rate = g['learning-rate'], epsilon=1e-08)
        else:
            assert False
        loss_fn = SparseCategoricalCrossentropy(from_logits = True)
        metrics = ['sparse_categorical_accuracy']
        model.compile(optimizer = opt, loss = loss_fn,
                      metrics = metrics)
        model(tf.constant([[0]]))
    return model

def load_generating_model(g, vocab_size, batch_size):
    if g['network-type'] == 'lstm':
        model = lstm_model(vocab_size,
                           g['embedding-size'],
                           g['lstm1-units'],
                           g['lstm2-units'],
                           0.0, 0.0,
                           True, batch_size)
    elif g['network-type'] == 'gpt2':
        strategy = select_strategy()
        with strategy.scope():
            model = GPT2(vocab_size)
            model.compile()
        model(tf.constant([[0]]))
    return model

def temperature_skew(P, temp):
    P = np.exp(np.log(P) / temp)
    return P / P.sum()

def top_p_skew(P, top_p):
    prob_ixs = np.argsort(-P)
    PC = np.cumsum(P[prob_ixs])
    top_n = len(PC[PC <= top_p]) + 1

    # Clear the prob of those who didn't make it.
    P[prob_ixs[top_n:]] = np.finfo('float').eps
    return P / P.sum()

def skew_distribution(P, sampling_method):
    type, param = sampling_method
    if type == 'top-p':
        return top_p_skew(P, param)
    elif type == 'temperature':
        return temperature_skew(P, param)
    else:
        assert False

def sample_logits(logits, banned_ixs, skews):
    eps = np.finfo('float').eps
    Ps = softmax(logits).numpy()

    confusion_indices = np.where(np.max(Ps, axis = 1) < 0.5)[0]
    if len(confusion_indices) > 0:
        SP.print([skews[i] for i in confusion_indices])

    # Dont sample any "banned" tokens
    for ix in banned_ixs:
        Ps[:, ix] = eps

    for i in range(Ps.shape[0]):
        Ps[i] = skew_distribution(Ps[i], skews[i])
    ixs = np.array([np.random.choice(len(P), p = P) for P in Ps])
    return ixs, [np.log(Ps[i, ix]) for i, ix in enumerate(ixs)]

def generate_sequences_normal(model, n_generate, banned_ixs, prompt, skews,
                              max_seq_len):
    log_prob_sums = np.zeros(len(skews))
    preds = np.empty((0, prompt.shape[0]), int)
    for _ in trange(n_generate):
        logits = model(prompt, training = False)[:, -1, :]
        ixs, log_probs = sample_logits(logits, banned_ixs, skews)
        preds = np.vstack((preds, ixs))
        log_prob_sums += log_probs

        # Append column
        prompt = np.append(prompt, np.expand_dims(ixs, 1), axis = 1)
        if prompt.shape[1] >= max_seq_len:
            # Delete first column
            prompt = prompt[:, 1:]
    return preds.T, log_prob_sums

def generate_sequences_lstm(model, n_generate, banned_ixs,
                            prompt, skews):
    for i in trange(prompt.shape[1] - 1):
        model.predict(prompt[:, i])

    # The last item of the prompt is saved so that it can be used to
    # generate the first prediction.
    preds = np.expand_dims(prompt[:, -1], 0)
    log_prob_sums = np.zeros(len(skews))
    for _ in trange(n_generate):
        logits = model.predict(preds[-1])[:, -1, :]
        ixs, log_probs = sample_logits(logits, banned_ixs, skews)
        log_prob_sums += log_probs
        preds = np.vstack((preds, ixs))
    # Skip the first element which is not actually a prediction.
    return preds.T[:,1:], log_prob_sums

def generate_sequences(g, model, prompt, n_generate, banned_ixs, skews):
    if g['network-type'] == 'lstm':
        return generate_sequences_lstm(model, n_generate,
                                       banned_ixs,
                                       prompt, skews)
    else:
        return generate_sequences_normal(model, n_generate,
                                         banned_ixs, prompt, skews,
                                         g['sequence-length'])

def unlikelihood_loss(y_true, y_pred, from_logits):
    if len(y_true.shape) != 2:
        raise ValueError('y_true must have shape [bs x sl]')
    if len(y_pred.shape) != 3:
        raise ValueError('y_pred must have shape [bs x sl x vs]')
    bs, sl, vs = y_pred.shape

    # Produces a tensor of shape [bs x sl x vs]. The tensor is 1 if
    # the unlikelihood loss should be applied for the token type and 0
    # otherwise.
    candidate_mask = tf.reshape(tf.tile(y_true, [1, sl]), (bs, sl, sl))
    candidate_mask += 1
    candidate_mask *= tf.sequence_mask(tf.range(sl), sl, dtype=tf.int32)
    candidate_mask = tf.reduce_sum(
        tf.one_hot(candidate_mask, vs + 1)[:, :, :, 1:], axis=2)
    candidate_mask = tf.cast(tf.cast(candidate_mask, tf.bool), tf.float32)

    # True mask has shape [bs x sl x vs] and is 0 if the token type
    # is the true token and 1 otherwise.
    true_mask = tf.cast(tf.math.logical_not(
        tf.cast(tf.one_hot(y_true, vs, dtype=tf.int32), tf.bool)),
                        tf.float32)

    final_mask = candidate_mask * true_mask

    if from_logits:
        y_true = tf.nn.softmax(y_true, axis = -1)
    unlikelihood = tf.math.log(1 - y_true) * final_mask

    # Reduce!
    return -tf.reduce_sum(unlikelihood, axis = -1)
